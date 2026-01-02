import os
import json
import sys
import threading
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Dict, Set
import time
import math
import subprocess

import torch
import torch.distributed as dist

from nmoe.data.mixture import MixturePlan, resolve_plan


TRACKER = "latest_checkpointed_iteration.txt"


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _world_size() -> int:
    return dist.get_world_size() if _is_dist() else 1


def iteration_dir(base: str | Path, step: int) -> str:
    return str(Path(base) / f"iter_{step:07d}")


def tracker_path(base: str | Path) -> str:
    return str(Path(base) / TRACKER)


def write_tracker(base: str | Path, step: int) -> None:
    if _rank() != 0:
        return
    os.makedirs(base, exist_ok=True)
    out = tracker_path(base)
    tmp = out + f".tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(str(int(step)))
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, out)
    try:
        _fsync_file(out)
        _fsync_dir(str(Path(base)))
    except Exception:
        pass


def read_tracker(base: str | Path) -> int:
    try:
        with open(tracker_path(base), "r") as f:
            txt = f.read().strip()
            return int(txt)
    except Exception:
        return -1


def _safe_load(path: str) -> Optional[dict]:
    """Best-effort torch.load that never throws and maps to CPU."""
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
        return None


def read_latest_rd_info(base: str | Path) -> Optional[dict]:
    """Read run_info from the latest checkpoint's rd.pt, if present.

    Returns a dict including the original run_info fields plus 'step' when
    available, or None if no readable rd.pt is found.
    """
    step = read_tracker(base)
    if step <= 0:
        return None
    rd_path = os.path.join(iteration_dir(base, step), 'rd.pt')
    rd = _safe_load(rd_path)
    if not rd:
        return None
    info = rd.get('run_info')
    if isinstance(info, dict):
        out = dict(info)
        try:
            out['step'] = int(rd.get('step', 0))
        except Exception:
            pass
        return out
    return None


@dataclass
class AsyncTask:
    path: str
    base: str
    step: int
    state: dict[str, Any]


def _materialize_to_cpu_sync(state: dict[str, Any]) -> dict[str, Any]:
    """Materialize tensors in a nested state (dict/list/tuple) onto pinned CPU.

    Mirrors the async path but without spawning a thread; uses a dedicated CUDA
    stream when available. Non-tensor leaves are copied by reference.
    """
    @torch.no_grad()
    def _copy(obj: Any, ckpt_stream: Optional[torch.cuda.Stream]) -> Any:
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda:
                dst = torch.empty_like(obj, device='cpu', pin_memory=False)
                if ckpt_stream is not None:
                    with torch.cuda.stream(ckpt_stream):
                        dst.copy_(obj, non_blocking=True)
                else:
                    dst.copy_(obj)
                return dst
            return obj
        if isinstance(obj, dict):
            return {k: _copy(v, ckpt_stream) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_copy(v, ckpt_stream) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_copy(v, ckpt_stream) for v in obj)
        return obj

    # Avoid creating a new CUDA stream per checkpoint. Stream creation can
    # allocate device resources and fail under tight memory, even though the
    # checkpoint copy itself is host-side.
    out = _copy(state, None)
    return out


def _fsync_file(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _iter_dir(base: str | Path, step: int) -> str:
    return iteration_dir(base, step)


def _rd_path(base: str | Path, step: int) -> str:
    return os.path.join(_iter_dir(base, step), 'rd.pt')


def _dp_path(base: str | Path, step: int, rank: int) -> str:
    return os.path.join(_iter_dir(base, step), f'dp_rank_{rank:03d}.pt')


def _read_git_sha_from_rd(base: str | Path, step: int) -> str:
    rd = _safe_load(_rd_path(base, step))
    if not rd:
        return 'unknown'
    info = rd.get('run_info', {}) if isinstance(rd, dict) else {}
    return str(info.get('git_sha', 'unknown'))


def _read_world_from_rd(base: str | Path, step: int) -> int:
    rd = _safe_load(_rd_path(base, step))
    if not rd:
        return -1
    info = rd.get('run_info', {}) if isinstance(rd, dict) else {}
    try:
        return int(info.get('world', -1))
    except Exception:
        return -1


def _all_dp_present(base: str | Path, step: int, world: int) -> bool:
    if world <= 0:
        return False
    it = _iter_dir(base, step)
    if not os.path.isdir(it):
        return False
    for r in range(world):
        if not os.path.exists(_dp_path(base, step, r)):
            return False
    return True


def _write_manifest(base: str | Path, step: int, world: int) -> None:
    it_dir = _iter_dir(base, step)
    os.makedirs(it_dir, exist_ok=True)
    rd = _rd_path(base, step)
    files = [rd] + [_dp_path(base, step, r) for r in range(world)]
    bytes_total = 0
    for p in files:
        try:
            bytes_total += os.path.getsize(p)
        except Exception:
            pass
    git_sha = _read_git_sha_from_rd(base, step)
    manifest = {
        'step': int(step),
        'world': int(world),
        'dp_count': int(world),
        'bytes_total': int(bytes_total),
        'git_sha': git_sha,
        'files': [os.path.basename(p) for p in files],
    }
    tmp = os.path.join(it_dir, 'manifest.json.tmp')
    out = os.path.join(it_dir, 'manifest.json')
    with open(tmp, 'w') as f:
        json.dump(manifest, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out)
    _fsync_file(out)
    _fsync_dir(it_dir)


def try_finalize_step(base: str | Path, step: int) -> bool:
    """Rank‑0 helper: if rd.pt and all dp shards exist, write manifest then flip tracker.
    Returns True if completion succeeded.
    """
    try:
        world = _read_world_from_rd(base, step)
        if world <= 0:
            return False
        # Require rd.pt present
        if not os.path.exists(_rd_path(base, step)):
            return False
        if not _all_dp_present(base, step, world):
            return False
        _write_manifest(base, step, world)
        write_tracker(base, step)
        return True
    except Exception:
        return False


class _AsyncSaver(threading.Thread):
    def __init__(self, max_queue:int = 1, on_saved: Optional[Callable[[int, float, int], None]] = None) -> None:
        super().__init__(daemon=True)
        self.q: "queue.Queue[AsyncTask | None]" = queue.Queue(maxsize=max(1, int(max_queue)))
        self._on_saved = on_saved
        self._error: Exception | None = None
        self.start()

    def run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                return
            assert isinstance(item, AsyncTask)
            try:
                bytes_written, ms = self._atomic_save(item.path, item.state)
                # Do not flip tracker here; finalize step will update when complete
                if self._on_saved is not None:
                    try:
                        self._on_saved(bytes_written, ms, item.step)
                    except Exception:
                        pass
            except Exception as e:
                if self._error is None:
                    self._error = e
                # Fail loud, but do not crash training from a background thread.
                try:
                    sys.stderr.write(f"[ckpt] async save failed step={item.step} path={item.path}: {e}\n")
                    sys.stderr.flush()
                except Exception:
                    pass
            finally:
                self.q.task_done()

    def submit(self, path: str, base: str, step: int, state: dict[str, Any]) -> None:
        if self._error is not None:
            raise RuntimeError(f"checkpoint async saver previously failed: {self._error}") from self._error
        # Reliability invariant: enqueue CPU materialized state only. The background thread must
        # never hold references to live CUDA tensors from the training step.
        cpu_state = _materialize_to_cpu_sync(state)
        try:
            self.q.put(AsyncTask(path, base, step, cpu_state), block=True, timeout=60.0)
        except queue.Full as e:
            raise RuntimeError(
                f"checkpoint async queue full (step={step}, path={path}). "
                "This indicates checkpoint I/O cannot keep up; refusing to drop checkpoints."
            ) from e

    def close(self) -> None:
        # Signal thread to exit and wait for a clean shutdown to avoid interpreter
        # teardown while the background thread still references Python objects.
        try:
            self.q.put(None)
        except Exception:
            return
        try:
            self.join(timeout=60.0)
        except Exception:
            pass

    def wait(self) -> None:
        # Wait until all queued tasks are processed
        self.q.join()
        if self._error is not None:
            raise RuntimeError(f"checkpoint async saver failed: {self._error}") from self._error

    def _atomic_save(self, path: str, state: dict[str, Any]) -> tuple[int, float]:
        t0 = time.perf_counter()
        tmp = path + ".tmp"
        base = os.path.dirname(path)
        os.makedirs(base, exist_ok=True)
        # Write temp file first
        torch.save(state, tmp)
        # Robust finalize: best-effort atomic replace; fall back to direct save on failure
        try:
            os.replace(tmp, path)
            try:
                _fsync_file(path)
                _fsync_dir(base)
            except Exception:
                pass
        except FileNotFoundError:
            # Temp may not be visible yet on some backends; try a direct save as a fallback
            try:
                # Ensure directory still exists in fallback path
                os.makedirs(base, exist_ok=True)
                torch.save(state, path)
                try:
                    _fsync_file(path)
                    _fsync_dir(base)
                except Exception:
                    pass
            finally:
                # Cleanup temp if present
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass
        ms = (time.perf_counter() - t0) * 1000.0
        try:
            size = os.path.getsize(path)
        except Exception:
            size = 0
        return size, ms


class _Purger(threading.Thread):
    def __init__(self, base: str, keep_last: int) -> None:
        super().__init__(daemon=True)
        self.base = base
        self.keep_last = keep_last
        self._req: "queue.Queue[None]" = queue.Queue()
        self.start()

    def run(self) -> None:
        while True:
            self._req.get()
            try:
                self._purge_once()
            finally:
                self._req.task_done()

    def trigger(self) -> None:
        # Collapse multiple triggers
        if self._req.empty():
            self._req.put(None)

    def _purge_once(self) -> None:
        if self.keep_last <= 0:
            return
        base = Path(self.base)
        if not base.exists():
            return
        iters = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("iter_")])
        if len(iters) <= self.keep_last:
            return
        for p in iters[:-self.keep_last]:
            try:
                # Remove directory tree for this iteration
                for sub in p.glob("**/*"):
                    try:
                        sub.unlink()
                    except IsADirectoryError:
                        pass
                for sub in sorted(p.glob("**/*"), reverse=True):
                    try:
                        sub.rmdir()
                    except Exception:
                        pass
                p.rmdir()
            except Exception:
                # Best-effort purge
                pass


class Checkpointer:
    """Megatron/TorchTitan-style checkpoint manager (minimal).

    - Per-step directory: base/iter_0000123/
    - Rank-local files:   dp_rank_000.pt (per rank to capture sharded experts)
    - Tracker file:       latest_checkpointed_iteration.txt (integer step)
    - Keep-last rotation: background purger
    - Optional async rank-local torch.save via a background thread
    """

    def __init__(self, base: str, keep_last: int = 5, async_io: bool = False, async_max_queue:int = 1) -> None:
        # Normalize base to absolute to avoid CWD-related surprises in background threads
        self.base = str(Path(base).absolute())
        self.keep_last = keep_last
        self.async_io = async_io
        self._purger = _Purger(self.base, keep_last) if keep_last > 0 else None
        # metrics
        self.last_bytes: int = 0
        self.last_ms: float = 0.0
        self._last_requested_step: int = -1
        def _on_saved(bytes_written:int, ms:float, step:int):
            self.last_bytes = bytes_written
            self.last_ms = ms
            if self._purger:
                self._purger.trigger()
            # Rank 0 opportunistically attempts to finalize this step
            try:
                if _rank() == 0:
                    try_finalize_step(self.base, step)
            except Exception:
                pass
        # Ensure capacity for rd.pt + dp_rank_XXX.pt on rank 0 without dropping
        max_q = max(2, async_max_queue)
        self._saver = _AsyncSaver(max_queue=max_q, on_saved=_on_saved) if async_io else None

    def _raise_if_async_failed(self) -> None:
        if self._saver is None:
            return
        # wait() raises, but we also fail fast on submit paths.
        if getattr(self._saver, "_error", None) is not None:
            err = self._saver._error  # type: ignore[attr-defined]
            raise RuntimeError(f"checkpoint async saver failed: {err}") from err

    def save_rank_local(self, step: int, state: dict[str, Any]) -> str:
        self._raise_if_async_failed()
        self._last_requested_step = max(self._last_requested_step, int(step))
        it_dir = iteration_dir(self.base, step)
        os.makedirs(it_dir, exist_ok=True)
        path = os.path.join(it_dir, f"dp_rank_{_rank():03d}.pt")

        if self._saver is not None:
            self._saver.submit(path, self.base, step, state)
        else:
            # Synchronous path: materialize and save inline (rare / tests)
            cpu_state = _materialize_to_cpu_sync(state)
            tmp = path + ".tmp"
            torch.save(cpu_state, tmp)
            os.replace(tmp, path)
            # Finalize may flip tracker if complete
            if _rank() == 0:
                try_finalize_step(self.base, step)
            if self._purger is not None:
                self._purger.trigger()

        return path

    def save_dense(self, step: int, state: dict[str, Any]) -> str:
        """Save replicated (dense/router) parameters once per step (call on rank 0)."""
        self._raise_if_async_failed()
        self._last_requested_step = max(self._last_requested_step, int(step))
        it_dir = iteration_dir(self.base, step)
        os.makedirs(it_dir, exist_ok=True)
        path = os.path.join(it_dir, "rd.pt")

        if self._saver is not None:
            self._saver.submit(path, self.base, step, state)
        else:
            # Synchronous path
            cpu_state = _materialize_to_cpu_sync(state)
            tmp = path + ".tmp"
            torch.save(cpu_state, tmp)
            os.replace(tmp, path)
            if _rank() == 0:
                try_finalize_step(self.base, step)
            if self._purger is not None:
                self._purger.trigger()

        return path

    def find_latest(self) -> tuple[int, str] | tuple[int, None]:
        def _candidate(step: int) -> str:
            return os.path.join(iteration_dir(self.base, step), f"dp_rank_{_rank():03d}.pt")

        step = read_tracker(self.base)
        if step > 0:
            it = iteration_dir(self.base, step)
            if os.path.exists(os.path.join(it, "manifest.json")) and os.path.exists(_candidate(step)):
                return step, _candidate(step)

        base = Path(self.base)
        if not base.exists():
            return -1, None
        iters = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("iter_")], reverse=True)
        for p in iters:
            try:
                s = int(p.name.split("_", 1)[1])
            except Exception:
                continue
            if not (p / "manifest.json").exists():
                continue
            path = str(p / f"dp_rank_{_rank():03d}.pt")
            if os.path.exists(path):
                return s, path
        return -1, None

    def try_finalize(self, step: int) -> bool:
        """Attempt to mark a step complete by writing manifest and flipping tracker.
        Rank-0 only; returns True if completion succeeded.
        """
        if _rank() != 0:
            return False
        return try_finalize_step(self.base, step)

    def recommend_interval(self, step_time_ms: float, safety: float = 1.3) -> int:
        """Compute save_every given last measured save time and step time.
        Returns at least 1.
        """
        if self.last_ms <= 0 or step_time_ms <= 0:
            return 1
        n = math.ceil(safety * (self.last_ms / step_time_ms))
        return max(1, int(n))

    def close(self) -> None:
        if self._saver is not None:
            # Wait for all pending saves to complete before closing
            self._saver.wait()
            if _rank() == 0 and self._last_requested_step > 0:
                try:
                    try_finalize_step(self.base, int(self._last_requested_step))
                except Exception:
                    pass
            self._saver.close()


"""Split‑format checkpointing utilities (no legacy paths)."""


def _split_param_names(model: torch.nn.Module) -> Tuple[Set[str], Set[str]]:
    """Return (dense_names, expert_names) using model.param_sets(), if available."""
    dense_names: Set[str] = set()
    expert_names: Set[str] = set()
    name_to_param: Dict[int, str] = {id(p): n for n, p in model.named_parameters()}
    if hasattr(model, 'param_sets'):
        expert_params, dense_params = model.param_sets()  # type: ignore[attr-defined]
        expert_names = {name_to_param.get(id(p)) for p in expert_params}
        dense_names = {name_to_param.get(id(p)) for p in dense_params}
        expert_names = {n for n in expert_names if n is not None}
        dense_names = {n for n in dense_names if n is not None}
    else:
        # Fallback: everything is dense
        dense_names = {n for n, _ in model.named_parameters()}
        expert_names = set()
    return dense_names, expert_names


def build_states(
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokens: int,
    loader,
    config_fingerprint: str = "",
    zero2_state: Optional[dict[str, Any]] = None,
    plan: Optional[MixturePlan] = None,
) -> Tuple[Optional[dict[str, Any]], dict[str, Any]]:
    """Build (replicated_dense_state, rank_local_state).

    - replicated_dense_state (rank 0 writes rd.pt): only replicated parameters (dense/router).
    - rank_local_state (every rank writes dp_rank_XXX.pt): expert (sharded) params + local optimizer state.
    """
    full_sd = model.state_dict()
    dense_names, expert_names = _split_param_names(model)

    # Keep all non-expert state (including buffers) in rd.pt for correctness.
    # Only expert weights are sharded across ranks.
    expert_sd = {k: v for k, v in full_sd.items() if k in expert_names}
    dense_sd = {k: v for k, v in full_sd.items() if k not in expert_names}

    # Run metadata (immutable) for rd.pt
    world = _world_size()
    cfg = getattr(model, 'config', None)
    try:
        git_sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.getcwd(),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_sha = 'unknown'

    rd_state: Optional[dict[str, Any]] = {
        'step': step,
        'model_dense': dense_sd,
        'tokens': tokens,
        'config_fingerprint': config_fingerprint,
        'run_info': {
            'git_sha': git_sha,
            'preset': getattr(cfg, 'preset', None),
            'dtype': getattr(cfg, 'dtype', None),
            'E': getattr(cfg, 'n_routed_experts', None),
            'K': getattr(cfg, 'n_activated_experts', None),
            'H': getattr(cfg, 'dim', None),
            'L': getattr(cfg, 'n_layers', None),
            'world': world,
            'dataset_version': getattr(loader, 'dataset_version', None),
            'tokenizer_id': getattr(loader, 'tokenizer_id', None),
        }
    }

    if plan is not None:
        # Save plan bundle (rank 0 only)
        rd_state['plan_bundle'] = {
            'mixture_id': plan.mixture_id,
            'flow_mode': plan.flow_mode,
            'plan_hash': plan.plan_hash,
            'plan_json': plan.to_json(),
        }

    dp_state: dict[str, Any] = {
        'step': step,
        'model_expert': expert_sd,
        'optimizer': optimizer.state_dict(),
        'loader': loader.state_dict() if hasattr(loader, 'state_dict') else None,
        'rng': {
            'torch': torch.random.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        'config_fingerprint': config_fingerprint,
    }
    if zero2_state is not None:
        dp_state['zero2'] = zero2_state

    return rd_state, dp_state


def load_state(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader=None,
    print_fn=print,
) -> tuple[int, int, Optional[dict[str, Any]]]:
    """Load split checkpoint and restore state. Returns (step, tokens, zero2_state).

    Expects rd.pt (replicated dense/router) in the same iteration directory,
    and dp_rank_XXX.pt for the calling rank.
    """
    it_dir = os.path.dirname(path)
    rd_path = os.path.join(it_dir, 'rd.pt')

    map_location = 'cpu'
    if torch.cuda.is_available():
        map_location = f'cuda:{torch.cuda.current_device()}'

    # Load replicated dense/router weights
    rd = torch.load(rd_path, map_location=map_location, weights_only=False)
    dense_sd = rd['model_dense']
    model.load_state_dict(dense_sd, strict=False)
    tokens = int(rd.get('tokens', 0))
    step = int(rd.get('step', 0))
    # Validate config hash against current model config (best-effort)
    try:
        from nmoe.config import fingerprint as _fingerprint
        current_fp = _fingerprint(getattr(model, 'config', None))
        saved_fp = str(rd.get('config_fingerprint', ''))
        if saved_fp and current_fp and saved_fp != current_fp:
            raise RuntimeError(
                f"config_fingerprint mismatch on resume (saved={saved_fp[:8]} current={current_fp[:8]}). "
                "Refusing to resume with a different config."
            )
    except Exception:
        raise

    # Check plan bundle if loader expects one
    if loader is not None and hasattr(loader, 'plan') and 'plan_bundle' in rd:
        saved_plan = rd['plan_bundle']
        current_plan = loader.plan
        if saved_plan['plan_hash'] != current_plan.plan_hash:
             # Mismatch policy: strictly fail unless user explicitly allows (not implemented yet)
            raise RuntimeError(
                f"Plan mismatch on resume! Saved hash={saved_plan['plan_hash'][:8]}, "
                f"Current hash={current_plan.plan_hash[:8]}. "
                "Cannot resume deterministically with different mixture plan."
            )
        print_fn(f"[resume] Plan hash match: {current_plan.plan_hash[:8]}")

    # Load rank-local shard
    print_fn(f'Loading checkpoint: {path}')
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt['model_expert'], strict=False)
    optimizer.load_state_dict(ckpt['optimizer'])

    if loader is not None and ckpt.get('loader'):
        loader.load_state_dict(ckpt['loader'])

    if ckpt.get('rng'):
        torch_state = ckpt['rng'].get('torch')
        if torch_state is not None:
            if not torch.is_tensor(torch_state):
                torch_state = torch.as_tensor(torch_state, dtype=torch.uint8)
            torch.random.set_rng_state(torch_state.cpu())
        cuda_states = ckpt['rng'].get('cuda')
        if cuda_states is not None and torch.cuda.is_available():
            for dev, state in enumerate(cuda_states):
                if state is not None:
                    state = torch.as_tensor(state, dtype=torch.uint8).cpu()
                    torch.cuda.set_rng_state(state, dev)

    return int(step), tokens, ckpt.get('zero2')


def load_checkpoint(
    checkpointer: Checkpointer,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader,
    plan,
    cfg,
    rank: int,
    print_fn=print,
) -> tuple[int, int, dict]:
    """Wrapper for checkpoint resume. Returns (start_step, tokens_seen, zero2_state)."""
    start_step = 0
    tokens_seen = 0
    zero2_state = {}

    if getattr(cfg, 'resume', True):
        step, path = checkpointer.find_latest()
        if path is not None:
            start_step, tokens_seen, z2 = load_state(path, model, optimizer, loader, print_fn)
            if z2 is not None:
                zero2_state = z2
            if rank == 0:
                print_fn(f"[nmoe] Resumed from step {start_step}, {tokens_seen:,} tokens")

    return start_step, tokens_seen, zero2_state


def save_checkpoint(
    checkpointer: Checkpointer,
    step: int,
    tokens_seen: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader,
    plan,
    zero2_state: dict,
    cfg,
    rank: int,
    config_fingerprint: str,
    checkpoint_every: int,
    print_fn=print,
) -> None:
    """Wrapper for checkpoint save. Handles RD/DP split and finalization."""
    if (step % checkpoint_every == 0) or (step == cfg.steps):
        rd_state, dp_state = build_states(
            step, model, optimizer, tokens_seen, loader, config_fingerprint,
            zero2_state=zero2_state, plan=plan,
        )
        if rank == 0:
            checkpointer.save_dense(step, rd_state)
        checkpointer.save_rank_local(step, dp_state)

        if rank == 0 and checkpointer.last_ms > 0:
            size_mb = checkpointer.last_bytes / (1024 * 1024)
            print_fn(f"[ckpt] saved step={step} size={size_mb:.1f}MB time={checkpointer.last_ms:.0f}ms")

        # Try to finalize: rank 0 writes manifest.json and flips tracker
        if rank == 0:
            try:
                done = checkpointer.try_finalize(step)
                if done:
                    print_fn(f"[ckpt] complete step={step}")
            except Exception:
                pass
