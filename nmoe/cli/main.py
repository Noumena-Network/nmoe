"""n - nmoe training CLI."""

import json
import os
import subprocess
import time
import shutil
import sqlite3
import sys
from pathlib import Path

import typer
from rich.console import Console

from nmoe.campaigns import (
  CampaignError,
  claim_candidate,
  discover_campaign_specs,
  evaluate_metrics,
  load_campaign,
  new_receipt_path,
  parse_candidate_overrides,
  propose_next_candidate,
  release_candidate_claim,
  resolve_receipt_dir,
  select_baseline,
  validate_candidate_overrides,
  write_json,
)

app = typer.Typer(
  name="n",
  help="nmoe training CLI",
  add_completion=False,
  no_args_is_help=True,
)
campaign_app = typer.Typer(
  help="Bounded autoresearch campaigns",
  add_completion=False,
)
console = Console()

NMOE_ROOT = Path(__file__).parent.parent.parent
NPROC = os.environ.get("NPROC", "8")
PYTHON_BIN = os.environ.get("PYTHON_BIN") or sys.executable or "python"
TORCHRUN_BIN = str(Path(PYTHON_BIN).with_name("torchrun"))


def _extra_python_paths() -> list[str]:
  paths = [
    str(NMOE_ROOT),
    str(NMOE_ROOT / "third_party" / "quack"),
    str(NMOE_ROOT / "third_party" / "flash_attn"),
    str(NMOE_ROOT / "triton" / "python"),
    "/opt/third_party/quack",
    "/opt/third_party/flash_attn",
    "/opt/third_party/triton/python",
  ]
  for env_key in ("NMOE_QUACK_PATH", "NMOE_FLASH_ATTN_PATH", "NMOE_TRITON_PYTHON_PATH"):
    value = os.environ.get(env_key, "").strip()
    if value:
      paths.append(value)
  seen: list[str] = []
  for path in paths:
    if path and path not in seen:
      seen.append(path)
  return seen


def _discover_data_dir() -> Path:
  """Discover data directory from env or common locations."""
  # Explicit env var takes precedence
  if "DATA_DIR" in os.environ:
    return Path(os.environ["DATA_DIR"])
  # Check common locations
  candidates = [
    Path("/data"),
    NMOE_ROOT / "data",
    Path.home() / "nmoe_data",
  ]
  for p in candidates:
    if p.exists():
      return p
  # Default to /data, will be created as needed
  return Path("/data")


DATA_DIR = _discover_data_dir()


def _experiments_db_path() -> Path:
  value = os.environ.get("NMOE_EXPERIMENTS_DB", "").strip()
  if value:
    return Path(value)
  return DATA_DIR / "experiments.db"


def _campaign_checkpoint_dir(experiment_id: str) -> Path | None:
  if not experiment_id:
    return None
  return DATA_DIR / "checkpoints" / "campaigns" / experiment_id


def _get_port(name: str, default: int) -> int:
  """Get port from env, handling service discovery collisions."""
  val = os.environ.get(f"NMOE_{name}", os.environ.get(name, str(default)))
  try:
    return int(val)
  except ValueError:
    return default

JUPYTER_PORT = _get_port("JUPYTER_PORT", 8888)
NVIZ_PORT = _get_port("NVIZ_PORT", 3000)

app.add_typer(campaign_app, name="campaign")


def _with_nmoe_env(env: dict | None = None) -> dict:
  """Return env with required PYTHONPATH for nmoe's vendored deps.

  This makes `n speedrun` work in minimal environments (e.g. k8s debug pods)
  without requiring users to manually export PYTHONPATH.
  """
  out = (env or os.environ).copy()

  # Ensure training/runtime deps are importable from vendored or image-provided roots.
  required = _extra_python_paths()
  # Only add entries that exist on disk (allows pip-installed fallbacks).
  required = [p for p in required if Path(p).exists()]

  parts = [p for p in str(out.get("PYTHONPATH", "")).split(":") if p]
  for p in required:
    if p not in parts:
      parts.append(p)
  if parts:
    out["PYTHONPATH"] = ":".join(parts)

  return out


def run(
  cmd: list[str],
  cwd: Path | None = None,
  env: dict | None = None,
  timeout_s: int | None = None,
) -> None:
  """Run command, exit on failure."""
  console.print(f"[blue]>[/blue] {' '.join(cmd)}")
  try:
    result = subprocess.run(
      cmd,
      cwd=cwd or NMOE_ROOT,
      env=_with_nmoe_env(env),
      timeout=timeout_s,
    )
  except subprocess.TimeoutExpired:
    console.print(f"[red]Command timed out after {timeout_s}s[/red]")
    raise typer.Exit(124)
  if result.returncode != 0:
    raise typer.Exit(result.returncode)


def run_background(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> subprocess.Popen:
  """Run command in background."""
  console.print(f"[blue]>[/blue] {' '.join(cmd)} &")
  return subprocess.Popen(
    cmd,
    cwd=cwd or NMOE_ROOT,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    env=_with_nmoe_env(env),
  )


def _torchrun_cmd() -> list[str]:
  if Path(TORCHRUN_BIN).exists():
    return [TORCHRUN_BIN]
  return [PYTHON_BIN, "-m", "torch.distributed.run"]


def _has_npy_shards(dir_path: Path) -> bool:
  try:
    return dir_path.exists() and any(dir_path.rglob("*.npy"))
  except Exception:
    return False


def ensure_eval_bundle() -> Path:
  """Ensure CORE eval bundle exists. Fails if missing (should be installed by bootstrap.sh).

  Returns path to eval_bundle directory.
  """
  bundle_dir = DATA_DIR / "eval" / "eval_bundle"

  if bundle_dir.exists() and any(bundle_dir.rglob("*.jsonl")):
    return bundle_dir

  console.print(f"[red]CORE eval bundle not found at {bundle_dir}[/red]")
  console.print("[yellow]Run 'bash scripts/bootstrap.sh' to install it, or download manually:[/yellow]")
  console.print("  curl -L -o /tmp/eval_bundle.zip https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip")
  console.print(f"  unzip /tmp/eval_bundle.zip -d {bundle_dir.parent}")
  raise typer.Exit(1)


def _token_budget_to_int(value: str) -> int:
  v = value.strip().upper()
  mult = 1
  if v.endswith("K"):
    mult = 1_000
    v = v[:-1]
  elif v.endswith("M"):
    mult = 1_000_000
    v = v[:-1]
  elif v.endswith("B"):
    mult = 1_000_000_000
    v = v[:-1]
  return int(float(v) * mult)


def ensure_speedrun_data(
  *,
  hf_dataset: str | None = None,
  val_data_file: str | None = None,
  train_tokens_budget: str = "10B",
  val_tokens_budget: str = "10485760",
) -> Path:
  """Ensure speedrun train/val datasets exist.

  Canonical dataset: karpathy/fineweb-edu-100b-shuffle (tokenized to GPT-2 shards).
  Returns path to DATA_DIR/speedrun.
  """
  hf_dataset = hf_dataset or os.environ.get("NMOE_SPEEDRUN_DATASET", "karpathy/fineweb-edu-100b-shuffle")
  val_data_file = val_data_file or os.environ.get("NMOE_SPEEDRUN_VAL_DATA_FILE", "shard_01822.parquet")
  train_tokens_min = _token_budget_to_int(train_tokens_budget)
  val_tokens_min = _token_budget_to_int(val_tokens_budget)

  train_dir = DATA_DIR / "speedrun" / "train"
  val_dir = DATA_DIR / "speedrun" / "val"

  def _manifest_ok(dir_path: Path, *, min_tokens: int) -> bool:
    m = dir_path / "manifest.json"
    if not m.exists():
      return False
    try:
      import json
      obj = json.loads(m.read_text())
    except Exception:
      return False
    src = str(obj.get("source_info", {}).get("source", ""))
    if not src.startswith(hf_dataset):
      return False
    if obj.get("tokenizer") != "gpt2":
      return False
    if int(obj.get("vocab_size", 0)) != 50304:
      return False
    if int(obj.get("eos_token_id", -1)) != 50256:
      return False
    if int(obj.get("total_tokens", 0)) < int(min_tokens):
      return False
    return True

  if _has_npy_shards(train_dir) and _has_npy_shards(val_dir):
    if not (_manifest_ok(train_dir, min_tokens=train_tokens_min) and _manifest_ok(val_dir, min_tokens=val_tokens_min)):
      raise typer.Exit(
        f"Non-canonical speedrun dataset detected at {DATA_DIR / 'speedrun'} (missing/mismatched manifest).\n"
        f"Expected: {hf_dataset} (gpt2 vocab=50304 eos=50256) with >= {train_tokens_budget} train tokens and >= {val_tokens_budget} val tokens.\n"
        f"To rebuild: rm -rf {DATA_DIR / 'speedrun'}"
      )
    console.print(f"[green]Data:[/green] {train_dir} (+ val)")
    return DATA_DIR / "speedrun"

  train_dir.mkdir(parents=True, exist_ok=True)
  val_dir.mkdir(parents=True, exist_ok=True)

  if not _has_npy_shards(train_dir):
    console.print(f"[yellow]Preparing speedrun train dataset → {train_dir}[/yellow]")
    run([
      PYTHON_BIN, "-m", "nmoe.data.cli", "prep",
      "--source", "hub_parquet",
      "--dataset", hf_dataset,
      "--split", "train",
      "--output", str(train_dir),
      "--name", "speedrun_train",
      "--tokenizer", "gpt2",
      "--vocab-size", "50304",
      "--eos-token-id", "50256",
      "--max-tokens-total", train_tokens_budget,
      "--num-shards", "64",
      "--parallel",
    ])
    if not _manifest_ok(train_dir, min_tokens=train_tokens_min):
      raise typer.Exit(f"speedrun train dataset prepared but manifest is not canonical: {train_dir / 'manifest.json'}")

  if not _has_npy_shards(val_dir):
    console.print(f"[yellow]Preparing speedrun val dataset → {val_dir}[/yellow]")
    run([
      PYTHON_BIN, "-m", "nmoe.data.cli", "prep",
      "--source", "hub_parquet",
      "--dataset", hf_dataset,
      "--split", "train",
      "--data-files", val_data_file,
      "--output", str(val_dir),
      "--name", "speedrun_val",
      "--tokenizer", "gpt2",
      "--vocab-size", "50304",
      "--eos-token-id", "50256",
      "--max-tokens-total", val_tokens_budget,
      "--num-shards", "8",
      "--parallel",
    ])
    if not _manifest_ok(val_dir, min_tokens=val_tokens_min):
      raise typer.Exit(f"speedrun val dataset prepared but manifest is not canonical: {val_dir / 'manifest.json'}")

  return DATA_DIR / "speedrun"


def ensure_data(name: str, tokens: str) -> Path:
  """Download data if not present, return path.

  Note: This is intentionally a small convenience helper. Golden-path training
  configs should still specify canonical data paths; this helper is for quick
  bring-up and smoke tests.
  """
  data_path = DATA_DIR / name
  if _has_npy_shards(data_path):
    console.print(f"[green]Data:[/green] {data_path}")
    return data_path

  data_path.mkdir(parents=True, exist_ok=True)
  console.print(f"[yellow]Downloading {tokens} tokens to {data_path}...[/yellow]")
  run([
    PYTHON_BIN, "-m", "nmoe.data.cli", "prep",
    "--source", "hf",
    "--dataset", "HuggingFaceFW/fineweb-edu",
    "--split", "train",
    "--output", str(data_path),
    "--name", name,
    "--tokenizer", "gpt2",
    "--vocab-size", "50304",
    "--eos-token-id", "50256",
    "--max-tokens-total", tokens,
    "--num-shards", "32",
    "--parallel",
  ])
  return data_path


def start_training(config: str, *, data_path_override: Path | None, background: bool = False):
  """Start training, optionally in background."""
  cmd = _torchrun_cmd() + [
    "--nproc_per_node", NPROC,
    "-m", "nmoe.train",
    config,
  ]
  if data_path_override is not None:
    cmd.append(f"--data_path={data_path_override}")
  if background:
    run_background(cmd)
    time.sleep(2)  # Let training initialize
  else:
    run(cmd)


def start_tunnel(port: int) -> str | None:
  """Start cloudflared tunnel, return public URL."""
  if not Path("/usr/local/bin/cloudflared").exists():
    console.print("[yellow]cloudflared not installed, skipping tunnel[/yellow]")
    return None

  console.print(f"[blue]Starting tunnel for port {port}...[/blue]")
  proc = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
  )

  # Wait for URL
  for _ in range(30):
    if proc.stdout is None:
      break
    line = proc.stdout.readline()
    if "trycloudflare.com" in line:
      import re
      match = re.search(r'https://[^\s]+\.trycloudflare\.com', line)
      if match:
        url = match.group(0)
        console.print(f"[green]Public URL:[/green] {url}")
        return url
    time.sleep(0.5)

  console.print("[yellow]Tunnel started but URL not captured[/yellow]")
  return None


def start_nviz_with_tunnel(dev: bool = False):
  """Start nviz and cloudflared tunnel."""
  nviz_path = NMOE_ROOT / "nviz"
  if not nviz_path.exists():
    console.print("[yellow]nviz not found[/yellow]")
    return
  if shutil.which("bun") is None:
    console.print("[red]bun not found. Run: bash scripts/bootstrap.sh[/red]")
    return

  env = os.environ.copy()
  env["NVIZ_METRICS_DIR"] = str(DATA_DIR / "metrics")
  env["PORT"] = str(NVIZ_PORT)

  if not (nviz_path / "node_modules").exists():
    console.print("[blue]Installing nviz deps (bun install)...[/blue]")
    run(["bun", "install"], cwd=nviz_path, env=env)

  if dev:
    # Dev mode: hot reload
    console.print("[blue]Starting nviz (dev mode)...[/blue]")
    run_background(["bun", "run", "dev"], cwd=nviz_path, env=env)
  else:
    # Production mode: build if needed, then start
    next_dir = nviz_path / ".next"
    if not next_dir.exists():
      console.print("[blue]Building nviz...[/blue]")
      run(["bun", "run", "build"], cwd=nviz_path, env=env)
    run_background(["bun", "run", "start"], cwd=nviz_path, env=env)

  time.sleep(2)
  start_tunnel(NVIZ_PORT)


def open_nmon():
  """Open nmon TUI (replaces current process)."""
  nmon_path = NMOE_ROOT / "tools" / "nmon" / "nmon"
  if not nmon_path.exists():
    console.print("[yellow]Building nmon...[/yellow]")
    run(["go", "build", "-o", "nmon", "./cmd/nmon"], cwd=NMOE_ROOT / "tools" / "nmon")
  console.print("[blue]Opening nmon...[/blue]")
  args = [str(nmon_path), f"--leaderboard={LEADERBOARD_PATH}"]
  os.execv(str(nmon_path), args)


def open_jupyter():
  """Start JupyterLab with cloudflared tunnel."""
  console.print(f"[blue]Starting JupyterLab on port {JUPYTER_PORT}...[/blue]")

  # Start JupyterLab in background
  run_background([
    PYTHON_BIN, "-m", "jupyter", "lab",
    "--ip=0.0.0.0",
    f"--port={JUPYTER_PORT}",
    "--no-browser",
    "--allow-root",
    "--ServerApp.token=''",
    "--ServerApp.password=''",
    f"--notebook-dir={str(NMOE_ROOT)}",
  ])
  time.sleep(3)

  # Start tunnel
  start_tunnel(JUPYTER_PORT)

  console.print("[blue]JupyterLab running. Press Ctrl+C to stop.[/blue]")
  console.print("[dim]from nmoe.research import lab[/dim]")
  try:
    while True:
      time.sleep(1)
  except KeyboardInterrupt:
    pass


# -----------------------------------------------------------------------------
# Commands: Start (new run by default, --attach for existing)
# -----------------------------------------------------------------------------

SPEEDRUN_CONFIGS = {
  "dense": "configs/speedrun/dense.toml",
  "moe": "configs/speedrun/moe.toml",         # 64 experts
  "super": "configs/speedrun/super.toml",     # 256 experts
  "ultra": "configs/speedrun/ultra.toml",     # 4096 experts
}

LEADERBOARD_PATH = NMOE_ROOT / "LEADERBOARD.json"


def _resolve_speedrun_config(config: str) -> str:
  if config not in SPEEDRUN_CONFIGS:
    console.print(f"[red]Unknown config: {config}[/red]")
    console.print(f"[yellow]Available: {', '.join(SPEEDRUN_CONFIGS.keys())}[/yellow]")
    raise typer.Exit(1)
  return SPEEDRUN_CONFIGS[config]


def _resolve_speedrun_dtype(explicit_dtype: str = "") -> str:
  dtype = explicit_dtype or "fp8"

  # Contract: speedruns default to fp8 on B200/SM100; H100/SM90 is BF16-only bring-up.
  if explicit_dtype:
    return dtype

  arch = os.environ.get("NMOE_CUDA_ARCH", "").strip().lower()
  if arch in ("90", "sm90", "9.0"):
    return "bf16"
  try:
    import torch
    if torch.cuda.is_available():
      cap = tuple(torch.cuda.get_device_capability())
      if cap == (9, 0):
        return "bf16"
  except Exception:
    pass
  return dtype


def _build_speedrun_train_cmd(
  config: str,
  *,
  explicit_dtype: str = "",
  activation: str = "",
  steps: int = 0,
  experiment_id: str = "",
  extra_overrides: dict[str, str] | None = None,
  prepare_data: bool = True,
  eval_enabled: bool = True,
  train_tokens_budget: str | None = None,
  val_tokens_budget: str | None = None,
) -> tuple[list[str], str, str]:
  config_path = _resolve_speedrun_config(config)
  dtype = _resolve_speedrun_dtype(explicit_dtype)

  if prepare_data:
    speedrun_dir = ensure_speedrun_data(
      train_tokens_budget=train_tokens_budget or "10B",
      val_tokens_budget=val_tokens_budget or "10485760",
    )
    if eval_enabled:
      ensure_eval_bundle()
  else:
    speedrun_dir = DATA_DIR / "speedrun"

  cmd = [
    *_torchrun_cmd(), "--nproc_per_node", NPROC,
    "-m", "nmoe.train",
    config_path,
    f"--dtype={dtype}",
    f"--data_root={DATA_DIR}",
    f"--data_path={speedrun_dir / 'train'}",
    f"--validation_data_path={speedrun_dir / 'val'}",
    f"--experiments_db={_experiments_db_path()}",
  ]
  if eval_enabled:
    cmd.extend([
      "--eval_enabled=true",
      "--eval_tasks=core",
    ])
  else:
    cmd.append("--eval_enabled=false")
  if experiment_id:
    cmd.append(f"--experiment_id={experiment_id}")
    checkpoint_dir = _campaign_checkpoint_dir(experiment_id)
    if checkpoint_dir is not None:
      cmd.append(f"--checkpoint_dir={checkpoint_dir}")
  if steps > 0:
    cmd.append(f"--steps={steps}")
  if activation:
    if activation not in ("swiglu", "relu_squared", "squared_reglu"):
      console.print(f"[red]Unknown activation: {activation}[/red]")
      console.print("[yellow]Available: swiglu, relu_squared, squared_reglu[/yellow]")
      raise typer.Exit(1)
    cmd.append(f"--activation={activation}")
  for key, value in (extra_overrides or {}).items():
    cmd.append(f"--{key}={value}")

  return cmd, dtype, config_path


def _latest_experiment_run(experiments_db: Path, experiment_id: str) -> dict | None:
  if not experiments_db.exists():
    return None
  conn = sqlite3.connect(str(experiments_db))
  try:
    row = conn.execute(
      """
      SELECT id, status, started_at, ended_at, results_json
      FROM runs
      WHERE experiment_id = ?
      ORDER BY started_at DESC
      LIMIT 1
      """,
      (experiment_id,),
    ).fetchone()
  finally:
    conn.close()
  if row is None:
    return None

  results_json = row[4]
  results = {}
  if results_json:
    try:
      results = json.loads(results_json)
    except Exception:
      results = {}
  return {
    "run_id": row[0],
    "status": row[1],
    "started_at": row[2],
    "ended_at": row[3],
    "results": results,
  }


def _leaderboard_entry_for_experiment(experiment_id: str) -> dict | None:
  if not LEADERBOARD_PATH.exists():
    return None
  try:
    data = json.loads(LEADERBOARD_PATH.read_text())
  except Exception:
    return None

  runs = data.get("runs", [])
  if not isinstance(runs, list):
    return None
  for entry in runs:
    if isinstance(entry, dict) and entry.get("experiment_id") == experiment_id:
      return entry
  return None


def _campaign_metrics_for_experiment(experiment_id: str, experiments_db: Path) -> tuple[dict | None, dict | None, dict]:
  run = _latest_experiment_run(experiments_db, experiment_id)
  leaderboard_entry = _leaderboard_entry_for_experiment(experiment_id)

  metrics: dict[str, object] = {}
  if run is not None:
    metrics.update(run.get("results", {}))
  if leaderboard_entry is not None:
    metrics.update(leaderboard_entry)
    if "core_score" in leaderboard_entry and "core" not in metrics:
      metrics["core"] = leaderboard_entry["core_score"]
  if "core" in metrics and "core_score" not in metrics:
    metrics["core_score"] = metrics["core"]
  if "val_loss_to_target" in metrics and "final_valid_loss" not in metrics:
    metrics["final_valid_loss"] = metrics["val_loss_to_target"]

  return run, leaderboard_entry, metrics


def _display_path(path: Path) -> str:
  try:
    return str(path.relative_to(NMOE_ROOT))
  except ValueError:
    return str(path)


def _speedrun_leaderboard():
  """Print speedrun leaderboard from LEADERBOARD.json."""
  import json

  if not LEADERBOARD_PATH.exists():
    console.print("[yellow]No speedrun results yet. Run: n speedrun dense[/yellow]")
    return

  try:
    data = json.loads(LEADERBOARD_PATH.read_text())
    runs = data.get("runs", [])
  except Exception as e:
    console.print(f"[red]Error reading leaderboard: {e}[/red]")
    return

  if not runs:
    console.print("[yellow]No speedrun results yet. Run: n speedrun dense[/yellow]")
    return

  # For safety, sort here too (train.py also writes sorted).
  runs = list(runs)
  runs.sort(key=lambda r: (float(r.get("wall_time_s") or 1e18), int(r.get("steps") or 1e18)))

  console.print(f"\n{'═' * 80}")
  console.print("  nmoe Speedrun Leaderboard")
  console.print(f"{'═' * 80}")
  console.print(f"  {'Config':<12} {'HW':<6} {'Dtype':<8} {'Time':>8} {'Steps':>6} {'Loss':>8} {'CORE':>8} {'Tokens':>8} {'Date':<12}")
  console.print(f"  {'─' * 74}")

  for r in runs[:20]:
    config = r.get('config', '?')
    hw = r.get('hardware', '?')
    dtype = r.get('dtype', '?')
    loss = r.get('final_loss', 0)
    core = r.get('core_score', 0)
    tokens = r.get('tokens', 0)
    steps = r.get('steps', 0)
    wall_time = r.get('wall_time_s', 0)
    date = r.get('date', '?')[:10]

    tokens_str = f"{tokens/1e9:.1f}B" if tokens else "?"
    time_str = f"{wall_time/60:.1f}m" if wall_time else "?"
    steps_str = f"{int(steps)}" if steps else "?"
    loss_str = f"{loss:.4f}" if loss else "?"
    core_str = f"{core:.3f}" if core else "?"

    console.print(f"  {config:<12} {hw:<6} {dtype:<8} {time_str:>8} {steps_str:>6} {loss_str:>8} {core_str:>8} {tokens_str:>8} {date:<12}")

  console.print(f"{'═' * 80}\n")


@app.command()
def speedrun(
  config: str = typer.Argument("super", help="Config: dense, moe, super, ultra"),
  bf16: bool = typer.Option(False, "--bf16", help="Use bf16 instead of nvfp4"),
  fp8: bool = typer.Option(False, "--fp8", help="Use fp8 instead of nvfp4"),
  activation: str = typer.Option("", "--activation", "-a", help="Activation: swiglu, relu_squared, squared_reglu"),
  steps: int = typer.Option(0, "--steps", "-s", help="Override steps (0=use config default)"),
  attach: bool = typer.Option(False, "--attach", help="Attach to existing run"),
  leaderboard: bool = typer.Option(False, "--leaderboard", "-l", help="Show leaderboard"),
  no_nmon: bool = typer.Option(False, "--no-nmon", help="Don't open nmon TUI (run in foreground; headless/CI friendly)"),
):
  """Run speedrun benchmark. Opens nmon for monitoring by default.

  Examples:
    n speedrun dense          # Dense baseline (nvfp4)
    n speedrun moe            # MoE-64 (nvfp4)
    n speedrun moe --bf16     # MoE-64 (bf16)
    n speedrun super          # MoE-256 (nvfp4)
    n speedrun ultra          # MoE-4096 (nvfp4)
    n speedrun --leaderboard  # Show results
    n speedrun dense --activation=relu_squared  # Ablation
    n speedrun super --no-nmon  # Foreground run (no TUI)
  """
  if leaderboard:
    _speedrun_leaderboard()
    return

  if attach:
    open_nmon()
    return

  # Determine dtype
  if bf16 and fp8:
    console.print("[red]Cannot use both --bf16 and --fp8[/red]")
    raise typer.Exit(1)
  dtype_explicit = "bf16" if bf16 else ("fp8" if fp8 else "")
  cmd, dtype, config_path = _build_speedrun_train_cmd(
    config,
    explicit_dtype=dtype_explicit,
    activation=activation,
    steps=steps,
  )

  console.print(f"\n[bold]Speedrun: {config} ({dtype})[/bold]")
  console.print(f"[dim]Config: {config_path}[/dim]\n")

  if no_nmon:
    # Headless/CI friendly: keep logs/exit code in the invoking shell.
    run(cmd)
    return

  # Default: start training in background and open nmon.
  run_background(cmd)
  time.sleep(2)
  open_nmon()


@campaign_app.command(name="list")
def campaign_list():
  """List available bounded autoresearch campaigns."""
  paths = discover_campaign_specs(NMOE_ROOT)
  if not paths:
    console.print("[yellow]No campaigns found[/yellow]")
    return

  console.print("[bold]Available campaigns:[/bold]\n")
  for path in paths:
    try:
      spec = load_campaign(NMOE_ROOT, str(path))
      stages = ", ".join(sorted(spec.budget.keys()))
      console.print(
        f"  {spec.name:<28} runner={spec.runner:<8} metric={spec.objective.primary_metric:<12} "
        f"stages=({stages})  {_display_path(spec.path)}",
        markup=False,
      )
    except CampaignError as e:
      console.print(f"  [red]{path.name}[/red]  invalid: {e}")


@campaign_app.command()
def show(
  campaign: str = typer.Argument(..., help="Campaign name or TOML path"),
):
  """Show the fully-resolved campaign spec."""
  try:
    spec = load_campaign(NMOE_ROOT, campaign)
  except CampaignError as e:
    console.print(f"[red]{e}[/red]")
    raise typer.Exit(1)

  console.print(json.dumps(spec.to_dict(), indent=2, sort_keys=True))


def _execute_campaign_speedrun(
  spec,
  *,
  stage_cfg,
  candidate_id: str,
  requested_overrides: dict[str, str],
  receipt_root: Path,
  proposal: dict | None = None,
  dry_run: bool = False,
) -> tuple[Path, dict, int]:
  if spec.runner != "speedrun" or spec.speedrun is None:
    console.print(f"[red]campaign runner {spec.runner!r} is not implemented in phase 1[/red]")
    raise typer.Exit(1)

  overrides = dict(requested_overrides)
  validate_candidate_overrides(spec, overrides)

  activation = spec.speedrun.activation
  if "activation" in overrides:
    activation = overrides.pop("activation")

  explicit_dtype = "" if spec.speedrun.dtype == "auto" else spec.speedrun.dtype
  if "dtype" in overrides:
    explicit_dtype = overrides.pop("dtype")

  steps = stage_cfg.steps or 0
  if "steps" in overrides:
    try:
      steps = int(overrides.pop("steps"))
    except ValueError:
      console.print("[red]campaign override steps must be an integer[/red]")
      raise typer.Exit(1)

  baseline = select_baseline(
    spec,
    stage=stage_cfg.name,
    receipt_dir=receipt_root,
    leaderboard_path=LEADERBOARD_PATH,
  )

  ts = int(time.time())
  resolved_candidate_id = candidate_id or f"{spec.name}_{stage_cfg.name}_{ts}"
  experiment_id = f"campaign_{spec.name}_{stage_cfg.name}_{ts}"

  cmd, dtype, config_path = _build_speedrun_train_cmd(
    spec.speedrun.config,
    explicit_dtype=explicit_dtype,
    activation=activation,
    steps=steps,
    experiment_id=experiment_id,
    extra_overrides=overrides,
    prepare_data=not dry_run,
    eval_enabled=spec.speedrun.eval_enabled,
    train_tokens_budget=spec.speedrun.train_tokens,
    val_tokens_budget=spec.speedrun.val_tokens,
  )

  receipt_path = new_receipt_path(
    spec,
    stage=stage_cfg.name,
    candidate_id=resolved_candidate_id,
    receipt_dir=receipt_root,
  )
  experiments_db = _experiments_db_path()
  receipt = {
    "schema_version": 1,
    "campaign_name": spec.name,
    "campaign_kind": spec.kind,
    "candidate_id": resolved_candidate_id,
    "stage": stage_cfg.name,
    "status": "planned",
    "description": spec.description,
    "spec_path": _display_path(spec.path),
    "receipt_path": _display_path(receipt_path),
    "runner": spec.runner,
    "runner_config": {
      "config": spec.speedrun.config,
      "config_path": config_path,
      "dtype": dtype,
      "activation": activation,
      "eval_enabled": spec.speedrun.eval_enabled,
      "train_tokens": spec.speedrun.train_tokens,
      "val_tokens": spec.speedrun.val_tokens,
    },
    "experiment_id": experiment_id,
    "budget": {
      "steps": stage_cfg.steps,
      "max_wall_s": stage_cfg.max_wall_s,
    },
    "mutation": {
      "tier": spec.mutation.tier,
      "allowed_overrides": list(spec.mutation.allowed_overrides),
      "allowed_files": list(spec.mutation.allowed_files),
    },
    "baseline": baseline,
    "proposal": proposal,
    "worker": {
      "id": os.environ.get("NMOE_AUTORESEARCH_WORKER_ID", "").strip() or os.environ.get("HOSTNAME", "").strip() or None,
    },
    "overrides": dict(requested_overrides),
    "command": cmd,
    "started_at": None,
    "ended_at": None,
    "metrics": {},
    "run": None,
    "leaderboard_entry": None,
    "decision": None,
  }

  console.print(f"\n[bold]Campaign:[/bold] {spec.name} [{stage_cfg.name}]")
  console.print(f"[dim]Spec: {_display_path(spec.path)}[/dim]")
  console.print(f"[dim]Config: {config_path} ({dtype})[/dim]")
  if baseline is None:
    console.print("[dim]Baseline: none[/dim]")
  else:
    baseline_metrics = baseline.get("metrics", {})
    baseline_value = baseline_metrics.get(spec.objective.primary_metric)
    console.print(
      f"[dim]Baseline: {baseline.get('source')} "
      f"{spec.objective.primary_metric}={baseline_value}[/dim]"
    )

  if dry_run:
    console.print("")
    console.print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt_path, receipt, 0

  receipt["status"] = "running"
  receipt["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
  write_json(receipt_path, receipt)

  exit_code = 0
  try:
    run(cmd, timeout_s=stage_cfg.max_wall_s)
  except typer.Exit as exc:
    exit_code = int(exc.exit_code)

  run_record, leaderboard_entry, metrics = _campaign_metrics_for_experiment(experiment_id, experiments_db)
  receipt["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
  receipt["metrics"] = metrics
  receipt["run"] = run_record
  receipt["leaderboard_entry"] = leaderboard_entry

  if exit_code != 0:
    receipt["status"] = "failed"
    if metrics:
      receipt["decision"] = evaluate_metrics(spec, metrics, baseline)
    else:
      receipt["decision"] = {
        "primary_metric": spec.objective.primary_metric,
        "direction": spec.objective.direction,
        "current_value": None,
        "baseline_value": None,
        "constraints_pass": False,
        "constraint_failures": ["runner exited non-zero"],
        "improved": False,
        "kept": False,
        "reason": "runner exited non-zero",
      }
    write_json(receipt_path, receipt)
    return receipt_path, receipt, exit_code

  decision = evaluate_metrics(spec, metrics, baseline)
  receipt["status"] = "completed"
  receipt["decision"] = decision
  write_json(receipt_path, receipt)

  verdict = "keep" if decision["kept"] else "discard"
  console.print(
    f"\n[bold]Decision:[/bold] {verdict} "
    f"({spec.objective.primary_metric}={decision['current_value']}, reason={decision['reason']})"
  )
  console.print(f"[dim]Receipt: {_display_path(receipt_path)}[/dim]")
  return receipt_path, receipt, 0


@campaign_app.command(name="run")
def campaign_run(
  campaign: str = typer.Argument(..., help="Campaign name or TOML path"),
  stage: str = typer.Option("smoke", "--stage", "-s", help="Budget stage to run"),
  candidate: str = typer.Option("", "--candidate", "-c", help="Candidate identifier for receipts"),
  set_: list[str] = typer.Option(None, "--set", help="Runtime override(s): key=value"),
  receipt_dir: str = typer.Option("", "--receipt-dir", help="Override receipt directory"),
  dry_run: bool = typer.Option(False, "--dry-run", help="Print resolved command and exit"),
):
  """Run a bounded campaign using the canonical nmoe runner."""
  try:
    spec = load_campaign(NMOE_ROOT, campaign)
    stage_cfg = spec.stage(stage)
    overrides = parse_candidate_overrides(list(set_ or []))
  except CampaignError as e:
    console.print(f"[red]{e}[/red]")
    raise typer.Exit(1)

  receipt_root = resolve_receipt_dir(spec, receipt_dir or None)
  _, _, exit_code = _execute_campaign_speedrun(
    spec,
    stage_cfg=stage_cfg,
    candidate_id=candidate,
    requested_overrides=overrides,
    receipt_root=receipt_root,
    proposal=None,
    dry_run=dry_run,
  )
  if exit_code != 0:
    raise typer.Exit(exit_code)


@campaign_app.command(name="auto")
def campaign_auto(
  campaign: str = typer.Argument(..., help="Campaign name or TOML path"),
  stage: str = typer.Option("benchmark", "--stage", "-s", help="Budget stage to run"),
  receipt_dir: str = typer.Option("", "--receipt-dir", help="Override receipt directory"),
  max_trials: int = typer.Option(0, "--max-trials", help="Override search.max_trials"),
  max_no_improve: int = typer.Option(0, "--max-no-improve", help="Stop after this many non-kept trials"),
  dry_run: bool = typer.Option(False, "--dry-run", help="Print the next autonomous candidate and exit"),
):
  """Run a TOML-defined autonomous config search loop."""
  try:
    spec = load_campaign(NMOE_ROOT, campaign)
    stage_cfg = spec.stage(stage)
  except CampaignError as e:
    console.print(f"[red]{e}[/red]")
    raise typer.Exit(1)

  if spec.search is None:
    console.print(f"[red]campaign {spec.name} does not define a [search] section[/red]")
    raise typer.Exit(1)

  receipt_root = resolve_receipt_dir(spec, receipt_dir or None)
  trial_limit = max_trials or spec.search.max_trials
  stop_after_no_improve = max_no_improve or spec.search.max_no_improve
  worker_id = os.environ.get("NMOE_AUTORESEARCH_WORKER_ID", "").strip() or os.environ.get("HOSTNAME", "").strip() or None
  if trial_limit <= 0:
    console.print("[red]max_trials must be > 0[/red]")
    raise typer.Exit(1)

  console.print(
    f"\n[bold]Autoresearch:[/bold] {spec.name} [{stage_cfg.name}] "
    f"strategy={spec.search.strategy} max_trials={trial_limit}"
  )
  if stop_after_no_improve is not None:
    console.print(f"[dim]Stop after {stop_after_no_improve} non-kept trial(s)[/dim]")

  proposal = propose_next_candidate(
    spec,
    stage=stage_cfg.name,
    receipt_dir=receipt_root,
    leaderboard_path=LEADERBOARD_PATH,
  )
  if proposal is None:
    console.print("[yellow]No unexplored autonomous candidate remains for this campaign[/yellow]")
    return

  if dry_run:
    console.print(json.dumps({
      "campaign": spec.name,
      "stage": stage_cfg.name,
      "candidate_id": proposal.candidate_id,
      "reason": proposal.reason,
      "overrides": proposal.overrides,
    }, indent=2, sort_keys=True))
    return

  trials_run = 0
  consecutive_non_kept = 0
  last_exit_code = 0
  while trials_run < trial_limit:
    proposal = propose_next_candidate(
      spec,
      stage=stage_cfg.name,
      receipt_dir=receipt_root,
      leaderboard_path=LEADERBOARD_PATH,
    )
    if proposal is None:
      console.print("\n[yellow]Autoresearch exhausted the configured search space[/yellow]")
      break

    console.print(
      f"\n[bold]Autoresearch Trial {trials_run + 1}/{trial_limit}:[/bold] "
      f"{proposal.candidate_id}"
    )
    console.print(f"[dim]{proposal.reason}[/dim]")
    proposal_payload = {
      "strategy": spec.search.strategy,
      "reason": proposal.reason,
      "axis": proposal.axis,
      "current_value": proposal.current_value,
      "proposed_value": proposal.proposed_value,
      "worker_id": worker_id,
    }
    claim = claim_candidate(
      spec,
      stage=stage_cfg.name,
      candidate_id=proposal.candidate_id,
      overrides=proposal.overrides,
      receipt_dir=receipt_root,
      proposal=proposal_payload,
      worker_id=worker_id,
    )
    if claim is None:
      console.print(f"[dim]candidate already claimed elsewhere: {proposal.candidate_id}[/dim]")
      continue

    try:
      _, receipt, exit_code = _execute_campaign_speedrun(
        spec,
        stage_cfg=stage_cfg,
        candidate_id=proposal.candidate_id,
        requested_overrides=proposal.overrides,
        receipt_root=receipt_root,
        proposal=proposal_payload,
        dry_run=False,
      )
    finally:
      release_candidate_claim(
        spec,
        stage=stage_cfg.name,
        candidate_id=proposal.candidate_id,
        receipt_dir=receipt_root,
      )
    trials_run += 1
    last_exit_code = exit_code

    decision = receipt.get("decision") or {}
    if bool(decision.get("kept")):
      consecutive_non_kept = 0
    else:
      consecutive_non_kept += 1

    if stop_after_no_improve is not None and consecutive_non_kept >= stop_after_no_improve:
      console.print(
        f"\n[yellow]Stopping after {consecutive_non_kept} consecutive non-kept trial(s)[/yellow]"
      )
      break

  if last_exit_code != 0:
    raise typer.Exit(last_exit_code)


@app.command()
def research():
  """Open JupyterLab for research. Use: from nmoe.research import lab"""
  open_jupyter()


@app.command()
def train(
  config: str = typer.Argument("configs/moonlight.toml", help="Training config"),
  attach: bool = typer.Option(False, "--attach", "-a", help="Attach to existing run"),
):
  """Production training. Starts nviz dashboard."""
  start_nviz_with_tunnel()
  if not attach:
    data_path = ensure_data("fineweb_train", "10B")
    start_training(config, data_path_override=data_path, background=False)
  else:
    console.print("[blue]Attached to nviz. Press Ctrl+C to stop.[/blue]")
    try:
      while True:
        time.sleep(1)
    except KeyboardInterrupt:
      pass


# -----------------------------------------------------------------------------
# Commands: List runs
# -----------------------------------------------------------------------------

@app.command(name="list")
def list_runs(limit: int = typer.Option(20, "--limit", "-n", help="Max runs to show")):
  """List available runs."""
  from datetime import datetime
  experiments_db = _experiments_db_path()
  metrics_dir = DATA_DIR / "metrics"

  run_info = []

  # Prefer experiments.db as source of truth
  if experiments_db.exists():
    try:
      import sqlite3
      conn = sqlite3.connect(str(experiments_db))
      cursor = conn.execute("""
        SELECT run, status, started_at, ended_at
        FROM runs
        ORDER BY started_at DESC
        LIMIT ?
      """, (limit * 2,))  # Fetch extra to account for filtering
      for row in cursor:
        run_id, status, started_at, ended_at = row
        try:
          ts = datetime.fromisoformat(started_at.replace("Z", "+00:00")).timestamp()
        except Exception:
          ts = 0
        run_info.append((run_id, ts, status))
      conn.close()
    except Exception as e:
      console.print(f"[yellow]experiments.db error: {e}, falling back to metrics dir[/yellow]")
      run_info = []

  # Fallback: scan /data/metrics
  if not run_info and metrics_dir.exists():
    run_dirs = [d for d in metrics_dir.iterdir() if d.is_dir()]
    for run_dir in run_dirs:
      run_id = run_dir.name
      last_step = 0
      last_ts = 0.0

      # Authoritative live store: step_XXXXXXXX.parquet (rank 0 only).
      parquet = sorted(run_dir.glob("step_*.parquet"))
      if parquet:
        newest = max(parquet, key=lambda p: p.stat().st_mtime)
        last_ts = newest.stat().st_mtime
        try:
          # step_00001234.parquet
          stem = newest.name.split(".", 1)[0]
          last_step = int(stem.split("_", 1)[1])
        except Exception:
          last_step = 0
      else:
        # Backward-compat (older runs).
        db_path = run_dir / "rank_0.duckdb"
        if db_path.exists():
          last_ts = db_path.stat().st_mtime
          last_step = 0

      status = f"step {last_step:,}" if last_step else "no data"
      run_info.append((run_id, last_ts, status))

    run_info.sort(key=lambda x: x[1], reverse=True)

  if not run_info:
    console.print("[yellow]No runs found[/yellow]")
    return

  console.print("[bold]Available runs:[/bold]\n")
  for i, (run_id, ts, status) in enumerate(run_info[:limit]):
    time_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "unknown"
    marker = "[green]← latest[/green]" if i == 0 else ""
    console.print(f"  {run_id:<30} {time_str}  {status:<12} {marker}")

  if len(run_info) > limit:
    console.print(f"\n  ... and {len(run_info) - limit} more (use --limit to show more)")


# -----------------------------------------------------------------------------
# Commands: Monitor (attach to latest run, --run <id> to specify)
# -----------------------------------------------------------------------------

@app.command(name="mon")
def mon(run_id: str = typer.Option(None, "--run", "-r", help="Run ID to monitor")):
  """TUI monitor. Attaches to latest run."""
  if run_id:
    os.environ["NMOE_RUN"] = run_id
  open_nmon()


@app.command(name="viz")
def viz(
  run_id: str = typer.Option(None, "--run", "-r", help="Run ID to monitor"),
  dev: bool = typer.Option(False, "--dev", "-d", help="Dev mode with hot reload"),
):
  """Web dashboard. Attaches to latest run."""
  if run_id:
    os.environ["NMOE_RUN"] = run_id
  start_nviz_with_tunnel(dev=dev)
  console.print("[blue]viz running. Press Ctrl+C to stop.[/blue]")
  try:
    while True:
      time.sleep(1)
  except KeyboardInterrupt:
    pass


@app.command(name="nb")
def nb(run_id: str = typer.Option(None, "--run", "-r", help="Run ID context")):
  """Jupyter notebook. Attaches to latest run context."""
  if run_id:
    os.environ["NMOE_RUN"] = run_id
  open_jupyter()


def main():
  app()


if __name__ == "__main__":
  main()
