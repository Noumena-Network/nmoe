from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'repro' / 'eval_0005_forward_floor.py'
SPEC = importlib.util.spec_from_file_location('eval_0005_forward_floor', MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _mk_checkpoint(path: Path) -> Path:
  path.mkdir(parents=True, exist_ok=True)
  (path / 'rd.pt').write_text('dense')
  (path / 'dp_rank_000.pt').write_text('expert')
  return path


def test_resolve_checkpoint_dir_accepts_iter_dirs(tmp_path: Path) -> None:
  _mk_checkpoint(tmp_path / 'iter_0000100')
  latest = _mk_checkpoint(tmp_path / 'iter_0009536')
  assert MODULE._resolve_checkpoint_dir(tmp_path) == latest


def test_resolve_checkpoint_dir_accepts_legacy_numeric_dirs(tmp_path: Path) -> None:
  _mk_checkpoint(tmp_path / '100')
  latest = _mk_checkpoint(tmp_path / '9536')
  assert MODULE._resolve_checkpoint_dir(tmp_path) == latest


def test_resolve_checkpoint_dir_accepts_direct_leaf_dir(tmp_path: Path) -> None:
  leaf = _mk_checkpoint(tmp_path / 'iter_0009536')
  assert MODULE._resolve_checkpoint_dir(leaf) == leaf


def test_resolve_checkpoint_dir_ignores_non_checkpoint_dirs(tmp_path: Path) -> None:
  (tmp_path / 'iter_0001000').mkdir()
  latest = _mk_checkpoint(tmp_path / 'iter_0002000')
  (tmp_path / 'notes').mkdir()
  assert MODULE._resolve_checkpoint_dir(tmp_path) == latest
