from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ExecResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int

    @property
    def success(self) -> bool:
        # Backward-compatible alias used by older task implementations.
        return bool(self.ok)


def _strip_code_fences(s: str) -> str:
    t = s.strip()
    if t.startswith("```"):
        # Remove first fence line and trailing fence if present.
        lines = t.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip() + "\n"
    return t + ("\n" if not t.endswith("\n") else "")


def run_python(
    code: str,
    *,
    timeout_s: float = 5.0,
    python: str = "python",
    cwd: Optional[str] = None,
) -> ExecResult:
    """Execute code in a subprocess with a timeout.

    This is a pragmatic sandbox for HumanEval-style unit tests. It is not a
    perfect security boundary; it is intended for trusted evaluation inputs.
    """
    with tempfile.TemporaryDirectory(prefix="nmoe_eval_") as d:
        path = Path(d) / "prog.py"
        path.write_text(code, encoding="utf-8")
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        try:
            p = subprocess.run(
                [python, str(path)],
                cwd=cwd or d,
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
            return ExecResult(ok=(p.returncode == 0), stdout=p.stdout, stderr=p.stderr, returncode=p.returncode)
        except subprocess.TimeoutExpired as e:
            return ExecResult(ok=False, stdout=e.stdout or "", stderr=(e.stderr or "") + "\nTIMEOUT\n", returncode=124)


def humaneval_check(
    prompt: str,
    completion: str,
    tests: str,
    *,
    timeout_s: float = 5.0,
) -> ExecResult:
    """Run HumanEval tests for a prompt+completion."""
    code = prompt + _strip_code_fences(completion) + "\n" + tests + "\n"
    return run_python(code, timeout_s=timeout_s)


def execute_code(code: str, timeout: float = 5.0) -> ExecResult:
    """Backward-compatible entrypoint for HumanEval task implementations."""
    return run_python(code, timeout_s=float(timeout))
