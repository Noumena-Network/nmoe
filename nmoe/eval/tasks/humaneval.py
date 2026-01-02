from __future__ import annotations

from typing import Any

from nmoe.eval.execution import humaneval_check


class HumanEval:
    name = "HumanEval"

    def __init__(self, *, split: str = "test"):
        from datasets import load_dataset

        self.ds = load_dataset("openai/openai_humaneval", split=split)

    def __len__(self) -> int:
        return int(len(self.ds))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return dict(self.ds[int(idx)])

    def format_prompt(self, ex: dict[str, Any]) -> str:
        return str(ex.get("prompt", ""))

    def evaluate(self, ex: dict[str, Any], completion: str) -> bool:
        prompt = str(ex.get("prompt", ""))
        tests = str(ex.get("test", ""))
        if not prompt or not tests:
            return False
        res = humaneval_check(prompt, completion, tests, timeout_s=10.0)
        return bool(res.ok)
