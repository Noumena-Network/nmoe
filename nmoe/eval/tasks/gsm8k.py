from __future__ import annotations

import re
from typing import Any, Optional


_ANS_RE = re.compile(r"####\\s*([-+]?\\d+(?:\\.\\d+)?)")
_NUM_RE = re.compile(r"[-+]?\\d+(?:\\.\\d+)?")


def _extract_gold(answer: str) -> Optional[str]:
    m = _ANS_RE.search(answer)
    if not m:
        return None
    return m.group(1).strip()


def _extract_pred(text: str) -> Optional[str]:
    # Prefer the final numeric-looking token.
    nums = _NUM_RE.findall(text.replace(",", ""))
    if not nums:
        return None
    return nums[-1].strip()


class GSM8K:
    name = "GSM8K"

    def __init__(self, *, subset: str = "main", split: str = "test"):
        from datasets import load_dataset

        self.ds = load_dataset("openai/gsm8k", subset, split=split)

    def __len__(self) -> int:
        return int(len(self.ds))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return dict(self.ds[int(idx)])

    def format_prompt(self, ex: dict[str, Any]) -> str:
        q = str(ex.get("question", "")).strip()
        return f"{q}\nAnswer:"

    def evaluate(self, ex: dict[str, Any], completion: str) -> bool:
        gold = _extract_gold(str(ex.get("answer", "")))
        pred = _extract_pred(completion)
        if gold is None or pred is None:
            return False
        try:
            # GSM8K gold is effectively integer-valued.
            return int(float(pred)) == int(float(gold))
        except Exception:
            return False
