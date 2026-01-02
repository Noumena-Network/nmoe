from __future__ import annotations

import re
from typing import Any, Optional


_CHOICE_RE = re.compile(r"\\b([A-D])\\b", re.IGNORECASE)


def _extract_choice(text: str) -> Optional[str]:
    m = _CHOICE_RE.search(text)
    if not m:
        return None
    return m.group(1).upper()


class GPQA:
    name = "GPQA"

    def __init__(self, *, split: str = "test"):
        from datasets import load_dataset

        try:
            # Public, MC-formatted GPQA variant (no auth required).
            self.ds = load_dataset("hendrydong/gpqa_diamond_mc", split=split)
        except Exception as e:
            raise RuntimeError(
                "Failed to load GPQA (HF dataset 'hendrydong/gpqa_diamond_mc'). "
                "Verify internet/HF access and try again."
            ) from e

    def __len__(self) -> int:
        return int(len(self.ds))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return dict(self.ds[int(idx)])

    def format_prompt(self, ex: dict[str, Any]) -> str:
        # Dataset encodes the full MC question+choices in a single "problem" field.
        problem = str(ex.get("problem", "")).strip()
        return f"{problem}\nAnswer (A-D):"

    def evaluate(self, ex: dict[str, Any], completion: str) -> bool:
        # Solution is typically like "\\boxed{D}".
        gold_raw = str(ex.get("solution", "")).strip()
        gold = _extract_choice(gold_raw)
        pred = _extract_choice(completion)
        return bool(gold is not None and pred is not None and pred == gold)
