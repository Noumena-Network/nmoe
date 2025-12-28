from __future__ import annotations

"""
Lightweight dataset adapters for CORE eval tasks.

Each adapter yields standardized records:
  - choices: {prompt: str, options: list[str], label: int}
  - span:    {prompt: str, answers: list[str]}

Sources are encoded as strings like:
  hf:<dataset>:<subset>:<split>
    examples:
      hf:ai2_arc:ARC-Easy:test
      hf:hellaswag:validation
      hf:hendrycks_test:all:test
      hf:squad:plain_text:validation

Adapters are intentionally compact and robust to minor schema variations.
"""

from typing import Dict, Iterable, Iterator, List, Optional, Tuple


def _parse_source(src: str) -> Tuple[str, List[str]]:
    parts = src.split(":")
    scheme = parts[0]
    args = parts[1:]
    return scheme, args


# ----------------------
# Choices task adapters
# ----------------------

def iter_choices(name: str, source: str, max_examples: int) -> Iterator[Dict]:
    # Lazy import so `python -c "import nmoe.eval.*"` works in minimal environments.
    from datasets import load_dataset

    scheme, args = _parse_source(source)
    assert scheme == "hf", f"Unsupported scheme for choices: {scheme}"

    # Resolve dataset / subset / split
    if len(args) == 3:
        dataset, subset, split = args
        ds = load_dataset(dataset, subset, split=split)
    elif len(args) == 2:
        dataset, split = args
        subset = None
        ds = load_dataset(dataset, split=split)
    else:
        raise ValueError(f"Bad source spec: {source}")

    n = 0
    dataset_base = dataset.split("/")[-1]
    subset_name = subset
    for ex in ds:
        try:
            if dataset_base == "ai2_arc":  # ARC Easy/Challenge (allenai/ai2_arc)
                q = ex.get("question")
                choices = ex.get("choices", {})
                opts = choices.get("text") or []
                labels = choices.get("label") or []
                key = ex.get("answerKey")
                idx = labels.index(key) if key in labels else None
                if idx is None:
                    continue
                prompt = f"{q}\nChoices:\n" + "\n".join([f"{chr(65+i)}. {o}" for i, o in enumerate(opts)]) + "\nAnswer:"
                yield {"prompt": prompt, "options": opts, "label": idx}

            elif dataset_base == "hellaswag":  # HellaSwag (Rowan/hellaswag)
                ctx = ex.get("ctx") or ex.get("context") or ""
                endings = ex.get("endings") or []
                label = int(ex.get("label")) if ex.get("label") is not None else 0
                prompt = f"{ctx}\nEnding:"
                yield {"prompt": prompt, "options": list(endings), "label": label}

            elif dataset_base in ("hendrycks_test", "mmlu") or dataset == "cais/mmlu":  # MMLU variants
                q = ex.get("question") or ex.get("input")
                opts = ex.get("choices") or ex.get("options")
                ans = ex.get("answer")
                # cais/mmlu uses integer index (0-3), hendrycks uses letter A-D
                if isinstance(ans, str) and ans in "ABCD":
                    idx = "ABCD".index(ans)
                elif isinstance(ans, int):
                    idx = ans
                else:
                    try:
                        idx = int(ans)
                    except (TypeError, ValueError):
                        continue
                if idx is None or not isinstance(opts, list) or idx >= len(opts):
                    continue
                prompt = f"{q}\nChoices:\n" + "\n".join([f"{chr(65+i)}. {o}" for i, o in enumerate(opts)]) + "\nAnswer:"
                yield {"prompt": prompt, "options": opts, "label": idx}

            elif dataset_base == "MMLU-Pro":  # TIGER-Lab/MMLU-Pro (10 choices A-J)
                q = ex.get("question")
                opts = ex.get("options")
                # Use answer_index directly if available, else parse letter
                ans_idx = ex.get("answer_index")
                ans = ex.get("answer")
                if not q or not opts:
                    continue
                if ans_idx is not None:
                    idx = int(ans_idx)
                elif isinstance(ans, str) and ans in "ABCDEFGHIJ":
                    idx = "ABCDEFGHIJ".index(ans)
                else:
                    continue
                if idx >= len(opts):
                    continue
                prompt = f"{q}\nChoices:\n" + "\n".join([f"{chr(65+i)}. {o}" for i, o in enumerate(opts)]) + "\nAnswer:"
                yield {"prompt": prompt, "options": opts, "label": idx}

            elif dataset_base == "openbookqa":  # allenai/openbookqa
                q = ex.get("question_stem") or ex.get("question")
                choices = ex.get("choices") or {}
                opts = choices.get("text") or []
                labels = choices.get("label") or []
                key = ex.get("answerKey")
                idx = labels.index(key) if key in labels else None
                if idx is None:
                    continue
                prompt = f"{q}\nChoices:\n" + "\n".join([f"{chr(65+i)}. {o}" for i, o in enumerate(opts)]) + "\nAnswer:"
                yield {"prompt": prompt, "options": opts, "label": idx}

            elif dataset_base == "piqa":
                goal = ex.get("goal")
                sol1 = ex.get("sol1"); sol2 = ex.get("sol2")
                label = int(ex.get("label"))
                prompt = f"Goal: {goal}\nSelect the better solution:"
                yield {"prompt": prompt, "options": [sol1, sol2], "label": label}

            elif dataset_base == "social_i_qa":
                q = ex.get("question")
                opts = [ex.get("answerA"), ex.get("answerB"), ex.get("answerC")]
                label = int(ex.get("label")) - 1
                prompt = f"{q}\nChoices:\nA. {opts[0]}\nB. {opts[1]}\nC. {opts[2]}\nAnswer:"
                yield {"prompt": prompt, "options": opts, "label": label}

            elif dataset_base == "winogrande":  # allenai/winogrande
                sent = ex.get("sentence")
                o1 = ex.get("option1"); o2 = ex.get("option2")
                label = int(ex.get("answer")) - 1
                prompt = f"Fill in the blank: {sent}\nOptions:\nA. {o1}\nB. {o2}\nAnswer:"
                yield {"prompt": prompt, "options": [o1, o2], "label": label}

            elif dataset_base == "boolq":  # google/boolq
                passage = ex.get("passage")
                question = ex.get("question")
                ans = bool(ex.get("answer"))
                opts = ["yes", "no"]
                label = 0 if ans else 1
                prompt = f"{passage}\nQuestion: {question}\nAnswer (yes/no):"
                yield {"prompt": prompt, "options": opts, "label": label}

            elif dataset_base == "super_glue" and (subset_name in ("copa",)):
                prem = ex.get("premise"); c1 = ex.get("choice1"); c2 = ex.get("choice2")
                label = int(ex.get("label"))
                prompt = f"Premise: {prem}\nChoose the more plausible alternative:"
                yield {"prompt": prompt, "options": [c1, c2], "label": label}

            elif dataset_base == "story_cloze":
                sents = [ex.get(f"input_sentence_{i}") for i in range(1, 5)]
                o1 = ex.get("sentence_quiz1"); o2 = ex.get("sentence_quiz2")
                label = int(ex.get("answer_right_ending")) - 1
                story = " ".join([s for s in sents if s])
                prompt = f"Story: {story}\nSelect the best ending:"
                yield {"prompt": prompt, "options": [o1, o2], "label": label}

            else:
                # Unknown schema; skip
                continue

            n += 1
            if n >= max_examples:
                break
        except Exception:
            # Robust to occasional bad rows
            continue


# -------------------
# Span task adapters
# -------------------

def iter_span(name: str, source: str, max_examples: int) -> Iterator[Dict]:
    # Lazy import so `python -c "import nmoe.eval.*"` works in minimal environments.
    from datasets import load_dataset

    scheme, args = _parse_source(source)
    assert scheme == "hf", f"Unsupported scheme for span: {scheme}"
    if len(args) == 3:
        dataset, subset, split = args
        ds = load_dataset(dataset, subset, split=split, trust_remote_code=True)
    elif len(args) == 2:
        dataset, split = args
        ds = load_dataset(dataset, split=split, trust_remote_code=True)
    else:
        raise ValueError(f"Bad source spec: {source}")

    n = 0
    for ex in ds:
        try:
            # Identify by dataset name from the source rather than builder metadata
            if dataset in ("squad", "rajpurkar/squad"):
                context = ex.get("context"); question = ex.get("question")
                answers = ex.get("answers", {}).get("text", [])
                if not context or not question:
                    continue
                prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
                yield {"prompt": prompt, "answers": list(answers) if isinstance(answers, list) else []}
            else:
                continue
            n += 1
            if n >= max_examples:
                break
        except Exception:
            continue
