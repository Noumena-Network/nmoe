from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from nmoe.checkpoint import build_states, load_state
from nmoe.config import Config, fingerprint


class _DummyLoader:
    dataset_version = "unit"
    tokenizer_id = "unit"

    def __init__(self) -> None:
        self.state = {"cursor": 7}

    def state_dict(self) -> dict:
        return dict(self.state)

    def load_state_dict(self, state: dict) -> None:
        self.state = dict(state)


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32))
        self.expert = torch.nn.Parameter(torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32))
        self.config = Config(
            dim=4,
            n_layers=1,
            n_heads=1,
            inter_dim=4,
            moe_inter_dim=4,
            n_routed_experts=1,
            n_activated_experts=1,
        )

    def param_sets(self):
        return [self.expert], [self.dense]


def _prime_adamw(opt: torch.optim.Optimizer, params: list[torch.nn.Parameter]) -> None:
    for p in params:
        p.grad = torch.ones_like(p)
    opt.step()
    opt.zero_grad(set_to_none=True)


class CheckpointResumeSurfaceTest(unittest.TestCase):
    def test_checkpoint_roundtrip_restores_extra_optimizer_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model = _DummyModel()
            loader = _DummyLoader()
            expert_opt = torch.optim.AdamW([model.expert], lr=0.1)
            _prime_adamw(expert_opt, [model.expert])
            extra_opt = torch.optim.AdamW([model.dense], lr=0.01)
            _prime_adamw(extra_opt, [model.dense])

            rd_state, dp_state = build_states(
                12,
                model,
                expert_opt,
                3456,
                loader,
                config_fingerprint=fingerprint(model.config),
                extra_optimizers={"muon": extra_opt},
            )
            it_dir = tmp_path / "iter_0000012"
            it_dir.mkdir()
            torch.save(rd_state, it_dir / "rd.pt")
            torch.save(dp_state, it_dir / "dp_rank_000.pt")

            model2 = _DummyModel()
            loader2 = _DummyLoader()
            loader2.state = {"cursor": -1}
            expert_opt2 = torch.optim.AdamW([model2.expert], lr=0.1)
            extra_opt2 = torch.optim.AdamW([model2.dense], lr=0.01)

            step, tokens, zero2_state = load_state(
                str(it_dir / "dp_rank_000.pt"),
                model2,
                expert_opt2,
                loader2,
                extra_optimizers={"muon": extra_opt2},
                print_fn=lambda *_args, **_kwargs: None,
            )

            self.assertEqual(step, 12)
            self.assertEqual(tokens, 3456)
            self.assertIsNone(zero2_state)
            self.assertEqual(loader2.state, {"cursor": 7})
            self.assertTrue(torch.equal(model2.dense, model.dense))
            self.assertTrue(torch.equal(model2.expert, model.expert))
            loaded_state = extra_opt2.state_dict()["state"]
            self.assertEqual(len(loaded_state), 1)
            self.assertTrue(
                all("exp_avg" in state and "exp_avg_sq" in state for state in loaded_state.values())
            )

    def test_resume_fails_loudly_when_extra_optimizer_state_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model = _DummyModel()
            loader = _DummyLoader()
            expert_opt = torch.optim.AdamW([model.expert], lr=0.1)
            _prime_adamw(expert_opt, [model.expert])

            rd_state, dp_state = build_states(
                3,
                model,
                expert_opt,
                128,
                loader,
                config_fingerprint=fingerprint(model.config),
            )
            it_dir = tmp_path / "iter_0000003"
            it_dir.mkdir()
            torch.save(rd_state, it_dir / "rd.pt")
            torch.save(dp_state, it_dir / "dp_rank_000.pt")

            model2 = _DummyModel()
            loader2 = _DummyLoader()
            expert_opt2 = torch.optim.AdamW([model2.expert], lr=0.1)
            extra_opt2 = torch.optim.AdamW([model2.dense], lr=0.01)

            with self.assertRaisesRegex(RuntimeError, "missing required extra optimizer state"):
                load_state(
                    str(it_dir / "dp_rank_000.pt"),
                    model2,
                    expert_opt2,
                    loader2,
                    extra_optimizers={"muon": extra_opt2},
                    print_fn=lambda *_args, **_kwargs: None,
                )
