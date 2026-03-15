import pytest


class _DummyRouter:
    def __init__(self, bias):
        self.bias = bias


def test_collect_param_group_stats_splits_dense_router_expert():
    torch = pytest.importorskip("torch")

    from nmoe.metrics import collect_param_group_stats

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = torch.nn.Linear(2, 2, bias=False)
            self.router = torch.nn.Linear(2, 2, bias=False)
            self.expert = torch.nn.Linear(2, 2, bias=False)

        def param_sets(self):
            return [self.expert.weight], [self.dense.weight, self.router.weight]

    model = DummyModel()
    model.dense.weight.data.fill_(3.0)
    model.router.weight.data.fill_(4.0)
    model.expert.weight.data.fill_(12.0)
    model.dense.weight.grad = torch.full_like(model.dense.weight, 2.0)
    model.router.weight.grad = torch.full_like(model.router.weight, 5.0)
    model.expert.weight.grad = torch.full_like(model.expert.weight, 1.5)

    stats = collect_param_group_stats(model)

    assert stats["dense"]["param_count"] == 4
    assert stats["router"]["param_count"] == 4
    assert stats["expert"]["param_count"] == 4
    assert stats["dense"]["grad_norm"] == pytest.approx(4.0)
    assert stats["router"]["grad_norm"] == pytest.approx(10.0)
    assert stats["expert"]["grad_norm"] == pytest.approx(3.0)
    assert stats["dense"]["param_norm"] == pytest.approx(6.0)
    assert stats["router"]["param_norm"] == pytest.approx(8.0)
    assert stats["expert"]["param_norm"] == pytest.approx(24.0)
    assert stats["dense"]["grad_to_param"] == pytest.approx(4.0 / 6.0)
    assert stats["router"]["grad_to_param"] == pytest.approx(10.0 / 8.0)
    assert stats["expert"]["grad_to_param"] == pytest.approx(3.0 / 24.0)


def test_collect_router_stats_reports_percentiles():
    torch = pytest.importorskip("torch")

    from nmoe.metrics import collect_router_stats

    class DummyFFN(torch.nn.Module):
        def __init__(self, loads, importance, bias):
            super().__init__()
            self.last_aux_loss = torch.tensor(0.0)
            self.last_loads = loads
            self.last_importance = importance
            self.router = _DummyRouter(bias)

    class DummyBlock(torch.nn.Module):
        def __init__(self, ffn):
            super().__init__()
            self.ffn = ffn

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([
                DummyBlock(
                    DummyFFN(
                        torch.tensor([0.0, 0.25, 0.25, 0.5]),
                        torch.tensor([0.1, 0.2, 0.3, 0.4]),
                        torch.tensor([-1.0, 0.0, 0.5, 1.0]),
                    )
                )
            ])

    model = DummyModel()
    per, agg = collect_router_stats(model)

    assert len(per) == 1
    layer = per[0]
    loads = torch.tensor([0.0, 0.25, 0.25, 0.5], dtype=torch.float32)
    importance = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)

    assert layer["min_load"] == pytest.approx(0.0)
    assert layer["p10_load"] == pytest.approx(float(torch.quantile(loads, 0.10).item() * 100.0))
    assert layer["p50_load"] == pytest.approx(float(torch.quantile(loads, 0.50).item() * 100.0))
    assert layer["p90_load"] == pytest.approx(float(torch.quantile(loads, 0.90).item() * 100.0))
    assert layer["max_load"] == pytest.approx(50.0)
    assert layer["experts_active"] == 3
    assert layer["importance_cv"] == pytest.approx(float((importance.std(unbiased=False) / importance.mean() * 100.0).item()))
    assert layer["p90_importance"] == pytest.approx(float(torch.quantile(importance, 0.90).item() * 100.0))
    assert layer["max_importance"] == pytest.approx(40.0)
    assert agg["dead_experts_count"] == pytest.approx(1.0)
    assert agg["experts_active_mean"] == pytest.approx(3.0)
    assert agg["mean_max_load"] == pytest.approx(50.0)
    assert agg["mean_max_importance"] == pytest.approx(40.0)
    assert agg["mean_p50_importance"] == pytest.approx(float(torch.quantile(importance, 0.50).item() * 100.0))


def test_train_signal_overrides_round_trip():
    from nmoe.metrics import publish_train_signal_overrides, pop_train_signal_overrides

    pop_train_signal_overrides()
    publish_train_signal_overrides({
        "expert": {
            "update_norm": 1.25,
            "update_to_pre_param": 0.05,
            "weight_decay_update_norm": None,
        }
    })

    stats = pop_train_signal_overrides()
    assert stats["expert"]["update_norm"] == pytest.approx(1.25)
    assert stats["expert"]["update_to_pre_param"] == pytest.approx(0.05)
    assert pop_train_signal_overrides() == {}


def test_summarize_param_snapshot_splits_weight_decay_and_optimizer_update():
    torch = pytest.importorskip("torch")

    from nmoe.opt import _snapshot_params, _summarize_param_snapshot

    p = torch.nn.Parameter(torch.tensor([10.0, 20.0], dtype=torch.float32))
    snapshot = _snapshot_params([p])
    p.data = torch.tensor([8.8, 17.6], dtype=torch.float32)

    stats = _summarize_param_snapshot(snapshot, lr=0.1, weight_decay=0.2)

    pre_param_norm = (10.0 ** 2 + 20.0 ** 2) ** 0.5
    wd_norm = (0.2 ** 2 + 0.4 ** 2) ** 0.5
    opt_norm = (1.0 ** 2 + 2.0 ** 2) ** 0.5
    total_norm = (1.2 ** 2 + 2.4 ** 2) ** 0.5

    assert stats["pre_param_norm"] == pytest.approx(pre_param_norm)
    assert stats["weight_decay_update_norm"] == pytest.approx(wd_norm)
    assert stats["optimizer_update_norm"] == pytest.approx(opt_norm)
    assert stats["update_norm"] == pytest.approx(total_norm)
    assert stats["update_to_pre_param"] == pytest.approx(total_norm / pre_param_norm)
    assert stats["optimizer_update_to_pre_param"] == pytest.approx(opt_norm / pre_param_norm)
