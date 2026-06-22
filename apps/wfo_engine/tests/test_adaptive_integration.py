import os
import sys
import importlib.util

import numpy as np
import pandas as pd
import pytest

TEST_DIR = os.path.dirname(__file__)
WFO_ENGINE_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
sys.path.insert(0, WFO_ENGINE_DIR)

HAS_VBT = importlib.util.find_spec("vectorbtpro") is not None
pytestmark = pytest.mark.skipif(not HAS_VBT, reason="vectorbtpro required")

from adaptive_optimization import AdaptiveValueModel
from config import WFOSettings

if HAS_VBT:
    from adaptive_optimization import adaptive_continuous_optimization


def _make_settings(**kwargs):
    s = WFOSettings()
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


def _make_model(param_grid=None, **setting_kwargs):
    if param_grid is None:
        param_grid = {"timeperiod": [10, 12, 14], "StDev": [1.0, 1.5]}
    settings = _make_settings(**setting_kwargs)
    rng = np.random.default_rng(0)
    return AdaptiveValueModel(param_grid, settings, rng)


def generate_mock_data(n=300):
    rng = np.random.default_rng(42)
    index = pd.date_range("2025-01-01", periods=n, freq="5s")
    close = rng.uniform(100, 200, size=n)
    return pd.DataFrame(
        {"Close": close, "Open": close, "High": close + 1.0, "Low": close - 1.0},
        index=index,
    )


# ── AdaptiveValueModel unit tests (no VBT) ────────────────────────────────────

class TestAdaptiveValueModelDecay:
    def test_decay_reduces_sum_w(self):
        model = _make_model(adaptive_decay=0.5)
        model.update_single({"timeperiod": 10, "StDev": 1.0}, score=1.0)
        w_before = model.sum_w["timeperiod"].copy()
        model.apply_decay()
        assert np.all(model.sum_w["timeperiod"] <= w_before)
        assert np.any(model.sum_w["timeperiod"] < w_before)

    def test_decay_1_0_is_noop(self):
        model = _make_model(adaptive_decay=1.0)
        model.update_single({"timeperiod": 10, "StDev": 1.0}, score=1.0)
        w_before = model.sum_w["timeperiod"].copy()
        model.apply_decay()
        np.testing.assert_array_almost_equal(model.sum_w["timeperiod"], w_before)

    def test_total_trials_decays(self):
        model = _make_model(adaptive_decay=0.9)
        model.update_single({"timeperiod": 10, "StDev": 1.0}, score=1.0)
        t_before = model.total_trials
        model.apply_decay()
        assert model.total_trials < t_before


class TestAdaptiveValueModelUpdateSingle:
    def test_valid_update_increments_sum_w(self):
        model = _make_model()
        model.update_single({"timeperiod": 10, "StDev": 1.0}, score=0.5)
        assert model.sum_w["timeperiod"][0] > 0
        assert model.total_trials > 0

    def test_nan_score_ignored(self):
        model = _make_model()
        model.update_single({"timeperiod": 10, "StDev": 1.0}, score=float("nan"))
        assert model.total_trials == 0.0

    def test_inf_score_ignored(self):
        model = _make_model()
        model.update_single({"timeperiod": 10, "StDev": 1.0}, score=float("inf"))
        assert model.total_trials == 0.0

    def test_unknown_param_value_skipped(self):
        model = _make_model()
        model.update_single({"timeperiod": 999, "StDev": 1.0}, score=1.0)
        assert np.sum(model.sum_w["timeperiod"]) == 0.0


class TestAdaptiveValueModelBuildGrid:
    def test_returns_dict_during_warmup(self):
        model = _make_model(adaptive_warmup_trials=1000)
        grid, meta = model.build_active_grid()
        assert isinstance(grid, dict)
        assert meta["enabled"] is False

    def test_active_grid_values_within_base(self):
        model = _make_model(adaptive_warmup_trials=1)
        for _ in range(5):
            model.update_single({"timeperiod": 10, "StDev": 1.0}, score=1.0)
            model.update_single({"timeperiod": 14, "StDev": 1.5}, score=0.5)
        model.total_trials = 2.0
        grid, meta = model.build_active_grid()
        assert isinstance(grid, dict)
        for name, values in grid.items():
            base_values = set(model.base_param_grid[name])
            assert all(v in base_values for v in values), f"{name}: {values} not subset of {base_values}"


# ── adaptive_continuous_optimization smoke test ───────────────────────────────

def test_adaptive_continuous_optimization_smoke():
    """One adaptive cycle on tiny data must return the expected result structure."""
    df = generate_mock_data(300)
    param_grid = {"timeperiod": [10, 12, 14]}

    settings = _make_settings(
        adaptive_train_bars=80,
        adaptive_cycle_bars=40,
        adaptive_trials_per_cycle=5,
        adaptive_candidate_pool_size=15,
        adaptive_max_cycles=1,
        adaptive_warmup_trials=0,
        optimization_metric="total_return",
        secondary_metric="total_return",
        metric_weights=(1.0, 0.0),
    )

    results = adaptive_continuous_optimization(
        df=df,
        param_grid=param_grid,
        timeframe="5s",
        settings=settings,
    )

    assert isinstance(results, dict)
    assert "window_results" in results
    assert "cycle_results" in results
    cycle_results = results["cycle_results"]
    assert isinstance(cycle_results, list)
    assert len(cycle_results) >= 1
