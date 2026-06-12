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

from stagewise_optimizer import (
    propose_stage_settings,
    _consensus_params,
    PARAM_DEFAULTS,
    FULL_PARAM_REGISTRY,
)

if HAS_VBT:
    from stagewise_optimizer import run_stagewise_campaign


def generate_mock_data(n=200):
    rng = np.random.default_rng(42)
    index = pd.date_range("2025-01-01", periods=n, freq="5s")
    close = rng.uniform(100, 200, size=n)
    return pd.DataFrame(
        {"Close": close, "Open": close, "High": close + 1.0, "Low": close - 1.0},
        index=index,
    )


# ── propose_stage_settings (pure, no VBT) ─────────────────────────────────────

class TestProposeStageSettings:
    def test_returns_required_keys(self):
        result = propose_stage_settings(["timeperiod", "StDev"])
        assert {"method", "max_trials", "patience", "stability", "n_combos"}.issubset(result)

    def test_small_combo_selects_grid(self):
        result = propose_stage_settings(["use_t2_signal"])
        assert result["method"] == "grid"

    def test_large_combo_selects_bayesian(self):
        # timeperiod (21 values) × StDev (23 values) × fenetre_lowest (19) > 100k → bayesian
        result = propose_stage_settings(["timeperiod", "StDev", "fenetre_lowest"])
        assert result["method"] == "bayesian"
        assert result["max_trials"] > 0

    def test_param_ranges_override(self):
        # With narrow custom ranges, combo count drops → grid may be selected
        result = propose_stage_settings(
            ["timeperiod", "StDev"],
            param_ranges={"timeperiod": (10, 12, 2), "StDev": (1.0, 1.5, 0.5)},
        )
        assert result["n_combos"] <= 6
        assert result["method"] == "grid"

    def test_empty_params_returns_defaults(self):
        result = propose_stage_settings([])
        assert result["method"] == "grid"
        assert result["n_combos"] == 0


# ── _consensus_params (pure, no VBT) ──────────────────────────────────────────

class TestConsensusParams:
    def _make_wfo_results(self, rows):
        return {"best_params": rows}

    def test_numeric_median(self):
        rows = [{"timeperiod": 10}, {"timeperiod": 12}, {"timeperiod": 14}]
        result = _consensus_params(self._make_wfo_results(rows))
        assert result["timeperiod"] == 12

    def test_float_median(self):
        rows = [{"StDev": 1.0}, {"StDev": 2.0}, {"StDev": 3.0}]
        result = _consensus_params(self._make_wfo_results(rows))
        assert abs(result["StDev"] - 2.0) < 1e-9

    def test_boolean_majority_vote_true(self):
        rows = [
            {"use_t2_signal": True},
            {"use_t2_signal": True},
            {"use_t2_signal": False},
        ]
        result = _consensus_params(self._make_wfo_results(rows))
        assert result["use_t2_signal"] is True

    def test_boolean_majority_vote_false(self):
        rows = [
            {"exit_sar_enabled": False},
            {"exit_sar_enabled": False},
            {"exit_sar_enabled": True},
        ]
        result = _consensus_params(self._make_wfo_results(rows))
        assert result["exit_sar_enabled"] is False

    def test_empty_best_params_returns_empty(self):
        result = _consensus_params({"best_params": []})
        assert result == {}

    def test_mixed_params(self):
        rows = [
            {"timeperiod": 10, "use_t2_signal": True,  "StDev": 1.0},
            {"timeperiod": 14, "use_t2_signal": False, "StDev": 2.0},
            {"timeperiod": 12, "use_t2_signal": True,  "StDev": 1.5},
        ]
        result = _consensus_params(self._make_wfo_results(rows))
        assert result["timeperiod"] == 12
        assert result["use_t2_signal"] is True
        assert abs(result["StDev"] - 1.5) < 1e-9


# ── run_stagewise_campaign smoke test ─────────────────────────────────────────

MINI_STAGE_PLAN = [
    {
        "name": "Mini Stage 1 — timeperiod only",
        "optimize": ["timeperiod"],
        "fixed_overrides": {
            "use_t2_signal": True,
            "use_roc_filter": False,
            "use_divergence_bb": False,
            "exit_sar_enabled": False,
            "exit_macd_enabled": False,
            "exit_cross_sar_sma_enabled": False,
            "exit_retour_bb_enabled": False,
            "exit_regline_enabled": False,
            "exit_volat_down_enabled": False,
        },
        "method": "grid",
        "max_trials": 2,
        "patience": "Low",
        "stability": 3,
    },
]


def test_run_stagewise_campaign_smoke(tmp_path):
    """Stagewise campaign with 1 mini stage must return final_report with accumulated_best_params."""
    df = generate_mock_data(200)
    report = run_stagewise_campaign(
        df=df,
        timeframe="5s",
        direction="long_only",
        stage_plan=MINI_STAGE_PLAN,
        n_windows=1,
        train_size=0.6,
        output_dir=tmp_path / "stagewise_out",
        optimization_metric="sharpe_ratio",
        secondary_metric="total_return",
        metric_weights=(1.0, 0.0),
        parallel_backend="sequential",
        neighbor_count=3,
        param_ranges={"timeperiod": (10, 12, 2)},
    )
    assert isinstance(report, dict)
    assert "stages" in report
    assert len(report["stages"]) == 1
    assert "final_best_params" in report
    final_params = report["final_best_params"]
    assert isinstance(final_params, dict)
    assert "timeperiod" in final_params
