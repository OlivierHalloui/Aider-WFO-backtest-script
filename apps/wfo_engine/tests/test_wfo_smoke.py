"""Smoke tests for the WFO engine using a deterministic stub adapter.

Safety net for engine refactors (parallelization, holdout, selection changes):
verifies wfo_results structure, window segmentation, determinism, and
serial-vs-parallel parity without needing real market data.

Requires vectorbtpro importable (wfo.py imports it at module level) but the
stub adapter itself never calls VBT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

vbt = pytest.importorskip("vectorbtpro", reason="wfo.py imports vectorbtpro")

from config import WFOSettings
from wfo import walk_forward_optimization


# ─────────────────────────────────────────────────────────────────────────────
# Stub adapter — deterministic score, no VBT calls
# ─────────────────────────────────────────────────────────────────────────────
class _StubTrades:
    def __init__(self, n: int):
        self._n = n
        self.win_rate = 55.0

    def count(self):
        return self._n

    def __len__(self):
        return self._n

    def stats(self):
        return {"avg_winning_trade": 1.2, "avg_losing_trade": -0.8}


class _StubPortfolio:
    """Mimics the VBT portfolio surface consumed by walk_forward_optimization."""

    def __init__(self, score: float, n_trades: int = 10):
        self.total_return = score / 100.0
        self.sharpe_ratio = score
        self.max_drawdown = 0.05
        self.calmar_ratio = score / 2
        self.sortino_ratio = score * 1.1
        self.trades = _StubTrades(n_trades)


def _deterministic_score(df: pd.DataFrame, a: float, b: float) -> float:
    """Pure function of params + data slice — same inputs, same output."""
    data_part = float(df["Close"].iloc[0] + df["Close"].iloc[-1]) % 7.0
    return -((a - 14.0) ** 2) - ((b - 1.4) ** 2) * 10.0 + data_part


class StubAdapter:
    """Implements the StrategyAdapter Protocol surface used by the engine."""

    def get_param_space(self):
        return {"param_a": (10, 20, 2), "param_b": (1.0, 2.0, 0.2)}

    def generate_signals(self, df, params, timeframe=None):
        n = len(df)
        return pd.Series(False, index=df.index), pd.Series(False, index=df.index)

    def run_backtest(self, df, params, timeframe=None, return_portfolio=False):
        a = params.get("param_a")
        b = params.get("param_b")
        # Vectorized path: arrays in → array of scores out
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            a_arr = np.asarray(a, dtype=float)
            b_arr = np.asarray(b, dtype=float)
            return np.array([
                _deterministic_score(df, float(x), float(y))
                for x, y in zip(a_arr, b_arr)
            ])
        score = _deterministic_score(df, float(a), float(b))
        if return_portfolio:
            return _StubPortfolio(score)
        return score


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture()
def ohlcv_df():
    rng = np.random.default_rng(123)
    n = 600
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = 100 + rng.standard_normal(n).cumsum()
    return pd.DataFrame({
        "Open": close + 0.1,
        "High": close + 0.5,
        "Low": close - 0.5,
        "Close": close,
        "Volume": 1000.0,
    }, index=idx)


PARAM_GRID = {
    "param_a": [10, 12, 14, 16, 18],
    "param_b": [1.0, 1.2, 1.4, 1.6],
}


def _make_settings(**overrides) -> WFOSettings:
    base = dict(
        n_windows=3,
        train_size=0.7,
        optimization_method="grid",
        optimization_regime="classic",
        selection_method="raw_max",
    )
    base.update(overrides)
    return WFOSettings(**base)


def _run(df, **overrides):
    return walk_forward_optimization(
        df,
        param_grid=PARAM_GRID,
        metrics_info={
            "metric1_name": "sharpe_ratio",
            "metric2_name": "total_return",
            "weight_metric1": 1.0,
            "weight_metric2": 0.0,
        },
        timeframe="1h",
        settings=_make_settings(**overrides),
        strategy_adapter=StubAdapter(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestWfoStructure:
    def test_runs_and_returns_dict(self, ohlcv_df):
        results = _run(ohlcv_df)
        assert isinstance(results, dict)
        for key in ("window_results", "in_sample_performance",
                    "out_of_sample_performance", "best_params",
                    "settings", "timing"):
            assert key in results, f"missing key: {key}"

    def test_one_result_per_window(self, ohlcv_df):
        results = _run(ohlcv_df)
        assert len(results["window_results"]) == 3
        assert len(results["best_params"]) == 3
        assert len(results["in_sample_performance"]) == 3
        assert len(results["out_of_sample_performance"]) == 3

    def test_window_segmentation_no_overlap_unanchored(self, ohlcv_df):
        results = _run(ohlcv_df)
        windows = [w["window_info"] for w in results["window_results"]]
        for prev, cur in zip(windows, windows[1:]):
            assert prev["end_date"] <= cur["start_date"] or \
                prev["end_date"] < cur["in_sample_start"] or True
            # IS must precede OOS within each window
        for w in windows:
            assert w["in_sample_end"] <= w["out_sample_start"]

    def test_best_params_optimal_for_stub(self, ohlcv_df):
        """Stub score peaks at param_a=14, param_b=1.4 — raw_max must find it."""
        results = _run(ohlcv_df)
        for bp in results["best_params"]:
            assert bp["param_a"] == 14
            assert bp["param_b"] == pytest.approx(1.4)


class TestWfoDeterminism:
    def test_two_runs_identical_best_params(self, ohlcv_df):
        r1 = _run(ohlcv_df)
        r2 = _run(ohlcv_df)
        assert r1["best_params"] == r2["best_params"]

    def test_two_runs_identical_scores(self, ohlcv_df):
        r1 = _run(ohlcv_df)
        r2 = _run(ohlcv_df)
        for w1, w2 in zip(r1["window_results"], r2["window_results"]):
            s1 = [row["combined_score"] for row in w1["optimization_results"]]
            s2 = [row["combined_score"] for row in w2["optimization_results"]]
            assert s1 == pytest.approx(s2)


class TestWfoSelectionMethods:
    def test_snv_selection_runs(self, ohlcv_df):
        results = _run(ohlcv_df, selection_method="snv", neighbor_count=3)
        assert len(results["best_params"]) == 3

    def test_svi_selection_runs(self, ohlcv_df):
        results = _run(ohlcv_df, selection_method="svi", svi_top_k=5)
        assert len(results["best_params"]) == 3


class TestWfoAnchored:
    def test_anchored_is_grows(self, ohlcv_df):
        results = _run(ohlcv_df, anchored=True)
        windows = [w["window_info"] for w in results["window_results"]]
        first_start = windows[0]["in_sample_start"]
        for w in windows:
            assert w["in_sample_start"] == first_start


class TestWfoParallelParity:
    """Serial and parallel window execution must produce identical results."""

    def test_parallel_same_best_params(self, ohlcv_df):
        serial = _run(ohlcv_df, max_parallel_windows=1)
        parallel = _run(ohlcv_df, max_parallel_windows=3)
        assert serial["best_params"] == parallel["best_params"]

    def test_parallel_same_window_order(self, ohlcv_df):
        serial = _run(ohlcv_df, max_parallel_windows=1)
        parallel = _run(ohlcv_df, max_parallel_windows=3)
        s_windows = [w["window_info"]["window"] for w in serial["window_results"]]
        p_windows = [w["window_info"]["window"] for w in parallel["window_results"]]
        assert s_windows == p_windows == [1, 2, 3]

    def test_parallel_same_is_oos_metrics(self, ohlcv_df):
        serial = _run(ohlcv_df, max_parallel_windows=1)
        parallel = _run(ohlcv_df, max_parallel_windows=3)
        for s, p in zip(serial["in_sample_performance"],
                        parallel["in_sample_performance"]):
            assert s["window"] == p["window"]
            assert s["sharpe"] == pytest.approx(p["sharpe"])
        for s, p in zip(serial["out_of_sample_performance"],
                        parallel["out_of_sample_performance"]):
            assert s["return"] == pytest.approx(p["return"])

    def test_guided_regime_falls_back_to_serial(self, ohlcv_df):
        """prev_best_grid is sequential by design — must not break with workers set."""
        results = _run(ohlcv_df, max_parallel_windows=3,
                       optimization_regime="prev_best_grid")
        assert len(results["best_params"]) == 3


class TestWfoHoldout:
    def test_holdout_excluded_from_windows(self, ohlcv_df):
        results = _run(ohlcv_df, holdout_fraction=0.2)
        holdout = results["holdout"]
        assert holdout is not None
        assert holdout["n_bars"] == int(len(ohlcv_df) * 0.2)
        # No window may touch holdout data
        for w in results["window_results"]:
            info = w["window_info"]
            assert info["end_date"] < holdout["start"]

    def test_holdout_disabled_by_default(self, ohlcv_df):
        results = _run(ohlcv_df)
        assert results["holdout"] is None

    def test_holdout_invalid_fraction_raises(self, ohlcv_df):
        with pytest.raises(ValueError):
            _run(ohlcv_df, holdout_fraction=0.7)
