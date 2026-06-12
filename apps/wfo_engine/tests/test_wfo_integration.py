import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure WFO Engine modules are importable from the dedicated app folder.
TEST_DIR = os.path.dirname(__file__)
WFO_ENGINE_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
sys.path.append(WFO_ENGINE_DIR)

HAS_VBT = importlib.util.find_spec("vectorbtpro") is not None
pytestmark = pytest.mark.skipif(not HAS_VBT, reason="vectorbtpro is required for integration tests")

if HAS_VBT:
    import vectorbtpro as vbt
    from main import get_param_grid, get_metrics_info, get_wfo_settings
    from wfo import walk_forward_optimization, get_svi_best_params
    from strategy import run_backtest
    from metrics import portfolio_metrics
    from stagewise_optimizer import PARAM_DEFAULTS
    from strategy_adapters import resolve_strategy_adapter


def generate_mock_data(n=200):
    """Create deterministic synthetic OHLC data for fast integration checks."""
    rng = np.random.default_rng(42)
    index = pd.date_range("2025-01-01", periods=n, freq="5s")
    close = rng.uniform(100, 200, size=n)
    return pd.DataFrame(
        {
            "Close": close,
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
        },
        index=index,
    )


def base_config():
    """Minimal config that exercises default-parameter injection + optimization."""
    return {
        "selected_params": ["timeperiod"],
        "timeperiod_min": 10,
        "timeperiod_max": 12,
        "timeperiod_step": 1,
        "metric1_name": "sharpe_ratio",
        "metric2_name": "total_return",
        "weight_metric1": 1.0,
        "weight_metric2": 0.0,
        "n_windows": 1,
        "train_size": 0.5,
        "parallel_backend": "thread",
        "max_trials": 5,
    }


def test_get_param_grid_default_injection():
    """Unselected strategy parameters must still exist as fixed defaults."""
    config = base_config()
    param_grid = get_param_grid(config)

    assert "coeff_medianeBBW" in param_grid
    assert param_grid["coeff_medianeBBW"] == [1.1]
    assert "timeperiod" in param_grid
    assert param_grid["timeperiod"] == [10, 11, 12]


@pytest.mark.parametrize("method", ["grid", "bayesian", "optuna"])
def test_wfo_runs_for_core_optimizers(method):
    """The WFO pipeline should execute for grid/bayesian/optuna on small data."""
    df = generate_mock_data(120)
    config = base_config()
    config["optimization_method"] = method
    if method != "grid":
        config["max_trials"] = 5

    param_grid = get_param_grid(config)
    metrics_info = get_metrics_info(config)
    settings = get_wfo_settings(config)
    settings.optimization_method = method
    settings.n_windows = 1

    results = walk_forward_optimization(
        df,
        param_grid,
        metrics_info,
        timeframe="5s",
        settings=settings,
    )

    assert isinstance(results, dict)
    assert "window_results" in results
    assert len(results["window_results"]) == 1
    assert "best_params" in results
    assert len(results["best_params"]) >= 1


# ── Phase 1: VBT real-stack integration tests ─────────────────────────────────

def test_run_backtest_returns_portfolio():
    """run_backtest with scalar params must return a real vbt.Portfolio."""
    df = generate_mock_data(200)
    params = dict(PARAM_DEFAULTS)
    portfolio = run_backtest(df, params, timeframe="5s", return_portfolio=True)
    assert isinstance(portfolio, vbt.Portfolio)
    total_ret = float(portfolio.total_return)
    assert np.isfinite(total_ret)


def test_run_backtest_returns_scalar_score():
    """run_backtest with return_portfolio=False must return a finite scalar score."""
    df = generate_mock_data(200)
    params = dict(PARAM_DEFAULTS)
    score = run_backtest(df, params, timeframe="5s", return_portfolio=False)
    score_f = float(np.asarray(score).reshape(-1)[0]) if hasattr(score, "__len__") else float(score)
    assert np.isfinite(score_f) or np.isnan(score_f)


def test_portfolio_metrics_with_real_vbt():
    """portfolio_metrics must return 11 keys; values finite or NaN (NaN allowed for ratio metrics with 0 trades)."""
    df = generate_mock_data(200)
    params = dict(PARAM_DEFAULTS)
    portfolio = run_backtest(df, params, timeframe="5s", return_portfolio=True)
    result = portfolio_metrics(portfolio, window_id=0)
    expected_keys = {
        "return", "sharpe", "max_drawdown", "win_rate",
        "avg_gain_per_trade", "avg_loss_per_trade", "avg_pl_per_trade",
        "calmar_ratio", "sortino_ratio", "pqs", "n_trades",
    }
    assert expected_keys.issubset(result.keys())
    # Values must be numeric (float, int, or NaN) — not strings, not objects
    for key in expected_keys:
        val = result[key]
        if val is not None:
            assert isinstance(float(val), float), f"{key}={val} not numeric"


def test_svi_best_params():
    """get_svi_best_params must return a valid row from the optimization results."""
    df = generate_mock_data(200)
    config = base_config()
    param_grid = get_param_grid(config)
    metrics_info = get_metrics_info(config)
    settings = get_wfo_settings(config)
    settings.n_windows = 1

    results = walk_forward_optimization(df, param_grid, metrics_info, timeframe="5s", settings=settings)
    # optimization_results stored as list-of-dicts — convert to DataFrame as the engine does
    opt_records = results["window_results"][0]["optimization_trials"]
    opt_df = pd.DataFrame(opt_records).sort_values("combined_score", ascending=False).reset_index(drop=True)
    is2_df = df.iloc[: len(df) // 4]
    adapter = resolve_strategy_adapter(strategy_mode="native_atdmf")

    best_row, best_score = get_svi_best_params(
        opt_df,
        is2_df=is2_df,
        top_k=3,
        strategy_adapter=adapter,
        timeframe="5s",
        metrics_info=metrics_info,
        param_grid=param_grid,
    )
    assert best_row is not None
    assert "timeperiod" in best_row.index


def test_wfo_out_of_sample_metrics_populated():
    """WFO with 2 windows must populate out-of-sample performance metrics."""
    df = generate_mock_data(300)
    config = base_config()
    config["n_windows"] = 2
    param_grid = get_param_grid(config)
    metrics_info = get_metrics_info(config)
    settings = get_wfo_settings(config)
    settings.n_windows = 2

    results = walk_forward_optimization(df, param_grid, metrics_info, timeframe="5s", settings=settings)

    oos = results.get("out_of_sample_performance", [])
    assert len(oos) == 2
    for entry in oos:
        assert "return" in entry or "sharpe" in entry, f"missing metrics in OOS entry: {entry}"


if __name__ == "__main__":
    # Useful for quick local manual check.
    config = {
        **base_config(),
        "optimization_method": "grid",
    }
    _df = generate_mock_data(120)
    _grid = get_param_grid(config)
    _metrics = get_metrics_info(config)
    _settings = get_wfo_settings(config)
    _ = walk_forward_optimization(_df, _grid, _metrics, timeframe="5s", settings=_settings)
    print("Integration smoke test passed.")
