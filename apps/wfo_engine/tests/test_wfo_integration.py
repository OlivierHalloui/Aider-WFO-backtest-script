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
    from main import get_param_grid, get_metrics_info, get_wfo_settings
    from wfo import walk_forward_optimization


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
