import importlib.util
import os
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

# Ensure WFO Engine modules are importable from the dedicated app folder.
TEST_DIR = os.path.dirname(__file__)
WFO_ENGINE_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
sys.path.append(WFO_ENGINE_DIR)

HAS_VBT = importlib.util.find_spec("vectorbtpro") is not None
pytestmark = pytest.mark.skipif(not HAS_VBT, reason="vectorbtpro is required for pine adapter tests")

if HAS_VBT:
    from strategy_adapters import resolve_strategy_adapter


def _mock_ohlc(n=240):
    rng = np.random.default_rng(123)
    index = pd.date_range("2025-01-01", periods=n, freq="5s")
    close = 100 + np.cumsum(rng.normal(0, 0.1, size=n))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
        },
        index=index,
    )


def test_resolve_pine_strategy_test_adapter():
    adapter = resolve_strategy_adapter(
        strategy_mode="pine_imported",
        strategy_id="pine_strategy_test",
        config={"pine_source_name": "strategy_test.txt"},
    )
    assert adapter.strategy_mode == "pine_imported"
    assert adapter.strategy_id == "pine_strategy_test"


def test_resolve_pine_strict_blocking_raises():
    with pytest.raises(NotImplementedError):
        resolve_strategy_adapter(
            strategy_mode="pine_imported",
            strategy_id="pine_strategy_test",
            config={
                "pine_source_name": "strategy_test.txt",
                "pine_compat_mode": "strict",
                "pine_compatibility_blocking": True,
            },
        )


def test_resolve_pine_adapter_keeps_runtime_config():
    cfg = {
        "pine_source_name": "strategy_test.txt",
        "pine_import_mapping": {"BBT1": "apps/wfo_engine/indicators.py"},
    }
    adapter = resolve_strategy_adapter(
        strategy_mode="pine_imported",
        strategy_id="pine_strategy_test",
        config=cfg,
    )
    assert isinstance(getattr(adapter, "runtime_config", None), dict)
    assert adapter.runtime_config.get("pine_import_mapping", {}).get("BBT1")


def test_resolve_generated_module_adapter(tmp_path):
    module_path = tmp_path / "generated_strategy.py"
    module_path.write_text(
        textwrap.dedent(
            """
            from pine_v3.runtime_adapter import PineStrategyTestAdapter

            def create_adapter(runtime_config=None):
                return PineStrategyTestAdapter(strategy_id="pine_strategy_test", runtime_config=runtime_config)
            """
        ),
        encoding="utf-8",
    )

    adapter = resolve_strategy_adapter(
        strategy_mode="pine_imported",
        strategy_id="pine_strategy_test",
        config={
            "pine_source_name": "strategy_test.txt",
            "pine_generated_module_path": str(module_path),
        },
    )
    assert adapter.strategy_mode == "pine_imported"
    assert adapter.strategy_id == "pine_strategy_test"


def test_pine_strategy_test_backtest_scalar_and_vectorized():
    df = _mock_ohlc(240)
    adapter = resolve_strategy_adapter(
        strategy_mode="pine_imported",
        strategy_id="pine_strategy_test",
        config={"pine_source_name": "strategy_test.txt"},
    )

    params_scalar = {
        "timeperiod": 20,
        "StDev": 2.0,
        "fenetre_lowest": 30,
        "seuil_lowest": 3.5,
        "user_exit_sma_length": 20,
        "metric1_name": "sharpe_ratio",
        "metric2_name": "total_return",
        "weight_metric1": 1.0,
        "weight_metric2": 0.0,
        "order_sizing_mode": "percent_equity",
        "order_fixed_cash": 10000.0,
        "fees_pct": 0.0,
    }

    score_scalar = adapter.run_backtest(df, params_scalar, timeframe="5s", return_portfolio=False)
    assert np.isscalar(score_scalar)

    params_vec = dict(params_scalar)
    params_vec.update(
        {
            "timeperiod": np.array([20, 21]),
            "StDev": np.array([2.0, 2.0]),
            "fenetre_lowest": np.array([30, 30]),
            "seuil_lowest": np.array([3.5, 3.5]),
            "user_exit_sma_length": np.array([20, 20]),
        }
    )
    score_vec = adapter.run_backtest(df, params_vec, timeframe="5s", return_portfolio=False)
    assert hasattr(score_vec, "shape")
    assert int(len(score_vec)) == 2


def test_pine_strategy_test_mtf_filter_reduces_or_equals_entries():
    df = _mock_ohlc(500)
    adapter = resolve_strategy_adapter(
        strategy_mode="pine_imported",
        strategy_id="pine_strategy_test",
        config={"pine_source_name": "strategy_test.txt"},
    )

    params_base = {
        "timeperiod": 20,
        "StDev": 2.0,
        "fenetre_lowest": 30,
        "seuil_lowest": 3.5,
        "user_exit_sma_length": 20,
        "mtf_filter_enabled": False,
        "metric1_name": "sharpe_ratio",
        "metric2_name": "total_return",
        "weight_metric1": 1.0,
        "weight_metric2": 0.0,
        "order_sizing_mode": "percent_equity",
        "order_fixed_cash": 10000.0,
        "fees_pct": 0.0,
    }

    sig_base = adapter.generate_signals(df, params_base)
    base_entries = int(sig_base["entry_signal"].sum().sum())

    params_mtf = dict(params_base)
    params_mtf.update(
        {
            "mtf_filter_enabled": True,
            "mtf_filter_timeframe": "30s",
            "mtf_filter_sma_length": 10,
            "mtf_filter_timing": "closing",
            "mtf_filter_gaps": "off",
            "mtf_filter_mode": "above",
        }
    )
    sig_mtf = adapter.generate_signals(df, params_mtf)
    mtf_entries = int(sig_mtf["entry_signal"].sum().sum())

    # Confirmation filter should not create additional entries.
    assert mtf_entries <= base_entries
