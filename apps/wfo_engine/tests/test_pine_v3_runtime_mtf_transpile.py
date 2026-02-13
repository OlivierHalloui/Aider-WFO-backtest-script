"""Tests for transpiled runtime support of request.security (MTF)."""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

TEST_DIR = os.path.dirname(__file__)
WFO_ENGINE_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
sys.path.append(WFO_ENGINE_DIR)

HAS_VBT = importlib.util.find_spec("vectorbtpro") is not None
pytestmark = pytest.mark.skipif(not HAS_VBT, reason="vectorbtpro is required for runtime MTF tests")

if HAS_VBT:
    from pine_v3.runtime_adapter import (
        GeneratedPineRuntimeAdapter,
        build_request_security_diagnostics,
        run_transpiled_pine_backtest,
    )


def _mock_df(n=120, freq="1h"):
    idx = pd.date_range("2025-01-01", periods=n, freq=freq, tz="UTC")
    base = np.linspace(100.0, 130.0, n)
    wave = 2.5 * np.sin(np.linspace(0, 15, n))
    close = base + wave
    return pd.DataFrame(
        {
            "Open": close - 0.2,
            "High": close + 0.4,
            "Low": close - 0.4,
            "Close": close,
            "Volume": np.full(n, 1.0),
        },
        index=idx,
    )


def _mtf_spec():
    return {
        "schema_version": "strategy_spec.v1",
        "strategy": {"id": "pine_mtf_demo", "name": "Pine MTF Demo", "kind": "pine_imported"},
        "source": {
            "file_name": "demo.pine",
            "source_sha1": "x",
            "line_count": 10,
            "char_count": 100,
            "pine_version": 6,
        },
        "imports": [],
        "inputs": [],
        "warnings": [],
        "capabilities": {
            "uses_request_security": True,
            "uses_request_security_lower_tf": False,
            "uses_strategy_entry": True,
            "uses_strategy_exit": True,
            "uses_strategy_close": False,
            "uses_strategy_cancel": False,
        },
        "logic": {
            "assignments": [
                {"line": 1, "targets": ["TF"], "op": "=", "expr": "'4h'"},
                {
                    "line": 2,
                    "targets": ["sma_htf"],
                    "op": "=",
                    "expr": "request.security(syminfo.tickerid, TF, ta.sma(close, 3))",
                },
            ],
            "order_rules": [
                {
                    "line": 3,
                    "action": "entry",
                    "id": "L",
                    "direction": "strategy.long",
                    "condition_expr": "ta.crossover(close, sma_htf)",
                    "call": "strategy.entry(id='L', direction=strategy.long)",
                },
                {
                    "line": 4,
                    "action": "exit",
                    "id": "X",
                    "direction": "",
                    "condition_expr": "ta.crossunder(close, sma_htf)",
                    "call": "strategy.exit(id='X', from_entry='L')",
                },
            ],
        },
    }


def _short_order_spec():
    return {
        "schema_version": "strategy_spec.v1",
        "strategy": {"id": "pine_short_order_demo", "name": "Pine Short Order Demo", "kind": "pine_imported"},
        "source": {
            "file_name": "demo_short.pine",
            "source_sha1": "x",
            "line_count": 10,
            "char_count": 100,
            "pine_version": 6,
        },
        "imports": [],
        "inputs": [],
        "warnings": [],
        "capabilities": {
            "uses_request_security": False,
            "uses_request_security_lower_tf": False,
            "uses_strategy_entry": False,
            "uses_strategy_order": True,
            "uses_strategy_exit": False,
            "uses_strategy_close": True,
            "uses_strategy_cancel": False,
        },
        "logic": {
            "assignments": [
                {"line": 1, "targets": ["slow"], "op": "=", "expr": "ta.sma(close, 8)"},
            ],
            "order_rules": [
                {
                    "line": 2,
                    "action": "order",
                    "id": "S",
                    "direction": "strategy.short",
                    "condition_expr": "close < slow",
                    "call": "strategy.order('S', strategy.short)",
                },
                {
                    "line": 3,
                    "action": "close",
                    "id": "S",
                    "direction": "",
                    "condition_expr": "close > slow",
                    "call": "strategy.close('S')",
                },
            ],
        },
    }


def test_request_security_diagnostics_available():
    df = _mock_df()
    spec = _mtf_spec()
    diag = build_request_security_diagnostics(df=df, params={}, strategy_spec=spec, external_bindings={})
    assert diag["status"] == "available"
    assert int(diag.get("request_security_count", 0)) >= 1
    rows = diag.get("rows") or []
    assert rows
    assert str(rows[0].get("timeframe")) == "4h"
    assert int(rows[0].get("non_na_count", 0)) > 0


def test_run_transpiled_backtest_with_request_security():
    df = _mock_df()
    spec = _mtf_spec()
    score = run_transpiled_pine_backtest(
        df=df,
        params={"metric1_name": "total_return", "metric2_name": "sharpe_ratio", "weight_metric1": 1.0, "weight_metric2": 0.0},
        strategy_spec=spec,
        timeframe="1h",
        return_portfolio=False,
        external_bindings={},
    )
    assert np.isfinite(float(score))


def test_generated_runtime_supports_short_order_signals():
    df = _mock_df()
    adapter = GeneratedPineRuntimeAdapter(
        strategy_id="pine_short_order_demo",
        runtime_config={},
        strategy_spec=_short_order_spec(),
    )
    signals = adapter.generate_signals(df, params={})
    assert "short_entry_signal" in signals
    assert "short_exit_signal" in signals
    assert int(signals["short_entry_signal"].sum()) > 0

    score = adapter.run_backtest(
        df=df,
        params={"metric1_name": "total_return", "metric2_name": "sharpe_ratio", "weight_metric1": 1.0, "weight_metric2": 0.0},
        timeframe="1h",
        return_portfolio=False,
    )
    assert np.isfinite(float(score))
