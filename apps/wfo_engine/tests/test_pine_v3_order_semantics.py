"""Tests for Pine V3 order semantics contract (strategy.order/entry args)."""

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
pytestmark = pytest.mark.skipif(not HAS_VBT, reason="vectorbtpro is required for runtime order semantics tests")

if HAS_VBT:
    from pine_v3.runtime_adapter import (
        GeneratedPineRuntimeAdapter,
        build_order_semantics_report,
    )


def _mock_df(n: int = 180) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    base = np.linspace(100.0, 120.0, n)
    wave = np.sin(np.linspace(0, 8, n))
    close = base + wave
    return pd.DataFrame(
        {
            "Open": close - 0.2,
            "High": close + 0.3,
            "Low": close - 0.3,
            "Close": close,
            "Volume": np.full(n, 1.0),
        },
        index=idx,
    )


def _spec_with_limit_order() -> dict:
    return {
        "schema_version": "strategy_spec.v1",
        "strategy": {"id": "pine_limit_demo", "name": "Pine Limit Demo", "kind": "pine_imported"},
        "source": {"file_name": "demo.pine", "source_sha1": "x", "line_count": 1, "char_count": 1, "pine_version": 6},
        "imports": [],
        "inputs": [],
        "warnings": [],
        "capabilities": {
            "uses_request_security": False,
            "uses_request_security_lower_tf": False,
            "uses_strategy_entry": False,
            "uses_strategy_order": True,
            "uses_strategy_exit": True,
            "uses_strategy_close": False,
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
                    "id": "L",
                    "direction": "strategy.long",
                    "condition_expr": "close > slow",
                    "call": "strategy.order(id='L', direction=strategy.long, limit=close)",
                    "args_positional": [],
                    "args_named": {"id": "'L'", "direction": "strategy.long", "limit": "close"},
                },
                {
                    "line": 3,
                    "action": "exit",
                    "id": "X",
                    "direction": "",
                    "condition_expr": "close < slow",
                    "call": "strategy.exit(id='X', from_entry='L')",
                    "args_positional": [],
                    "args_named": {"id": "'X'", "from_entry": "'L'"},
                },
            ],
        },
    }


def _spec_with_qty_percent_order() -> dict:
    return {
        "schema_version": "strategy_spec.v1",
        "strategy": {"id": "pine_qtyp_demo", "name": "Pine QtyP Demo", "kind": "pine_imported"},
        "source": {"file_name": "demo_qtyp.pine", "source_sha1": "x", "line_count": 1, "char_count": 1, "pine_version": 6},
        "imports": [],
        "inputs": [],
        "warnings": [],
        "capabilities": {
            "uses_request_security": False,
            "uses_request_security_lower_tf": False,
            "uses_strategy_entry": False,
            "uses_strategy_order": True,
            "uses_strategy_exit": True,
            "uses_strategy_close": False,
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
                    "id": "L",
                    "direction": "strategy.long",
                    "condition_expr": "close > slow",
                    "call": "strategy.order(id='L', direction=strategy.long, qty_percent=25)",
                    "args_positional": [],
                    "args_named": {"id": "'L'", "direction": "strategy.long", "qty_percent": "25"},
                },
                {
                    "line": 3,
                    "action": "close",
                    "id": "L",
                    "direction": "",
                    "condition_expr": "close < slow",
                    "call": "strategy.close('L')",
                    "args_positional": ["'L'"],
                    "args_named": {},
                },
            ],
        },
    }


def _spec_with_mixed_qty_modes() -> dict:
    return {
        "schema_version": "strategy_spec.v1",
        "strategy": {"id": "pine_qty_mixed_demo", "name": "Pine Qty Mixed Demo", "kind": "pine_imported"},
        "source": {"file_name": "demo_qty_mixed.pine", "source_sha1": "x", "line_count": 1, "char_count": 1, "pine_version": 6},
        "imports": [],
        "inputs": [],
        "warnings": [],
        "capabilities": {
            "uses_request_security": False,
            "uses_request_security_lower_tf": False,
            "uses_strategy_entry": False,
            "uses_strategy_order": True,
            "uses_strategy_exit": True,
            "uses_strategy_close": False,
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
                    "id": "L1",
                    "direction": "strategy.long",
                    "condition_expr": "close > slow",
                    "call": "strategy.order(id='L1', direction=strategy.long, qty_percent=25)",
                    "args_positional": [],
                    "args_named": {"id": "'L1'", "direction": "strategy.long", "qty_percent": "25"},
                },
                {
                    "line": 3,
                    "action": "order",
                    "id": "L2",
                    "direction": "strategy.long",
                    "condition_expr": "close > slow",
                    "call": "strategy.order(id='L2', direction=strategy.long, qty=2)",
                    "args_positional": [],
                    "args_named": {"id": "'L2'", "direction": "strategy.long", "qty": "2"},
                },
            ],
        },
    }


def test_order_semantics_report_detects_limit_controls():
    report = build_order_semantics_report(
        strategy_spec=_spec_with_limit_order(),
        runtime_config={"pine_compat_mode": "strict"},
    )
    assert report["passed"] is False
    assert int(report.get("rules_with_price_controls", 0)) >= 1
    blockers = report.get("blockers") or []
    assert any(str(x.get("code")) == "unsupported_order_price_controls" for x in blockers if isinstance(x, dict))


def test_generated_adapter_blocks_limit_controls_in_strict_mode():
    adapter = GeneratedPineRuntimeAdapter(
        strategy_id="pine_limit_demo",
        runtime_config={"pine_compat_mode": "strict"},
        strategy_spec=_spec_with_limit_order(),
    )
    with pytest.raises(NotImplementedError):
        adapter.generate_signals(_mock_df(), params={})


def test_generated_adapter_allows_limit_controls_in_assist_mode():
    adapter = GeneratedPineRuntimeAdapter(
        strategy_id="pine_limit_demo",
        runtime_config={"pine_compat_mode": "assist"},
        strategy_spec=_spec_with_limit_order(),
    )
    signals = adapter.generate_signals(_mock_df(), params={})
    report = signals.get("order_semantics", {})
    assert isinstance(report, dict)
    assert report.get("status") == "failed"


def test_order_semantics_supports_qty_percent_mode():
    report = build_order_semantics_report(
        strategy_spec=_spec_with_qty_percent_order(),
        runtime_config={"pine_compat_mode": "strict"},
    )
    assert report["passed"] is True
    assert report.get("entry_qty_modes") == ["percent"]

    adapter = GeneratedPineRuntimeAdapter(
        strategy_id="pine_qtyp_demo",
        runtime_config={"pine_compat_mode": "strict"},
        strategy_spec=_spec_with_qty_percent_order(),
    )
    signals = adapter.generate_signals(_mock_df(), params={})
    assert str(signals.get("entry_size_mode_hint")) == "percent"
    long_sizes = signals.get("entry_size_long_override")
    assert isinstance(long_sizes, pd.Series)
    # qty_percent=25 should map to size=0.25 on matching entry bars.
    assert float(long_sizes.max(skipna=True)) <= 0.25 + 1e-12
    assert float(long_sizes.max(skipna=True)) >= 0.25 - 1e-12


def test_order_semantics_blocks_mixed_qty_modes_in_strict():
    report = build_order_semantics_report(
        strategy_spec=_spec_with_mixed_qty_modes(),
        runtime_config={"pine_compat_mode": "strict"},
    )
    assert report["passed"] is False
    blockers = report.get("blockers") or []
    assert any(str(x.get("code")) == "mixed_qty_modes" for x in blockers if isinstance(x, dict))

    adapter = GeneratedPineRuntimeAdapter(
        strategy_id="pine_qty_mixed_demo",
        runtime_config={"pine_compat_mode": "strict"},
        strategy_spec=_spec_with_mixed_qty_modes(),
    )
    with pytest.raises(NotImplementedError):
        adapter.generate_signals(_mock_df(), params={})
