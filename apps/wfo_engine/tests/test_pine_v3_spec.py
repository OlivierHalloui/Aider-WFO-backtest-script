"""Tests for Pine V3 strategy spec builder/validator."""

from __future__ import annotations

import os
import sys

TEST_DIR = os.path.dirname(__file__)
WFO_ENGINE_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
sys.path.append(WFO_ENGINE_DIR)

from pine_v3.spec import build_strategy_spec_v1_from_pine_text, validate_strategy_spec_v1


def test_spec_builder_keeps_import_details_and_resolution() -> None:
    pine_text = """
//@version=6
strategy("Demo")
import foo/bar/1 as LIB
x = input.int(defval = 10, title = "Len")
"""
    compat_report = {
        "import_resolution": [
            {
                "module_ref": "foo/bar/1",
                "alias": "LIB",
                "resolved": True,
            }
        ]
    }

    spec = build_strategy_spec_v1_from_pine_text(
        pine_text=pine_text,
        source_name="demo.txt",
        strategy_id="pine_demo",
        compatibility_report=compat_report,
    )

    assert spec["schema_version"] == "strategy_spec.v1"
    assert isinstance(spec.get("imports"), list)
    assert spec["imports"] == ["import foo/bar/1 as LIB"]
    assert isinstance(spec.get("imports_detail"), list)
    assert spec["imports_detail"][0]["alias"] == "LIB"
    assert isinstance(spec.get("import_resolution"), list)
    assert spec["import_resolution"][0]["resolved"] is True
    assert isinstance(spec.get("logic"), dict)
    assert isinstance(spec["logic"].get("assignments"), list)
    assert isinstance(spec["logic"].get("order_rules"), list)


def test_spec_validator_accepts_optional_import_fields() -> None:
    pine_text = """
//@version=6
strategy("Demo")
"""
    spec = build_strategy_spec_v1_from_pine_text(
        pine_text=pine_text,
        source_name="demo.txt",
        strategy_id="pine_demo",
        compatibility_report={},
    )
    # Optional fields can be present and still validate.
    spec["imports_detail"] = []
    spec["import_resolution"] = []

    validation = validate_strategy_spec_v1(spec)
    assert validation["valid"] is True, validation


def test_spec_extracts_order_rules_with_if_condition() -> None:
    pine_text = """
//@version=6
strategy("Demo")
signal = close > ta.sma(close, 20)
if signal
    strategy.entry(id="L", direction=strategy.long)
if ta.crossunder(close, ta.sma(close, 20))
    strategy.exit(id="X", from_entry="L")
"""
    spec = build_strategy_spec_v1_from_pine_text(
        pine_text=pine_text,
        source_name="demo.txt",
        strategy_id="pine_demo_orders",
        compatibility_report={},
    )
    validation = validate_strategy_spec_v1(spec)
    assert validation["valid"] is True, validation
    logic = spec.get("logic") or {}
    rules = logic.get("order_rules") or []
    assert len(rules) >= 2
    actions = {str(r.get("action")) for r in rules if isinstance(r, dict)}
    assert "entry" in actions
    assert "exit" in actions


def test_spec_extracts_multiline_request_security_calls() -> None:
    pine_text = """
//@version=6
strategy("Demo MTF")
TF_UTP = input.timeframe('30S', 'UTP Timeframe')
[sma1_UTP, sma2_UTP] = request.security(
    syminfo.tickerid,
    TF_UTP,
    [ta.sma(close, 7), ta.sma(close, 18)]
)
if ta.crossover(close, sma1_UTP)
    strategy.entry(id="L", direction=strategy.long)
"""
    spec = build_strategy_spec_v1_from_pine_text(
        pine_text=pine_text,
        source_name="demo_mtf.txt",
        strategy_id="pine_demo_mtf",
        compatibility_report={},
    )
    validation = validate_strategy_spec_v1(spec)
    assert validation["valid"] is True, validation
    logic = spec.get("logic") or {}
    calls = logic.get("request_security_calls") or []
    assert len(calls) >= 1
    first = calls[0]
    assert first.get("timeframe_expr") == "TF_UTP"
    assert "ta.sma(close, 7)" in str(first.get("expression_expr"))
