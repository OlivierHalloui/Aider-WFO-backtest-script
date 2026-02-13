"""Tests for Pine V3 external call contract checks."""

from __future__ import annotations

import importlib.util
import os
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

TEST_DIR = os.path.dirname(__file__)
WFO_ENGINE_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
sys.path.append(WFO_ENGINE_DIR)

HAS_VBT = importlib.util.find_spec("vectorbtpro") is not None
pytestmark = pytest.mark.skipif(not HAS_VBT, reason="vectorbtpro is required for runtime contract tests")

if HAS_VBT:
    from pine_v3.runtime_adapter import (
        GeneratedPineRuntimeAdapter,
        build_external_call_contract_report,
    )


def _base_spec() -> dict:
    return {
        "schema_version": "strategy_spec.v1",
        "strategy": {"id": "pine_ext_demo", "name": "Pine External Demo", "kind": "pine_imported"},
        "source": {"file_name": "demo.pine", "source_sha1": "x", "line_count": 1, "char_count": 1, "pine_version": 6},
        "imports": ["import x/y/1 as BBT1"],
        "imports_detail": [{"module_ref": "x/y/1", "alias": "BBT1", "parse_ok": True}],
        "import_resolution": [
            {
                "alias": "BBT1",
                "python_mapping_target": "dummy",
                "python_mapping_valid": True,
                "missing_functions_count": 0,
                "missing_functions": [],
            }
        ],
        "inputs": [],
        "warnings": [],
        "capabilities": {
            "uses_request_security": False,
            "uses_request_security_lower_tf": False,
            "uses_strategy_entry": True,
            "uses_strategy_exit": True,
            "uses_strategy_close": False,
            "uses_strategy_cancel": False,
        },
        "logic": {
            "assignments": [],
            "order_rules": [
                {
                    "line": 10,
                    "action": "entry",
                    "id": "L",
                    "direction": "strategy.long",
                    "condition_expr": "BBT1.always_true(close)",
                    "call": "strategy.entry(id='L', direction=strategy.long)",
                },
                {
                    "line": 11,
                    "action": "exit",
                    "id": "X",
                    "direction": "",
                    "condition_expr": "ta.crossunder(close, ta.sma(close, 5))",
                    "call": "strategy.exit(id='X', from_entry='L')",
                },
            ],
        },
    }


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


def test_external_contract_fails_when_alias_unresolved():
    report = build_external_call_contract_report(
        strategy_spec=_base_spec(),
        runtime_config={"pine_compat_mode": "strict"},
        external_bindings={},
    )
    assert report["passed"] is False
    blockers = report.get("blockers", [])
    assert any(str(x.get("code")) == "alias_unresolved" for x in blockers if isinstance(x, dict))


def test_external_contract_fails_when_function_missing(tmp_path):
    module_path = tmp_path / "lib_empty.py"
    module_path.write_text("x = 1\n", encoding="utf-8")

    cfg = {
        "pine_compat_mode": "strict",
        "pine_import_mapping": {"BBT1": str(module_path)},
        "pine_precheck_report": {
            "import_resolution": [
                {
                    "alias": "BBT1",
                    "python_mapping_target": str(module_path),
                    "python_mapping_valid": True,
                    "missing_functions_count": 0,
                    "missing_functions": [],
                }
            ]
        },
    }

    adapter = GeneratedPineRuntimeAdapter(
        strategy_id="pine_ext_demo",
        runtime_config=cfg,
        strategy_spec=_base_spec(),
    )
    with pytest.raises(NotImplementedError):
        adapter.generate_signals(_mock_df(), params={})


def test_external_contract_passes_with_callable_mapping(tmp_path):
    module_path = tmp_path / "lib_ok.py"
    module_path.write_text(
        textwrap.dedent(
            """
            import pandas as pd
            def always_true(close):
                if isinstance(close, pd.Series):
                    return pd.Series(True, index=close.index)
                return True
            """
        ),
        encoding="utf-8",
    )

    cfg = {
        "pine_compat_mode": "strict",
        "pine_import_mapping": {"BBT1": str(module_path)},
        "pine_precheck_report": {
            "import_resolution": [
                {
                    "alias": "BBT1",
                    "python_mapping_target": str(module_path),
                    "python_mapping_valid": True,
                    "missing_functions_count": 0,
                    "missing_functions": [],
                }
            ]
        },
    }
    adapter = GeneratedPineRuntimeAdapter(
        strategy_id="pine_ext_demo",
        runtime_config=cfg,
        strategy_spec=_base_spec(),
    )
    signals = adapter.generate_signals(_mock_df(), params={})
    contract = signals.get("external_call_contract", {})
    assert isinstance(contract, dict)
    assert contract.get("passed") is True
    assert int(contract.get("resolved_calls_count", 0)) >= 1

