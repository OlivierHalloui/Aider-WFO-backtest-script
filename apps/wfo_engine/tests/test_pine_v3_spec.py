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
