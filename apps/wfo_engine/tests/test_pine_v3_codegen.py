import os
import sys

TEST_DIR = os.path.dirname(__file__)
WFO_ENGINE_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
sys.path.append(WFO_ENGINE_DIR)

from pine_v3.codegen import generate_strategy_module_from_spec
from pine_v3.spec import build_strategy_spec_v1_from_pine_text, validate_strategy_spec_v1


def test_generate_strategy_module_from_spec(tmp_path):
    pine_text = """
//@version=6
strategy("Demo")
import foo/bar/1 as BBT1
x = input.int(defval = 20, title = "Len")
"""
    spec = build_strategy_spec_v1_from_pine_text(
        pine_text=pine_text,
        source_name="demo.txt",
        strategy_id="pine_demo",
        precheck_report={},
        compatibility_report={"import_resolution": [{"alias": "BBT1", "resolved": True}]},
    )
    validation = validate_strategy_spec_v1(spec)
    assert validation["valid"] is True, validation

    report = generate_strategy_module_from_spec(
        strategy_spec=spec,
        output_dir=str(tmp_path),
        import_mapping={"BBT1": "apps/wfo_engine/indicators.py"},
        import_resolution=[{"alias": "BBT1", "resolved": True}],
    )
    assert report["status"] == "ok"
    output_path = report["output_path"]
    assert os.path.exists(output_path)

    content = open(output_path, "r", encoding="utf-8").read()
    assert "class GeneratedPineAdapter" in content
    assert "create_adapter" in content
