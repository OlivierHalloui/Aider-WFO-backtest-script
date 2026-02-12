"""Tests for Pine V3 LLM migration helper (P2.1)."""

from __future__ import annotations

import json
import os
import sys

TEST_DIR = os.path.dirname(__file__)
WFO_ENGINE_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
sys.path.append(WFO_ENGINE_DIR)

from expert import LLMConfig
from pine_v3.llm_migration import run_llm_spec_migration
from pine_v3.spec import build_strategy_spec_v1_from_pine_text


class _FakeGatewayValid:
    def __init__(self, payload: dict):
        self.payload = payload

    def generate(self, system_prompt: str, user_prompt: str, config: LLMConfig) -> str:
        return json.dumps(self.payload, ensure_ascii=False)


class _FakeGatewayInvalidJson:
    def generate(self, system_prompt: str, user_prompt: str, config: LLMConfig) -> str:
        return "not a json payload"


def _llm_cfg() -> LLMConfig:
    return LLMConfig(
        provider="openai",
        model="gpt-5-mini",
        api_key="test-key",
        base_url="",
        temperature=0.2,
        timeout_s=20.0,
        max_tokens=1500,
        retries=0,
    )


def test_llm_migration_accepts_valid_candidate() -> None:
    pine_text = """
//@version=6
strategy("Demo")
len = input.int(defval = 14, title = "Len")
if close > ta.sma(close, len)
    strategy.entry(id="L", direction=strategy.long)
"""
    baseline = build_strategy_spec_v1_from_pine_text(
        pine_text=pine_text,
        source_name="demo.txt",
        strategy_id="pine_demo",
        parser_backend="regex",
    )
    # Slightly tweak warning text but keep contract valid.
    baseline["warnings"] = list(baseline.get("warnings") or []) + ["llm candidate"]

    report = run_llm_spec_migration(
        pine_text=pine_text,
        source_name="demo.txt",
        strategy_id="pine_demo",
        precheck_report={},
        compatibility_report={},
        parser_backend="auto",
        llm_config=_llm_cfg(),
        gateway=_FakeGatewayValid(baseline),
    )

    assert report["status"] == "ok"
    assert bool((report.get("accepted_validation") or {}).get("valid", False)) is True
    assert isinstance(report.get("accepted_spec"), dict)
    assert ((report.get("trace") or {}).get("llm_used")) is True


def test_llm_migration_fallbacks_on_invalid_json() -> None:
    pine_text = """
//@version=6
strategy("Demo")
"""
    report = run_llm_spec_migration(
        pine_text=pine_text,
        source_name="demo.txt",
        strategy_id="pine_demo",
        precheck_report={},
        compatibility_report={},
        parser_backend="auto",
        llm_config=_llm_cfg(),
        gateway=_FakeGatewayInvalidJson(),
    )

    assert report["status"] in {"fallback", "error"}
    assert bool((report.get("accepted_validation") or {}).get("valid", False)) is True
    assert isinstance(report.get("accepted_spec"), dict)
    assert "schema_version" in (report.get("accepted_spec") or {})
