"""Tests for Pine V3 parity CI campaign (P2.3)."""

from __future__ import annotations

import json
import os
import sys

TEST_DIR = os.path.dirname(__file__)
WFO_ENGINE_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
sys.path.append(WFO_ENGINE_DIR)

from pine_v3.parity_ci import run_parity_ci_campaign, write_parity_ci_report


def test_campaign_passes_without_runtime_requirement(monkeypatch) -> None:
    monkeypatch.setattr("pine_v3.parity_ci._is_vectorbt_available", lambda: False)
    report = run_parity_ci_campaign(require_runtime=False)
    assert report["schema_version"] == "pine_parity_ci_report.v1"
    assert report["status"] == "passed"
    summary = report.get("summary", {})
    assert int(summary.get("total_scenarios", 0)) >= 3
    assert int(summary.get("failed", 0)) == 0


def test_campaign_fails_when_runtime_is_required_and_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("pine_v3.parity_ci._is_vectorbt_available", lambda: False)
    report = run_parity_ci_campaign(require_runtime=True)
    assert report["status"] == "failed"
    blockers = report.get("blockers", [])
    assert any(str(item.get("code")) == "runtime_campaign_missing" for item in blockers if isinstance(item, dict))


def test_write_report(tmp_path) -> None:
    report = run_parity_ci_campaign(require_runtime=False)
    output_path = tmp_path / "parity_ci_report.json"
    written = write_parity_ci_report(report, str(output_path))
    assert os.path.exists(written)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload.get("schema_version") == "pine_parity_ci_report.v1"

