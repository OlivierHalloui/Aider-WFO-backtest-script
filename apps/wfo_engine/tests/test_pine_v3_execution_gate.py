"""Tests for Pine V3 execution gate helper."""

from __future__ import annotations

import os
import sys

TEST_DIR = os.path.dirname(__file__)
WFO_ENGINE_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
sys.path.append(WFO_ENGINE_DIR)

from pine_v3.execution_gate import build_execution_gate_report


def test_gate_not_applicable_in_native_mode() -> None:
    report = build_execution_gate_report(strategy_mode="native_atdmf")
    assert report["status"] == "not_applicable"
    assert report["can_run"] is True
    assert report["blockers"] == []


def test_gate_blocks_when_beta_readiness_is_missing() -> None:
    report = build_execution_gate_report(strategy_mode="pine_imported")
    assert report["status"] == "blocked"
    assert report["can_run"] is False
    codes = {b.get("code") for b in report.get("blockers", [])}
    assert "beta_readiness_missing" in codes


def test_gate_blocks_when_beta_not_ready() -> None:
    report = build_execution_gate_report(
        strategy_mode="pine_imported",
        beta_readiness_report={"beta_ready": False, "readiness_score": 75.0, "runtime_blockers": [{}]},
    )
    assert report["status"] == "blocked"
    assert report["can_run"] is False
    codes = {b.get("code") for b in report.get("blockers", [])}
    assert "beta_readiness_failed" in codes


def test_gate_passes_when_beta_ready_and_no_parity_reference() -> None:
    report = build_execution_gate_report(
        strategy_mode="pine_imported",
        beta_readiness_report={"beta_ready": True, "readiness_score": 100.0, "runtime_blockers": []},
    )
    assert report["status"] == "pass"
    assert report["can_run"] is True


def test_gate_blocks_when_reference_is_present_but_parity_missing() -> None:
    report = build_execution_gate_report(
        strategy_mode="pine_imported",
        beta_readiness_report={"beta_ready": True},
        parity_reference_payload={"schema_version": "pine_parity_reference.v1"},
        parity_reference_validation={"valid": True, "errors": []},
        parity_report={},
    )
    assert report["status"] == "blocked"
    codes = {b.get("code") for b in report.get("blockers", [])}
    assert "parity_report_missing" in codes


def test_gate_blocks_when_parity_is_not_passed() -> None:
    report = build_execution_gate_report(
        strategy_mode="pine_imported",
        beta_readiness_report={"beta_ready": True},
        parity_reference_payload={"schema_version": "pine_parity_reference.v1"},
        parity_reference_validation={"valid": True, "errors": []},
        parity_report={"status": "failed", "parity_pass": False},
    )
    assert report["status"] == "blocked"
    codes = {b.get("code") for b in report.get("blockers", [])}
    assert "parity_not_passed" in codes


def test_gate_passes_when_parity_is_passed() -> None:
    report = build_execution_gate_report(
        strategy_mode="pine_imported",
        beta_readiness_report={"beta_ready": True},
        parity_reference_payload={"schema_version": "pine_parity_reference.v1"},
        parity_reference_validation={"valid": True, "errors": []},
        parity_report={"status": "passed", "parity_pass": True},
    )
    assert report["status"] == "pass"
    assert report["can_run"] is True


def test_gate_can_ignore_parity_when_disabled() -> None:
    report = build_execution_gate_report(
        strategy_mode="pine_imported",
        beta_readiness_report={"beta_ready": True},
        parity_reference_payload={"schema_version": "pine_parity_reference.v1"},
        parity_reference_validation={"valid": True, "errors": []},
        parity_report={},
        enforce_parity_when_reference=False,
    )
    assert report["status"] == "pass"
    assert report["can_run"] is True

