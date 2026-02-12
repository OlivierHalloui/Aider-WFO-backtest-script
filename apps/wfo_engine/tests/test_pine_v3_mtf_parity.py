"""Tests for dedicated MTF parity proof report."""

from __future__ import annotations

import os
import sys

TEST_DIR = os.path.dirname(__file__)
WFO_ENGINE_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
sys.path.append(WFO_ENGINE_DIR)

from pine_v3.mtf_parity import build_mtf_parity_proof_report


def test_mtf_parity_proof_pass_when_diagnostics_are_present() -> None:
    spec = {"capabilities": {"uses_request_security": True}}
    diagnostics = {
        "status": "available",
        "request_security_count": 1,
        "rows": [
            {
                "line": 10,
                "target": "sma_htf",
                "timeframe": "4h",
                "non_na_count": 100,
                "change_count": 12,
                "base_bar_count": 200,
            }
        ],
        "warnings": [],
    }
    parity = {"parity_pass": True}
    report = build_mtf_parity_proof_report(spec, diagnostics, parity)
    assert report["status"] == "pass"
    assert report["proof_pass"] is True


def test_mtf_parity_proof_fails_when_diagnostics_missing() -> None:
    spec = {"capabilities": {"uses_request_security": True}}
    report = build_mtf_parity_proof_report(spec, {}, {"parity_pass": True})
    assert report["status"] == "failed"
    assert report["proof_pass"] is False
    assert any(str(b.get("code")) == "mtf_diagnostics_available" for b in report.get("blockers", []))

