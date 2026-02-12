"""Tests for Pine V3 parity helpers."""

from __future__ import annotations

import os
import sys

TEST_DIR = os.path.dirname(__file__)
WFO_ENGINE_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
sys.path.append(WFO_ENGINE_DIR)

from pine_v3.parity import (
    build_parity_report,
    build_parity_reference_payload,
    validate_parity_reference_payload,
    normalize_metrics,
    normalize_events,
    normalize_trade_pairs,
    PARITY_REFERENCE_SCHEMA_VERSION,
)


def test_normalize_metrics_handles_aliases() -> None:
    raw = {
        "trades": 100,
        "entry_trades": 100,
        "exit_trades": 100,
        "return_percent": 12.5,
        "mdd_pct": -3.2,
        "ignored": "abc",
    }
    out = normalize_metrics(raw)
    assert out["trade_count"] == 100.0
    assert out["entry_count"] == 100.0
    assert out["exit_count"] == 100.0
    assert out["total_return_pct"] == 12.5
    assert out["max_drawdown_pct"] == -3.2
    assert "ignored" not in out


def test_build_parity_report_passes_when_diffs_are_within_thresholds() -> None:
    report = build_parity_report(
        reference_metrics={
            "trade_count": 100,
            "entry_count": 100,
            "exit_count": 100,
            "total_return_pct": 10.0,
            "max_drawdown_pct": -5.0,
        },
        current_metrics={
            "trade_count": 101,
            "entry_count": 101,
            "exit_count": 101,
            "total_return_pct": 9.0,
            "max_drawdown_pct": -4.0,
        },
        thresholds={
            "trade_count_rel_pct": 2.0,
            "entry_count_rel_pct": 2.0,
            "exit_count_rel_pct": 2.0,
            "total_return_abs_pct": 2.0,
            "max_drawdown_abs_pct": 2.0,
        },
    )
    assert report["schema_version"] == "pine_parity_report.v1"
    assert report["status"] == "passed"
    assert report["parity_pass"] is True


def test_build_parity_report_fails_when_required_metric_exceeds_threshold() -> None:
    report = build_parity_report(
        reference_metrics={
            "trade_count": 100,
            "total_return_pct": 10.0,
            "max_drawdown_pct": -5.0,
        },
        current_metrics={
            "trade_count": 150,  # 50% diff
            "total_return_pct": 2.0,
            "max_drawdown_pct": -5.0,
        },
        thresholds={"trade_count_rel_pct": 2.0, "total_return_abs_pct": 3.0, "max_drawdown_abs_pct": 3.0},
    )
    assert report["status"] == "failed"
    assert report["parity_pass"] is False


def test_build_parity_report_marks_insufficient_reference() -> None:
    report = build_parity_report(
        reference_metrics={"trade_count": 100},
        current_metrics={"trade_count": 100, "total_return_pct": 8.0, "max_drawdown_pct": -3.0},
    )
    assert report["status"] == "insufficient_reference"
    assert report["parity_pass"] is None


def test_build_parity_reference_payload_sets_canonical_schema() -> None:
    payload = build_parity_reference_payload(
        reference_metrics={"trades": 10, "total_return": 4.2, "max_drawdown": -1.3},
        source={"provider": "tradingview", "strategy_id": "s1"},
    )
    assert payload["schema_version"] == PARITY_REFERENCE_SCHEMA_VERSION
    assert payload["source"]["provider"] == "tradingview"
    assert payload["reference_metrics"]["trade_count"] == 10.0
    assert payload["reference_metrics"]["total_return_pct"] == 4.2


def test_validate_parity_reference_payload_accepts_valid_v1() -> None:
    payload = {
        "schema_version": PARITY_REFERENCE_SCHEMA_VERSION,
        "source": {"provider": "tv"},
        "reference_metrics": {
            "trade_count": 100,
            "entry_count": 100,
            "exit_count": 100,
            "total_return_pct": 12.0,
            "max_drawdown_pct": -3.0,
        },
    }
    validation = validate_parity_reference_payload(payload, allow_legacy=False)
    assert validation["valid"] is True, validation
    normalized = validation["normalized_payload"]
    assert normalized["schema_version"] == PARITY_REFERENCE_SCHEMA_VERSION
    assert normalized["reference_metrics"]["trade_count"] == 100.0


def test_validate_parity_reference_payload_can_migrate_legacy_dict() -> None:
    legacy_payload = {
        "trades": 50,
        "total_return": 8.0,
        "max_drawdown": -2.0,
    }
    validation = validate_parity_reference_payload(legacy_payload, allow_legacy=True)
    assert validation["valid"] is True, validation
    assert validation["normalized_payload"]["schema_version"] == PARITY_REFERENCE_SCHEMA_VERSION


def test_validate_parity_reference_payload_rejects_missing_required_metrics() -> None:
    payload = {
        "schema_version": PARITY_REFERENCE_SCHEMA_VERSION,
        "reference_metrics": {
            "trade_count": 50,
            "total_return_pct": 8.0,
        },
    }
    validation = validate_parity_reference_payload(payload, allow_legacy=False)
    assert validation["valid"] is False
    assert any("max_drawdown_pct" in err for err in validation["errors"])


def test_normalize_events_and_trades() -> None:
    events = normalize_events(
        {
            "entry_events": ["2025-01-01T00:00:00+00:00", "2025-01-01T00:01:00+00:00"],
            "exit_events": ["2025-01-01T00:02:00+00:00"],
        }
    )
    trades = normalize_trade_pairs(
        [
            {"entry_time": "2025-01-01T00:00:00+00:00", "exit_time": "2025-01-01T00:02:00+00:00"},
            {"entry_ts": "2025-01-01T00:03:00+00:00", "exit_ts": "2025-01-01T00:05:00+00:00"},
        ]
    )
    assert len(events["entries"]) == 2
    assert len(events["exits"]) == 1
    assert len(trades) == 2


def test_build_parity_report_detailed_pass() -> None:
    report = build_parity_report(
        reference_metrics={
            "trade_count": 2,
            "entry_count": 2,
            "exit_count": 2,
            "total_return_pct": 5.0,
            "max_drawdown_pct": -2.0,
        },
        current_metrics={
            "trade_count": 2,
            "entry_count": 2,
            "exit_count": 2,
            "total_return_pct": 5.0,
            "max_drawdown_pct": -2.0,
        },
        reference_events={
            "entries": ["2025-01-01T00:00:00+00:00", "2025-01-01T00:03:00+00:00"],
            "exits": ["2025-01-01T00:02:00+00:00", "2025-01-01T00:05:00+00:00"],
        },
        current_events={
            "entries": ["2025-01-01T00:00:01+00:00", "2025-01-01T00:03:01+00:00"],
            "exits": ["2025-01-01T00:02:01+00:00", "2025-01-01T00:05:01+00:00"],
        },
        reference_trades=[
            {"entry_time": "2025-01-01T00:00:00+00:00", "exit_time": "2025-01-01T00:02:00+00:00"},
            {"entry_time": "2025-01-01T00:03:00+00:00", "exit_time": "2025-01-01T00:05:00+00:00"},
        ],
        current_trades=[
            {"entry_time": "2025-01-01T00:00:01+00:00", "exit_time": "2025-01-01T00:02:01+00:00"},
            {"entry_time": "2025-01-01T00:03:01+00:00", "exit_time": "2025-01-01T00:05:01+00:00"},
        ],
        detail_thresholds={
            "event_time_tolerance_sec": 2.0,
            "trade_time_tolerance_sec": 2.0,
            "entry_event_match_min_ratio": 1.0,
            "exit_event_match_min_ratio": 1.0,
            "trade_match_min_ratio": 1.0,
        },
    )
    assert report["detail_available"] is True
    assert report["detail_pass"] is True
    assert report["parity_pass"] is True


def test_build_parity_report_detailed_fail_on_time_mismatch() -> None:
    report = build_parity_report(
        reference_metrics={
            "trade_count": 1,
            "entry_count": 1,
            "exit_count": 1,
            "total_return_pct": 1.0,
            "max_drawdown_pct": -1.0,
        },
        current_metrics={
            "trade_count": 1,
            "entry_count": 1,
            "exit_count": 1,
            "total_return_pct": 1.0,
            "max_drawdown_pct": -1.0,
        },
        reference_events={
            "entries": ["2025-01-01T00:00:00+00:00"],
            "exits": ["2025-01-01T00:02:00+00:00"],
        },
        current_events={
            "entries": ["2025-01-01T01:00:00+00:00"],
            "exits": ["2025-01-01T01:02:00+00:00"],
        },
        detail_thresholds={
            "event_time_tolerance_sec": 1.0,
            "entry_event_match_min_ratio": 1.0,
            "exit_event_match_min_ratio": 1.0,
        },
    )
    assert report["detail_available"] is True
    assert report["detail_pass"] is False
    assert report["parity_pass"] is False
