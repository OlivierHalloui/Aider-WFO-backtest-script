"""P2.3 parity CI campaign helpers for Pine V3."""

from __future__ import annotations

import datetime
import importlib.util
import json
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .mtf_parity import build_mtf_parity_proof_report
from .parity import (
    DEFAULT_PARITY_THRESHOLDS,
    build_parity_report,
)


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _is_vectorbt_available() -> bool:
    return importlib.util.find_spec("vectorbtpro") is not None


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if hasattr(value, "iloc"):
            return float(value.iloc[0])  # type: ignore[index]
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except Exception:
        return float(default)


def _safe_rel_diff_pct(reference: float, current: float) -> float:
    denom = max(abs(float(reference)), 1.0)
    return abs(float(current) - float(reference)) / denom * 100.0


def _report_to_scenario(
    *,
    scenario_id: str,
    strategy_id: str,
    timeframe: str,
    parity_report: dict[str, Any],
    mtf_proof_report: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    checks = parity_report.get("checks") if isinstance(parity_report.get("checks"), list) else []
    failed_checks = [
        {
            "metric": str(item.get("metric") or ""),
            "diff": item.get("diff"),
            "threshold": item.get("threshold"),
            "status": str(item.get("status") or ""),
        }
        for item in checks
        if isinstance(item, dict) and item.get("passed") is False
    ]

    max_required_diff = 0.0
    for item in checks:
        if not isinstance(item, dict):
            continue
        if str(item.get("status")) != "ok":
            continue
        metric = str(item.get("metric") or "")
        if metric not in {"trade_count", "total_return_pct", "max_drawdown_pct"}:
            continue
        diff = item.get("diff")
        try:
            max_required_diff = max(max_required_diff, float(diff))
        except Exception:
            continue

    status = str(parity_report.get("status") or "unknown")
    parity_pass = parity_report.get("parity_pass")
    scenario_pass = bool(parity_pass is True)

    mtf_status = None
    mtf_pass = None
    mtf_blockers: list[dict[str, Any]] = []
    if isinstance(mtf_proof_report, dict) and mtf_proof_report:
        mtf_status = str(mtf_proof_report.get("status") or "unknown")
        mtf_pass = mtf_proof_report.get("proof_pass")
        blockers = mtf_proof_report.get("blockers")
        mtf_blockers = blockers if isinstance(blockers, list) else []
        if mtf_pass is False:
            scenario_pass = False

    return {
        "scenario_id": scenario_id,
        "strategy_id": strategy_id,
        "timeframe": timeframe,
        "status": "passed" if scenario_pass else "failed",
        "parity_status": status,
        "parity_pass": parity_pass,
        "max_required_diff_pct": round(float(max_required_diff), 6),
        "failed_checks": failed_checks,
        "mtf_proof_status": mtf_status,
        "mtf_proof_pass": mtf_pass,
        "mtf_blockers": mtf_blockers,
        "notes": [str(x) for x in (notes or []) if str(x).strip()],
        "parity_report": parity_report,
        "mtf_proof_report": mtf_proof_report if isinstance(mtf_proof_report, dict) else None,
    }


def _deterministic_campaign_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []

    report_core = build_parity_report(
        reference_metrics={
            "trade_count": 100,
            "entry_count": 100,
            "exit_count": 100,
            "total_return_pct": 12.0,
            "max_drawdown_pct": -3.5,
        },
        current_metrics={
            "trade_count": 101,
            "entry_count": 101,
            "exit_count": 101,
            "total_return_pct": 11.7,
            "max_drawdown_pct": -3.2,
        },
        thresholds={
            "trade_count_rel_pct": 2.0,
            "entry_count_rel_pct": 2.0,
            "exit_count_rel_pct": 2.0,
            "total_return_abs_pct": 1.0,
            "max_drawdown_abs_pct": 1.0,
        },
    )
    scenarios.append(
        _report_to_scenario(
            scenario_id="core-parity-atdmf-proxy-5s",
            strategy_id="atdmf_proxy",
            timeframe="5s",
            parity_report=report_core,
            notes=["Deterministic baseline parity scenario for core metrics."],
        )
    )

    report_detail = build_parity_report(
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
    scenarios.append(
        _report_to_scenario(
            scenario_id="detail-parity-pine-demo-1m",
            strategy_id="pine_demo",
            timeframe="1m",
            parity_report=report_detail,
            notes=["Deterministic event/trade-level parity scenario."],
        )
    )

    mtf_diag = {
        "status": "available",
        "request_security_count": 1,
        "rows": [
            {
                "line": 10,
                "target": "sma_htf",
                "timeframe": "4h",
                "non_na_count": 120,
                "change_count": 14,
                "base_bar_count": 240,
            }
        ],
        "warnings": [],
    }
    report_mtf = build_parity_report(
        reference_metrics={"trade_count": 3, "total_return_pct": 9.0, "max_drawdown_pct": -1.0},
        current_metrics={"trade_count": 3, "total_return_pct": 9.1, "max_drawdown_pct": -0.9},
        thresholds={"trade_count_rel_pct": 2.0, "total_return_abs_pct": 1.0, "max_drawdown_abs_pct": 1.0},
    )
    mtf_proof = build_mtf_parity_proof_report(
        strategy_spec={"capabilities": {"uses_request_security": True}},
        request_security_diagnostics=mtf_diag,
        parity_report=report_mtf,
    )
    scenarios.append(
        _report_to_scenario(
            scenario_id="mtf-proof-pine-demo-1h",
            strategy_id="pine_demo_mtf",
            timeframe="1h",
            parity_report=report_mtf,
            mtf_proof_report=mtf_proof,
            notes=["Deterministic MTF proof scenario tied to parity report."],
        )
    )
    return scenarios


_RUNTIME_BASELINES = {
    "pine-mtf-runtime-1h": {
        "trade_count": 3.0,
        "entry_count": 3.0,
        "exit_count": 3.0,
        "total_return_pct": 30.951433716290193,
        "max_drawdown_pct": 0.04277596534800754,
    },
    "pine-mtf-runtime-30min": {
        "trade_count": 3.0,
        "entry_count": 3.0,
        "exit_count": 3.0,
        "total_return_pct": 28.97160242340762,
        "max_drawdown_pct": 0.20075328760320588,
    },
}


@dataclass(frozen=True)
class _RuntimeScenario:
    scenario_id: str
    strategy_id: str
    timeframe: str
    bars: int
    htf_expr: str


def _mock_ohlcv_df(n: int, freq: str) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq=freq, tz="UTC")
    base = np.linspace(100.0, 130.0, n)
    wave = 2.5 * np.sin(np.linspace(0, 15, n))
    close = base + wave
    return pd.DataFrame(
        {
            "Open": close - 0.2,
            "High": close + 0.4,
            "Low": close - 0.4,
            "Close": close,
            "Volume": np.full(n, 1.0),
        },
        index=idx,
    )


def _runtime_mtf_spec(htf_expr: str = "'4h'") -> dict[str, Any]:
    return {
        "schema_version": "strategy_spec.v1",
        "strategy": {"id": "pine_mtf_demo", "name": "Pine MTF Demo", "kind": "pine_imported"},
        "source": {
            "file_name": "runtime_ci_demo.pine",
            "source_sha1": "runtime_ci",
            "line_count": 8,
            "char_count": 200,
            "pine_version": 6,
        },
        "imports": [],
        "inputs": [],
        "warnings": [],
        "capabilities": {
            "uses_request_security": True,
            "uses_request_security_lower_tf": False,
            "uses_strategy_entry": True,
            "uses_strategy_exit": True,
            "uses_strategy_close": False,
            "uses_strategy_cancel": False,
        },
        "logic": {
            "assignments": [
                {"line": 1, "targets": ["TF"], "op": "=", "expr": htf_expr},
                {
                    "line": 2,
                    "targets": ["sma_htf"],
                    "op": "=",
                    "expr": "request.security(syminfo.tickerid, TF, ta.sma(close, 3))",
                },
            ],
            "order_rules": [
                {
                    "line": 3,
                    "action": "entry",
                    "id": "L",
                    "direction": "strategy.long",
                    "condition_expr": "ta.crossover(close, sma_htf)",
                    "call": "strategy.entry(id='L', direction=strategy.long)",
                },
                {
                    "line": 4,
                    "action": "exit",
                    "id": "X",
                    "direction": "",
                    "condition_expr": "ta.crossunder(close, sma_htf)",
                    "call": "strategy.exit(id='X', from_entry='L')",
                },
            ],
        },
    }


def _runtime_campaign_scenarios() -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if not _is_vectorbt_available():
        warnings.append("vectorbtpro non disponible: scénarios runtime MTF ignorés.")
        return [], warnings

    try:
        from .runtime_adapter import (
            build_request_security_diagnostics,
            run_transpiled_pine_backtest,
        )
    except Exception as e:
        warnings.append(f"Import runtime_adapter impossible: {e}")
        return [], warnings

    scenarios_def = [
        _RuntimeScenario(
            scenario_id="pine-mtf-runtime-1h",
            strategy_id="pine_mtf_demo",
            timeframe="1h",
            bars=240,
            htf_expr="'4h'",
        ),
        _RuntimeScenario(
            scenario_id="pine-mtf-runtime-30min",
            strategy_id="pine_mtf_demo",
            timeframe="30min",
            bars=240,
            htf_expr="'4h'",
        ),
    ]
    out: list[dict[str, Any]] = []

    params = {
        "metric1_name": "total_return",
        "metric2_name": "sharpe_ratio",
        "weight_metric1": 1.0,
        "weight_metric2": 0.0,
    }
    thresholds = dict(DEFAULT_PARITY_THRESHOLDS)
    thresholds.update(
        {
            "trade_count_rel_pct": 1.0,
            "entry_count_rel_pct": 1.0,
            "exit_count_rel_pct": 1.0,
            "total_return_abs_pct": 2.5,
            "max_drawdown_abs_pct": 0.5,
        }
    )

    for scenario in scenarios_def:
        spec = _runtime_mtf_spec(htf_expr=scenario.htf_expr)
        df = _mock_ohlcv_df(n=scenario.bars, freq=scenario.timeframe)
        note_messages: list[str] = []
        try:
            portfolio = run_transpiled_pine_backtest(
                df=df,
                params=params,
                strategy_spec=spec,
                timeframe=scenario.timeframe,
                return_portfolio=True,
                external_bindings={},
            )
            diagnostics = build_request_security_diagnostics(
                df=df,
                params=params,
                strategy_spec=spec,
                external_bindings={},
            )
        except Exception as e:
            out.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "strategy_id": scenario.strategy_id,
                    "timeframe": scenario.timeframe,
                    "status": "failed",
                    "parity_status": "runtime_error",
                    "parity_pass": False,
                    "max_required_diff_pct": math.inf,
                    "failed_checks": [{"metric": "runtime", "status": f"error: {e}"}],
                    "mtf_proof_status": "failed",
                    "mtf_proof_pass": False,
                    "mtf_blockers": [{"code": "runtime_exception", "label": "Runtime exception", "detail": str(e)}],
                    "notes": [],
                    "parity_report": {},
                    "mtf_proof_report": {},
                }
            )
            continue

        current_metrics = {
            "trade_count": _to_float(portfolio.trades.count()),
            "entry_count": _to_float(portfolio.trades.count()),
            "exit_count": _to_float(portfolio.trades.count()),
            "total_return_pct": _to_float(portfolio.total_return) * 100.0,
            "max_drawdown_pct": _to_float(portfolio.max_drawdown) * -100.0,
        }

        baseline = _RUNTIME_BASELINES.get(scenario.scenario_id, {})
        if not baseline:
            note_messages.append("Baseline runtime manquante: scénario exécuté mais non scoré.")
            parity_report = build_parity_report(
                reference_metrics={
                    "trade_count": current_metrics["trade_count"],
                    "entry_count": current_metrics["entry_count"],
                    "exit_count": current_metrics["exit_count"],
                    "total_return_pct": current_metrics["total_return_pct"],
                    "max_drawdown_pct": current_metrics["max_drawdown_pct"],
                },
                current_metrics=current_metrics,
                thresholds=thresholds,
            )
        else:
            parity_report = build_parity_report(
                reference_metrics=baseline,
                current_metrics=current_metrics,
                thresholds=thresholds,
            )
            ret_diff = _safe_rel_diff_pct(
                float(baseline.get("total_return_pct", 0.0)),
                float(current_metrics.get("total_return_pct", 0.0)),
            )
            note_messages.append(
                f"runtime_baseline_rel_diff_total_return_pct={ret_diff:.4f}%"
            )

        mtf_proof = build_mtf_parity_proof_report(
            strategy_spec=spec,
            request_security_diagnostics=diagnostics,
            parity_report=parity_report,
        )
        out.append(
            _report_to_scenario(
                scenario_id=scenario.scenario_id,
                strategy_id=scenario.strategy_id,
                timeframe=scenario.timeframe,
                parity_report=parity_report,
                mtf_proof_report=mtf_proof,
                notes=note_messages,
            )
        )
    return out, warnings


def run_parity_ci_campaign(require_runtime: bool = False) -> dict[str, Any]:
    """
    Run parity campaign for CI and return a machine-readable report.

    Campaign contains:
    - deterministic parity checks (always available),
    - optional runtime MTF checks when `vectorbtpro` is available.
    """
    deterministic = _deterministic_campaign_scenarios()
    runtime, runtime_warnings = _runtime_campaign_scenarios()
    scenarios = [*deterministic, *runtime]

    runtime_executed = len(runtime) > 0
    blockers: list[dict[str, Any]] = []
    if require_runtime and not runtime_executed:
        blockers.append(
            {
                "code": "runtime_campaign_missing",
                "label": "Scénarios runtime requis mais non exécutés",
                "detail": "Installe vectorbtpro ou lance la campagne sans --require-runtime.",
            }
        )

    failed = [row for row in scenarios if isinstance(row, dict) and str(row.get("status")) == "failed"]
    skipped = [row for row in scenarios if isinstance(row, dict) and str(row.get("status")) == "skipped"]
    passed = [row for row in scenarios if isinstance(row, dict) and str(row.get("status")) == "passed"]

    status = "passed"
    if blockers or failed:
        status = "failed"
    elif not scenarios:
        status = "failed"
        blockers.append(
            {
                "code": "empty_campaign",
                "label": "Aucun scénario de parité exécuté",
                "detail": "Campagne invalide: aucune preuve de parité n'a été produite.",
            }
        )

    return {
        "schema_version": "pine_parity_ci_report.v1",
        "generated_at_utc": _utc_now_iso(),
        "status": status,
        "require_runtime": bool(require_runtime),
        "runtime_executed": runtime_executed,
        "runtime_warnings": runtime_warnings,
        "summary": {
            "total_scenarios": len(scenarios),
            "passed": len(passed),
            "failed": len(failed),
            "skipped": len(skipped),
            "blockers": len(blockers),
        },
        "scenarios": scenarios,
        "blockers": blockers,
    }


def write_parity_ci_report(report: dict[str, Any], output_path: str) -> str:
    """Persist parity campaign report to disk."""
    target = str(output_path or "").strip()
    if not target:
        target = os.path.join("reports", "ci", "pine_parity_ci_report.json")
    folder = os.path.dirname(target)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return target

