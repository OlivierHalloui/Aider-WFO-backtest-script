"""services/error_log.py — Write a structured error/warning log after each WFO run.

The log is a JSON file saved to a configurable directory.  It captures:
  - Fatal run exceptions
  - Per-window / per-stage failures and warnings
  - Config summary for traceability

Usage
-----
    from services.error_log import write_error_log

    path = write_error_log(
        run_type="classic",
        config=config,
        entries=[{"level": "ERROR", "context": "window 3", "message": "..."}],
        output_dir="reports/error_logs",
        run_ts="20260419_120000",
        status="FAILED",
        fatal_error="ValueError: ...",
    )
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_LOG_DIR = "reports/error_logs"


def write_error_log(
    run_type: str,
    config: dict,
    entries: list[dict],
    output_dir: str | Path | None = None,
    run_ts: str | None = None,
    status: str = "OK",
    fatal_error: str | None = None,
) -> str:
    """Write a JSON error log for a WFO run.

    Parameters
    ----------
    run_type    : "classic" | "stagewise"
    config      : full config dict (only a summary is stored)
    entries     : list of dicts with keys: level ("ERROR"|"WARNING"|"INFO"),
                  context (e.g. "stage 3", "window 2"), message (str)
    output_dir  : directory to write to; defaults to reports/error_logs/
    run_ts      : timestamp string used in the filename (YYYYMMDD_HHMMSS)
    status      : overall run status "OK" | "FAILED" | "INTERRUPTED"
    fatal_error : top-level exception message if run crashed

    Returns
    -------
    str  path to the written file
    """
    ts = run_ts or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir) if output_dir else Path(_DEFAULT_LOG_DIR)
    out.mkdir(parents=True, exist_ok=True)

    n_errors   = sum(1 for e in entries if e.get("level") == "ERROR")
    n_warnings = sum(1 for e in entries if e.get("level") == "WARNING")

    payload = {
        "run_type":     run_type,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "run_ts":       ts,
        "status":       status,
        "fatal_error":  fatal_error,
        "n_errors":     n_errors,
        "n_warnings":   n_warnings,
        "config_summary": _config_summary(config, run_type),
        "entries":      entries,
    }

    path = out / f"error_log_{run_type}_{ts}.json"
    try:
        path.write_text(
            json.dumps(payload, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        if n_errors or fatal_error:
            logger.warning(
                "Error log written → %s  (%d error(s), %d warning(s))",
                path, n_errors, n_warnings,
            )
        else:
            logger.info("Error log written → %s  (no errors)", path)
    except Exception as exc:
        logger.error("Could not write error log: %s", exc)

    return str(path)


# ── helpers ───────────────────────────────────────────────────────────────────

def _config_summary(config: dict, run_type: str) -> dict:
    """Extract traceability fields from the full config dict."""
    base = {
        "timeframe":     config.get("timeframe"),
        "direction":     config.get("strategy_direction") or config.get("direction"),
        "start_date":    str(config.get("start_date", "")),
        "end_date":      str(config.get("end_date", "")),
        "n_windows":     config.get("n_windows"),
        "train_size":    config.get("train_size"),
        "data_file":     config.get("file_path") or config.get("data_file", ""),
    }
    if run_type == "classic":
        base.update({
            "optimization_method": config.get("optimization_method"),
            "max_trials":          config.get("max_trials"),
            "selection_method":    config.get("selection_method", "snv"),
            "cross_window_method": config.get("cross_window_method", "best_is_oos"),
        })
    elif run_type == "stagewise":
        plan = config.get("stage_plan", [])
        base.update({
            "n_stages":            len(plan),
            "stage_consensus_method": config.get("stage_consensus_method", "median_mode"),
            "stages": [
                {"name": s.get("name"), "method": s.get("method"),
                 "params": s.get("optimize", [])}
                for s in plan
            ],
        })
    return base


def collect_classic_wfo_entries(wfo_results: dict | None) -> list[dict]:
    """Extract warning entries from a completed classic WFO results dict."""
    if not wfo_results:
        return []
    entries = []
    for wr in wfo_results.get("window_results", []):
        winfo = wr.get("window_info", {})
        wnum  = winfo.get("window", "?")
        oos   = wr.get("oos_performance", {})
        if oos and oos.get("n_trades", 1) == 0:
            entries.append({
                "level":   "WARNING",
                "context": f"window {wnum} OOS",
                "message": "0 trades generated on OOS period.",
            })
        is_p = wr.get("is_performance", {})
        if is_p and is_p.get("n_trades", 1) == 0:
            entries.append({
                "level":   "WARNING",
                "context": f"window {wnum} IS",
                "message": "0 trades generated on IS period.",
            })
    return entries


def collect_stagewise_entries(final_report: dict) -> list[dict]:
    """Extract error/warning entries from a stagewise final_report dict."""
    entries = []
    for stage in final_report.get("stages", []):
        snum = stage.get("stage", "?")
        name = stage.get("name", f"Stage {snum}")
        if stage.get("status") == "FAILED":
            entries.append({
                "level":   "ERROR",
                "context": f"stage {snum} — {name}",
                "message": stage.get("error", "Unknown error"),
            })
        elif stage.get("status") == "OK":
            oos_sharpe = stage.get("oos_avg_sharpe")
            if oos_sharpe is not None and oos_sharpe <= 0:
                entries.append({
                    "level":   "WARNING",
                    "context": f"stage {snum} — {name}",
                    "message": f"OOS avg Sharpe non-positif : {oos_sharpe:.3f}",
                })
    fb = final_report.get("final_backtest", {})
    if fb.get("status") == "FAILED":
        entries.append({
            "level":   "ERROR",
            "context": "final_backtest",
            "message": fb.get("error", "Unknown error"),
        })
    return entries
