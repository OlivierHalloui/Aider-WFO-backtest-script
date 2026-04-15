#!/usr/bin/env python3
"""
stagewise_optimizer.py — Stagewise WFO campaign, fully headless.

Runs N stages of Walk-Forward Optimization sequentially.  Each stage
optimizes a sub-set of parameters while holding the best values found in
previous stages fixed.  The final report aggregates all per-stage results
into a single JSON artefact.

Usage (from repo root):
    PYTHONPATH=apps/wfo_engine python apps/wfo_engine/stagewise_optimizer.py \
        --data-file  "Data/DBAD/spot/BTCUSDT/Binance_BTCUSDT_2025-01-01_2026-04-06_5s.csv" \
        --timeframe  5s \
        --start      2025-04-01 \
        --end        2025-08-15 \
        --direction  long_only \
        --output     reports/stagewise/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── project imports ──────────────────────────────────────────────────────────
from config import WFOSettings, DEFAULT_PARAM_GRID
from strategy_adapters import resolve_strategy_adapter
from wfo import walk_forward_optimization

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [stagewise] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  FULL PARAMETER REGISTRY
#     All optimisable params with their search ranges.
#     Continuous/integer → (min, max, step)
#     Boolean            → [True, False]
# ═══════════════════════════════════════════════════════════════════════════════
FULL_PARAM_REGISTRY: dict[str, Any] = {
    # ── BB core ──
    "timeperiod":              (8,   20,  2),
    "StDev":                   (0.8,  2.0, 0.2),
    # ── BBW compression ──
    "coeff_medianeBBW":        (0.9,  1.5, 0.1),
    "nb_bars_under_bbw_mini":  (1,    8,   1),
    # ── BB% filter ──
    "coef_mediane":            (0.7,  1.1, 0.1),
    "fenetre_lowest":          (40,   120, 20),
    "seuil_lowest":            (1.0,  3.5, 0.5),
    "longueur_mediane":        (50,   150, 50),
    "Nb_bars_above":           (1,    6,   1),
    "nb_bars_entre_bb":        (1,    10,  1),
    # ── SMA exit ──
    "user_exit_sma_length":    (8,   20,  2),
    # ── SAR exit ──
    "sar_start":               (0.02, 0.05, 0.01),
    "sar_increment":           (0.02, 0.05, 0.01),
    "sar_maximum":             (0.1,  0.3,  0.05),
    # ── MACD exit ──
    "macd_fast_length":        (6,   14,  2),
    "macd_slow_length":        (14,  26,  2),
    "macd_signal_length":      (4,   10,  2),
    # ── entry filter booleans ──
    "use_t2_signal":           [True, False],
    "use_roc_filter":          [True, False],
    "use_divergence_bb":       [True, False],
    # ── exit booleans ──
    "exit_sar_enabled":        [True, False],
    "exit_macd_enabled":       [True, False],
    "exit_macd_type_a":        [True, False],
    "exit_macd_type_b":        [True, False],
    "exit_cross_sar_sma_enabled": [True, False],
    "exit_retour_bb_enabled":  [True, False],
    "exit_regline_enabled":    [True, False],
    "exit_volat_down_enabled": [True, False],
}

# Default fixed values for every param not currently being optimised.
PARAM_DEFAULTS: dict[str, Any] = {
    "timeperiod":              14,
    "StDev":                   1.5,
    "coeff_medianeBBW":        1.1,
    "nb_bars_under_bbw_mini":  3,
    "coef_mediane":            0.9,
    "fenetre_lowest":          80,
    "seuil_lowest":            2.0,
    "longueur_mediane":        100,
    "Nb_bars_above":           3,
    "nb_bars_entre_bb":        5,
    "user_exit_sma_length":    14,
    "sar_start":               0.02,
    "sar_increment":           0.02,
    "sar_maximum":             0.2,
    "macd_fast_length":        8,
    "macd_slow_length":        20,
    "macd_signal_length":      6,
    "use_t2_signal":           True,
    "use_roc_filter":          False,
    "use_divergence_bb":       False,
    "exit_sar_enabled":        False,
    "exit_macd_enabled":       False,
    "exit_macd_type_a":        True,
    "exit_macd_type_b":        True,
    "exit_cross_sar_sma_enabled": False,
    "exit_retour_bb_enabled":  False,
    "exit_regline_enabled":    False,
    "exit_volat_down_enabled": False,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  STAGE PLAN  (7 stages)
#     Each stage defines:
#       name            – human label
#       optimize        – params to search (keys from FULL_PARAM_REGISTRY)
#       fixed_overrides – params to FORCE for this stage (ignore accumulated best)
#       method          – 'bayesian' | 'grid'
#       max_trials      – Bayesian trial budget
#       patience        – 'Low' | 'Medium' | 'High'
#       stability       – neighbor count for get_stable_best_params
# ═══════════════════════════════════════════════════════════════════════════════
STAGE_PLAN: list[dict] = [
    {
        "name":     "Run 1 — BB core + BBW compression",
        "optimize": [
            "timeperiod", "StDev",
            "coeff_medianeBBW", "nb_bars_under_bbw_mini",
        ],
        "fixed_overrides": {
            "use_t2_signal":           True,
            "use_roc_filter":          False,
            "use_divergence_bb":       False,
            "exit_sar_enabled":        False,
            "exit_macd_enabled":       False,
            "exit_cross_sar_sma_enabled": False,
            "exit_retour_bb_enabled":  False,
            "exit_regline_enabled":    False,
            "exit_volat_down_enabled": False,
        },
        "method":     "bayesian",
        "max_trials": 100,
        "patience":   "Low",
        "stability":  3,
    },
    {
        "name":     "Run 2 — BB% filter + SMA exit",
        "optimize": [
            "coef_mediane", "fenetre_lowest", "seuil_lowest",
            "longueur_mediane", "Nb_bars_above", "nb_bars_entre_bb",
            "user_exit_sma_length",
        ],
        "fixed_overrides": {},
        "method":     "bayesian",
        "max_trials": 150,
        "patience":   "Medium",
        "stability":  3,
    },
    {
        "name":     "Run 3 — Entry filter booleans (T2 / RoC / Divergence)",
        "optimize": [
            "use_t2_signal", "use_roc_filter", "use_divergence_bb",
        ],
        "fixed_overrides": {},
        "method":     "grid",
        "max_trials": 8,        # 2³ — exhaustive
        "patience":   "Low",
        "stability":  3,
    },
    {
        "name":     "Run 4 — SAR exit calibration",
        "optimize": [
            "exit_sar_enabled",
            "sar_start", "sar_increment", "sar_maximum",
        ],
        "fixed_overrides": {},
        "method":     "bayesian",
        "max_trials": 100,
        "patience":   "Low",
        "stability":  3,
    },
    {
        "name":     "Run 5 — MACD exit calibration",
        "optimize": [
            "exit_macd_enabled",
            "exit_macd_type_a", "exit_macd_type_b",
            "macd_fast_length", "macd_slow_length", "macd_signal_length",
        ],
        "fixed_overrides": {},
        "method":     "bayesian",
        "max_trials": 150,
        "patience":   "Medium",
        "stability":  3,
    },
    {
        "name":     "Run 6 — Cross / supplementary exits (booleans)",
        "optimize": [
            "exit_cross_sar_sma_enabled",
            "exit_retour_bb_enabled",
            "exit_regline_enabled",
            "exit_volat_down_enabled",
        ],
        "fixed_overrides": {},
        "method":     "grid",
        "max_trials": 16,       # 2⁴ — exhaustive
        "patience":   "Low",
        "stability":  3,
    },
    {
        "name":     "Run 7 — Re-calibration signal (BB core + BB% + SMA)",
        "optimize": [
            "timeperiod", "StDev",
            "coeff_medianeBBW", "nb_bars_under_bbw_mini",
            "coef_mediane", "fenetre_lowest", "seuil_lowest",
            "longueur_mediane", "Nb_bars_above", "nb_bars_entre_bb",
            "user_exit_sma_length",
        ],
        "fixed_overrides": {},   # exits + filters fixed from Runs 3-6
        "method":     "bayesian",
        "max_trials": 200,
        "patience":   "Medium",
        "stability":  5,        # final run — higher stability
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  FIXED-PARAM ADAPTER WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════
class _FixedParamAdapter:
    """
    Wraps any StrategyAdapter and injects fixed_params into every call.
    The optimisation engine only varies *stage* params; all others are
    transparently provided by this wrapper.
    """

    def __init__(self, base_adapter, fixed_params: dict[str, Any]):
        self._base = base_adapter
        self._fixed = fixed_params
        self.strategy_mode = getattr(base_adapter, "strategy_mode", "native_atdmf")
        self.strategy_id = getattr(base_adapter, "strategy_id", "atdmf")

    def _merge(self, params: dict) -> dict:
        return {**self._fixed, **params}

    def get_param_space(self) -> dict:
        return self._base.get_param_space()

    def generate_signals(self, df, params: dict, **kw):
        return self._base.generate_signals(df, self._merge(params), **kw)

    def run_backtest(self, df, params: dict, timeframe: str = "5s",
                     return_portfolio: bool = True):
        return self._base.run_backtest(
            df, self._merge(params),
            timeframe=timeframe, return_portfolio=return_portfolio,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _build_stage_param_grid(stage: dict) -> dict:
    """Return a param_grid dict containing only the params to optimise."""
    grid = {}
    for key in stage["optimize"]:
        if key not in FULL_PARAM_REGISTRY:
            raise KeyError(f"Unknown param '{key}' in stage '{stage['name']}'")
        grid[key] = FULL_PARAM_REGISTRY[key]
    return grid


def _consensus_params(wfo_results: dict) -> dict:
    """
    Aggregate per-window best_params to a single representative dict.
    Numeric  → median across windows.
    Boolean  → majority vote (mode).
    Integer  → rounded median.
    """
    rows = wfo_results.get("best_params", [])
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    result: dict[str, Any] = {}
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        unique = set(series.unique())
        if unique.issubset({True, False, 0, 1}):
            result[col] = bool(series.mode().iloc[0])
        elif pd.api.types.is_integer_dtype(series):
            result[col] = int(round(float(series.median())))
        elif all(float(v) == int(float(v)) for v in series if v == v):
            result[col] = int(round(float(series.median())))
        else:
            result[col] = float(series.median())
    return result


def _make_wfo_settings(stage: dict, direction: str,
                        n_windows: int = 6,
                        train_size: float = 0.75) -> WFOSettings:
    return WFOSettings(
        n_windows=n_windows,
        train_size=train_size,
        anchored=False,
        optimization_method=stage["method"],
        max_trials=stage["max_trials"],
        patience_level=stage["patience"],
        parallel_backend="dask",
        optimization_metric="sharpe_ratio",
        secondary_metric="total_return",
        metric_weights=(1.0, 0.0),
        strategy_direction=direction,
    )


def _json_safe(obj):
    """Recursively make an object JSON-serialisable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN CAMPAIGN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════
def run_final_backtest_headless(
    best_params: dict,
    df: pd.DataFrame,
    timeframe: str,
    direction: str,
    output_dir: Path,
) -> dict:
    """
    Run a full backtest with the accumulated best params on *df* and save
    the metrics to ``final_backtest_result.json`` inside *output_dir*.

    Parameters
    ----------
    best_params  : complete param dict (all params, not just optimised ones)
    df           : OHLCV DataFrame for the final backtest period
    timeframe    : pandas frequency string
    direction    : 'long_only' | 'short_only' | 'both'
    output_dir   : directory where the JSON result is written

    Returns
    -------
    dict  with keys: start, end, n_bars, timeframe, direction, params, metrics
    """
    from strategy import run_backtest
    from metrics import portfolio_metrics as _portfolio_metrics

    params = {**best_params, "strategy_direction": direction}
    logger.info("Final backtest: %d bars  [%s → %s]",
                len(df), df.index.min(), df.index.max())

    try:
        pf = run_backtest(df, params, timeframe=timeframe, return_portfolio=True)
        metrics = _portfolio_metrics(pf, window_id="final")
    except Exception as exc:
        logger.error("Final backtest failed: %s", exc, exc_info=True)
        return {"status": "FAILED", "error": str(exc)}

    result = {
        "status":    "OK",
        "start":     str(df.index.min()),
        "end":       str(df.index.max()),
        "n_bars":    len(df),
        "timeframe": timeframe,
        "direction": direction,
        "params":    _json_safe(best_params),
        "metrics":   _json_safe(metrics),
    }

    path = output_dir / "final_backtest_result.json"
    with path.open("w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info("Final backtest saved → %s", path)
    logger.info(
        "  Return=%.2f%%  Sharpe=%.3f  MaxDD=%.2f%%  Trades=%d",
        metrics.get("return", 0),
        metrics.get("sharpe", 0),
        metrics.get("max_drawdown", 0),
        metrics.get("n_trades", 0),
    )
    return result


def run_stagewise_campaign(
    df: pd.DataFrame,
    timeframe: str = "5s",
    direction: str = "long_only",
    stage_plan: list[dict] | None = None,
    n_windows: int = 6,
    train_size: float = 0.75,
    output_dir: Path | None = None,
    resume_from_stage: int = 0,
    initial_fixed_params: dict | None = None,
    final_df: pd.DataFrame | None = None,
    stage_callback=None,
) -> dict:
    """
    Run the full stagewise WFO campaign.

    Parameters
    ----------
    df                  : full OHLCV DataFrame (date-filtered externally)
    timeframe           : pandas frequency string ('5s', '1m', …)
    direction           : 'long_only' | 'short_only' | 'both'
    stage_plan          : list of stage dicts (defaults to STAGE_PLAN)
    n_windows           : number of WFO windows (identical across all stages)
    train_size          : IS fraction (identical across all stages)
    output_dir          : directory for per-stage JSON files + final report
    resume_from_stage   : skip stages 0…N-1 (load their results from output_dir)
    initial_fixed_params: pre-set param values (e.g. loaded from a previous run)
    final_df            : if provided, run a full final backtest after stage 7 and
                          save results to ``final_backtest_result.json``

    Returns
    -------
    dict  final_report with per-stage results + accumulated_best_params
    """
    if stage_plan is None:
        stage_plan = STAGE_PLAN
    if output_dir is None:
        output_dir = Path("reports/stagewise")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_adapter = resolve_strategy_adapter(strategy_mode="native_atdmf")
    accumulated_best: dict[str, Any] = {**PARAM_DEFAULTS}
    if initial_fixed_params:
        accumulated_best.update(initial_fixed_params)

    campaign_start = datetime.now(timezone.utc).isoformat()
    stage_reports: list[dict] = []

    # ── resume: reload previous stage results ───────────────────────────────
    for i in range(resume_from_stage):
        stage_file = output_dir / f"stage_{i+1:02d}_result.json"
        if stage_file.exists():
            with stage_file.open() as f:
                prev = json.load(f)
            accumulated_best.update(prev.get("consensus_best_params", {}))
            stage_reports.append(prev)
            logger.info("Resumed stage %d from %s", i + 1, stage_file)
        else:
            logger.warning("Resume requested but %s not found — starting fresh", stage_file)
            resume_from_stage = i
            break

    # ── run stages ──────────────────────────────────────────────────────────
    for stage_idx, stage in enumerate(stage_plan):
        if stage_idx < resume_from_stage:
            continue

        stage_num = stage_idx + 1
        logger.info("=" * 70)
        logger.info("STAGE %d / %d : %s", stage_num, len(stage_plan), stage["name"])
        logger.info("  Params to optimise : %s", stage["optimize"])
        logger.info("  Method             : %s  (max_trials=%d, patience=%s)",
                    stage["method"], stage["max_trials"], stage["patience"])
        logger.info("=" * 70)

        # Build fixed params = accumulated best + stage forced overrides
        fixed_for_stage = {**accumulated_best, **stage.get("fixed_overrides", {})}

        # Remove stage-optimised params from fixed (they will be searched)
        for key in stage["optimize"]:
            fixed_for_stage.pop(key, None)

        logger.info("  Fixed params (%d) : %s", len(fixed_for_stage),
                    {k: v for k, v in fixed_for_stage.items()
                     if k in stage.get("fixed_overrides", {}) or k not in PARAM_DEFAULTS})

        # Build adapter + settings
        adapter = _FixedParamAdapter(base_adapter, fixed_for_stage)
        settings = _make_wfo_settings(stage, direction, n_windows, train_size)

        param_grid = _build_stage_param_grid(stage)
        metrics_info = {
            "metric1_name":  settings.optimization_metric,
            "metric2_name":  settings.secondary_metric,
            "weight_metric1": settings.metric_weights[0],
            "weight_metric2": settings.metric_weights[1],
        }

        stage_t0 = datetime.now(timezone.utc)
        try:
            wfo_results = walk_forward_optimization(
                df=df,
                param_grid=param_grid,
                metrics_info=metrics_info,
                timeframe=timeframe,
                settings=settings,
                strategy_adapter=adapter,
                status_callback=lambda msg: logger.info("  [wfo] %s", msg),
                control=None,
            )
        except Exception as exc:
            logger.error("Stage %d FAILED: %s", stage_num, exc, exc_info=True)
            stage_report = {
                "stage": stage_num,
                "name": stage["name"],
                "status": "FAILED",
                "error": str(exc),
                "consensus_best_params": {},
            }
            stage_reports.append(stage_report)
            _save_stage(output_dir, stage_num, stage_report)
            if stage_callback is not None:
                try:
                    stage_callback(stage_num, stage["name"], "FAILED", stage_report)
                except Exception:
                    pass
            continue

        elapsed = (datetime.now(timezone.utc) - stage_t0).total_seconds()

        # Extract consensus best params across windows
        consensus = _consensus_params(wfo_results)
        accumulated_best.update(consensus)

        # Summary metrics
        is_perfs  = wfo_results.get("in_sample_performance", [])
        oos_perfs = wfo_results.get("out_of_sample_performance", [])
        is_sharpe  = float(np.nanmean([r.get("sharpe_ratio", np.nan) for r in is_perfs]))  if is_perfs  else None
        oos_sharpe = float(np.nanmean([r.get("sharpe_ratio", np.nan) for r in oos_perfs])) if oos_perfs else None
        is_ret   = float(np.nanmean([r.get("total_return",  np.nan) for r in is_perfs]))   if is_perfs  else None
        oos_ret  = float(np.nanmean([r.get("total_return",  np.nan) for r in oos_perfs]))  if oos_perfs else None

        logger.info("  Stage %d done in %.0fs", stage_num, elapsed)
        logger.info("  IS  Sharpe=%.3f  Return=%.2f%%", is_sharpe or 0, (is_ret or 0) * 100)
        logger.info("  OOS Sharpe=%.3f  Return=%.2f%%", oos_sharpe or 0, (oos_ret or 0) * 100)
        logger.info("  Consensus best params: %s", consensus)

        stage_report = {
            "stage":  stage_num,
            "name":   stage["name"],
            "status": "OK",
            "elapsed_seconds": elapsed,
            "method":     stage["method"],
            "max_trials": stage["max_trials"],
            "patience":   stage["patience"],
            "params_optimised": stage["optimize"],
            "fixed_overrides":  stage.get("fixed_overrides", {}),
            "consensus_best_params": _json_safe(consensus),
            "accumulated_best_params": _json_safe(accumulated_best),
            "is_avg_sharpe":  is_sharpe,
            "oos_avg_sharpe": oos_sharpe,
            "is_avg_return":  is_ret,
            "oos_avg_return": oos_ret,
            "n_windows_with_oos": len(oos_perfs),
        }
        stage_reports.append(stage_report)
        _save_stage(output_dir, stage_num, stage_report)
        if stage_callback is not None:
            try:
                stage_callback(stage_num, stage["name"], "OK", stage_report)
            except Exception:
                pass

    # ── final report ────────────────────────────────────────────────────────
    final_report = {
        "campaign_start": campaign_start,
        "campaign_end":   datetime.now(timezone.utc).isoformat(),
        "timeframe":      timeframe,
        "direction":      direction,
        "n_windows":      n_windows,
        "train_size":     train_size,
        "n_stages":       len(stage_plan),
        "final_best_params": _json_safe(accumulated_best),
        "stages": stage_reports,
    }
    # ── optional headless final backtest ────────────────────────────────────
    if final_df is not None and not final_df.empty:
        logger.info("=" * 70)
        logger.info("FINAL BACKTEST with accumulated best params")
        logger.info("=" * 70)
        fb_result = run_final_backtest_headless(
            best_params=accumulated_best,
            df=final_df,
            timeframe=timeframe,
            direction=direction,
            output_dir=output_dir,
        )
        final_report["final_backtest"] = fb_result

    final_path = output_dir / "stagewise_final_report.json"
    with final_path.open("w") as f:
        json.dump(final_report, f, indent=2, default=str)
    logger.info("Final report saved → %s", final_path)

    return final_report


def _save_stage(output_dir: Path, stage_num: int, report: dict) -> None:
    path = output_dir / f"stage_{stage_num:02d}_result.json"
    with path.open("w") as f:
        json.dump(_json_safe(report), f, indent=2, default=str)
    logger.info("  Stage result saved → %s", path)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  CLI
# ═══════════════════════════════════════════════════════════════════════════════
def _load_csv(path: str, start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="Open time", parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index <= pd.Timestamp(end + " 23:59:59", tz="UTC")]
    logger.info("Loaded %d bars  [%s → %s]", len(df), df.index.min(), df.index.max())
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stagewise WFO campaign — headless, no Streamlit required"
    )
    parser.add_argument(
        "--data-file", required=True,
        help="Path to the OHLCV CSV (5s recommended)",
    )
    parser.add_argument("--timeframe", default="5s", help="Timeframe string (default: 5s)")
    parser.add_argument("--start", default="", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default="", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--direction", default="long_only",
        choices=["long_only", "short_only", "both"],
    )
    parser.add_argument("--n-windows",   type=int,   default=6,    help="WFO windows")
    parser.add_argument("--train-size",  type=float, default=0.75, help="IS fraction")
    parser.add_argument("--output", default="reports/stagewise/", help="Output directory")
    parser.add_argument(
        "--resume-from", type=int, default=0,
        help="Resume from stage N (load previous stage JSON files)",
    )
    parser.add_argument(
        "--initial-params", default="",
        help="Path to a JSON file with initial fixed param values",
    )
    parser.add_argument(
        "--stages", default="",
        help="Comma-separated stage numbers to run (e.g. '1,2,3'). Default: all.",
    )
    parser.add_argument(
        "--final-backtest", action="store_true",
        help="Run a final backtest after stage 7 with the accumulated best params.",
    )
    parser.add_argument(
        "--final-start", default="",
        help="Start date for final backtest (YYYY-MM-DD). Defaults to --start.",
    )
    parser.add_argument(
        "--final-end", default="",
        help="End date for final backtest (YYYY-MM-DD). Defaults to --end.",
    )
    args = parser.parse_args()

    df = _load_csv(args.data_file, args.start, args.end)

    initial_params: dict = {}
    if args.initial_params and Path(args.initial_params).exists():
        with open(args.initial_params) as f:
            initial_params = json.load(f)
        logger.info("Loaded initial params from %s", args.initial_params)

    # Filter stage plan if --stages provided
    plan = STAGE_PLAN
    if args.stages:
        requested = {int(s) for s in args.stages.split(",") if s.strip()}
        plan = [s for i, s in enumerate(STAGE_PLAN, 1) if i in requested]
        logger.info("Running stages: %s", sorted(requested))

    final_df = None
    if args.final_backtest:
        fb_start = args.final_start or args.start
        fb_end   = args.final_end   or args.end
        final_df = _load_csv(args.data_file, fb_start, fb_end)
        logger.info("Final backtest period: %s → %s", fb_start, fb_end)

    run_stagewise_campaign(
        df=df,
        timeframe=args.timeframe,
        direction=args.direction,
        stage_plan=plan,
        n_windows=args.n_windows,
        train_size=args.train_size,
        output_dir=Path(args.output),
        resume_from_stage=args.resume_from,
        initial_fixed_params=initial_params or None,
        final_df=final_df,
    )


if __name__ == "__main__":
    main()
