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
    "timeperiod":              (10,   30,   1),
    "StDev":                   (0.8,   3.0,  0.1),
    # ── BBW compression ──
    "coeff_medianeBBW":        (0.9,   1.5,  0.1),
    "nb_bars_under_bbw_mini":  (1,    10,   2),
    # ── BB% filter ──
    "coef_mediane":            (0.6,   1.3,  0.1),
    "fenetre_lowest":          (20,   200,  10),
    "seuil_lowest":            (1.0,   4.0,  0.1),
    "longueur_mediane":        (50,   200,  10),
    "Nb_bars_above":           (1,     8,   1),
    "nb_bars_entre_bb":        (1,    10,   2),
    # ── SMA exit ──
    "user_exit_sma_length":    (6,    30,   1),
    # ── SAR exit ──
    "sar_start":               (0.02,  0.05, 0.01),
    "sar_increment":           (0.02,  0.05, 0.01),
    "sar_maximum":             (0.1,   0.3,  0.05),
    # ── MACD exit ──
    "macd_fast_length":        (6,    14,   2),
    "macd_slow_length":        (14,   26,   2),
    "macd_signal_length":      (4,    10,   2),
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
# 2.  STAGE PLAN  (11 stages)
#     Each stage defines:
#       name            – human label
#       optimize        – params to search (keys from FULL_PARAM_REGISTRY)
#       fixed_overrides – params to FORCE for this stage (ignore accumulated best)
#       method          – 'bayesian' | 'grid'
#       max_trials      – trial budget (= n_combos for grid; budget for bayesian)
#       patience        – 'Low' | 'Medium' | 'High'
#       stability       – neighbor count for get_stable_best_params
#
#  Sequencing rationale:
#    BB geometry → BBW filter → BB% filter → lowest filter → median →
#    entry count → entry booleans → SMA exit → SAR exit → MACD exit →
#    exit booleans
#  Each group calibrated after upstream dependencies are fixed.
#  No parameter appears in more than one run.
#
#  Combo counts (from reference config ranges):
#    Run 1  : 21×23 =    483  → Bayésien
#    Run 2  :  7× 5 =     35  → Grid exhaustif
#    Run 3  :  8× 5 =     40  → Grid exhaustif
#    Run 4  : 19×31 =    589  → Bayésien
#    Run 5  :      16         → Grid exhaustif
#    Run 6  :       8         → Grid exhaustif
#    Run 7  :  2³  = 8        → Grid exhaustif
#    Run 8  :      25         → Grid exhaustif
#    Run 9  : 2×4×4×5 = 160  → Bayésien
#    Run 10 : 2×4×5×7×4=1120 → Bayésien
#    Run 11 :  2⁴ = 16        → Grid exhaustif
# ═══════════════════════════════════════════════════════════════════════════════
STAGE_PLAN: list[dict] = [
    {
        # timeperiod + StDev jointly define the BB band geometry.
        # All other params depend on this geometry — must be first.
        # 21×23 = 483 combos → Bayésien.
        "name":     "Run 1 — BB core geometry",
        "optimize": ["timeperiod", "StDev"],
        "fixed_overrides": {
            # Force all optional features off so the BB signal is evaluated clean.
            "use_t2_signal":              True,
            "use_roc_filter":             False,
            "use_divergence_bb":          False,
            "exit_sar_enabled":           False,
            "exit_macd_enabled":          False,
            "exit_cross_sar_sma_enabled": False,
            "exit_retour_bb_enabled":     False,
            "exit_regline_enabled":       False,
            "exit_volat_down_enabled":    False,
        },
        "method":     "bayesian",
        "max_trials": 150,      # 21×23 = 483 combos
        "patience":   "Medium",
        "stability":  3,
    },
    {
        # BBW compression threshold and bar-count depend on the BB width
        # established in Run 1. 7×5 = 35 combos → Grid exhaustif.
        "name":     "Run 2 — BBW compression filter",
        "optimize": ["coeff_medianeBBW", "nb_bars_under_bbw_mini"],
        "fixed_overrides": {},
        "method":     "grid",
        "max_trials": 35,       # 7×5 — exhaustive
        "patience":   "Low",
        "stability":  3,
    },
    {
        # coef_mediane (BB% threshold) and nb_bars_entre_bb (price-position
        # filter) are both positional filters relative to the BB bands.
        # 8×5 = 40 combos → Grid exhaustif.
        "name":     "Run 3 — BB% filter + band spacing",
        "optimize": ["coef_mediane", "nb_bars_entre_bb"],
        "fixed_overrides": {},
        "method":     "grid",
        "max_trials": 40,       # 8×5 — exhaustive
        "patience":   "Low",
        "stability":  3,
    },
    {
        # fenetre_lowest and seuil_lowest are tightly coupled (optimal threshold
        # depends on lookback window). 19×31 = 589 combos → Bayésien.
        "name":     "Run 4 — Lowest filter (fenêtre + seuil)",
        "optimize": ["fenetre_lowest", "seuil_lowest"],
        "fixed_overrides": {},
        "method":     "bayesian",
        "max_trials": 150,      # 19×31 = 589 combos
        "patience":   "Medium",
        "stability":  3,
    },
    {
        # longueur_mediane separated from Run 4: only 16 values → Grid exhaustif.
        "name":     "Run 5 — Longueur médiane",
        "optimize": ["longueur_mediane"],
        "fixed_overrides": {},
        "method":     "grid",
        "max_trials": 16,       # 16 values — exhaustive
        "patience":   "Low",
        "stability":  3,
    },
    {
        # Nb_bars_above is a temporal entry filter independent of BB params.
        # 8 values → Grid exhaustif.
        "name":     "Run 6 — Entry bar count",
        "optimize": ["Nb_bars_above"],
        "fixed_overrides": {},
        "method":     "grid",
        "max_trials": 8,        # 8 values — exhaustive
        "patience":   "Low",
        "stability":  3,
    },
    {
        # Boolean entry-feature selection after all entry thresholds are fixed.
        # 2³ = 8 combos → Grid exhaustif.
        "name":     "Run 7 — Entry filter booleans (T2 / RoC / Divergence)",
        "optimize": ["use_t2_signal", "use_roc_filter", "use_divergence_bb"],
        "fixed_overrides": {},
        "method":     "grid",
        "max_trials": 8,        # 2³ — exhaustive
        "patience":   "Low",
        "stability":  3,
    },
    {
        # SMA cross-exit: simple, independent of SAR/MACD; calibrate first
        # once all entry signals are fixed. 25 values → Grid exhaustif.
        "name":     "Run 8 — SMA exit calibration",
        "optimize": ["user_exit_sma_length"],
        "fixed_overrides": {},
        "method":     "grid",
        "max_trials": 25,       # 25 values — exhaustive
        "patience":   "Low",
        "stability":  3,
    },
    {
        # SAR triplet (start/increment/maximum) is internally coupled;
        # include the enable boolean to avoid biasing the calibration.
        # 2×4×4×5 = 160 combos → Bayésien.
        "name":     "Run 9 — SAR exit calibration",
        "optimize": [
            "exit_sar_enabled",
            "sar_start", "sar_increment", "sar_maximum",
        ],
        "fixed_overrides": {},
        "method":     "bayesian",
        "max_trials": 100,      # 2×4×4×5 = 160 combos
        "patience":   "Low",
        "stability":  3,
    },
    {
        # MACD fast/slow/signal are indissociable; type_a/type_b select the
        # signal mode and interact directly with the lengths.
        # 2×2×2×5×7×4 = 1120 combos → Bayésien.
        "name":     "Run 10 — MACD exit calibration",
        "optimize": [
            "exit_macd_enabled",
            "exit_macd_type_a", "exit_macd_type_b",
            "macd_fast_length", "macd_slow_length", "macd_signal_length",
        ],
        "fixed_overrides": {},
        "method":     "bayesian",
        "max_trials": 200,      # 2×2×2×5×7×4 = 1120 combos
        "patience":   "Medium",
        "stability":  5,
    },
    {
        # Final boolean selection for supplementary exits — all thresholds
        # already fixed, pure on/off feature selection.
        # 2⁴ = 16 combos → Grid exhaustif.
        "name":     "Run 11 — Supplementary exit booleans",
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
def _build_stage_param_grid(stage: dict, param_ranges: dict | None = None) -> dict:
    """Return a param_grid dict containing only the params to optimise.

    param_ranges overrides FULL_PARAM_REGISTRY for continuous params:
      {param_name: (min, max, step)}  — sourced from the Strategy panel sidebar.
    Boolean params ([True, False]) always come from FULL_PARAM_REGISTRY.
    """
    grid = {}
    for key in stage["optimize"]:
        if key not in FULL_PARAM_REGISTRY:
            raise KeyError(f"Unknown param '{key}' in stage '{stage['name']}'")
        spec = FULL_PARAM_REGISTRY[key]
        if isinstance(spec, list):
            grid[key] = spec  # boolean — no range override
        elif param_ranges and key in param_ranges:
            grid[key] = param_ranges[key]  # sidebar range
        else:
            grid[key] = spec
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


def propose_stage_settings(param_keys: list[str], param_ranges: dict | None = None) -> dict:
    """Propose WFO engine settings based on the combinatorial complexity of param_keys.

    param_ranges overrides FULL_PARAM_REGISTRY for continuous params:
      {param_name: (min, max, step)}  — sourced from the Strategy panel sidebar.

    Algorithm:
      1. For each param compute its discrete value count:
           list spec   → len(spec)
           tuple spec  → round((max - min) / step) + 1
      2. n_combos = product of all counts
      3. Map n_combos to method / max_trials / patience / stability via thresholds

    Thresholds (tuned against the hardcoded STAGE_PLAN as reference):
      n ≤ 32 (or ≤ 64 when all-boolean)  → Grid exhaustif
      32 < n ≤ 500                        → Bayesian  60 trials  Low   k=3
      500 < n ≤ 5 000                     → Bayesian 100 trials  Low   k=3
      5 000 < n ≤ 100 000                 → Bayesian 150 trials  Medium k=3
      n > 100 000                         → Bayesian 200 trials  Medium k=5

    Returns dict: method, max_trials, patience, stability, n_combos.
    """
    if not param_keys:
        return {"method": "grid", "max_trials": 1, "patience": "Low",
                "stability": 3, "n_combos": 0}

    n_combos = 1
    all_bool = True
    for key in param_keys:
        spec = FULL_PARAM_REGISTRY.get(key)
        if spec is None:
            continue
        if isinstance(spec, list):
            n_combos *= len(spec)
        else:  # (min, max, step) tuple — prefer sidebar range if available
            if param_ranges and key in param_ranges:
                min_v, max_v, step = param_ranges[key]
            else:
                min_v, max_v, step = spec
            n_combos *= max(round((max_v - min_v) / step) + 1, 1)
            all_bool = False

    grid_threshold = 64 if all_bool else 32

    if n_combos <= grid_threshold:
        return {"method": "grid",     "max_trials": n_combos, "patience": "Low",
                "stability": 3, "n_combos": n_combos}
    if n_combos <= 500:
        return {"method": "bayesian", "max_trials": 60,       "patience": "Low",
                "stability": 3, "n_combos": n_combos}
    if n_combos <= 5_000:
        return {"method": "bayesian", "max_trials": 100,      "patience": "Low",
                "stability": 3, "n_combos": n_combos}
    if n_combos <= 100_000:
        return {"method": "bayesian", "max_trials": 150,      "patience": "Medium",
                "stability": 3, "n_combos": n_combos}
    return {"method": "bayesian", "max_trials": 200, "patience": "Medium",
            "stability": 5, "n_combos": n_combos}


def _weighted_median_1d(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median of a 1-D float array."""
    sort_idx = np.argsort(values)
    values  = values[sort_idx]
    weights = weights[sort_idx]
    cumw    = np.cumsum(weights)
    midpoint = cumw[-1] / 2.0
    idx = int(np.searchsorted(cumw, midpoint))
    return float(values[min(idx, len(values) - 1)])


def _best_window_consensus(wfo_results: dict) -> dict:
    """Level 3: best_params from the window with highest avg IS+OOS Sharpe."""
    is_map  = {r.get("window"): r for r in wfo_results.get("in_sample_performance",  [])}
    oos_map = {r.get("window"): r for r in wfo_results.get("out_of_sample_performance", [])}

    best_params: dict = {}
    best_score = float("-inf")

    for wr in wfo_results.get("window_results", []):
        wid = wr.get("window_info", {}).get("window")
        bp  = wr.get("best_params", {})
        if not bp or wid is None:
            continue
        is_s  = float((is_map.get(wid)  or {}).get("sharpe", np.nan))
        oos_s = float((oos_map.get(wid) or {}).get("sharpe", np.nan))
        valid = [s for s in [is_s, oos_s] if np.isfinite(s)]
        if not valid:
            continue
        score = float(np.mean(valid))
        if score > best_score:
            best_score  = score
            best_params = dict(bp)

    return best_params


def _weighted_oos_median_consensus(wfo_results: dict) -> dict:
    """Level 3: weighted median of per-window best_params, weighted by OOS Sharpe."""
    oos_map = {r.get("window"): r for r in wfo_results.get("out_of_sample_performance", [])}

    param_rows: list[dict]  = []
    weights:    list[float] = []

    for wr in wfo_results.get("window_results", []):
        wid = wr.get("window_info", {}).get("window")
        bp  = wr.get("best_params")
        if not bp or wid is None:
            continue
        oos_s = float((oos_map.get(wid) or {}).get("sharpe", np.nan))
        if not np.isfinite(oos_s):
            continue
        param_rows.append(bp)
        weights.append(max(oos_s, 0.0))

    if not param_rows:
        return {}

    w_arr = np.array(weights, dtype=float)
    if w_arr.sum() == 0:
        w_arr = np.ones(len(w_arr))

    all_keys = set().union(*param_rows)
    result: dict = {}
    for key in all_keys:
        v_aln, w_aln = [], []
        for i, r in enumerate(param_rows):
            v = r.get(key)
            if v is not None:
                v_aln.append(v)
                w_aln.append(w_arr[i])
        if not v_aln:
            continue
        wa = np.array(w_aln, dtype=float)
        unique = set(v_aln)
        if unique.issubset({True, False, 0, 1}):
            true_w = sum(ww for v, ww in zip(v_aln, w_aln) if v)
            result[key] = bool(true_w >= wa.sum() / 2)
        else:
            vals = np.array([float(v) for v in v_aln], dtype=float)
            med  = _weighted_median_1d(vals, wa)
            if all(float(v) == int(float(v)) for v in v_aln):
                result[key] = int(round(med))
            else:
                result[key] = med

    return result


def _apply_stage_consensus(wfo_results: dict, method: str) -> dict:
    """Dispatch to the configured Level 3 consensus method."""
    if method == "best_window":
        return _best_window_consensus(wfo_results)
    if method == "weighted_oos_median":
        return _weighted_oos_median_consensus(wfo_results)
    return _consensus_params(wfo_results)   # default: median/mode


def _make_wfo_settings(
    stage: dict,
    direction: str,
    n_windows: int = 6,
    train_size: float = 0.75,
    anchored: bool = False,
    optimization_metric: str = "sharpe_ratio",
    secondary_metric: str = "total_return",
    metric_weights: tuple = (1.0, 0.0),
    parallel_backend: str = "dask",
    neighbor_count: int = 5,
    pqs_n_ref: int = 50,
) -> WFOSettings:
    return WFOSettings(
        n_windows=n_windows,
        train_size=train_size,
        anchored=anchored,
        optimization_method=stage["method"],
        max_trials=stage["max_trials"],
        patience_level=stage["patience"],
        parallel_backend=parallel_backend,
        optimization_metric=optimization_metric,
        secondary_metric=secondary_metric,
        metric_weights=metric_weights,
        strategy_direction=direction,
        neighbor_count=stage.get("stability", neighbor_count),
        pqs_n_ref=pqs_n_ref,
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


def _slim_wfo_results(wfo_results: dict) -> dict:
    """
    Remove heavy per-trial data from a wfo_results dict.
    Keeps window_info, best_params, top-5 optimization_results, and counts.
    Removes optimization_trials (can be hundreds of rows × N windows).
    """
    slimmed = dict(wfo_results)
    slimmed["window_results"] = [
        {k: v for k, v in wr.items() if k != "optimization_trials"}
        for wr in slimmed.get("window_results", [])
    ]
    for key in ("robust_set", "robust_set_summary", "nn_guidance"):
        slimmed.pop(key, None)
    return slimmed


def build_stagewise_wfo_results(final_report: dict) -> dict:
    """
    Build a wfo_results-compatible dict from a stagewise final_report.

    Uses the last successful stage's WFO output (stored as
    ``last_stage_wfo_results``) and reconstructs complete per-window
    best_params by merging the fixed params from that stage with each
    window's stage-specific optimised params.

    This dict is compatible with all existing WFO visualisation, ZIP
    export, and Final Backtest flows.  Returns {} when no WFO data is
    available (all stages failed).
    """
    lswr = final_report.get("last_stage_wfo_results")
    if not lswr:
        return {}

    fixed = final_report.get("last_stage_fixed_params", {})

    # Rebuild per-window complete params = fixed(all prior stages) + last-stage window params
    window_results = []
    best_params_list = []
    for wr in lswr.get("window_results", []):
        partial = wr.get("best_params", {})
        complete = {**fixed, **partial}
        window_results.append({**wr, "best_params": complete})
        best_params_list.append(complete)

    return {
        "window_results":            window_results,
        "in_sample_performance":     lswr.get("in_sample_performance", []),
        "out_of_sample_performance": lswr.get("out_of_sample_performance", []),
        "best_params":               best_params_list,
        "settings":                  lswr.get("settings", {}),
        "timing":                    lswr.get("timing", {}),
        # Stagewise-specific metadata (underscore-prefixed to avoid conflicts)
        "_source":            "stagewise",
        "_n_stages":          final_report.get("n_stages"),
        "_campaign_start":    final_report.get("campaign_start"),
        "_campaign_end":      final_report.get("campaign_end"),
        "_final_best_params": final_report.get("final_best_params", {}),
        "_stages_summary": [
            {
                "stage":          s.get("stage"),
                "name":           s.get("name"),
                "status":         s.get("status"),
                "is_avg_sharpe":  s.get("is_avg_sharpe"),
                "oos_avg_sharpe": s.get("oos_avg_sharpe"),
                "is_avg_return":  s.get("is_avg_return"),
                "oos_avg_return": s.get("oos_avg_return"),
            }
            for s in final_report.get("stages", [])
        ],
    }


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
    anchored: bool = False,
    output_dir: Path | None = None,
    resume_from_stage: int = 0,
    initial_fixed_params: dict | None = None,
    final_df: pd.DataFrame | None = None,
    stage_callback=None,
    data_file_path: str = "",
    optimization_metric: str = "sharpe_ratio",
    secondary_metric: str = "total_return",
    metric_weights: tuple = (1.0, 0.0),
    parallel_backend: str = "dask",
    neighbor_count: int = 5,
    pqs_n_ref: int = 50,
    stage_consensus_method: str = "median_mode",
    param_ranges: dict | None = None,
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
    anchored            : use anchored (fixed-start) WFO windows
    output_dir          : directory for per-stage JSON files + final report
    resume_from_stage   : skip stages 0…N-1 (load their results from output_dir)
    initial_fixed_params: pre-set param values (e.g. loaded from a previous run)
    final_df            : if provided, run a full final backtest after stage 7 and
                          save results to ``final_backtest_result.json``
    optimization_metric : primary metric name (from UI Performance Metrics panel)
    secondary_metric    : secondary metric name
    metric_weights      : (w1, w2) weights for combined score
    parallel_backend    : 'dask' | 'joblib' | 'sequential'
    neighbor_count      : stability neighbor count for get_stable_best_params
    pqs_n_ref           : PQS reference trade count (√(n_trades/n_ref))

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

    # Track last successful stage WFO results for synthetic wfo_results reconstruction
    _last_wfo_results: dict | None = None
    _last_fixed_params: dict = {}

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
        settings = _make_wfo_settings(
            stage, direction, n_windows, train_size,
            anchored=anchored,
            optimization_metric=optimization_metric,
            secondary_metric=secondary_metric,
            metric_weights=metric_weights,
            parallel_backend=parallel_backend,
            neighbor_count=neighbor_count,
            pqs_n_ref=pqs_n_ref,
        )

        param_grid = _build_stage_param_grid(stage, param_ranges=param_ranges)
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

        # Capture last successful stage data (slimmed) for wfo_results reconstruction
        _last_wfo_results = _slim_wfo_results(wfo_results)
        _last_fixed_params = dict(fixed_for_stage)

        # Extract consensus best params across windows (Level 3 method)
        consensus = _apply_stage_consensus(wfo_results, stage_consensus_method)
        accumulated_best.update(consensus)

        # Summary metrics
        is_perfs  = wfo_results.get("in_sample_performance", [])
        oos_perfs = wfo_results.get("out_of_sample_performance", [])
        is_sharpe  = float(np.nanmean([r.get("sharpe", np.nan) for r in is_perfs]))  if is_perfs  else None
        oos_sharpe = float(np.nanmean([r.get("sharpe", np.nan) for r in oos_perfs])) if oos_perfs else None
        is_ret   = float(np.nanmean([r.get("return",  np.nan) for r in is_perfs]))   if is_perfs  else None
        oos_ret  = float(np.nanmean([r.get("return",  np.nan) for r in oos_perfs]))  if oos_perfs else None

        logger.info("  Stage %d done in %.0fs", stage_num, elapsed)
        logger.info("  IS  Sharpe=%.3f  Return=%.2f%%", is_sharpe or 0, is_ret or 0)
        logger.info("  OOS Sharpe=%.3f  Return=%.2f%%", oos_sharpe or 0, oos_ret or 0)
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
        "data_file_path": data_file_path,
        "final_best_params": _json_safe(accumulated_best),
        "stages": stage_reports,
        # Last successful stage WFO data — used by build_stagewise_wfo_results()
        # to produce a standard wfo_results dict for visualization and ZIP export.
        "last_stage_wfo_results": _json_safe(_last_wfo_results) if _last_wfo_results else None,
        "last_stage_fixed_params": _json_safe(_last_fixed_params),
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
def _normalize_date(s: str) -> str:
    """Normalize date strings: replace dots/slashes with dashes in the YYYY?MM?DD part."""
    import re
    return re.sub(r'(\d{4})[.\-/](\d{2})[.\-/](\d{2})', r'\1-\2-\3', s.strip())


def _load_csv(path: str, start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="Open time", parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    if start:
        df = df[df.index >= pd.Timestamp(_normalize_date(start), tz="UTC")]
    if end:
        df = df[df.index <= pd.Timestamp(_normalize_date(end) + " 23:59:59", tz="UTC")]
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
