"""Campaign automation panel for the WFOE Streamlit application.

Provides two input paths that both feed the same execution queue:
  1. Campaign Builder  — select variation axes; configs are auto-generated
  2. JSON File Queue   — upload pre-made config JSON files directly

Both paths add entries to cmp_configs, which is then executed sequentially
via run_optimization_job() with a live comparison dashboard.

Session state keys (prefixed cmp_):
    cmp_configs       list[dict]  [{'filename': str, 'config': dict}, ...]
    cmp_status        str         'idle' | 'running' | 'completed'
    cmp_current_idx   int         index of config currently running
    cmp_results       list[dict]  per-run result records
    cmp_thread        Thread      background runner thread
    cmp_control       CampaignControl

Session state keys (prefixed cmb_ — Campaign Builder):
    cmb_base_config   dict|None   uploaded/snapped base config
    cmb_base_filename str         filename of uploaded base config
"""

from __future__ import annotations

import io
import itertools
import json
import threading
import time
from math import ceil
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

import logging

from config import DEFAULT_PARAM_GRID as _DEFAULT_PARAM_GRID
from data_loading import load_data as _load_data
from domain.serialization import sanitize_for_json as _sanitize_for_json
from services.run_service import run_optimization_job
from wfo import OptimizationInterrupted

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stop-control (mirrors app.py WFOControl without importing from it)
# ---------------------------------------------------------------------------

class CampaignControl:
    """Lightweight stop signal passed to run_optimization_job.

    Uses threading.Event for reliable cross-thread visibility, and raises
    OptimizationInterrupted from wait_if_paused() so the inner optimizer
    loop reacts immediately (same contract as WFOControl in app.py).
    """

    def __init__(self):
        self._stop_event = threading.Event()

    def request_stop(self):
        self._stop_event.set()

    def should_stop(self):
        return self._stop_event.is_set()

    def wait_if_paused(self, log=None):
        if self._stop_event.is_set():
            raise OptimizationInterrupted()


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {"optimization_method", "optimization_regime", "n_windows", "max_trials"}

_DEFAULTS: dict[str, Any] = {
    "cmp_configs": [],
    "cmp_status": "idle",
    "cmp_current_idx": 0,
    "cmp_results": [],
    "cmp_thread": None,
    "cmp_control": None,
    "cmp_job_state": {},    # live state of the currently running backtest
    "cmp_start_time": None, # time.time() when campaign was started
    "cmp_stopping": False,  # True after Stop button clicked, until campaign ends
    # Shared mutable dict — background thread mutates in-place (never reassigns).
    # Holds: status, current_idx, job_state.  Lives alongside cmp_results (shared list).
    "cmp_run_state": None,
}

_BUILDER_DEFAULTS: dict[str, Any] = {
    "cmb_base_config": None,
    "cmb_base_filename": "",
}

# Historical records are intentionally excluded from _DEFAULTS so that
# _reset_campaign() never wipes imported results.
_HISTORICAL_KEY = "cmp_historical_records"


def _init_state():
    for k, v in _DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v
    for k, v in _BUILDER_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if _HISTORICAL_KEY not in st.session_state:
        st.session_state[_HISTORICAL_KEY] = []


def _reset_campaign():
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def _validate_config(config: dict, filename: str) -> list[str]:
    """Return list of validation errors (empty = valid)."""
    errors = []
    for key in _REQUIRED_KEYS:
        if key not in config:
            errors.append(f"Missing required key: `{key}`")
    method = config.get("optimization_method", "")
    if method not in ("bayesian", "optuna", "grid", ""):
        errors.append(f"Unknown optimization_method: `{method}`")
    regime = config.get("optimization_regime", "")
    if regime not in ("classic", "nn_guided", "prev_best_grid", "adaptive_continuous", ""):
        errors.append(f"Unknown optimization_regime: `{regime}`")
    return errors


# ---------------------------------------------------------------------------
# Queue labels
# ---------------------------------------------------------------------------

def _config_label(entry: dict) -> str:
    cfg = entry["config"]
    regime = cfg.get("optimization_regime", "?")
    method = cfg.get("optimization_method", "?")
    anchored = "anchored" if cfg.get("anchored") else "rolling"
    robust = "robust" if cfg.get("robust_tests_enabled") else "no-robust"
    return f"{regime}/{method}/{anchored}/{robust}"


def _short_filename(filename: str) -> str:
    name = filename.replace(".json", "")
    if len(name) > 48:
        name = "..." + name[-45:]
    return name


# ===========================================================================
# Campaign Builder — constants & helpers
# ===========================================================================

# Axis option maps: display label → config value
_METHOD_MAP: dict[str, str] = {
    "Bayesian": "bayesian",
    "Optuna": "optuna",
    "Grid ⚠": "grid",
}
_REGIME_MAP: dict[str, str] = {
    "Classic": "classic",
    "NN-Guided": "nn_guided",
    "Adaptive continuous": "adaptive_continuous",
}
_ANCHOR_MAP: dict[str, bool] = {
    "Rolling": False,
    "Anchored": True,
}
# robust: label → (tag, robust_enabled, use_for_final)
_ROBUST_MAP: dict[str, tuple[bool, bool]] = {
    "Off":              (False, False),
    "Screen only":      (True,  False),
    "+ Final backtest": (True,  True),
}
_METRIC_MAP: dict[str, str] = {
    "Sharpe ratio": "sharpe_ratio",
    "Total return": "total_return",
    "Calmar ratio": "calmar_ratio",
}
_TRAIN_MAP: dict[str, float] = {
    "30%": 0.3,
    "50%": 0.5,
    "70%": 0.7,
}

# Stable config keys — copied from base config (Single Run session or uploaded JSON)
# These are NOT varied by the builder axes.
_STABLE_CONFIG_KEYS = [
    # Data
    "start_date", "end_date", "timeframe", "from_file", "file_path",
    # Strategy
    "strategy_mode", "strategy_id",
    # Pine V3
    "pine_file_path", "pine_compat_mode", "pine_enforce_external_call_contract",
    "pine_enforce_order_semantics", "pine_spec_parser_backend",
    "pine_llm_provider", "pine_llm_model", "pine_llm_base_url",
    "pine_llm_temperature", "pine_llm_max_tokens", "pine_llm_timeout_s",
    "pine_llm_retries", "pine_source_name", "pine_library_paths",
    "pine_library_names", "pine_import_mapping", "pine_generated_module_path",
    "pine_libraries_count", "pine_enforce_parity_gate",
    "pine_parity_trade_count_rel_pct", "pine_parity_entry_count_rel_pct",
    "pine_parity_exit_count_rel_pct", "pine_parity_total_return_abs_pct",
    "pine_parity_max_drawdown_abs_pct", "pine_parity_entry_event_count_rel_pct",
    "pine_parity_exit_event_count_rel_pct", "pine_parity_entry_event_match_min_ratio",
    "pine_parity_exit_event_match_min_ratio", "pine_parity_trade_match_min_ratio",
    "pine_parity_event_time_tolerance_sec", "pine_parity_trade_time_tolerance_sec",
    # Param selection & ranges
    "selected_params",
    "timeperiod_min", "timeperiod_max", "timeperiod_step",
    "StDev_min", "StDev_max", "StDev_step",
    "coeff_medianeBBW_min", "coeff_medianeBBW_max", "coeff_medianeBBW_step",
    "coef_mediane_min", "coef_mediane_max", "coef_mediane_step",
    "fenetre_lowest_min", "fenetre_lowest_max", "fenetre_lowest_step",
    "seuil_lowest_min", "seuil_lowest_max", "seuil_lowest_step",
    "longueur_mediane_min", "longueur_mediane_max", "longueur_mediane_step",
    "Nb_bars_above_min", "Nb_bars_above_max", "Nb_bars_above_step",
    "user_exit_sma_length_min", "user_exit_sma_length_max", "user_exit_sma_length_step",
    "sar_start_min", "sar_start_max", "sar_start_step",
    "sar_increment_min", "sar_increment_max", "sar_increment_step",
    "sar_maximum_min", "sar_maximum_max", "sar_maximum_step",
    "macd_fast_length_min", "macd_fast_length_max", "macd_fast_length_step",
    "macd_slow_length_min", "macd_slow_length_max", "macd_slow_length_step",
    "macd_signal_length_min", "macd_signal_length_max", "macd_signal_length_step",
    # Exit toggles
    "exit_sar_enabled", "exit_macd_enabled", "exit_macd_type_a", "exit_macd_type_b",
    # Order sizing
    "order_sizing_mode", "order_fixed_cash", "fees_pct",
    # WFO core (non-axis)
    "n_windows", "patience_level", "max_trials", "neighbor_count",
    "parallel_backend", "max_workers", "use_numba",
    # Metrics (secondary weight — primary metric is an axis)
    "metric2_name", "weight_metric1", "weight_metric2",
    # Robust (non-axis parts)
    "robust_top_n_per_window", "robust_min_windows",
    # NN guidance
    "nn_min_samples", "nn_candidate_pool_size", "nn_top_k",
    "nn_exploration_ratio", "nn_hidden_size", "nn_epochs",
    "nn_learning_rate", "nn_l2",
    # Adaptive
    "adaptive_train_bars", "adaptive_cycle_bars", "adaptive_trials_per_cycle",
    "adaptive_candidate_pool_size", "adaptive_profile", "adaptive_keep_ratio",
    "adaptive_exploration_ratio", "adaptive_min_values_per_param",
    "adaptive_decay", "adaptive_ucb_beta", "adaptive_warmup_trials",
    "adaptive_max_cycles", "adaptive_oos_weight",
]

# Fallback defaults used when Single Run session is not yet configured
_DEFAULT_BASE_CONFIG: dict[str, Any] = {
    "start_date": "2025-01-01",
    "end_date": "2025-01-30",
    "timeframe": "5s",
    "strategy_mode": "native_atdmf",
    "strategy_id": "atdmf_native_v2",
    "from_file": False,
    "file_path": "",
    "selected_params": [
        "timeperiod", "StDev", "coeff_medianeBBW", "coef_mediane",
        "fenetre_lowest", "seuil_lowest", "longueur_mediane",
        "Nb_bars_above", "user_exit_sma_length",
    ],
    "metric2_name": "total_return",
    "weight_metric1": 1.0,
    "weight_metric2": 0.0,
    "exit_sar_enabled": False,
    "exit_macd_enabled": False,
    "exit_macd_type_a": False,
    "exit_macd_type_b": False,
    "order_sizing_mode": "fixed_cash",
    "order_fixed_cash": 10000.0,
    "fees_pct": 0.0,
    "n_windows": 10,
    "patience_level": "High",
    "max_trials": 2000,
    "neighbor_count": 5,
    "robust_top_n_per_window": 20,
    "robust_min_windows": 3,
    "parallel_backend": "thread",
    "max_workers": 32,
    "use_numba": True,
    "nn_min_samples": 500,
    "nn_candidate_pool_size": 3000,
    "nn_top_k": 250,
    "nn_exploration_ratio": 0.15,
    "nn_hidden_size": 32,
    "nn_epochs": 60,
    "nn_learning_rate": 0.01,
    "nn_l2": 0.0001,
    "adaptive_train_bars": 5000,
    "adaptive_cycle_bars": 5000,
    "adaptive_trials_per_cycle": 150,
    "adaptive_candidate_pool_size": 3000,
    "adaptive_profile": "balanced",
    "adaptive_keep_ratio": 0.4,
    "adaptive_exploration_ratio": 0.2,
    "adaptive_min_values_per_param": 2,
    "adaptive_decay": 0.98,
    "adaptive_ucb_beta": 0.75,
    "adaptive_warmup_trials": 300,
    "adaptive_max_cycles": 0,
    "adaptive_oos_weight": 2.0,
    # Param ranges
    "timeperiod_min": 8.0, "timeperiod_max": 20.0, "timeperiod_step": 1.0,
    "StDev_min": 0.8, "StDev_max": 2.0, "StDev_step": 0.1,
    "coeff_medianeBBW_min": 0.9, "coeff_medianeBBW_max": 1.5, "coeff_medianeBBW_step": 0.05,
    "coef_mediane_min": 0.7, "coef_mediane_max": 1.1, "coef_mediane_step": 0.05,
    "fenetre_lowest_min": 40.0, "fenetre_lowest_max": 200.0, "fenetre_lowest_step": 10.0,
    "seuil_lowest_min": 1.0, "seuil_lowest_max": 3.5, "seuil_lowest_step": 0.1,
    "longueur_mediane_min": 50.0, "longueur_mediane_max": 150.0, "longueur_mediane_step": 5.0,
    "Nb_bars_above_min": 1.0, "Nb_bars_above_max": 6.0, "Nb_bars_above_step": 1.0,
    "user_exit_sma_length_min": 8.0, "user_exit_sma_length_max": 30.0, "user_exit_sma_length_step": 1.0,
    # Pine defaults
    "pine_file_path": "",
    "pine_compat_mode": "strict",
    "pine_enforce_external_call_contract": True,
    "pine_enforce_order_semantics": True,
    "pine_spec_parser_backend": "auto",
    "pine_llm_provider": "openai",
    "pine_llm_model": "gpt-5-mini",
    "pine_llm_base_url": "",
    "pine_llm_temperature": 0.2,
    "pine_llm_max_tokens": 4000,
    "pine_llm_timeout_s": 120,
    "pine_llm_retries": 1,
    "pine_source_name": "",
    "pine_library_paths": [],
    "pine_library_names": [],
    "pine_import_mapping": {},
    "pine_generated_module_path": "",
    "pine_libraries_count": 0,
    "pine_enforce_parity_gate": True,
    "pine_parity_trade_count_rel_pct": 2.0,
    "pine_parity_entry_count_rel_pct": 2.0,
    "pine_parity_exit_count_rel_pct": 2.0,
    "pine_parity_total_return_abs_pct": 3.0,
    "pine_parity_max_drawdown_abs_pct": 3.0,
    "pine_parity_entry_event_count_rel_pct": 5.0,
    "pine_parity_exit_event_count_rel_pct": 5.0,
    "pine_parity_entry_event_match_min_ratio": 0.85,
    "pine_parity_exit_event_match_min_ratio": 0.85,
    "pine_parity_trade_match_min_ratio": 0.8,
    "pine_parity_event_time_tolerance_sec": 5.0,
    "pine_parity_trade_time_tolerance_sec": 5.0,
}


def _estimate_grid_size(base_config: dict) -> int:
    """Estimate number of grid search combinations from parameter ranges."""
    params = base_config.get("selected_params") or []
    total = 1
    for p in params:
        try:
            mn = float(base_config.get(f"{p}_min", 0))
            mx = float(base_config.get(f"{p}_max", 0))
            step = float(base_config.get(f"{p}_step", 1))
            if step <= 0:
                continue
            n_vals = int(ceil((mx - mn) / step)) + 1
            total *= max(n_vals, 1)
        except (TypeError, ValueError):
            continue
    return total


def _read_session_base_config() -> dict:
    """Build base config from Single Run session state, falling back to defaults.

    Key translation notes (app.py widget keys differ from config keys in 3 cases):
    - ``from_file``      → session state ``data_source`` (string "Local File"/"Binance API")
    - ``selected_params``→ assembled from ``check_{param}`` checkbox booleans
    - ``{param}_min/max/step`` → session state ``min/max/step_{param}`` (prefix swapped)
    """
    base = dict(_DEFAULT_BASE_CONFIG)

    for key in _STABLE_CONFIG_KEYS:
        # --- Special case 1: from_file lives as 'data_source' string in session state ---
        if key == "from_file":
            data_source = st.session_state.get("data_source")
            if data_source is not None:
                base["from_file"] = (data_source == "Local File")
            continue

        # --- Special case 2a: pine_libraries_count computed from pine_library_files ---
        # Never stored as its own session key; app.py computes it inline when building cfg.
        if key == "pine_libraries_count":
            lib_files = st.session_state.get("pine_library_files", [])
            base["pine_libraries_count"] = int(len(list(lib_files or [])))
            continue

        # --- Special case 2b: selected_params assembled from per-checkbox session keys ---
        if key == "selected_params":
            if any(f"check_{p}" in st.session_state for p in _DEFAULT_PARAM_GRID):
                base["selected_params"] = [
                    p for p in _DEFAULT_PARAM_GRID
                    if st.session_state.get(f"check_{p}", False)
                ]
            continue

        # --- Special case 3: {param}_min/max/step → min/max/step_{param} in session ---
        if key.endswith("_min"):
            param = key[:-4]  # strip "_min"
            val = st.session_state.get(f"min_{param}")
            if val is not None:
                base[key] = val
            continue
        if key.endswith("_max"):
            param = key[:-4]  # strip "_max"
            val = st.session_state.get(f"max_{param}")
            if val is not None:
                base[key] = val
            continue
        if key.endswith("_step"):
            param = key[:-5]  # strip "_step"
            val = st.session_state.get(f"step_{param}")
            if val is not None:
                base[key] = val
            continue

        # --- Default: config key matches session state key directly ---
        val = st.session_state.get(key)
        if val is not None:
            base[key] = val

    return base


def _combo_to_config(base: dict, combo: dict) -> dict:
    """Apply one axis combo to a base config dict, returning a new config."""
    cfg = dict(base)
    cfg["optimization_method"] = combo["method"]
    cfg["optimization_regime"] = combo["regime"]
    cfg["anchored"] = combo["anchored"]
    cfg["metric1_name"] = combo["metric"]
    cfg["train_size"] = combo["train_size"]

    robust_enabled, robust_final = _ROBUST_MAP[combo["robust"]]
    cfg["robust_tests_enabled"] = robust_enabled
    cfg["robust_use_for_final_backtest"] = robust_final

    # adaptive_continuous is windowless — force n_windows=1
    if combo["regime"] == "adaptive_continuous":
        cfg["n_windows"] = 1

    return cfg


def _combo_label(combo: dict) -> str:
    rob_short = {
        "Off": "no-robust",
        "Screen only": "robust-screen",
        "+ Final backtest": "robust-final",
    }[combo["robust"]]
    anchored_short = "anchored" if combo["anchored"] else "rolling"
    train_short = f"{int(combo['train_size'] * 100)}pct"
    metric_short = combo["metric"].replace("_ratio", "").replace("_return", "ret")
    return f"{combo['regime']}/{combo['method']}/{anchored_short}/{rob_short}/{metric_short}/{train_short}"


def _combo_filename(combo: dict) -> str:
    rob_short = {
        "Off": "no-robust",
        "Screen only": "robust-screen",
        "+ Final backtest": "robust-final",
    }[combo["robust"]]
    anchored_short = "anchored" if combo["anchored"] else "rolling"
    train_short = f"{int(combo['train_size'] * 100)}pct"
    metric_short = combo["metric"].replace("_ratio", "").replace("_return", "ret")
    return (
        f"builder_{combo['regime']}_{combo['method']}_"
        f"{anchored_short}_{rob_short}_{metric_short}_{train_short}.json"
    )


def _add_combos_to_queue(combos: list[dict], base_config: dict, replace: bool = False):
    """Convert axis combos to config entries and add them to cmp_configs."""
    if replace:
        st.session_state["cmp_configs"] = []

    existing_names = {e["filename"] for e in st.session_state["cmp_configs"]}
    added = 0
    for combo in combos:
        filename = _combo_filename(combo)
        if filename in existing_names:
            continue
        config = _combo_to_config(base_config, combo)
        st.session_state["cmp_configs"].append({"filename": filename, "config": config})
        existing_names.add(filename)
        added += 1
    return added


def _show_builder_warnings(combos: list[dict], base_config: dict):
    """Display smart constraint warnings for the current combo set."""
    grid_combos = [c for c in combos if c["method"] == "grid"]
    if grid_combos:
        grid_size = _estimate_grid_size(base_config)
        n_windows = base_config.get("n_windows", 1)
        if grid_size > 1_000_000:
            st.error(
                f"Grid search: estimated **{grid_size:,}** combinations per window "
                f"× {n_windows} windows = **{grid_size * n_windows:,}** total evaluations. "
                "This will be extremely slow or infeasible. Reduce parameter ranges or deselect Grid."
            )
        elif grid_size > 100_000:
            st.warning(
                f"Grid search: estimated **{grid_size:,}** combinations per window. "
                "May take a long time."
            )
        else:
            st.info(f"Grid search: estimated **{grid_size:,}** combinations per window. Feasible.")

    adaptive_combos = [c for c in combos if c["regime"] == "adaptive_continuous"]
    if adaptive_combos:
        st.info(
            "Adaptive continuous: window count is forced to 1 — classic WFO windows are ignored "
            "and replaced by rolling cycles."
        )

    nn_combos = [c for c in combos if c["regime"] == "nn_guided"]
    if nn_combos:
        min_samples = base_config.get("nn_min_samples", 500)
        st.info(
            f"NN-Guided: neural guidance activates after ≥ **{min_samples}** cumulative trials. "
            "Early windows run as classic Bayesian/Optuna."
        )


# ---------------------------------------------------------------------------
# Section: Campaign Builder (axis selector)
# ---------------------------------------------------------------------------

def _render_campaign_builder():
    """Inner content of the Campaign Builder expander."""
    st.caption(
        "Select a base configuration and the dimensions you want to compare. "
        "All combinations are generated automatically and added to the queue."
    )

    # ---- Base config source ----
    st.markdown("**Base configuration**")
    base_source = st.radio(
        "Source",
        options=["Use current Single Run settings", "Upload base JSON file"],
        horizontal=True,
        key="cmb_base_source",
        label_visibility="collapsed",
    )

    base_config: dict | None = None

    if base_source == "Use current Single Run settings":
        base_config = _read_session_base_config()
        # Show a brief summary of what was read
        has_session = any(k in st.session_state for k in ("start_date", "timeframe", "n_windows"))
        if has_session:
            st.caption(
                f"Period: **{base_config.get('start_date')}** → **{base_config.get('end_date')}** | "
                f"Timeframe: **{base_config.get('timeframe')}** | "
                f"Windows: **{base_config.get('n_windows')}** | "
                f"Max trials: **{base_config.get('max_trials')}**"
            )
        else:
            st.caption("Single Run not configured — using default parameters as base.")

    else:
        col_up, col_clear = st.columns([3, 1])
        with col_up:
            uploaded_base = st.file_uploader(
                "Base JSON",
                type=["json"],
                key="cmb_base_upload",
                label_visibility="collapsed",
            )
        with col_clear:
            if st.button(
                "Clear base",
                key="cmb_base_clear",
                disabled=st.session_state.get("cmb_base_config") is None,
                width="stretch",
            ):
                st.session_state["cmb_base_config"] = None
                st.session_state["cmb_base_filename"] = ""
                st.rerun()

        if uploaded_base is not None:
            try:
                st.session_state["cmb_base_config"] = json.load(uploaded_base)
                st.session_state["cmb_base_filename"] = uploaded_base.name
            except Exception as exc:
                st.error(f"Cannot parse `{uploaded_base.name}`: {exc}")

        base_config = st.session_state.get("cmb_base_config")
        base_filename = st.session_state.get("cmb_base_filename", "")

        if base_config is None:
            st.info("Upload a base config JSON file to continue.")
            return
        st.caption(
            f"Using: **{base_filename}** | "
            f"{base_config.get('start_date')} → {base_config.get('end_date')} | "
            f"Windows: {base_config.get('n_windows')} | "
            f"Max trials: {base_config.get('max_trials')}"
        )

    st.divider()

    # ---- Axis selectors ----
    st.markdown("**Variation axes** *(select at least one per row)*")

    col1, col2 = st.columns(2)
    with col1:
        sel_methods = st.multiselect(
            "Optimization method",
            options=list(_METHOD_MAP.keys()),
            default=["Bayesian"],
            key="cmb_sel_methods",
        )
        sel_anchored = st.multiselect(
            "WFO style",
            options=list(_ANCHOR_MAP.keys()),
            default=["Rolling"],
            key="cmb_sel_anchored",
        )
    with col2:
        sel_regimes = st.multiselect(
            "Regime",
            options=list(_REGIME_MAP.keys()),
            default=["Classic"],
            key="cmb_sel_regimes",
        )
        sel_robust = st.multiselect(
            "Robust tests",
            options=list(_ROBUST_MAP.keys()),
            default=["Off"],
            key="cmb_sel_robust",
        )

    with st.expander("Advanced axes — metric & train size", expanded=False):
        col3, col4 = st.columns(2)
        with col3:
            sel_metrics = st.multiselect(
                "Primary metric",
                options=list(_METRIC_MAP.keys()),
                default=["Sharpe ratio"],
                key="cmb_sel_metrics",
            )
        with col4:
            sel_train = st.multiselect(
                "Train size",
                options=list(_TRAIN_MAP.keys()),
                default=["50%"],
                key="cmb_sel_train",
            )

    # Guard: all axes need at least one selection
    all_axes_filled = all([sel_methods, sel_regimes, sel_anchored, sel_robust, sel_metrics, sel_train])
    if not all_axes_filled:
        st.info("Select at least one option per axis to preview combinations.")
        return

    # ---- Generate combos ----
    methods_vals = [_METHOD_MAP[l] for l in sel_methods]
    regimes_vals = [_REGIME_MAP[l] for l in sel_regimes]
    anchored_vals = [_ANCHOR_MAP[l] for l in sel_anchored]
    robust_vals = list(sel_robust)  # keep display labels, _ROBUST_MAP keyed by them
    metrics_vals = [_METRIC_MAP[l] for l in sel_metrics]
    train_vals = [_TRAIN_MAP[l] for l in sel_train]

    all_combos = [
        {
            "method": m,
            "regime": r,
            "anchored": a,
            "robust": ro,
            "metric": mt,
            "train_size": tr,
        }
        for m, r, a, ro, mt, tr in itertools.product(
            methods_vals, regimes_vals, anchored_vals, robust_vals, metrics_vals, train_vals
        )
    ]

    st.divider()

    # ---- Smart warnings ----
    _show_builder_warnings(all_combos, base_config)

    # ---- Preview table ----
    n_total = len(all_combos)
    st.markdown(f"**Preview — {n_total} combination{'s' if n_total != 1 else ''}**")

    if n_total > 30:
        st.warning(
            f"{n_total} combinations will be queued. "
            "Each run can take minutes to hours depending on trial count."
        )

    preview_rows = [
        {
            "#": i + 1,
            "Method": c["method"],
            "Regime": c["regime"],
            "WFO": "Anchored" if c["anchored"] else "Rolling",
            "Robust": c["robust"],
            "Metric": c["metric"].replace("_ratio", "").replace("_return", "ret"),
            "Train": f"{int(c['train_size'] * 100)}%",
        }
        for i, c in enumerate(all_combos)
    ]
    st.dataframe(pd.DataFrame(preview_rows), width="stretch", hide_index=True)

    # Exclusion selector
    excluded_nums = st.multiselect(
        "Exclude combinations (by #)",
        options=[str(i + 1) for i in range(n_total)],
        default=[],
        key="cmb_exclude_nums",
        help="Select combination numbers to remove before adding to the queue.",
    )
    active_combos = [c for i, c in enumerate(all_combos) if str(i + 1) not in excluded_nums]
    n_active = len(active_combos)
    if n_active != n_total:
        st.caption(f"{n_active} of {n_total} combinations will be added.")

    # ---- Action buttons ----
    st.divider()
    is_running = st.session_state["cmp_status"] == "running"
    col_add, col_replace = st.columns(2)
    with col_add:
        if st.button(
            f"Add {n_active} to Queue",
            disabled=not active_combos or is_running,
            width="stretch",
            key="cmb_btn_add",
            help="Append generated configs to the existing queue (skips duplicates).",
        ):
            with st.spinner(f"Generating {n_active} config(s)…"):
                added = _add_combos_to_queue(active_combos, base_config, replace=False)
            st.success(f"Added {added} config(s) to the queue.")
            st.rerun()
    with col_replace:
        if st.button(
            f"Replace Queue ({n_active} combos)",
            disabled=not active_combos or is_running,
            width="stretch",
            key="cmb_btn_replace",
            help="Clear the existing queue and fill it with the generated configs.",
        ):
            with st.spinner(f"Generating {n_active} config(s)…"):
                _add_combos_to_queue(active_combos, base_config, replace=True)
            st.success(f"Queue replaced with {n_active} config(s).")
            st.rerun()


# ---------------------------------------------------------------------------
# Section: JSON File Queue loader
# ---------------------------------------------------------------------------

def _render_json_file_loader():
    """Inner content of the JSON File Queue expander."""
    st.caption(
        "Upload pre-made WFO config JSON files directly. "
        "Useful for custom configurations (different date ranges, Pine strategies, etc.)."
    )

    uploaded = st.file_uploader(
        "Load config JSON files",
        type=["json"],
        accept_multiple_files=True,
        key="cmp_file_uploader",
        help="Upload one or more WFO config JSON files to add to the campaign queue.",
    )

    if uploaded:
        existing_names = {e["filename"] for e in st.session_state["cmp_configs"]}
        added = 0
        for f in uploaded:
            if f.name in existing_names:
                continue
            try:
                config = json.load(f)
            except Exception as exc:
                st.error(f"Cannot parse `{f.name}`: {exc}")
                continue
            errors = _validate_config(config, f.name)
            if errors:
                st.warning(f"`{f.name}` — validation issues: " + "; ".join(errors))
            st.session_state["cmp_configs"].append({"filename": f.name, "config": config})
            added += 1
        if added:
            st.rerun()
    else:
        st.info("Upload JSON files to add them to the queue.")


# ---------------------------------------------------------------------------
# Section: Queue summary
# ---------------------------------------------------------------------------

def _render_queue_summary():
    """Always-visible queue table with remove/clear controls."""
    st.subheader("1 — Queue")

    configs = st.session_state["cmp_configs"]
    if not configs:
        st.info(
            "Queue is empty. Use the **Campaign Builder** or **JSON File Queue** above to add configs."
        )
        return

    rows = []
    for i, entry in enumerate(configs):
        cfg = entry["config"]
        rows.append({
            "#": i + 1,
            "File / Label": _short_filename(entry["filename"]),
            "Regime": cfg.get("optimization_regime", "?"),
            "Method": cfg.get("optimization_method", "?"),
            "Anchored": "✓" if cfg.get("anchored") else "✗",
            "Robust": "✓" if cfg.get("robust_tests_enabled") else "✗",
            "Windows": cfg.get("n_windows", "?"),
            "Trials": cfg.get("max_trials", "?"),
            "Train": f"{int(float(cfg.get('train_size', 0.5)) * 100)}%",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if st.session_state["cmp_status"] == "idle":
        col_idx, col_rm, col_clr = st.columns([2, 1, 1])
        with col_idx:
            rm_idx = st.number_input(
                "Remove entry #",
                min_value=1,
                max_value=len(configs),
                step=1,
                key="cmp_rm_idx",
                label_visibility="visible",
            )
        with col_rm:
            st.write("")  # vertical spacer
            if st.button("Remove", key="cmp_rm_btn", width="stretch"):
                st.session_state["cmp_configs"].pop(int(rm_idx) - 1)
                st.rerun()
        with col_clr:
            st.write("")  # vertical spacer
            if st.button("Clear all", key="cmp_clear_queue", width="stretch"):
                st.session_state["cmp_configs"] = []
                st.rerun()


# ---------------------------------------------------------------------------
# Background runner
# ---------------------------------------------------------------------------

def _slim_results(results: dict | None) -> dict | None:
    """Return a lightweight copy of a results dict for session-state storage.

    Keeps only the keys needed by the campaign UI (comparison table, charts,
    JSON export config reconstruction).  Drops per-trial bulk data
    (all_trials, robust_set, etc.) which can be several MB per run.
    """
    if not isinstance(results, dict):
        return results
    _KEEP = {
        "out_of_sample_performance",
        "in_sample_performance",
        "best_params_per_window",
        "stable_params",
        "settings",
    }
    return {k: v for k, v in results.items() if k in _KEEP}


def _run_campaign_worker(
    configs: list[dict],
    control: CampaignControl,
    run_state: dict,
    shared_results: list,
):
    """Sequential runner executed in a daemon thread.

    CRITICAL — Threading contract:
    This function must NEVER reassign st.session_state keys directly.
    Streamlit silently drops session-state *assignments* from background threads
    (the new value is not visible to the UI on the next script run).

    Instead we rely solely on in-place mutations of Python objects that are
    already stored by reference in session state:
      - run_state  (dict)  — current_idx, status, job_state
      - shared_results (list) — per-run result records

    The UI reads these via the same dict/list references it received at campaign
    start, so mutations are immediately visible on the next fragment rerun.
    """
    n = len(configs)
    _SEP_HEAVY = "━" * 72
    _SEP_LIGHT = "─" * 72

    # Data cache: keyed by (from_file, file_path, start_date, end_date, timeframe).
    # Configs that share the same dataset reuse the already-loaded DataFrame instead
    # of re-reading and re-resampling the full CSV file for every run.
    _df_cache: dict = {}

    logger.info(_SEP_HEAVY)
    logger.info("CAMPAIGN START — %d config(s) queued", n)
    logger.info(_SEP_HEAVY)

    for i, entry in enumerate(configs):
        if control.should_stop():
            for j in range(i, n):
                _append_skipped_in_place(configs[j], shared_results)
            logger.info(_SEP_HEAVY)
            logger.info(
                "CAMPAIGN STOPPED — %d/%d done | ✓%d ✗%d ⏭%d",
                len(shared_results), n,
                sum(1 for r in shared_results if r["status"] == "done"),
                sum(1 for r in shared_results if r["status"] == "error"),
                sum(1 for r in shared_results if r["status"] == "skipped"),
            )
            logger.info(_SEP_HEAVY)
            run_state["current_idx"] = n
            run_state["status"] = "completed"
            return

        run_state["current_idx"] = i

        cfg = entry["config"]
        logger.info(_SEP_LIGHT)
        logger.info(
            "[%d/%d] %s",
            i + 1, n, entry["filename"],
        )
        logger.info(
            "  Method: %s | Regime: %s | Windows: %s | IS: %.0f%% | Trials: %s | Anchored: %s",
            cfg.get("optimization_method", "?"),
            cfg.get("optimization_regime", "?"),
            cfg.get("n_windows", "?"),
            float(cfg.get("train_size", 0.5)) * 100,
            cfg.get("max_trials", "?"),
            "yes" if cfg.get("anchored") else "no",
        )
        logger.info(_SEP_LIGHT)

        # Replace job_state sub-dict for this run.
        # Assigning to a key of run_state is an in-place mutation of the dict
        # (same object stored in session state), so the UI sees the new sub-dict.
        job_state: dict[str, Any] = {
            "status": "running",
            "progress": 0.0,
            "message": "Preparing...",
            "error": None,
        }
        run_state["job_state"] = job_state   # in-place key assignment on shared dict

        # Pre-load data once per unique (from_file, file_path, start, end, timeframe).
        # run_optimization_job accepts an optional df= to skip its own load step.
        _data_key = (
            bool(cfg.get("from_file")),
            str(cfg.get("file_path", "")),
            str(cfg.get("start_date", "")),
            str(cfg.get("end_date", "")),
            str(cfg.get("timeframe", "")),
        )
        preloaded_df = _df_cache.get(_data_key)
        if preloaded_df is None:
            job_state["message"] = "Loading data..."
            logger.info("  Pre-loading data for %s → %s (%s)…",
                        cfg.get("start_date"), cfg.get("end_date"), cfg.get("timeframe"))
            _t0 = time.time()
            try:
                if cfg.get("from_file"):
                    preloaded_df = _load_data(
                        cfg["start_date"], cfg["end_date"], cfg["timeframe"],
                        from_file=True, file_path=cfg["file_path"],
                    )
                else:
                    preloaded_df = _load_data(
                        cfg["start_date"], cfg["end_date"], cfg["timeframe"],
                        from_file=False,
                    )
            except Exception as _exc:
                preloaded_df = None
                job_state["error"] = f"Data load failed: {_exc}"
                logger.error("  Data load failed: %s", _exc)
            if preloaded_df is not None and not preloaded_df.empty:
                _lm, _ls = divmod(int(time.time() - _t0), 60)
                logger.info("  Data cached: %d bars in %dm %02ds", len(preloaded_df), _lm, _ls)
                _df_cache[_data_key] = preloaded_df
            else:
                job_state["error"] = job_state.get("error") or "No data found."
        else:
            logger.info("  Data reused from cache: %d bars", len(preloaded_df))

        if job_state.get("error"):
            results, elapsed = None, None
        else:
            try:
                results, _df, elapsed = run_optimization_job(
                    config=entry["config"],
                    control=control,
                    job_state=job_state,
                    df=preloaded_df,
                )
            except Exception as exc:
                results, elapsed = None, None
                job_state["error"] = str(exc)

        error = job_state.get("error")
        record: dict[str, Any] = {
            "config_name": entry["filename"],
            "config": entry["config"],
            "label": _config_label(entry),
            "results": _slim_results(results),   # drop all_trials/robust_set bulk
            "elapsed": elapsed,
            "error": error,
            "status": "error" if error or results is None else "done",
        }
        shared_results.append(record)   # in-place mutation of shared list

        # Per-run completion log with OOS summary
        _elapsed_str = _format_duration(elapsed) if elapsed is not None else "?"
        if error or results is None:
            logger.info("[%d/%d] ✗ FAILED in %s | %s", i + 1, n, _elapsed_str, str(error)[:120])
        else:
            oos_perfs = results.get("out_of_sample_performance", []) if isinstance(results, dict) else []
            if oos_perfs:
                _sharpe = [w.get("sharpe", float("nan")) for w in oos_perfs]
                _ret    = [w.get("return",  float("nan")) for w in oos_perfs]
                _dd     = [w.get("max_drawdown", float("nan")) for w in oos_perfs]
                _trades = [w.get("n_trades", 0) for w in oos_perfs]
                logger.info(
                    "[%d/%d] ✓ Done in %s | OOS avg: Sharpe=%.3f | Return=%.1f%% "
                    "| DD=%.1f%% | Trades/win=%.0f | n_windows=%d",
                    i + 1, n, _elapsed_str,
                    float(np.nanmean(_sharpe)),
                    float(np.nanmean(_ret)),
                    float(np.nanmean(_dd)),
                    float(np.nanmean(_trades)),
                    len(oos_perfs),
                )
            else:
                logger.info("[%d/%d] ✓ Done in %s | no OOS metrics", i + 1, n, _elapsed_str)

    logger.info(_SEP_HEAVY)
    _n_done  = sum(1 for r in shared_results if r["status"] == "done")
    _n_err   = sum(1 for r in shared_results if r["status"] == "error")
    _n_skip  = sum(1 for r in shared_results if r["status"] == "skipped")
    _total_s = sum(r["elapsed"] for r in shared_results if r.get("elapsed") is not None)
    logger.info(
        "CAMPAIGN COMPLETE — %d/%d configs | ✓%d ✗%d ⏭%d | total time: %s",
        len(shared_results), n, _n_done, _n_err, _n_skip, _format_duration(_total_s),
    )
    logger.info(_SEP_HEAVY)

    run_state["current_idx"] = n
    run_state["status"] = "completed"


def _append_skipped_in_place(entry: dict, shared_results: list) -> None:
    record: dict[str, Any] = {
        "config_name": entry["filename"],
        "config": entry["config"],
        "label": _config_label(entry),
        "results": None,
        "elapsed": None,
        "error": "Skipped (campaign stopped)",
        "status": "skipped",
    }
    shared_results.append(record)   # in-place mutation of shared list


# ---------------------------------------------------------------------------
# Section: Run controls
# ---------------------------------------------------------------------------

def _render_controls():
    st.subheader("2 — Run Campaign")

    status = st.session_state["cmp_status"]
    configs = st.session_state["cmp_configs"]
    n = len(configs)

    col_start, col_stop, col_reset = st.columns(3)

    with col_start:
        start_disabled = status != "idle" or n == 0
        if st.button(
            "▶ Start Campaign",
            disabled=start_disabled,
            type="primary",
            width="stretch",
            key="cmp_btn_start",
        ):
            control = CampaignControl()
            st.session_state["cmp_control"] = control

            # --- Shared mutable objects ---
            # The background thread must ONLY mutate these in-place — it must
            # never reassign st.session_state keys, which Streamlit drops silently.
            shared_results: list = []          # worker appends records here
            run_state: dict[str, Any] = {     # worker updates current_idx, status, job_state
                "status": "running",
                "current_idx": 0,
                "job_state": {},
            }

            # Store shared objects in session state by reference so the UI
            # fragment reads the same objects the worker is mutating.
            st.session_state["cmp_results"] = shared_results
            st.session_state["cmp_run_state"] = run_state
            st.session_state["cmp_current_idx"] = 0
            st.session_state["cmp_status"] = "running"
            st.session_state["cmp_start_time"] = time.time()
            st.session_state["cmp_job_state"] = run_state["job_state"]

            t = threading.Thread(
                target=_run_campaign_worker,
                args=(list(configs), control, run_state, shared_results),
                daemon=True,
            )
            add_script_run_ctx(t, get_script_run_ctx())
            st.session_state["cmp_thread"] = t
            t.start()
            st.rerun()

    with col_stop:
        stop_disabled = status != "running" or st.session_state.get("cmp_stopping", False)
        if st.button(
            "⏹ Stop",
            disabled=stop_disabled,
            width="stretch",
            key="cmp_btn_stop",
        ):
            ctrl = st.session_state.get("cmp_control")
            if ctrl is not None:
                ctrl.request_stop()
            st.session_state["cmp_stopping"] = True
            js = st.session_state.get("cmp_job_state")
            if isinstance(js, dict):
                js["message"] = "Stop requested — finishing current computation…"
            st.rerun()

    with col_reset:
        reset_disabled = status == "running"
        if st.button(
            "↺ Reset",
            disabled=reset_disabled,
            width="stretch",
            key="cmp_btn_reset",
        ):
            _reset_campaign()
            st.rerun()


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _format_duration(seconds: float | None) -> str:
    """Human-readable duration string from a number of seconds."""
    if seconds is None or seconds < 0:
        return "—"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    elif s < 3600:
        m, sec = divmod(s, 60)
        return f"{m}m {sec:02d}s"
    else:
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        return f"{h}h {m:02d}m {sec:02d}s"


def _render_per_config_status(
    configs: list[dict],
    results: list[dict],
    current_idx: int,
    campaign_status: str,
) -> None:
    """Render the per-config status table (compact, no repeated subheader)."""
    result_map = {r["config_name"]: r for r in results}
    icons = {"done": "✅", "error": "❌", "skipped": "⏭", "running": "🔄", "pending": "⏳"}

    rows = []
    for i, entry in enumerate(configs):
        fn = entry["filename"]
        rec = result_map.get(fn)
        if rec:
            cfg_status = rec["status"]
            elapsed = rec.get("elapsed")
            elapsed_str = _format_duration(elapsed)
            err_str = (str(rec.get("error", ""))[:80]) if cfg_status == "error" else ""
        else:
            cfg_status = (
                "running" if (i == current_idx and campaign_status == "running") else "pending"
            )
            elapsed_str = ""
            err_str = ""

        icon = icons.get(cfg_status, "⏳")
        row = {
            " ": icon,
            "Config": _short_filename(fn),
            "Regime": entry["config"].get("optimization_regime", "?"),
            "Method": entry["config"].get("optimization_method", "?"),
            "WFO": "Anchored" if entry["config"].get("anchored") else "Rolling",
            "Robust": "✓" if entry["config"].get("robust_tests_enabled") else "✗",
            "Elapsed": elapsed_str,
        }
        if err_str:
            row["Error"] = err_str
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Section: Live progress
# ---------------------------------------------------------------------------

@st.fragment(run_every=None)
def _render_live_progress():
    """Live progress dashboard — runs as an isolated fragment so the
    polling st.rerun() inside only re-executes this section, not the
    entire campaign panel page.

    State reading strategy:
      - cmp_status      (session state str)  — used for Start/Stop button logic
      - cmp_run_state   (shared dict)        — mutated in-place by worker thread;
                                               gives current_idx, job_state, worker status
      - cmp_results     (shared list)        — appended in-place by worker thread
    """
    cmp_status: str = st.session_state.get("cmp_status", "idle")
    run_state: dict = st.session_state.get("cmp_run_state") or {}
    worker_status: str = run_state.get("status", cmp_status)

    configs = st.session_state["cmp_configs"]
    # cmp_results is the shared list the worker appends records to in-place.
    results = st.session_state.get("cmp_results", [])
    current_idx: int = run_state.get("current_idx", st.session_state.get("cmp_current_idx", 0))
    job_state: dict = run_state.get("job_state") or {}
    start_time: float | None = st.session_state.get("cmp_start_time")

    # Use worker_status for display (more up-to-date than cmp_status which
    # is only updated by the UI polling loop on full-page reruns).
    status = worker_status

    n = len(configs)
    n_done = len(results)

    if status == "idle" and not results:
        return

    st.subheader("3 — Progress")

    # ── Timing helpers ──────────────────────────────────────────────────────
    now = time.time()
    elapsed_campaign = (now - start_time) if start_time else None

    # Average elapsed per completed run → ETA
    done_timed = [r for r in results if r.get("elapsed") is not None]
    eta_campaign: float | None = None
    if done_timed and n_done < n:
        avg_per_run = sum(r["elapsed"] for r in done_timed) / len(done_timed)
        n_remaining = n - n_done
        # Credit what the current run has already burned
        current_run_eta = float(job_state.get("eta_seconds") or 0)
        eta_campaign = current_run_eta + avg_per_run * max(n_remaining - 1, 0)

    # ── Best result so far ───────────────────────────────────────────────────
    best_sharpe = float("nan")
    best_label = ""
    for r in results:
        if r["status"] == "done" and r.get("results"):
            s = _extract_avg_oos(r["results"], "sharpe")
            if s == s and (best_sharpe != best_sharpe or s > best_sharpe):  # NaN-safe max
                best_sharpe = s
                best_label = r.get("label", "")

    # ── Current-run live info ────────────────────────────────────────────────
    is_running = status == "running"
    cur_window = job_state.get("window")
    cur_n_windows = (
        configs[current_idx]["config"].get("n_windows", 1)
        if current_idx < n else 1
    )
    cur_progress = float(job_state.get("progress") or 0.0)
    cur_evals = job_state.get("evaluations")
    cur_speed = job_state.get("speed")
    cur_msg = str(job_state.get("message") or "")

    # ── Card panel ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            label="Campaign",
            value=f"{n_done} / {n}",
            delta=f"{n_done / max(n, 1) * 100:.0f}% complete",
            delta_color="off",
        )

    with c2:
        if is_running and cur_window is not None:
            win_val = f"{cur_window} / {cur_n_windows}"
            win_delta = cur_msg[:48] if cur_msg else None
        elif status == "completed":
            win_val = "Done"
            win_delta = None
        else:
            win_val = "—"
            win_delta = None
        st.metric(label="Window", value=win_val, delta=win_delta, delta_color="off")

    with c3:
        if is_running and cur_evals is not None:
            evals_val = f"{cur_evals:,}"
            evals_delta = f"{cur_speed:.1f} eval/s" if cur_speed else None
        else:
            evals_val = "—"
            evals_delta = None
        st.metric(label="Evaluations", value=evals_val, delta=evals_delta, delta_color="off")

    with c4:
        elapsed_str = _format_duration(elapsed_campaign)
        eta_str = _format_duration(eta_campaign) if eta_campaign is not None else "—"
        st.metric(
            label="Elapsed",
            value=elapsed_str,
            delta=f"ETA ≈ {eta_str}" if is_running and eta_campaign is not None else None,
            delta_color="off",
        )

    # Second row of cards: best result + status breakdown
    c5, c6 = st.columns(2)

    with c5:
        sharpe_str = f"{best_sharpe:.4f}" if best_sharpe == best_sharpe else "—"
        st.metric(
            label="Best OOS Sharpe so far",
            value=sharpe_str,
            delta=_short_filename(best_label) if best_label else None,
            delta_color="off",
        )

    with c6:
        n_ok = sum(1 for r in results if r["status"] == "done")
        n_err = sum(1 for r in results if r["status"] == "error")
        n_skip = sum(1 for r in results if r["status"] == "skipped")
        n_pending = n - n_done
        status_val = f"✅ {n_ok}  ❌ {n_err}  ⏭ {n_skip}  ⏳ {n_pending}"
        st.metric(label="Status breakdown", value=status_val, delta=None)

    # ── Dual progress bars ──────────────────────────────────────────────────
    st.divider()

    campaign_frac = n_done / max(n, 1)
    st.progress(
        campaign_frac,
        text=f"Campaign: {n_done} / {n} configs  ({campaign_frac * 100:.0f}%)",
    )

    if is_running and current_idx < n:
        cur_entry = configs[current_idx]
        cur_label = _config_label(cur_entry)
        win_text = f" · Window {cur_window}/{cur_n_windows}" if cur_window else ""
        eval_text = f" · {cur_evals:,} evals" if cur_evals else ""
        st.progress(
            min(max(cur_progress, 0.0), 1.0),
            text=f"Current run: {cur_label}{win_text}{eval_text}",
        )

    # ── Per-config status table ─────────────────────────────────────────────
    st.divider()
    _render_per_config_status(configs, results, current_idx, status)

    if status == "completed":
        total_elapsed = _format_duration(elapsed_campaign)
        st.success(
            f"Campaign completed in **{total_elapsed}** — "
            f"✅ {n_ok} done | ❌ {n_err} errors | ⏭ {n_skip} skipped"
        )

    # Auto-refresh while running (0.5 s polling) — MUST be last so all
    # content above is rendered before st.rerun() aborts the script run.
    #
    # We check cmp_status (session state) rather than status/worker_status:
    #   - cmp_status == "running" means the UI launched a campaign and hasn't yet
    #     done the completion full-page rerun.
    #   - worker_status (from run_state) may already be "completed" while
    #     cmp_status is still "running" — that's exactly when we trigger the sync.
    if cmp_status == "running":
        if worker_status == "completed":
            # Worker finished cleanly.  Sync persistent session state keys and
            # trigger a full-page rerun so the comparison table renders.
            st.session_state["cmp_status"] = "completed"
            st.session_state["cmp_stopping"] = False
            st.session_state["cmp_current_idx"] = run_state.get("current_idx", n)
            st.rerun(scope="app")

        thread = st.session_state.get("cmp_thread")
        if thread is not None and not thread.is_alive():
            # Thread died unexpectedly (unhandled exception).  Same sync + full rerun.
            st.session_state["cmp_status"] = "completed"
            st.session_state["cmp_stopping"] = False
            st.session_state["cmp_current_idx"] = run_state.get("current_idx", n)
            st.rerun(scope="app")

        time.sleep(1.0)
        st.rerun()   # fragment-only rerun for live progress update


# ---------------------------------------------------------------------------
# Section: Comparison table
# ---------------------------------------------------------------------------

def _extract_avg_oos(results_dict: dict | None, metric: str, default=float("nan")) -> float:
    if not isinstance(results_dict, dict):
        return default
    oos = results_dict.get("out_of_sample_performance", [])
    if not oos:
        return default
    try:
        vals = [float(w.get(metric, float("nan"))) for w in oos]
        vals = [v for v in vals if not (v != v)]  # drop NaN
        return float(np.mean(vals)) if vals else default
    except Exception:
        return default


def _build_comparison_df(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        cfg = rec.get("config", {})
        res = rec.get("results")
        csv_m = rec.get("_csv_metrics") or {}
        st_ = rec.get("status", "?")
        elapsed = rec.get("elapsed")

        def _get(oos_key: str, csv_key: str) -> float:
            if not res and csv_key in csv_m:
                try:
                    return float(csv_m[csv_key])
                except (TypeError, ValueError):
                    pass
            return _extract_avg_oos(res, oos_key)

        rows.append({
            "Config": _short_filename(rec["config_name"]),
            "Regime": cfg.get("optimization_regime", "?"),
            "Method": cfg.get("optimization_method", "?"),
            "Anchored": "✓" if cfg.get("anchored") else "✗",
            "Robust": "✓" if cfg.get("robust_tests_enabled") else "✗",
            "Avg OOS Sharpe": _get("sharpe", "sharpe"),
            "Avg OOS Return %": _get("return", "return"),
            "Avg OOS Drawdown %": _get("max_drawdown", "drawdown"),
            "Avg Win Rate %": _get("win_rate", "win_rate"),
            "Avg Trades/Win": _get("n_trades", "n_trades"),
            "Runtime (s)": round(elapsed, 1) if elapsed is not None else float("nan"),
            "Status": "✅" if st_ == "done" else ("❌" if st_ == "error" else "⏭"),
        })
    return pd.DataFrame(rows)


def _render_comparison_table(records: list[dict]):
    done_records = [r for r in records if r["status"] == "done"]
    if not done_records:
        return

    st.subheader("4 — Comparison Table")

    df = _build_comparison_df(records)
    # Columns where higher is better (highlight max) vs lower is better (highlight min)
    _higher_is_better = {"Avg OOS Sharpe", "Avg OOS Return %", "Avg Win Rate %", "Avg Trades/Win"}
    _lower_is_better  = {"Avg OOS Drawdown %", "Runtime (s)"}
    numeric_cols = list(_higher_is_better | _lower_is_better)

    def _highlight_best(s: pd.Series) -> list[str]:
        if s.name not in numeric_cols:
            return [""] * len(s)
        vals = pd.to_numeric(s, errors="coerce")
        if vals.isna().all():
            return [""] * len(s)
        best_idx = vals.idxmin() if s.name in _lower_is_better else vals.idxmax()
        return [
            "background-color: #d4edda; font-weight: bold" if i == best_idx else ""
            for i in s.index
        ]

    styled = df.style.apply(_highlight_best, axis=0)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    err_records = [r for r in records if r["status"] == "error"]
    if err_records:
        with st.expander(f"❌ {len(err_records)} error(s)", expanded=False):
            for r in err_records:
                st.error(f"**{_short_filename(r['config_name'])}**: {r.get('error', 'unknown error')}")


# ---------------------------------------------------------------------------
# Section: Per-window charts
# ---------------------------------------------------------------------------

def _render_window_charts(records: list[dict]):
    done_records = [r for r in records if r["status"] == "done" and r.get("results")]
    if len(done_records) < 2:
        return

    st.subheader("5 — Per-Window Charts")
    chart_tab1, chart_tab2 = st.tabs(["OOS Sharpe", "OOS Return %"])

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf",
    ]

    def _build_fig(metric: str, y_label: str) -> go.Figure:
        fig = go.Figure()
        for idx, rec in enumerate(done_records):
            oos = rec["results"].get("out_of_sample_performance", [])
            if not oos:
                continue
            windows = [w.get("window", i + 1) for i, w in enumerate(oos)]
            values = [w.get(metric, None) for w in oos]
            fig.add_trace(go.Scatter(
                x=windows,
                y=values,
                mode="lines+markers",
                name=rec["label"],
                line=dict(color=colors[idx % len(colors)], width=2),
                marker=dict(size=6),
            ))
        fig.update_layout(
            xaxis_title="Window",
            yaxis_title=y_label,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            height=400,
            margin=dict(l=40, r=20, t=60, b=40),
        )
        return fig

    with chart_tab1:
        st.plotly_chart(_build_fig("sharpe", "OOS Sharpe Ratio"), use_container_width=True)
    with chart_tab2:
        st.plotly_chart(_build_fig("return", "OOS Return (%)"), use_container_width=True)


# ---------------------------------------------------------------------------
# Section: Export
# ---------------------------------------------------------------------------

def _render_export(records: list[dict]):
    done_records = [r for r in records if r["status"] == "done"]
    if not done_records:
        return

    st.subheader("6 — Export")
    col_csv, col_json = st.columns(2)

    with col_csv:
        df = _build_comparison_df(records)
        st.download_button(
            "⬇ Download CSV (comparison table)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="campaign_comparison.csv",
            mime="text/csv",
            width="stretch",
            key="cmp_dl_csv",
        )

    with col_json:
        export_payload = [
            {
                "config_name": rec["config_name"],
                "label": rec["label"],
                "status": rec["status"],
                "elapsed": rec.get("elapsed"),
                "error": rec.get("error"),
                "results": _sanitize_for_json(rec.get("results")) if rec.get("results") else None,
            }
            for rec in records
        ]
        json_bytes = json.dumps(export_payload, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button(
            "⬇ Download JSON (full results)",
            data=json_bytes,
            file_name="campaign_results.json",
            mime="application/json",
            width="stretch",
            key="cmp_dl_json",
        )


# ---------------------------------------------------------------------------
# Section: Import historical results
# ---------------------------------------------------------------------------

def _reconstruct_config_from_record(rec: dict) -> dict:
    """Rebuild a minimal config dict from an imported JSON record."""
    results = rec.get("results") or {}
    settings = results.get("settings") or {}
    label = rec.get("label", "")
    parts = label.split("/") if label else []

    regime = settings.get("optimization_regime") or (parts[0] if len(parts) > 0 else "?")
    method = settings.get("optimization_method") or (parts[1] if len(parts) > 1 else "?")
    anchored = settings.get("anchored", "anchored" in label)
    robust = "robust" in label and "no-robust" not in label
    return {
        "optimization_regime": regime,
        "optimization_method": method,
        "anchored": anchored,
        "robust_tests_enabled": robust,
    }


def _parse_json_records(data: list) -> tuple[list[dict], list[str]]:
    """Parse a campaign_results.json payload into records + warnings."""
    records: list[dict] = []
    warnings: list[str] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            warnings.append(f"Item {i}: not a dict, skipped.")
            continue
        rec: dict[str, Any] = {
            "config_name": item.get("config_name", f"run_{i}"),
            "label": item.get("label", ""),
            "status": item.get("status", "done"),
            "elapsed": item.get("elapsed"),
            "error": item.get("error"),
            "results": item.get("results"),
        }
        rec["config"] = _reconstruct_config_from_record(rec)
        records.append(rec)
    return records, warnings


def _parse_csv_records(df: pd.DataFrame) -> tuple[list[dict], list[str]]:
    """Parse a campaign_comparison.csv into minimal records (no per-window data)."""
    records: list[dict] = []
    warnings: list[str] = []
    required = {"Config", "Regime", "Method"}
    missing = required - set(df.columns)
    if missing:
        warnings.append(f"CSV missing expected columns: {', '.join(missing)}")
        return records, warnings

    for _, row in df.iterrows():
        anchored = str(row.get("Anchored", "✗")) == "✓"
        robust = str(row.get("Robust", "✗")) == "✓"
        regime = str(row.get("Regime", "?"))
        method = str(row.get("Method", "?"))
        label = f"{regime}/{method}/{'anchored' if anchored else 'rolling'}/{'robust' if robust else 'no-robust'}"
        raw_status = str(row.get("Status", ""))
        status = "done" if raw_status in ("✅", "done", "ok") else "error"

        def _safe_float(val) -> float | None:
            try:
                v = float(val)
                return None if (v != v) else v
            except (TypeError, ValueError):
                return None

        rec: dict[str, Any] = {
            "config_name": str(row.get("Config", f"row_{_}")),
            "label": label,
            "status": status,
            "elapsed": _safe_float(row.get("Runtime (s)")),
            "error": None,
            "results": None,
            "config": {
                "optimization_regime": regime,
                "optimization_method": method,
                "anchored": anchored,
                "robust_tests_enabled": robust,
            },
            "_csv_metrics": {
                "sharpe": _safe_float(row.get("Avg OOS Sharpe")),
                "return": _safe_float(row.get("Avg OOS Return %")),
                "drawdown": _safe_float(row.get("Avg OOS Drawdown %")),
                "win_rate": _safe_float(row.get("Avg Win Rate %")),
                "n_trades": _safe_float(row.get("Avg Trades/Win")),
            },
        }
        records.append(rec)
    return records, warnings


def _render_comparison_table_keyed(records: list[dict], key_suffix: str = ""):
    done_records = [r for r in records if r["status"] == "done"]
    if not done_records:
        st.info("No completed runs to display.")
        return

    df = _build_comparison_df(records)
    # Columns where higher is better (highlight max) vs lower is better (highlight min)
    _higher_is_better = {"Avg OOS Sharpe", "Avg OOS Return %", "Avg Win Rate %", "Avg Trades/Win"}
    _lower_is_better  = {"Avg OOS Drawdown %", "Runtime (s)"}
    numeric_cols = list(_higher_is_better | _lower_is_better)

    def _highlight_best(s: pd.Series) -> list[str]:
        if s.name not in numeric_cols:
            return [""] * len(s)
        vals = pd.to_numeric(s, errors="coerce")
        if vals.isna().all():
            return [""] * len(s)
        best_idx = vals.idxmin() if s.name in _lower_is_better else vals.idxmax()
        return [
            "background-color: #d4edda; font-weight: bold" if i == best_idx else ""
            for i in s.index
        ]

    styled = df.style.apply(_highlight_best, axis=0)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    err_records = [r for r in records if r["status"] == "error"]
    if err_records:
        with st.expander(f"❌ {len(err_records)} error(s)", expanded=False):
            for r in err_records:
                st.error(f"**{_short_filename(r['config_name'])}**: {r.get('error', 'unknown error')}")


def _render_window_charts_keyed(records: list[dict], key_suffix: str = ""):
    done_records = [r for r in records if r["status"] == "done" and r.get("results")]
    if len(done_records) < 2:
        if any(r.get("_csv_metrics") for r in records):
            st.caption("Per-window charts unavailable — CSV import contains summary metrics only.")
        return

    chart_tab1, chart_tab2 = st.tabs(["OOS Sharpe", "OOS Return %"])
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf",
    ]

    def _build_fig(metric: str, y_label: str) -> go.Figure:
        fig = go.Figure()
        for idx, rec in enumerate(done_records):
            oos = rec["results"].get("out_of_sample_performance", [])
            if not oos:
                continue
            windows = [w.get("window", i + 1) for i, w in enumerate(oos)]
            values = [w.get(metric, None) for w in oos]
            fig.add_trace(go.Scatter(
                x=windows, y=values, mode="lines+markers",
                name=rec["label"],
                line=dict(color=colors[idx % len(colors)], width=2),
                marker=dict(size=6),
            ))
        fig.update_layout(
            xaxis_title="Window", yaxis_title=y_label,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            height=400, margin=dict(l=40, r=20, t=60, b=40),
        )
        return fig

    with chart_tab1:
        st.plotly_chart(_build_fig("sharpe", "OOS Sharpe Ratio"), use_container_width=True)
    with chart_tab2:
        st.plotly_chart(_build_fig("return", "OOS Return (%)"), use_container_width=True)


def _render_export_keyed(records: list[dict], key_suffix: str = ""):
    done_records = [r for r in records if r["status"] == "done"]
    if not done_records:
        return

    col_csv, col_json = st.columns(2)
    with col_csv:
        df = _build_comparison_df(records)
        st.download_button(
            "⬇ Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"campaign_comparison_{key_suffix}.csv",
            mime="text/csv",
            width="stretch",
            key=f"cmp_dl_csv_{key_suffix}",
        )
    with col_json:
        payload = [
            {
                "config_name": r["config_name"],
                "label": r["label"],
                "status": r["status"],
                "elapsed": r.get("elapsed"),
                "error": r.get("error"),
                "results": _sanitize_for_json(r.get("results")) if r.get("results") else None,
            }
            for r in records
        ]
        st.download_button(
            "⬇ Download JSON",
            data=json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name=f"campaign_results_{key_suffix}.json",
            mime="application/json",
            width="stretch",
            key=f"cmp_dl_json_{key_suffix}",
        )


def _render_import_historical():
    """Render the historical results import section (collapsible)."""
    hist_records: list[dict] = st.session_state.get(_HISTORICAL_KEY, [])

    label = (
        f"📂 Load Historical Results ({len(hist_records)} loaded)"
        if hist_records
        else "📂 Load Historical Results"
    )
    with st.expander(label, expanded=bool(hist_records)):
        st.caption(
            "Import a previously exported `campaign_results.json` (full results + charts) "
            "or `campaign_comparison.csv` (table only)."
        )

        col_up, col_clear = st.columns([3, 1])
        with col_up:
            uploaded = st.file_uploader(
                "Select file",
                type=["json", "csv"],
                key="cmph_import_file",
                label_visibility="collapsed",
            )
        with col_clear:
            if st.button(
                "Clear",
                key="cmph_btn_clear",
                disabled=not hist_records,
                width="stretch",
            ):
                st.session_state[_HISTORICAL_KEY] = []
                st.rerun()

        if uploaded is not None:
            try:
                if uploaded.name.lower().endswith(".json"):
                    data = json.load(uploaded)
                    if not isinstance(data, list):
                        st.error("Expected a JSON array (`campaign_results.json` format).")
                    else:
                        records, warns = _parse_json_records(data)
                        for w in warns:
                            st.warning(w)
                        st.session_state[_HISTORICAL_KEY] = records
                        st.success(f"Loaded {len(records)} records from JSON — full results available.")
                        st.rerun()

                elif uploaded.name.lower().endswith(".csv"):
                    df_csv = pd.read_csv(uploaded)
                    records, warns = _parse_csv_records(df_csv)
                    for w in warns:
                        st.warning(w)
                    st.session_state[_HISTORICAL_KEY] = records
                    st.info(
                        f"Loaded {len(records)} records from CSV. "
                        "Per-window charts are not available (CSV contains summary metrics only)."
                    )
                    st.rerun()

            except Exception as exc:
                st.error(f"Failed to parse `{uploaded.name}`: {exc}")

        if not hist_records:
            return

        st.divider()
        _render_comparison_table_keyed(hist_records, key_suffix="hist")
        st.divider()
        _render_window_charts_keyed(hist_records, key_suffix="hist")
        st.divider()
        _render_export_keyed(hist_records, key_suffix="hist")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def render_campaign_panel():
    """Render the full campaign automation panel in the main content area."""
    _init_state()

    st.title("🗂 Campaign — Batch Backtest Comparison")
    st.caption(
        "Build a campaign using the **Campaign Builder** (auto-generate configs from axes) "
        "or upload pre-made **JSON files** directly. Both input methods feed the same queue."
    )
    st.divider()

    # ── Input sources (two expanders feeding the same cmp_configs queue) ──
    with st.expander("🛠 Campaign Builder", expanded=True):
        _render_campaign_builder()

    with st.expander("📂 JSON File Queue"):
        _render_json_file_loader()

    st.divider()

    # ── Queue summary (always visible) ──
    _render_queue_summary()
    st.divider()

    # ── Run controls ──
    _render_controls()

    # ── Progress + results ──
    records = st.session_state.get("cmp_results", [])
    status = st.session_state.get("cmp_status", "idle")

    if status != "idle" or records:
        st.divider()
        _render_live_progress()

    if records:
        st.divider()
        _render_comparison_table(records)
        st.divider()
        _render_window_charts(records)
        st.divider()
        _render_export(records)

    st.divider()
    _render_import_historical()
