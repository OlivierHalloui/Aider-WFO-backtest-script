import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import json
import datetime
import io
import zipfile
import threading
import hashlib
import html
import tempfile
import re
import importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Plotly expects np.bool8 on older releases; alias for numpy>=2.0 compatibility.
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import from existing modules
from config import (
    DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_TIMEFRAME, DEFAULT_DATA_FILE, DEFAULT_PARAM_GRID,
    DEFAULT_STRATEGY_MODE, DEFAULT_STRATEGY_ID,
    WFOSettings
)
from wfo import OptimizationInterrupted
from data_loading import load_data, get_csv_date_range
from expert import (
    LLMConfig,
    ExpertRunContext,
    ExpertInputData,
    ExpertRequest,
    OpenAICompatibleGateway,
    ExpertPromptBuilder,
    ExpertAnalyzer,
    ExpertStorage,
    ExpertService,
)
from domain.serialization import (
    sanitize_for_json as _sanitize_for_json,
    utc_now_iso as _utc_now_iso,
    sha256_json as _sha256_json,
    to_jsonable as _to_jsonable,
    safe_float_scalar as _safe_float_scalar,
)
from ui.data_utils import (
    downsample_series as _downsample_series,
    downsample_df as _downsample_df,
    get_return_series as _get_return_series,
    compute_trade_pnl_metrics as _compute_trade_pnl_metrics,
)
from ui.expert_report import (
    render_deterministic_alerts as _render_deterministic_alerts,
    render_interpretation_guide as _render_interpretation_guide,
    render_expert_human_report as _render_expert_human_report,
)
from services.traceability import (
    build_traceability_payload as _build_traceability_payload,
)
from services.export_utils import (
    build_data_source_descriptor as _build_data_source_descriptor_base,
    build_replay_manifest as _build_replay_manifest_base,
)
from services.runtime_utils import (
    timeframe_to_seconds as _timeframe_to_seconds,
    parse_iso_date as _parse_iso_date,
    humanize_seconds as _humanize_seconds,
    compute_running_elapsed_seconds as _compute_running_elapsed_seconds,
)
from services.run_service import run_optimization_job
from strategy_adapters import resolve_strategy_adapter
from pine_v3 import (
    build_strategy_spec_v1_from_pine_text as _build_strategy_spec_v1_from_pine_text,
    validate_strategy_spec_v1 as _validate_strategy_spec_v1,
    generate_strategy_module_from_spec as _generate_strategy_module_from_spec,
)
from pine_v3.parity import (
    build_parity_report as _build_parity_report,
    build_parity_reference_payload as _build_parity_reference_payload,
    validate_parity_reference_payload as _validate_parity_reference_payload,
    DEFAULT_PARITY_THRESHOLDS as _DEFAULT_PARITY_THRESHOLDS,
    DEFAULT_PARITY_DETAIL_THRESHOLDS as _DEFAULT_PARITY_DETAIL_THRESHOLDS,
    PARITY_REFERENCE_SCHEMA_VERSION as _PARITY_REFERENCE_SCHEMA_VERSION,
    normalize_metrics as _normalize_parity_metrics,
)
from pine_v3.execution_gate import (
    build_execution_gate_report as _build_pine_execution_gate_report,
)
from pine_v3.mtf_parity import (
    build_mtf_parity_proof_report as _build_mtf_parity_proof_report,
)
from pine_v3.runtime_adapter import (
    build_request_security_diagnostics as _build_request_security_diagnostics,
)

# Set page config
st.set_page_config(
    page_title="ATDMF Strategy Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and Description
st.title("📈 ATDMF Strategy Walk-Forward Optimizer")
st.markdown("""
This dashboard performs **Walk-Forward Optimization (WFO)** on the ATDMF strategy using **VectorBT Pro**.
Configure your data, strategy parameters, and optimization settings in the sidebar to begin.
""")

# Global UX: widen main content area and keep sidebar readable.
st.markdown(
    """
    <style>
    /* Expand central area to use full available width next to the sidebar. */
    div[data-testid="stAppViewContainer"] > section.main > div.block-container {
        max-width: none !important;
        width: 100% !important;
        padding-left: 0.45rem !important;
        padding-right: 0.45rem !important;
    }
    /* Compact tab labels and allow wrapping so all tabs remain visible. */
    div[data-testid="stTabs"] button[role="tab"] {
        padding: 0.22rem 0.42rem !important;
        font-size: 0.78rem !important;
        white-space: nowrap;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0.06rem;
        flex-wrap: wrap;
    }
    /* Hide +/- steppers in sidebar number inputs for cleaner range editing. */
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button {
        display: none !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stExpander"] details {
        border: 1px solid rgba(112, 141, 173, 0.55);
        border-radius: 10px;
        overflow: hidden;
        background: rgba(16, 25, 39, 0.55);
        margin-bottom: 8px;
    }
    section[data-testid="stSidebar"] div[data-testid="stExpander"] summary {
        background: linear-gradient(135deg, rgba(33, 52, 76, 0.96), rgba(24, 39, 59, 0.96));
        border-bottom: 1px solid rgba(133, 171, 208, 0.30);
        padding-top: 0.28rem;
        padding-bottom: 0.28rem;
    }
    section[data-testid="stSidebar"] div[data-testid="stExpander"] summary:hover {
        background: linear-gradient(135deg, rgba(44, 68, 98, 0.98), rgba(30, 50, 75, 0.98));
    }
    section[data-testid="stSidebar"] div[data-testid="stExpander"] details[open] summary {
        background: linear-gradient(135deg, rgba(52, 79, 113, 0.98), rgba(35, 57, 86, 0.98));
        border-bottom: 1px solid rgba(163, 201, 236, 0.42);
    }
    section[data-testid="stSidebar"] div[data-testid="stExpander"] div[role="region"] {
        background: rgba(12, 20, 31, 0.72);
        padding-top: 0.40rem;
        padding-bottom: 0.25rem;
    }
    /* Make the Expert run button more visible. */
    div.st-key-expert_generate_btn button {
        background: linear-gradient(135deg, #0f766e, #0ea5e9) !important;
        color: #f8fbff !important;
        border: 1px solid rgba(173, 227, 255, 0.55) !important;
        font-weight: 700 !important;
        box-shadow: 0 0 0 1px rgba(12, 111, 161, 0.28), 0 8px 18px rgba(5, 70, 110, 0.28) !important;
    }
    div.st-key-expert_generate_btn button:hover {
        background: linear-gradient(135deg, #109684, #1aa9f0) !important;
        border-color: rgba(208, 242, 255, 0.75) !important;
    }
    div.st-key-expert_generate_btn button:focus {
        outline: 2px solid rgba(136, 226, 255, 0.55) !important;
        outline-offset: 2px !important;
    }
    /* Expert follow-up UI: distinct backgrounds for user prompt and AI answer. */
    div.st-key-expert_followup_prompt textarea {
        background: linear-gradient(135deg, rgba(20, 44, 75, 0.92), rgba(17, 35, 58, 0.92)) !important;
        color: #e8f3ff !important;
        border: 1px solid rgba(111, 158, 211, 0.55) !important;
    }
    .expert-followup-answer-box {
        background: linear-gradient(135deg, rgba(18, 69, 43, 0.86), rgba(16, 54, 36, 0.86));
        border: 1px solid rgba(118, 217, 164, 0.44);
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
        color: #eafff5;
        margin-top: 0.35rem;
    }
    .expert-followup-question-preview {
        background: linear-gradient(135deg, rgba(24, 56, 97, 0.88), rgba(21, 44, 73, 0.88));
        border: 1px solid rgba(126, 175, 231, 0.44);
        border-radius: 10px;
        padding: 0.65rem 0.9rem;
        color: #edf6ff;
        margin-top: 0.45rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================================================================
# SIDEBAR CONFIGURATION
# ==============================================================================

# NOTE: Data-processing, serialization and traceability helpers were extracted
# into `ui.data_utils`, `domain.serialization`, and `services.traceability`.

def _select_best_params_from_results(results, config):
    metric1_name = config.get('metric1_name', 'sharpe_ratio')
    metric2_name = config.get('metric2_name', 'total_return')
    weight_metric1 = float(config.get('weight_metric1', 1.0))
    weight_metric2 = float(config.get('weight_metric2', 0.0))

    def get_metric_value(row, name):
        if not row:
            return None
        if name == 'max_drawdown':
            value = row.get('max_drawdown')
            return None if value is None else -value
        if name == 'sharpe_ratio':
            return row.get('sharpe')
        if name == 'total_return':
            return row.get('return')
        if name == 'win_rate':
            return row.get('win_rate')
        if name == 'avg_gain_per_trade':
            return row.get('avg_gain_per_trade')
        if name == 'avg_loss_per_trade':
            value = row.get('avg_loss_per_trade')
            return None if value is None else -value
        if name == 'avg_pl_per_trade':
            return row.get('avg_pl_per_trade')
        return None

    def combined_score(row):
        total_weight = weight_metric1 + weight_metric2
        if total_weight == 0:
            return None
        m1 = get_metric_value(row, metric1_name)
        m2 = get_metric_value(row, metric2_name)
        if weight_metric1 != 0 and m1 is None:
            return None
        if weight_metric2 != 0 and m2 is None:
            return None
        if m1 is None:
            m1 = 0.0
        if m2 is None:
            m2 = 0.0
        return (weight_metric1 * m1 + weight_metric2 * m2) / total_weight

    best_params = None
    best_score = None
    best_window = None
    best_is_metrics = None
    best_oos_metrics = None

    is_map = {row.get('window'): row for row in results.get('in_sample_performance', [])}
    oos_map = {row.get('window'): row for row in results.get('out_of_sample_performance', [])}

    for window in results.get('window_results', []):
        window_id = window.get('window_info', {}).get('window')
        if window_id is None:
            continue
        is_row = is_map.get(window_id)
        oos_row = oos_map.get(window_id)
        is_score = combined_score(is_row)
        oos_score = combined_score(oos_row)

        if is_score is None and oos_score is None:
            continue
        if is_score is None:
            window_score = oos_score
        elif oos_score is None:
            window_score = is_score
        else:
            window_score = (is_score + oos_score) / 2

        if best_score is None or window_score > best_score:
            best_score = window_score
            best_params = (window.get('best_params') or {}).copy()
            best_window = window_id
            best_is_metrics = is_row
            best_oos_metrics = oos_row

    return best_params, best_score, best_window, best_is_metrics, best_oos_metrics


def _normalize_vote_value(value):
    """Normalize heterogeneous trial values for robust voting/aggregation."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return ""
        s_l = s.lower()
        if s_l in {"true", "1", "yes", "y"}:
            return True
        if s_l in {"false", "0", "no", "n"}:
            return False
        try:
            f = float(s)
            if np.isfinite(f):
                if abs(f - round(f)) < 1e-9:
                    return int(round(f))
                return float(f)
        except Exception:
            return s
        return s
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if not np.isfinite(f):
            return None
        if abs(f - round(f)) < 1e-9:
            return int(round(f))
        return f
    return value


def _weighted_median(values, weights):
    """Compute weighted median for numeric vectors."""
    if len(values) == 0:
        return None
    vals = np.asarray(values, dtype=float)
    wts = np.asarray(weights, dtype=float)
    mask = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
    vals = vals[mask]
    wts = wts[mask]
    if vals.size == 0:
        return None
    order = np.argsort(vals)
    vals = vals[order]
    wts = wts[order]
    cumsum = np.cumsum(wts)
    cutoff = 0.5 * float(np.sum(wts))
    idx = int(np.searchsorted(cumsum, cutoff, side="left"))
    idx = max(0, min(idx, len(vals) - 1))
    return float(vals[idx])


def _build_robust_set_summary(results, config):
    """
    Build level-1 robust set summary:
    Top-N in each window + weighted vote/median across windows.
    """
    enabled = bool(config.get("robust_tests_enabled", False)) if isinstance(config, dict) else False
    top_n = max(1, int(config.get("robust_top_n_per_window", 20) or 20)) if isinstance(config, dict) else 20
    min_windows = max(1, int(config.get("robust_min_windows", 3) or 3)) if isinstance(config, dict) else 3
    use_for_final = bool(config.get("robust_use_for_final_backtest", False)) if isinstance(config, dict) else False
    selected_params = [str(p) for p in (config.get("selected_params") or [])] if isinstance(config, dict) else []

    summary = {
        "schema_version": "robust_set.v1",
        "enabled": enabled,
        "use_for_final_backtest": use_for_final,
        "method": "top_n_weighted_vote",
        "top_n_per_window": int(top_n),
        "min_windows_required": int(min_windows),
        "status": "disabled" if not enabled else "pending",
        "windows_total": int(len(results.get("window_results", []) or [])) if isinstance(results, dict) else 0,
        "windows_used": [],
        "candidates_total": 0,
        "robust_params": {},
        "support_by_param": [],
        "notes": [],
    }

    if not enabled:
        summary["notes"].append("Robust set désactivé dans la configuration.")
        return summary
    if not isinstance(results, dict):
        summary["status"] = "invalid_input"
        summary["notes"].append("Résultats WFO invalides.")
        return summary

    pool_rows = []
    windows_with_trials = []
    score_col = "combined_score"

    for wr in results.get("window_results", []) or []:
        if not isinstance(wr, dict):
            continue
        info = wr.get("window_info", {}) or {}
        window_id = info.get("window")
        trials = wr.get("optimization_trials") or wr.get("optimization_results") or []
        if not isinstance(trials, list) or not trials:
            continue
        tdf = pd.DataFrame(trials)
        if tdf.empty or score_col not in tdf.columns:
            continue
        tdf[score_col] = pd.to_numeric(tdf[score_col], errors="coerce")
        tdf = tdf.replace([np.inf, -np.inf], np.nan).dropna(subset=[score_col])
        if tdf.empty:
            continue
        tdf = tdf.sort_values(score_col, ascending=False).head(top_n).reset_index(drop=True)
        windows_with_trials.append(window_id)
        for rank_idx, (_, row) in enumerate(tdf.iterrows(), start=1):
            item = row.to_dict()
            # Higher weight for higher rank in each window.
            item["_weight"] = float(top_n - rank_idx + 1)
            item["_window"] = window_id
            item["_rank"] = rank_idx
            pool_rows.append(item)

    if not pool_rows:
        summary["status"] = "no_trials"
        summary["notes"].append("Aucun trial exploitable trouvé dans les fenêtres WFO.")
        return summary

    pool_df = pd.DataFrame(pool_rows)
    windows_used = sorted(list({w for w in windows_with_trials if w is not None}))
    summary["windows_used"] = windows_used
    summary["candidates_total"] = int(len(pool_df))
    if len(windows_used) < min_windows:
        summary["status"] = "insufficient_windows"
        summary["notes"].append(
            f"Fenêtres exploitables insuffisantes ({len(windows_used)}/{min_windows})."
        )
    else:
        summary["status"] = "ok"

    strategy_param_names = set(DEFAULT_PARAM_GRID.keys()) | {
        "exit_sar_enabled",
        "exit_macd_enabled",
        "exit_macd_type_a",
        "exit_macd_type_b",
        "order_sizing_mode",
        "order_fixed_cash",
        "fees_pct",
    }
    candidate_cols = [c for c in pool_df.columns if c in strategy_param_names]
    if selected_params:
        ordered = [p for p in selected_params if p in candidate_cols]
        extra = [p for p in candidate_cols if p not in ordered]
        candidate_cols = ordered + extra

    support_rows = []
    robust_params = {}
    for param in candidate_cols:
        sub = pool_df[[param, "_weight", "_window"]].copy()
        sub = sub.dropna(subset=[param])
        if sub.empty:
            continue
        sub["norm_value"] = sub[param].map(_normalize_vote_value)
        sub = sub.dropna(subset=["norm_value"])
        if sub.empty:
            continue

        numeric_vals = pd.to_numeric(sub["norm_value"], errors="coerce")
        is_numeric = numeric_vals.notna().all() and sub["norm_value"].map(lambda x: isinstance(x, (int, float, np.integer, np.floating))).all()

        if is_numeric:
            values = numeric_vals.astype(float).to_numpy()
            weights = pd.to_numeric(sub["_weight"], errors="coerce").fillna(0).to_numpy(dtype=float)
            med = _weighted_median(values, weights)
            observed_unique = np.unique(values[np.isfinite(values)])
            if med is None or observed_unique.size == 0:
                continue
            # Keep value on observed grid to avoid out-of-domain drift.
            chosen = float(observed_unique[int(np.argmin(np.abs(observed_unique - med)))])
            if abs(chosen - round(chosen)) < 1e-9:
                chosen = int(round(chosen))
            chosen_token = _normalize_vote_value(chosen)
        else:
            weight_by_value = (
                sub.groupby("norm_value")["_weight"]
                .sum()
                .sort_values(ascending=False)
            )
            if weight_by_value.empty:
                continue
            chosen_token = _normalize_vote_value(weight_by_value.index[0])
            chosen = chosen_token

        total_weight = float(pd.to_numeric(sub["_weight"], errors="coerce").fillna(0).sum())
        selected_mask = sub["norm_value"].map(_normalize_vote_value) == chosen_token
        selected_weight = float(pd.to_numeric(sub.loc[selected_mask, "_weight"], errors="coerce").fillna(0).sum())
        support_ratio = (selected_weight / total_weight) if total_weight > 0 else 0.0
        windows_hit = sorted(list({w for w in sub.loc[selected_mask, "_window"].tolist() if w is not None}))

        robust_params[param] = _to_jsonable(chosen)
        support_rows.append(
            {
                "parameter": param,
                "selected_value": _to_jsonable(chosen),
                "support_ratio": float(support_ratio),
                "weighted_support": float(selected_weight),
                "weighted_total": float(total_weight),
                "windows_covered": int(len(windows_hit)),
                "distinct_values": int(sub["norm_value"].nunique(dropna=True)),
            }
        )

    summary["robust_params"] = _sanitize_for_json(robust_params)
    summary["support_by_param"] = _sanitize_for_json(support_rows)
    if summary["status"] == "ok" and not robust_params:
        summary["status"] = "insufficient_data"
        summary["notes"].append("Impossible de construire des paramètres robustes à partir des Top-N.")
    return summary


def _select_final_params_from_results(results, config):
    """Select final params using classic best-window or robust-set mode with safe fallback."""
    best_params, best_score, best_window, best_is_metrics, best_oos_metrics = _select_best_params_from_results(results, config)
    robust_summary = _build_robust_set_summary(results, config)
    use_robust = bool(config.get("robust_tests_enabled", False)) and bool(
        config.get("robust_use_for_final_backtest", False)
    )
    robust_params = robust_summary.get("robust_params") if isinstance(robust_summary, dict) else None

    if use_robust and isinstance(robust_params, dict) and robust_params and robust_summary.get("status") == "ok":
        return (
            robust_params.copy(),
            None,
            None,
            None,
            None,
            "robust_set",
            robust_summary,
        )
    return (
        best_params,
        best_score,
        best_window,
        best_is_metrics,
        best_oos_metrics,
        "best_window",
        robust_summary,
    )

def _build_results_payload():
    config_snapshot = get_current_config()
    results_snapshot = st.session_state.get("wfo_results")
    if isinstance(results_snapshot, dict):
        robust_summary = _build_robust_set_summary(results_snapshot, config_snapshot)
        results_snapshot = dict(results_snapshot)
        results_snapshot["robust_set_summary"] = robust_summary
        st.session_state["wfo_results"] = results_snapshot
    run_metadata = st.session_state.get("wfo_run_metadata")
    traceability = _build_traceability_payload(
        config_snapshot=config_snapshot,
        results_snapshot=results_snapshot,
        run_metadata=run_metadata
    )
    payload = {
        "exported_at": datetime.datetime.now().isoformat(),
        "config": config_snapshot,
        "wfo_results": results_snapshot,
        "has_final_portfolio": "final_portfolio" in st.session_state,
        "traceability": traceability,
        "pine_precheck_report": st.session_state.get("pine_precheck_report"),
        "pine_compatibility_report": st.session_state.get("pine_compatibility_report"),
        "pine_strategy_spec": st.session_state.get("pine_strategy_spec"),
        "pine_strategy_spec_validation": st.session_state.get("pine_strategy_spec_validation"),
        "pine_codegen_report": st.session_state.get("pine_codegen_report"),
        "pine_generated_module_path": st.session_state.get("pine_generated_module_path"),
        "pine_generation_trace": st.session_state.get("pine_generation_trace"),
        "pine_artifacts_manifest": st.session_state.get("pine_artifacts_manifest"),
        "pine_beta_readiness_report": st.session_state.get("pine_beta_readiness_report"),
        "pine_execution_gate_report": st.session_state.get("pine_execution_gate_report"),
        "pine_parity_report": st.session_state.get("pine_parity_report"),
        "pine_request_security_diagnostics": st.session_state.get("pine_request_security_diagnostics"),
        "pine_mtf_parity_proof_report": st.session_state.get("pine_mtf_parity_proof_report"),
        "pine_parity_reference_payload": st.session_state.get("pine_parity_reference_payload"),
        "pine_parity_reference_validation": st.session_state.get("pine_parity_reference_validation"),
        "pine_parity_reference_metrics": st.session_state.get("pine_parity_reference_metrics"),
        "pine_source_name": st.session_state.get("pine_source_name"),
        "pine_source_encoding": st.session_state.get("pine_source_encoding"),
        "pine_library_files": st.session_state.get("pine_library_files", []),
        "pine_library_names": st.session_state.get("pine_library_names", []),
        "pine_library_paths": st.session_state.get("pine_library_paths", []),
        "pine_import_mapping": st.session_state.get("pine_import_mapping", {}),
    }
    return _sanitize_for_json(payload)

def _build_trials_dataframe_from_results(results):
    rows = []
    if not isinstance(results, dict):
        return pd.DataFrame()
    for window_result in results.get("window_results", []):
        if not isinstance(window_result, dict):
            continue
        info = window_result.get("window_info", {}) or {}
        window_id = info.get("window")
        trials = window_result.get("optimization_trials") or []
        trial_source = "optimization_trials"
        # Backward compatibility for historical ZIPs that only store top
        # optimization rows under `optimization_results`.
        if not isinstance(trials, list) or len(trials) == 0:
            trials = window_result.get("optimization_results") or []
            trial_source = "optimization_results"
        for idx, trial in enumerate(trials):
            if not isinstance(trial, dict):
                continue
            row = dict(trial)
            row["window"] = window_id
            row["trial_rank_in_window"] = idx + 1
            row["trial_source"] = trial_source
            row["in_sample_start"] = info.get("in_sample_start")
            row["in_sample_end"] = info.get("in_sample_end")
            row["out_sample_start"] = info.get("out_sample_start")
            row["out_sample_end"] = info.get("out_sample_end")
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def _build_window_info_dataframe(results):
    rows = []
    if not isinstance(results, dict):
        return pd.DataFrame()
    for window_result in results.get("window_results", []):
        if not isinstance(window_result, dict):
            continue
        info = window_result.get("window_info", {}) or {}
        row = dict(info)
        row["optimization_trials_count"] = int(window_result.get("optimization_trials_count", 0))
        row["evaluations"] = int(window_result.get("evaluations", 0))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

# NOTE: `_to_jsonable` and `_safe_float_scalar` now come from
# `domain.serialization`.

def _compact_trials_for_expert(trials_df, selected_params=None, top_per_window=5, max_windows=60):
    if selected_params is None:
        selected_params = []
    if trials_df is None or trials_df.empty:
        return {"top_trials": [], "window_stats": [], "total_trials": 0}
    if "window" not in trials_df.columns or "combined_score" not in trials_df.columns:
        return {"top_trials": [], "window_stats": [], "total_trials": int(len(trials_df))}

    df_tmp = trials_df.copy()
    df_tmp["window"] = pd.to_numeric(df_tmp["window"], errors="coerce")
    df_tmp["combined_score"] = pd.to_numeric(df_tmp["combined_score"], errors="coerce")
    df_tmp = df_tmp.dropna(subset=["window", "combined_score"])
    if df_tmp.empty:
        return {"top_trials": [], "window_stats": [], "total_trials": int(len(trials_df))}

    df_tmp = df_tmp.sort_values(["window", "combined_score"], ascending=[True, False])
    windows = sorted(df_tmp["window"].unique().tolist())
    if len(windows) > max_windows:
        windows = windows[-max_windows:]
    df_tmp = df_tmp[df_tmp["window"].isin(windows)]

    safe_params = [p for p in selected_params if p in df_tmp.columns]
    if not safe_params:
        candidate_cols = [c for c in df_tmp.columns if c not in {"window", "combined_score", "trial_rank_in_window"}]
        safe_params = candidate_cols[:8]

    top_rows = []
    grouped = df_tmp.groupby("window", sort=True)
    for window_id, g in grouped:
        for _, row in g.head(int(max(1, top_per_window))).iterrows():
            item = {
                "window": int(row["window"]),
                "combined_score": float(row["combined_score"]),
            }
            for p in safe_params:
                item[p] = _to_jsonable(row.get(p))
            top_rows.append(item)

    stats_rows = (
        grouped["combined_score"]
        .agg(best="max", median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75), n="count")
        .reset_index()
    )
    window_stats = []
    for _, row in stats_rows.iterrows():
        window_stats.append({
            "window": int(row["window"]),
            "best": float(row["best"]),
            "median": float(row["median"]),
            "q25": float(row["q25"]),
            "q75": float(row["q75"]),
            "n": int(row["n"]),
        })

    return {
        "top_trials": top_rows,
        "window_stats": window_stats,
        "total_trials": int(len(df_tmp)),
        "parameters_used": safe_params,
    }

def _build_pine_artifacts_summary_for_expert():
    """Build a compact Pine artifacts summary consumable by Expert diagnostics."""
    precheck = st.session_state.get("pine_precheck_report")
    compat = st.session_state.get("pine_compatibility_report")
    spec = st.session_state.get("pine_strategy_spec")
    spec_validation = st.session_state.get("pine_strategy_spec_validation")
    trace = st.session_state.get("pine_generation_trace")
    codegen = st.session_state.get("pine_codegen_report")
    beta_readiness = st.session_state.get("pine_beta_readiness_report")
    execution_gate = st.session_state.get("pine_execution_gate_report")
    parity_report = st.session_state.get("pine_parity_report")
    request_security_diagnostics = st.session_state.get("pine_request_security_diagnostics")
    mtf_parity_proof = st.session_state.get("pine_mtf_parity_proof_report")
    parity_reference = st.session_state.get("pine_parity_reference_metrics")
    parity_reference_payload = st.session_state.get("pine_parity_reference_payload")
    parity_reference_validation = st.session_state.get("pine_parity_reference_validation")
    generated_module_path = st.session_state.get("pine_generated_module_path")
    source_text = st.session_state.get("pine_source_text")
    library_files = st.session_state.get("pine_library_files")
    import_mapping = st.session_state.get("pine_import_mapping")
    if not isinstance(library_files, list):
        library_files = []
    if not isinstance(import_mapping, dict):
        import_mapping = {}

    if (
        not any(isinstance(x, dict) and x for x in (precheck, compat, spec, spec_validation, trace))
        and not isinstance(beta_readiness, dict)
        and not isinstance(execution_gate, dict)
        and not isinstance(parity_report, dict)
        and not isinstance(request_security_diagnostics, dict)
        and not isinstance(mtf_parity_proof, dict)
        and not isinstance(source_text, str)
        and not library_files
    ):
        return {}

    strategy_meta = (spec.get("strategy") or {}) if isinstance(spec, dict) else {}
    source_meta = (spec.get("source") or {}) if isinstance(spec, dict) else {}
    transcription_meta = (spec.get("transcription") or {}) if isinstance(spec, dict) else {}
    source_sha1 = None
    if isinstance(precheck, dict):
        source_sha1 = precheck.get("source_sha1")
    if not source_sha1 and isinstance(source_meta, dict):
        source_sha1 = source_meta.get("source_sha1")

    source_preview = ""
    if isinstance(source_text, str) and source_text.strip():
        source_preview = "\n".join(source_text.splitlines()[:60]).strip()

    return _sanitize_for_json(
        {
            "available": True,
            "strategy_id": strategy_meta.get("id"),
            "strategy_name": strategy_meta.get("name"),
            "source_name": st.session_state.get("pine_source_name") or source_meta.get("file_name"),
            "source_sha1": source_sha1,
            "source_encoding": st.session_state.get("pine_source_encoding"),
            "precheck_status": precheck.get("status") if isinstance(precheck, dict) else None,
            "compatibility_status": compat.get("status") if isinstance(compat, dict) else None,
            "compatibility_score": compat.get("compatibility_score") if isinstance(compat, dict) else None,
            "blocking_items_count": len((compat.get("blocking_items") or [])) if isinstance(compat, dict) else 0,
            "spec_schema_version": spec.get("schema_version") if isinstance(spec, dict) else None,
            "spec_valid": bool(spec_validation.get("valid", False)) if isinstance(spec_validation, dict) else None,
            "spec_errors_count": len((spec_validation.get("errors") or [])) if isinstance(spec_validation, dict) else 0,
            "spec_parser_requested": (
                transcription_meta.get("parser_backend_requested")
                if isinstance(transcription_meta, dict)
                else None
            ),
            "spec_parser_used": (
                transcription_meta.get("parser_backend_used")
                if isinstance(transcription_meta, dict)
                else None
            ),
            "spec_parser_fallback": (
                bool(transcription_meta.get("fallback_to_regex", False))
                if isinstance(transcription_meta, dict)
                else None
            ),
            "imports": list(spec.get("imports") or []) if isinstance(spec, dict) else [],
            "import_resolution_count": len((precheck.get("import_resolution") or []))
            if isinstance(precheck, dict)
            else 0,
            "external_functions_called_count": int(precheck.get("external_functions_called_count", 0))
            if isinstance(precheck, dict)
            else 0,
            "external_functions_missing_count": int(precheck.get("external_functions_missing_count", 0))
            if isinstance(precheck, dict)
            else 0,
            "libraries_count": len(library_files),
            "libraries_names": [str(item.get("source_name") or "") for item in library_files if isinstance(item, dict)],
            "import_mapping_count": len(import_mapping),
            "import_mapping_keys": sorted([str(k) for k in import_mapping.keys()]),
            "inputs_count": len(spec.get("inputs") or []) if isinstance(spec, dict) else 0,
            "generation_trace": trace if isinstance(trace, dict) else {},
            "codegen_status": (codegen.get("status") if isinstance(codegen, dict) else None),
            "beta_ready": bool(beta_readiness.get("beta_ready")) if isinstance(beta_readiness, dict) else None,
            "beta_readiness_score": (
                float(beta_readiness.get("readiness_score", 0.0)) if isinstance(beta_readiness, dict) else None
            ),
            "execution_gate_status": execution_gate.get("status") if isinstance(execution_gate, dict) else None,
            "execution_gate_can_run": execution_gate.get("can_run") if isinstance(execution_gate, dict) else None,
            "execution_gate_blockers_count": len((execution_gate.get("blockers") or []))
            if isinstance(execution_gate, dict)
            else 0,
            "parity_status": parity_report.get("status") if isinstance(parity_report, dict) else None,
            "parity_pass": (
                parity_report.get("parity_pass")
                if isinstance(parity_report, dict) and "parity_pass" in parity_report
                else None
            ),
            "parity_checks_count": len((parity_report.get("checks") or [])) if isinstance(parity_report, dict) else 0,
            "parity_detail_available": bool(parity_report.get("detail_available", False))
            if isinstance(parity_report, dict)
            else None,
            "parity_detail_pass": (
                parity_report.get("detail_pass")
                if isinstance(parity_report, dict) and "detail_pass" in parity_report
                else None
            ),
            "parity_reference_schema_version": (
                parity_reference_payload.get("schema_version")
                if isinstance(parity_reference_payload, dict)
                else None
            ),
            "parity_reference_source": (
                parity_reference_payload.get("source")
                if isinstance(parity_reference_payload, dict)
                else {}
            ),
            "parity_reference_valid": (
                bool(parity_reference_validation.get("valid", False))
                if isinstance(parity_reference_validation, dict)
                else None
            ),
            "parity_reference_metrics_count": (
                len((parity_reference_payload.get("reference_metrics") or {}))
                if isinstance(parity_reference_payload, dict)
                else 0
            ),
            "request_security_diagnostics_status": (
                request_security_diagnostics.get("status")
                if isinstance(request_security_diagnostics, dict)
                else None
            ),
            "request_security_diagnostics_count": (
                int(request_security_diagnostics.get("request_security_count", 0))
                if isinstance(request_security_diagnostics, dict)
                else 0
            ),
            "mtf_parity_proof_status": (
                mtf_parity_proof.get("status") if isinstance(mtf_parity_proof, dict) else None
            ),
            "mtf_parity_proof_pass": (
                mtf_parity_proof.get("proof_pass")
                if isinstance(mtf_parity_proof, dict) and "proof_pass" in mtf_parity_proof
                else None
            ),
            "parity_reference_metrics": parity_reference if isinstance(parity_reference, dict) else {},
            "generated_module_path": generated_module_path if isinstance(generated_module_path, str) else None,
            "source_preview": source_preview,
        }
    )

def _build_strategy_context_for_expert(current_conf):
    if not isinstance(current_conf, dict):
        return {}
    selected_params = [str(p) for p in (current_conf.get("selected_params") or [])]

    entry_params = [
        "timeperiod", "StDev", "coeff_medianeBBW", "coef_mediane",
        "fenetre_lowest", "seuil_lowest", "longueur_mediane", "Nb_bars_above"
    ]
    exit_params = [
        "user_exit_sma_length", "sar_start", "sar_increment", "sar_maximum",
        "macd_fast_length", "macd_slow_length", "macd_signal_length"
    ]

    parameter_ranges = {}
    for p in selected_params:
        p_min = current_conf.get(f"{p}_min")
        p_max = current_conf.get(f"{p}_max")
        p_step = current_conf.get(f"{p}_step")
        parameter_ranges[p] = {
            "min": _to_jsonable(p_min),
            "max": _to_jsonable(p_max),
            "step": _to_jsonable(p_step),
            "role": PARAMETER_HELP.get(p, ""),
        }

    enabled_entry = [p for p in selected_params if p in entry_params]
    enabled_exit = [p for p in selected_params if p in exit_params]
    fixed_exit = [p for p in exit_params if p not in enabled_exit]

    exit_modules = {
        "exit_sar_enabled": bool(current_conf.get("exit_sar_enabled", True)),
        "exit_macd_enabled": bool(current_conf.get("exit_macd_enabled", True)),
        "exit_macd_type_a": bool(current_conf.get("exit_macd_type_a", True)),
        "exit_macd_type_b": bool(current_conf.get("exit_macd_type_b", True)),
    }

    entry_logic = [
        "Bollinger/BBW: detection de compression via coeff_medianeBBW, coef_mediane et lowest sur fenetre_lowest/seuil_lowest.",
        "Validation du momentum/filtre via Nb_bars_above et longueur_mediane.",
        "timeperiod et StDev reglent l'echantillonnage et la largeur des bandes."
    ]
    exit_logic = [
        "Sortie SMA via user_exit_sma_length.",
        "Sortie Parabolic SAR conditionnee par exit_sar_enabled (sar_start/sar_increment/sar_maximum).",
        "Sortie MACD conditionnee par exit_macd_enabled (types A/B + macd_fast_length/macd_slow_length/macd_signal_length)."
    ]

    pine_artifacts = _build_pine_artifacts_summary_for_expert()

    context = {
        "selected_params": selected_params,
        "entry_params_enabled": enabled_entry,
        "exit_params_enabled": enabled_exit,
        "exit_params_fixed": fixed_exit,
        "exit_modules": exit_modules,
        "parameter_ranges": parameter_ranges,
        "entry_logic_summary": entry_logic,
        "exit_logic_summary": exit_logic,
    }
    if pine_artifacts:
        context["pine_v3_artifacts"] = pine_artifacts
    return context

def _extract_final_backtest_for_expert(results, current_conf, df_source=None):
    summary = {
        "available": False,
        "source": "none",
        "run_final_backtest": False,
        "final_params": st.session_state.get("final_params"),
        "final_params_score": _to_jsonable(st.session_state.get("final_params_score")),
        "final_params_window": _to_jsonable(st.session_state.get("final_params_window")),
        "final_params_source": _to_jsonable(st.session_state.get("final_params_source")),
        "is_score": _to_jsonable(st.session_state.get("final_params_is_score")),
        "oos_score": _to_jsonable(st.session_state.get("final_params_oos_score")),
    }

    pf = st.session_state.get("final_portfolio")
    if pf is not None:
        summary["available"] = True
        summary["source"] = "final_portfolio"
        summary["run_final_backtest"] = True
        try:
            summary["strategy_total_return_pct"] = float(_safe_float_scalar(getattr(pf, "total_return", np.nan) * 100))
        except Exception:
            summary["strategy_total_return_pct"] = None
        try:
            summary["strategy_sharpe"] = float(_safe_float_scalar(getattr(pf, "sharpe_ratio", np.nan)))
        except Exception:
            summary["strategy_sharpe"] = None
        try:
            summary["strategy_max_drawdown_pct"] = float(_safe_float_scalar(getattr(pf, "max_drawdown", np.nan) * 100))
        except Exception:
            summary["strategy_max_drawdown_pct"] = None
        try:
            summary["strategy_win_rate_pct"] = float(_safe_float_scalar(getattr(getattr(pf, "trades", object()), "win_rate", np.nan) * 100))
        except Exception:
            summary["strategy_win_rate_pct"] = None
        try:
            summary["strategy_n_trades"] = int(len(pf.trades))
        except Exception:
            summary["strategy_n_trades"] = None
        try:
            summary["strategy_calmar"] = float(_safe_float_scalar(getattr(pf, "calmar_ratio", np.nan)))
        except Exception:
            summary["strategy_calmar"] = None
        try:
            summary["strategy_sortino"] = float(_safe_float_scalar(getattr(pf, "sortino_ratio", np.nan)))
        except Exception:
            summary["strategy_sortino"] = None
    else:
        trades_df = st.session_state.get("final_trades_df")
        stats_df = st.session_state.get("final_trade_stats_df")
        if trades_df is not None or stats_df is not None:
            summary["available"] = True
            summary["source"] = "zip_import"
            if trades_df is not None and hasattr(trades_df, "empty") and not trades_df.empty:
                tdf = trades_df.copy()
                summary["strategy_n_trades"] = int(len(tdf))
                if "pnl" in tdf.columns:
                    pnl_s = pd.to_numeric(tdf["pnl"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                    if not pnl_s.empty:
                        summary["trades_pnl_mean"] = float(pnl_s.mean())
                        summary["trades_pnl_median"] = float(pnl_s.median())
                ret_s = _get_return_series(tdf)
                if ret_s is not None:
                    ret_s = pd.to_numeric(ret_s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                    if not ret_s.empty:
                        summary["trades_return_mean_pct"] = float(ret_s.mean() * 100.0)
            if stats_df is not None and hasattr(stats_df, "empty") and not stats_df.empty:
                summary["final_trade_stats_preview"] = stats_df.head(40).to_dict("records")

    # Buy & Hold comparison when price data is available.
    price_df = st.session_state.get("final_backtest_df")
    if price_df is None or getattr(price_df, "empty", True):
        price_df = df_source
    if price_df is not None and hasattr(price_df, "empty") and not price_df.empty:
        if "Close" in price_df.columns:
            px_series = price_df["Close"]
        else:
            px_series = price_df.iloc[:, 0]
        px_series = pd.to_numeric(px_series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(px_series) > 1:
            initial_price = float(px_series.iloc[0])
            final_price = float(px_series.iloc[-1])
            if initial_price != 0 and np.isfinite(initial_price) and np.isfinite(final_price):
                buy_hold_return_pct = ((final_price / initial_price) - 1.0) * 100.0
                summary["buy_hold_return_pct"] = float(buy_hold_return_pct)
                st_ret = summary.get("strategy_total_return_pct")
                if isinstance(st_ret, (int, float)) and np.isfinite(float(st_ret)):
                    summary["outperformance_vs_buy_hold_pct"] = float(st_ret - buy_hold_return_pct)

    return summary

def _compute_deterministic_expert_alerts(results, current_conf, trials_df=None, final_backtest_summary=None):
    """Generate deterministic pre-diagnostic alerts before running the Expert LLM."""
    alerts = []
    if not isinstance(results, dict):
        return alerts

    is_df = pd.DataFrame(results.get("in_sample_performance", []))
    oos_df = pd.DataFrame(results.get("out_of_sample_performance", []))
    n_windows = int(max(len(is_df), len(oos_df)))

    if n_windows < 3:
        alerts.append({
            "type": "inconclusive",
            "severity": "medium",
            "message": f"Peu de fenêtres/cycles exploitables ({n_windows}). Robustesse statistique limitée.",
            "rule": "n_windows < 3",
        })

    # Gap IS/OOS overfitting alert.
    if not is_df.empty and not oos_df.empty and "window" in is_df.columns and "window" in oos_df.columns:
        merged = pd.merge(
            is_df[["window", "return", "sharpe"]],
            oos_df[["window", "return", "sharpe"]],
            on="window",
            how="inner",
            suffixes=("_is", "_oos")
        )
        if not merged.empty:
            for col in ["return_is", "return_oos", "sharpe_is", "sharpe_oos"]:
                merged[col] = pd.to_numeric(merged[col], errors="coerce")
            merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["return_is", "return_oos", "sharpe_is", "sharpe_oos"])
            if not merged.empty:
                ret_gap = float((merged["return_is"] - merged["return_oos"]).mean())
                shp_gap = float((merged["sharpe_is"] - merged["sharpe_oos"]).mean())
                if ret_gap > 8.0 or shp_gap > 0.8:
                    sev = "high" if ret_gap > 15.0 or shp_gap > 1.2 else "medium"
                    alerts.append({
                        "type": "overfitting",
                        "severity": sev,
                        "message": (
                            "Écart IS/OOS élevé détecté "
                            f"(return_gap_moyen={ret_gap:.2f}, sharpe_gap_moyen={shp_gap:.2f})."
                        ),
                        "rule": "mean(IS-OOS) return > 8 or sharpe > 0.8",
                    })
                oos_return_series = pd.to_numeric(
                    oos_df["return"] if "return" in oos_df.columns else pd.Series(dtype=float),
                    errors="coerce"
                ).replace([np.inf, -np.inf], np.nan).dropna()
                oos_return_mean = float(oos_return_series.mean()) if not oos_return_series.empty else 0.0
                if oos_return_mean <= 0:
                    alerts.append({
                        "type": "oos_underperformance",
                        "severity": "medium",
                        "message": "Rendement OOS moyen <= 0. Vérifier robustesse et adéquation des paramètres.",
                        "rule": "mean(oos_return) <= 0",
                    })

    # Parameter instability alert.
    best_params_df = pd.DataFrame(results.get("best_params", []))
    selected_params = current_conf.get("selected_params", []) if isinstance(current_conf, dict) else []
    numeric_instability = []
    if not best_params_df.empty and selected_params:
        for p in selected_params:
            if p in best_params_df.columns:
                s = pd.to_numeric(best_params_df[p], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if len(s) >= 4:
                    mean_abs = float(abs(s.mean()))
                    std_v = float(s.std(ddof=0))
                    if mean_abs > 1e-9:
                        cv = std_v / mean_abs
                        if cv > 0.35:
                            numeric_instability.append((p, cv))
    if numeric_instability:
        numeric_instability.sort(key=lambda x: x[1], reverse=True)
        top = ", ".join([f"{p} (cv={cv:.2f})" for p, cv in numeric_instability[:4]])
        alerts.append({
            "type": "instability",
            "severity": "medium",
            "message": f"Instabilité paramétrique détectée: {top}.",
            "rule": "cv(param) > 0.35 sur >=4 fenêtres",
        })

    # Trial volume alert.
    if trials_df is not None and not getattr(trials_df, "empty", True):
        total_trials = int(len(trials_df))
    else:
        total_trials = 0
        for wr in results.get("window_results", []):
            try:
                total_trials += int(wr.get("optimization_trials_count", 0))
            except Exception:
                continue
    if n_windows > 0:
        avg_trials = total_trials / max(1, n_windows)
        if avg_trials < 30:
            alerts.append({
                "type": "insufficient_trials",
                "severity": "medium",
                "message": f"Volume d'essais faible: ~{avg_trials:.1f} trials/fenêtre.",
                "rule": "avg_trials_per_window < 30",
            })

    # Final backtest underperformance vs Buy & Hold.
    if isinstance(final_backtest_summary, dict) and final_backtest_summary:
        outperf = final_backtest_summary.get("outperformance_vs_buy_hold_pct")
        if isinstance(outperf, (int, float)) and np.isfinite(float(outperf)):
            outperf = float(outperf)
            if outperf < -5.0:
                sev = "high" if outperf < -12.0 else "medium"
                alerts.append({
                    "type": "final_backtest_underperformance",
                    "severity": sev,
                    "message": (
                        "Final backtest sous-performe le buy&hold "
                        f"de {abs(outperf):.2f} points de pourcentage."
                    ),
                    "rule": "strategy_return - buy_hold_return < -5%",
                })

    return alerts

def _repo_root_dir():
    """Return repository root from current app location."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _expert_reports_dir():
    """Canonical expert reports folder shared across sessions/layout changes."""
    folder = os.path.join(_repo_root_dir(), "reports", "expert")
    os.makedirs(folder, exist_ok=True)
    return folder


def _expert_prompt_templates_path():
    """Canonical prompt templates path."""
    return os.path.join(_expert_reports_dir(), "prompt_templates.json")


def _legacy_expert_prompt_templates_path():
    """Legacy path used when app.py lived in repo root."""
    return os.path.join(os.path.dirname(__file__), "reports", "expert", "prompt_templates.json")


def _read_prompt_templates_file(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        templates = data.get("templates", {})
        return templates if isinstance(templates, dict) else {}
    except Exception:
        return {}

def _load_expert_prompt_templates():
    canonical_path = _expert_prompt_templates_path()
    legacy_path = _legacy_expert_prompt_templates_path()

    # Merge legacy + canonical for backward compatibility.
    # Canonical has priority on key collisions.
    templates_legacy = _read_prompt_templates_file(legacy_path)
    templates_canonical = _read_prompt_templates_file(canonical_path)
    merged = {}
    if isinstance(templates_legacy, dict):
        merged.update(templates_legacy)
    if isinstance(templates_canonical, dict):
        merged.update(templates_canonical)

    # Self-heal: if merged content exists but canonical file is missing/empty, persist it.
    if merged and (not os.path.exists(canonical_path) or not templates_canonical):
        try:
            _save_expert_prompt_templates(merged)
        except Exception:
            pass
    return merged

def _save_expert_prompt_templates(templates):
    path = _expert_prompt_templates_path()
    payload = {
        "schema_version": "expert_prompt_templates.v1",
        "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "templates": templates if isinstance(templates, dict) else {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _resolve_template_prompts(template_obj, default_system_prompt, default_user_prompt):
    """
    Resolve template prompt fields with backward-compatible aliases.

    Supported aliases:
    - system: system_prompt, system, prompt_system, system_message, systemPrompt
    - user:   user_prompt, user, prompt_user, prompt_utilisateur, user_message, prompt, instruction
    """
    template_obj = template_obj if isinstance(template_obj, dict) else {}

    system_keys = ("system_prompt", "system", "prompt_system", "system_message", "systemPrompt")
    user_keys = ("user_prompt", "user", "prompt_user", "prompt_utilisateur", "user_message", "prompt", "instruction")

    def _pick(keys):
        for key in keys:
            val = template_obj.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""

    resolved_system = _pick(system_keys)
    resolved_user = _pick(user_keys)

    missing_system = not bool(resolved_system)
    missing_user = not bool(resolved_user)

    if missing_system:
        resolved_system = str(default_system_prompt or "")
    if missing_user:
        resolved_user = str(default_user_prompt or "")

    return {
        "system_prompt": resolved_system,
        "user_prompt": resolved_user,
        "missing_system": missing_system,
        "missing_user": missing_user,
    }


def _build_effective_expert_user_prompt(user_prompt_text, default_user_prompt_text):
    """
    Build final user prompt sent to LLM.

    Rules:
    - If template contains a context marker, replace it with auto-built prompt data.
    - If no obvious WFO payload is detected, append auto-built prompt data.
    - Preserve user's custom instructions at the top.
    """
    raw_prompt = str(user_prompt_text or "").strip()
    auto_prompt = str(default_user_prompt_text or "").strip()
    if not raw_prompt:
        return auto_prompt, "auto_only"

    markers = ("{{AUTO_WFO_CONTEXT}}", "{{AUTO_USER_PROMPT}}", "{{PROMPT_UTILISATEUR_AUTO}}")
    for marker in markers:
        if marker in raw_prompt:
            return raw_prompt.replace(marker, auto_prompt), f"marker:{marker}"

    probe = raw_prompt.lower()
    has_embedded_payload = (
        "donnees:" in probe
        or "run_context" in probe
        or "oos_performance" in probe
        or "is_performance" in probe
        or "best_params_by_window" in probe
        or "trials_compact" in probe
    )
    if has_embedded_payload:
        return raw_prompt, "custom_with_payload"

    final_prompt = (
        f"{raw_prompt}\n\n"
        "Contexte WFO auto-injecte pour analyse factuelle:\n"
        f"{auto_prompt}"
    )
    return final_prompt, "auto_appended"


def _list_saved_expert_reports(limit=300):
    folder = _expert_reports_dir()
    try:
        files = [
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if name.endswith(".json") and name.startswith("expert_")
        ]
    except Exception:
        return []

    files = [p for p in files if os.path.isfile(p)]
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    if isinstance(limit, int) and limit > 0:
        files = files[:limit]
    return files


def _load_saved_expert_report(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("Format de rapport invalide (objet JSON attendu).")
    except Exception as exc:
        return None, str(exc)

    result_json = payload.get("result")
    if not isinstance(result_json, dict):
        result_json = payload.get("result_json")
    if not isinstance(result_json, dict):
        result_json = {}

    loaded = {
        "status": payload.get("status", "loaded"),
        "result_json": result_json,
        "raw_text": payload.get("raw_text", ""),
        "model_info": payload.get("model_info", {}),
        "timings_ms": payload.get("timings_ms", {}),
        "warnings": payload.get("warnings", []),
        "run_id": payload.get("run_id", "unknown"),
        "deterministic_alerts": payload.get("deterministic_alerts", []),
        "strategy_context": payload.get("strategy_context", {}),
        "final_backtest": payload.get("final_backtest", {}),
        "used_system_prompt": payload.get("used_system_prompt", ""),
        "used_user_prompt": payload.get("used_user_prompt", ""),
        "used_template_name": payload.get("used_template_name"),
        "prompt_injection_mode": payload.get("prompt_injection_mode"),
        "loaded_from_path": path,
        "loaded_saved_at": payload.get("saved_at"),
    }
    return loaded, None


def _format_saved_expert_report_label(path):
    try:
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        mtime = "unknown_time"
    name = os.path.basename(path)
    return f"{mtime} | {name}"

# NOTE: `_render_deterministic_alerts` is now provided by `ui.expert_report`.

def _build_expert_input_data(results, current_conf):
    if not isinstance(results, dict):
        return None

    # If a context pack was loaded from ZIP and matches the current run, reuse it
    # to preserve rich historical context for Expert analysis.
    imported_pack = st.session_state.get("expert_context_pack")
    if isinstance(imported_pack, dict):
        imported_input = imported_pack.get("expert_input_data")
        imported_run_id = str(imported_pack.get("run_id") or "").strip()

        run_meta = st.session_state.get("wfo_run_metadata") or {}
        current_run_id = str(run_meta.get("run_id") or "").strip()
        if not current_run_id and isinstance(results, dict):
            try:
                current_run_id = str(
                    ((results.get("traceability") or {}).get("run") or {}).get("run_id") or ""
                ).strip()
            except Exception:
                current_run_id = ""

        run_matches = True
        if imported_run_id and current_run_id:
            run_matches = imported_run_id == current_run_id

        if run_matches and isinstance(imported_input, dict):
            try:
                ctx = imported_input.get("context") or {}
                imported_obj = ExpertInputData(
                    context=ExpertRunContext(
                        run_id=str(ctx.get("run_id") or current_run_id or f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                        optimization_regime=str(ctx.get("optimization_regime") or "classic"),
                        timeframe=str(ctx.get("timeframe") or ""),
                        start_date=str(ctx.get("start_date") or ""),
                        end_date=str(ctx.get("end_date") or ""),
                        selected_params=[str(p) for p in (ctx.get("selected_params") or [])],
                    ),
                    out_of_sample_performance=list(imported_input.get("out_of_sample_performance") or []),
                    in_sample_performance=list(imported_input.get("in_sample_performance") or []),
                    best_params=list(imported_input.get("best_params") or []),
                    all_trials=list(imported_input.get("all_trials") or []),
                    adaptive_guidance=list(imported_input.get("adaptive_guidance") or []),
                    adaptive_summary=dict(imported_input.get("adaptive_summary") or {}),
                    price_features=dict(imported_input.get("price_features") or {}),
                    strategy_context=dict(imported_input.get("strategy_context") or {}),
                    deterministic_alerts=list(imported_input.get("deterministic_alerts") or []),
                    final_backtest=dict(imported_input.get("final_backtest") or {}),
                )

                # Refresh final backtest summary from current session artifacts if available.
                fresh_final = _extract_final_backtest_for_expert(
                    results=results,
                    current_conf=current_conf,
                    df_source=st.session_state.get("df")
                )
                if isinstance(fresh_final, dict) and fresh_final:
                    imported_obj.final_backtest = fresh_final
                return imported_obj
            except Exception:
                pass

    run_meta = st.session_state.get("wfo_run_metadata") or {}
    run_id = str(run_meta.get("run_id") or f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    selected_params = current_conf.get("selected_params", []) if isinstance(current_conf, dict) else []

    context = ExpertRunContext(
        run_id=run_id,
        optimization_regime=str((results.get("mode") or current_conf.get("optimization_regime", "classic"))),
        timeframe=str(current_conf.get("timeframe", "")),
        start_date=str(current_conf.get("start_date", "")),
        end_date=str(current_conf.get("end_date", "")),
        selected_params=[str(p) for p in selected_params],
    )

    trials_df = st.session_state.get("all_trials_df")
    if trials_df is None or getattr(trials_df, "empty", True):
        trials_df = _build_trials_dataframe_from_results(results)

    compact_trials = _compact_trials_for_expert(
        trials_df=trials_df,
        selected_params=selected_params,
        top_per_window=5,
        max_windows=60
    )

    adaptive_guidance = results.get("adaptive_guidance", [])
    if isinstance(adaptive_guidance, list) and len(adaptive_guidance) > 60:
        adaptive_guidance = adaptive_guidance[-60:]

    strategy_context = _build_strategy_context_for_expert(current_conf)
    final_backtest_summary = _extract_final_backtest_for_expert(
        results=results,
        current_conf=current_conf,
        df_source=st.session_state.get("df")
    )
    deterministic_alerts = _compute_deterministic_expert_alerts(
        results=results,
        current_conf=current_conf,
        trials_df=trials_df,
        final_backtest_summary=final_backtest_summary
    )

    best_params_rows = results.get("best_params", [])
    if not isinstance(best_params_rows, list):
        best_params_rows = []

    # Enrich best-params rows with window identifiers and best score proxies when
    # available (important for imported historical ZIPs).
    wr_list = results.get("window_results", [])
    if isinstance(wr_list, list) and wr_list:
        enriched = []
        for idx, params_row in enumerate(best_params_rows):
            row = dict(params_row) if isinstance(params_row, dict) else {}
            wr = wr_list[idx] if idx < len(wr_list) and isinstance(wr_list[idx], dict) else {}
            info = wr.get("window_info", {}) if isinstance(wr.get("window_info"), dict) else {}
            if row.get("window") is None and info.get("window") is not None:
                row["window"] = info.get("window")

            opt_rows = wr.get("optimization_results")
            if isinstance(opt_rows, list) and opt_rows:
                top = opt_rows[0] if isinstance(opt_rows[0], dict) else {}
                if row.get("combined_score") is None and isinstance(top, dict):
                    row["combined_score"] = _to_jsonable(top.get("combined_score"))
            enriched.append(row)
        best_params_rows = enriched

    # Attach IS/OOS metrics by window for stronger optimization diagnostics.
    if best_params_rows:
        is_by_window = {}
        for row in results.get("in_sample_performance", []) or []:
            if isinstance(row, dict) and row.get("window") is not None:
                is_by_window[row.get("window")] = row
        oos_by_window = {}
        for row in results.get("out_of_sample_performance", []) or []:
            if isinstance(row, dict) and row.get("window") is not None:
                oos_by_window[row.get("window")] = row

        enriched_perf = []
        for row in best_params_rows:
            if not isinstance(row, dict):
                continue
            item = dict(row)
            w = item.get("window")
            if w in is_by_window:
                item["is_return"] = _to_jsonable(is_by_window[w].get("return"))
                item["is_sharpe"] = _to_jsonable(is_by_window[w].get("sharpe"))
            if w in oos_by_window:
                item["oos_return"] = _to_jsonable(oos_by_window[w].get("return"))
                item["oos_sharpe"] = _to_jsonable(oos_by_window[w].get("sharpe"))
            enriched_perf.append(item)
        best_params_rows = enriched_perf

    return ExpertInputData(
        context=context,
        out_of_sample_performance=results.get("out_of_sample_performance", []),
        in_sample_performance=results.get("in_sample_performance", []),
        best_params=best_params_rows,
        all_trials=compact_trials.get("top_trials", []),
        adaptive_guidance=adaptive_guidance,
        adaptive_summary=results.get("adaptive_summary", {}),
        price_features={
            "trials_window_stats": compact_trials.get("window_stats", []),
            "total_trials_compact_source": compact_trials.get("total_trials", 0),
            "parameters_included_in_trials": compact_trials.get("parameters_used", []),
        },
        strategy_context=strategy_context,
        deterministic_alerts=deterministic_alerts,
        final_backtest=final_backtest_summary,
    )


def _extract_window_ids_from_question(question_text):
    """Extract referenced window ids (w3, W10, fenêtre 5) from a free-text question."""
    if not isinstance(question_text, str) or not question_text.strip():
        return []
    text = question_text.lower()
    found = set()
    for pat in [r"\bw\s*(\d{1,3})\b", r"fen[êe]tre\s*(\d{1,3})", r"window\s*(\d{1,3})"]:
        for m in re.findall(pat, text):
            try:
                found.add(int(m))
            except Exception:
                continue
    return sorted(found)


def _build_oos_rankings(results):
    """Build deterministic rankings to help follow-up Q/A answer precisely."""
    oos = results.get("out_of_sample_performance", []) if isinstance(results, dict) else []
    df_oos = pd.DataFrame(oos)
    if df_oos.empty:
        return {}

    for col in ["window", "return", "sharpe", "max_drawdown", "win_rate", "n_trades"]:
        if col in df_oos.columns:
            df_oos[col] = pd.to_numeric(df_oos[col], errors="coerce")
    df_oos = df_oos.dropna(subset=["window"])
    if df_oos.empty:
        return {}

    keep_cols = [c for c in ["window", "return", "sharpe", "max_drawdown", "win_rate", "n_trades"] if c in df_oos.columns]
    by_return = df_oos.sort_values("return", ascending=False)[keep_cols] if "return" in df_oos.columns else pd.DataFrame()
    by_sharpe = df_oos.sort_values("sharpe", ascending=False)[keep_cols] if "sharpe" in df_oos.columns else pd.DataFrame()

    def rows(df):
        if df is None or df.empty:
            return []
        out = []
        for _, r in df.iterrows():
            item = {}
            for c in keep_cols:
                v = r.get(c)
                item[c] = _to_jsonable(v)
            out.append(item)
        return out

    return {
        "by_return_desc": rows(by_return),
        "by_sharpe_desc": rows(by_sharpe),
    }


def _build_trials_context_for_followup(trials_df, question_text, max_rows_full=800):
    """Provide full trials when small, otherwise rich summaries + targeted slices."""
    if trials_df is None or getattr(trials_df, "empty", True):
        return {"available": False, "rows_total": 0}

    df_t = trials_df.copy()
    if "window" in df_t.columns:
        df_t["window"] = pd.to_numeric(df_t["window"], errors="coerce")
    if "combined_score" in df_t.columns:
        df_t["combined_score"] = pd.to_numeric(df_t["combined_score"], errors="coerce")

    rows_total = int(len(df_t))
    out = {
        "available": True,
        "rows_total": rows_total,
        "columns": list(df_t.columns),
    }

    if rows_total <= int(max_rows_full):
        out["records_full"] = _sanitize_for_json(df_t.to_dict("records"))
        return out

    # Global summaries for large tables.
    if {"window", "combined_score"}.issubset(df_t.columns):
        grouped = (
            df_t.dropna(subset=["window", "combined_score"])
            .groupby("window")["combined_score"]
            .agg(count="count", best="max", median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
            .reset_index()
        )
        out["window_score_stats"] = _sanitize_for_json(grouped.to_dict("records"))

    if "combined_score" in df_t.columns:
        df_s = df_t.dropna(subset=["combined_score"]).sort_values("combined_score", ascending=False)
        out["top_global"] = _sanitize_for_json(df_s.head(200).to_dict("records"))
        out["bottom_global"] = _sanitize_for_json(df_s.tail(120).to_dict("records"))

    # Targeted slices for referenced windows in the question.
    window_ids = _extract_window_ids_from_question(question_text)
    if window_ids and "window" in df_t.columns and "combined_score" in df_t.columns:
        targeted = {}
        for wid in window_ids:
            wd = df_t[df_t["window"] == wid].dropna(subset=["combined_score"]).sort_values("combined_score", ascending=False)
            if wd.empty:
                continue
            targeted[str(wid)] = {
                "top": _sanitize_for_json(wd.head(80).to_dict("records")),
                "bottom": _sanitize_for_json(wd.tail(30).to_dict("records")),
            }
        out["targeted_windows"] = targeted
    else:
        out["sample_head"] = _sanitize_for_json(df_t.head(250).to_dict("records"))

    return out


def _build_followup_context_pack(results, expert_input_preview, expert_last, question_text):
    """Assemble a rich context pack for follow-up Q/A over current or imported WFO."""
    results = results if isinstance(results, dict) else {}
    oos = results.get("out_of_sample_performance", []) or []
    ins = results.get("in_sample_performance", []) or []
    best_params = results.get("best_params", []) or []
    window_results = results.get("window_results", []) or []
    window_info_df = st.session_state.get("window_info_df")
    all_trials_df = st.session_state.get("all_trials_df")
    if (all_trials_df is None or getattr(all_trials_df, "empty", True)) and window_results:
        all_trials_df = _build_trials_dataframe_from_results(results)

    context_pack = {
        "run_id": (expert_last or {}).get("run_id"),
        "analysis_json": (expert_last or {}).get("result_json", {}),
        "deterministic_alerts": (expert_last or {}).get("deterministic_alerts", []),
        "strategy_context": (expert_last or {}).get("strategy_context", {}),
        "final_backtest": (expert_last or {}).get("final_backtest", {}),
        "wfo_metrics": {
            "in_sample_performance": ins,
            "out_of_sample_performance": oos,
            "best_params_by_window": best_params,
            "window_count": len(window_results),
            "is_count": len(ins),
            "oos_count": len(oos),
            "best_params_count": len(best_params),
            "robust_set_summary": results.get("robust_set_summary", {}),
        },
        "window_info": _sanitize_for_json(window_info_df.to_dict("records")) if isinstance(window_info_df, pd.DataFrame) and not window_info_df.empty else [],
        "oos_rankings": _build_oos_rankings(results),
        "trials_context": _build_trials_context_for_followup(all_trials_df, question_text),
        "expert_input_compact": {
            "all_trials_compact": (expert_input_preview.all_trials if expert_input_preview else []),
            "adaptive_guidance": (expert_input_preview.adaptive_guidance if expert_input_preview else []),
            "adaptive_summary": (expert_input_preview.adaptive_summary if expert_input_preview else {}),
            "price_features": (expert_input_preview.price_features if expert_input_preview else {}),
        },
    }
    return _sanitize_for_json(context_pack)


def _parse_json_like_text(raw_text):
    """Best-effort parse for JSON-like LLM outputs (raw JSON, fenced JSON, nested JSON string)."""
    text = str(raw_text or "").strip()
    if not text:
        return None

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    for candidate in (text,):
        try:
            parsed = json.loads(candidate)
            # Some models return JSON encoded as string: "\"{...}\""
            if isinstance(parsed, str):
                inner = parsed.strip()
                try:
                    parsed2 = json.loads(inner)
                    if isinstance(parsed2, (dict, list)):
                        return parsed2
                except Exception:
                    return parsed
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            pass

    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        try:
            parsed = json.loads(text[first:last + 1])
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            pass

    return None


def _normalize_followup_text(raw_text):
    """Clean escaped characters to improve readability in Expert follow-up answers."""
    text = str(raw_text or "")
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n")
    text = text.replace("\\t", " ")
    text = text.replace("\\r", "\n")
    text = text.replace("\\\"", "\"")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _format_followup_value(value, indent=0):
    """Format scalar/list/dict payload into readable plain text blocks."""
    prefix = "  " * max(0, int(indent))
    lines = []

    if value is None:
        return [f"{prefix}(non renseigné)"]

    if isinstance(value, dict):
        if not value:
            return [f"{prefix}(vide)"]
        for key, sub_value in value.items():
            key_label = str(key).replace("_", " ").strip().capitalize()
            if isinstance(sub_value, (dict, list)):
                lines.append(f"{prefix}- {key_label}:")
                lines.extend(_format_followup_value(sub_value, indent=indent + 1))
            else:
                lines.append(f"{prefix}- {key_label}: {_normalize_followup_text(sub_value)}")
        return lines

    if isinstance(value, list):
        if not value:
            return [f"{prefix}(liste vide)"]
        for i, item in enumerate(value, start=1):
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{i}.")
                lines.extend(_format_followup_value(item, indent=indent + 1))
            else:
                lines.append(f"{prefix}{i}. {_normalize_followup_text(item)}")
        return lines

    text = _normalize_followup_text(value)
    if not text:
        return [f"{prefix}(vide)"]
    text_lines = text.splitlines() or [text]
    return [f"{prefix}{line}" if idx == 0 else f"{prefix}{line}" for idx, line in enumerate(text_lines)]


def _format_followup_answer_for_display(answer_text):
    """Render follow-up answer into a human-friendly text layout (even when model returns JSON)."""
    raw = str(answer_text or "").strip()
    if not raw:
        return ""

    parsed = _parse_json_like_text(raw)
    if parsed is None:
        return _normalize_followup_text(raw)

    # If model returned a plain string via JSON quoting.
    if isinstance(parsed, str):
        return _normalize_followup_text(parsed)

    key_labels = {
        "focus": "Point clé",
        "definition": "Définition",
        "why_this_matters": "Pourquoi c'est important",
        "evidence": "Éléments de preuve",
        "how_to_apply": "Comment l'appliquer concrètement",
        "a_1_a_3_actions_recommandees_maintenant": "Actions recommandées maintenant",
        "actions_recommandees": "Actions recommandées",
        "recommended_actions": "Actions recommandées",
        "explication_simple": "Explication simple",
        "global_assessment": "Évaluation globale",
        "key_findings": "Constats clés",
        "alerts": "Alertes",
        "limitations": "Limites",
        "disclaimer": "Note",
        "run_id": "Run ID",
    }
    preferred_order = [
        "focus",
        "definition",
        "why_this_matters",
        "evidence",
        "how_to_apply",
        "a_1_a_3_actions_recommandees_maintenant",
        "actions_recommandees",
        "recommended_actions",
        "explication_simple",
        "global_assessment",
        "key_findings",
        "alerts",
        "limitations",
        "disclaimer",
        "run_id",
    ]

    if isinstance(parsed, list):
        return "\n".join(_format_followup_value(parsed))

    if not isinstance(parsed, dict):
        return _normalize_followup_text(raw)

    lines = []
    used = set()
    for key in preferred_order:
        if key not in parsed:
            continue
        used.add(key)
        title = key_labels.get(key, key.replace("_", " ").strip().capitalize())
        lines.append(f"{title}")
        lines.append("-" * len(title))
        lines.extend(_format_followup_value(parsed.get(key)))
        lines.append("")

    for key in parsed.keys():
        if key in used:
            continue
        title = key_labels.get(key, key.replace("_", " ").strip().capitalize())
        lines.append(f"{title}")
        lines.append("-" * len(title))
        lines.extend(_format_followup_value(parsed.get(key)))
        lines.append("")

    out = "\n".join(lines).strip()
    return _normalize_followup_text(out)


def _expert_input_to_dict(expert_input):
    """Serialize ExpertInputData dataclass to exportable dict."""
    if expert_input is None:
        return {}
    try:
        return _sanitize_for_json(
            {
                "context": {
                    "run_id": expert_input.context.run_id,
                    "optimization_regime": expert_input.context.optimization_regime,
                    "timeframe": expert_input.context.timeframe,
                    "start_date": expert_input.context.start_date,
                    "end_date": expert_input.context.end_date,
                    "selected_params": expert_input.context.selected_params,
                },
                "out_of_sample_performance": expert_input.out_of_sample_performance,
                "in_sample_performance": expert_input.in_sample_performance,
                "best_params": expert_input.best_params,
                "all_trials": expert_input.all_trials,
                "adaptive_guidance": expert_input.adaptive_guidance,
                "adaptive_summary": expert_input.adaptive_summary,
                "price_features": expert_input.price_features,
                "strategy_context": expert_input.strategy_context,
                "deterministic_alerts": expert_input.deterministic_alerts,
                "final_backtest": expert_input.final_backtest,
            }
        )
    except Exception:
        return {}


def _build_expert_context_pack_for_export(results, config_snapshot):
    """Build a rich Expert context pack to persist in ZIP exports."""
    expert_input = _build_expert_input_data(results, config_snapshot if isinstance(config_snapshot, dict) else {})
    expert_last = st.session_state.get("expert_last_response") or {}
    followup_context = _build_followup_context_pack(
        results=results,
        expert_input_preview=expert_input,
        expert_last=expert_last,
        question_text="",
    )
    run_meta = st.session_state.get("wfo_run_metadata") or {}
    run_id = str(run_meta.get("run_id") or expert_last.get("run_id") or "")
    if not run_id and isinstance(expert_input, ExpertInputData):
        run_id = str(expert_input.context.run_id or "")

    payload = {
        "schema_version": "expert_context_pack.v1",
        "generated_at_utc": _utc_now_iso(),
        "run_id": run_id,
        "expert_input_data": _expert_input_to_dict(expert_input),
        "followup_context": followup_context,
        "followup_history": _sanitize_for_json(st.session_state.get("expert_followup_history") or []),
    }
    return _sanitize_for_json(payload)

def _resolve_pine_source_for_artifacts():
    """Resolve Pine source text and metadata from session state for ZIP artifacts."""
    source_text = ""
    source_encoding = str(st.session_state.get("pine_source_encoding") or "")
    source_name = str(st.session_state.get("pine_source_name") or "").strip()
    source_path = str(st.session_state.get("pine_file_path") or "").strip()

    if source_path and os.path.exists(source_path):
        try:
            source_text, source_encoding = _read_text_file_with_fallback(source_path)
        except Exception:
            source_text = ""
    elif isinstance(st.session_state.get("pine_source_text"), str):
        source_text = st.session_state.get("pine_source_text") or ""

    if not source_name and source_path:
        source_name = os.path.basename(source_path)
    if not source_name:
        source_name = "strategy_source.pine.txt"

    if source_text and not source_name.lower().endswith((".pine", ".txt")):
        source_name = f"{source_name}.txt"

    source_sha1 = hashlib.sha1(source_text.encode("utf-8", errors="ignore")).hexdigest() if source_text else None
    return _sanitize_for_json(
        {
            "text": source_text,
            "encoding": source_encoding or None,
            "source_name": source_name,
            "source_path": source_path or None,
            "source_sha1": source_sha1,
            "line_count": source_text.count("\n") + (1 if source_text else 0),
            "char_count": len(source_text),
        }
    )


def _resolve_pine_libraries_for_artifacts():
    """Resolve uploaded Pine libraries from session state for ZIP artifacts."""
    raw_manifest = st.session_state.get("pine_library_files")
    if not isinstance(raw_manifest, list) or len(raw_manifest) == 0:
        return []

    out = []
    for entry in raw_manifest:
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("path") or "").strip()
        source_name = str(entry.get("source_name") or "").strip()
        if not path or not os.path.exists(path):
            continue
        try:
            text, encoding = _read_text_file_with_fallback(path)
        except Exception:
            continue
        if not source_name:
            source_name = os.path.basename(path)
        source_sha1 = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest() if text else None
        out.append(
            {
                "source_name": source_name,
                "path": path,
                "text": text,
                "encoding": encoding,
                "source_sha1": source_sha1,
                "line_count": text.count("\n") + (1 if text else 0),
                "char_count": len(text),
            }
        )
    return _sanitize_for_json(out)


def _build_pine_generation_trace(
    source_artifact: dict,
    library_artifacts: list[dict] | None,
    import_mapping: dict | None,
    codegen_report: dict | None,
    generated_module_path: str | None,
    precheck_report: dict | None,
    compatibility_report: dict | None,
    strategy_spec: dict | None,
    strategy_spec_validation: dict | None,
):
    """Build deterministic generation trace for Pine V3 artifacts."""
    pre = precheck_report if isinstance(precheck_report, dict) else {}
    compat = compatibility_report if isinstance(compatibility_report, dict) else {}
    spec = strategy_spec if isinstance(strategy_spec, dict) else {}
    spec_val = strategy_spec_validation if isinstance(strategy_spec_validation, dict) else {}
    src = source_artifact if isinstance(source_artifact, dict) else {}
    libs = library_artifacts if isinstance(library_artifacts, list) else []
    mapping = import_mapping if isinstance(import_mapping, dict) else {}
    codegen = codegen_report if isinstance(codegen_report, dict) else {}
    generated_path = str(generated_module_path or "").strip()

    has_any = (
        bool(src.get("text"))
        or bool(pre)
        or bool(compat)
        or bool(spec)
        or bool(spec_val)
        or bool(libs)
        or bool(mapping)
        or bool(codegen)
        or bool(generated_path)
        or bool(parity_ref_payload)
        or bool(parity_ref_validation)
    )
    if not has_any:
        return {}

    spec_strategy = spec.get("strategy") if isinstance(spec.get("strategy"), dict) else {}
    trace = {
        "schema_version": "pine_generation_trace.v1",
        "generated_at_utc": _utc_now_iso(),
        "pipeline_stage": "P0.6",
        "strategy_mode": st.session_state.get("strategy_mode"),
        "strategy_id": st.session_state.get("strategy_id") or spec_strategy.get("id"),
        "source": {
            "name": src.get("source_name"),
            "path": src.get("source_path"),
            "encoding": src.get("encoding"),
            "sha1": src.get("source_sha1") or pre.get("source_sha1"),
            "line_count": src.get("line_count"),
            "char_count": src.get("char_count"),
        },
        "libraries": {
            "count": len(libs),
            "names": [str(item.get("source_name")) for item in libs if isinstance(item, dict)],
            "sha1_list": [str(item.get("source_sha1")) for item in libs if isinstance(item, dict) and item.get("source_sha1")],
        },
        "import_mapping": {
            "count": len(mapping),
            "keys": sorted([str(k) for k in mapping.keys()]),
        },
        "codegen": {
            "status": codegen.get("status"),
            "output_path": codegen.get("output_path") or generated_path or None,
            "module_name": codegen.get("module_name"),
            "changed": codegen.get("changed"),
        },
        "precheck": {
            "status": pre.get("status"),
            "detected_version": pre.get("detected_version"),
            "error_count": len(pre.get("errors") or []),
            "warning_count": len(pre.get("warnings") or []),
        },
        "compatibility": {
            "status": compat.get("status"),
            "score": compat.get("compatibility_score"),
            "is_blocking": bool(compat.get("is_blocking", False)),
            "blocking_items_count": len(compat.get("blocking_items") or []),
        },
        "spec": {
            "schema_version": spec.get("schema_version"),
            "strategy_id": spec_strategy.get("id"),
            "strategy_name": spec_strategy.get("name"),
            "valid": bool(spec_val.get("valid", False)) if spec_val else None,
            "validation_error_count": len(spec_val.get("errors") or []) if spec_val else None,
            "sha256": _sha256_json(spec) if spec else None,
            "imports_count": len(spec.get("imports") or []) if spec else 0,
            "inputs_count": len(spec.get("inputs") or []) if spec else 0,
        },
        "llm_used": False,
    }
    return _sanitize_for_json(trace)


def _build_pine_artifacts_manifest(
    source_artifact: dict,
    library_artifacts: list[dict] | None,
    import_mapping: dict | None,
    codegen_report: dict | None,
    generated_module_path: str | None,
    precheck_report: dict | None,
    compatibility_report: dict | None,
    strategy_spec: dict | None,
    strategy_spec_validation: dict | None,
    generation_trace: dict | None,
    beta_readiness_report: dict | None = None,
    execution_gate_report: dict | None = None,
    parity_report: dict | None = None,
    request_security_diagnostics: dict | None = None,
    mtf_parity_proof_report: dict | None = None,
    parity_reference_payload: dict | None = None,
    parity_reference_validation: dict | None = None,
):
    """Build export manifest describing Pine artifacts bundled in the ZIP."""
    src = source_artifact if isinstance(source_artifact, dict) else {}
    libs = library_artifacts if isinstance(library_artifacts, list) else []
    pre = precheck_report if isinstance(precheck_report, dict) else {}
    compat = compatibility_report if isinstance(compatibility_report, dict) else {}
    spec = strategy_spec if isinstance(strategy_spec, dict) else {}
    spec_val = strategy_spec_validation if isinstance(strategy_spec_validation, dict) else {}
    trace = generation_trace if isinstance(generation_trace, dict) else {}
    beta = beta_readiness_report if isinstance(beta_readiness_report, dict) else {}
    gate = execution_gate_report if isinstance(execution_gate_report, dict) else {}
    parity = parity_report if isinstance(parity_report, dict) else {}
    mtf_diag = request_security_diagnostics if isinstance(request_security_diagnostics, dict) else {}
    mtf_proof = mtf_parity_proof_report if isinstance(mtf_parity_proof_report, dict) else {}
    parity_ref_payload = parity_reference_payload if isinstance(parity_reference_payload, dict) else {}
    parity_ref_validation = parity_reference_validation if isinstance(parity_reference_validation, dict) else {}
    mapping = import_mapping if isinstance(import_mapping, dict) else {}
    codegen = codegen_report if isinstance(codegen_report, dict) else {}
    generated_path = str(generated_module_path or "").strip()

    has_any = (
        bool(src.get("text"))
        or bool(pre)
        or bool(compat)
        or bool(spec)
        or bool(spec_val)
        or bool(trace)
        or bool(gate)
        or bool(libs)
        or bool(mapping)
        or bool(codegen)
        or bool(generated_path)
        or bool(mtf_diag)
        or bool(mtf_proof)
    )
    if not has_any:
        return {}

    return _sanitize_for_json(
        {
            "schema_version": "pine_artifacts_manifest.v1",
            "generated_at_utc": _utc_now_iso(),
            "has_source": bool(src.get("text")),
            "libraries_count": len(libs),
            "library_names": [str(item.get("source_name")) for item in libs if isinstance(item, dict)],
            "import_mapping_count": len(mapping),
            "has_generated_strategy_module": bool(codegen.get("output_path") or generated_path),
            "has_precheck_report": bool(pre),
            "has_compatibility_report": bool(compat),
            "has_strategy_spec": bool(spec),
            "has_strategy_spec_validation": bool(spec_val),
            "has_generation_trace": bool(trace),
            "has_beta_readiness_report": bool(beta),
            "has_execution_gate_report": bool(gate),
            "has_parity_report": bool(parity),
            "has_request_security_diagnostics": bool(mtf_diag),
            "has_mtf_parity_proof_report": bool(mtf_proof),
            "has_parity_reference_payload": bool(parity_ref_payload),
            "has_parity_reference_validation": bool(parity_ref_validation),
            "source_name": src.get("source_name"),
            "source_sha1": src.get("source_sha1") or pre.get("source_sha1"),
            "compatibility_status": compat.get("status"),
            "strategy_spec_valid": bool(spec_val.get("valid", False)) if spec_val else None,
            "beta_ready": bool(beta.get("beta_ready")) if beta else None,
            "execution_gate_status": gate.get("status") if gate else None,
            "execution_gate_can_run": gate.get("can_run") if gate else None,
            "parity_pass": parity.get("parity_pass") if parity else None,
            "request_security_diagnostics_status": mtf_diag.get("status") if mtf_diag else None,
            "request_security_diagnostics_count": mtf_diag.get("request_security_count") if mtf_diag else None,
            "mtf_parity_proof_status": mtf_proof.get("status") if mtf_proof else None,
            "mtf_parity_proof_pass": mtf_proof.get("proof_pass") if mtf_proof else None,
            "parity_detail_available": bool(parity.get("detail_available", False)) if parity else None,
            "parity_detail_pass": parity.get("detail_pass") if parity else None,
            "parity_reference_schema_version": parity_ref_payload.get("schema_version") if parity_ref_payload else None,
            "parity_reference_valid": parity_ref_validation.get("valid") if parity_ref_validation else None,
            "artifact_files": {
                "strategy_source": "strategy_source.pine.txt",
                "libraries_manifest": "pine_libraries_manifest.json",
                "libraries_dir": "pine_libraries/",
                "import_mapping": "import_mapping.json",
                "compatibility_report": "compatibility_report.json",
                "strategy_spec": "strategy_spec.v1.json",
                "strategy_spec_validation": "strategy_spec_validation.json",
                "generated_strategy_module": "generated_strategy.py",
                "generation_trace": "generation_trace.json",
                "beta_readiness_report": "pine_beta_readiness_report.json",
                "execution_gate_report": "pine_execution_gate_report.json",
                "parity_report": "pine_parity_report.json",
                "request_security_diagnostics": "pine_request_security_diagnostics.json",
                "mtf_parity_proof_report": "pine_mtf_parity_proof_report.json",
                "parity_reference_payload": "pine_parity_reference.v1.json",
                "parity_reference_validation": "pine_parity_reference_validation.json",
                "parity_reference_metrics_legacy": "pine_parity_reference_metrics.json",
            },
        }
    )

def _build_data_source_descriptor(config_snapshot, df):
    return _build_data_source_descriptor_base(config_snapshot=config_snapshot, df=df)


def _build_replay_manifest(payload, snapshot_mode_requested, snapshot_mode_actual, has_df_snapshot, data_source):
    return _build_replay_manifest_base(
        payload=payload,
        snapshot_mode_requested=snapshot_mode_requested,
        snapshot_mode_actual=snapshot_mode_actual,
        has_df_snapshot=has_df_snapshot,
        data_source=data_source,
        run_meta=st.session_state.get("wfo_run_metadata") or {},
        final_params=st.session_state.get("final_params"),
        final_start_date=st.session_state.get("final_start_date"),
        final_end_date=st.session_state.get("final_end_date"),
        final_file_path=st.session_state.get("final_file_path"),
        v3_artifacts=st.session_state.get("pine_artifacts_manifest") or {},
    )

def _export_results_zip(data_snapshot_mode="manifest_only", df_max_rows=200000, full_package=False):
    if "wfo_results" not in st.session_state:
        st.error("No results available to export.")
        return None

    results = st.session_state["wfo_results"]
    df = st.session_state.get("df")

    pine_precheck_report = st.session_state.get("pine_precheck_report")
    pine_compatibility_report = st.session_state.get("pine_compatibility_report")
    pine_strategy_spec = st.session_state.get("pine_strategy_spec")
    pine_strategy_spec_validation = st.session_state.get("pine_strategy_spec_validation")
    pine_import_mapping = st.session_state.get("pine_import_mapping", {})
    if not isinstance(pine_import_mapping, dict):
        pine_import_mapping = {}
    pine_codegen_report = st.session_state.get("pine_codegen_report")
    pine_generated_module_path = st.session_state.get("pine_generated_module_path")
    pine_parity_report = st.session_state.get("pine_parity_report")
    pine_request_security_diagnostics = st.session_state.get("pine_request_security_diagnostics")
    pine_mtf_parity_proof_report = st.session_state.get("pine_mtf_parity_proof_report")
    pine_parity_reference_payload = st.session_state.get("pine_parity_reference_payload")
    pine_parity_reference_validation = st.session_state.get("pine_parity_reference_validation")
    pine_parity_reference_metrics = st.session_state.get("pine_parity_reference_metrics")
    pine_beta_readiness_report = _build_pine_beta_readiness_report(
        precheck_report=pine_precheck_report,
        compatibility_report=pine_compatibility_report,
        strategy_spec=pine_strategy_spec,
        strategy_spec_validation=pine_strategy_spec_validation,
        codegen_report=pine_codegen_report,
        generated_module_path=pine_generated_module_path,
    )
    pine_execution_gate_report = _build_pine_execution_gate_report(
        strategy_mode=st.session_state.get("strategy_mode", DEFAULT_STRATEGY_MODE),
        beta_readiness_report=pine_beta_readiness_report,
        parity_reference_payload=pine_parity_reference_payload,
        parity_reference_validation=pine_parity_reference_validation,
        parity_report=pine_parity_report,
        enforce_parity_when_reference=bool(st.session_state.get("pine_enforce_parity_gate", True)),
    )
    pine_source_artifact = _resolve_pine_source_for_artifacts()
    pine_library_artifacts = _resolve_pine_libraries_for_artifacts()
    pine_generation_trace = _build_pine_generation_trace(
        source_artifact=pine_source_artifact,
        library_artifacts=pine_library_artifacts,
        import_mapping=pine_import_mapping,
        codegen_report=pine_codegen_report,
        generated_module_path=pine_generated_module_path,
        precheck_report=pine_precheck_report,
        compatibility_report=pine_compatibility_report,
        strategy_spec=pine_strategy_spec,
        strategy_spec_validation=pine_strategy_spec_validation,
    )
    pine_artifacts_manifest = _build_pine_artifacts_manifest(
        source_artifact=pine_source_artifact,
        library_artifacts=pine_library_artifacts,
        import_mapping=pine_import_mapping,
        codegen_report=pine_codegen_report,
        generated_module_path=pine_generated_module_path,
        precheck_report=pine_precheck_report,
        compatibility_report=pine_compatibility_report,
        strategy_spec=pine_strategy_spec,
        strategy_spec_validation=pine_strategy_spec_validation,
        generation_trace=pine_generation_trace,
        beta_readiness_report=pine_beta_readiness_report,
        execution_gate_report=pine_execution_gate_report,
        parity_report=pine_parity_report,
        request_security_diagnostics=pine_request_security_diagnostics,
        mtf_parity_proof_report=pine_mtf_parity_proof_report,
        parity_reference_payload=pine_parity_reference_payload,
        parity_reference_validation=pine_parity_reference_validation,
    )
    if isinstance(pine_source_artifact, dict) and pine_source_artifact.get("text"):
        st.session_state["pine_source_text"] = pine_source_artifact.get("text")
    else:
        st.session_state.pop("pine_source_text", None)
    if isinstance(pine_library_artifacts, list):
        st.session_state["pine_library_files"] = [
            {
                "source_name": str(item.get("source_name") or ""),
                "path": str(item.get("path") or ""),
                "source_sha1": str(item.get("source_sha1") or ""),
                "size_bytes": int(len(str(item.get("text") or "").encode("utf-8", errors="ignore"))),
            }
            for item in pine_library_artifacts
            if isinstance(item, dict)
        ]
        st.session_state["pine_library_paths"] = [str(item.get("path") or "") for item in st.session_state["pine_library_files"]]
        st.session_state["pine_library_names"] = [str(item.get("source_name") or "") for item in st.session_state["pine_library_files"]]
    if isinstance(pine_import_mapping, dict):
        st.session_state["pine_import_mapping"] = dict(pine_import_mapping)
    if isinstance(pine_codegen_report, dict) and pine_codegen_report:
        st.session_state["pine_codegen_report"] = pine_codegen_report
    else:
        st.session_state.pop("pine_codegen_report", None)
    if isinstance(pine_generated_module_path, str) and pine_generated_module_path.strip():
        st.session_state["pine_generated_module_path"] = pine_generated_module_path.strip()
    else:
        st.session_state.pop("pine_generated_module_path", None)
    if isinstance(pine_generation_trace, dict) and pine_generation_trace:
        st.session_state["pine_generation_trace"] = pine_generation_trace
    else:
        st.session_state.pop("pine_generation_trace", None)
    if isinstance(pine_beta_readiness_report, dict) and pine_beta_readiness_report:
        st.session_state["pine_beta_readiness_report"] = pine_beta_readiness_report
    else:
        st.session_state.pop("pine_beta_readiness_report", None)
    if isinstance(pine_execution_gate_report, dict) and pine_execution_gate_report:
        st.session_state["pine_execution_gate_report"] = pine_execution_gate_report
    else:
        st.session_state.pop("pine_execution_gate_report", None)
    if isinstance(pine_parity_report, dict) and pine_parity_report:
        st.session_state["pine_parity_report"] = pine_parity_report
    else:
        st.session_state.pop("pine_parity_report", None)
    if isinstance(pine_request_security_diagnostics, dict) and pine_request_security_diagnostics:
        st.session_state["pine_request_security_diagnostics"] = pine_request_security_diagnostics
    else:
        st.session_state.pop("pine_request_security_diagnostics", None)
    if isinstance(pine_mtf_parity_proof_report, dict) and pine_mtf_parity_proof_report:
        st.session_state["pine_mtf_parity_proof_report"] = pine_mtf_parity_proof_report
    else:
        st.session_state.pop("pine_mtf_parity_proof_report", None)
    if isinstance(pine_parity_reference_payload, dict) and pine_parity_reference_payload:
        st.session_state["pine_parity_reference_payload"] = pine_parity_reference_payload
    else:
        st.session_state.pop("pine_parity_reference_payload", None)
    if isinstance(pine_parity_reference_validation, dict) and pine_parity_reference_validation:
        st.session_state["pine_parity_reference_validation"] = pine_parity_reference_validation
    else:
        st.session_state.pop("pine_parity_reference_validation", None)
    if isinstance(pine_parity_reference_metrics, dict) and pine_parity_reference_metrics:
        st.session_state["pine_parity_reference_metrics"] = pine_parity_reference_metrics
    else:
        st.session_state.pop("pine_parity_reference_metrics", None)
        st.session_state.pop("pine_parity_reference_text", None)
    if isinstance(pine_artifacts_manifest, dict) and pine_artifacts_manifest:
        st.session_state["pine_artifacts_manifest"] = pine_artifacts_manifest
    else:
        st.session_state.pop("pine_artifacts_manifest", None)

    payload = _build_results_payload()
    config_snapshot = payload.get("config", {})
    zip_buffer = io.BytesIO()
    expert_context_pack = _build_expert_context_pack_for_export(results, config_snapshot)

    snapshot_mode_requested = str(data_snapshot_mode)
    snapshot_mode_actual = "manifest_only"
    has_df_snapshot = False
    data_source_descriptor = _build_data_source_descriptor(config_snapshot, df)

    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("results.json", json.dumps(payload, indent=2))
        zf.writestr("audit_trace.json", json.dumps(payload.get("traceability", {}), indent=2))
        zf.writestr("expert_context_pack.json", json.dumps(expert_context_pack, indent=2, ensure_ascii=False))
        if isinstance(pine_precheck_report, dict) and pine_precheck_report:
            zf.writestr("pine_precheck_report.json", json.dumps(pine_precheck_report, indent=2, ensure_ascii=False))
        if isinstance(pine_compatibility_report, dict) and pine_compatibility_report:
            zf.writestr(
                "pine_compatibility_report.json",
                json.dumps(pine_compatibility_report, indent=2, ensure_ascii=False),
            )
            # P0.6 canonical name for replayable V3 compatibility artifact.
            zf.writestr(
                "compatibility_report.json",
                json.dumps(pine_compatibility_report, indent=2, ensure_ascii=False),
            )
        if isinstance(pine_strategy_spec, dict) and pine_strategy_spec:
            zf.writestr("strategy_spec.v1.json", json.dumps(pine_strategy_spec, indent=2, ensure_ascii=False))
        if isinstance(pine_strategy_spec_validation, dict) and pine_strategy_spec_validation:
            zf.writestr(
                "strategy_spec_validation.json",
                json.dumps(pine_strategy_spec_validation, indent=2, ensure_ascii=False),
            )
        if isinstance(pine_beta_readiness_report, dict) and pine_beta_readiness_report:
            zf.writestr(
                "pine_beta_readiness_report.json",
                json.dumps(pine_beta_readiness_report, indent=2, ensure_ascii=False),
            )
        if isinstance(pine_execution_gate_report, dict) and pine_execution_gate_report:
            zf.writestr(
                "pine_execution_gate_report.json",
                json.dumps(pine_execution_gate_report, indent=2, ensure_ascii=False),
            )
        if isinstance(pine_parity_report, dict) and pine_parity_report:
            zf.writestr(
                "pine_parity_report.json",
                json.dumps(pine_parity_report, indent=2, ensure_ascii=False),
            )
        if isinstance(pine_request_security_diagnostics, dict) and pine_request_security_diagnostics:
            zf.writestr(
                "pine_request_security_diagnostics.json",
                json.dumps(pine_request_security_diagnostics, indent=2, ensure_ascii=False),
            )
        if isinstance(pine_mtf_parity_proof_report, dict) and pine_mtf_parity_proof_report:
            zf.writestr(
                "pine_mtf_parity_proof_report.json",
                json.dumps(pine_mtf_parity_proof_report, indent=2, ensure_ascii=False),
            )
        if isinstance(pine_parity_reference_payload, dict) and pine_parity_reference_payload:
            zf.writestr(
                "pine_parity_reference.v1.json",
                json.dumps(pine_parity_reference_payload, indent=2, ensure_ascii=False),
            )
        if isinstance(pine_parity_reference_validation, dict) and pine_parity_reference_validation:
            zf.writestr(
                "pine_parity_reference_validation.json",
                json.dumps(pine_parity_reference_validation, indent=2, ensure_ascii=False),
            )
        if isinstance(pine_parity_reference_metrics, dict) and pine_parity_reference_metrics:
            zf.writestr(
                "pine_parity_reference_metrics.json",
                json.dumps(pine_parity_reference_metrics, indent=2, ensure_ascii=False),
            )
        if isinstance(pine_import_mapping, dict) and pine_import_mapping:
            zf.writestr(
                "import_mapping.json",
                json.dumps(pine_import_mapping, indent=2, ensure_ascii=False),
            )
        if isinstance(pine_generated_module_path, str) and pine_generated_module_path.strip():
            generated_path = pine_generated_module_path.strip()
            if os.path.exists(generated_path):
                try:
                    with open(generated_path, "r", encoding="utf-8") as f:
                        zf.writestr("generated_strategy.py", f.read())
                except Exception:
                    pass
        if isinstance(pine_source_artifact, dict) and pine_source_artifact.get("text"):
            zf.writestr("strategy_source.pine.txt", str(pine_source_artifact.get("text")))
        if isinstance(pine_library_artifacts, list) and pine_library_artifacts:
            libs_manifest = []
            for idx, lib in enumerate(pine_library_artifacts, start=1):
                if not isinstance(lib, dict):
                    continue
                lib_name = str(lib.get("source_name") or f"library_{idx}.txt")
                safe_name = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in lib_name) or f"library_{idx}.txt"
                if not safe_name.lower().endswith((".pine", ".txt")):
                    safe_name = f"{safe_name}.txt"
                zip_name = f"pine_libraries/{idx:02d}_{safe_name}"
                lib_text = str(lib.get("text") or "")
                zf.writestr(zip_name, lib_text)
                libs_manifest.append(
                    {
                        "source_name": lib_name,
                        "source_sha1": lib.get("source_sha1"),
                        "encoding": lib.get("encoding"),
                        "line_count": lib.get("line_count"),
                        "char_count": lib.get("char_count"),
                        "zip_path": zip_name,
                    }
                )
            if libs_manifest:
                zf.writestr("pine_libraries_manifest.json", json.dumps(libs_manifest, indent=2, ensure_ascii=False))
        if isinstance(pine_generation_trace, dict) and pine_generation_trace:
            zf.writestr("generation_trace.json", json.dumps(pine_generation_trace, indent=2, ensure_ascii=False))
            # Backward-compatible alias.
            zf.writestr("pine_generation_trace.json", json.dumps(pine_generation_trace, indent=2, ensure_ascii=False))
        if isinstance(pine_artifacts_manifest, dict) and pine_artifacts_manifest:
            zf.writestr("pine_artifacts_manifest.json", json.dumps(pine_artifacts_manifest, indent=2, ensure_ascii=False))

        if results.get("out_of_sample_performance"):
            oos_df = pd.DataFrame(results["out_of_sample_performance"])
            zf.writestr("out_of_sample_performance.csv", oos_df.to_csv(index=False))
        if results.get("in_sample_performance"):
            is_df = pd.DataFrame(results["in_sample_performance"])
            zf.writestr("in_sample_performance.csv", is_df.to_csv(index=False))
        if results.get("best_params"):
            params_df = pd.DataFrame(results["best_params"])
            zf.writestr("best_params.csv", params_df.to_csv(index=False))

        window_info_df = _build_window_info_dataframe(results)
        if not window_info_df.empty:
            zf.writestr("window_info.csv", window_info_df.to_csv(index=False))

        trials_df = _build_trials_dataframe_from_results(results)
        if not trials_df.empty:
            zf.writestr("all_trials.csv", trials_df.to_csv(index=False))
            if "window" in trials_df.columns:
                for window_id, win_df in trials_df.groupby("window", dropna=False):
                    safe_window = str(window_id).replace("/", "_")
                    zf.writestr(f"trials/window_{safe_window}.csv", win_df.to_csv(index=False))

        if full_package and snapshot_mode_requested == "manifest_only":
            st.info("Package complet actif sans snapshot de prix: rejeu strict non garanti.")

        if df is not None and not df.empty and snapshot_mode_requested in ("csv_full", "csv_downsampled", "parquet_zstd"):
            df_out = df.copy()
            if snapshot_mode_requested == "csv_downsampled":
                df_out = _downsample_df(df_out, max_rows=df_max_rows)
            try:
                if snapshot_mode_requested == "parquet_zstd":
                    # Compact snapshot format for large market datasets.
                    buf = io.BytesIO()
                    df_out.to_parquet(buf, compression="zstd")
                    zf.writestr("df.parquet", buf.getvalue())
                    snapshot_mode_actual = "parquet_zstd"
                    has_df_snapshot = True
                else:
                    df_out.index.name = "Open time"
                    zf.writestr("df.csv", df_out.to_csv())
                    snapshot_mode_actual = snapshot_mode_requested
                    has_df_snapshot = True
            except Exception as e:
                # Fallback to CSV if parquet dependencies are missing or serialization fails.
                df_out.index.name = "Open time"
                zf.writestr("df.csv", df_out.to_csv())
                snapshot_mode_actual = "csv_full" if snapshot_mode_requested == "parquet_zstd" else snapshot_mode_requested
                has_df_snapshot = True
                st.warning(f"Snapshot parquet indisponible, fallback CSV appliqué: {e}")

        if "final_portfolio" in st.session_state:
            pf = st.session_state["final_portfolio"]
            try:
                trades_df = pd.DataFrame(pf.trades.records)
                zf.writestr("final_trades.csv", trades_df.to_csv(index=False))
            except Exception:
                pass
            try:
                stats_df = pf.trades.stats().reset_index()
                stats_df.columns = ["metric", "value"]
                zf.writestr("final_trade_stats.csv", stats_df.to_csv(index=False))
            except Exception:
                pass

        replay_manifest = _build_replay_manifest(
            payload,
            snapshot_mode_requested=snapshot_mode_requested,
            snapshot_mode_actual=snapshot_mode_actual,
            has_df_snapshot=has_df_snapshot,
            data_source=data_source_descriptor
        )
        zf.writestr("replay_manifest.json", json.dumps(replay_manifest, indent=2))

    zip_buffer.seek(0)
    return zip_buffer

def _build_results_zip_filename(config):
    """
    Build ZIP filename using the same naming rule components as config export.
    Example:
    results_wfo_20260210_132530_01m_5s_bayes_05w_tr5000_classic.zip
    """
    cfg_name = _build_config_filename(config)
    stem = cfg_name.removeprefix("config_wfo_").removesuffix(".json")
    return f"results_wfo_{stem}.zip"


def _build_results_pdf_filename(config):
    """Build PDF filename aligned with config naming rule."""
    cfg_name = _build_config_filename(config)
    stem = cfg_name.removeprefix("config_wfo_").removesuffix(".json")
    return f"report_wfo_{stem}.pdf"


def _safe_series(values):
    if values is None:
        return pd.Series(dtype=float)
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    try:
        return pd.to_numeric(pd.Series(values), errors="coerce")
    except Exception:
        return pd.Series(dtype=float)


def _generate_wfo_pdf_report(results, config):
    """
    Generate a rich multi-page PDF report aligned with GUI result tabs.
    Includes optimization charts/tables, adaptive insights (if available),
    raw data tables, and final backtest analysis.
    """
    if not isinstance(results, dict):
        return None

    oos_df = pd.DataFrame(results.get("out_of_sample_performance", []) or [])
    is_df = pd.DataFrame(results.get("in_sample_performance", []) or [])
    best_params_df = pd.DataFrame(results.get("best_params", []) or [])
    window_results = results.get("window_results", []) or []
    traceability = results.get("traceability") or st.session_state.get("wfo_traceability")
    run_mode = str(
        results.get("mode")
        or results.get("settings", {}).get("optimization_regime")
        or config.get("optimization_regime", "classic")
    ).lower()
    market_df = st.session_state.get("df")
    final_backtest_df = st.session_state.get("final_backtest_df")
    if final_backtest_df is None or (isinstance(final_backtest_df, pd.DataFrame) and final_backtest_df.empty):
        final_backtest_df = market_df
    all_trials_df = st.session_state.get("all_trials_df")
    if all_trials_df is None or (isinstance(all_trials_df, pd.DataFrame) and all_trials_df.empty):
        all_trials_df = _build_trials_dataframe_from_results(results)
    window_info_df = st.session_state.get("window_info_df")
    if window_info_df is None or (isinstance(window_info_df, pd.DataFrame) and window_info_df.empty):
        window_info_df = _build_window_info_dataframe(results)
    robust_summary = results.get("robust_set_summary")
    if not isinstance(robust_summary, dict) or not robust_summary:
        robust_summary = _build_robust_set_summary(results, config)

    final_summary = _extract_final_backtest_for_expert(results, config, df_source=st.session_state.get("df"))

    def _safe_float(value):
        try:
            val = float(value)
            return val if np.isfinite(val) else np.nan
        except Exception:
            return np.nan

    def _fmt_num(value, ndigits=2, suffix=""):
        val = _safe_float(value)
        if np.isfinite(val):
            return f"{val:.{ndigits}f}{suffix}"
        return "n/a"

    def _coerce_datetime(value):
        try:
            ts = pd.to_datetime(value, errors="coerce")
            if pd.isna(ts):
                return None
            return ts
        except Exception:
            return None

    def _to_display_df(df_in, max_cell=90):
        if df_in is None or not isinstance(df_in, pd.DataFrame) or df_in.empty:
            return pd.DataFrame()
        df_out = df_in.copy()
        for col in df_out.columns:
            series = df_out[col]
            if pd.api.types.is_datetime64_any_dtype(series):
                df_out[col] = series.dt.strftime("%Y-%m-%d %H:%M:%S")
                continue
            if pd.api.types.is_numeric_dtype(series):
                continue

            def _cell_to_str(x):
                if isinstance(x, (dict, list, tuple, set)):
                    try:
                        txt = json.dumps(_to_jsonable(x), ensure_ascii=False)
                    except Exception:
                        txt = str(x)
                else:
                    txt = str(x)
                if len(txt) > max_cell:
                    return txt[: max_cell - 3] + "..."
                return txt

            df_out[col] = series.map(_cell_to_str)

        df_out = df_out.replace({np.nan: "", np.inf: "inf", -np.inf: "-inf"})
        return df_out

    def _add_text_page(pdf, title, lines):
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_subplot(111)
        ax.axis("off")
        fig.suptitle(title, fontsize=16, y=0.98)
        ax.text(0.02, 0.96, "\n".join(lines), va="top", ha="left", fontsize=10)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def _add_table_pages(pdf, title, df_in, rows_per_page=26, cols_per_page=9):
        df_show = _to_display_df(df_in)
        if df_show.empty:
            return

        columns = list(df_show.columns)
        for col_start in range(0, len(columns), cols_per_page):
            cols_chunk = columns[col_start: col_start + cols_per_page]
            df_col = df_show[cols_chunk]
            for row_start in range(0, len(df_col), rows_per_page):
                chunk = df_col.iloc[row_start: row_start + rows_per_page]
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                ax.axis("off")
                title_suffix = (
                    f"rows {row_start + 1}-{row_start + len(chunk)} / {len(df_col)}"
                    f", cols {col_start + 1}-{col_start + len(cols_chunk)} / {len(columns)}"
                )
                ax.set_title(f"{title}\n({title_suffix})", fontsize=12, pad=10)
                table = ax.table(
                    cellText=chunk.values,
                    colLabels=[str(c) for c in cols_chunk],
                    loc="center",
                    cellLoc="left",
                    colLoc="left",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(7.5)
                table.scale(1.0, 1.25)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    def _series_from_price_df(df_in):
        if not isinstance(df_in, pd.DataFrame) or df_in.empty:
            return pd.Series(dtype=float)
        if "Close" in df_in.columns:
            out = pd.to_numeric(df_in["Close"], errors="coerce")
        elif "close" in df_in.columns:
            out = pd.to_numeric(df_in["close"], errors="coerce")
        else:
            out = pd.to_numeric(df_in.iloc[:, 0], errors="coerce")
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out[~out.index.isna()].dropna()
        return out

    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        # 1) Executive summary
        fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        fig.suptitle("WFO Engine Report", fontsize=18, y=0.98)
        ax = fig.add_subplot(111)
        ax.axis("off")

        mode = str(config.get("optimization_regime", run_mode))
        method = str(config.get("optimization_method", "grid"))
        timeframe = str(config.get("timeframe", "n/a"))
        period = f"{config.get('start_date', 'n/a')} -> {config.get('end_date', 'n/a')}"
        n_windows = int(config.get("n_windows", 0) or 0)

        oos_return_mean = np.nan
        oos_sharpe_mean = np.nan
        if not oos_df.empty:
            if "return" in oos_df.columns:
                oos_return_mean = float(pd.to_numeric(oos_df["return"], errors="coerce").mean())
            if "sharpe" in oos_df.columns:
                oos_sharpe_mean = float(pd.to_numeric(oos_df["sharpe"], errors="coerce").mean())

        summary_lines = [
            f"Run date (UTC): {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Mode: {mode}",
            f"Method: {method}",
            f"Timeframe: {timeframe}",
            f"Period: {period}",
            f"Configured windows: {n_windows}",
            f"Computed window results: {len(window_results)}",
            f"OOS mean return: {oos_return_mean:.2f}%" if np.isfinite(oos_return_mean) else "OOS mean return: n/a",
            f"OOS mean sharpe: {oos_sharpe_mean:.2f}" if np.isfinite(oos_sharpe_mean) else "OOS mean sharpe: n/a",
            f"Best params rows: {len(best_params_df)}",
            f"All trials rows: {len(all_trials_df) if isinstance(all_trials_df, pd.DataFrame) else 0}",
        ]
        if isinstance(robust_summary, dict):
            summary_lines.extend([
                f"Robust set status: {robust_summary.get('status', 'n/a')}",
                f"Robust Top-N / window: {robust_summary.get('top_n_per_window', 'n/a')}",
                (
                    "Robust windows used: "
                    f"{len(robust_summary.get('windows_used', []) or [])}"
                    f"/{robust_summary.get('min_windows_required', 'n/a')}"
                ),
                f"Robust candidates pool: {robust_summary.get('candidates_total', 0)}",
            ])

        fb_ret = final_summary.get("strategy_total_return_pct")
        fb_sharpe = final_summary.get("strategy_sharpe")
        fb_dd = final_summary.get("strategy_max_drawdown_pct")
        fb_outperf = final_summary.get("outperformance_vs_buy_hold_pct")
        summary_lines.extend(
            [
                "",
                "Final backtest:",
                f"- Return strategy: {fb_ret:.2f}%" if isinstance(fb_ret, (int, float)) and np.isfinite(fb_ret) else "- Return strategy: n/a",
                f"- Sharpe: {fb_sharpe:.2f}" if isinstance(fb_sharpe, (int, float)) and np.isfinite(fb_sharpe) else "- Sharpe: n/a",
                f"- Max drawdown: {fb_dd:.2f}%" if isinstance(fb_dd, (int, float)) and np.isfinite(fb_dd) else "- Max drawdown: n/a",
                f"- Outperformance vs buy&hold: {fb_outperf:.2f} pts"
                if isinstance(fb_outperf, (int, float)) and np.isfinite(fb_outperf)
                else "- Outperformance vs buy&hold: n/a",
            ]
        )
        if isinstance(traceability, dict):
            run_meta = traceability.get("run", {})
            summary_lines.extend([
                "",
                "Traceability:",
                f"- Run ID: {run_meta.get('run_id', 'n/a')}",
                f"- Status: {run_meta.get('status', 'n/a')}",
                f"- Config SHA256: {str(run_meta.get('config_sha256', 'n/a'))[:18]}...",
            ])

        ax.text(0.02, 0.95, "\n".join(summary_lines), va="top", ha="left", fontsize=11)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 2) Price + WFO windows (same spirit as OOS tab chart)
        if isinstance(market_df, pd.DataFrame) and not market_df.empty:
            from matplotlib.patches import Patch

            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            price_series = _series_from_price_df(market_df)
            price_series = _downsample_series(price_series, max_points=4000)
            if not price_series.empty:
                ax.plot(price_series.index, price_series.values, color="#1f77b4", linewidth=0.9, label="Price")

            first_train = True
            first_test = True
            for wr in window_results:
                info = wr.get("window_info", {}) if isinstance(wr, dict) else {}
                is_start = _coerce_datetime(info.get("in_sample_start"))
                is_end = _coerce_datetime(info.get("in_sample_end"))
                oos_start = _coerce_datetime(info.get("out_sample_start"))
                oos_end = _coerce_datetime(info.get("out_sample_end"))
                if is_start is not None and is_end is not None:
                    ax.axvspan(
                        is_start,
                        is_end,
                        color="green",
                        alpha=0.08,
                        label="Train (IS)" if first_train else None,
                    )
                    first_train = False
                if oos_start is not None and oos_end is not None:
                    ax.axvspan(
                        oos_start,
                        oos_end,
                        color="red",
                        alpha=0.08,
                        label="Test (OOS)" if first_test else None,
                    )
                    first_test = False

            handles, labels = ax.get_legend_handles_labels()
            if not handles:
                handles = [
                    Patch(facecolor="green", edgecolor="none", alpha=0.12, label="Train (IS)"),
                    Patch(facecolor="red", edgecolor="none", alpha=0.12, label="Test (OOS)"),
                ]
                labels = [h.get_label() for h in handles]
            ax.set_title("Price Series with Walk-Forward Windows")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.grid(alpha=0.25)
            ax.legend(handles, labels, loc="best")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # 3) IS/OOS performance by window
        if not oos_df.empty or not is_df.empty:
            fig, (ax_r, ax_s) = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=True)
            fig.suptitle("In-Sample vs Out-of-Sample Performance by Window", fontsize=16, y=0.98)

            is_tmp = is_df.copy()
            oos_tmp = oos_df.copy()
            if "window" in is_tmp.columns:
                is_tmp["window"] = pd.to_numeric(is_tmp["window"], errors="coerce")
            if "window" in oos_tmp.columns:
                oos_tmp["window"] = pd.to_numeric(oos_tmp["window"], errors="coerce")

            windows = sorted(
                set(is_tmp.get("window", pd.Series(dtype=float)).dropna().tolist())
                | set(oos_tmp.get("window", pd.Series(dtype=float)).dropna().tolist())
            )
            x = np.array(windows, dtype=float) if windows else np.array([], dtype=float)
            w = 0.36

            if len(x) > 0:
                if {"window", "return"}.issubset(is_tmp.columns):
                    is_ret = pd.to_numeric(
                        is_tmp.set_index("window").reindex(x)["return"], errors="coerce"
                    ).to_numpy()
                    ax_r.bar(x - w / 2, is_ret, width=w, label="IS Return %", alpha=0.78, color="#ff7f0e")
                if {"window", "return"}.issubset(oos_tmp.columns):
                    oos_ret = pd.to_numeric(
                        oos_tmp.set_index("window").reindex(x)["return"], errors="coerce"
                    ).to_numpy()
                    ax_r.bar(x + w / 2, oos_ret, width=w, label="OOS Return %", alpha=0.86, color="#37536D")
                ax_r.set_ylabel("Return %")
                ax_r.grid(axis="y", alpha=0.25)
                ax_r.legend(loc="best")

                if {"window", "sharpe"}.issubset(is_tmp.columns):
                    is_sh = pd.to_numeric(
                        is_tmp.set_index("window").reindex(x)["sharpe"], errors="coerce"
                    ).to_numpy()
                    ax_s.plot(x, is_sh, marker="o", label="IS Sharpe", color="#d62728")
                if {"window", "sharpe"}.issubset(oos_tmp.columns):
                    oos_sh = pd.to_numeric(
                        oos_tmp.set_index("window").reindex(x)["sharpe"], errors="coerce"
                    ).to_numpy()
                    ax_s.plot(x, oos_sh, marker="o", label="OOS Sharpe", color="#1a76ff")
                ax_s.set_xlabel("Window")
                ax_s.set_ylabel("Sharpe")
                ax_s.grid(alpha=0.25)
                ax_s.legend(loc="best")
            else:
                ax_r.text(0.5, 0.5, "Window metrics unavailable", ha="center", va="center")
                ax_s.text(0.5, 0.5, "Window metrics unavailable", ha="center", va="center")
                ax_r.set_axis_off()
                ax_s.set_axis_off()

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # 4) Parameter stability heatmap + parameter table
        if not best_params_df.empty:
            selected_for_opt = []
            if isinstance(traceability, dict):
                cfg_trace = traceability.get("config")
                if isinstance(cfg_trace, dict):
                    selected_for_opt = cfg_trace.get("selected_params") or []
            if not selected_for_opt and isinstance(config, dict):
                selected_for_opt = config.get("selected_params") or []
            if not selected_for_opt:
                selected_for_opt = st.session_state.get("selected_params", [])

            numeric_cols = best_params_df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c not in ["window", "metric1_name", "metric2_name"]]
            if selected_for_opt:
                numeric_cols = [c for c in numeric_cols if c in selected_for_opt]
            varying_cols = [
                c for c in numeric_cols
                if pd.to_numeric(best_params_df[c], errors="coerce").nunique(dropna=True) > 1
            ]

            if varying_cols:
                ndf = best_params_df[varying_cols].copy()
                ndf = ndf.apply(pd.to_numeric, errors="coerce")
                for col in varying_cols:
                    vmin = ndf[col].min()
                    vmax = ndf[col].max()
                    if pd.notna(vmin) and pd.notna(vmax) and vmax > vmin:
                        ndf[col] = (ndf[col] - vmin) / (vmax - vmin)
                    else:
                        ndf[col] = 0.5

                heat = ndf.T.values
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                im = ax.imshow(heat, aspect="auto", cmap="viridis", interpolation="nearest")
                ax.set_title("Parameter Stability (Normalized across windows)")
                ax.set_xlabel("Window index")
                ax.set_ylabel("Parameter")
                ax.set_yticks(range(len(varying_cols)))
                ax.set_yticklabels(varying_cols, fontsize=8)
                ax.set_xticks(range(len(ndf)))
                ax.set_xticklabels([str(i + 1) for i in range(len(ndf))], fontsize=8)
                cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
                cbar.set_label("Normalized value")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            params_table_df = best_params_df.copy()
            if not params_table_df.empty:
                params_table_df.insert(0, "Window", np.arange(1, len(params_table_df) + 1))
                _add_table_pages(pdf, "Best Parameters Used for Backtesting in Each WFO Window", params_table_df)
            if isinstance(robust_summary, dict):
                robust_params = robust_summary.get("robust_params", {}) or {}
                support_rows = robust_summary.get("support_by_param", []) or []
                if robust_params:
                    robust_params_df = pd.DataFrame(
                        [{"parameter": k, "robust_value": v} for k, v in robust_params.items()]
                    ).sort_values("parameter")
                    _add_table_pages(pdf, "Robust Set - Selected Parameters", robust_params_df)
                if support_rows:
                    _add_table_pages(pdf, "Robust Set - Support by Parameter", pd.DataFrame(support_rows))

        # 5) Drawdowns & returns distributions
        if not oos_df.empty:
            fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
            fig.suptitle("Drawdowns & Returns Distributions (OOS)", fontsize=15, y=0.98)

            if "win_rate" in oos_df.columns:
                win = pd.to_numeric(oos_df["win_rate"], errors="coerce").dropna()
                axes[0].hist(win, bins=12, color="#4E79A7", edgecolor="white")
                axes[0].set_title("Win Rate Distribution")
                axes[0].set_xlabel("Win Rate (%)")
                axes[0].set_ylabel("Count")
                axes[0].grid(alpha=0.25)
            else:
                axes[0].text(0.5, 0.5, "win_rate unavailable", ha="center", va="center")
                axes[0].set_axis_off()

            if "max_drawdown" in oos_df.columns:
                dd = pd.to_numeric(oos_df["max_drawdown"], errors="coerce").dropna()
                axes[1].hist(dd, bins=12, color="#E15759", edgecolor="white")
                axes[1].set_title("Max Drawdown Distribution")
                axes[1].set_xlabel("Max Drawdown (%)")
                axes[1].set_ylabel("Count")
                axes[1].grid(alpha=0.25)
            else:
                axes[1].text(0.5, 0.5, "max_drawdown unavailable", ha="center", va="center")
                axes[1].set_axis_off()

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # 6) Adaptive insights (same family as GUI tab)
        guidance_df = pd.DataFrame(results.get("adaptive_guidance", []) or [])
        if run_mode == "adaptive_continuous":
            if guidance_df.empty and window_results:
                fallback_rows = []
                baseline_default = (
                    results.get("settings", {}).get("baseline_param_combinations")
                    or results.get("timing", {}).get("param_combinations")
                )
                for wr in window_results:
                    cycle_info = wr.get("cycle_info", {}) if isinstance(wr, dict) else {}
                    w_info = wr.get("window_info", {}) if isinstance(wr, dict) else {}
                    cycle_id = cycle_info.get("cycle", w_info.get("window"))
                    fallback_rows.append({
                        "window": cycle_id,
                        "cycle": cycle_id,
                        "baseline_combinations": baseline_default,
                        "active_combinations": cycle_info.get("active_combinations"),
                        "trials_tested": wr.get("optimization_trials_count", cycle_info.get("trials_tested")),
                        "parameter_weights": {},
                    })
                guidance_df = pd.DataFrame(fallback_rows)

            if isinstance(all_trials_df, pd.DataFrame) and not all_trials_df.empty and {"window", "combined_score"}.issubset(all_trials_df.columns):
                conv_df = all_trials_df[["window", "combined_score"]].copy()
                conv_df["window"] = pd.to_numeric(conv_df["window"], errors="coerce")
                conv_df["combined_score"] = pd.to_numeric(conv_df["combined_score"], errors="coerce")
                conv_df = conv_df.dropna(subset=["window", "combined_score"])
                if not conv_df.empty:
                    conv = conv_df.groupby("window")["combined_score"].agg(
                        best="max",
                        median="median",
                        q25=lambda x: x.quantile(0.25),
                        q75=lambda x: x.quantile(0.75),
                    ).reset_index().sort_values("window")
                    fig, ax = plt.subplots(figsize=(11.69, 5.8))
                    ax.fill_between(conv["window"], conv["q25"], conv["q75"], alpha=0.20, color="#636efa", label="IQR Q25-Q75")
                    ax.plot(conv["window"], conv["best"], marker="o", color="#00CC96", label="Best score")
                    ax.plot(conv["window"], conv["median"], marker="o", linestyle="--", color="#FECB52", label="Median score")
                    ax.set_title("Adaptive Insights - Convergence des scores par cycle")
                    ax.set_xlabel("Cycle")
                    ax.set_ylabel("combined_score")
                    ax.grid(alpha=0.25)
                    ax.legend(loc="best")
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

            if not is_df.empty and not oos_df.empty and "window" in is_df.columns and "window" in oos_df.columns:
                gap_df = pd.merge(
                    is_df[["window", "return", "sharpe"]].rename(columns={"return": "is_return", "sharpe": "is_sharpe"}),
                    oos_df[["window", "return", "sharpe"]].rename(columns={"return": "oos_return", "sharpe": "oos_sharpe"}),
                    on="window",
                    how="inner",
                )
                gap_df["window"] = pd.to_numeric(gap_df["window"], errors="coerce")
                for col in ["is_return", "oos_return", "is_sharpe", "oos_sharpe"]:
                    gap_df[col] = pd.to_numeric(gap_df[col], errors="coerce")
                gap_df = gap_df.dropna(subset=["window"]).sort_values("window")
                if not gap_df.empty:
                    gap_df["return_gap"] = gap_df["is_return"] - gap_df["oos_return"]
                    gap_df["sharpe_gap"] = gap_df["is_sharpe"] - gap_df["oos_sharpe"]
                    fig, ax1 = plt.subplots(figsize=(11.69, 5.8))
                    ax2 = ax1.twinx()
                    ax1.bar(gap_df["window"], gap_df["return_gap"], alpha=0.70, color="#ef553b", label="Gap Return")
                    ax2.plot(gap_df["window"], gap_df["sharpe_gap"], marker="o", color="#19D3F3", label="Gap Sharpe")
                    ax1.axhline(0, linewidth=1, linestyle=":", color="#999999")
                    ax1.set_title("Adaptive Insights - Gap IS vs OOS")
                    ax1.set_xlabel("Cycle")
                    ax1.set_ylabel("Gap Return (IS - OOS)")
                    ax2.set_ylabel("Gap Sharpe (IS - OOS)")
                    ax1.grid(alpha=0.20)
                    h1, l1 = ax1.get_legend_handles_labels()
                    h2, l2 = ax2.get_legend_handles_labels()
                    ax1.legend(h1 + h2, l1 + l2, loc="best")
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

            if not guidance_df.empty:
                gdf = guidance_df.copy()
                gdf["cycle"] = pd.to_numeric(
                    gdf["cycle"] if "cycle" in gdf.columns else gdf.get("window"),
                    errors="coerce",
                )
                gdf["active_combinations"] = pd.to_numeric(gdf.get("active_combinations"), errors="coerce")
                gdf["baseline_combinations"] = pd.to_numeric(gdf.get("baseline_combinations"), errors="coerce")
                gdf["trials_tested"] = pd.to_numeric(gdf.get("trials_tested"), errors="coerce")
                gdf = gdf.dropna(subset=["cycle"]).sort_values("cycle")
                if not gdf.empty:
                    fig, ax1 = plt.subplots(figsize=(11.69, 5.8))
                    ax2 = ax1.twinx()
                    ax1.bar(gdf["cycle"], gdf["trials_tested"], alpha=0.70, color="#FECB52", label="Trials testes")
                    ax2.plot(gdf["cycle"], gdf["active_combinations"], marker="o", color="#00CC96", label="Active combinations")
                    if gdf["baseline_combinations"].notna().any():
                        ax2.plot(gdf["cycle"], gdf["baseline_combinations"], linestyle="--", color="#AB63FA", label="Baseline combinations")
                    if (gdf["active_combinations"] > 0).any():
                        ax2.set_yscale("log")
                    ax1.set_title("Adaptive Insights - Effort de test vs taille de grille")
                    ax1.set_xlabel("Cycle")
                    ax1.set_ylabel("Trials")
                    ax2.set_ylabel("Combinations")
                    ax1.grid(alpha=0.22)
                    h1, l1 = ax1.get_legend_handles_labels()
                    h2, l2 = ax2.get_legend_handles_labels()
                    ax1.legend(h1 + h2, l1 + l2, loc="best")
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

                weight_rows = []
                for _, row in gdf.iterrows():
                    weights = row.get("parameter_weights", {})
                    if isinstance(weights, str):
                        try:
                            weights = json.loads(weights)
                        except Exception:
                            weights = {}
                    if not isinstance(weights, dict):
                        continue
                    for pname, pweight in weights.items():
                        weight_rows.append({
                            "cycle": row.get("cycle"),
                            "parameter": str(pname),
                            "weight": _safe_float(pweight),
                        })
                weights_df = pd.DataFrame(weight_rows)
                if not weights_df.empty:
                    weights_df = weights_df.dropna(subset=["cycle", "weight"])
                    top_params = (
                        weights_df.groupby("parameter")["weight"].mean().sort_values(ascending=False).head(8).index.tolist()
                    )
                    weights_df = weights_df[weights_df["parameter"].isin(top_params)].sort_values("cycle")
                    fig, ax = plt.subplots(figsize=(11.69, 5.8))
                    for p in top_params:
                        sub = weights_df[weights_df["parameter"] == p]
                        ax.plot(sub["cycle"], sub["weight"], marker="o", linewidth=1.4, label=p)
                    ax.set_title("Adaptive Insights - Evolution des poids parametres (top 8)")
                    ax.set_xlabel("Cycle")
                    ax.set_ylabel("Poids relatif")
                    ax.grid(alpha=0.25)
                    ax.legend(loc="upper left", ncols=2, fontsize=8)
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

                _add_table_pages(pdf, "Adaptive Guidance Table", gdf)

            top_values = (results.get("adaptive_summary") or {}).get("top_values_by_parameter", {})
            if isinstance(top_values, dict) and top_values:
                flat_rows = []
                for pname, rows in top_values.items():
                    if not isinstance(rows, list):
                        continue
                    for rank, row in enumerate(rows, start=1):
                        if not isinstance(row, dict):
                            continue
                        flat_rows.append({
                            "parameter": pname,
                            "rank": rank,
                            "value": row.get("value"),
                            "mean_score": row.get("mean_score"),
                            "effective_trials": row.get("effective_trials"),
                        })
                _add_table_pages(pdf, "Adaptive Top Values by Parameter", pd.DataFrame(flat_rows))

        # 7) Raw tables (GUI tab Raw Data)
        _add_table_pages(pdf, "Out-of-Sample Metrics", oos_df)
        _add_table_pages(pdf, "In-Sample Metrics", is_df)
        if isinstance(window_info_df, pd.DataFrame) and not window_info_df.empty:
            _add_table_pages(pdf, "Window Info", window_info_df)
        if isinstance(all_trials_df, pd.DataFrame) and not all_trials_df.empty:
            trials_for_pdf = all_trials_df.copy()
            if "combined_score" in trials_for_pdf.columns:
                trials_for_pdf["combined_score"] = pd.to_numeric(trials_for_pdf["combined_score"], errors="coerce")
                trials_for_pdf = trials_for_pdf.sort_values("combined_score", ascending=False)
            _add_table_pages(pdf, "All Trials (sorted by combined_score desc)", trials_for_pdf)

        # 8) Final backtest: summary and curve
        final_params = st.session_state.get("final_params")
        final_score = st.session_state.get("final_params_score")
        final_window = st.session_state.get("final_params_window")
        final_is_score = st.session_state.get("final_params_is_score")
        final_oos_score = st.session_state.get("final_params_oos_score")
        final_is_metrics = st.session_state.get("final_params_is_metrics")
        final_oos_metrics = st.session_state.get("final_params_oos_metrics")
        final_params_source = str(st.session_state.get("final_params_source", "best_window"))
        final_lines = [
            "Final backtest summary:",
            f"- Strategy return: {_fmt_num(fb_ret, 2, '%')}",
            f"- Sharpe: {_fmt_num(fb_sharpe)}",
            f"- Max drawdown: {_fmt_num(fb_dd, 2, '%')}",
            f"- Outperformance vs buy&hold: {_fmt_num(fb_outperf, 2, ' pts')}",
            f"- Parameter source: {final_params_source}",
            f"- Best optimization score (combined_score): {_fmt_num(final_score, 4)}",
            f"- Best window: {final_window if final_window is not None else 'n/a'}",
            f"- IS combined score: {_fmt_num(final_is_score, 4)}",
            f"- OOS combined score: {_fmt_num(final_oos_score, 4)}",
        ]
        if isinstance(final_is_metrics, dict):
            final_lines.append(
                "- IS metrics: "
                f"window={final_is_metrics.get('window', 'n/a')}, "
                f"ret={_fmt_num(final_is_metrics.get('return'), 2, '%')}, "
                f"sharpe={_fmt_num(final_is_metrics.get('sharpe'))}, "
                f"dd={_fmt_num(final_is_metrics.get('max_drawdown'), 2, '%')}, "
                f"wr={_fmt_num(final_is_metrics.get('win_rate'), 2, '%')}, "
                f"trades={final_is_metrics.get('n_trades', 'n/a')}"
            )
        if isinstance(final_oos_metrics, dict):
            final_lines.append(
                "- OOS metrics: "
                f"window={final_oos_metrics.get('window', 'n/a')}, "
                f"ret={_fmt_num(final_oos_metrics.get('return'), 2, '%')}, "
                f"sharpe={_fmt_num(final_oos_metrics.get('sharpe'))}, "
                f"dd={_fmt_num(final_oos_metrics.get('max_drawdown'), 2, '%')}, "
                f"wr={_fmt_num(final_oos_metrics.get('win_rate'), 2, '%')}, "
                f"trades={final_oos_metrics.get('n_trades', 'n/a')}"
            )
        if isinstance(final_params, dict):
            final_lines.append("- Used parameters:")
            final_lines.append(json.dumps(_to_jsonable(final_params), ensure_ascii=False))
        _add_text_page(pdf, "Final Backtest Results", final_lines)

        fig = plt.figure(figsize=(11.69, 8.27))
        gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0])
        ax_curve = fig.add_subplot(gs[0, 0])
        ax_text = fig.add_subplot(gs[1, 0])
        ax_text.axis("off")

        curve_plotted = False
        pf = st.session_state.get("final_portfolio")
        price_df = final_backtest_df
        if pf is not None and isinstance(price_df, pd.DataFrame) and not price_df.empty:
            try:
                value_series = pf.value() if callable(getattr(pf, "value", None)) else pf.value
                value_series = _safe_series(value_series).dropna()
                value_series.index = pd.to_datetime(value_series.index, errors="coerce")
                value_series = value_series[~value_series.index.isna()]

                close_col = "Close" if "Close" in price_df.columns else ("close" if "close" in price_df.columns else None)
                if close_col is not None and not value_series.empty:
                    price_series = pd.to_numeric(price_df[close_col], errors="coerce").dropna()
                    price_series.index = pd.to_datetime(price_series.index, errors="coerce")
                    price_series = price_series[~price_series.index.isna()]

                    idx = value_series.index.intersection(price_series.index)
                    if len(idx) > 2:
                        value_aligned = value_series.reindex(idx).dropna()
                        price_aligned = price_series.reindex(idx).dropna()
                        idx2 = value_aligned.index.intersection(price_aligned.index)
                        value_aligned = value_aligned.reindex(idx2)
                        price_aligned = price_aligned.reindex(idx2)
                        if len(idx2) > 2:
                            initial_capital = float(config.get("order_fixed_cash", 10000.0))
                            p0 = float(price_aligned.iloc[0]) if float(price_aligned.iloc[0]) != 0 else np.nan
                            if np.isfinite(p0):
                                buy_hold = initial_capital * (price_aligned / p0)
                                value_plot = _downsample_series(value_aligned, max_points=3000)
                                buy_hold_plot = _downsample_series(buy_hold, max_points=3000)
                                price_plot = _downsample_series(price_aligned, max_points=3000)
                                ax_curve.plot(value_plot.index, value_plot.values, label="Portfolio Value")
                                ax_curve.plot(buy_hold_plot.index, buy_hold_plot.values, label="Buy & Hold")
                                ax_price = ax_curve.twinx()
                                ax_price.plot(price_plot.index, price_plot.values, label="Asset Price", color="#ff7f0e", linewidth=1.0, alpha=0.85)
                                ax_price.set_ylabel("Asset Price")
                                ax_curve.set_title("Final Backtest: Portfolio vs Buy & Hold")
                                ax_curve.set_xlabel("Date")
                                ax_curve.set_ylabel("Value")
                                ax_curve.grid(alpha=0.3)
                                h1, l1 = ax_curve.get_legend_handles_labels()
                                h2, l2 = ax_price.get_legend_handles_labels()
                                ax_curve.legend(h1 + h2, l1 + l2, loc="best")
                                curve_plotted = True
            except Exception:
                curve_plotted = False

        if not curve_plotted:
            ax_curve.text(0.5, 0.5, "Final backtest curve unavailable", ha="center", va="center")
            ax_curve.set_axis_off()

        curve_summary_lines = [
            "Final backtest summary:",
            f"- Strategy return: {fb_ret:.2f}%"
            if isinstance(fb_ret, (int, float)) and np.isfinite(fb_ret) else "- Strategy return: n/a",
            f"- Sharpe: {fb_sharpe:.2f}"
            if isinstance(fb_sharpe, (int, float)) and np.isfinite(fb_sharpe) else "- Sharpe: n/a",
            f"- Max drawdown: {fb_dd:.2f}%"
            if isinstance(fb_dd, (int, float)) and np.isfinite(fb_dd) else "- Max drawdown: n/a",
            f"- Outperformance vs buy&hold: {fb_outperf:.2f} pts"
            if isinstance(fb_outperf, (int, float)) and np.isfinite(fb_outperf) else "- Outperformance vs buy&hold: n/a",
        ]
        ax_text.text(0.01, 0.95, "\n".join(curve_summary_lines), va="top", ha="left", fontsize=11)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 9) Final backtest trade tables and diagnostics
        final_trades_df = pd.DataFrame()
        final_trade_stats_df = pd.DataFrame()
        if pf is not None:
            try:
                final_trades_df = pd.DataFrame(pf.trades.records)
            except Exception:
                final_trades_df = pd.DataFrame()
            try:
                stats_raw = pf.trades.stats()
                if isinstance(stats_raw, pd.Series):
                    final_trade_stats_df = stats_raw.rename_axis("metric").reset_index(name="value")
                elif isinstance(stats_raw, pd.DataFrame):
                    final_trade_stats_df = stats_raw.reset_index()
            except Exception:
                final_trade_stats_df = pd.DataFrame()
        if final_trade_stats_df.empty and isinstance(st.session_state.get("final_trade_stats_df"), pd.DataFrame):
            final_trade_stats_df = st.session_state.get("final_trade_stats_df")
        if final_trades_df.empty and isinstance(st.session_state.get("final_trades_df"), pd.DataFrame):
            final_trades_df = st.session_state.get("final_trades_df")

        if not final_trade_stats_df.empty:
            _add_table_pages(pdf, "Final Backtest - Trade Stats", final_trade_stats_df, rows_per_page=34, cols_per_page=6)

        if not final_trades_df.empty:
            pnl_metrics_df = _compute_trade_pnl_metrics(final_trades_df, trim=0.05)
            if isinstance(pnl_metrics_df, pd.DataFrame) and not pnl_metrics_df.empty:
                _add_table_pages(pdf, "Final Backtest - Average P&L per Trade", pnl_metrics_df, rows_per_page=34, cols_per_page=7)

            # Time-of-day/day-of-week charts
            trades_time = final_trades_df.copy()
            had_trades = not trades_time.empty
            if had_trades:
                if "entry_ts" in trades_time.columns:
                    trades_time["entry_ts"] = pd.to_datetime(trades_time["entry_ts"], errors="coerce")
                elif "entry_idx" in trades_time.columns and pf is not None and hasattr(pf, "wrapper"):
                    try:
                        entry_index = pf.wrapper.index
                        trades_time["entry_ts"] = pd.to_datetime(entry_index.take(trades_time["entry_idx"].to_numpy()), errors="coerce")
                    except Exception:
                        trades_time["entry_ts"] = pd.NaT
                else:
                    trades_time["entry_ts"] = pd.NaT
                trades_time = trades_time.dropna(subset=["entry_ts"])

            if not trades_time.empty and "pnl" in trades_time.columns:
                trades_time["day_of_week"] = trades_time["entry_ts"].dt.day_name()
                trades_time["hour_of_day"] = trades_time["entry_ts"].dt.hour
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                pnl_by_time = trades_time.groupby(["day_of_week", "hour_of_day"])["pnl"].sum().unstack(fill_value=0)
                pnl_by_time = pnl_by_time.reindex(day_order)

                fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), gridspec_kw={"height_ratios": [1.2, 1.0]})
                fig.suptitle("Final Backtest - P&L by Time", fontsize=15, y=0.98)

                im = axes[0, 0].imshow(pnl_by_time.values, aspect="auto", cmap="RdYlGn")
                axes[0, 0].set_title("Total PnL by Day/Hour")
                axes[0, 0].set_yticks(np.arange(len(pnl_by_time.index)))
                axes[0, 0].set_yticklabels(list(pnl_by_time.index), fontsize=8)
                axes[0, 0].set_xticks(np.arange(len(pnl_by_time.columns)))
                axes[0, 0].set_xticklabels([str(c) for c in pnl_by_time.columns], fontsize=7)
                axes[0, 0].set_xlabel("Hour")
                axes[0, 0].set_ylabel("Day")
                fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

                pnl_by_day = trades_time.groupby("day_of_week")["pnl"].sum().reindex(day_order)
                axes[0, 1].bar(pnl_by_day.index, pnl_by_day.values, color="#6BAED6")
                axes[0, 1].set_title("Total PnL per Day")
                axes[0, 1].tick_params(axis="x", rotation=35)
                axes[0, 1].grid(axis="y", alpha=0.25)

                pnl_by_hour = trades_time.groupby("hour_of_day")["pnl"].sum()
                axes[1, 0].bar(pnl_by_hour.index.astype(int), pnl_by_hour.values, color="#9ecae1")
                axes[1, 0].set_title("Total PnL per Hour")
                axes[1, 0].set_xlabel("Hour")
                axes[1, 0].grid(axis="y", alpha=0.25)

                axes[1, 1].axis("off")
                axes[1, 1].text(
                    0.02,
                    0.95,
                    (
                        f"Trades analyzed: {len(trades_time):,}\n"
                        f"Best day pnl: {_fmt_num(pnl_by_day.max())}\n"
                        f"Worst day pnl: {_fmt_num(pnl_by_day.min())}\n"
                        f"Best hour pnl: {_fmt_num(pnl_by_hour.max())}\n"
                        f"Worst hour pnl: {_fmt_num(pnl_by_hour.min())}"
                    ),
                    va="top",
                    ha="left",
                    fontsize=10,
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            _add_table_pages(pdf, "Final Backtest - Trades", final_trades_df)

        # 10) Rolling metrics page
        if pf is not None:
            try:
                returns = pf.returns() if callable(getattr(pf, "returns", None)) else pf.returns
                returns = _safe_series(returns).dropna()
                returns = _downsample_series(returns, max_points=7000)
                rolling_window = 30
                if len(returns) > rolling_window:
                    rolling_returns = ((1 + returns).rolling(window=rolling_window).apply(np.prod, raw=True) - 1) * 100
                    rolling_mean = returns.rolling(window=rolling_window).mean()
                    rolling_std = returns.rolling(window=rolling_window).std(ddof=0)
                    rolling_sharpe = rolling_mean.divide(rolling_std).multiply(np.sqrt(rolling_window))
                    rolling_returns = rolling_returns.dropna()
                    rolling_sharpe = rolling_sharpe.dropna()
                    if not rolling_returns.empty and not rolling_sharpe.empty:
                        fig, ax1 = plt.subplots(figsize=(11.69, 5.8))
                        ax2 = ax1.twinx()
                        ax1.plot(rolling_returns.index, rolling_returns.values, color="#1f77b4", label="Rolling Returns (%)")
                        ax2.plot(rolling_sharpe.index, rolling_sharpe.values, color="#ff7f0e", label="Rolling Sharpe")
                        ax1.set_title(f"Final Backtest - {rolling_window}-Period Rolling Performance")
                        ax1.set_xlabel("Date")
                        ax1.set_ylabel("Rolling Returns (%)")
                        ax2.set_ylabel("Rolling Sharpe")
                        ax1.grid(alpha=0.22)
                        h1, l1 = ax1.get_legend_handles_labels()
                        h2, l2 = ax2.get_legend_handles_labels()
                        ax1.legend(h1 + h2, l1 + l2, loc="best")
                        pdf.savefig(fig, bbox_inches="tight")
                        plt.close(fig)
            except Exception:
                pass

    pdf_buffer.seek(0)
    return pdf_buffer


def _save_results_zip_to_disk(zip_buffer, config):
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    filename = _build_results_zip_filename(config)
    path = os.path.join(reports_dir, filename)
    path = os.path.abspath(path)
    with open(path, "wb") as f:
        f.write(zip_buffer.getvalue())
    return path


def _save_results_pdf_to_disk(pdf_buffer, config):
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    filename = _build_results_pdf_filename(config)
    path = os.path.join(reports_dir, filename)
    path = os.path.abspath(path)
    with open(path, "wb") as f:
        f.write(pdf_buffer.getvalue())
    return path


def _load_results_zip(zip_file):
    try:
        # Reset optional imported context to avoid stale cross-run usage.
        st.session_state.pop("expert_context_pack", None)
        st.session_state.pop("pending_strategy_id", None)
        st.session_state.pop("pine_precheck_report", None)
        st.session_state.pop("pine_compatibility_report", None)
        st.session_state.pop("pine_strategy_spec", None)
        st.session_state.pop("pine_strategy_spec_validation", None)
        st.session_state.pop("pine_codegen_report", None)
        st.session_state.pop("pine_generated_module_path", None)
        st.session_state.pop("pine_generation_trace", None)
        st.session_state.pop("pine_artifacts_manifest", None)
        st.session_state.pop("pine_beta_readiness_report", None)
        st.session_state.pop("pine_execution_gate_report", None)
        st.session_state.pop("pine_parity_report", None)
        st.session_state.pop("pine_request_security_diagnostics", None)
        st.session_state.pop("pine_mtf_parity_proof_report", None)
        st.session_state.pop("pine_parity_reference_payload", None)
        st.session_state.pop("pine_parity_reference_validation", None)
        st.session_state.pop("pine_parity_reference_metrics", None)
        st.session_state.pop("pine_parity_reference_text", None)
        st.session_state.pop("pine_source_text", None)
        st.session_state.pop("pine_library_files", None)
        st.session_state.pop("pine_library_paths", None)
        st.session_state.pop("pine_library_names", None)
        st.session_state.pop("pine_import_mapping", None)
        with zipfile.ZipFile(zip_file) as zf:
            names = set(zf.namelist())

            def _read_json_from_candidates(candidates, label):
                for name in candidates:
                    if name not in names:
                        continue
                    try:
                        data = json.loads(zf.read(name).decode("utf-8"))
                        return data, name
                    except Exception as e:
                        st.warning(f"Impossible de lire {name} ({label}): {e}")
                return None, None

            def _read_text_from_candidates(candidates, label):
                for name in candidates:
                    if name not in names:
                        continue
                    try:
                        raw = zf.read(name)
                        try:
                            return raw.decode("utf-8"), "utf-8", name
                        except UnicodeDecodeError:
                            return raw.decode("latin-1", errors="replace"), "latin-1", name
                    except Exception as e:
                        st.warning(f"Impossible de lire {name} ({label}): {e}")
                return None, None, None

            payload = {}
            if "results.json" in names:
                payload = json.loads(zf.read("results.json").decode("utf-8"))
                if payload.get("wfo_results"):
                    st.session_state["wfo_results"] = payload["wfo_results"]
                pine_precheck = payload.get("pine_precheck_report")
                if isinstance(pine_precheck, dict):
                    st.session_state["pine_precheck_report"] = pine_precheck
                pine_compat = payload.get("pine_compatibility_report")
                if isinstance(pine_compat, dict):
                    st.session_state["pine_compatibility_report"] = pine_compat
                pine_spec = payload.get("pine_strategy_spec")
                if isinstance(pine_spec, dict):
                    st.session_state["pine_strategy_spec"] = pine_spec
                pine_spec_validation = payload.get("pine_strategy_spec_validation")
                if isinstance(pine_spec_validation, dict):
                    st.session_state["pine_strategy_spec_validation"] = pine_spec_validation
                pine_codegen_report = payload.get("pine_codegen_report")
                if isinstance(pine_codegen_report, dict):
                    st.session_state["pine_codegen_report"] = pine_codegen_report
                if isinstance(payload.get("pine_generated_module_path"), str):
                    st.session_state["pine_generated_module_path"] = payload.get("pine_generated_module_path")
                pine_trace = payload.get("pine_generation_trace")
                if isinstance(pine_trace, dict):
                    st.session_state["pine_generation_trace"] = pine_trace
                pine_manifest = payload.get("pine_artifacts_manifest")
                if isinstance(pine_manifest, dict):
                    st.session_state["pine_artifacts_manifest"] = pine_manifest
                pine_beta = payload.get("pine_beta_readiness_report")
                if isinstance(pine_beta, dict):
                    st.session_state["pine_beta_readiness_report"] = pine_beta
                pine_gate = payload.get("pine_execution_gate_report")
                if isinstance(pine_gate, dict):
                    st.session_state["pine_execution_gate_report"] = pine_gate
                pine_parity = payload.get("pine_parity_report")
                if isinstance(pine_parity, dict):
                    st.session_state["pine_parity_report"] = pine_parity
                pine_mtf_diag = payload.get("pine_request_security_diagnostics")
                if isinstance(pine_mtf_diag, dict):
                    st.session_state["pine_request_security_diagnostics"] = pine_mtf_diag
                pine_mtf_proof = payload.get("pine_mtf_parity_proof_report")
                if isinstance(pine_mtf_proof, dict):
                    st.session_state["pine_mtf_parity_proof_report"] = pine_mtf_proof
                pine_parity_payload = payload.get("pine_parity_reference_payload")
                if isinstance(pine_parity_payload, dict):
                    st.session_state["pine_parity_reference_payload"] = pine_parity_payload
                pine_parity_validation = payload.get("pine_parity_reference_validation")
                if isinstance(pine_parity_validation, dict):
                    st.session_state["pine_parity_reference_validation"] = pine_parity_validation
                pine_parity_ref = payload.get("pine_parity_reference_metrics")
                if isinstance(pine_parity_ref, dict):
                    st.session_state["pine_parity_reference_metrics"] = pine_parity_ref
                    if isinstance(st.session_state.get("pine_parity_reference_payload"), dict):
                        st.session_state["pine_parity_reference_text"] = json.dumps(
                            st.session_state["pine_parity_reference_payload"], indent=2, ensure_ascii=False
                        )
                    else:
                        st.session_state["pine_parity_reference_text"] = json.dumps(
                            pine_parity_ref, indent=2, ensure_ascii=False
                        )
                if isinstance(payload.get("pine_source_name"), str):
                    st.session_state["pine_source_name"] = payload.get("pine_source_name")
                if isinstance(payload.get("pine_source_encoding"), str):
                    st.session_state["pine_source_encoding"] = payload.get("pine_source_encoding")
                if isinstance(payload.get("pine_library_files"), list):
                    st.session_state["pine_library_files"] = payload.get("pine_library_files")
                if isinstance(payload.get("pine_library_paths"), list):
                    st.session_state["pine_library_paths"] = payload.get("pine_library_paths")
                if isinstance(payload.get("pine_library_names"), list):
                    st.session_state["pine_library_names"] = payload.get("pine_library_names")
                if isinstance(payload.get("pine_import_mapping"), dict):
                    st.session_state["pine_import_mapping"] = payload.get("pine_import_mapping")
            if "audit_trace.json" in names:
                traceability = json.loads(zf.read("audit_trace.json").decode("utf-8"))
                st.session_state["wfo_traceability"] = traceability
                if isinstance(traceability, dict) and isinstance(traceability.get("run"), dict):
                    st.session_state["wfo_run_metadata"] = traceability["run"]
            elif isinstance(payload, dict) and payload.get("traceability"):
                st.session_state["wfo_traceability"] = payload.get("traceability")
                if isinstance(payload["traceability"], dict) and isinstance(payload["traceability"].get("run"), dict):
                    st.session_state["wfo_run_metadata"] = payload["traceability"]["run"]
            if "df.parquet" in names:
                try:
                    st.session_state["df"] = pd.read_parquet(io.BytesIO(zf.read("df.parquet")))
                except Exception as e:
                    st.warning(f"Impossible de lire df.parquet: {e}")
            elif "df.csv" in names:
                df = pd.read_csv(io.BytesIO(zf.read("df.csv")))
                if "Open time" in df.columns:
                    df["Open time"] = pd.to_datetime(df["Open time"], errors="coerce")
                    df.set_index("Open time", inplace=True)
                st.session_state["df"] = df
            if "all_trials.csv" in names:
                st.session_state["all_trials_df"] = pd.read_csv(io.BytesIO(zf.read("all_trials.csv")))
            if "window_info.csv" in names:
                st.session_state["window_info_df"] = pd.read_csv(io.BytesIO(zf.read("window_info.csv")))

            if "expert_context_pack.json" in names:
                try:
                    expert_context_pack = json.loads(zf.read("expert_context_pack.json").decode("utf-8"))
                    if isinstance(expert_context_pack, dict):
                        st.session_state["expert_context_pack"] = expert_context_pack
                        history = expert_context_pack.get("followup_history")
                        if isinstance(history, list):
                            st.session_state["expert_followup_history"] = history[-50:]
                        # Optional fallback: recover compact trials if CSV is missing.
                        if "all_trials_df" not in st.session_state:
                            trials_compact = (
                                (expert_context_pack.get("expert_input_data") or {}).get("all_trials")
                                if isinstance(expert_context_pack.get("expert_input_data"), dict)
                                else None
                            )
                            if isinstance(trials_compact, list) and trials_compact:
                                st.session_state["all_trials_df"] = pd.DataFrame(trials_compact)
                except Exception as e:
                    st.warning(f"Impossible de lire expert_context_pack.json: {e}")

            pine_precheck, _ = _read_json_from_candidates(
                ["pine_precheck_report.json", "precheck_report.json"],
                "pine_precheck",
            )
            if isinstance(pine_precheck, dict):
                st.session_state["pine_precheck_report"] = pine_precheck

            pine_compat, _ = _read_json_from_candidates(
                ["compatibility_report.json", "pine_compatibility_report.json"],
                "pine_compatibility",
            )
            if isinstance(pine_compat, dict):
                st.session_state["pine_compatibility_report"] = pine_compat

            pine_spec, _ = _read_json_from_candidates(
                ["strategy_spec.v1.json"],
                "strategy_spec",
            )
            if isinstance(pine_spec, dict):
                st.session_state["pine_strategy_spec"] = pine_spec

            pine_spec_validation, _ = _read_json_from_candidates(
                ["strategy_spec_validation.json"],
                "strategy_spec_validation",
            )
            if isinstance(pine_spec_validation, dict):
                st.session_state["pine_strategy_spec_validation"] = pine_spec_validation

            pine_trace, _ = _read_json_from_candidates(
                ["generation_trace.json", "pine_generation_trace.json"],
                "pine_generation_trace",
            )
            if isinstance(pine_trace, dict):
                st.session_state["pine_generation_trace"] = pine_trace

            pine_manifest, _ = _read_json_from_candidates(
                ["pine_artifacts_manifest.json"],
                "pine_artifacts_manifest",
            )
            if isinstance(pine_manifest, dict):
                st.session_state["pine_artifacts_manifest"] = pine_manifest

            pine_beta, _ = _read_json_from_candidates(
                ["pine_beta_readiness_report.json"],
                "pine_beta_readiness",
            )
            if isinstance(pine_beta, dict):
                st.session_state["pine_beta_readiness_report"] = pine_beta

            pine_gate, _ = _read_json_from_candidates(
                ["pine_execution_gate_report.json"],
                "pine_execution_gate",
            )
            if isinstance(pine_gate, dict):
                st.session_state["pine_execution_gate_report"] = pine_gate

            pine_parity, _ = _read_json_from_candidates(
                ["pine_parity_report.json"],
                "pine_parity_report",
            )
            if isinstance(pine_parity, dict):
                st.session_state["pine_parity_report"] = pine_parity

            pine_mtf_diag, _ = _read_json_from_candidates(
                ["pine_request_security_diagnostics.json"],
                "pine_request_security_diagnostics",
            )
            if isinstance(pine_mtf_diag, dict):
                st.session_state["pine_request_security_diagnostics"] = pine_mtf_diag

            pine_mtf_proof, _ = _read_json_from_candidates(
                ["pine_mtf_parity_proof_report.json"],
                "pine_mtf_parity_proof_report",
            )
            if isinstance(pine_mtf_proof, dict):
                st.session_state["pine_mtf_parity_proof_report"] = pine_mtf_proof

            pine_parity_payload, _ = _read_json_from_candidates(
                ["pine_parity_reference.v1.json"],
                "pine_parity_reference_payload",
            )
            if isinstance(pine_parity_payload, dict):
                st.session_state["pine_parity_reference_payload"] = pine_parity_payload

            pine_parity_validation, _ = _read_json_from_candidates(
                ["pine_parity_reference_validation.json"],
                "pine_parity_reference_validation",
            )
            if isinstance(pine_parity_validation, dict):
                st.session_state["pine_parity_reference_validation"] = pine_parity_validation

            pine_parity_ref, _ = _read_json_from_candidates(
                ["pine_parity_reference_metrics.json"],
                "pine_parity_reference_metrics",
            )
            if isinstance(pine_parity_ref, dict):
                st.session_state["pine_parity_reference_metrics"] = pine_parity_ref
                if isinstance(st.session_state.get("pine_parity_reference_payload"), dict):
                    st.session_state["pine_parity_reference_text"] = json.dumps(
                        st.session_state["pine_parity_reference_payload"], indent=2, ensure_ascii=False
                    )
                else:
                    st.session_state["pine_parity_reference_text"] = json.dumps(
                        pine_parity_ref, indent=2, ensure_ascii=False
                    )

            # Normalize/validate parity reference after ZIP load (v1 preferred, legacy tolerated).
            loaded_parity_payload = st.session_state.get("pine_parity_reference_payload")
            if isinstance(loaded_parity_payload, dict):
                loaded_validation = _validate_parity_reference_payload(
                    loaded_parity_payload,
                    allow_legacy=False,
                )
                normalized_payload = loaded_validation.get("normalized_payload")
                if isinstance(normalized_payload, dict) and normalized_payload:
                    _apply_parity_reference_payload(normalized_payload, loaded_validation, update_text=True)
            elif isinstance(st.session_state.get("pine_parity_reference_metrics"), dict):
                legacy_payload = _build_parity_reference_payload(
                    st.session_state.get("pine_parity_reference_metrics"),
                    source={
                        "provider": "legacy_zip",
                        "strategy_id": str(st.session_state.get("strategy_id") or ""),
                    },
                )
                legacy_validation = _validate_parity_reference_payload(legacy_payload, allow_legacy=False)
                _apply_parity_reference_payload(legacy_payload, legacy_validation, update_text=True)

            pine_import_mapping, _ = _read_json_from_candidates(
                ["import_mapping.json", "pine_import_mapping.json"],
                "pine_import_mapping",
            )
            if isinstance(pine_import_mapping, dict):
                st.session_state["pine_import_mapping"] = pine_import_mapping

            pine_source_text, pine_source_encoding, pine_source_file = _read_text_from_candidates(
                ["strategy_source.pine.txt", "v3/strategy_source.pine.txt", "artifacts/v3/strategy_source.pine.txt"],
                "pine_source",
            )
            if isinstance(pine_source_text, str) and pine_source_text.strip():
                st.session_state["pine_source_text"] = pine_source_text
                if isinstance(pine_source_encoding, str):
                    st.session_state["pine_source_encoding"] = pine_source_encoding
                source_name = None
                if isinstance(st.session_state.get("pine_generation_trace"), dict):
                    source_name = ((st.session_state["pine_generation_trace"].get("source") or {}).get("name"))
                if not source_name and isinstance(payload, dict):
                    source_name = payload.get("pine_source_name")
                if not source_name:
                    source_name = os.path.basename(str(pine_source_file or "strategy_source.pine.txt"))
                st.session_state["pine_source_name"] = source_name
                persisted_pine_path = _persist_pine_source_text(pine_source_text, source_name=source_name)
                if isinstance(persisted_pine_path, str):
                    st.session_state["pine_file_path"] = persisted_pine_path

            generated_strategy_text, _, generated_strategy_file = _read_text_from_candidates(
                ["generated_strategy.py"],
                "generated_strategy",
            )
            if isinstance(generated_strategy_text, str) and generated_strategy_text.strip():
                generated_name = os.path.basename(str(generated_strategy_file or "generated_strategy.py"))
                persisted_generated_path = _persist_generated_strategy_text(
                    generated_strategy_text,
                    source_name=generated_name,
                )
                if isinstance(persisted_generated_path, str):
                    st.session_state["pine_generated_module_path"] = persisted_generated_path
                    if not isinstance(st.session_state.get("pine_codegen_report"), dict):
                        st.session_state["pine_codegen_report"] = {
                            "status": "ok",
                            "output_path": persisted_generated_path,
                            "module_name": os.path.basename(persisted_generated_path),
                            "changed": False,
                        }

            libs_manifest, _ = _read_json_from_candidates(
                ["pine_libraries_manifest.json"],
                "pine_libraries_manifest",
            )
            if isinstance(libs_manifest, list) and libs_manifest:
                reloaded_libs = []
                for lib in libs_manifest:
                    if not isinstance(lib, dict):
                        continue
                    zip_path = str(lib.get("zip_path") or "").strip()
                    source_name = str(lib.get("source_name") or "").strip()
                    if not zip_path or zip_path not in names:
                        continue
                    lib_text, lib_encoding, _ = _read_text_from_candidates([zip_path], "pine_library")
                    if not isinstance(lib_text, str):
                        continue
                    persisted_path = _persist_pine_library_text(lib_text, source_name=source_name or os.path.basename(zip_path))
                    if not persisted_path:
                        continue
                    reloaded_libs.append(
                        {
                            "source_name": source_name or os.path.basename(persisted_path),
                            "path": persisted_path,
                            "source_sha1": hashlib.sha1(lib_text.encode("utf-8", errors="ignore")).hexdigest(),
                            "size_bytes": int(len(lib_text.encode("utf-8", errors="ignore"))),
                            "encoding": lib_encoding or lib.get("encoding"),
                        }
                    )
                if reloaded_libs:
                    st.session_state["pine_library_files"] = reloaded_libs
                    st.session_state["pine_library_paths"] = [str(item.get("path") or "") for item in reloaded_libs]
                    st.session_state["pine_library_names"] = [str(item.get("source_name") or "") for item in reloaded_libs]

            # Reinforced replay loading: if spec exists but validation is missing,
            # validate immediately to keep replay diagnostics deterministic.
            loaded_spec = st.session_state.get("pine_strategy_spec")
            loaded_spec_validation = st.session_state.get("pine_strategy_spec_validation")
            if isinstance(loaded_spec, dict):
                if not isinstance(loaded_spec_validation, dict):
                    try:
                        st.session_state["pine_strategy_spec_validation"] = _sanitize_for_json(
                            _validate_strategy_spec_v1(loaded_spec)
                        )
                    except Exception as e:
                        st.warning(f"Validation auto du strategy_spec impossible: {e}")
                # Align strategy metadata for replay traceability.
                st.session_state["strategy_mode"] = "pine_imported"
                strategy_id = ((loaded_spec.get("strategy") or {}).get("id"))
                if isinstance(strategy_id, str) and strategy_id.strip():
                    st.session_state["strategy_id"] = strategy_id

            if (
                isinstance(st.session_state.get("pine_precheck_report"), dict)
                and isinstance(st.session_state.get("pine_source_text"), str)
            ):
                expected_sha1 = str(st.session_state["pine_precheck_report"].get("source_sha1") or "").strip()
                if expected_sha1:
                    current_sha1 = hashlib.sha1(
                        st.session_state["pine_source_text"].encode("utf-8", errors="ignore")
                    ).hexdigest()
                    if current_sha1 != expected_sha1:
                        st.warning(
                            "Le SHA1 de la source Pine rechargée diffère du précheck enregistré. "
                            "Vérifie la cohérence des artefacts."
                        )

            # Optional: load backtest artifacts for display
            if "final_trades.csv" in names:
                trades_df = pd.read_csv(io.BytesIO(zf.read("final_trades.csv")))
                st.session_state["final_trades_df"] = trades_df
            if "final_trade_stats.csv" in names:
                stats_df = pd.read_csv(io.BytesIO(zf.read("final_trade_stats.csv")))
                st.session_state["final_trade_stats_df"] = stats_df
    except Exception as e:
        st.error(f"Error loading results ZIP: {e}")

def sync_dates_from_file(force=False):
    file_path = st.session_state.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return

    if not force and st.session_state.get('last_data_file_path') == file_path:
        return

    min_date, max_date = get_csv_date_range(file_path)
    if min_date and max_date:
        # Never write directly to widget-bound keys here; this callback can
        # run after widgets are instantiated in the same Streamlit cycle.
        st.session_state['pending_start_date'] = min_date
        st.session_state['pending_end_date'] = max_date
        st.session_state['last_data_file_path'] = file_path


def _persist_uploaded_data_file(uploaded_file):
    if uploaded_file is None:
        return None, False
    try:
        file_id = getattr(uploaded_file, "file_id", f"{uploaded_file.name}:{uploaded_file.size}")
        existing_id = st.session_state.get("uploaded_data_file_id")
        existing_path = st.session_state.get("uploaded_data_file_path")
        if existing_id == file_id and isinstance(existing_path, str) and os.path.exists(existing_path):
            return existing_path, False

        base_name = os.path.basename(str(uploaded_file.name or "uploaded_data.csv"))
        safe_name = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in base_name)
        if not safe_name:
            safe_name = "uploaded_data.csv"
        if not safe_name.lower().endswith(".csv"):
            safe_name = f"{safe_name}.csv"

        digest = hashlib.sha1(f"{file_id}_{time.time_ns()}".encode("utf-8")).hexdigest()[:12]
        target_dir = os.path.join(tempfile.gettempdir(), "atdmf_streamlit_uploads")
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, f"{digest}_{safe_name}")

        with open(target_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state["uploaded_data_file_id"] = file_id
        st.session_state["uploaded_data_file_path"] = target_path
        return target_path, True
    except Exception as e:
        st.error(f"Error while saving uploaded CSV: {e}")
        return None, False


def _pine_imports_dir():
    """Return persistent folder used to store imported Pine strategy files."""
    folder = os.path.join(_repo_root_dir(), "reports", "pine_imports")
    os.makedirs(folder, exist_ok=True)
    return folder


def _pine_generated_dir():
    """Return persistent folder used to store generated Pine adapter modules."""
    folder = os.path.join(_pine_imports_dir(), "generated")
    os.makedirs(folder, exist_ok=True)
    return folder

def _persist_pine_source_text(source_text: str, source_name: str = "strategy_source.pine.txt"):
    """Persist Pine source text restored from ZIP and return local file path."""
    text = str(source_text or "")
    if not text.strip():
        return None
    safe_name = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(source_name or "strategy_source.pine.txt"))
    if not safe_name:
        safe_name = "strategy_source.pine.txt"
    if not safe_name.lower().endswith((".pine", ".txt")):
        safe_name = f"{safe_name}.txt"

    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    target_path = os.path.join(_pine_imports_dir(), f"zip_{digest}_{safe_name}")
    if not os.path.exists(target_path):
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(text)
    return target_path


def _persist_generated_strategy_text(source_text: str, source_name: str = "generated_strategy.py"):
    """Persist generated Python strategy module restored from ZIP and return local file path."""
    text = str(source_text or "")
    if not text.strip():
        return None
    safe_name = "".join(
        ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(source_name or "generated_strategy.py")
    )
    if not safe_name:
        safe_name = "generated_strategy.py"
    if not safe_name.lower().endswith(".py"):
        safe_name = f"{safe_name}.py"

    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    target_path = os.path.join(_pine_generated_dir(), f"zipgen_{digest}_{safe_name}")
    if not os.path.exists(target_path):
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(text)
    return target_path


def _persist_uploaded_pine_file(uploaded_file):
    """Persist uploaded Pine/TXT strategy file to disk and return saved path."""
    if uploaded_file is None:
        return None, False
    try:
        file_id = getattr(uploaded_file, "file_id", f"{uploaded_file.name}:{uploaded_file.size}")
        existing_id = st.session_state.get("uploaded_pine_file_id")
        existing_path = st.session_state.get("uploaded_pine_file_path")
        if existing_id == file_id and isinstance(existing_path, str) and os.path.exists(existing_path):
            return existing_path, False

        base_name = os.path.basename(str(uploaded_file.name or "strategy.pine"))
        safe_name = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in base_name)
        if not safe_name:
            safe_name = "strategy.pine"
        if not safe_name.lower().endswith((".pine", ".txt")):
            safe_name = f"{safe_name}.txt"

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        digest = hashlib.sha1(f"{file_id}_{time.time_ns()}".encode("utf-8")).hexdigest()[:12]
        target_path = os.path.join(_pine_imports_dir(), f"{timestamp}_{digest}_{safe_name}")
        with open(target_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state["uploaded_pine_file_id"] = file_id
        st.session_state["uploaded_pine_file_path"] = target_path
        st.session_state["pine_source_name"] = base_name
        st.session_state["pine_file_path"] = target_path
        return target_path, True
    except Exception as e:
        st.error(f"Error while saving uploaded Pine file: {e}")
        return None, False


def _persist_pine_library_text(source_text: str, source_name: str = "library_source.pine.txt"):
    """Persist a Pine library text restored from ZIP and return local file path."""
    text = str(source_text or "")
    if not text.strip():
        return None
    safe_name = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(source_name or "library_source.pine.txt"))
    if not safe_name:
        safe_name = "library_source.pine.txt"
    if not safe_name.lower().endswith((".pine", ".txt")):
        safe_name = f"{safe_name}.txt"

    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    target_path = os.path.join(_pine_imports_dir(), f"ziplib_{digest}_{safe_name}")
    if not os.path.exists(target_path):
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(text)
    return target_path


def _persist_uploaded_pine_library_files(uploaded_files):
    """
    Persist uploaded Pine library files and keep a session manifest.

    Returns:
        tuple[list[dict], bool]: (manifest, has_new_file)
    """
    files = list(uploaded_files or [])
    if len(files) == 0:
        existing = st.session_state.get("pine_library_files")
        if isinstance(existing, list):
            return existing, False
        st.session_state["pine_library_files"] = []
        st.session_state["pine_library_paths"] = []
        st.session_state["pine_library_names"] = []
        return [], False

    manifest = []
    has_new_file = False
    try:
        for uploaded in files:
            base_name = os.path.basename(str(getattr(uploaded, "name", "") or "library.pine"))
            safe_name = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in base_name)
            if not safe_name:
                safe_name = "library.pine"
            if not safe_name.lower().endswith((".pine", ".txt")):
                safe_name = f"{safe_name}.txt"

            raw = bytes(uploaded.getbuffer())
            source_sha1 = hashlib.sha1(raw).hexdigest()
            target_path = os.path.join(_pine_imports_dir(), f"lib_{source_sha1[:12]}_{safe_name}")
            if not os.path.exists(target_path):
                with open(target_path, "wb") as f:
                    f.write(raw)
                has_new_file = True

            manifest.append(
                {
                    "source_name": base_name,
                    "path": target_path,
                    "source_sha1": source_sha1,
                    "size_bytes": int(len(raw)),
                }
            )
    except Exception as e:
        st.error(f"Error while saving uploaded Pine libraries: {e}")
        return st.session_state.get("pine_library_files", []), False

    st.session_state["pine_library_files"] = manifest
    st.session_state["pine_library_paths"] = [str(item.get("path", "")) for item in manifest]
    st.session_state["pine_library_names"] = [str(item.get("source_name", "")) for item in manifest]
    return manifest, has_new_file


def _normalize_token(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _parse_pine_import_lines(import_lines: list[str] | None):
    """Parse Pine import lines into structured records."""
    rows = []
    for raw in list(import_lines or []):
        line = str(raw or "").strip()
        if not line:
            continue
        match = re.match(
            r"^\s*import\s+([A-Za-z0-9_./-]+)\s+as\s+([A-Za-z_][A-Za-z0-9_]*)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if not match:
            rows.append(
                {
                    "raw": line,
                    "module_ref": None,
                    "alias": None,
                    "parse_ok": False,
                }
            )
            continue
        module_ref = str(match.group(1) or "").strip()
        alias = str(match.group(2) or "").strip()
        module_parts = [part for part in module_ref.split("/") if part]
        module_tail = module_parts[-1] if module_parts else ""
        if re.fullmatch(r"\d+", module_tail) and len(module_parts) >= 2:
            module_tail = module_parts[-2]
        rows.append(
            {
                "raw": line,
                "module_ref": module_ref,
                "alias": alias,
                "module_tail": module_tail,
                "parse_ok": True,
            }
        )
    return rows


def _extract_library_decl_name(source_text: str) -> str:
    """Best-effort parse of Pine `library(...)` declaration title/name."""
    text = str(source_text or "")
    if not text.strip():
        return ""
    lib_block_match = re.search(r"\blibrary\s*\((.*?)\)", text, flags=re.IGNORECASE | re.DOTALL)
    if not lib_block_match:
        return ""
    block = lib_block_match.group(1)
    title_match = re.search(r"""title\s*=\s*(['"])(.*?)\1""", block, flags=re.IGNORECASE | re.DOTALL)
    if title_match:
        return str(title_match.group(2) or "").strip()
    first_string = re.search(r"""(['"])(.*?)\1""", block, flags=re.DOTALL)
    if first_string:
        return str(first_string.group(2) or "").strip()
    return ""


def _extract_pine_library_functions(source_text: str) -> list[str]:
    """
    Extract function names declared in a Pine library file.

    Supports common forms:
    - `export foo(args) =>`
    - `foo(args) =>`
    """
    text = str(source_text or "")
    if not text.strip():
        return []
    names = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        m = re.match(
            r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*=>",
            stripped,
            flags=re.IGNORECASE,
        )
        if m:
            names.add(str(m.group(1) or "").strip())
    return sorted([n for n in names if n])


def _extract_alias_function_calls(source_text: str) -> dict[str, list[str]]:
    """
    Extract calls of the form `Alias.func(...)` from Pine source.
    """
    text = str(source_text or "")
    calls: dict[str, set[str]] = {}
    if not text.strip():
        return {}
    for alias, func in re.findall(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        text,
        flags=re.IGNORECASE,
    ):
        alias_key = str(alias).strip()
        func_name = str(func).strip()
        if not alias_key or not func_name:
            continue
        calls.setdefault(alias_key, set()).add(func_name)
    return {k: sorted(list(v)) for k, v in calls.items()}


def _analyze_provided_libraries(provided_library_files: list[dict] | None):
    """
    Build lightweight metadata used to match Pine imports with uploaded libraries.
    """
    out = []
    for item in list(provided_library_files or []):
        if not isinstance(item, dict):
            continue
        source_name = str(item.get("source_name") or "").strip()
        path = str(item.get("path") or "").strip()
        stem = os.path.splitext(os.path.basename(source_name or path))[0]
        normalized_candidates = set()
        if stem:
            normalized_candidates.add(_normalize_token(stem))
        declared_name = ""
        declared_functions: list[str] = []
        if path and os.path.exists(path):
            try:
                txt, _ = _read_text_file_with_fallback(path)
                declared_name = _extract_library_decl_name(txt)
                declared_functions = _extract_pine_library_functions(txt)
            except Exception:
                declared_name = ""
                declared_functions = []
        if declared_name:
            normalized_candidates.add(_normalize_token(declared_name))
        out.append(
            {
                "source_name": source_name,
                "path": path,
                "source_sha1": str(item.get("source_sha1") or ""),
                "declared_name": declared_name,
                "declared_functions": declared_functions,
                "normalized_candidates": sorted([c for c in normalized_candidates if c]),
            }
        )
    return out


def _validate_python_mapping_target(target: str):
    """
    Validate mapping target:
    - `/abs/path/module.py` or `relative/path/module.py`
    - dotted module path (`package.module`) importable in current env
    """
    value = str(target or "").strip()
    if not value:
        return False, "mapping vide"
    # File path mode.
    if value.endswith(".py") or "/" in value or "\\" in value:
        candidate = value
        if not os.path.isabs(candidate):
            candidate = os.path.join(_repo_root_dir(), candidate)
        if os.path.exists(candidate):
            return True, f"path:{os.path.abspath(candidate)}"
        return False, f"fichier introuvable: {candidate}"
    # Dotted module mode.
    try:
        spec = importlib.util.find_spec(value)
        if spec is not None:
            return True, f"module:{value}"
    except Exception:
        pass
    return False, f"module non importable: {value}"


def _resolve_pine_imports(
    import_lines: list[str] | None,
    source_text: str | None = None,
    provided_library_files: list[dict] | None = None,
    import_mapping: dict | None = None,
):
    """
    Resolve Pine imports against uploaded library files + explicit Python mapping.
    """
    parsed_imports = _parse_pine_import_lines(import_lines)
    libraries = _analyze_provided_libraries(provided_library_files)
    mapping_dict = import_mapping if isinstance(import_mapping, dict) else {}
    alias_calls = _extract_alias_function_calls(str(source_text or ""))
    resolution = []
    resolved_count = 0
    functions_called_total = 0
    functions_missing_total = 0

    for item in parsed_imports:
        module_ref = item.get("module_ref")
        alias = item.get("alias")
        raw = item.get("raw")
        parse_ok = bool(item.get("parse_ok"))
        module_tail = _normalize_token(item.get("module_tail") or "")
        alias_norm = _normalize_token(alias or "")

        matched_lib = None
        for lib in libraries:
            candidates = set(lib.get("normalized_candidates") or [])
            if module_tail and module_tail in candidates:
                matched_lib = lib
                break
            if alias_norm and alias_norm in candidates:
                matched_lib = lib
                break

        mapping_target = ""
        if alias and alias in mapping_dict:
            mapping_target = str(mapping_dict.get(alias) or "").strip()
        elif module_ref and module_ref in mapping_dict:
            mapping_target = str(mapping_dict.get(module_ref) or "").strip()
        elif alias_norm and alias_norm in mapping_dict:
            mapping_target = str(mapping_dict.get(alias_norm) or "").strip()

        mapping_valid = False
        mapping_detail = ""
        if mapping_target:
            mapping_valid, mapping_detail = _validate_python_mapping_target(mapping_target)

        called_functions = alias_calls.get(str(alias or ""), []) if alias else []
        library_functions = []
        if isinstance(matched_lib, dict):
            library_functions = list(matched_lib.get("declared_functions") or [])
        library_function_set = set(str(x) for x in library_functions)
        missing_functions = [f for f in called_functions if str(f) not in library_function_set]
        found_functions_count = len([f for f in called_functions if str(f) in library_function_set])
        functions_called_total += len(called_functions)
        functions_missing_total += len(missing_functions)

        functions_ok = len(missing_functions) == 0
        resolved = parse_ok and bool(matched_lib) and bool(mapping_target) and bool(mapping_valid) and functions_ok
        if resolved:
            resolved_count += 1

        resolution.append(
            {
                "raw": raw,
                "module_ref": module_ref,
                "alias": alias,
                "parse_ok": parse_ok,
                "library_file_found": bool(matched_lib),
                "library_source_name": matched_lib.get("source_name") if isinstance(matched_lib, dict) else None,
                "library_path": matched_lib.get("path") if isinstance(matched_lib, dict) else None,
                "library_functions_count": len(library_functions),
                "called_functions": called_functions,
                "called_functions_count": len(called_functions),
                "found_functions_count": int(found_functions_count),
                "missing_functions": missing_functions,
                "missing_functions_count": len(missing_functions),
                "python_mapping_target": mapping_target or None,
                "python_mapping_valid": bool(mapping_valid),
                "python_mapping_detail": mapping_detail or None,
                "resolved": bool(resolved),
            }
        )

    return {
        "imports": resolution,
        "import_count": len(parsed_imports),
        "resolved_count": int(resolved_count),
        "unresolved_count": int(max(0, len(parsed_imports) - resolved_count)),
        "functions_called_total": int(functions_called_total),
        "functions_missing_total": int(functions_missing_total),
    }


def _precheck_pine_script_text(
    source_text: str,
    source_name: str = "",
    provided_library_files: list[dict] | None = None,
    import_mapping: dict | None = None,
):
    """Run a lightweight Pine pre-check to provide immediate actionable feedback."""
    text = str(source_text or "")
    errors = []
    warnings = []
    info = []

    line_count = text.count("\n") + (1 if text else 0)
    char_count = len(text)
    if not text.strip():
        errors.append("Le fichier est vide.")

    version_match = re.search(r"//@version\s*=\s*(\d+)", text, flags=re.IGNORECASE)
    detected_version = int(version_match.group(1)) if version_match else None
    if detected_version is None:
        errors.append("Directive `//@version=...` manquante.")
    elif detected_version != 6:
        warnings.append(
            f"Version Pine détectée: v{detected_version}. WFOE V3 cible prioritairement Pine v6."
        )

    has_strategy_decl = bool(re.search(r"^\s*strategy\s*\(", text, flags=re.IGNORECASE | re.MULTILINE))
    has_indicator_decl = bool(re.search(r"^\s*indicator\s*\(", text, flags=re.IGNORECASE | re.MULTILINE))
    if not has_strategy_decl:
        errors.append("Déclaration `strategy(...)` introuvable.")
    if has_indicator_decl and not has_strategy_decl:
        warnings.append("Le script semble être un `indicator`, pas une `strategy` backtestable.")

    opens = text.count("(")
    closes = text.count(")")
    if opens != closes:
        warnings.append(
            f"Déséquilibre de parenthèses détecté: ouvrantes={opens}, fermantes={closes}."
        )

    import_lines = re.findall(r"^\s*import\s+.+$", text, flags=re.IGNORECASE | re.MULTILINE)
    provided_library_files = provided_library_files if isinstance(provided_library_files, list) else []
    import_mapping = import_mapping if isinstance(import_mapping, dict) else {}
    provided_library_names = [
        str(item.get("source_name") or "").strip()
        for item in provided_library_files
        if isinstance(item, dict)
    ]
    provided_library_names = [name for name in provided_library_names if name]
    provided_libraries_count = len(provided_library_names)
    import_resolution = _resolve_pine_imports(
        import_lines=import_lines,
        source_text=text,
        provided_library_files=provided_library_files,
        import_mapping=import_mapping,
    )
    resolved_import_count = int(import_resolution.get("resolved_count", 0))
    unresolved_import_count = int(import_resolution.get("unresolved_count", 0))
    functions_called_total = int(import_resolution.get("functions_called_total", 0))
    functions_missing_total = int(import_resolution.get("functions_missing_total", 0))
    if import_lines:
        if provided_libraries_count == 0:
            warnings.append(
                "Des imports Pine externes sont présents. Aucun fichier de librairie n'est fourni: "
                "ajoute les fichiers .txt/.pine des librairies avec la stratégie."
            )
        elif unresolved_import_count > 0:
            warnings.append(
                f"Des imports Pine externes sont présents ({len(import_lines)} import(s)) et "
                f"{provided_libraries_count} fichier(s) de librairie ont été fournis. "
                f"Résolution incomplète: {resolved_import_count}/{len(import_lines)} import(s) mappés. "
                f"Fonctions manquantes: {functions_missing_total}/{functions_called_total}."
            )
        else:
            info.append(
                f"Imports Pine externes résolus: {resolved_import_count}/{len(import_lines)} "
                f"(fichiers librairie + mapping Python). Fonctions vérifiées: "
                f"{functions_called_total - functions_missing_total}/{functions_called_total}."
            )

    detected_features = {
        "uses_request_security": bool(re.search(r"\brequest\.security\s*\(", text, flags=re.IGNORECASE)),
        "uses_request_security_lower_tf": bool(
            re.search(r"\brequest\.security_lower_tf\s*\(", text, flags=re.IGNORECASE)
        ),
        "uses_strategy_entry": bool(re.search(r"\bstrategy\.entry\s*\(", text, flags=re.IGNORECASE)),
        "uses_strategy_exit": bool(re.search(r"\bstrategy\.exit\s*\(", text, flags=re.IGNORECASE)),
        "uses_strategy_close": bool(re.search(r"\bstrategy\.close\s*\(", text, flags=re.IGNORECASE)),
        "uses_strategy_cancel": bool(re.search(r"\bstrategy\.cancel\s*\(", text, flags=re.IGNORECASE)),
        "uses_strategy_order": bool(re.search(r"\bstrategy\.order\s*\(", text, flags=re.IGNORECASE)),
        "uses_short_entry": bool(
            re.search(
                r"\bstrategy\.entry\s*\([^)]*direction\s*=\s*strategy\.short",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
        ),
        "uses_pyramiding": bool(re.search(r"\bpyramiding\s*=", text, flags=re.IGNORECASE)),
        "uses_loops": bool(re.search(r"^\s*(for|while)\b", text, flags=re.IGNORECASE | re.MULTILINE)),
        "uses_switch": bool(re.search(r"^\s*switch\b", text, flags=re.IGNORECASE | re.MULTILINE)),
        "import_count": len(import_lines),
        "resolved_import_count": resolved_import_count,
        "unresolved_import_count": unresolved_import_count,
        "external_functions_called_count": functions_called_total,
        "external_functions_missing_count": functions_missing_total,
    }
    if detected_features["uses_request_security_lower_tf"]:
        warnings.append(
            "`request.security_lower_tf` détecté: support prévu en mode limité, parité à vérifier."
        )

    source_sha1 = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    status = "valid" if not errors else "invalid"
    if status == "valid":
        info.append("Pré-analyse OK: script exploitable pour la phase suivante de compatibilité.")

    return _sanitize_for_json(
        {
            "status": status,
            "source_name": source_name or None,
            "line_count": line_count,
            "char_count": char_count,
            "detected_version": detected_version,
            "has_strategy_declaration": has_strategy_decl,
            "has_indicator_declaration": has_indicator_decl,
            "detected_features": detected_features,
            "errors": errors,
            "warnings": warnings,
            "info": info,
            "source_sha1": source_sha1,
            "import_lines": import_lines,
            "import_resolution": import_resolution.get("imports", []),
            "provided_libraries_count": provided_libraries_count,
            "provided_libraries_names": provided_library_names,
            "resolved_import_count": resolved_import_count,
            "unresolved_import_count": unresolved_import_count,
            "external_functions_called_count": functions_called_total,
            "external_functions_missing_count": functions_missing_total,
            "pine_import_mapping": import_mapping,
        }
    )


def _precheck_pine_script_file(
    file_path: str,
    provided_library_files: list[dict] | None = None,
    import_mapping: dict | None = None,
):
    """Read a Pine file from disk and return a pre-check report."""
    path = str(file_path or "").strip()
    if not path:
        return None
    if not os.path.exists(path):
        return {
            "status": "invalid",
            "source_name": os.path.basename(path) if path else None,
            "errors": [f"Fichier introuvable: {path}"],
            "warnings": [],
            "info": [],
        }
    try:
        with open(path, "rb") as f:
            raw = f.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="replace")
        return _precheck_pine_script_text(
            text,
            source_name=os.path.basename(path),
            provided_library_files=provided_library_files,
            import_mapping=import_mapping,
        )
    except Exception as e:
        return {
            "status": "invalid",
            "source_name": os.path.basename(path),
            "errors": [f"Erreur de lecture: {e}"],
            "warnings": [],
            "info": [],
        }


def _build_pine_compatibility_report(precheck_report: dict, compat_mode: str = "strict"):
    """Build Pine compatibility report aligned with Lot P0.3 (S0/S1/S2/S3)."""
    if not isinstance(precheck_report, dict) or not precheck_report:
        return None

    mode = str(compat_mode or "strict").lower()
    if mode not in ("strict", "assist", "manual"):
        mode = "strict"

    detected = precheck_report.get("detected_features", {}) or {}
    import_count = int(detected.get("import_count", 0) or 0)
    resolved_import_count = int(precheck_report.get("resolved_import_count", 0) or 0)
    unresolved_import_count = int(precheck_report.get("unresolved_import_count", 0) or 0)
    external_functions_called_count = int(
        precheck_report.get("external_functions_called_count", 0) or 0
    )
    external_functions_missing_count = int(
        precheck_report.get("external_functions_missing_count", 0) or 0
    )
    provided_libraries_count = int(precheck_report.get("provided_libraries_count", 0) or 0)
    uses_security = bool(detected.get("uses_request_security", False))
    uses_lower_tf = bool(detected.get("uses_request_security_lower_tf", False))
    uses_strategy_order = bool(detected.get("uses_strategy_order", False))
    uses_short_entry = bool(detected.get("uses_short_entry", False))
    uses_pyramiding = bool(detected.get("uses_pyramiding", False))
    uses_loops = bool(detected.get("uses_loops", False))
    uses_switch = bool(detected.get("uses_switch", False))
    has_strategy_decl = bool(precheck_report.get("has_strategy_declaration", False))
    version = precheck_report.get("detected_version")
    pre_errors = list(precheck_report.get("errors", []) or [])
    pre_warnings = list(precheck_report.get("warnings", []) or [])

    items = []

    def _add_item(code, label, target_level, status, detected_flag, detail="", blocking=False):
        items.append(
            {
                "code": code,
                "label": label,
                "target_level": target_level,
                "status": status,
                "detected": bool(detected_flag),
                "blocking": bool(blocking),
                "detail": str(detail or ""),
            }
        )

    # Baseline structure checks.
    if has_strategy_decl:
        _add_item("strategy_decl", "Déclaration strategy(...)", "S3", "supported", True)
    else:
        _add_item(
            "strategy_decl",
            "Déclaration strategy(...)",
            "S3",
            "blocked",
            False,
            detail="Le script n'expose pas de stratégie backtestable.",
            blocking=True,
        )

    if version == 6:
        _add_item("pine_version", "Directive //@version=6", "S2", "supported", True)
    elif version is None:
        _add_item(
            "pine_version",
            "Directive //@version=6",
            "S2",
            "blocked",
            False,
            detail="Directive de version absente.",
            blocking=True,
        )
    else:
        _add_item(
            "pine_version",
            "Directive //@version=6",
            "S2",
            "partial",
            True,
            detail=f"Version détectée v{version}, adaptation potentiellement nécessaire.",
        )

    # Feature-level compatibility.
    if import_count > 0:
        detail = (
            f"{import_count} import(s) externe(s) détecté(s) | "
            f"résolus={resolved_import_count}, non résolus={unresolved_import_count}."
        )
        if external_functions_called_count > 0:
            detail += (
                f" Fonctions externes appelées={external_functions_called_count}, "
                f"manquantes={external_functions_missing_count}."
            )
        if provided_libraries_count > 0:
            detail += f" {provided_libraries_count} fichier(s) de librairie fourni(s)."
        if unresolved_import_count == 0 and resolved_import_count == import_count:
            _add_item(
                "external_imports",
                "Imports Pine externes",
                "S1",
                "partial",
                True,
                detail=detail + " Mapping fourni (phase assistée), validation de parité encore requise.",
                blocking=False,
            )
        else:
            _add_item(
                "external_imports",
                "Imports Pine externes",
                "S0",
                "blocked",
                True,
                detail=detail + " Mapping Python complet requis en mode strict.",
                blocking=True,
            )
    else:
        _add_item("external_imports", "Imports Pine externes", "S0", "not_applicable", False)

    if uses_security:
        _add_item(
            "request_security",
            "request.security",
            "S1",
            "partial",
            True,
            detail="Support MTF partiel prévu (lookahead/politiques à valider).",
        )
    else:
        _add_item("request_security", "request.security", "S1", "not_applicable", False)

    if uses_lower_tf:
        _add_item(
            "request_security_lower_tf",
            "request.security_lower_tf",
            "S0",
            "blocked",
            True,
            detail="Support bas timeframe non garanti au stade actuel.",
            blocking=True,
        )
    else:
        _add_item("request_security_lower_tf", "request.security_lower_tf", "S0", "not_applicable", False)

    if uses_strategy_order:
        _add_item(
            "strategy_order",
            "strategy.order",
            "S0",
            "blocked",
            True,
            detail="`strategy.order` n'est pas encore couvert par le runtime V3 beta.",
            blocking=True,
        )
    else:
        _add_item("strategy_order", "strategy.order", "S0", "not_applicable", False)

    if uses_short_entry:
        _add_item(
            "short_entries",
            "Entrées short",
            "S0",
            "blocked",
            True,
            detail="Le runtime V3 beta est long-only (entries short non supportées).",
            blocking=True,
        )
    else:
        _add_item("short_entries", "Entrées short", "S0", "not_applicable", False)

    if uses_pyramiding:
        _add_item(
            "pyramiding",
            "Pyramiding",
            "S0",
            "blocked",
            True,
            detail="Le runtime V3 beta ne reproduit pas encore le pyramiding Pine.",
            blocking=True,
        )
    else:
        _add_item("pyramiding", "Pyramiding", "S0", "not_applicable", False)

    if uses_loops or uses_switch:
        labels = []
        if uses_loops:
            labels.append("boucles")
        if uses_switch:
            labels.append("switch")
        _add_item(
            "control_flow_advanced",
            "Contrôle de flux avancé",
            "S0",
            "blocked",
            True,
            detail=f"Transpilation V3 beta partielle: {', '.join(labels)} non validés.",
            blocking=True,
        )
    else:
        _add_item("control_flow_advanced", "Contrôle de flux avancé", "S0", "not_applicable", False)

    for code, label, flag in [
        ("strategy_entry", "strategy.entry", bool(detected.get("uses_strategy_entry", False))),
        ("strategy_exit", "strategy.exit", bool(detected.get("uses_strategy_exit", False))),
        ("strategy_close", "strategy.close", bool(detected.get("uses_strategy_close", False))),
        ("strategy_cancel", "strategy.cancel", bool(detected.get("uses_strategy_cancel", False))),
    ]:
        if flag:
            _add_item(code, label, "S2", "supported", True)
        else:
            _add_item(code, label, "S2", "not_applicable", False)

    if pre_errors:
        _add_item(
            "precheck_errors",
            "Erreurs de pré-analyse",
            "S0",
            "blocked",
            True,
            detail=f"{len(pre_errors)} erreur(s): " + " | ".join(str(e) for e in pre_errors[:3]),
            blocking=True,
        )
    else:
        _add_item("precheck_errors", "Erreurs de pré-analyse", "S0", "supported", False)

    # Score model (deterministic and transparent).
    score_map = {
        "supported": 1.0,
        "partial": 0.5,
        "blocked": 0.0,
        "not_applicable": None,
    }
    scored = [score_map[it["status"]] for it in items if score_map.get(it["status"]) is not None]
    compatibility_score = round((sum(scored) / len(scored)) * 100, 1) if scored else 0.0

    blocking_items = [it for it in items if bool(it.get("blocking", False))]
    has_blocking_features = len(blocking_items) > 0
    is_blocking = mode == "strict" and has_blocking_features

    status = "compatible"
    if has_blocking_features:
        status = "incompatible_strict" if mode == "strict" else "incompatible_non_blocking"
    elif any(it.get("status") == "partial" for it in items):
        status = "compatible_with_warnings"

    recommendations = []
    if import_count > 0:
        if provided_libraries_count == 0:
            recommendations.append(
                "Importer les fichiers de librairie Pine (.txt/.pine) en même temps que la stratégie."
            )
        if unresolved_import_count > 0:
            recommendations.append(
                "Compléter le mapping des imports Pine vers des modules Python locaux (100% requis en strict)."
            )
        else:
            recommendations.append(
                "Mapping imports complété: lancer un test de parité Pine/Python avant usage production."
            )
    if uses_lower_tf:
        recommendations.append(
            "Remplacer ou simplifier request.security_lower_tf pour réduire le risque de non-parité."
        )
    if uses_strategy_order:
        recommendations.append(
            "Remplacer `strategy.order` par des blocs `strategy.entry/exit/close` pour la V3 beta."
        )
    if uses_short_entry:
        recommendations.append(
            "Retirer les entrées short ou attendre le support short/pyramiding d'une version ultérieure."
        )
    if uses_pyramiding:
        recommendations.append(
            "Désactiver le pyramiding pour la phase beta (requis pour fiabilité du replay)."
        )
    if uses_loops or uses_switch:
        recommendations.append(
            "Éviter `for/while/switch` dans la stratégie Pine cible beta ou fournir une stratégie simplifiée."
        )
    if pre_errors:
        recommendations.append(
            "Corriger d'abord les erreurs de pré-analyse (`//@version`, `strategy(...)`, syntaxe)."
        )
    if mode == "strict" and has_blocking_features:
        recommendations.append(
            "Le mode strict bloque ce script. Utiliser `assist` uniquement pour diagnostic exploratoire."
        )
    if not recommendations:
        recommendations.append("Script prêt pour l'étape suivante de compilation `strategy_spec.v1`.")

    return _sanitize_for_json(
        {
            "schema_version": "pine_compatibility.v1",
            "generated_at_utc": _utc_now_iso(),
            "source_name": precheck_report.get("source_name"),
            "compat_mode": mode,
            "status": status,
            "compatibility_score": compatibility_score,
            "has_blocking_features": has_blocking_features,
            "is_blocking": is_blocking,
            "items": items,
            "blocking_items": blocking_items,
            "import_resolution": precheck_report.get("import_resolution", []),
            "external_functions_called_count": external_functions_called_count,
            "external_functions_missing_count": external_functions_missing_count,
            "precheck_warnings": pre_warnings,
            "precheck_errors": pre_errors,
            "recommendations": recommendations,
        }
    )


def _build_pine_beta_readiness_report(
    precheck_report: dict | None,
    compatibility_report: dict | None,
    strategy_spec: dict | None,
    strategy_spec_validation: dict | None,
    codegen_report: dict | None,
    generated_module_path: str | None = None,
):
    """Compute a deterministic beta-readiness gate for Pine V3 execution."""
    pre = precheck_report if isinstance(precheck_report, dict) else {}
    compat = compatibility_report if isinstance(compatibility_report, dict) else {}
    spec = strategy_spec if isinstance(strategy_spec, dict) else {}
    spec_val = strategy_spec_validation if isinstance(strategy_spec_validation, dict) else {}
    codegen = codegen_report if isinstance(codegen_report, dict) else {}
    module_path = str(generated_module_path or "").strip()

    logic = spec.get("logic") if isinstance(spec.get("logic"), dict) else {}
    order_rules = logic.get("order_rules") if isinstance(logic.get("order_rules"), list) else []
    assignments = logic.get("assignments") if isinstance(logic.get("assignments"), list) else []

    runtime_blockers = []
    for item in (compat.get("items") or []):
        if not isinstance(item, dict):
            continue
        if bool(item.get("blocking", False)):
            runtime_blockers.append(
                {
                    "code": item.get("code"),
                    "label": item.get("label"),
                    "detail": item.get("detail"),
                }
            )

    checks = [
        {
            "id": "precheck_valid",
            "label": "Pré-analyse Pine valide",
            "required": True,
            "passed": str(pre.get("status", "")).lower() == "valid",
            "detail": None,
        },
        {
            "id": "compatibility_non_blocking",
            "label": "Compatibilité non bloquante",
            "required": True,
            "passed": not bool(compat.get("is_blocking", False)),
            "detail": f"status={compat.get('status')}",
        },
        {
            "id": "strategy_spec_valid",
            "label": "strategy_spec.v1 valide",
            "required": True,
            "passed": bool(spec_val.get("valid", False)),
            "detail": f"errors={len(spec_val.get('errors', []) or [])}",
        },
        {
            "id": "logic_order_rules",
            "label": "Règles d'ordres extraites",
            "required": True,
            "passed": len(order_rules) > 0,
            "detail": f"order_rules={len(order_rules)}, assignments={len(assignments)}",
        },
        {
            "id": "codegen_ok",
            "label": "Codegen module Python",
            "required": True,
            "passed": str(codegen.get("status", "")).lower() == "ok",
            "detail": f"status={codegen.get('status')}",
        },
        {
            "id": "generated_module_exists",
            "label": "Module généré disponible",
            "required": True,
            "passed": bool(module_path and os.path.exists(module_path)),
            "detail": module_path or None,
        },
        {
            "id": "runtime_blockers_absent",
            "label": "Aucun blocker runtime beta",
            "required": True,
            "passed": len(runtime_blockers) == 0,
            "detail": f"blockers={len(runtime_blockers)}",
        },
    ]

    required_checks = [c for c in checks if bool(c.get("required", False))]
    required_passed = [c for c in required_checks if bool(c.get("passed", False))]
    readiness_score = round((len(required_passed) / max(1, len(required_checks))) * 100.0, 1)
    beta_ready = len(required_passed) == len(required_checks)

    next_actions = []
    for check in required_checks:
        if not bool(check.get("passed", False)):
            next_actions.append(f"{check.get('label')}: {check.get('detail') or 'à corriger'}")
    if not next_actions and runtime_blockers:
        next_actions.extend(
            [f"{b.get('label')}: {b.get('detail')}" for b in runtime_blockers if isinstance(b, dict)]
        )

    return _sanitize_for_json(
        {
            "schema_version": "pine_beta_readiness.v1",
            "generated_at_utc": _utc_now_iso(),
            "beta_ready": bool(beta_ready),
            "status": "ready" if beta_ready else "not_ready",
            "readiness_score": readiness_score,
            "checks": checks,
            "runtime_blockers": runtime_blockers,
            "next_actions": next_actions,
            "strategy_id": (spec.get("strategy") or {}).get("id") if isinstance(spec, dict) else None,
            "compatibility_status": compat.get("status"),
            "compatibility_score": compat.get("compatibility_score"),
        }
    )


def _get_pine_parity_thresholds_from_state() -> dict:
    """Return parity thresholds from session state with safe defaults."""
    thresholds = dict(_DEFAULT_PARITY_THRESHOLDS)
    for key, default_value in _DEFAULT_PARITY_THRESHOLDS.items():
        state_key = f"pine_parity_{key}"
        raw_value = st.session_state.get(state_key, default_value)
        try:
            thresholds[key] = float(raw_value)
        except Exception:
            thresholds[key] = float(default_value)
    return thresholds


def _get_pine_parity_detail_thresholds_from_state() -> dict:
    """Return detailed parity thresholds (events/trades) from session state."""
    thresholds = dict(_DEFAULT_PARITY_DETAIL_THRESHOLDS)
    for key, default_value in _DEFAULT_PARITY_DETAIL_THRESHOLDS.items():
        state_key = f"pine_parity_{key}"
        raw_value = st.session_state.get(state_key, default_value)
        try:
            thresholds[key] = float(raw_value)
        except Exception:
            thresholds[key] = float(default_value)
    return thresholds


def _to_iso_utc(value) -> str | None:
    """Best-effort conversion to UTC ISO timestamp."""
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    try:
        return ts.to_pydatetime().isoformat()
    except Exception:
        return str(ts)


def _extract_current_events_and_trades_for_parity(max_items: int = 30000) -> dict:
    """Extract current runtime entry/exit events and trades for detailed parity."""
    max_rows = int(max(100, max_items))
    trades_df = pd.DataFrame()
    pf = st.session_state.get("final_portfolio")
    index_ref = None

    if pf is not None:
        try:
            trades_df = pd.DataFrame(pf.trades.records)
        except Exception:
            trades_df = pd.DataFrame()
        try:
            if hasattr(pf, "wrapper"):
                index_ref = pf.wrapper.index
        except Exception:
            index_ref = None

    if trades_df.empty and isinstance(st.session_state.get("final_trades_df"), pd.DataFrame):
        trades_df = st.session_state.get("final_trades_df").copy()
    if trades_df.empty:
        return {"entries": [], "exits": [], "trades": []}

    if len(trades_df) > max_rows:
        trades_df = trades_df.head(max_rows).copy()

    def _ts_from_row(row, kind: str) -> str | None:
        direct_cols = [
            f"{kind}_ts",
            f"{kind}_time",
            f"{kind}_timestamp",
        ]
        for col in direct_cols:
            if col in row and pd.notna(row.get(col)):
                iso = _to_iso_utc(row.get(col))
                if iso:
                    return iso

        idx_col = f"{kind}_idx"
        if idx_col in row and index_ref is not None and pd.notna(row.get(idx_col)):
            try:
                idx_val = int(row.get(idx_col))
                if 0 <= idx_val < len(index_ref):
                    return _to_iso_utc(index_ref[idx_val])
            except Exception:
                return None
        return None

    entries: list[str] = []
    exits: list[str] = []
    trades: list[dict] = []

    for _, row in trades_df.iterrows():
        entry_iso = _ts_from_row(row, "entry")
        exit_iso = _ts_from_row(row, "exit")
        if entry_iso:
            entries.append(entry_iso)
        if exit_iso:
            exits.append(exit_iso)
        if entry_iso and exit_iso:
            trade_row = {"entry_time": entry_iso, "exit_time": exit_iso}
            if "pnl" in row and pd.notna(row.get("pnl")):
                try:
                    trade_row["pnl"] = float(row.get("pnl"))
                except Exception:
                    pass
            trades.append(trade_row)

    entries = sorted(set(entries))
    exits = sorted(set(exits))
    return {"entries": entries, "exits": exits, "trades": trades}


def _extract_current_metrics_for_parity() -> dict:
    """Extract current runtime metrics for parity checks from final portfolio."""
    metrics = {}
    pf = st.session_state.get("final_portfolio")
    if pf is not None:
        try:
            n_trades = int(len(pf.trades))
            metrics["trade_count"] = n_trades
            metrics["entry_count"] = n_trades
            metrics["exit_count"] = n_trades
        except Exception:
            pass
        try:
            metrics["total_return_pct"] = float(_safe_float_scalar(getattr(pf, "total_return", np.nan) * 100))
        except Exception:
            pass
        try:
            metrics["max_drawdown_pct"] = float(_safe_float_scalar(getattr(pf, "max_drawdown", np.nan) * 100))
        except Exception:
            pass
        return _normalize_parity_metrics(metrics)

    # Fallback for historical ZIP without final portfolio object.
    final_summary = _extract_final_backtest_for_expert(
        st.session_state.get("wfo_results"),
        get_current_config(),
        df_source=st.session_state.get("df"),
    )
    if isinstance(final_summary, dict):
        if isinstance(final_summary.get("strategy_n_trades"), (int, float)):
            n_trades = float(final_summary.get("strategy_n_trades"))
            metrics["trade_count"] = n_trades
            metrics["entry_count"] = n_trades
            metrics["exit_count"] = n_trades
        if isinstance(final_summary.get("strategy_total_return_pct"), (int, float)):
            metrics["total_return_pct"] = float(final_summary.get("strategy_total_return_pct"))
        if isinstance(final_summary.get("strategy_max_drawdown_pct"), (int, float)):
            metrics["max_drawdown_pct"] = float(final_summary.get("strategy_max_drawdown_pct"))
    return _normalize_parity_metrics(metrics)


def _parse_parity_reference_payload_from_text(
    raw_text: str,
    allow_legacy: bool = True,
) -> tuple[dict, dict, str | None]:
    """Parse and validate parity reference JSON text to canonical v1 payload."""
    txt = str(raw_text or "").strip()
    if not txt:
        return {}, {}, None
    try:
        data = json.loads(txt)
    except Exception as e:
        return {}, {}, f"JSON invalide: {e}"

    validation = _validate_parity_reference_payload(data, allow_legacy=allow_legacy)
    normalized_payload = (
        validation.get("normalized_payload")
        if isinstance(validation, dict) and isinstance(validation.get("normalized_payload"), dict)
        else {}
    )
    if not bool((validation or {}).get("valid", False)):
        err_lines = (validation or {}).get("errors") or []
        if not isinstance(err_lines, list):
            err_lines = [str(err_lines)]
        error_message = "; ".join([str(x) for x in err_lines if str(x).strip()]) or "Référence Pine invalide."
        return normalized_payload, validation, error_message
    return normalized_payload, validation, None


def _apply_parity_reference_payload(
    payload: dict,
    validation: dict | None = None,
    update_text: bool = False,
):
    """Persist canonical parity reference payload and derived session fields."""
    if not isinstance(payload, dict):
        return
    metrics = payload.get("reference_metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    st.session_state["pine_parity_reference_payload"] = payload
    st.session_state["pine_parity_reference_metrics"] = metrics
    if bool(update_text):
        st.session_state["pine_parity_reference_text"] = json.dumps(payload, indent=2, ensure_ascii=False)
    if isinstance(validation, dict):
        st.session_state["pine_parity_reference_validation"] = validation


def _read_text_file_with_fallback(path: str):
    """Read text file using UTF-8 then Latin-1 fallback."""
    with open(path, "rb") as f:
        raw = f.read()
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace"), "latin-1"


def load_best_params_into_inputs():
    final_params = None
    final_source = None
    if 'wfo_results' in st.session_state:
        best_params, best_score, best_window, best_is, best_oos, final_source, robust_summary = _select_final_params_from_results(
            st.session_state['wfo_results'],
            get_current_config()
        )
        if best_params:
            st.session_state['final_params'] = best_params
            st.session_state['final_params_score'] = best_score
            st.session_state['final_params_window'] = best_window
            st.session_state['final_params_is_metrics'] = best_is
            st.session_state['final_params_oos_metrics'] = best_oos
            st.session_state['final_params_source'] = final_source
            st.session_state['final_params_robust_summary'] = robust_summary
            final_params = best_params
            final_source = final_source or "best_window"
    if final_params is None:
        final_params = st.session_state.get("final_params")
        final_source = st.session_state.get("final_params_source")
    if not final_params:
        st.sidebar.error("No final parameters available.")
        return

    int_params = {'timeperiod', 'fenetre_lowest', 'longueur_mediane', 'Nb_bars_above', 'user_exit_sma_length',
                  'macd_fast_length', 'macd_slow_length', 'macd_signal_length'}
    for param in DEFAULT_PARAM_GRID:
        st.session_state[f"check_{param}"] = False

    for param, value in final_params.items():
        if param not in DEFAULT_PARAM_GRID:
            continue
        try:
            if param in int_params:
                value = int(round(float(value)))
            else:
                value = float(value)
        except Exception:
            continue

        st.session_state[f"check_{param}"] = True
        st.session_state[f"min_{param}"] = value
        st.session_state[f"max_{param}"] = value
        st.session_state[f"step_{param}"] = 1 if param in int_params else 0.01

    window_id = st.session_state.get('final_params_window')
    if str(final_source or "").lower() == "robust_set":
        st.sidebar.success("Loaded robust-set parameters into input ranges.")
    elif window_id is not None:
        st.sidebar.success(f"Loaded best parameters from window {window_id}.")
    else:
        st.sidebar.success("Loaded best parameters.")


PARAMETER_HELP = {
    'timeperiod': "Période des bandes de Bollinger (lookback).",
    'StDev': "Nombre d'écarts-types utilisé pour les bandes de Bollinger.",
    'coeff_medianeBBW': "Coefficient du signal de compression/horizontalité BBW.",
    'coef_mediane': "Coefficient du signal écart Bollinger borné.",
    'fenetre_lowest': "Fenêtre utilisée pour détecter les plus bas de BBW.",
    'seuil_lowest': "Seuil appliqué sur le signal 'lowest' de BBW.",
    'longueur_mediane': "Longueur de fenêtre pour la médiane de référence.",
    'Nb_bars_above': "Nombre de barres de validation du signal d'entrée.",
    'user_exit_sma_length': "Longueur de SMA pour le signal de sortie.",
    'sar_start': "Valeur initiale du Parabolic SAR.",
    'sar_increment': "Incrément du Parabolic SAR.",
    'sar_maximum': "Valeur maximale du facteur d'accélération SAR.",
    'macd_fast_length': "Période EMA rapide du MACD.",
    'macd_slow_length': "Période EMA lente du MACD.",
    'macd_signal_length': "Période de la ligne signal MACD."
}

EXPERT_MODEL_CATALOG = {
    "grok": [
        {
            "id": "grok-4-1-fast-reasoning",
            "label": "grok-4-1-fast-reasoning",
            "description": "Optimisé pour le tool-calling agentique et le raisonnement avancé.",
        },
        {
            "id": "grok-4-1-fast-non-reasoning",
            "label": "grok-4-1-fast-non-reasoning",
            "description": "Version plus rapide pour des tâches sans raisonnement lourd.",
        },
    ],
    "openai": [
        {
            "id": "gpt-5.2",
            "label": "GPT-5.2",
            "description": "Meilleur modèle pour le codage et les tâches agentiques complexes.",
        },
        {
            "id": "gpt-5-mini",
            "label": "GPT-5 mini",
            "description": "Version rapide et économique pour des tâches bien définies.",
        },
    ],
    "gemini": [
        {
            "id": "gemini-3-pro-preview",
            "label": "gemini-3-pro-preview",
            "description": "Flagship actuel, excellent en raisonnement multimodal et agentique (preview).",
        },
        {
            "id": "gemini-3-flash-preview",
            "label": "gemini-3-flash-preview",
            "description": "Version rapide et efficace, très performante en raisonnement complexe avec faible latence.",
        },
    ],
}

def _get_expert_model_entries(provider):
    return EXPERT_MODEL_CATALOG.get(str(provider or "").lower(), EXPERT_MODEL_CATALOG["openai"])


def _get_expert_provider_defaults(provider):
    key = str(provider or "").lower()
    if key == "grok":
        return {"base_url": "https://api.x.ai/v1", "label": "Grok"}
    if key == "gemini":
        return {"base_url": "https://generativelanguage.googleapis.com/v1beta", "label": "Gemini"}
    return {"base_url": "https://api.openai.com/v1", "label": "OpenAI"}

ADAPTIVE_PROFILE_DEFS = {
    "custom": {
        "label": "Custom (manuel)",
        "summary": "Aucun preset appliqué; tous les réglages restent manuels.",
        "advantages": "Contrôle total sur chaque hyperparamètre adaptatif.",
        "drawbacks": "Plus de risque d'erreur de calibration, temps moins prévisible.",
        "specificity": "À utiliser si tu maîtrises déjà ton régime de marché et ton budget compute.",
        "duration_note": "Variable selon les valeurs saisies.",
        "params": None
    },
    "smoke_test": {
        "label": "Smoke Test (ultra rapide)",
        "summary": "Validation technique rapide du pipeline et de l'UI.",
        "advantages": "Très rapide, utile pour vérifier que tout fonctionne.",
        "drawbacks": "Peu robuste statistiquement, forte variance.",
        "specificity": "Profil de debug, pas de décision de production.",
        "duration_note": "Très court.",
        "params": {
            "adaptive_train_bars": 20000,
            "adaptive_cycle_bars": 5000,
            "adaptive_trials_per_cycle": 40,
            "adaptive_candidate_pool_size": 400,
            "adaptive_keep_ratio": 0.50,
            "adaptive_exploration_ratio": 0.35,
            "adaptive_min_values_per_param": 2,
            "adaptive_decay": 0.98,
            "adaptive_ucb_beta": 1.00,
            "adaptive_warmup_trials": 150,
            "adaptive_max_cycles": 0,
            "adaptive_oos_weight": 1.5
        }
    },
    "fast": {
        "label": "Rapide",
        "summary": "Bon compromis vitesse/qualité pour itérations fréquentes.",
        "advantages": "Boucles courtes, feedback rapide.",
        "drawbacks": "Moins stable qu'un profil robuste sur longues périodes.",
        "specificity": "Idéal en phase de prototypage ou tuning quotidien.",
        "duration_note": "Court à moyen.",
        "params": {
            "adaptive_train_bars": 86400,
            "adaptive_cycle_bars": 10000,
            "adaptive_trials_per_cycle": 80,
            "adaptive_candidate_pool_size": 1200,
            "adaptive_keep_ratio": 0.45,
            "adaptive_exploration_ratio": 0.25,
            "adaptive_min_values_per_param": 2,
            "adaptive_decay": 0.98,
            "adaptive_ucb_beta": 0.85,
            "adaptive_warmup_trials": 250,
            "adaptive_max_cycles": 0,
            "adaptive_oos_weight": 2.0
        }
    },
    "balanced": {
        "label": "Équilibré (recommandé)",
        "summary": "Compromis robustesse/coût adapté à la plupart des runs.",
        "advantages": "Résultats généralement stables avec durée contenue.",
        "drawbacks": "Plus lent qu'un profil rapide.",
        "specificity": "Point de départ conseillé pour la plupart des backtests.",
        "duration_note": "Moyen.",
        "params": {
            "adaptive_train_bars": 345600,
            "adaptive_cycle_bars": 17280,
            "adaptive_trials_per_cycle": 180,
            "adaptive_candidate_pool_size": 4000,
            "adaptive_keep_ratio": 0.35,
            "adaptive_exploration_ratio": 0.20,
            "adaptive_min_values_per_param": 3,
            "adaptive_decay": 0.985,
            "adaptive_ucb_beta": 0.90,
            "adaptive_warmup_trials": 500,
            "adaptive_max_cycles": 0,
            "adaptive_oos_weight": 2.5
        }
    },
    "robust": {
        "label": "Robuste",
        "summary": "Favorise la stabilité OOS et la régularité.",
        "advantages": "Moins sensible au bruit; meilleure résilience out-of-sample.",
        "drawbacks": "Temps de calcul plus élevé.",
        "specificity": "À privilégier pour les runs de référence.",
        "duration_note": "Long.",
        "params": {
            "adaptive_train_bars": 500000,
            "adaptive_cycle_bars": 15000,
            "adaptive_trials_per_cycle": 260,
            "adaptive_candidate_pool_size": 6000,
            "adaptive_keep_ratio": 0.30,
            "adaptive_exploration_ratio": 0.20,
            "adaptive_min_values_per_param": 3,
            "adaptive_decay": 0.99,
            "adaptive_ucb_beta": 0.75,
            "adaptive_warmup_trials": 800,
            "adaptive_max_cycles": 0,
            "adaptive_oos_weight": 3.0
        }
    },
    "reactive": {
        "label": "Réactif (changement de régime)",
        "summary": "S'adapte plus vite aux shifts de marché.",
        "advantages": "Réagit rapidement aux phases de rupture.",
        "drawbacks": "Plus de variance, risque de sur-réaction.",
        "specificity": "Pertinent si le marché change fréquemment de régime.",
        "duration_note": "Moyen à long.",
        "params": {
            "adaptive_train_bars": 120000,
            "adaptive_cycle_bars": 8000,
            "adaptive_trials_per_cycle": 160,
            "adaptive_candidate_pool_size": 3500,
            "adaptive_keep_ratio": 0.40,
            "adaptive_exploration_ratio": 0.28,
            "adaptive_min_values_per_param": 2,
            "adaptive_decay": 0.97,
            "adaptive_ucb_beta": 1.00,
            "adaptive_warmup_trials": 350,
            "adaptive_max_cycles": 0,
            "adaptive_oos_weight": 2.2
        }
    },
    "conservative": {
        "label": "Conservateur anti-overfit",
        "summary": "Contraint davantage la grille pour maximiser la robustesse.",
        "advantages": "Réduit les risques de sur-ajustement.",
        "drawbacks": "Peut rater des niches de performance.",
        "specificity": "Utile si priorité absolue à la robustesse OOS.",
        "duration_note": "Moyen.",
        "params": {
            "adaptive_train_bars": 345600,
            "adaptive_cycle_bars": 17280,
            "adaptive_trials_per_cycle": 150,
            "adaptive_candidate_pool_size": 3000,
            "adaptive_keep_ratio": 0.30,
            "adaptive_exploration_ratio": 0.25,
            "adaptive_min_values_per_param": 3,
            "adaptive_decay": 0.99,
            "adaptive_ucb_beta": 0.80,
            "adaptive_warmup_trials": 800,
            "adaptive_max_cycles": 0,
            "adaptive_oos_weight": 3.0
        }
    },
    "exploratory": {
        "label": "Exploratoire",
        "summary": "Recherche agressive de nouvelles zones de paramètres.",
        "advantages": "Découverte plus large de combinaisons candidates.",
        "drawbacks": "Coût compute élevé, résultats parfois moins stables.",
        "specificity": "Adapté pour ouvrir la recherche avant un profil robuste.",
        "duration_note": "Long à très long.",
        "params": {
            "adaptive_train_bars": 200000,
            "adaptive_cycle_bars": 10000,
            "adaptive_trials_per_cycle": 220,
            "adaptive_candidate_pool_size": 8000,
            "adaptive_keep_ratio": 0.55,
            "adaptive_exploration_ratio": 0.35,
            "adaptive_min_values_per_param": 2,
            "adaptive_decay": 0.98,
            "adaptive_ucb_beta": 1.10,
            "adaptive_warmup_trials": 400,
            "adaptive_max_cycles": 0,
            "adaptive_oos_weight": 2.0
        }
    }
}

# NOTE: runtime/date helpers are now provided by `services.runtime_utils`.

def _inject_running_animation_css():
    st.markdown(
        """
        <style>
        @keyframes runPulse {
            0% { transform: scale(1); box-shadow: 0 0 0 rgba(76, 175, 255, 0.0); }
            70% { transform: scale(1.08); box-shadow: 0 0 0 8px rgba(76, 175, 255, 0.0); }
            100% { transform: scale(1); box-shadow: 0 0 0 rgba(76, 175, 255, 0.0); }
        }
        @keyframes runShimmer {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }
        @keyframes stopPulse {
            0% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.30); }
            70% { box-shadow: 0 0 0 10px rgba(255, 75, 75, 0.00); }
            100% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.00); }
        }
        .run-status-card {
            border: 1px solid rgba(111, 168, 220, 0.55);
            border-radius: 12px;
            background: linear-gradient(135deg, rgba(16, 34, 54, 0.88), rgba(17, 43, 71, 0.88));
            padding: 0.8rem 0.95rem;
            margin: 0.3rem 0 0.8rem 0;
        }
        .run-status-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 700;
            color: #e8f3ff;
            margin-bottom: 0.45rem;
        }
        .run-status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #4cafef;
            animation: runPulse 1.6s ease-in-out infinite;
        }
        .run-status-meta {
            color: #b9d6f3;
            font-size: 0.90rem;
            line-height: 1.4;
        }
        .run-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            border: 1px solid rgba(102, 182, 255, 0.55);
            border-radius: 999px;
            background: rgba(20, 51, 84, 0.88);
            color: #dff0ff;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            padding: 0.20rem 0.55rem;
            margin: 0.2rem 0 0.35rem 0;
        }
        .run-badge-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #4cafef;
            animation: runPulse 1.4s ease-in-out infinite;
        }
        section[data-testid="stSidebar"] div[data-testid="stProgressBar"] div[role="progressbar"] > div {
            background-image: linear-gradient(110deg, #3a93ff 20%, #88c6ff 40%, #3a93ff 60%);
            background-size: 220% 100%;
            animation: runShimmer 2.1s linear infinite;
        }
        .st-key-stop_wfo_btn button {
            border: 1px solid rgba(255, 109, 109, 0.72) !important;
            background: linear-gradient(135deg, rgba(95, 24, 24, 0.88), rgba(72, 20, 20, 0.88)) !important;
            animation: stopPulse 1.9s ease-out infinite;
        }
        .st-key-stop_wfo_btn button:hover {
            background: linear-gradient(135deg, rgba(120, 32, 32, 0.92), rgba(95, 24, 24, 0.92)) !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def _render_running_status_card(job_state, config):
    if not isinstance(job_state, dict):
        return
    message = str(job_state.get("message") or "Optimization running...")
    progress = float(job_state.get("progress", 0.0) or 0.0)
    progress = min(max(progress, 0.0), 1.0)

    elapsed = _compute_running_elapsed_seconds(job_state)
    elapsed_text = _humanize_seconds(elapsed) if elapsed is not None else "n/a"

    eta_seconds = job_state.get("eta_seconds")
    if eta_seconds is None and elapsed is not None and progress > 1e-6:
        eta_seconds = elapsed * (1.0 - progress) / progress
    eta_text = _humanize_seconds(float(eta_seconds)) if eta_seconds is not None else "n/a"

    window_text = job_state.get("window")
    evaluations = job_state.get("evaluations")
    run_id = str(job_state.get("run_id") or "n/a")
    run_short = run_id[-10:] if len(run_id) > 10 else run_id
    mode = str(config.get("optimization_regime", "classic")).lower()
    mode_label = (
        "Adaptive Continuous" if mode == "adaptive_continuous"
        else "WFO classique"
    )

    meta_parts = [
        f"Mode: {mode_label}",
        f"Progression: {progress * 100:.1f}%",
        f"Elapsed: {elapsed_text}",
        f"ETA: {eta_text}",
        f"Run ID: {html.escape(run_short)}"
    ]
    if window_text is not None:
        meta_parts.append(f"Fenêtre/Cycle: {window_text}")
    if evaluations is not None:
        meta_parts.append(f"Évaluations (dernier update): {evaluations}")

    meta_html = "<br>".join(html.escape(str(x)) for x in meta_parts)
    card_html = f"""
    <div class="run-status-card">
        <div class="run-status-header"><span class="run-status-dot"></span>Optimisation en cours</div>
        <div class="run-status-meta"><strong>Statut:</strong> {html.escape(message)}<br>{meta_html}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def _estimate_adaptive_load(start_date, end_date, timeframe_str, train_bars, cycle_bars, trials_per_cycle, max_cycles):
    step_seconds = _timeframe_to_seconds(timeframe_str)
    start_dt = _parse_iso_date(start_date)
    end_dt = _parse_iso_date(end_date)
    if step_seconds is None or start_dt is None or end_dt is None:
        return None
    if end_dt <= start_dt:
        return None

    total_seconds = (end_dt - start_dt).total_seconds()
    n_bars = int(total_seconds // step_seconds) + 1
    train_bars = int(max(1, train_bars))
    cycle_bars = int(max(1, cycle_bars))
    trials_per_cycle = int(max(1, trials_per_cycle))
    max_cycles = int(max(0, max_cycles))

    if n_bars <= train_bars:
        cycles = 0
    else:
        cycles = int((n_bars - train_bars) // cycle_bars)
    if max_cycles > 0:
        cycles = min(cycles, max_cycles)

    total_trials = int(cycles * trials_per_cycle)
    return {
        "bars": int(n_bars),
        "cycles": int(cycles),
        "total_trials": total_trials
    }

def _get_observed_seconds_per_trial():
    results = st.session_state.get("wfo_results")
    if not isinstance(results, dict):
        return None
    timing = results.get("timing", {})
    if not isinstance(timing, dict):
        return None
    total_time = timing.get("total_time")
    total_trials = timing.get("total_trials")
    try:
        total_time = float(total_time)
        total_trials = float(total_trials)
        if total_time > 0 and total_trials > 0:
            return total_time / total_trials
    except Exception:
        return None
    return None

with st.sidebar:
    st.header("⚙️ Configuration")
    st.caption("Parcours rapide: 1) Données 2) Paramètres 3) Moteur WFO 4) Lancer 5) Exporter")

    has_final_params = 'final_params' in st.session_state or 'wfo_results' in st.session_state
    st.sidebar.button(
        "📥 Load Best Params into Inputs",
        width="stretch",
        on_click=load_best_params_into_inputs if has_final_params else None,
        disabled=not has_final_params,
        help="Charge les meilleurs paramètres trouvés dans les champs Min/Max/Step pour préparer un nouveau run."
    )
    if not has_final_params:
        st.sidebar.info("Run the final backtest to enable loading best parameters.")
    
    # --- File Uploader for Config ---
    uploaded_config = st.file_uploader(
        "📂 Load Config (JSON)",
        type=['json'],
        help="Importe une configuration sauvegardée et met à jour les contrôles de la sidebar."
    )
    
    if uploaded_config is not None:
        try:
            # Use file_id (or name+size as proxy) to detect if it's a new file upload
            # Streamlit reruns script on interaction, so we must not re-apply config if file hasn't changed.
            file_id = getattr(uploaded_config, 'file_id', uploaded_config.name + str(uploaded_config.size))
            
            if 'last_loaded_file_id' not in st.session_state or st.session_state['last_loaded_file_id'] != file_id:
                loaded_config = json.load(uploaded_config)
                st.session_state['loaded_config'] = loaded_config
                st.session_state['last_loaded_file_id'] = file_id
                
                # --- APPLY CONFIG TO WIDGET STATE ---
                # 1. General Settings
                state_map = {
                    'start_date': 'start_date', 'end_date': 'end_date', 'timeframe': 'timeframe',
                    'strategy_mode': 'strategy_mode', 'strategy_id': 'strategy_id',
                    'pine_file_path': 'pine_file_path', 'pine_compat_mode': 'pine_compat_mode',
                    'pine_spec_parser_backend': 'pine_spec_parser_backend',
                    'pine_generated_module_path': 'pine_generated_module_path',
                    'pine_library_paths': 'pine_library_paths', 'pine_library_names': 'pine_library_names',
                    'pine_import_mapping': 'pine_import_mapping',
                    'pine_parity_trade_count_rel_pct': 'pine_parity_trade_count_rel_pct',
                    'pine_parity_entry_count_rel_pct': 'pine_parity_entry_count_rel_pct',
                    'pine_parity_exit_count_rel_pct': 'pine_parity_exit_count_rel_pct',
                    'pine_parity_total_return_abs_pct': 'pine_parity_total_return_abs_pct',
                    'pine_parity_max_drawdown_abs_pct': 'pine_parity_max_drawdown_abs_pct',
                    'pine_parity_entry_event_count_rel_pct': 'pine_parity_entry_event_count_rel_pct',
                    'pine_parity_exit_event_count_rel_pct': 'pine_parity_exit_event_count_rel_pct',
                    'pine_parity_entry_event_match_min_ratio': 'pine_parity_entry_event_match_min_ratio',
                    'pine_parity_exit_event_match_min_ratio': 'pine_parity_exit_event_match_min_ratio',
                    'pine_parity_trade_match_min_ratio': 'pine_parity_trade_match_min_ratio',
                    'pine_parity_event_time_tolerance_sec': 'pine_parity_event_time_tolerance_sec',
                    'pine_parity_trade_time_tolerance_sec': 'pine_parity_trade_time_tolerance_sec',
                    'file_path': 'file_path', 'n_windows': 'n_windows', 'train_size': 'train_size',
                    'anchored': 'anchored', 'optimization_method': 'optimization_method',
                    'optimization_regime': 'optimization_regime',
                    'parallel_backend': 'parallel_backend', 'max_workers': 'max_workers',
                    'use_numba': 'use_numba', 'metric1_name': 'metric1_name', 
                    'metric2_name': 'metric2_name', 'weight_metric1': 'weight_metric1',
                    'weight_metric2': 'weight_metric2', 'patience_level': 'patience_level',
                    'max_trials': 'max_trials', 'neighbor_count': 'neighbor_count',
                    'nn_min_samples': 'nn_min_samples',
                    'nn_candidate_pool_size': 'nn_candidate_pool_size',
                    'nn_top_k': 'nn_top_k',
                    'nn_exploration_ratio': 'nn_exploration_ratio',
                    'nn_hidden_size': 'nn_hidden_size',
                    'nn_epochs': 'nn_epochs',
                    'nn_learning_rate': 'nn_learning_rate',
                    'nn_l2': 'nn_l2',
                    'adaptive_train_bars': 'adaptive_train_bars',
                    'adaptive_cycle_bars': 'adaptive_cycle_bars',
                    'adaptive_trials_per_cycle': 'adaptive_trials_per_cycle',
                    'adaptive_candidate_pool_size': 'adaptive_candidate_pool_size',
                    'adaptive_profile': 'adaptive_profile',
                    'adaptive_keep_ratio': 'adaptive_keep_ratio',
                    'adaptive_exploration_ratio': 'adaptive_exploration_ratio',
                    'adaptive_min_values_per_param': 'adaptive_min_values_per_param',
                    'adaptive_decay': 'adaptive_decay',
                    'adaptive_ucb_beta': 'adaptive_ucb_beta',
                    'adaptive_warmup_trials': 'adaptive_warmup_trials',
                    'adaptive_max_cycles': 'adaptive_max_cycles',
                    'adaptive_oos_weight': 'adaptive_oos_weight',
                    'robust_tests_enabled': 'robust_tests_enabled',
                    'robust_top_n_per_window': 'robust_top_n_per_window',
                    'robust_min_windows': 'robust_min_windows',
                    'robust_use_for_final_backtest': 'robust_use_for_final_backtest',
                    'exit_sar_enabled': 'exit_sar_enabled', 'exit_macd_enabled': 'exit_macd_enabled',
                    'exit_macd_type_a': 'exit_macd_type_a', 'exit_macd_type_b': 'exit_macd_type_b',
                    'order_sizing_mode': 'order_sizing_mode', 'order_fixed_cash': 'order_fixed_cash',
                    'fees_pct': 'fees_pct'
                }
                for conf_key, widget_key in state_map.items():
                    if conf_key in loaded_config:
                        st.session_state[widget_key] = loaded_config[conf_key]
                
                # 2. Data Source
                if 'from_file' in loaded_config:
                    st.session_state['data_source'] = "Local File" if loaded_config['from_file'] else "Binance API"
                    
                # 3. Parameters (Ranges and Selection)
                for param in DEFAULT_PARAM_GRID:
                    # Checkbox
                    if 'selected_params' in loaded_config:
                        st.session_state[f"check_{param}"] = param in loaded_config['selected_params']
                    
                    # Ranges
                    if f'{param}_min' in loaded_config: st.session_state[f"min_{param}"] = loaded_config[f'{param}_min']
                    if f'{param}_max' in loaded_config: st.session_state[f"max_{param}"] = loaded_config[f'{param}_max']
                    if f'{param}_step' in loaded_config: st.session_state[f"step_{param}"] = loaded_config[f'{param}_step']

                # 4. Pine libraries manifest (best-effort restore from configured paths)
                loaded_lib_paths = loaded_config.get("pine_library_paths")
                loaded_lib_names = loaded_config.get("pine_library_names")
                if isinstance(loaded_lib_paths, list) and loaded_lib_paths:
                    rebuilt = []
                    for i, p in enumerate(loaded_lib_paths):
                        path = str(p or "").strip()
                        if not path or not os.path.exists(path):
                            continue
                        source_name = None
                        if isinstance(loaded_lib_names, list) and i < len(loaded_lib_names):
                            source_name = str(loaded_lib_names[i] or "").strip()
                        if not source_name:
                            source_name = os.path.basename(path)
                        try:
                            text, _ = _read_text_file_with_fallback(path)
                            source_sha1 = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
                        except Exception:
                            source_sha1 = ""
                        rebuilt.append(
                            {
                                "source_name": source_name,
                                "path": path,
                                "source_sha1": source_sha1,
                                "size_bytes": int(os.path.getsize(path)),
                            }
                        )
                    st.session_state["pine_library_files"] = rebuilt
                    st.session_state["pine_library_paths"] = [str(item.get("path") or "") for item in rebuilt]
                    st.session_state["pine_library_names"] = [str(item.get("source_name") or "") for item in rebuilt]

                if loaded_config.get('from_file'):
                    sync_dates_from_file(force=True)

                st.success(f"Loaded config: {uploaded_config.name}")
        except Exception as e:
            st.error(f"Error loading config: {e}")

    # --- Results Loader ---
    uploaded_results = st.file_uploader(
        "📦 Load Results (ZIP)",
        type=['zip'],
        help="Recharge des résultats exportés (métriques, paramètres, éventuellement trades et df)."
    )
    if uploaded_results is not None:
        _load_results_zip(uploaded_results)
    
    # --- Data Settings ---
    with st.expander("1. Data Configuration", expanded=True):
        pending_start_date = st.session_state.pop("pending_start_date", None)
        pending_end_date = st.session_state.pop("pending_end_date", None)
        if pending_start_date is not None:
            st.session_state["start_date"] = pending_start_date
        if pending_end_date is not None:
            st.session_state["end_date"] = pending_end_date

        # NOTE: Removed 'get_conf' usage for value=. The value argument is only used for initialization
        # when key is NOT in session_state. If key IS in session_state (e.g. from loader above), 
        # Streamlit ignores value=. This allows user edits to persist.
        
        start_date = st.text_input(
            "Start Date (YYYY-MM-DD)",
            value=DEFAULT_START_DATE,
            key='start_date',
            help="Date de début utilisée pour charger les données d'optimisation."
        )
        end_date = st.text_input(
            "End Date (YYYY-MM-DD)",
            value=DEFAULT_END_DATE,
            key='end_date',
            help="Date de fin utilisée pour charger les données d'optimisation."
        )
        
        # Timeframe selection
        tf_options = ['1s', '5s', '10s', '15s', '30s', '1m', '5m', '15m', '30m', '1h', '4h', '1d']
        default_tf_idx = tf_options.index(DEFAULT_TIMEFRAME) if DEFAULT_TIMEFRAME in tf_options else 1
        timeframe = st.selectbox(
            "Timeframe",
            options=tf_options,
            index=default_tf_idx,
            key='timeframe',
            help="Résolution temporelle des bougies utilisées par la stratégie et le backtest."
        )
        
        # Data Source
        ds_options = ["Local File", "Binance API"]
        # Default index 0 (Local File) if not in state
        data_source = st.radio(
            "Data Source",
            options=ds_options,
            index=0,
            key='data_source',
            help="Choisis entre un fichier local et un chargement via API Binance."
        )
        
        if data_source == "Local File":
            uploaded_market_csv = st.file_uploader(
                "Browse CSV file from disk",
                type=["csv"],
                key="market_data_file_upload",
                help="Choisis un fichier CSV depuis ton disque. Le fichier est copié localement pour être utilisé par le run."
            )
            if uploaded_market_csv is not None:
                uploaded_path, is_new_upload = _persist_uploaded_data_file(uploaded_market_csv)
                if uploaded_path:
                    st.session_state["file_path"] = uploaded_path
                    if is_new_upload:
                        sync_dates_from_file(force=True)
                    st.caption(f"Selected file: `{uploaded_market_csv.name}`")

            file_path = st.text_input(
                "File Path",
                value=DEFAULT_DATA_FILE,
                key='file_path',
                on_change=sync_dates_from_file,
                help="Chemin du CSV OHLCV local. Les dates peuvent être synchronisées automatiquement avec le fichier."
            )
            uploaded_path = st.session_state.get("uploaded_data_file_path")
            if isinstance(uploaded_path, str) and os.path.exists(uploaded_path):
                st.caption(f"Uploaded local copy: `{uploaded_path}`")
            if not os.path.exists(file_path):
                st.error("File not found! Please check the path.")
        else:
            file_path = DEFAULT_DATA_FILE

    # Helper to create param inputs
    def param_input(key, label, default_min, default_max, default_step):
        display_label = str(label).replace("_", " ")
        # Row 1: activation + parameter label on full width (prevents crushed names).
        c_toggle, c_label = st.columns([0.14, 0.86])
        with c_toggle:
            enabled = st.checkbox(
                "Activer",
                value=True,
                key=f"check_{key}",
                label_visibility="collapsed",
                help=PARAMETER_HELP.get(key, "Active/désactive ce paramètre dans l'optimisation.")
            )
        with c_label:
            st.markdown(f"**{display_label}**")
            st.caption(PARAMETER_HELP.get(key, ""))

        # Row 2: numeric controls in a clean 3-column grid.
        c2, c3, c4 = st.columns([1, 1, 1])
        with c2:
            min_val = st.number_input(
                "Min",
                value=float(default_min),
                key=f"min_{key}",
                disabled=not enabled,
                help=f"Borne minimale testée pour `{key}`."
            )
        with c3:
            max_val = st.number_input(
                "Max",
                value=float(default_max),
                key=f"max_{key}",
                disabled=not enabled,
                help=f"Borne maximale testée pour `{key}`."
            )
        with c4:
            step_val = st.number_input(
                "Step",
                value=float(default_step),
                key=f"step_{key}",
                disabled=not enabled,
                help=f"Pas d'incrément entre Min et Max pour `{key}`."
            )
        st.markdown(
            "<div style='height: 0.15rem; border-bottom: 1px solid rgba(120,145,170,0.20); margin: 0.25rem 0 0.45rem 0;'></div>",
            unsafe_allow_html=True
        )
        return enabled, min_val, max_val, step_val

    # --- Entry Parameters ---
    with st.expander("2. Entry Parameters", expanded=False):
        st.info("Configure l'espace de recherche des paramètres d'entrée.")
        
        entry_params = [
            'timeperiod', 'StDev', 'coeff_medianeBBW', 'coef_mediane',
            'fenetre_lowest', 'seuil_lowest', 'longueur_mediane', 'Nb_bars_above'
        ]

        config_params = {}
        selected_params = []

        for param in entry_params:
            d_min, d_max, d_step = DEFAULT_PARAM_GRID[param]
            enabled, p_min, p_max, p_step = param_input(param, param, d_min, d_max, d_step)
            if enabled:
                selected_params.append(param)
                config_params[f'{param}_min'] = p_min
                config_params[f'{param}_max'] = p_max
                config_params[f'{param}_step'] = p_step

    # --- Exit Parameters ---
    with st.expander("3. Exit Parameters", expanded=False):
        st.info("Configure l'espace de recherche des paramètres de sortie.")

        exit_sar_enabled = st.checkbox(
            "Enable Parabolic SAR Exit",
            value=True,
            key='exit_sar_enabled'
        )

        exit_macd_enabled = st.checkbox(
            "Enable MACD Exit",
            value=True,
            key='exit_macd_enabled'
        )
        exit_macd_type_a = st.checkbox(
            "MACD Exit Type A (signal falling)",
            value=True,
            key='exit_macd_type_a'
        )
        exit_macd_type_b = st.checkbox(
            "MACD Exit Type B (simple crossunder)",
            value=True,
            key='exit_macd_type_b'
        )

        exit_params = [
            'user_exit_sma_length', 'sar_start', 'sar_increment', 'sar_maximum',
            'macd_fast_length', 'macd_slow_length', 'macd_signal_length'
        ]
        for param in exit_params:
            d_min, d_max, d_step = DEFAULT_PARAM_GRID[param]
            enabled, p_min, p_max, p_step = param_input(param, param, d_min, d_max, d_step)
            if enabled:
                selected_params.append(param)
                config_params[f'{param}_min'] = p_min
                config_params[f'{param}_max'] = p_max
                config_params[f'{param}_step'] = p_step

    # --- WFO Settings ---
    with st.expander("4. WFO Engine Settings", expanded=False):
        if "strategy_mode" not in st.session_state:
            st.session_state["strategy_mode"] = DEFAULT_STRATEGY_MODE
        if "strategy_id" not in st.session_state:
            st.session_state["strategy_id"] = DEFAULT_STRATEGY_ID
        pending_strategy_id = str(st.session_state.pop("pending_strategy_id", "") or "").strip()
        if pending_strategy_id and st.session_state.get("strategy_mode") != "native_atdmf":
            st.session_state["strategy_id"] = pending_strategy_id
        if st.session_state.get("strategy_mode") == "native_atdmf":
            st.session_state["strategy_id"] = DEFAULT_STRATEGY_ID

        strategy_mode = st.selectbox(
            "Strategy Mode",
            options=["native_atdmf", "pine_imported"],
            index=0 if st.session_state.get("strategy_mode", DEFAULT_STRATEGY_MODE) == "native_atdmf" else 1,
            key="strategy_mode",
            format_func=lambda v: "Native ATDMF" if v == "native_atdmf" else "Pine Imported (V3)",
            help=(
                "Sélectionne la source logique de stratégie. "
                "`native_atdmf` utilise le moteur actuel. "
                "`pine_imported` est réservé à la V3 (pipeline d'import Pine)."
            ),
        )
        strategy_id = st.text_input(
            "Strategy ID",
            key="strategy_id",
            disabled=(strategy_mode == "native_atdmf"),
            help=(
                "Identifiant fonctionnel de la stratégie (traçabilité/export). "
                "En mode natif, l'ID est forcé automatiquement."
            ),
        )
        if strategy_mode != "native_atdmf":
            st.warning(
                "Mode `pine_imported` en phase expérimentale: exécutable uniquement pour "
                "`strategy_test.txt` (runtime V3 block 1)."
            )
            st.caption("Référence Pine v6 (LLM): https://github.com/codenamedevan/pinescriptv6")
            if "pine_compat_mode" not in st.session_state:
                st.session_state["pine_compat_mode"] = "strict"
            if "pine_spec_parser_backend" not in st.session_state:
                st.session_state["pine_spec_parser_backend"] = "auto"
            pine_compat_mode = st.selectbox(
                "Pine Compatibility Mode",
                options=["strict", "assist", "manual"],
                index=["strict", "assist", "manual"].index(
                    st.session_state.get("pine_compat_mode", "strict")
                    if st.session_state.get("pine_compat_mode", "strict") in ["strict", "assist", "manual"]
                    else "strict"
                ),
                key="pine_compat_mode",
                format_func=lambda v: (
                    "Strict (bloque S0)"
                    if v == "strict"
                    else ("Assist (diagnostic permissif)" if v == "assist" else "Manual (analyse seule)")
                ),
                help=(
                    "`strict`: bloque si features incompatibles (S0). "
                    "`assist`: n'empêche pas l'analyse mais signale les risques. "
                    "`manual`: mode exploratoire sans blocage automatique."
                ),
            )
            st.selectbox(
                "Pine Spec Parser Backend",
                options=["auto", "regex", "pynescript"],
                index=["auto", "regex", "pynescript"].index(
                    st.session_state.get("pine_spec_parser_backend", "auto")
                    if st.session_state.get("pine_spec_parser_backend", "auto") in ["auto", "regex", "pynescript"]
                    else "auto"
                ),
                key="pine_spec_parser_backend",
                format_func=lambda v: (
                    "Auto (pynescript -> fallback regex)"
                    if v == "auto"
                    else ("Regex déterministe" if v == "regex" else "pynescript (AST, expérimental)")
                ),
                help=(
                    "Choisit le backend d'analyse pour générer `strategy_spec.v1`. "
                    "`auto` tente `pynescript` puis bascule en regex si indisponible/échec. "
                    "`regex` force le parser déterministe actuel. "
                    "`pynescript` force AST avec fallback regex sécurisé."
                ),
            )
        else:
            pine_compat_mode = st.session_state.get("pine_compat_mode", "strict")

        uploaded_pine_strategy = st.file_uploader(
            "Import Pine Strategy (.txt/.pine)",
            type=["txt", "pine"],
            key="uploaded_pine_strategy",
            disabled=(strategy_mode != "pine_imported"),
            help=(
                "Charge un fichier texte contenant une stratégie Pine Script. "
                "Le fichier est sauvegardé localement puis pré-analysé."
            ),
        )
        uploaded_pine_path, is_new_pine_upload = None, False
        if strategy_mode == "pine_imported":
            uploaded_pine_path, is_new_pine_upload = _persist_uploaded_pine_file(uploaded_pine_strategy)
            if uploaded_pine_path and is_new_pine_upload:
                st.success(f"Fichier Pine importé: {uploaded_pine_path}")

        uploaded_pine_libraries = st.file_uploader(
            "Import Pine Libraries (.txt/.pine)",
            type=["txt", "pine"],
            key="uploaded_pine_libraries",
            accept_multiple_files=True,
            disabled=(strategy_mode != "pine_imported"),
            help=(
                "Ajoute les fichiers de librairie Pine utilisés par la stratégie (lignes `import ...`). "
                "Ces fichiers sont sauvegardés localement et exportés avec les résultats."
            ),
        )
        if strategy_mode == "pine_imported":
            pine_libraries_manifest, has_new_libraries = _persist_uploaded_pine_library_files(uploaded_pine_libraries)
            if has_new_libraries:
                st.success(f"Librairies Pine importées: {len(pine_libraries_manifest)} fichier(s).")
            if isinstance(pine_libraries_manifest, list) and pine_libraries_manifest:
                with st.expander("Librairies Pine associées", expanded=False):
                    st.caption("Ces librairies accompagneront la stratégie dans les exports ZIP.")
                    for lib in pine_libraries_manifest:
                        if not isinstance(lib, dict):
                            continue
                        st.caption(
                            f"- `{lib.get('source_name')}` | SHA1: `{str(lib.get('source_sha1') or '')[:12]}`"
                        )

        pine_file_path = st.text_input(
            "Pine File Path",
            value=st.session_state.get("pine_file_path", ""),
            key="pine_file_path",
            disabled=(strategy_mode != "pine_imported"),
            help=(
                "Chemin du fichier Pine à analyser (précheck). "
                "Tu peux aussi importer via le bouton ci-dessus."
            ),
        )
        pine_precheck_report = None
        if strategy_mode == "pine_imported":
            provided_library_files = st.session_state.get("pine_library_files", [])
            if not isinstance(provided_library_files, list):
                provided_library_files = []
            current_import_mapping = st.session_state.get("pine_import_mapping", {})
            if not isinstance(current_import_mapping, dict):
                current_import_mapping = {}

            parsed_import_entries = []
            pine_path_value = str(pine_file_path or "").strip()
            if pine_path_value and os.path.exists(pine_path_value):
                try:
                    pine_text_for_mapping, _ = _read_text_file_with_fallback(pine_path_value)
                    import_lines_for_mapping = re.findall(
                        r"^\s*import\s+.+$",
                        pine_text_for_mapping,
                        flags=re.IGNORECASE | re.MULTILINE,
                    )
                    parsed_import_entries = _parse_pine_import_lines(import_lines_for_mapping)
                except Exception:
                    parsed_import_entries = []

            with st.expander("Mapping imports Pine -> modules Python (P1.3 assisté)", expanded=False):
                st.caption(
                    "Renseigne un mapping par import (`alias` recommandé) vers un module Python "
                    "local (`chemin.py` ou `package.module`)."
                )
                updated_mapping = {}
                if parsed_import_entries:
                    for idx, imp in enumerate(parsed_import_entries, start=1):
                        if not isinstance(imp, dict):
                            continue
                        label = str(imp.get("alias") or imp.get("module_ref") or f"import_{idx}")
                        key_candidates = [
                            str(imp.get("alias") or "").strip(),
                            str(imp.get("module_ref") or "").strip(),
                            _normalize_token(str(imp.get("alias") or "").strip()),
                        ]
                        existing_value = ""
                        for k in key_candidates:
                            if k and k in current_import_mapping and str(current_import_mapping.get(k) or "").strip():
                                existing_value = str(current_import_mapping.get(k) or "").strip()
                                break

                        mapping_value = st.text_input(
                            f"Import `{label}`",
                            value=existing_value,
                            key=f"pine_import_map_{idx}_{_normalize_token(label) or 'import'}",
                            help=(
                                "Exemples: `apps/wfo_engine/pine_libs/bbt1.py` "
                                "ou `apps.wfo_engine.pine_libs.bbt1`."
                            ),
                        ).strip()
                        if key_candidates[0]:
                            updated_mapping[key_candidates[0]] = mapping_value
                        elif key_candidates[1]:
                            updated_mapping[key_candidates[1]] = mapping_value

                        if mapping_value:
                            ok_target, detail_target = _validate_python_mapping_target(mapping_value)
                            if ok_target:
                                st.caption(f"Validation: OK ({detail_target})")
                            else:
                                st.caption(f"Validation: KO ({detail_target})")
                        else:
                            st.caption("Validation: mapping absent")
                else:
                    st.info("Aucun import Pine détecté dans le fichier courant.")

                st.session_state["pine_import_mapping"] = {
                    str(k): str(v).strip()
                    for k, v in updated_mapping.items()
                    if str(v).strip()
                }

            if str(pine_file_path or "").strip():
                pine_precheck_report = _precheck_pine_script_file(
                    pine_file_path,
                    provided_library_files=provided_library_files,
                    import_mapping=st.session_state.get("pine_import_mapping", {}),
                )
            else:
                st.info("Importe une stratégie Pine pour lancer la pré-analyse.")
                st.session_state.pop("pine_precheck_report", None)
                st.session_state.pop("pine_compatibility_report", None)
                st.session_state.pop("pine_strategy_spec", None)
                st.session_state.pop("pine_strategy_spec_validation", None)
                st.session_state.pop("pine_codegen_report", None)
                st.session_state.pop("pine_generated_module_path", None)
                st.session_state.pop("pine_generation_trace", None)
                st.session_state.pop("pine_artifacts_manifest", None)
                st.session_state.pop("pine_beta_readiness_report", None)
                st.session_state.pop("pine_execution_gate_report", None)
                st.session_state.pop("pine_parity_report", None)
                st.session_state.pop("pine_request_security_diagnostics", None)
                st.session_state.pop("pine_mtf_parity_proof_report", None)
                st.session_state.pop("pine_parity_reference_payload", None)
                st.session_state.pop("pine_parity_reference_validation", None)
                st.session_state.pop("pine_parity_reference_metrics", None)
                st.session_state.pop("pine_parity_reference_text", None)
                st.session_state.pop("pending_strategy_id", None)
                st.session_state.pop("pine_source_text", None)
                st.session_state.pop("pine_library_files", None)
                st.session_state.pop("pine_library_paths", None)
                st.session_state.pop("pine_library_names", None)
                st.session_state.pop("pine_import_mapping", None)

            if isinstance(pine_precheck_report, dict):
                st.session_state["pine_precheck_report"] = pine_precheck_report
                st.session_state["pine_source_name"] = (
                    pine_precheck_report.get("source_name") or st.session_state.get("pine_source_name")
                )
                pine_compatibility_report = _build_pine_compatibility_report(
                    pine_precheck_report,
                    compat_mode=st.session_state.get("pine_compat_mode", "strict"),
                )
                st.session_state["pine_compatibility_report"] = pine_compatibility_report

                status = str(pine_precheck_report.get("status", "invalid")).lower()
                if status == "valid":
                    st.success("Pré-analyse Pine: valide")
                else:
                    st.error("Pré-analyse Pine: invalide")

                for err in pine_precheck_report.get("errors", []):
                    st.caption(f"Erreur: {err}")
                for warn in pine_precheck_report.get("warnings", []):
                    st.caption(f"Avertissement: {warn}")

                with st.expander("Détails pré-analyse Pine", expanded=False):
                    c_pre_a, c_pre_b, c_pre_c = st.columns(3)
                    c_pre_a.metric("Version", str(pine_precheck_report.get("detected_version", "n/a")))
                    c_pre_b.metric("Lignes", str(pine_precheck_report.get("line_count", "n/a")))
                    c_pre_c.metric("SHA1 source", str(pine_precheck_report.get("source_sha1", "n/a"))[:12] + "...")
                    st.json(pine_precheck_report)

                import_resolution_rows = pine_precheck_report.get("import_resolution", []) or []
                if import_resolution_rows:
                    with st.expander("Résolution des imports Pine", expanded=False):
                        df_resolution = pd.DataFrame(import_resolution_rows)
                        st.dataframe(df_resolution, width="stretch")

                if isinstance(pine_compatibility_report, dict):
                    c_comp_1, c_comp_2, c_comp_3 = st.columns(3)
                    c_comp_1.metric(
                        "Compat Score",
                        f"{float(pine_compatibility_report.get('compatibility_score', 0.0)):.1f}/100",
                    )
                    c_comp_2.metric(
                        "Blocking Items",
                        str(len(pine_compatibility_report.get("blocking_items", []) or [])),
                    )
                    c_comp_3.metric(
                        "Compat Status",
                        str(pine_compatibility_report.get("status", "n/a")),
                    )
                    if pine_compatibility_report.get("is_blocking"):
                        st.error(
                            "Compatibilité Pine bloquante en mode strict: des éléments S0 empêchent l'exécution."
                        )
                    elif pine_compatibility_report.get("has_blocking_features"):
                        st.warning(
                            "Éléments incompatibles détectés, mais non bloquants dans le mode courant."
                        )
                    else:
                        st.success("Aucun blocage S0 détecté pour ce script.")

                    with st.expander("Rapport de compatibilité Pine (P0.3)", expanded=False):
                        st.json(pine_compatibility_report)
                        recos = pine_compatibility_report.get("recommendations", []) or []
                        if recos:
                            st.markdown("**Actions recommandées**")
                            for reco in recos:
                                st.caption(f"- {reco}")

                # P0.4: Build and validate strategy_spec.v1 after successful precheck.
                if status == "valid" and os.path.exists(str(pine_file_path or "")):
                    try:
                        pine_text, source_encoding = _read_text_file_with_fallback(str(pine_file_path))
                        strategy_spec = _build_strategy_spec_v1_from_pine_text(
                            pine_text=pine_text,
                            source_name=st.session_state.get("pine_source_name", ""),
                            strategy_id=st.session_state.get("strategy_id", ""),
                            precheck_report=pine_precheck_report,
                            compatibility_report=pine_compatibility_report,
                            parser_backend=st.session_state.get("pine_spec_parser_backend", "auto"),
                        )
                        strategy_spec_validation = _validate_strategy_spec_v1(strategy_spec)
                        st.session_state["pine_strategy_spec"] = _sanitize_for_json(strategy_spec)
                        st.session_state["pine_strategy_spec_validation"] = _sanitize_for_json(
                            strategy_spec_validation
                        )
                        st.session_state["pine_source_encoding"] = source_encoding
                        st.session_state["pine_source_text"] = pine_text
                        spec_id = ((strategy_spec.get("strategy") or {}).get("id"))
                        if isinstance(spec_id, str) and spec_id.strip():
                            st.session_state["pending_strategy_id"] = spec_id.strip()

                        if bool(strategy_spec_validation.get("valid", False)):
                            try:
                                codegen_report = _generate_strategy_module_from_spec(
                                    strategy_spec=strategy_spec,
                                    output_dir=_pine_generated_dir(),
                                    import_mapping=st.session_state.get("pine_import_mapping", {}),
                                    import_resolution=(pine_precheck_report.get("import_resolution") or []),
                                )
                                st.session_state["pine_codegen_report"] = _sanitize_for_json(codegen_report)
                                output_path = str((codegen_report or {}).get("output_path") or "").strip()
                                if output_path:
                                    st.session_state["pine_generated_module_path"] = output_path
                            except Exception as codegen_error:
                                st.session_state["pine_codegen_report"] = {
                                    "status": "error",
                                    "errors": [f"Codegen error: {codegen_error}"],
                                }
                                st.session_state.pop("pine_generated_module_path", None)
                        else:
                            st.session_state.pop("pine_codegen_report", None)
                            st.session_state.pop("pine_generated_module_path", None)
                    except Exception as e:
                        st.session_state["pine_strategy_spec"] = None
                        st.session_state["pine_strategy_spec_validation"] = {
                            "schema_version": "strategy_spec_validation.v1",
                            "valid": False,
                            "errors": [f"Spec build error: {e}"],
                            "warnings": [],
                        }
                        st.session_state.pop("pine_codegen_report", None)
                        st.session_state.pop("pine_generated_module_path", None)
                        st.session_state.pop("pine_execution_gate_report", None)
                        st.session_state.pop("pine_parity_report", None)
                        st.session_state.pop("pine_request_security_diagnostics", None)
                        st.session_state.pop("pine_mtf_parity_proof_report", None)
                        st.session_state.pop("pine_parity_reference_payload", None)
                        st.session_state.pop("pine_parity_reference_validation", None)
                        st.session_state.pop("pine_parity_reference_metrics", None)
                        st.session_state.pop("pine_parity_reference_text", None)
                elif status != "valid":
                    st.session_state.pop("pine_strategy_spec", None)
                    st.session_state.pop("pine_strategy_spec_validation", None)
                    st.session_state.pop("pine_codegen_report", None)
                    st.session_state.pop("pine_generated_module_path", None)
                    st.session_state.pop("pine_execution_gate_report", None)
                    st.session_state.pop("pine_parity_report", None)
                    st.session_state.pop("pine_request_security_diagnostics", None)
                    st.session_state.pop("pine_mtf_parity_proof_report", None)
                    st.session_state.pop("pine_parity_reference_payload", None)
                    st.session_state.pop("pine_parity_reference_validation", None)
                    st.session_state.pop("pine_parity_reference_metrics", None)
                    st.session_state.pop("pine_parity_reference_text", None)
                    st.session_state.pop("pending_strategy_id", None)

                spec_validation = st.session_state.get("pine_strategy_spec_validation")
                strategy_spec = st.session_state.get("pine_strategy_spec")
                if isinstance(spec_validation, dict):
                    is_valid_spec = bool(spec_validation.get("valid", False))
                    err_count = len(spec_validation.get("errors", []) or [])
                    warn_count = len(spec_validation.get("warnings", []) or [])
                    c_spec_1, c_spec_2, c_spec_3 = st.columns(3)
                    c_spec_1.metric("Spec Status", "valid" if is_valid_spec else "invalid")
                    c_spec_2.metric("Spec Errors", str(err_count))
                    c_spec_3.metric("Spec Warnings", str(warn_count))
                    if is_valid_spec:
                        st.success("`strategy_spec.v1` valide et prêt pour les prochains lots.")
                    else:
                        st.error("`strategy_spec.v1` invalide: corrige les erreurs avant la suite.")
                    transcription = (
                        strategy_spec.get("transcription")
                        if isinstance(strategy_spec, dict) and isinstance(strategy_spec.get("transcription"), dict)
                        else {}
                    )
                    if transcription:
                        c_par_1, c_par_2, c_par_3 = st.columns(3)
                        c_par_1.metric("Parser Requested", str(transcription.get("parser_backend_requested", "n/a")))
                        c_par_2.metric("Parser Used", str(transcription.get("parser_backend_used", "n/a")))
                        c_par_3.metric(
                            "Fallback",
                            "yes" if bool(transcription.get("fallback_to_regex", False)) else "no",
                        )
                        py_meta = transcription.get("pynescript") if isinstance(transcription, dict) else {}
                        if isinstance(py_meta, dict):
                            st.caption(
                                "pynescript: "
                                f"available={bool(py_meta.get('available', False))}, "
                                f"parse_ok={bool(py_meta.get('parse_ok', False))}, "
                                f"entrypoint={py_meta.get('entrypoint') or 'n/a'}"
                            )
                    with st.expander("strategy_spec.v1 (P0.4)", expanded=False):
                        if isinstance(strategy_spec, dict):
                            st.json(strategy_spec)
                        st.markdown("**Validation**")
                        st.json(spec_validation)

                codegen_report = st.session_state.get("pine_codegen_report")
                generated_module_path = st.session_state.get("pine_generated_module_path")
                if isinstance(codegen_report, dict):
                    with st.expander("Generated Strategy Module (P1.1)", expanded=False):
                        st.json(codegen_report)
                        if isinstance(generated_module_path, str) and generated_module_path.strip():
                            st.caption(f"Generated module path: `{generated_module_path}`")
                            if os.path.exists(generated_module_path):
                                try:
                                    with open(generated_module_path, "r", encoding="utf-8") as f:
                                        preview = "".join(f.readlines()[:160])
                                    st.code(preview, language="python")
                                except Exception as preview_error:
                                    st.caption(f"Preview unavailable: {preview_error}")

                beta_readiness_report = _build_pine_beta_readiness_report(
                    precheck_report=pine_precheck_report,
                    compatibility_report=pine_compatibility_report,
                    strategy_spec=st.session_state.get("pine_strategy_spec"),
                    strategy_spec_validation=st.session_state.get("pine_strategy_spec_validation"),
                    codegen_report=st.session_state.get("pine_codegen_report"),
                    generated_module_path=st.session_state.get("pine_generated_module_path"),
                )
                if isinstance(beta_readiness_report, dict) and beta_readiness_report:
                    st.session_state["pine_beta_readiness_report"] = beta_readiness_report
                    c_beta_1, c_beta_2, c_beta_3 = st.columns(3)
                    c_beta_1.metric(
                        "V3 Beta Ready",
                        "yes" if bool(beta_readiness_report.get("beta_ready", False)) else "no",
                    )
                    c_beta_2.metric(
                        "Readiness Score",
                        f"{float(beta_readiness_report.get('readiness_score', 0.0)):.1f}/100",
                    )
                    c_beta_3.metric(
                        "Runtime Blockers",
                        str(len(beta_readiness_report.get("runtime_blockers", []) or [])),
                    )
                    if bool(beta_readiness_report.get("beta_ready", False)):
                        st.success("V3 beta: script prêt pour exécution/replay dans le périmètre supporté.")
                    else:
                        st.warning("V3 beta: des conditions bloquantes restent à corriger.")
                    with st.expander("Pine V3 Beta Readiness", expanded=False):
                        st.json(beta_readiness_report)
                else:
                    st.session_state.pop("pine_beta_readiness_report", None)
                    st.session_state.pop("pine_execution_gate_report", None)

                if "pine_enforce_parity_gate" not in st.session_state:
                    st.session_state["pine_enforce_parity_gate"] = True
                st.checkbox(
                    "Verrou d'exécution: exiger `parity_pass=true` si une référence Pine est fournie",
                    key="pine_enforce_parity_gate",
                    help=(
                        "Si activé, un run Pine est bloqué tant que la référence de parité est invalide "
                        "ou que `pine_parity_report.parity_pass` n'est pas vrai."
                    ),
                )

                with st.expander("Validation de parité Pine/Python (P1.4)", expanded=False):
                    st.caption(
                        "Compare les métriques de référence (Pine) aux métriques runtime Python "
                        "pour quantifier la parité."
                    )
                    st.caption(
                        f"Format canonique requis: `{_PARITY_REFERENCE_SCHEMA_VERSION}` "
                        "(JSON structuré avec `reference_metrics`)."
                    )
                    st.code(
                        """{
  "schema_version": "pine_parity_reference.v1",
  "generated_at_utc": "2026-02-12T00:00:00+00:00",
  "source": {
    "provider": "tradingview",
    "strategy_id": "my_strategy_v1",
    "symbol": "BTCUSDT",
    "timeframe": "5s",
    "start_date": "2025-01-01",
    "end_date": "2025-01-30",
    "notes": "backtest Pine de référence"
  },
  "reference_metrics": {
    "trade_count": 120,
    "entry_count": 120,
    "exit_count": 120,
    "total_return_pct": 18.4,
    "max_drawdown_pct": -4.2
  }
}""",
                        language="json",
                    )

                    uploaded_parity_reference = st.file_uploader(
                        "Charger référence Pine (JSON)",
                        type=["json"],
                        key="pine_parity_reference_upload",
                        help=(
                            "Charge un JSON de référence Pine. "
                            "Le format legacy est toléré puis migré vers le format v1."
                        ),
                    )
                    if uploaded_parity_reference is not None:
                        try:
                            uploaded_ref_raw = uploaded_parity_reference.read().decode("utf-8")
                            parsed_payload, parsed_validation, parse_err = _parse_parity_reference_payload_from_text(
                                uploaded_ref_raw,
                                allow_legacy=True,
                            )
                            if parse_err:
                                if isinstance(parsed_validation, dict):
                                    st.session_state["pine_parity_reference_validation"] = parsed_validation
                                st.error(parse_err)
                            elif isinstance(parsed_payload, dict) and parsed_payload:
                                _apply_parity_reference_payload(
                                    parsed_payload,
                                    parsed_validation,
                                    update_text=True,
                                )
                                if bool((parsed_validation or {}).get("valid", False)):
                                    st.success("Référence Pine chargée et validée.")
                                else:
                                    st.warning("Référence Pine chargée mais invalide: corrige les erreurs.")
                                for warn in (parsed_validation or {}).get("warnings", []):
                                    st.caption(f"Avertissement: {warn}")
                            else:
                                st.warning("Le JSON chargé est vide ou non exploitable.")
                        except Exception as e:
                            st.warning(f"Impossible de lire le JSON de référence: {e}")

                    pending_ref_text = st.session_state.pop("pending_pine_parity_reference_text", None)
                    if isinstance(pending_ref_text, str):
                        st.session_state["pine_parity_reference_text"] = pending_ref_text
                    if (
                        "pine_parity_reference_text" not in st.session_state
                        and isinstance(st.session_state.get("pine_parity_reference_payload"), dict)
                        and st.session_state.get("pine_parity_reference_payload")
                    ):
                        st.session_state["pine_parity_reference_text"] = json.dumps(
                            st.session_state.get("pine_parity_reference_payload"),
                            indent=2,
                            ensure_ascii=False,
                        )
                    elif (
                        "pine_parity_reference_text" not in st.session_state
                        and isinstance(st.session_state.get("pine_parity_reference_metrics"), dict)
                        and st.session_state.get("pine_parity_reference_metrics")
                    ):
                        bootstrap_payload = _build_parity_reference_payload(
                            st.session_state.get("pine_parity_reference_metrics"),
                            source={
                                "provider": "session_metrics_bootstrap",
                                "strategy_id": str(st.session_state.get("strategy_id") or ""),
                            },
                        )
                        bootstrap_validation = _validate_parity_reference_payload(
                            bootstrap_payload,
                            allow_legacy=False,
                        )
                        _apply_parity_reference_payload(bootstrap_payload, bootstrap_validation, update_text=True)

                    parity_reference_text = st.text_area(
                        "Référence Pine (JSON)",
                        key="pine_parity_reference_text",
                        height=200,
                        help=(
                            "Colle ici le JSON de référence Pine au format "
                            f"`{_PARITY_REFERENCE_SCHEMA_VERSION}`."
                        ),
                    )
                    c_ref_btn_1, c_ref_btn_2 = st.columns(2)
                    with c_ref_btn_1:
                        if st.button("Appliquer la référence JSON", key="pine_apply_reference_btn", width="stretch"):
                            parsed_payload, parsed_validation, parse_err = _parse_parity_reference_payload_from_text(
                                parity_reference_text,
                                allow_legacy=True,
                            )
                            if parse_err:
                                if isinstance(parsed_validation, dict):
                                    st.session_state["pine_parity_reference_validation"] = parsed_validation
                                st.error(parse_err)
                            elif not parsed_payload:
                                st.warning("Référence Pine vide ou non reconnue.")
                            else:
                                _apply_parity_reference_payload(
                                    parsed_payload,
                                    parsed_validation,
                                    update_text=False,
                                )
                                st.session_state["pending_pine_parity_reference_text"] = json.dumps(
                                    parsed_payload, indent=2, ensure_ascii=False
                                )
                                if bool((parsed_validation or {}).get("valid", False)):
                                    st.success("Référence Pine validée et normalisée.")
                                else:
                                    st.error("Référence Pine invalide: corrige les erreurs ci-dessous.")
                    with c_ref_btn_2:
                        if st.button("Insérer un template v1", key="pine_insert_reference_template_btn", width="stretch"):
                            current_detail_template = _extract_current_events_and_trades_for_parity()
                            template_payload = _build_parity_reference_payload(
                                reference_metrics=_extract_current_metrics_for_parity(),
                                reference_events={
                                    "entries": current_detail_template.get("entries", []),
                                    "exits": current_detail_template.get("exits", []),
                                },
                                reference_trades=current_detail_template.get("trades", []),
                                source={
                                    "provider": "manual_template",
                                    "strategy_id": str(st.session_state.get("strategy_id") or ""),
                                    "symbol": "",
                                    "timeframe": str(st.session_state.get("timeframe") or ""),
                                    "start_date": str(st.session_state.get("start_date") or ""),
                                    "end_date": str(st.session_state.get("end_date") or ""),
                                    "notes": "Compléter avant validation.",
                                },
                            )
                            st.session_state["pending_pine_parity_reference_text"] = json.dumps(
                                template_payload,
                                indent=2,
                                ensure_ascii=False,
                            )
                            st.info("Template v1 prêt (inclut événements/trades si disponibles). Clique sur `Appliquer la référence JSON`.")

                    reference_validation = st.session_state.get("pine_parity_reference_validation")
                    if isinstance(reference_validation, dict) and reference_validation:
                        c_ref_v1, c_ref_v2, c_ref_v3 = st.columns(3)
                        c_ref_v1.metric(
                            "Référence v1 valide",
                            "yes" if bool(reference_validation.get("valid", False)) else "no",
                        )
                        c_ref_v2.metric("Erreurs", str(len(reference_validation.get("errors", []) or [])))
                        c_ref_v3.metric("Avertissements", str(len(reference_validation.get("warnings", []) or [])))
                        for err in reference_validation.get("errors", []) or []:
                            st.caption(f"Erreur: {err}")
                        for warn in reference_validation.get("warnings", []) or []:
                            st.caption(f"Avertissement: {warn}")
                        with st.expander("Validation référence Pine (JSON)", expanded=False):
                            st.json(reference_validation)

                    c_th_1, c_th_2 = st.columns(2)
                    with c_th_1:
                        st.number_input(
                            "Seuil rel. trades (%)",
                            min_value=0.0,
                            value=float(st.session_state.get("pine_parity_trade_count_rel_pct", _DEFAULT_PARITY_THRESHOLDS["trade_count_rel_pct"])),
                            step=0.1,
                            key="pine_parity_trade_count_rel_pct",
                            help="Écart relatif max autorisé sur le nombre de trades.",
                        )
                        st.number_input(
                            "Seuil rel. entries (%)",
                            min_value=0.0,
                            value=float(st.session_state.get("pine_parity_entry_count_rel_pct", _DEFAULT_PARITY_THRESHOLDS["entry_count_rel_pct"])),
                            step=0.1,
                            key="pine_parity_entry_count_rel_pct",
                        )
                        st.number_input(
                            "Seuil rel. exits (%)",
                            min_value=0.0,
                            value=float(st.session_state.get("pine_parity_exit_count_rel_pct", _DEFAULT_PARITY_THRESHOLDS["exit_count_rel_pct"])),
                            step=0.1,
                            key="pine_parity_exit_count_rel_pct",
                        )
                    with c_th_2:
                        st.number_input(
                            "Seuil abs. return (%)",
                            min_value=0.0,
                            value=float(st.session_state.get("pine_parity_total_return_abs_pct", _DEFAULT_PARITY_THRESHOLDS["total_return_abs_pct"])),
                            step=0.1,
                            key="pine_parity_total_return_abs_pct",
                            help="Écart absolu max autorisé sur le total return (%).",
                        )
                        st.number_input(
                            "Seuil abs. max drawdown (%)",
                            min_value=0.0,
                            value=float(st.session_state.get("pine_parity_max_drawdown_abs_pct", _DEFAULT_PARITY_THRESHOLDS["max_drawdown_abs_pct"])),
                            step=0.1,
                            key="pine_parity_max_drawdown_abs_pct",
                            help="Écart absolu max autorisé sur le max drawdown (%).",
                        )
                    with st.expander("Seuils détaillés (événements/trades) - Lot 3", expanded=False):
                        d_th_1, d_th_2 = st.columns(2)
                        with d_th_1:
                            st.number_input(
                                "Seuil rel. count entry-events (%)",
                                min_value=0.0,
                                value=float(st.session_state.get("pine_parity_entry_event_count_rel_pct", _DEFAULT_PARITY_DETAIL_THRESHOLDS["entry_event_count_rel_pct"])),
                                step=0.1,
                                key="pine_parity_entry_event_count_rel_pct",
                            )
                            st.number_input(
                                "Seuil rel. count exit-events (%)",
                                min_value=0.0,
                                value=float(st.session_state.get("pine_parity_exit_event_count_rel_pct", _DEFAULT_PARITY_DETAIL_THRESHOLDS["exit_event_count_rel_pct"])),
                                step=0.1,
                                key="pine_parity_exit_event_count_rel_pct",
                            )
                            st.number_input(
                                "Tolérance temps events (sec)",
                                min_value=0.0,
                                value=float(st.session_state.get("pine_parity_event_time_tolerance_sec", _DEFAULT_PARITY_DETAIL_THRESHOLDS["event_time_tolerance_sec"])),
                                step=0.5,
                                key="pine_parity_event_time_tolerance_sec",
                            )
                        with d_th_2:
                            st.number_input(
                                "Match min entry-events (ratio)",
                                min_value=0.0,
                                max_value=1.0,
                                value=float(st.session_state.get("pine_parity_entry_event_match_min_ratio", _DEFAULT_PARITY_DETAIL_THRESHOLDS["entry_event_match_min_ratio"])),
                                step=0.01,
                                key="pine_parity_entry_event_match_min_ratio",
                            )
                            st.number_input(
                                "Match min exit-events (ratio)",
                                min_value=0.0,
                                max_value=1.0,
                                value=float(st.session_state.get("pine_parity_exit_event_match_min_ratio", _DEFAULT_PARITY_DETAIL_THRESHOLDS["exit_event_match_min_ratio"])),
                                step=0.01,
                                key="pine_parity_exit_event_match_min_ratio",
                            )
                            st.number_input(
                                "Match min trades (ratio)",
                                min_value=0.0,
                                max_value=1.0,
                                value=float(st.session_state.get("pine_parity_trade_match_min_ratio", _DEFAULT_PARITY_DETAIL_THRESHOLDS["trade_match_min_ratio"])),
                                step=0.01,
                                key="pine_parity_trade_match_min_ratio",
                            )
                            st.number_input(
                                "Tolérance temps trades (sec)",
                                min_value=0.0,
                                value=float(st.session_state.get("pine_parity_trade_time_tolerance_sec", _DEFAULT_PARITY_DETAIL_THRESHOLDS["trade_time_tolerance_sec"])),
                                step=0.5,
                                key="pine_parity_trade_time_tolerance_sec",
                            )

                    current_metrics = _extract_current_metrics_for_parity()
                    current_detail = _extract_current_events_and_trades_for_parity()
                    reference_payload = st.session_state.get("pine_parity_reference_payload")
                    if not isinstance(reference_payload, dict):
                        reference_payload = {}
                    reference_metrics = reference_payload.get("reference_metrics")
                    reference_events = reference_payload.get("reference_events")
                    reference_trades = reference_payload.get("reference_trades")
                    if not isinstance(reference_metrics, dict):
                        reference_metrics = st.session_state.get("pine_parity_reference_metrics")
                        if not isinstance(reference_metrics, dict):
                            reference_metrics = {}
                    if not isinstance(reference_events, dict):
                        reference_events = {}
                    if not isinstance(reference_trades, list):
                        reference_trades = []

                    current_events = {
                        "entries": current_detail.get("entries", []),
                        "exits": current_detail.get("exits", []),
                    }
                    current_trades = current_detail.get("trades", [])
                    if not isinstance(current_trades, list):
                        current_trades = []
                    st.markdown("**Métriques runtime courantes (Python)**")
                    st.json(current_metrics or {})
                    st.markdown("**Référence Pine (métriques normalisées)**")
                    st.json(reference_metrics or {})
                    c_det_1, c_det_2, c_det_3 = st.columns(3)
                    c_det_1.metric("Entry events ref/current", f"{len(reference_events.get('entries', []) or [])}/{len(current_events.get('entries', []) or [])}")
                    c_det_2.metric("Exit events ref/current", f"{len(reference_events.get('exits', []) or [])}/{len(current_events.get('exits', []) or [])}")
                    c_det_3.metric("Trades ref/current", f"{len(reference_trades)}/{len(current_trades)}")

                    if st.button("Calculer le rapport de parité", key="pine_compute_parity_btn", width="stretch"):
                        if not reference_metrics:
                            st.warning("Référence Pine manquante: charge un JSON avant calcul.")
                        elif not bool((st.session_state.get("pine_parity_reference_validation") or {}).get("valid", False)):
                            st.warning("Référence Pine invalide: corrige d'abord les erreurs de validation.")
                        elif not current_metrics:
                            st.warning("Métriques runtime indisponibles: lance d'abord le final backtest.")
                        else:
                            parity_report = _build_parity_report(
                                reference_metrics=reference_metrics,
                                current_metrics=current_metrics,
                                thresholds=_get_pine_parity_thresholds_from_state(),
                                reference_events=reference_events,
                                current_events=current_events,
                                reference_trades=reference_trades,
                                current_trades=current_trades,
                                detail_thresholds=_get_pine_parity_detail_thresholds_from_state(),
                            )
                            if isinstance(reference_payload, dict) and reference_payload:
                                parity_report["reference_payload_schema_version"] = reference_payload.get("schema_version")
                                parity_report["reference_source"] = reference_payload.get("source")
                            st.session_state["pine_parity_report"] = _sanitize_for_json(parity_report)
                            if parity_report.get("parity_pass") is True:
                                st.success("Parité validée (parity_pass=true).")
                            elif parity_report.get("parity_pass") is False:
                                st.error("Parité non validée (parity_pass=false).")
                            else:
                                st.warning("Parité partielle: référence insuffisante pour statuer.")

                    pine_parity_report = st.session_state.get("pine_parity_report")
                    if isinstance(pine_parity_report, dict) and pine_parity_report:
                        c_pr_1, c_pr_2, c_pr_3 = st.columns(3)
                        c_pr_1.metric("Parity Status", str(pine_parity_report.get("status", "n/a")))
                        c_pr_2.metric(
                            "Parity Pass",
                            (
                                "yes"
                                if pine_parity_report.get("parity_pass") is True
                                else ("no" if pine_parity_report.get("parity_pass") is False else "n/a")
                            ),
                        )
                        c_pr_3.metric("Checks", str(len(pine_parity_report.get("checks", []) or [])))
                        c_pr_4, c_pr_5 = st.columns(2)
                        c_pr_4.metric(
                            "Detail Available",
                            "yes" if bool(pine_parity_report.get("detail_available", False)) else "no",
                        )
                        detail_pass_value = pine_parity_report.get("detail_pass")
                        c_pr_5.metric(
                            "Detail Pass",
                            "yes" if detail_pass_value is True else ("no" if detail_pass_value is False else "n/a"),
                        )
                        checks = pine_parity_report.get("checks") or []
                        if isinstance(checks, list) and checks:
                            checks_df = pd.DataFrame(checks)
                            st.dataframe(checks_df, width="stretch")
                        detail_checks = pine_parity_report.get("detail_checks") or []
                        if isinstance(detail_checks, list) and detail_checks:
                            st.markdown("**Detail checks (events/trades)**")
                            detail_checks_df = pd.DataFrame(detail_checks)
                            st.dataframe(detail_checks_df, width="stretch")
                        with st.expander("Rapport de parité (JSON)", expanded=False):
                            st.json(pine_parity_report)

                pine_spec_for_mtf = st.session_state.get("pine_strategy_spec")
                pine_spec_for_mtf = pine_spec_for_mtf if isinstance(pine_spec_for_mtf, dict) else {}
                pine_cap_for_mtf = (
                    pine_spec_for_mtf.get("capabilities")
                    if isinstance(pine_spec_for_mtf.get("capabilities"), dict)
                    else {}
                )
                if bool(pine_cap_for_mtf.get("uses_request_security", False)):
                    with st.expander("Preuve de parité MTF request.security (P1.2)", expanded=False):
                        st.caption(
                            "Diagnostic dédié aux séries `request.security` (timeframe, densité, stabilité) "
                            "et preuve MTF consolidée."
                        )
                        if st.button("Générer preuve MTF", key="pine_compute_mtf_proof_btn", width="stretch"):
                            df_for_mtf = st.session_state.get("final_backtest_df")
                            if not isinstance(df_for_mtf, pd.DataFrame) or df_for_mtf.empty:
                                df_for_mtf = st.session_state.get("df")
                            if not isinstance(df_for_mtf, pd.DataFrame) or df_for_mtf.empty:
                                st.warning(
                                    "Données indisponibles pour preuve MTF: lance d'abord un run/backtest."
                                )
                            else:
                                params_for_mtf = (
                                    st.session_state.get("final_params")
                                    if isinstance(st.session_state.get("final_params"), dict)
                                    else {}
                                )
                                mtf_diag = _build_request_security_diagnostics(
                                    df=df_for_mtf,
                                    params=params_for_mtf,
                                    strategy_spec=pine_spec_for_mtf,
                                    external_bindings={},
                                )
                                mtf_proof = _build_mtf_parity_proof_report(
                                    strategy_spec=pine_spec_for_mtf,
                                    request_security_diagnostics=mtf_diag,
                                    parity_report=st.session_state.get("pine_parity_report"),
                                )
                                st.session_state["pine_request_security_diagnostics"] = _sanitize_for_json(mtf_diag)
                                st.session_state["pine_mtf_parity_proof_report"] = _sanitize_for_json(mtf_proof)
                                if bool(mtf_proof.get("proof_pass", False)):
                                    st.success("Preuve MTF validée.")
                                else:
                                    st.warning("Preuve MTF partielle/échouée: consulte les blockers.")

                        mtf_diag_report = st.session_state.get("pine_request_security_diagnostics")
                        mtf_diag_report = mtf_diag_report if isinstance(mtf_diag_report, dict) else {}
                        mtf_proof_report = st.session_state.get("pine_mtf_parity_proof_report")
                        mtf_proof_report = mtf_proof_report if isinstance(mtf_proof_report, dict) else {}
                        c_mtf_1, c_mtf_2, c_mtf_3 = st.columns(3)
                        c_mtf_1.metric(
                            "MTF diagnostics",
                            str(mtf_diag_report.get("status", "n/a")),
                        )
                        c_mtf_2.metric(
                            "request.security rows",
                            str(int(mtf_diag_report.get("request_security_count", 0) or 0)),
                        )
                        c_mtf_3.metric(
                            "MTF proof",
                            (
                                "pass"
                                if bool(mtf_proof_report.get("proof_pass", False))
                                else str(mtf_proof_report.get("status", "n/a"))
                            ),
                        )
                        mtf_rows = mtf_diag_report.get("rows") or []
                        if isinstance(mtf_rows, list) and mtf_rows:
                            st.dataframe(pd.DataFrame(mtf_rows), width="stretch")
                        mtf_blockers = mtf_proof_report.get("blockers") or []
                        if isinstance(mtf_blockers, list) and mtf_blockers:
                            st.markdown("**Blockers MTF**")
                            st.dataframe(pd.DataFrame(mtf_blockers), width="stretch")
                        with st.expander("Rapport MTF parity proof (JSON)", expanded=False):
                            if mtf_proof_report:
                                st.json(mtf_proof_report)
                            else:
                                st.caption("Aucun rapport MTF généré.")
                else:
                    st.session_state.pop("pine_request_security_diagnostics", None)
                    st.session_state.pop("pine_mtf_parity_proof_report", None)

                execution_gate_report = _build_pine_execution_gate_report(
                    strategy_mode=st.session_state.get("strategy_mode", DEFAULT_STRATEGY_MODE),
                    beta_readiness_report=st.session_state.get("pine_beta_readiness_report"),
                    parity_reference_payload=st.session_state.get("pine_parity_reference_payload"),
                    parity_reference_validation=st.session_state.get("pine_parity_reference_validation"),
                    parity_report=st.session_state.get("pine_parity_report"),
                    enforce_parity_when_reference=bool(st.session_state.get("pine_enforce_parity_gate", True)),
                )
                st.session_state["pine_execution_gate_report"] = _sanitize_for_json(execution_gate_report)
                c_gate_1, c_gate_2, c_gate_3 = st.columns(3)
                c_gate_1.metric(
                    "Execution Gate",
                    "pass" if bool(execution_gate_report.get("can_run", False)) else "blocked",
                )
                c_gate_2.metric(
                    "Gate blockers",
                    str(len(execution_gate_report.get("blockers", []) or [])),
                )
                c_gate_3.metric(
                    "Parity enforced",
                    "yes" if bool(execution_gate_report.get("enforce_parity_when_reference", False)) else "no",
                )
                if bool(execution_gate_report.get("can_run", False)):
                    st.success("Gate exécution Pine V3: prêt à lancer le run.")
                else:
                    st.error("Gate exécution Pine V3: lancement bloqué tant que les blockers persistent.")
                for blocker in execution_gate_report.get("blockers", []) or []:
                    if not isinstance(blocker, dict):
                        continue
                    st.caption(
                        f"Blocker `{blocker.get('code', 'unknown')}`: "
                        f"{blocker.get('label', '')} | {blocker.get('detail', '')}"
                    )
                for warn in execution_gate_report.get("warnings", []) or []:
                    st.caption(f"Avertissement gate: {warn}")
                with st.expander("Pine V3 Execution Gate (Lot 4)", expanded=False):
                    st.json(execution_gate_report)
        else:
            pine_file_path = st.session_state.get("pine_file_path", "")
            pine_precheck_report = st.session_state.get("pine_precheck_report")

        n_windows = st.number_input(
            "Number of Windows",
            min_value=1,
            value=1,
            help="Nombre de fenêtres utilisées pour le processus WFO classique.",
            key='n_windows'
        )
        train_size = st.slider(
            "Train Size Ratio",
            0.1,
            0.9,
            0.5,
            0.05,
            help="Part de chaque fenêtre réservée à l'optimisation (IS) par rapport à la validation (OOS).",
            key='train_size'
        )
        anchored = st.checkbox(
            "Anchored WFO",
            value=False,
            help="Si activé, la zone d'entraînement s'agrandit au fil du temps; sinon elle glisse.",
            key='anchored'
        )
        
        regime_options = ['classic', 'prev_best_grid', 'nn_guided', 'adaptive_continuous']
        optimization_regime = st.selectbox(
            "WFO Mode",
            options=regime_options,
            index=0,
            key='optimization_regime',
            format_func=lambda v: (
                "Classic WFO"
                if v == "classic"
                else (
                    "Previous Best Grid WFO"
                    if v == "prev_best_grid"
                    else ("NN-Guided WFO" if v == "nn_guided" else "Adaptive Continuous")
                )
            ),
            help="Choisit la logique globale de construction de grille d'une fenêtre/cycle au suivant."
        )

        opt_methods = ['grid', 'bayesian', 'optuna']
        if optimization_regime == "adaptive_continuous":
            optimization_method = st.selectbox(
                "Optimization Method",
                options=opt_methods,
                index=opt_methods.index(st.session_state.get("optimization_method", "grid")) if st.session_state.get("optimization_method", "grid") in opt_methods else 0,
                key='optimization_method',
                disabled=True,
                help="Non utilisé en mode Adaptive Continuous (le moteur adaptatif applique sa propre logique)."
            )
            st.caption("En mode Adaptive Continuous, la méthode `grid/bayesian/optuna` n'est pas appliquée.")
        else:
            optimization_method = st.selectbox(
                "Optimization Method",
                options=opt_methods,
                index=opt_methods.index(st.session_state.get("optimization_method", "grid")) if st.session_state.get("optimization_method", "grid") in opt_methods else 0,
                key='optimization_method',
                help="`grid`: exhaustif, `bayesian/optuna`: recherche probabiliste plus efficace sur grands espaces."
            )

        if optimization_regime == "prev_best_grid":
            st.caption("Fenêtre 1: grille complète. Fenêtres suivantes: réutilisation des meilleures valeurs de la fenêtre précédente.")

        patience_levels = ['Low', 'Medium', 'High']
        patience_level = st.selectbox(
            "Patience Level (Bayesian/Optuna)",
            options=patience_levels,
            index=1,
            key='patience_level',
            help="Contrôle l'arrêt anticipé des méthodes probabilistes: Low plus rapide, High plus approfondi."
        )
        
        max_trials = st.number_input(
            "Max Trials (Bayesian/Optuna)",
            min_value=10,
            value=200,
            step=10,
            key='max_trials',
            help="Nombre maximum d'essais évalués par fenêtre pour les méthodes bayésiennes."
        )
        neighbor_count = st.number_input(
            "Stability Neighbor Count",
            min_value=1,
            value=5,
            step=1,
            key='neighbor_count',
            help="Lissage local utilisé pour sélectionner un meilleur paramètre plus robuste."
        )

        st.markdown("##### Robust Tests (Level 1)")
        robust_tests_enabled = st.checkbox(
            "Enable Robust Set (Top-N + vote multi-fenêtres)",
            value=bool(st.session_state.get("robust_tests_enabled", False)),
            key="robust_tests_enabled",
            help=(
                "Construit un ensemble robuste en prenant les Top-N trials de chaque fenêtre, "
                "puis en agrégeant les paramètres par vote/médiane pondérés."
            ),
        )
        robust_top_n_per_window = st.number_input(
            "Robust Top-N / fenêtre",
            min_value=3,
            max_value=500,
            value=int(st.session_state.get("robust_top_n_per_window", 20)),
            step=1,
            key="robust_top_n_per_window",
            disabled=not robust_tests_enabled,
            help=(
                "Nombre de meilleurs trials IS conservés par fenêtre pour construire le pool robuste. "
                "Plus N est élevé, plus la robustesse augmente, mais la sélection est moins agressive."
            ),
        )
        robust_min_windows = st.number_input(
            "Robust min fenêtres requises",
            min_value=1,
            max_value=100,
            value=int(st.session_state.get("robust_min_windows", 3)),
            step=1,
            key="robust_min_windows",
            disabled=not robust_tests_enabled,
            help=(
                "Nombre minimal de fenêtres avec trials exploitables pour valider le robust set. "
                "Si ce seuil n'est pas atteint, l'app revient automatiquement au mode classique."
            ),
        )
        robust_use_for_final_backtest = st.checkbox(
            "Use Robust Set for Final Backtest",
            value=bool(st.session_state.get("robust_use_for_final_backtest", False)),
            key="robust_use_for_final_backtest",
            disabled=not robust_tests_enabled,
            help=(
                "Si activé, le backtest final utilise les paramètres robustes. "
                "Sinon, il conserve le meilleur jeu de paramètres d'une fenêtre."
            ),
        )
        with st.expander("Guide utilisateur - Robust Tests", expanded=False):
            st.markdown(
                "Le mode `Robust Set` réduit la dépendance à un optimum local de fenêtre.\n"
                "- Étape 1: on prend les Top-N trials dans chaque fenêtre.\n"
                "- Étape 2: on agrège les paramètres via vote pondéré (catégoriels/bool) et médiane pondérée (numériques).\n"
                "- Étape 3: on obtient un jeu de paramètres plus stable inter-fenêtres.\n\n"
                "Conseils:\n"
                "- Commence avec `Top-N=20` et `min fenêtres=3`.\n"
                "- Active `Use Robust Set for Final Backtest` pour tester la robustesse OOS globale.\n"
                "- Si les fenêtres sont peu nombreuses, garde un fallback classique."
            )
        
        backends = ['thread', 'dask', 'ray', 'pathos']
        parallel_backend = st.selectbox(
            "Parallel Backend",
            options=backends,
            index=0,
            key='parallel_backend',
            help="Moteur de parallélisation de l'optimisation."
        )
        
        max_workers = st.number_input(
            "Max Workers",
            min_value=1,
            value=os.cpu_count() or 1,
            key='max_workers',
            help="Nombre max de workers CPU pour les tâches parallèles."
        )
        use_numba = st.checkbox(
            "Use Numba Acceleration",
            value=True,
            key='use_numba',
            help="Active les optimisations Numba lorsque disponibles."
        )

        if optimization_regime == "nn_guided":
            st.caption("Mode guidé RN: apprend des fenêtres précédentes et resserre l'espace de recherche.")
            nn_min_samples = st.number_input(
                "NN Min Cumulative Trials",
                min_value=50,
                value=500,
                step=50,
                key='nn_min_samples',
                help="Nombre minimal d'essais valides cumulés avant d'activer le guidage par réseau de neurones."
            )
            nn_candidate_pool_size = st.number_input(
                "NN Candidate Pool Size",
                min_value=500,
                value=3000,
                step=100,
                key='nn_candidate_pool_size',
                help="Nombre de candidats aléatoires scorés par le RN à chaque fenêtre."
            )
            nn_top_k = st.number_input(
                "NN Top-K Candidates",
                min_value=50,
                value=250,
                step=10,
                key='nn_top_k',
                help="Nombre de meilleurs candidats retenus pour construire la grille guidée."
            )
            nn_exploration_ratio = st.slider(
                "NN Exploration Ratio",
                min_value=0.0,
                max_value=0.5,
                value=0.15,
                step=0.01,
                key='nn_exploration_ratio',
                help="Part des valeurs de base conservées pour l'exploration à chaque fenêtre."
            )
            nn_hidden_size = st.number_input(
                "NN Hidden Size",
                min_value=8,
                value=32,
                step=4,
                key='nn_hidden_size'
            )
            nn_epochs = st.number_input(
                "NN Epochs/Window",
                min_value=10,
                value=60,
                step=5,
                key='nn_epochs'
            )
            nn_learning_rate = st.number_input(
                "NN Learning Rate",
                min_value=0.0001,
                value=0.01,
                step=0.0005,
                format="%.4f",
                key='nn_learning_rate'
            )
            nn_l2 = st.number_input(
                "NN L2 Regularization",
                min_value=0.0,
                value=0.0001,
                step=0.0001,
                format="%.4f",
                key='nn_l2'
            )
        elif optimization_regime == "adaptive_continuous":
            st.caption("Adaptive Continuous: pas de fenêtres WFO fixes, la grille évolue cycle après cycle via l'historique des trials.")
            profile_keys = list(ADAPTIVE_PROFILE_DEFS.keys())
            default_profile = st.session_state.get("adaptive_profile", "balanced")
            if default_profile not in profile_keys:
                default_profile = "balanced"
            if st.session_state.get("adaptive_profile") not in profile_keys:
                st.session_state["adaptive_profile"] = default_profile
            adaptive_profile = st.selectbox(
                "Adaptive Profile",
                options=profile_keys,
                index=profile_keys.index(default_profile),
                key='adaptive_profile',
                format_func=lambda k: ADAPTIVE_PROFILE_DEFS.get(k, {}).get("label", k),
                help="Profil préconfiguré pour vitesse/robustesse. `Custom` laisse les champs manuels."
            )
            selected_profile_def = ADAPTIVE_PROFILE_DEFS.get(adaptive_profile, ADAPTIVE_PROFILE_DEFS["custom"])
            selected_profile_params = selected_profile_def.get("params")

            notice = st.session_state.pop("adaptive_profile_notice", None)
            if notice:
                st.success(notice)

            if isinstance(selected_profile_params, dict):
                profile_applied = st.session_state.get("adaptive_profile_last_applied")
                apply_label = "Réappliquer ce profil" if profile_applied == adaptive_profile else "Appliquer ce profil"
                if profile_applied != adaptive_profile:
                    st.info("Ce profil n'est pas encore appliqué aux champs ci-dessous.")
                if st.button(
                    apply_label,
                    key="adaptive_profile_apply",
                    help="Injecte les valeurs du profil dans tous les champs adaptatifs."
                ):
                    for param_key, param_val in selected_profile_params.items():
                        st.session_state[param_key] = param_val
                    st.session_state["adaptive_profile_last_applied"] = adaptive_profile
                    st.session_state["adaptive_profile_notice"] = f"Profil adaptatif appliqué: {selected_profile_def.get('label', adaptive_profile)}"
                    st.rerun()
            else:
                st.session_state["adaptive_profile_last_applied"] = "custom"

            with st.expander("Détails du profil", expanded=False):
                st.markdown(f"**Objectif:** {selected_profile_def.get('summary', 'n/a')}")
                st.markdown(f"**Avantages:** {selected_profile_def.get('advantages', 'n/a')}")
                st.markdown(f"**Limites:** {selected_profile_def.get('drawbacks', 'n/a')}")
                st.markdown(f"**Spécificité:** {selected_profile_def.get('specificity', 'n/a')}")
                st.markdown(f"**Impact durée (qualitatif):** {selected_profile_def.get('duration_note', 'n/a')}")

            adaptive_train_bars = st.number_input(
                "Adaptive Train Bars",
                min_value=200,
                value=5000,
                step=100,
                key='adaptive_train_bars',
                help="Nombre de bougies historiques utilisées comme zone d'entraînement à chaque cycle."
            )
            adaptive_cycle_bars = st.number_input(
                "Adaptive Cycle Bars",
                min_value=50,
                value=1000,
                step=50,
                key='adaptive_cycle_bars',
                help="Nombre de bougies avancées et évaluées après chaque cycle adaptatif."
            )
            adaptive_trials_per_cycle = st.number_input(
                "Adaptive Trials per Cycle",
                min_value=10,
                value=150,
                step=10,
                key='adaptive_trials_per_cycle'
            )
            adaptive_candidate_pool_size = st.number_input(
                "Adaptive Candidate Pool",
                min_value=200,
                value=3000,
                step=100,
                key='adaptive_candidate_pool_size'
            )
            adaptive_keep_ratio = st.slider(
                "Adaptive Keep Ratio",
                min_value=0.10,
                max_value=1.00,
                value=0.40,
                step=0.05,
                key='adaptive_keep_ratio',
                help="Part des meilleures valeurs conservées par paramètre pour la grille active suivante."
            )
            adaptive_exploration_ratio = st.slider(
                "Adaptive Exploration Ratio",
                min_value=0.00,
                max_value=0.90,
                value=0.20,
                step=0.01,
                key='adaptive_exploration_ratio',
                help="Part d'exploration utilisée pour la sélection des valeurs et l'échantillonnage des essais."
            )
            adaptive_min_values_per_param = st.number_input(
                "Adaptive Min Values/Param",
                min_value=1,
                value=2,
                step=1,
                key='adaptive_min_values_per_param'
            )
            adaptive_decay = st.number_input(
                "Adaptive Memory Decay",
                min_value=0.50,
                max_value=1.00,
                value=0.98,
                step=0.01,
                format="%.2f",
                key='adaptive_decay',
                help="Facteur de décroissance de la mémoire historique à chaque cycle (1.00 = mémoire complète)."
            )
            adaptive_ucb_beta = st.number_input(
                "Adaptive UCB Beta",
                min_value=0.0,
                value=0.75,
                step=0.05,
                key='adaptive_ucb_beta',
                help="Bonus d'incertitude appliqué au classement des valeurs de paramètres."
            )
            adaptive_warmup_trials = st.number_input(
                "Adaptive Warmup Trials",
                min_value=50,
                value=300,
                step=50,
                key='adaptive_warmup_trials',
                help="Nombre d'essais cumulés avant de commencer à resserrer la grille."
            )
            adaptive_max_cycles = st.number_input(
                "Adaptive Max Cycles (0 = no cap)",
                min_value=0,
                value=0,
                step=1,
                key='adaptive_max_cycles'
            )
            adaptive_oos_weight = st.number_input(
                "Adaptive OOS Weight",
                min_value=0.0,
                value=2.0,
                step=0.1,
                key='adaptive_oos_weight',
                help="Poids appliqué au score OOS lors de la mise à jour des statistiques de valeurs."
            )

            estimate = _estimate_adaptive_load(
                start_date=start_date,
                end_date=end_date,
                timeframe_str=timeframe,
                train_bars=adaptive_train_bars,
                cycle_bars=adaptive_cycle_bars,
                trials_per_cycle=adaptive_trials_per_cycle,
                max_cycles=adaptive_max_cycles
            )
            if estimate:
                observed_sec_per_trial = _get_observed_seconds_per_trial()
                if observed_sec_per_trial is not None:
                    eta = estimate["total_trials"] * observed_sec_per_trial
                    eta_text = _humanize_seconds(eta)
                    eta_source = f"basée sur ton dernier run (~{observed_sec_per_trial:.3f}s/trial)"
                else:
                    eta_low = estimate["total_trials"] * 0.2
                    eta_high = estimate["total_trials"] * 1.0
                    eta_text = f"{_humanize_seconds(eta_low)} à {_humanize_seconds(eta_high)}"
                    eta_source = "fourchette générique (0.2s à 1.0s par trial)"
                st.info(
                    "Estimation charge/durée: "
                    f"~{estimate['bars']:,} bougies, ~{estimate['cycles']:,} cycles, "
                    f"~{estimate['total_trials']:,} trials, durée estimée {eta_text} ({eta_source})."
                )

    # --- Metrics ---
    with st.expander("5. Performance Metrics", expanded=False):
        metric_options = ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate', 'avg_gain_per_trade', 'avg_loss_per_trade', 'avg_pl_per_trade']
        
        m1_idx = 0 # Default sharpe
        metric1 = st.selectbox(
            "Primary Metric",
            options=metric_options,
            index=m1_idx,
            key='metric1_name',
            help="Métrique principale du score combiné d'optimisation."
        )
        weight1 = st.number_input(
            "Weight 1",
            value=1.0,
            key='weight_metric1',
            help="Poids de la métrique principale."
        )
        
        m2_idx = 1 # Default total_return
        metric2 = st.selectbox(
            "Secondary Metric",
            options=metric_options,
            index=m2_idx,
            key='metric2_name',
            help="Métrique secondaire ajoutée au score combiné."
        )
        weight2 = st.number_input(
            "Weight 2",
            value=0.0,
            key='weight_metric2',
            help="Poids de la métrique secondaire (0 = ignorée)."
        )

    # --- Execution Settings ---
    with st.expander("6. Execution Settings", expanded=False):
        sizing_options = {
            "percent_equity": "100% capital",
            "fixed_cash": "Fixed amount (10000)"
        }
        sizing_values = list(sizing_options.keys())
        default_idx = 0
        order_sizing_mode = st.selectbox(
            "Order sizing",
            options=sizing_values,
            index=default_idx,
            key="order_sizing_mode",
            format_func=lambda v: sizing_options.get(v, v),
            help="Choix du mode de taille d'ordre pendant le backtest."
        )

        order_fixed_cash = st.number_input(
            "Fixed amount per trade",
            min_value=0.0,
            value=10000.0,
            step=100.0,
            key="order_fixed_cash",
            disabled=(order_sizing_mode != "fixed_cash")
        )

        fees_pct = st.number_input(
            "Brokerage fees (%)",
            min_value=0.0,
            value=0.0,
            step=0.001,
            format="%.3f",
            key="fees_pct"
        )

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

class WFOControl:
    def __init__(self):
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def should_stop(self):
        return self._stop_requested

    def wait_if_paused(self, log=None):
        if self._stop_requested:
            raise OptimizationInterrupted()

def _capture_state_snapshot():
    keys = [
        'wfo_results', 'df', 'final_backtest_df', 'final_portfolio', 'final_params',
        'final_params_score', 'final_params_window', 'final_params_is_metrics',
        'final_params_oos_metrics', 'final_params_is_score', 'final_params_oos_score',
        'final_params_source', 'final_params_robust_summary'
    ]
    return {k: st.session_state[k] for k in keys if k in st.session_state}

def _restore_state_snapshot(snapshot):
    keys = [
        'wfo_results', 'df', 'final_backtest_df', 'final_portfolio', 'final_params',
        'final_params_score', 'final_params_window', 'final_params_is_metrics',
        'final_params_oos_metrics', 'final_params_is_score', 'final_params_oos_score',
        'final_params_source', 'final_params_robust_summary'
    ]
    for key in keys:
        if key in st.session_state:
            st.session_state.pop(key)
    for key, value in snapshot.items():
        st.session_state[key] = value

def get_current_config():
    """Collects all sidebar widgets into a configuration dictionary."""
    pine_report = st.session_state.get("pine_precheck_report")
    pine_report = pine_report if isinstance(pine_report, dict) else {}
    pine_compat_report = st.session_state.get("pine_compatibility_report")
    pine_compat_report = pine_compat_report if isinstance(pine_compat_report, dict) else {}
    pine_spec = st.session_state.get("pine_strategy_spec")
    pine_spec = pine_spec if isinstance(pine_spec, dict) else {}
    pine_spec_validation = st.session_state.get("pine_strategy_spec_validation")
    pine_spec_validation = pine_spec_validation if isinstance(pine_spec_validation, dict) else {}
    pine_beta = st.session_state.get("pine_beta_readiness_report")
    pine_beta = pine_beta if isinstance(pine_beta, dict) else {}
    pine_parity = st.session_state.get("pine_parity_report")
    pine_parity = pine_parity if isinstance(pine_parity, dict) else {}
    pine_mtf_diag = st.session_state.get("pine_request_security_diagnostics")
    pine_mtf_diag = pine_mtf_diag if isinstance(pine_mtf_diag, dict) else {}
    pine_mtf_proof = st.session_state.get("pine_mtf_parity_proof_report")
    pine_mtf_proof = pine_mtf_proof if isinstance(pine_mtf_proof, dict) else {}
    pine_execution_gate = st.session_state.get("pine_execution_gate_report")
    pine_execution_gate = pine_execution_gate if isinstance(pine_execution_gate, dict) else {}
    pine_parity_reference_payload = st.session_state.get("pine_parity_reference_payload")
    pine_parity_reference_payload = (
        pine_parity_reference_payload if isinstance(pine_parity_reference_payload, dict) else {}
    )
    pine_parity_reference_validation = st.session_state.get("pine_parity_reference_validation")
    pine_parity_reference_validation = (
        pine_parity_reference_validation if isinstance(pine_parity_reference_validation, dict) else {}
    )
    config = {
        'start_date': start_date,
        'end_date': end_date,
        'timeframe': timeframe,
        'strategy_mode': strategy_mode,
        'strategy_id': strategy_id,
        'pine_file_path': st.session_state.get("pine_file_path", ""),
        'pine_compat_mode': st.session_state.get("pine_compat_mode", "strict"),
        'pine_spec_parser_backend': st.session_state.get("pine_spec_parser_backend", "auto"),
        'pine_source_name': st.session_state.get("pine_source_name", ""),
        'pine_library_paths': list(st.session_state.get("pine_library_paths", []) or []),
        'pine_library_names': list(st.session_state.get("pine_library_names", []) or []),
        'pine_import_mapping': dict(st.session_state.get("pine_import_mapping", {}) or {}),
        'pine_generated_module_path': st.session_state.get("pine_generated_module_path", ""),
        'pine_libraries_count': int(len(list(st.session_state.get("pine_library_files", []) or []))),
        'pine_precheck_status': pine_report.get("status"),
        'pine_source_sha1': pine_report.get("source_sha1"),
        'pine_detected_version': pine_report.get("detected_version"),
        'pine_compatibility_status': pine_compat_report.get("status"),
        'pine_compatibility_score': pine_compat_report.get("compatibility_score"),
        'pine_compatibility_blocking': bool(pine_compat_report.get("is_blocking", False)),
        'strategy_spec_schema_version': pine_spec.get("schema_version"),
        'strategy_spec_valid': bool(pine_spec_validation.get("valid", False)),
        'strategy_spec_sha256': _sha256_json(pine_spec) if pine_spec else None,
        'strategy_spec_validation_errors': len(pine_spec_validation.get("errors", []) or []),
        'strategy_spec_parser_backend_used': (
            (pine_spec.get("transcription") or {}).get("parser_backend_used")
            if isinstance(pine_spec.get("transcription"), dict)
            else None
        ),
        'pine_beta_ready': bool(pine_beta.get("beta_ready", False)),
        'pine_beta_readiness_score': pine_beta.get("readiness_score"),
        'pine_enforce_parity_gate': bool(st.session_state.get("pine_enforce_parity_gate", True)),
        'pine_execution_gate_status': pine_execution_gate.get("status"),
        'pine_execution_gate_can_run': pine_execution_gate.get("can_run"),
        'pine_execution_gate_blockers_count': len(pine_execution_gate.get("blockers", []) or []),
        'pine_parity_status': pine_parity.get("status"),
        'pine_parity_pass': pine_parity.get("parity_pass"),
        'pine_request_security_diagnostics_status': pine_mtf_diag.get("status"),
        'pine_request_security_diagnostics_count': int(pine_mtf_diag.get("request_security_count", 0)),
        'pine_mtf_parity_proof_status': pine_mtf_proof.get("status"),
        'pine_mtf_parity_proof_pass': pine_mtf_proof.get("proof_pass"),
        'pine_parity_reference_schema_version': pine_parity_reference_payload.get("schema_version"),
        'pine_parity_reference_valid': bool(pine_parity_reference_validation.get("valid", False)),
        'pine_parity_reference_errors': len(pine_parity_reference_validation.get("errors", []) or []),
        'pine_parity_reference_warnings': len(pine_parity_reference_validation.get("warnings", []) or []),
        'pine_parity_trade_count_rel_pct': float(st.session_state.get("pine_parity_trade_count_rel_pct", _DEFAULT_PARITY_THRESHOLDS["trade_count_rel_pct"])),
        'pine_parity_entry_count_rel_pct': float(st.session_state.get("pine_parity_entry_count_rel_pct", _DEFAULT_PARITY_THRESHOLDS["entry_count_rel_pct"])),
        'pine_parity_exit_count_rel_pct': float(st.session_state.get("pine_parity_exit_count_rel_pct", _DEFAULT_PARITY_THRESHOLDS["exit_count_rel_pct"])),
        'pine_parity_total_return_abs_pct': float(st.session_state.get("pine_parity_total_return_abs_pct", _DEFAULT_PARITY_THRESHOLDS["total_return_abs_pct"])),
        'pine_parity_max_drawdown_abs_pct': float(st.session_state.get("pine_parity_max_drawdown_abs_pct", _DEFAULT_PARITY_THRESHOLDS["max_drawdown_abs_pct"])),
        'pine_parity_entry_event_count_rel_pct': float(st.session_state.get("pine_parity_entry_event_count_rel_pct", _DEFAULT_PARITY_DETAIL_THRESHOLDS["entry_event_count_rel_pct"])),
        'pine_parity_exit_event_count_rel_pct': float(st.session_state.get("pine_parity_exit_event_count_rel_pct", _DEFAULT_PARITY_DETAIL_THRESHOLDS["exit_event_count_rel_pct"])),
        'pine_parity_entry_event_match_min_ratio': float(st.session_state.get("pine_parity_entry_event_match_min_ratio", _DEFAULT_PARITY_DETAIL_THRESHOLDS["entry_event_match_min_ratio"])),
        'pine_parity_exit_event_match_min_ratio': float(st.session_state.get("pine_parity_exit_event_match_min_ratio", _DEFAULT_PARITY_DETAIL_THRESHOLDS["exit_event_match_min_ratio"])),
        'pine_parity_trade_match_min_ratio': float(st.session_state.get("pine_parity_trade_match_min_ratio", _DEFAULT_PARITY_DETAIL_THRESHOLDS["trade_match_min_ratio"])),
        'pine_parity_event_time_tolerance_sec': float(st.session_state.get("pine_parity_event_time_tolerance_sec", _DEFAULT_PARITY_DETAIL_THRESHOLDS["event_time_tolerance_sec"])),
        'pine_parity_trade_time_tolerance_sec': float(st.session_state.get("pine_parity_trade_time_tolerance_sec", _DEFAULT_PARITY_DETAIL_THRESHOLDS["trade_time_tolerance_sec"])),
        'from_file': (data_source == "Local File"),
        'file_path': file_path,
        'selected_params': selected_params,
        'metric1_name': metric1,
        'metric2_name': metric2,
        'weight_metric1': weight1,
        'weight_metric2': weight2,
        'exit_sar_enabled': exit_sar_enabled,
        'exit_macd_enabled': exit_macd_enabled,
        'exit_macd_type_a': exit_macd_type_a,
        'exit_macd_type_b': exit_macd_type_b,
        'order_sizing_mode': order_sizing_mode,
        'order_fixed_cash': order_fixed_cash,
        'fees_pct': fees_pct,
        'n_windows': n_windows,
        'train_size': train_size,
        'anchored': anchored,
        'optimization_method': optimization_method,
        'optimization_regime': optimization_regime,
        'patience_level': patience_level,
        'max_trials': max_trials,
        'neighbor_count': neighbor_count,
        'robust_tests_enabled': bool(st.session_state.get('robust_tests_enabled', False)),
        'robust_top_n_per_window': int(st.session_state.get('robust_top_n_per_window', 20)),
        'robust_min_windows': int(st.session_state.get('robust_min_windows', 3)),
        'robust_use_for_final_backtest': bool(st.session_state.get('robust_use_for_final_backtest', False)),
        'parallel_backend': parallel_backend,
        'max_workers': max_workers,
        'use_numba': use_numba,
        'nn_min_samples': int(st.session_state.get('nn_min_samples', 500)),
        'nn_candidate_pool_size': int(st.session_state.get('nn_candidate_pool_size', 3000)),
        'nn_top_k': int(st.session_state.get('nn_top_k', 250)),
        'nn_exploration_ratio': float(st.session_state.get('nn_exploration_ratio', 0.15)),
        'nn_hidden_size': int(st.session_state.get('nn_hidden_size', 32)),
        'nn_epochs': int(st.session_state.get('nn_epochs', 60)),
        'nn_learning_rate': float(st.session_state.get('nn_learning_rate', 0.01)),
        'nn_l2': float(st.session_state.get('nn_l2', 1e-4)),
        'adaptive_train_bars': int(st.session_state.get('adaptive_train_bars', 5000)),
        'adaptive_cycle_bars': int(st.session_state.get('adaptive_cycle_bars', 1000)),
        'adaptive_trials_per_cycle': int(st.session_state.get('adaptive_trials_per_cycle', 150)),
        'adaptive_candidate_pool_size': int(st.session_state.get('adaptive_candidate_pool_size', 3000)),
        'adaptive_profile': st.session_state.get('adaptive_profile', 'balanced'),
        'adaptive_keep_ratio': float(st.session_state.get('adaptive_keep_ratio', 0.40)),
        'adaptive_exploration_ratio': float(st.session_state.get('adaptive_exploration_ratio', 0.20)),
        'adaptive_min_values_per_param': int(st.session_state.get('adaptive_min_values_per_param', 2)),
        'adaptive_decay': float(st.session_state.get('adaptive_decay', 0.98)),
        'adaptive_ucb_beta': float(st.session_state.get('adaptive_ucb_beta', 0.75)),
        'adaptive_warmup_trials': int(st.session_state.get('adaptive_warmup_trials', 300)),
        'adaptive_max_cycles': int(st.session_state.get('adaptive_max_cycles', 0)),
        'adaptive_oos_weight': float(st.session_state.get('adaptive_oos_weight', 2.0))
    }
    # Merge parameter ranges
    config.update(config_params)
    return config

def calculate_combinations(config):
    """Calculates the total number of parameter combinations."""
    total = 1
    if not config.get('selected_params'):
        return 0
        
    for param in config['selected_params']:
        p_min = config.get(f'{param}_min')
        p_max = config.get(f'{param}_max')
        p_step = config.get(f'{param}_step')
        
        if p_step <= 0:
            continue
            
        # Robust calculation for float steps
        # Adding a small epsilon to handle floating point errors
        count = int(np.floor((p_max - p_min + 1e-10) / p_step)) + 1
        total *= max(1, count)
        
    return total


def _normalize_filename_token(value, default="na", max_len=24):
    """Normalize arbitrary text to a filesystem-safe short token."""
    text = str(value or "").strip().lower()
    if not text:
        return default
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        return default
    return text[:max_len]


def _build_period_label(start_date, end_date):
    """Build compact duration label (ex: 01m, 02w, 10d) from config dates."""
    start_dt = _parse_iso_date(start_date) if isinstance(start_date, str) else None
    end_dt = _parse_iso_date(end_date) if isinstance(end_date, str) else None
    if start_dt is None or end_dt is None:
        return "unk"
    if end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt
    days = max(1, (end_dt.date() - start_dt.date()).days + 1)
    if days >= 28:
        months = max(1, int(round(days / 30.0)))
        return f"{months:02d}m"
    if days >= 7:
        weeks = max(1, int(round(days / 7.0)))
        return f"{weeks:02d}w"
    return f"{days:02d}d"


def _build_config_filename(config):
    """
    Build an abbreviated, information-rich WFO config filename.
    Example:
    config_wfo_20260210_132530_01m_5s_bayes_05w_tr5000_classic.json
    """
    now_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    period = _build_period_label(config.get("start_date"), config.get("end_date"))
    timeframe = _normalize_filename_token(str(config.get("timeframe", "tf")).lower(), default="tf", max_len=8)

    method_map = {
        "bayesian": "bayes",
        "optuna": "optuna",
        "grid": "grid",
    }
    regime_map = {
        "classic": "classic",
        "adaptive_continuous": "adapt",
        "nn_guided": "nnguide",
        "prev_best_grid": "prevbest",
    }
    method_raw = str(config.get("optimization_method", "grid")).lower()
    regime_raw = str(config.get("optimization_regime", "classic")).lower()
    method = method_map.get(method_raw, _normalize_filename_token(method_raw, default="method", max_len=12))
    regime = regime_map.get(regime_raw, _normalize_filename_token(regime_raw, default="regime", max_len=14))

    try:
        windows = int(config.get("n_windows", 0))
    except Exception:
        windows = 0
    windows_label = f"{max(0, windows):02d}w"

    # Trials label: use max_trials for classic/nn/prevbest regimes,
    # and adaptive_trials_per_cycle for adaptive continuous mode.
    if regime_raw == "adaptive_continuous":
        trials_value = config.get("adaptive_trials_per_cycle")
    else:
        trials_value = config.get("max_trials")
    try:
        trials_label = f"tr{max(0, int(trials_value))}"
    except Exception:
        trials_label = "trna"

    return f"config_wfo_{now_tag}_{period}_{timeframe}_{method}_{windows_label}_{trials_label}_{regime}.json"

def run_final_backtest_logic():
    """Runs the final backtest using averaged parameters."""
    if 'wfo_results' not in st.session_state or 'df' not in st.session_state:
        st.error("No WFO results available to run final backtest.")
        return

    results = st.session_state['wfo_results']
    config = get_current_config()
    final_start_date = st.session_state.get('final_start_date', config.get('start_date'))
    final_end_date = st.session_state.get('final_end_date', config.get('end_date'))
    final_file_path = st.session_state.get('final_file_path', config.get('file_path'))

    with st.spinner("Loading data for final backtest..."):
        if config.get('from_file'):
            df = load_data(
                final_start_date,
                final_end_date,
                config.get('timeframe', DEFAULT_TIMEFRAME),
                from_file=True,
                file_path=final_file_path
            )
        else:
            df = load_data(
                final_start_date,
                final_end_date,
                config.get('timeframe', DEFAULT_TIMEFRAME),
                from_file=False
            )

    if df is None or df.empty:
        st.error("No data loaded for the final backtest range.")
        return
    st.session_state['final_backtest_df'] = df

    # Use either classic best-window params or robust-set params (if enabled).
    (
        chosen_params,
        best_score,
        best_window,
        best_is_metrics,
        best_oos_metrics,
        final_source,
        robust_summary,
    ) = _select_final_params_from_results(results, config)
    if not chosen_params:
        st.error("No valid parameters found for final backtest.")
        return
    if (
        bool(config.get("robust_tests_enabled", False))
        and bool(config.get("robust_use_for_final_backtest", False))
        and str(final_source or "").lower() != "robust_set"
    ):
        st.warning(
            "Robust Set demandé mais non applicable sur ce run. "
            "Fallback automatique vers la sélection classique best_window."
        )

    # Define integer parameters that should be rounded
    int_params = {
        'timeperiod', 'fenetre_lowest', 'longueur_mediane', 'Nb_bars_above', 'user_exit_sma_length',
        'macd_fast_length', 'macd_slow_length', 'macd_signal_length'
    }
    for param in list(chosen_params.keys()):
        if param in int_params:
            try:
                chosen_params[param] = int(round(float(chosen_params[param])))
            except Exception:
                pass
        elif isinstance(chosen_params[param], float):
            chosen_params[param] = round(chosen_params[param], 2)

    st.session_state['final_params'] = chosen_params
    st.session_state['final_params_score'] = best_score
    st.session_state['final_params_window'] = best_window
    st.session_state['final_params_is_metrics'] = best_is_metrics
    st.session_state['final_params_oos_metrics'] = best_oos_metrics
    st.session_state['final_params_source'] = final_source
    st.session_state['final_params_robust_summary'] = robust_summary
    def _combined_score_local(row):
        metric1_name = config.get('metric1_name', 'sharpe_ratio')
        metric2_name = config.get('metric2_name', 'total_return')
        weight_metric1 = float(config.get('weight_metric1', 1.0))
        weight_metric2 = float(config.get('weight_metric2', 0.0))

        def get_metric_value(row, name):
            if not row:
                return None
            if name == 'max_drawdown':
                value = row.get('max_drawdown')
                return None if value is None else -value
            if name == 'sharpe_ratio':
                return row.get('sharpe')
            if name == 'total_return':
                return row.get('return')
            if name == 'win_rate':
                return row.get('win_rate')
            if name == 'avg_gain_per_trade':
                return row.get('avg_gain_per_trade')
            if name == 'avg_loss_per_trade':
                value = row.get('avg_loss_per_trade')
                return None if value is None else -value
            if name == 'avg_pl_per_trade':
                return row.get('avg_pl_per_trade')
            return None

        total_weight = weight_metric1 + weight_metric2
        if total_weight == 0:
            return None
        m1 = get_metric_value(row, metric1_name)
        m2 = get_metric_value(row, metric2_name)
        if weight_metric1 != 0 and m1 is None:
            return None
        if weight_metric2 != 0 and m2 is None:
            return None
        if m1 is None:
            m1 = 0.0
        if m2 is None:
            m2 = 0.0
        return (weight_metric1 * m1 + weight_metric2 * m2) / total_weight

    st.session_state['final_params_is_score'] = _combined_score_local(best_is_metrics)
    st.session_state['final_params_oos_score'] = _combined_score_local(best_oos_metrics)

    # Inject execution settings into params for the final backtest
    chosen_params['order_sizing_mode'] = config.get('order_sizing_mode', 'percent_equity')
    chosen_params['order_fixed_cash'] = float(config.get('order_fixed_cash', 10000.0))
    chosen_params['fees_pct'] = float(config.get('fees_pct', 0.0))
    
    with st.spinner("Running Final Backtest on Full Dataset..."):
        try:
            strategy_adapter = resolve_strategy_adapter(
                strategy_mode=config.get("strategy_mode"),
                strategy_id=config.get("strategy_id"),
                config=config,
            )
            final_portfolio = strategy_adapter.run_backtest(
                df,
                chosen_params,
                config.get('timeframe', DEFAULT_TIMEFRAME),
                return_portfolio=True
            )
            st.session_state['final_portfolio'] = final_portfolio
            st.success("Final Backtest Complete!")
            if str(final_source or "").lower() == "robust_set":
                st.info("Final backtest executed with robust-set parameters (Top-N vote multi-fenêtres).")
        except Exception as e:
            st.error(f"Error in final backtest: {e}")


# ==============================================================================
# MAIN LOGIC
# ==============================================================================

def run_wfo(config, control=None, job_state=None):
    """Run WFO via the run service without direct UI calls."""
    return run_optimization_job(config=config, control=control, job_state=job_state)

# --- Action Buttons ---
st.sidebar.divider()

# Live Combination Count
current_conf = get_current_config()
total_combos = calculate_combinations(current_conf)
if str(current_conf.get("optimization_regime", "")).lower() == "adaptive_continuous":
    adaptive_estimate = _estimate_adaptive_load(
        start_date=current_conf.get("start_date"),
        end_date=current_conf.get("end_date"),
        timeframe_str=current_conf.get("timeframe"),
        train_bars=current_conf.get("adaptive_train_bars", 5000),
        cycle_bars=current_conf.get("adaptive_cycle_bars", 1000),
        trials_per_cycle=current_conf.get("adaptive_trials_per_cycle", 150),
        max_cycles=current_conf.get("adaptive_max_cycles", 0)
    )
    if adaptive_estimate:
        sec_per_trial = _get_observed_seconds_per_trial()
        if sec_per_trial is not None:
            eta_text = _humanize_seconds(adaptive_estimate["total_trials"] * sec_per_trial)
            eta_hint = f"ETA ~ {eta_text}"
        else:
            eta_hint = "ETA selon machine/données"
        st.sidebar.info(
            "📊 Charge adaptative estimée: "
            f"**{adaptive_estimate['cycles']:,} cycles** | "
            f"**{adaptive_estimate['total_trials']:,} trials** ({eta_hint})"
        )
    else:
        st.sidebar.info("📊 Charge adaptative: renseigne des dates/timeframe valides pour estimer cycles et trials.")
else:
    st.sidebar.info(f"📊 Total Parameter Combinations: **{total_combos:,}**")

if 'wfo_running' not in st.session_state:
    st.session_state['wfo_running'] = False

# Resolve finished background job and update/restore state once.
if st.session_state.get('wfo_running'):
    wfo_thread = st.session_state.get('wfo_thread')
    wfo_job_state = st.session_state.get('wfo_job_state')
    if wfo_thread is not None and not wfo_thread.is_alive() and wfo_job_state is not None:
        status = wfo_job_state.get('status')
        job_conf = st.session_state.get('wfo_job_config', {})
        run_metadata = {
            "run_id": wfo_job_state.get("run_id"),
            "status": status,
            "started_at_utc": wfo_job_state.get("started_at_utc"),
            "ended_at_utc": wfo_job_state.get("ended_at_utc"),
            "elapsed_seconds": wfo_job_state.get("elapsed"),
            "config_sha256": wfo_job_state.get("config_sha256"),
            "results_sha256": wfo_job_state.get("results_sha256"),
            "strategy_mode": job_conf.get("strategy_mode"),
            "strategy_id": job_conf.get("strategy_id"),
        }
        st.session_state["wfo_run_metadata"] = _sanitize_for_json(run_metadata)
        if status == 'completed' and wfo_job_state.get('results') is not None:
            st.session_state['wfo_results'] = wfo_job_state['results']
            st.session_state['df'] = wfo_job_state['df']
            st.session_state['opt_start_date'] = job_conf.get('start_date')
            st.session_state['opt_end_date'] = job_conf.get('end_date')
            st.session_state['wfo_notice'] = ("success", "Optimization finished.")
        elif status == 'stopped':
            _restore_state_snapshot(st.session_state.get('wfo_prev_state', {}))
            st.session_state['wfo_notice'] = ("warning", "Optimization stopped. Previous state restored.")
        else:
            _restore_state_snapshot(st.session_state.get('wfo_prev_state', {}))
            err = wfo_job_state.get('error') or "Unknown optimization error."
            st.session_state['wfo_notice'] = ("error", f"An error occurred during optimization: {err}")

        st.session_state["wfo_traceability"] = _build_traceability_payload(
            config_snapshot=job_conf,
            results_snapshot=wfo_job_state.get('results'),
            run_metadata=st.session_state.get("wfo_run_metadata")
        )

        for key in ['wfo_thread', 'wfo_control', 'wfo_job_state', 'wfo_prev_state', 'wfo_job_config']:
            st.session_state.pop(key, None)
        st.session_state['wfo_running'] = False
        st.rerun()

col_run, col_save = st.sidebar.columns([1, 1])

with col_run:
    strategy_mode_current = str(current_conf.get("strategy_mode", DEFAULT_STRATEGY_MODE)).lower()
    pine_gate_for_launch = _build_pine_execution_gate_report(
        strategy_mode=current_conf.get("strategy_mode", DEFAULT_STRATEGY_MODE),
        beta_readiness_report=st.session_state.get("pine_beta_readiness_report"),
        parity_reference_payload=st.session_state.get("pine_parity_reference_payload"),
        parity_reference_validation=st.session_state.get("pine_parity_reference_validation"),
        parity_report=st.session_state.get("pine_parity_report"),
        enforce_parity_when_reference=bool(st.session_state.get("pine_enforce_parity_gate", True)),
    )
    st.session_state["pine_execution_gate_report"] = _sanitize_for_json(pine_gate_for_launch)
    gate_blocks_launch = (
        strategy_mode_current == "pine_imported"
        and not bool(pine_gate_for_launch.get("can_run", False))
    )
    if gate_blocks_launch:
        blocker_codes = [
            str(item.get("code", "unknown"))
            for item in (pine_gate_for_launch.get("blockers") or [])
            if isinstance(item, dict)
        ]
        st.sidebar.warning(
            "Pine gate actif: lancement bloqué"
            + (f" ({', '.join(blocker_codes[:3])})" if blocker_codes else ".")
        )
    if not st.session_state.get('wfo_running'):
        if st.button(
            "🚀 Start WFO",
            type="primary",
            key="start_wfo_btn",
            width="stretch",
            help="Lance l'optimisation selon le mode choisi (WFO classique, grille précédente, NN, ou adaptatif continu).",
            disabled=gate_blocks_launch,
        ):
            adapter_ready = True
            if gate_blocks_launch:
                for blocker in pine_gate_for_launch.get("blockers", []) or []:
                    if isinstance(blocker, dict):
                        st.error(
                            f"[{blocker.get('code', 'unknown')}] "
                            f"{blocker.get('label', '')}: {blocker.get('detail', '')}"
                        )
                adapter_ready = False
            if strategy_mode_current != "native_atdmf":
                compat_mode = str(current_conf.get("pine_compat_mode", "strict")).lower()
                compat_report = st.session_state.get("pine_compatibility_report")
                if compat_mode == "strict" and isinstance(compat_report, dict) and compat_report.get("is_blocking"):
                    blocking_items = compat_report.get("blocking_items", []) or []
                    blocking_labels = ", ".join(
                        str(item.get("label", "unknown"))
                        for item in blocking_items[:4]
                        if isinstance(item, dict)
                    )
                    suffix = f" ({blocking_labels})" if blocking_labels else ""
                    st.error(
                        "Mode strict: script Pine incompatible, exécution bloquée" + suffix + "."
                    )
                    adapter_ready = False
                if adapter_ready:
                    try:
                        resolve_strategy_adapter(
                            strategy_mode=current_conf.get("strategy_mode"),
                            strategy_id=current_conf.get("strategy_id"),
                            config=current_conf,
                        )
                    except NotImplementedError as e:
                        st.error(f"Mode stratégie non exécutable: {e}")
                        adapter_ready = False

            if not adapter_ready:
                pass
            elif not selected_params:
                st.error("Select params!")
            else:
                run_started_at = _utc_now_iso()
                run_id = f"wfo-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
                config_sha = _sha256_json(current_conf)
                job_state = {
                    'status': 'running',
                    'progress': 0.0,
                    'message': "Preparing optimization...",
                    'results': None,
                    'df': None,
                    'error': None,
                    'elapsed': None,
                    'window': None,
                    'evaluations': None,
                    'speed': None,
                    'eta_seconds': None,
                    'run_id': run_id,
                    'started_at_utc': run_started_at,
                    'ended_at_utc': None,
                    'config_sha256': config_sha,
                    'results_sha256': None
                }
                control = WFOControl()
                st.session_state['wfo_prev_state'] = _capture_state_snapshot()
                st.session_state['wfo_job_state'] = job_state
                st.session_state['wfo_control'] = control
                st.session_state['wfo_job_config'] = current_conf.copy()
                st.session_state['wfo_running'] = True
                st.session_state['wfo_traceability'] = _build_traceability_payload(
                    config_snapshot=current_conf,
                    results_snapshot=None,
                    run_metadata={
                        "run_id": run_id,
                        "status": "running",
                        "started_at_utc": run_started_at,
                        "config_sha256": config_sha,
                        "strategy_mode": current_conf.get("strategy_mode"),
                        "strategy_id": current_conf.get("strategy_id"),
                    }
                )

                def _wfo_worker():
                    results, df, elapsed = run_wfo(current_conf, control=control, job_state=job_state)
                    if control.should_stop():
                        job_state['status'] = 'stopped'
                        job_state['ended_at_utc'] = _utc_now_iso()
                    elif results is not None and df is not None:
                        job_state['status'] = 'completed'
                        results["robust_set_summary"] = _build_robust_set_summary(results, current_conf)
                        run_meta_completed = {
                            "run_id": job_state.get("run_id"),
                            "status": "completed",
                            "started_at_utc": job_state.get("started_at_utc"),
                            "ended_at_utc": _utc_now_iso(),
                            "elapsed_seconds": elapsed,
                            "config_sha256": job_state.get("config_sha256"),
                            "strategy_mode": current_conf.get("strategy_mode"),
                            "strategy_id": current_conf.get("strategy_id"),
                        }
                        # Bind audit metadata to the produced results so the trace follows the data.
                        results['traceability'] = _build_traceability_payload(
                            config_snapshot=current_conf,
                            results_snapshot=results,
                            run_metadata=run_meta_completed
                        )
                        job_state['results'] = results
                        job_state['df'] = df
                        job_state['elapsed'] = elapsed
                        job_state['ended_at_utc'] = run_meta_completed["ended_at_utc"]
                        job_state['results_sha256'] = _sha256_json(results)
                    else:
                        if job_state.get('status') != 'stopped':
                            job_state['status'] = 'error'
                            if not job_state.get('error'):
                                job_state['error'] = "No data found for the specified range/source."
                            job_state['ended_at_utc'] = _utc_now_iso()

                worker = threading.Thread(target=_wfo_worker, daemon=True)
                st.session_state['wfo_thread'] = worker
                worker.start()
                st.rerun()
    else:
        if st.button("🛑 Stop WFO", key="stop_wfo_btn", width="stretch", help="Demande un arrêt propre après l'essai en cours."):
            control = st.session_state.get('wfo_control')
            if control:
                control.request_stop()
            if st.session_state.get('wfo_job_state') is not None:
                st.session_state['wfo_job_state']['message'] = "Stop requested. Waiting for clean shutdown..."
            st.sidebar.warning("Stop requested. Optimization is shutting down...")
            st.rerun()

if st.session_state.get('wfo_running'):
    job_state = st.session_state.get('wfo_job_state', {})
    _inject_running_animation_css()
    st.sidebar.markdown(
        '<div class="run-badge"><span class="run-badge-dot"></span>RUNNING</div>',
        unsafe_allow_html=True
    )
    st.sidebar.progress(float(job_state.get('progress', 0.0)))
    st.sidebar.caption(job_state.get('message', "Running..."))
    _render_running_status_card(job_state, current_conf)

with col_save:
    # Save Config Button
    json_config = json.dumps(current_conf, indent=4)
    config_filename = _build_config_filename(current_conf)
    st.download_button(
        label="💾 Save Config",
        data=json_config,
        file_name=config_filename,
        mime="application/json",
        width="stretch",
        help=(
            "Télécharge la configuration actuelle de tous les contrôles de la sidebar. "
            "Règle de nommage: "
            "`config_wfo_<YYYYMMDD_HHMMSS>_<periode>_<timeframe>_<methode>_<nb_fenetres>_<trials>_<regime>.json` "
            "(ex: `config_wfo_20260210_141530_01m_5s_bayes_05w_tr5000_classic.json`)."
        )
    )

if st.session_state.get('wfo_notice'):
    notice_type, notice_msg = st.session_state.pop('wfo_notice')
    if notice_type == "success":
        st.success(notice_msg)
    elif notice_type == "warning":
        st.warning(notice_msg)
    else:
        st.error(notice_msg)

st.sidebar.divider()
st.sidebar.subheader("📤 Export Results")
if "wfo_results" in st.session_state:
    full_export_bundle = st.sidebar.checkbox(
        "Package complet (rejeu + stats)",
        value=True,
        help="Inclut tous les trials, les fichiers d'analyse et un manifeste de rejeu."
    )
    data_snapshot_mode = st.sidebar.selectbox(
        "Snapshot des prix dans le ZIP",
        options=["manifest_only", "parquet_zstd", "csv_downsampled", "csv_full"],
        index=0,
        format_func=lambda v: (
            "Manifest only (léger, sans snapshot)"
            if v == "manifest_only"
            else (
                "Parquet zstd (recommandé)"
                if v == "parquet_zstd"
                else ("CSV downsampled" if v == "csv_downsampled" else "CSV full")
            )
        ),
        help=(
            "Choix du format de données marché exportées: "
            "`manifest_only` pour archive légère, `parquet_zstd` pour rejeu compact."
        )
    )
    if full_export_bundle and data_snapshot_mode == "manifest_only":
        st.sidebar.warning("Package complet sans snapshot prix: rejeu strict non garanti.")
    df_max_rows = 200000
    if data_snapshot_mode == "csv_downsampled":
        df_max_rows = st.sidebar.slider(
            "Max rows for df.csv",
            min_value=10000,
            max_value=1000000,
            value=200000,
            step=10000,
            help="Nombre maximal approximatif de lignes conservées dans `df.csv`."
        )
    if st.sidebar.button(
        "💾 Save Results to Disk",
        width="stretch",
        help="Crée une archive ZIP des résultats dans le dossier `reports/`."
    ):
        zip_buffer = _export_results_zip(
            data_snapshot_mode=data_snapshot_mode,
            df_max_rows=df_max_rows,
            full_package=full_export_bundle
        )
        if zip_buffer is not None:
            saved_path = _save_results_zip_to_disk(zip_buffer, current_conf)
            st.session_state["results_zip_bytes"] = zip_buffer.getvalue()
            st.session_state["results_zip_path"] = saved_path
            st.sidebar.success(f"Saved: {saved_path}")

    if st.sidebar.button(
        "📄 Générer rapport PDF",
        width="stretch",
        help="Génère un rapport PDF (résultats WFO + final backtest) dans le dossier `reports/`."
    ):
        try:
            pdf_buffer = _generate_wfo_pdf_report(st.session_state.get("wfo_results"), current_conf)
            if pdf_buffer is None:
                st.sidebar.error("Impossible de générer le PDF: résultats WFO indisponibles.")
            else:
                saved_pdf_path = _save_results_pdf_to_disk(pdf_buffer, current_conf)
                st.session_state["results_pdf_bytes"] = pdf_buffer.getvalue()
                st.session_state["results_pdf_path"] = saved_pdf_path
                st.sidebar.success(f"PDF saved: {saved_pdf_path}")
        except Exception as pdf_exc:
            st.sidebar.error(f"Erreur génération PDF: {pdf_exc}")

    if st.session_state.get("results_zip_bytes"):
        st.sidebar.download_button(
            label="⬇️ Download Results (ZIP)",
            data=st.session_state["results_zip_bytes"],
            file_name=os.path.basename(
                st.session_state.get("results_zip_path", _build_results_zip_filename(current_conf))
            ),
            mime="application/zip",
            width="stretch",
            help=(
                "Télécharge l'archive des résultats en mémoire (JSON/CSV/trades selon disponibilité). "
                "Règle de nommage: "
                "`results_wfo_<YYYYMMDD_HHMMSS>_<periode>_<timeframe>_<methode>_<nb_fenetres>_<trials>_<regime>.zip`."
            )
        )
    if st.session_state.get("results_pdf_bytes"):
        st.sidebar.download_button(
            label="⬇️ Download Report (PDF)",
            data=st.session_state["results_pdf_bytes"],
            file_name=os.path.basename(
                st.session_state.get("results_pdf_path", _build_results_pdf_filename(current_conf))
            ),
            mime="application/pdf",
            width="stretch",
            help=(
                "Télécharge le rapport PDF consolidé (WFO + final backtest). "
                "Règle de nommage: "
                "`report_wfo_<YYYYMMDD_HHMMSS>_<periode>_<timeframe>_<methode>_<nb_fenetres>_<trials>_<regime>.pdf`."
            )
        )
else:
    st.sidebar.info("Run an optimization or load a results ZIP to enable export.")

# Run Final Backtest Button (Conditional)
if 'wfo_results' in st.session_state:
    st.sidebar.divider()
    st.sidebar.subheader("🗓️ Final Backtest Range")
    default_final_start = st.session_state.get('opt_start_date', st.session_state.get('start_date', DEFAULT_START_DATE))
    default_final_end = st.session_state.get('opt_end_date', st.session_state.get('end_date', DEFAULT_END_DATE))
    final_start_date = st.sidebar.text_input(
        "Final Start Date (YYYY-MM-DD)",
        value=default_final_start,
        key="final_start_date",
        help="Date de début du jeu de données utilisé pour le backtest final."
    )
    final_end_date = st.sidebar.text_input(
        "Final End Date (YYYY-MM-DD)",
        value=default_final_end,
        key="final_end_date",
        help="Date de fin du jeu de données utilisé pour le backtest final."
    )
    final_file_path = st.sidebar.text_input(
        "Final Data File Path",
        value=st.session_state.get('file_path', DEFAULT_DATA_FILE),
        key="final_file_path",
        help="Chemin du fichier de données pour le backtest final (si source locale)."
    )
    if st.sidebar.button(
        "🏆 Run Final Backtest",
        width="stretch",
        help=(
            "Exécute un backtest complet avec le jeu de paramètres final. "
            "Source: best_window ou robust_set selon les options Robust Tests."
        )
    ):
        run_final_backtest_logic()

# ==============================================================================
# RESULTS VISUALIZATION
# ==============================================================================

if 'wfo_results' in st.session_state:
    results = st.session_state['wfo_results']
    if isinstance(results, dict):
        results["robust_set_summary"] = _build_robust_set_summary(results, current_conf)
        st.session_state["wfo_results"] = results
    df = st.session_state.get('df')
    traceability = results.get("traceability") or st.session_state.get("wfo_traceability")
    all_trials_df_live = _build_trials_dataframe_from_results(results)
    if not all_trials_df_live.empty:
        st.session_state["all_trials_df"] = all_trials_df_live
    window_info_df_live = _build_window_info_dataframe(results)
    if not window_info_df_live.empty:
        st.session_state["window_info_df"] = window_info_df_live
    
    st.divider()
    st.header("📊 Optimization Results")
    if traceability:
        with st.expander("🧾 Traçabilité du run", expanded=False):
            run_meta = traceability.get("run", {}) if isinstance(traceability, dict) else {}
            config_sha = run_meta.get("config_sha256")
            config_sha_display = f"{config_sha[:12]}..." if isinstance(config_sha, str) and config_sha else "n/a"
            c1, c2, c3 = st.columns(3)
            c1.metric("Run ID", str(run_meta.get("run_id", "n/a")))
            c2.metric("Status", str(run_meta.get("status", "n/a")))
            c3.metric("Config SHA256", config_sha_display)
            strategy_mode_label = str(
                run_meta.get("strategy_mode")
                or (traceability.get("config", {}) if isinstance(traceability, dict) else {}).get("strategy_mode")
                or "n/a"
            )
            strategy_id_label = str(
                run_meta.get("strategy_id")
                or (traceability.get("config", {}) if isinstance(traceability, dict) else {}).get("strategy_id")
                or "n/a"
            )
            st.caption(f"Strategy Mode: `{strategy_mode_label}` | Strategy ID: `{strategy_id_label}`")
            st.json(traceability)
    
    # 1. Summary Metrics
    if results['out_of_sample_performance']:
        oos_df = pd.DataFrame(results['out_of_sample_performance'])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Return", f"{oos_df['return'].mean():.2f}%")
        col2.metric("Avg Sharpe", f"{oos_df['sharpe'].mean():.2f}")
        col3.metric("Avg Max Drawdown", f"{oos_df['max_drawdown'].mean():.2f}%")
        col4.metric("Avg Win Rate", f"{oos_df['win_rate'].mean():.2f}%")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 OOS Performance",
        "🔍 Parameters",
        "📉 Drawdowns & Returns",
        "🧠 Adaptive Insights",
        "🤖 Expert IA",
        "📋 Raw Data",
        "🏆 Final Backtest"
    ])
    
    with tab1:
        # Combined Chart: Price + Windows
        st.subheader("Price Series with Walk-Forward Windows")
        
        if df is None or df.empty:
            st.warning("Price data (df) not available. Re-run optimization or include df.csv in the results ZIP.")
        else:
            # Using Plotly for interactive chart
            fig = go.Figure()
            
            # Price Line
            if 'Close' in df.columns:
                price_col = 'Close'
            else:
                price_col = df.columns[0]
                
            fig.add_trace(go.Scatter(x=df.index, y=df[price_col], mode='lines', name='Price', line=dict(color='#1f77b4', width=1)))
            
            # Add Windows
            colors = {'train': 'rgba(0, 255, 0, 0.1)', 'test': 'rgba(255, 0, 0, 0.1)'}
            first_train_labeled = False
            first_test_labeled = False
            out_of_range_windows = 0
            df_x_min = None
            df_x_max = None
            try:
                if len(df.index) > 0:
                    df_x_min = pd.to_datetime(df.index.min(), errors="coerce")
                    df_x_max = pd.to_datetime(df.index.max(), errors="coerce")
            except Exception:
                df_x_min = None
                df_x_max = None

            for i, window in enumerate(results['window_results']):
                info = window['window_info']
                # IS
                if info['in_sample_start'] and info['in_sample_end']:
                    rect_kwargs = dict(
                        x0=info['in_sample_start'], x1=info['in_sample_end'],
                        fillcolor=colors['train'], layer="below", line_width=0
                    )
                    if not first_train_labeled:
                        rect_kwargs["annotation_text"] = f"W{i+1} Train"
                        first_train_labeled = True
                    fig.add_vrect(**rect_kwargs)
                # OOS
                if info['out_sample_start'] and info['out_sample_end']:
                    rect_kwargs = dict(
                        x0=info['out_sample_start'], x1=info['out_sample_end'],
                        fillcolor=colors['test'], layer="below", line_width=0
                    )
                    if not first_test_labeled:
                        rect_kwargs["annotation_text"] = f"W{i+1} Test"
                        first_test_labeled = True
                    fig.add_vrect(**rect_kwargs)

                # Detect window/date mismatch between plotted df and WFO windows.
                if df_x_min is not None and df_x_max is not None:
                    try:
                        w_start = pd.to_datetime(info.get('start_date'), errors='coerce')
                        w_end = pd.to_datetime(info.get('end_date'), errors='coerce')
                        if pd.notna(w_start) and pd.notna(w_end):
                            if w_end < df_x_min or w_start > df_x_max:
                                out_of_range_windows += 1
                    except Exception:
                        pass
                    
            fig.update_layout(height=500, template="plotly_dark", title_text="Market Data & WFO Windows")
            if df_x_min is not None and df_x_max is not None and pd.notna(df_x_min) and pd.notna(df_x_max):
                # Keep x-axis aligned with the displayed price data range.
                fig.update_xaxes(range=[df_x_min, df_x_max])
            st.plotly_chart(fig, use_container_width=True)
            if out_of_range_windows > 0:
                st.info(
                    f"{out_of_range_windows} fenêtre(s) WFO sont hors de la plage de prix affichée. "
                    "Cela arrive si les résultats WFO importés et la série de prix active n'ont pas la même période."
                )
        
        # IS + OOS Performance per Window (shared scale)
        if results['out_of_sample_performance'] or results['in_sample_performance']:
            st.subheader("In-Sample vs Out-of-Sample Performance by Window")
            oos_metrics_df = pd.DataFrame(results['out_of_sample_performance'])
            is_metrics_df = pd.DataFrame(results['in_sample_performance'])

            if not oos_metrics_df.empty:
                oos_metrics_df['Window'] = oos_metrics_df['window'].astype(str)
            if not is_metrics_df.empty:
                is_metrics_df['Window'] = is_metrics_df['window'].astype(str)

            windows = sorted(
                set(oos_metrics_df.get('Window', [])) | set(is_metrics_df.get('Window', [])),
                key=lambda x: int(x)
            )

            fig_perf = make_subplots(specs=[[{"secondary_y": True}]])

            if not is_metrics_df.empty:
                fig_perf.add_trace(go.Bar(
                    x=windows,
                    y=is_metrics_df.set_index('Window').reindex(windows)['return'],
                    name="IS Return %",
                    marker_color='rgb(255, 127, 14)',
                    opacity=0.7
                ), secondary_y=False)

                fig_perf.add_trace(go.Scatter(
                    x=windows,
                    y=is_metrics_df.set_index('Window').reindex(windows)['sharpe'],
                    name="IS Sharpe",
                    mode='lines+markers',
                    line=dict(color='rgb(214, 39, 40)')
                ), secondary_y=True)

            if not oos_metrics_df.empty:
                fig_perf.add_trace(go.Bar(
                    x=windows,
                    y=oos_metrics_df.set_index('Window').reindex(windows)['return'],
                    name="OOS Return %",
                    marker_color='rgb(55, 83, 109)'
                ), secondary_y=False)

                fig_perf.add_trace(go.Scatter(
                    x=windows,
                    y=oos_metrics_df.set_index('Window').reindex(windows)['sharpe'],
                    name="OOS Sharpe",
                    mode='lines+markers',
                    line=dict(color='rgb(26, 118, 255)')
                ), secondary_y=True)

            fig_perf.update_layout(
                height=450,
                template="plotly_dark",
                barmode="group",
                title_text="Returns & Sharpe Ratio per Window (IS vs OOS)"
            )
            fig_perf.update_yaxes(title_text="Return %", secondary_y=False)
            fig_perf.update_yaxes(title_text="Sharpe Ratio", secondary_y=True)

            st.plotly_chart(fig_perf, use_container_width=True)

    with tab2:
        st.subheader("Parameter Stability Analysis")
        params_df = pd.DataFrame(results['best_params'])
        
        # Filter numeric parameters only
        numeric_cols = params_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ['window', 'metric1_name', 'metric2_name']] # Filter out non-params
        # Prefer selected params from the run being visualized (traceability/config in results ZIP),
        # then fallback to current UI config.
        selected_for_opt = []
        run_traceability = results.get("traceability") if isinstance(results, dict) else None
        if isinstance(run_traceability, dict):
            run_cfg = run_traceability.get("config")
            if isinstance(run_cfg, dict):
                selected_for_opt = run_cfg.get("selected_params") or []
        if not selected_for_opt:
            selected_for_opt = current_conf.get("selected_params", []) if isinstance(current_conf, dict) else []
        if not selected_for_opt:
            selected_for_opt = st.session_state.get('selected_params', [])
        if selected_for_opt:
            numeric_cols = [c for c in numeric_cols if c in selected_for_opt]
        
        if numeric_cols:
            # Normalize for heatmap
            norm_df = params_df[numeric_cols].copy()
            for col in norm_df.columns:
                if norm_df[col].max() != norm_df[col].min():
                    norm_df[col] = (norm_df[col] - norm_df[col].min()) / (norm_df[col].max() - norm_df[col].min())
                else:
                    norm_df[col] = 0.5 # Constant parameter
            
            fig_heat = px.imshow(
                norm_df.T, 
                labels=dict(x="Window", y="Parameter", color="Normalized Value"),
                x=list(range(1, len(params_df)+1)),
                aspect="auto",
                color_continuous_scale="Viridis"
            )
            fig_heat.update_layout(title="Parameter Evolution Across Windows (Normalized)", height=500)
            st.plotly_chart(fig_heat, use_container_width=True)
            
            # =========================================================================
            # BEST PARAMETERS TABLE PER WFO WINDOW
            # =========================================================================
            st.subheader("Best Parameters Table per WFO Window")

            # Create a table showing the best parameters for each WFO window
            if not params_df.empty:
                # Add a 'Window' column for clarity
                params_df_with_window = params_df.copy()
                params_df_with_window['Window'] = list(range(1, len(params_df) + 1))

                # Use Plotly for an interactive table
                fig_table = go.Figure(data=[go.Table(
                    header=dict(values=['Window'] + list(params_df.columns),
                                fill_color='paleturquoise',
                                align='left',
                                font=dict(size=12, color='black')),
                    cells=dict(values=[params_df_with_window['Window']] + [params_df_with_window[col] for col in params_df.columns],
                              fill_color='lavender',
                              align='left',
                              font=dict(size=11, color='black'))
                )])

                fig_table.update_layout(
                    title='Best Parameters Used for Backtesting in Each WFO Window',
                    title_font_size=16,
                    width=1200,
                    height=400
                )

                # Add description as annotation
                fig_table.add_annotation(
                    text="This table lists the optimal parameters selected during optimization for each Walk-Forward window.<br>"
                         "These were used to generate the backtest results shown in the out-of-sample performance.",
                    xref="paper", yref="paper",
                    x=0.5, y=-0.15,
                    showarrow=False,
                    font=dict(size=12),
                    align="center",
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=10
                )

                st.plotly_chart(fig_table, use_container_width=True)

            st.subheader("Robust Set (Level 1)")
            robust_summary = results.get("robust_set_summary", {}) if isinstance(results, dict) else {}
            if not isinstance(robust_summary, dict) or not robust_summary:
                st.info("Aucun résumé Robust Set disponible pour ce run.")
            else:
                r_c1, r_c2, r_c3, r_c4 = st.columns(4)
                r_c1.metric("Status", str(robust_summary.get("status", "n/a")))
                r_c2.metric("Top-N/fenêtre", str(robust_summary.get("top_n_per_window", "n/a")))
                windows_used = robust_summary.get("windows_used", []) or []
                min_req = robust_summary.get("min_windows_required", "n/a")
                r_c3.metric("Fenêtres utilisées", f"{len(windows_used)} / {min_req}")
                r_c4.metric("Candidats pool", str(robust_summary.get("candidates_total", 0)))

                if robust_summary.get("notes"):
                    st.info(" | ".join([str(x) for x in robust_summary.get("notes", [])]))

                robust_params = robust_summary.get("robust_params", {}) or {}
                if robust_params:
                    robust_params_df = pd.DataFrame(
                        [{"parameter": k, "robust_value": v} for k, v in robust_params.items()]
                    ).sort_values("parameter")
                    st.markdown("##### Paramètres robustes retenus")
                    st.dataframe(robust_params_df, width="stretch")

                support_df = pd.DataFrame(robust_summary.get("support_by_param", []) or [])
                if not support_df.empty:
                    st.markdown("##### Support par paramètre")
                    st.dataframe(support_df, width="stretch")
                    if {"parameter", "support_ratio"}.issubset(support_df.columns):
                        support_plot = support_df.copy()
                        support_plot["support_ratio"] = pd.to_numeric(support_plot["support_ratio"], errors="coerce")
                        support_plot = support_plot.dropna(subset=["support_ratio"]).sort_values("support_ratio", ascending=False)
                        if not support_plot.empty:
                            fig_support = px.bar(
                                support_plot,
                                x="parameter",
                                y="support_ratio",
                                template="plotly_dark",
                                title="Robust Set Support Ratio by Parameter",
                                labels={"support_ratio": "Support ratio pondéré", "parameter": "Paramètre"},
                            )
                            fig_support.update_layout(height=320)
                            st.plotly_chart(fig_support, use_container_width=True)

            # Removed duplicate raw parameters table
        else:
            st.warning("No numeric parameters to visualize.")
            st.subheader("Robust Set (Level 1)")
            robust_summary = results.get("robust_set_summary", {}) if isinstance(results, dict) else {}
            if not isinstance(robust_summary, dict) or not robust_summary:
                st.info("Aucun résumé Robust Set disponible pour ce run.")
            else:
                r_c1, r_c2, r_c3, r_c4 = st.columns(4)
                r_c1.metric("Status", str(robust_summary.get("status", "n/a")))
                r_c2.metric("Top-N/fenêtre", str(robust_summary.get("top_n_per_window", "n/a")))
                windows_used = robust_summary.get("windows_used", []) or []
                min_req = robust_summary.get("min_windows_required", "n/a")
                r_c3.metric("Fenêtres utilisées", f"{len(windows_used)} / {min_req}")
                r_c4.metric("Candidats pool", str(robust_summary.get("candidates_total", 0)))

    with tab3:
        if results['out_of_sample_performance']:
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Win Rate Distribution")
                fig_hist = px.histogram(oos_df, x="win_rate", nbins=10, title="Win Rate Distribution", template="plotly_dark")
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col_b:
                st.subheader("Drawdown Distribution")
                fig_dd = px.histogram(oos_df, x="max_drawdown", nbins=10, title="Max Drawdown Distribution", template="plotly_dark", color_discrete_sequence=['red'])
                st.plotly_chart(fig_dd, use_container_width=True)

    with tab4:
        st.subheader("Adaptive Optimization Insights")
        run_mode = str(
            results.get("mode")
            or results.get("settings", {}).get("optimization_regime")
            or current_conf.get("optimization_regime", "classic")
        ).lower()

        if run_mode != "adaptive_continuous":
            st.info("Ces graphiques sont disponibles pour le mode `Adaptive Continuous`.")
        else:
            trials_df = st.session_state.get("all_trials_df")
            if trials_df is None or trials_df.empty:
                trials_df = _build_trials_dataframe_from_results(results)
            guidance_df = pd.DataFrame(results.get("adaptive_guidance", []))
            is_perf_df = pd.DataFrame(results.get("in_sample_performance", []))
            oos_perf_df = pd.DataFrame(results.get("out_of_sample_performance", []))

            # Fallback if adaptive_guidance is not present in old runs/imports.
            if guidance_df.empty and results.get("window_results"):
                fallback_rows = []
                baseline_default = (
                    results.get("settings", {}).get("baseline_param_combinations")
                    or results.get("timing", {}).get("param_combinations")
                )
                for wr in results.get("window_results", []):
                    cycle_info = wr.get("cycle_info", {}) if isinstance(wr, dict) else {}
                    window_info = wr.get("window_info", {}) if isinstance(wr, dict) else {}
                    cycle_id = cycle_info.get("cycle", window_info.get("window"))
                    fallback_rows.append({
                        "window": cycle_id,
                        "cycle": cycle_id,
                        "baseline_combinations": baseline_default,
                        "active_combinations": cycle_info.get("active_combinations"),
                        "trials_tested": wr.get("optimization_trials_count", cycle_info.get("trials_tested")),
                        "parameter_weights": {},
                    })
                guidance_df = pd.DataFrame(fallback_rows)

            # --- 1) Convergence by cycle ---
            st.markdown("#### 1) Convergence des scores d'entraînement")
            if not trials_df.empty and {"window", "combined_score"}.issubset(trials_df.columns):
                cycle_scores = trials_df[["window", "combined_score"]].copy()
                cycle_scores["window"] = pd.to_numeric(cycle_scores["window"], errors="coerce")
                cycle_scores["combined_score"] = pd.to_numeric(cycle_scores["combined_score"], errors="coerce")
                cycle_scores = cycle_scores.dropna(subset=["window", "combined_score"])
                if not cycle_scores.empty:
                    conv_df = cycle_scores.groupby("window")["combined_score"].agg(
                        best="max",
                        median="median",
                        q25=lambda x: x.quantile(0.25),
                        q75=lambda x: x.quantile(0.75),
                        std="std",
                        n="count"
                    ).reset_index().sort_values("window")
                    conv_df["std"] = conv_df["std"].fillna(0.0)

                    fig_conv = go.Figure()
                    fig_conv.add_trace(go.Scatter(
                        x=conv_df["window"],
                        y=conv_df["q75"],
                        mode="lines",
                        line=dict(width=0),
                        name="Q75",
                        showlegend=False,
                        hovertemplate="Cycle %{x}<br>Q75: %{y:.2f}<extra></extra>"
                    ))
                    fig_conv.add_trace(go.Scatter(
                        x=conv_df["window"],
                        y=conv_df["q25"],
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(99, 110, 250, 0.16)",
                        name="IQR (Q25-Q75)",
                        hovertemplate="Cycle %{x}<br>Q25: %{y:.2f}<extra></extra>"
                    ))
                    fig_conv.add_trace(go.Scatter(
                        x=conv_df["window"],
                        y=conv_df["best"],
                        mode="lines+markers",
                        name="Best score",
                        line=dict(color="#00CC96", width=2),
                        hovertemplate="Cycle %{x}<br>Best: %{y:.2f}<extra></extra>"
                    ))
                    fig_conv.add_trace(go.Scatter(
                        x=conv_df["window"],
                        y=conv_df["median"],
                        mode="lines+markers",
                        name="Median score",
                        line=dict(color="#FECB52", width=1.7, dash="dot"),
                        hovertemplate="Cycle %{x}<br>Median: %{y:.2f}<extra></extra>"
                    ))
                    fig_conv.update_layout(
                        template="plotly_dark",
                        height=390,
                        title="Évolution des scores des trials par cycle",
                        xaxis_title="Cycle",
                        yaxis_title="Score (combined_score)"
                    )
                    st.plotly_chart(fig_conv, use_container_width=True)
                    _render_interpretation_guide([
                        "La courbe `Best score` qui monte indique que la recherche découvre de meilleures zones.",
                        "Une bande IQR (Q25-Q75) qui se resserre signale une recherche plus stable.",
                        "Si `Best` monte mais que `Median` reste basse, les bons résultats sont rares (risque de sur-ajustement local).",
                    ])
                else:
                    st.info("Données de trial insuffisantes pour afficher la convergence.")
            else:
                st.info("Les colonnes `window` et `combined_score` sont nécessaires dans `all_trials.csv`.")

            # --- 2) Generalization gap IS vs OOS ---
            st.markdown("#### 2) Gap de généralisation (IS vs OOS)")
            if not is_perf_df.empty and not oos_perf_df.empty and "window" in is_perf_df.columns and "window" in oos_perf_df.columns:
                is_gap = is_perf_df[["window", "return", "sharpe"]].copy()
                oos_gap = oos_perf_df[["window", "return", "sharpe"]].copy()
                is_gap = is_gap.rename(columns={"return": "is_return", "sharpe": "is_sharpe"})
                oos_gap = oos_gap.rename(columns={"return": "oos_return", "sharpe": "oos_sharpe"})
                gap_df = pd.merge(is_gap, oos_gap, on="window", how="inner")
                gap_df["window"] = pd.to_numeric(gap_df["window"], errors="coerce")
                for col in ["is_return", "oos_return", "is_sharpe", "oos_sharpe"]:
                    gap_df[col] = pd.to_numeric(gap_df[col], errors="coerce")
                gap_df = gap_df.dropna(subset=["window"]).sort_values("window")
                gap_df["return_gap"] = gap_df["is_return"] - gap_df["oos_return"]
                gap_df["sharpe_gap"] = gap_df["is_sharpe"] - gap_df["oos_sharpe"]

                if not gap_df.empty:
                    fig_gap = go.Figure()
                    fig_gap.add_trace(go.Bar(
                        x=gap_df["window"],
                        y=gap_df["return_gap"],
                        name="Gap Return (IS - OOS)",
                        marker_color="rgba(239, 85, 59, 0.75)",
                        hovertemplate="Cycle %{x}<br>Gap Return: %{y:.2f}<extra></extra>"
                    ))
                    fig_gap.add_trace(go.Scatter(
                        x=gap_df["window"],
                        y=gap_df["sharpe_gap"],
                        mode="lines+markers",
                        name="Gap Sharpe (IS - OOS)",
                        line=dict(color="#19D3F3", width=2),
                        hovertemplate="Cycle %{x}<br>Gap Sharpe: %{y:.2f}<extra></extra>"
                    ))
                    fig_gap.add_hline(y=0, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.6)")
                    fig_gap.update_layout(
                        template="plotly_dark",
                        height=360,
                        title="Écart de performance entre entraînement et validation",
                        xaxis_title="Cycle"
                    )
                    st.plotly_chart(fig_gap, use_container_width=True)
                    _render_interpretation_guide([
                        "Un gap proche de 0 signifie une meilleure généralisation hors-échantillon.",
                        "Un gap durablement positif (IS > OOS) signale un risque de sur-optimisation.",
                        "Si le gap se réduit au fil des cycles, l'adaptation devient plus robuste.",
                    ])
                else:
                    st.info("Impossible de calculer le gap IS/OOS sur ce run.")
            else:
                st.info("Les métriques IS et OOS par cycle sont requises pour ce graphique.")

            # --- 3) Search space vs effort ---
            st.markdown("#### 3) Compression de l'espace de recherche")
            if not guidance_df.empty:
                gdf = guidance_df.copy()
                gdf["cycle"] = pd.to_numeric(
                    gdf["cycle"] if "cycle" in gdf.columns else gdf.get("window"),
                    errors="coerce"
                )
                gdf["window"] = pd.to_numeric(gdf.get("window", gdf["cycle"]), errors="coerce")
                gdf["active_combinations"] = pd.to_numeric(gdf.get("active_combinations"), errors="coerce")
                gdf["baseline_combinations"] = pd.to_numeric(gdf.get("baseline_combinations"), errors="coerce")
                gdf["trials_tested"] = pd.to_numeric(gdf.get("trials_tested"), errors="coerce")
                gdf = gdf.dropna(subset=["cycle"]).sort_values("cycle")

                if not gdf.empty:
                    fig_space = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_space.add_trace(go.Bar(
                        x=gdf["cycle"],
                        y=gdf["trials_tested"],
                        name="Trials testés",
                        marker_color="rgba(254, 203, 82, 0.72)",
                        hovertemplate="Cycle %{x}<br>Trials: %{y:.0f}<extra></extra>"
                    ), secondary_y=False)
                    fig_space.add_trace(go.Scatter(
                        x=gdf["cycle"],
                        y=gdf["active_combinations"],
                        mode="lines+markers",
                        name="Combinaisons actives",
                        line=dict(color="#00CC96", width=2),
                        hovertemplate="Cycle %{x}<br>Active combos: %{y:,.0f}<extra></extra>"
                    ), secondary_y=True)
                    if gdf["baseline_combinations"].notna().any():
                        fig_space.add_trace(go.Scatter(
                            x=gdf["cycle"],
                            y=gdf["baseline_combinations"],
                            mode="lines",
                            name="Combinaisons baseline",
                            line=dict(color="#AB63FA", width=1.5, dash="dot"),
                            hovertemplate="Cycle %{x}<br>Baseline combos: %{y:,.0f}<extra></extra>"
                        ), secondary_y=True)

                    fig_space.update_layout(
                        template="plotly_dark",
                        height=380,
                        title="Effort de test vs taille de la grille active",
                        xaxis_title="Cycle"
                    )
                    fig_space.update_yaxes(title_text="Trials", secondary_y=False)
                    if (gdf["active_combinations"] > 0).any():
                        fig_space.update_yaxes(title_text="Combinaisons", type="log", secondary_y=True)
                    else:
                        fig_space.update_yaxes(title_text="Combinaisons", secondary_y=True)
                    st.plotly_chart(fig_space, use_container_width=True)
                    _render_interpretation_guide([
                        "La courbe `Combinaisons actives` doit en général baisser vs baseline: le moteur se focalise.",
                        "Les `Trials testés` doivent rester compatibles avec le temps de calcul visé.",
                        "Si les combinaisons explosent sans gain de score, la configuration est trop exploratoire.",
                    ])
                else:
                    st.info("Données de guidance adaptative insuffisantes.")
            else:
                st.info("Aucune donnée `adaptive_guidance` détectée pour ce run.")

            # --- 4) Parameter weights evolution ---
            st.markdown("#### 4) Évolution des poids relatifs par paramètre")
            weight_rows = []
            if not guidance_df.empty:
                for _, row in guidance_df.iterrows():
                    cycle_id = row.get("cycle", row.get("window"))
                    weights = row.get("parameter_weights", {})
                    if isinstance(weights, str):
                        try:
                            weights = json.loads(weights)
                        except Exception:
                            weights = {}
                    if not isinstance(weights, dict):
                        continue
                    for pname, weight in weights.items():
                        weight_rows.append({
                            "cycle": cycle_id,
                            "parameter": str(pname),
                            "weight": weight
                        })

            weights_df = pd.DataFrame(weight_rows)
            if not weights_df.empty:
                weights_df["cycle"] = pd.to_numeric(weights_df["cycle"], errors="coerce")
                weights_df["weight"] = pd.to_numeric(weights_df["weight"], errors="coerce")
                weights_df = weights_df.dropna(subset=["cycle", "weight"])
                if not weights_df.empty:
                    top_params = (
                        weights_df.groupby("parameter")["weight"]
                        .mean()
                        .sort_values(ascending=False)
                        .head(8)
                        .index
                    )
                    plot_weights_df = weights_df[weights_df["parameter"].isin(top_params)].sort_values("cycle")
                    fig_weights = px.area(
                        plot_weights_df,
                        x="cycle",
                        y="weight",
                        color="parameter",
                        template="plotly_dark",
                        title="Poids relatifs des paramètres (top 8)"
                    )
                    fig_weights.update_layout(height=380, xaxis_title="Cycle", yaxis_title="Poids relatif")
                    st.plotly_chart(fig_weights, use_container_width=True)
                    _render_interpretation_guide([
                        "Un poids élevé indique qu'un paramètre discrimine fortement les scores dans l'historique.",
                        "Des poids qui changent brutalement peuvent signaler un régime instable.",
                        "Un paramètre durablement proche de 0 a peu d'impact dans la configuration actuelle.",
                    ])
                else:
                    st.info("Aucun poids paramètre exploitable pour ce run.")
            else:
                st.info("Poids adaptatifs non disponibles (run ancien ou export incomplet).")

            # --- 5) Top values leaderboard ---
            st.markdown("#### 5) Valeurs les plus performantes par paramètre")
            top_values = (results.get("adaptive_summary") or {}).get("top_values_by_parameter", {})
            if isinstance(top_values, dict) and top_values:
                param_options = sorted(top_values.keys())
                selected_param = st.selectbox(
                    "Paramètre à analyser",
                    options=param_options,
                    key="adaptive_top_values_param"
                )
                selected_rows = top_values.get(selected_param) or []
                selected_df = pd.DataFrame(selected_rows)
                if not selected_df.empty and {"value", "mean_score"}.issubset(selected_df.columns):
                    selected_df["mean_score"] = pd.to_numeric(selected_df["mean_score"], errors="coerce")
                    selected_df["effective_trials"] = pd.to_numeric(selected_df.get("effective_trials"), errors="coerce")
                    selected_df["value_label"] = selected_df["value"].astype(str)
                    selected_df = selected_df.sort_values("mean_score", ascending=False)
                    fig_top_values = px.bar(
                        selected_df,
                        x="value_label",
                        y="mean_score",
                        color="effective_trials",
                        text=selected_df["effective_trials"].fillna(0).astype(int),
                        template="plotly_dark",
                        title=f"Top valeurs historiques pour `{selected_param}`",
                        labels={"value_label": "Valeur", "mean_score": "Score moyen", "effective_trials": "Trials effectifs"}
                    )
                    fig_top_values.update_layout(height=350)
                    st.plotly_chart(fig_top_values, use_container_width=True)
                    _render_interpretation_guide([
                        "La barre la plus haute donne la valeur historiquement la plus robuste sur ce run.",
                        "Le nombre de `trials effectifs` aide à distinguer un vrai signal d'un résultat peu observé.",
                        "Si plusieurs valeurs sont proches, garder de la diversité évite de sur-spécialiser la grille.",
                    ])
                else:
                    st.info("Aucune donnée exploitable pour ce paramètre.")
            else:
                st.info("Le résumé `top_values_by_parameter` n'est pas disponible pour ce run.")

    with tab5:
        st.subheader("🤖 Expert IA - Interprétation MVP")
        st.caption(
            "Analyse IA spécialisée des résultats d'optimisation. "
            "Le MVP utilise un agent unique et un endpoint OpenAI-compatible."
        )
        with st.expander("Historique des rapports Expert sauvegardés", expanded=False):
            saved_reports = _list_saved_expert_reports(limit=300)
            if not saved_reports:
                st.info("Aucun rapport Expert sauvegardé trouvé dans `reports/expert/`.")
            else:
                report_labels = [_format_saved_expert_report_label(p) for p in saved_reports]
                selected_report_label = st.selectbox(
                    "Rapport sauvegardé",
                    options=report_labels,
                    key="expert_saved_report_select",
                    help="Liste triée du plus récent au plus ancien."
                )
                selected_report_path = saved_reports[report_labels.index(selected_report_label)]
                st.caption(f"Fichier: `{selected_report_path}`")
                l1, l2 = st.columns([1.2, 1.2])
                with l1:
                    if st.button("Charger rapport sauvegardé", key="expert_load_saved_report_btn", width="stretch"):
                        loaded_report, load_error = _load_saved_expert_report(selected_report_path)
                        if load_error:
                            st.error(f"Impossible de charger le rapport: {load_error}")
                        else:
                            st.session_state["expert_last_response"] = loaded_report
                            st.success("Rapport sauvegardé chargé dans l'UI Expert.")
                            st.rerun()
                with l2:
                    if st.button("Vider rapport affiché", key="expert_clear_loaded_report_btn", width="stretch"):
                        st.session_state.pop("expert_last_response", None)
                        st.session_state.pop("expert_followup_history", None)
                        st.info("Rapport Expert retiré de l'affichage courant.")
                        st.rerun()

        exp_c1, exp_c2 = st.columns(2)
        with exp_c1:
            expert_provider = st.selectbox(
                "Provider",
                options=["openai", "grok", "gemini"],
                key="expert_provider",
                format_func=lambda v: {"openai": "OpenAI", "grok": "Grok", "gemini": "Gemini"}.get(v, v),
                help="Provider LLM à utiliser pour l'analyse Expert."
            )
            expert_api_key = st.text_input(
                "API Key Expert",
                type="password",
                key="expert_api_key",
                help="Non sauvegardée dans les exports.",
            )
            expert_mode = st.selectbox(
                "Mode d'analyse",
                options=["diagnostic", "summary", "action_plan", "alerts"],
                key="expert_mode",
            )
        with exp_c2:
            provider_models = _get_expert_model_entries(expert_provider)
            provider_model_ids = [m.get("id") for m in provider_models if isinstance(m, dict) and m.get("id")]
            provider_model_ids = provider_model_ids if provider_model_ids else ["gpt-5.2"]
            model_options = provider_model_ids + ["__custom__"]
            previous_model_choice = str(st.session_state.get("expert_model_select", model_options[0]))
            if previous_model_choice not in model_options:
                st.session_state["expert_model_select"] = model_options[0]

            expert_model_choice = st.selectbox(
                "Modèle LLM",
                options=model_options,
                key="expert_model_select",
                format_func=lambda mid: (
                    "Autre (saisie libre)"
                    if mid == "__custom__"
                    else next((m.get("label", mid) for m in provider_models if m.get("id") == mid), mid)
                ),
                help="Choisis un modèle préconfiguré ou saisis un identifiant personnalisé."
            )
            default_model = provider_model_ids[0]
            if expert_model_choice == "__custom__":
                expert_model = st.text_input(
                    "Modèle personnalisé",
                    value=str(st.session_state.get("expert_model_custom", default_model)),
                    key="expert_model_custom",
                    help="Identifiant exact du modèle à appeler via l'API du provider.",
                ).strip() or default_model
            else:
                expert_model = expert_model_choice
                model_desc = next((m.get("description", "") for m in provider_models if m.get("id") == expert_model_choice), "")
                if model_desc:
                    st.caption(f"Modèle: {model_desc}")

            with st.expander("Voir les modèles disponibles", expanded=False):
                for model_item in provider_models:
                    mid = str(model_item.get("id", ""))
                    mdesc = str(model_item.get("description", ""))
                    st.markdown(f"- `{mid}`: {mdesc}")

            expert_detail_level = st.selectbox(
                "Niveau de détail",
                options=["standard", "short", "expert"],
                key="expert_detail_level",
            )
            expert_question = st.text_area(
                "Question complémentaire (optionnel)",
                key="expert_user_question",
                placeholder="Ex: Où vois-tu le plus grand risque de sur-optimisation ?"
            )

        with st.expander("Paramètres avancés Expert", expanded=False):
            provider_defaults = _get_expert_provider_defaults(expert_provider)
            is_openai_gpt5_model = str(expert_provider).lower() == "openai" and str(expert_model).lower().startswith("gpt-5")
            adv_c1, adv_c2, adv_c3 = st.columns(3)
            with adv_c1:
                expert_base_url = st.text_input(
                    "Base URL (optionnel)",
                    key="expert_base_url",
                    placeholder=provider_defaults["base_url"],
                    help=f"Laisse vide pour utiliser l'URL par défaut {provider_defaults['label']}: {provider_defaults['base_url']}",
                )
                expert_temp = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.2,
                    step=0.05,
                    key="expert_temperature",
                )
                if is_openai_gpt5_model and float(expert_temp) != 1.0:
                    st.caption("Info: sur OpenAI GPT-5, la temperature personnalisée est ignorée (valeur par défaut imposée).")
            with adv_c2:
                expert_max_tokens = st.number_input(
                    "Max tokens",
                    min_value=300,
                    max_value=8000,
                    value=3000,
                    step=100,
                    key="expert_max_tokens",
                )
                expert_timeout = st.number_input(
                    "Timeout (s)",
                    min_value=10,
                    max_value=300,
                    value=90,
                    step=5,
                    key="expert_timeout_s",
                )
            with adv_c3:
                expert_retries = st.number_input(
                    "Retries",
                    min_value=0,
                    max_value=6,
                    value=2,
                    step=1,
                    key="expert_retries",
                )
                include_raw_evidence = st.checkbox(
                    "Inclure preuves brutes",
                    value=True,
                    key="expert_include_raw_evidence",
                )

        # Prompt preview/editing (phase 1): strategy-aware input + deterministic alerts + template management.
        expert_input_preview = _build_expert_input_data(results, current_conf)
        preview_alerts = expert_input_preview.deterministic_alerts if expert_input_preview else []
        preview_strategy = expert_input_preview.strategy_context if expert_input_preview else {}
        preview_final_backtest = expert_input_preview.final_backtest if expert_input_preview else {}

        st.markdown("#### Vérification des données envoyées à l'Expert")
        if expert_input_preview is not None:
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Fenêtres IS", str(len(expert_input_preview.in_sample_performance or [])))
            d2.metric("Fenêtres OOS", str(len(expert_input_preview.out_of_sample_performance or [])))
            d3.metric("Best Params (fenêtres)", str(len(expert_input_preview.best_params or [])))
            d4.metric("Trials compacts", str(len(expert_input_preview.all_trials or [])))
        else:
            st.warning("Aucune donnée Expert construite à partir des résultats courants.")

        st.markdown("#### Pré-diagnostic automatique (sans IA)")
        _render_deterministic_alerts(preview_alerts)

        with st.expander("Contexte stratégie transmis à l'Expert", expanded=False):
            if preview_strategy:
                st.markdown("**Modules de sortie actifs**")
                st.write(preview_strategy.get("exit_modules", {}))
                st.markdown("**Paramètres d'entrée optimisés**")
                st.write(preview_strategy.get("entry_params_enabled", []))
                st.markdown("**Paramètres de sortie optimisés**")
                st.write(preview_strategy.get("exit_params_enabled", []))
                st.markdown("**Règles d'entrée (résumé)**")
                for item in preview_strategy.get("entry_logic_summary", []):
                    st.markdown(f"- {item}")
                st.markdown("**Règles de sortie (résumé)**")
                for item in preview_strategy.get("exit_logic_summary", []):
                    st.markdown(f"- {item}")
            else:
                st.info("Contexte stratégie indisponible.")

        with st.expander("Résumé final backtest transmis à l'Expert", expanded=False):
            if isinstance(preview_final_backtest, dict) and preview_final_backtest:
                st.write(preview_final_backtest)
            else:
                st.info("Aucune donnée de final backtest disponible pour ce run.")

        preview_req = ExpertRequest(
            mode=expert_mode,
            detail_level=expert_detail_level,
            user_question=expert_question.strip() if expert_question else None,
            include_raw_evidence=bool(include_raw_evidence),
        )
        preview_builder = ExpertPromptBuilder()
        default_system_prompt = preview_builder.build_system_prompt(preview_req.mode, preview_req.detail_level)
        if expert_input_preview is not None:
            default_user_prompt = preview_builder.build_user_prompt(
                data=expert_input_preview,
                request=preview_req,
                output_schema={
                    "schema_version": "expert.v1",
                    "required_keys": [
                        "schema_version",
                        "run_id",
                        "mode",
                        "detail_level",
                        "global_assessment",
                        "key_findings",
                        "recommended_actions",
                        "alerts",
                        "limitations",
                        "disclaimer",
                    ],
                },
            )
        else:
            default_user_prompt = (
                "Analyse les donnees ci-dessous et renvoie un JSON unique conforme au schema cible.\n\n"
                "Donnees: insufficient_data"
            )

        if "expert_system_prompt_edit" not in st.session_state:
            st.session_state["expert_system_prompt_edit"] = default_system_prompt
        if "expert_user_prompt_edit" not in st.session_state:
            st.session_state["expert_user_prompt_edit"] = default_user_prompt

        # Auto-refresh prompts when Expert input data changed (e.g. new ZIP loaded),
        # unless user explicitly locks custom prompts.
        prompt_data_signature = _sha256_json(
            {
                "run_id": getattr(getattr(expert_input_preview, "context", None), "run_id", None) if expert_input_preview else None,
                "is_rows": len(expert_input_preview.in_sample_performance or []) if expert_input_preview else 0,
                "oos_rows": len(expert_input_preview.out_of_sample_performance or []) if expert_input_preview else 0,
                "best_rows": len(expert_input_preview.best_params or []) if expert_input_preview else 0,
                "trials_rows": len(expert_input_preview.all_trials or []) if expert_input_preview else 0,
                "fb_available": bool((expert_input_preview.final_backtest or {}).get("available")) if expert_input_preview else False,
            }
        )

        # Apply deferred lock toggle before widget instantiation to avoid StreamlitAPIException.
        if st.session_state.pop("expert_lock_prompts_pending", False):
            st.session_state["expert_lock_prompts"] = True

        lock_custom_prompts = st.checkbox(
            "Conserver mes prompts personnalisés",
            key="expert_lock_prompts",
            help="Si désactivé, les prompts se rechargent automatiquement lors d'un changement de dataset/run.",
        )
        previous_sig = st.session_state.get("expert_prompt_data_signature")
        if prompt_data_signature and previous_sig != prompt_data_signature:
            if not lock_custom_prompts:
                st.session_state["expert_system_prompt_edit"] = default_system_prompt
                st.session_state["expert_user_prompt_edit"] = default_user_prompt
                st.caption("Prompts auto rechargés pour les données du run courant.")
            st.session_state["expert_prompt_data_signature"] = prompt_data_signature

        templates = _load_expert_prompt_templates()
        template_names = sorted(list(templates.keys()))
        st.markdown("#### Templates de prompts")
        active_tpl = str(st.session_state.get("expert_active_template_name", "") or "").strip()
        if active_tpl:
            st.caption(f"Template actif: `{active_tpl}`")
        tpl_c1, tpl_c2 = st.columns([1.4, 1])
        with tpl_c1:
            selected_tpl_name = st.selectbox(
                "Template existant",
                options=["(aucun)"] + template_names,
                key="expert_prompt_template_select",
                help="Sélectionne un template sauvegardé pour le charger ou le supprimer."
            )
        with tpl_c2:
            new_tpl_name = st.text_input(
                "Nom template",
                key="expert_prompt_template_name",
                placeholder="ex: strategy_v2_fr"
            )

        target_name = str(new_tpl_name or "").strip()
        has_name = bool(target_name)
        has_selection = selected_tpl_name != "(aucun)" and selected_tpl_name in templates

        if not has_name:
            st.caption("Pour sauvegarder: renseigne d'abord un nom de template.")
        if not has_selection:
            st.caption("Pour charger/supprimer: sélectionne un template existant.")

        tpl_btn_1, tpl_btn_2, tpl_btn_3 = st.columns([1, 1, 1])
        with tpl_btn_1:
            if st.button(
                "Sauvegarder template",
                key="expert_prompt_template_save",
                width="stretch",
                disabled=not has_name,
                help="Enregistre les prompts courants sous le nom saisi."
            ):
                templates[target_name] = {
                    "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
                    "mode": expert_mode,
                    "detail_level": expert_detail_level,
                    "system_prompt": str(st.session_state.get("expert_system_prompt_edit", "")),
                    "user_prompt": str(st.session_state.get("expert_user_prompt_edit", "")),
                }
                _save_expert_prompt_templates(templates)
                st.session_state["expert_active_template_name"] = target_name
                st.success(f"Template `{target_name}` sauvegardé.")
        with tpl_btn_2:
            if st.button(
                "Charger template",
                key="expert_prompt_template_load",
                width="stretch",
                disabled=not has_selection,
                help="Charge un template existant dans les zones de prompts."
            ):
                selected_tpl = templates[selected_tpl_name]
                resolved = _resolve_template_prompts(
                    selected_tpl,
                    default_system_prompt=default_system_prompt,
                    default_user_prompt=default_user_prompt
                )
                st.session_state["expert_system_prompt_edit"] = resolved["system_prompt"]
                st.session_state["expert_user_prompt_edit"] = resolved["user_prompt"]
                st.session_state["expert_active_template_name"] = selected_tpl_name
                # Prevent silent auto-reload from overriding loaded template prompts.
                st.session_state["expert_lock_prompts_pending"] = True
                if resolved["missing_system"] or resolved["missing_user"]:
                    missing_parts = []
                    if resolved["missing_system"]:
                        missing_parts.append("système")
                    if resolved["missing_user"]:
                        missing_parts.append("utilisateur")
                    st.warning(
                        "Template chargé avec fallback prompt auto pour: "
                        + ", ".join(missing_parts)
                        + "."
                    )
                st.success(f"Template `{selected_tpl_name}` chargé.")
                st.rerun()
        with tpl_btn_3:
            if st.button(
                "Supprimer template",
                key="expert_prompt_template_delete",
                width="stretch",
                disabled=not has_selection,
                help="Supprime définitivement le template sélectionné."
            ):
                templates.pop(selected_tpl_name, None)
                _save_expert_prompt_templates(templates)
                if st.session_state.get("expert_active_template_name") == selected_tpl_name:
                    st.session_state.pop("expert_active_template_name", None)
                st.success(f"Template `{selected_tpl_name}` supprimé.")
                st.rerun()

        st.markdown("##### Import / Export templates")
        export_payload = {
            "schema_version": "expert_prompt_templates.v1",
            "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "templates": templates,
        }
        io_c1, io_c2 = st.columns([1.2, 1.2])
        with io_c1:
            st.download_button(
                "Exporter templates (JSON)",
                data=json.dumps(export_payload, ensure_ascii=False, indent=2),
                file_name=f"expert_prompt_templates_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                width="stretch",
                help="Télécharge tous les templates Expert au format JSON."
            )
        with io_c2:
            imported_templates_file = st.file_uploader(
                "Importer templates (JSON)",
                type=["json"],
                key="expert_prompt_templates_import_file",
                help="Accepte un JSON avec `templates` (ou un objet templates direct)."
            )
            import_mode = st.radio(
                "Mode d'import",
                options=["Fusionner", "Remplacer"],
                horizontal=True,
                key="expert_prompt_templates_import_mode",
                help="Fusionner: ajoute/écrase par nom. Remplacer: efface et remplace tous les templates."
            )
            if st.button(
                "Appliquer import",
                key="expert_prompt_templates_import_apply",
                width="stretch",
                disabled=imported_templates_file is None
            ):
                try:
                    imported_templates_file.seek(0)
                    imported_data = json.load(imported_templates_file)
                    if not isinstance(imported_data, dict):
                        raise ValueError("Le JSON importé doit être un objet.")
                    imported_templates = imported_data.get("templates", imported_data)
                    if not isinstance(imported_templates, dict):
                        raise ValueError("Le champ `templates` doit être un objet {nom: template}.")

                    normalized_templates = {}
                    for raw_name, raw_tpl in imported_templates.items():
                        name = str(raw_name or "").strip()
                        if not name:
                            continue
                        tpl = raw_tpl if isinstance(raw_tpl, dict) else {"prompt": str(raw_tpl)}
                        resolved = _resolve_template_prompts(
                            tpl,
                            default_system_prompt=default_system_prompt,
                            default_user_prompt=default_user_prompt
                        )
                        normalized_templates[name] = {
                            "updated_at": str(tpl.get("updated_at", datetime.datetime.utcnow().isoformat() + "Z")),
                            "mode": str(tpl.get("mode", expert_mode)),
                            "detail_level": str(tpl.get("detail_level", expert_detail_level)),
                            "system_prompt": resolved["system_prompt"],
                            "user_prompt": resolved["user_prompt"],
                        }

                    if import_mode == "Remplacer":
                        templates = normalized_templates
                    else:
                        templates.update(normalized_templates)

                    _save_expert_prompt_templates(templates)
                    st.success(f"Import terminé: {len(normalized_templates)} template(s) traité(s).")
                    st.rerun()
                except Exception as import_exc:
                    st.error(f"Import impossible: {import_exc}")

        c_prompt_a, c_prompt_b = st.columns([1.2, 1.2])
        with c_prompt_a:
            if st.button("Charger prompts auto", key="expert_reload_prompts"):
                st.session_state["expert_system_prompt_edit"] = default_system_prompt
                st.session_state["expert_user_prompt_edit"] = default_user_prompt
                st.session_state.pop("expert_active_template_name", None)
                st.rerun()
        with c_prompt_b:
            st.caption("Les prompts ci-dessous sont ceux utilisés pour l'appel API.")
            st.caption(
                "Astuce template: ajoute `{{AUTO_WFO_CONTEXT}}` dans le prompt utilisateur "
                "pour injecter explicitement le contexte WFO auto."
            )

        with st.expander("Prompts de l'agent Expert (éditables)", expanded=False):
            st.text_area(
                "Prompt système",
                key="expert_system_prompt_edit",
                height=130,
                help="Règles de rôle et de style de l'agent Expert."
            )
            st.text_area(
                "Prompt utilisateur",
                key="expert_user_prompt_edit",
                height=260,
                help="Instruction de tâche + données injectées pour l'analyse."
            )

        if st.button("Générer l'interprétation Expert", key="expert_generate_btn", width="stretch", type="primary"):
            clean_expert_api_key = str(expert_api_key or "").strip()
            if not clean_expert_api_key:
                st.warning("Renseigne une clé API Expert valide (non vide après suppression des espaces).")
            else:
                expert_input = _build_expert_input_data(results, current_conf)
                if expert_input is None:
                    st.error("Impossible de construire les données d'entrée Expert.")
                else:
                    user_system_prompt = str(st.session_state.get("expert_system_prompt_edit", "") or "").strip()
                    user_prompt_raw = str(st.session_state.get("expert_user_prompt_edit", "") or "").strip()
                    if not user_system_prompt or not user_prompt_raw:
                        st.error("Les prompts Expert ne peuvent pas être vides.")
                    else:
                        effective_user_prompt, prompt_injection_mode = _build_effective_expert_user_prompt(
                            user_prompt_raw,
                            default_user_prompt
                        )
                        if prompt_injection_mode in {"auto_only", "auto_appended"}:
                            st.info(
                                "Contexte WFO auto-injecté dans le prompt utilisateur pour garantir "
                                "l'accès aux données d'optimisation."
                            )
                        clean_base_url = str(expert_base_url or "").strip()
                        if clean_base_url.endswith("/chat/completions"):
                            clean_base_url = clean_base_url[: -len("/chat/completions")]
                        if clean_base_url.endswith("/v1/chat/completions"):
                            clean_base_url = clean_base_url[: -len("/chat/completions")]

                        llm_cfg = LLMConfig(
                            provider=expert_provider,
                            model=expert_model.strip() or default_model,
                            api_key=clean_expert_api_key,
                            base_url=(clean_base_url or None),
                            temperature=float(expert_temp),
                            timeout_s=float(expert_timeout),
                            max_tokens=int(expert_max_tokens),
                            retries=int(expert_retries),
                        )
                        expert_req = ExpertRequest(
                            mode=expert_mode,
                            detail_level=expert_detail_level,
                            user_question=expert_question.strip() if expert_question else None,
                            include_raw_evidence=bool(include_raw_evidence),
                        )
                        gateway = OpenAICompatibleGateway()
                        analyzer = ExpertAnalyzer(gateway, ExpertPromptBuilder())
                        storage = ExpertStorage()
                        service = ExpertService(analyzer, storage)
                        with st.spinner("Analyse Expert IA en cours..."):
                            response = service.run(
                                data=expert_input,
                                request=expert_req,
                                llm_config=llm_cfg,
                                persist=True,
                                system_prompt_override=user_system_prompt,
                                user_prompt_override=effective_user_prompt,
                            )
                        st.session_state["expert_last_response"] = {
                            "status": response.status,
                            "result_json": response.result_json,
                            "raw_text": response.raw_text,
                            "model_info": response.model_info,
                            "timings_ms": response.timings_ms,
                            "warnings": response.warnings,
                            "run_id": response.run_id,
                            "deterministic_alerts": expert_input.deterministic_alerts,
                            "strategy_context": expert_input.strategy_context,
                            "final_backtest": expert_input.final_backtest,
                            "used_system_prompt": user_system_prompt,
                            "used_user_prompt": effective_user_prompt,
                            "used_user_prompt_raw": user_prompt_raw,
                            "prompt_injection_mode": prompt_injection_mode,
                            "used_template_name": st.session_state.get("expert_active_template_name"),
                        }
                        if response.status == "ok":
                            st.success("Interprétation Expert générée et sauvegardée dans `reports/expert/`.")
                        elif response.status == "partial":
                            st.warning("Interprétation Expert partielle générée et sauvegardée dans `reports/expert/`.")
                        else:
                            st.error("Échec de génération Expert. Consulte les avertissements pour le diagnostic détaillé.")

        expert_last = st.session_state.get("expert_last_response")
        if isinstance(expert_last, dict):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Status", str(expert_last.get("status", "n/a")))
            m2.metric("Provider", str(expert_last.get("model_info", {}).get("provider", "n/a")))
            m3.metric("Model", str(expert_last.get("model_info", {}).get("model", "n/a")))
            m4.metric("Latency", f"{int(expert_last.get('timings_ms', {}).get('total', 0))} ms")
            if expert_last.get("used_template_name"):
                st.caption(f"Template utilisé pour cette analyse: `{expert_last.get('used_template_name')}`")
            else:
                st.caption("Template utilisé pour cette analyse: `(aucun - prompts édités manuellement/auto)`")
            if expert_last.get("prompt_injection_mode"):
                st.caption(f"Injection contexte prompt: `{expert_last.get('prompt_injection_mode')}`")

            warnings_list = expert_last.get("warnings") or []
            if warnings_list:
                st.warning("Avertissements Expert:\n- " + "\n- ".join([str(w) for w in warnings_list]))

            st.markdown("#### Rapport Expert")
            report_markdown = _render_expert_human_report(
                expert_last.get("result_json", {}),
                meta={
                    "run_id": expert_last.get("run_id"),
                    "status": expert_last.get("status"),
                    "provider": expert_last.get("model_info", {}).get("provider"),
                    "model": expert_last.get("model_info", {}).get("model"),
                    "latency_ms": expert_last.get("timings_ms", {}).get("total"),
                    "deterministic_alerts": expert_last.get("deterministic_alerts", []),
                    "final_backtest": expert_last.get("final_backtest", {}),
                }
            )

            st.markdown("#### Question de suivi à l'Expert")
            st.caption("Pose une question sur l'analyse ci-dessus ou donne une instruction supplémentaire.")
            followup_history = st.session_state.get("expert_followup_history", [])
            if not isinstance(followup_history, list):
                followup_history = []

            mt_c1, mt_c2 = st.columns([1.2, 1.2])
            with mt_c1:
                followup_multi_turn = st.checkbox(
                    "Mode multi-tour",
                    value=True,
                    key="expert_followup_multi_turn",
                    help="Conserve l'historique de conversation et le transmet au prochain tour.",
                )
            with mt_c2:
                followup_context_turns = st.number_input(
                    "Tours de contexte",
                    min_value=1,
                    max_value=12,
                    value=6,
                    step=1,
                    key="expert_followup_context_turns",
                    help="Nombre de tours précédents réinjectés dans la prochaine question.",
                    disabled=not followup_multi_turn,
                )
            followup_question = st.text_area(
                "Votre question/instruction",
                key="expert_followup_prompt",
                height=120,
                placeholder="Ex: Quelle fenêtre montre le meilleur compromis robustesse/performance et pourquoi ?"
            )
            ask_col, clear_col = st.columns([1.2, 1])
            with ask_col:
                if st.button("Envoyer la question à l'Expert", key="expert_followup_submit", width="stretch"):
                    clean_expert_api_key = str(expert_api_key or "").strip()
                    question_text = str(followup_question or "").strip()
                    if not clean_expert_api_key:
                        st.warning("Renseigne une clé API Expert pour envoyer une question de suivi.")
                    elif not question_text:
                        st.warning("Saisis une question ou une instruction avant l'envoi.")
                    else:
                        clean_base_url = str(expert_base_url or "").strip()
                        if clean_base_url.endswith("/chat/completions"):
                            clean_base_url = clean_base_url[: -len("/chat/completions")]
                        if clean_base_url.endswith("/v1/chat/completions"):
                            clean_base_url = clean_base_url[: -len("/chat/completions")]

                        llm_cfg = LLMConfig(
                            provider=expert_provider,
                            model=expert_model.strip() or default_model,
                            api_key=clean_expert_api_key,
                            base_url=(clean_base_url or None),
                            temperature=float(expert_temp),
                            timeout_s=float(expert_timeout),
                            max_tokens=int(expert_max_tokens),
                            retries=int(expert_retries),
                        )
                        followup_gateway = OpenAICompatibleGateway()

                        convo_context = []
                        if followup_multi_turn and followup_history:
                            safe_n = int(followup_context_turns)
                            for ex in followup_history[-safe_n:]:
                                if not isinstance(ex, dict):
                                    continue
                                convo_context.append(
                                    {
                                        "created_at": ex.get("created_at"),
                                        "question": str(ex.get("question", "")),
                                        "answer": str(ex.get("answer", "")),
                                    }
                                )

                        followup_context = _build_followup_context_pack(
                            results=results,
                            expert_input_preview=expert_input_preview,
                            expert_last=expert_last,
                            question_text=question_text,
                        )
                        followup_context["conversation_history"] = convo_context
                        followup_system_prompt = (
                            "Tu es l'agent Expert IA de l'application WFO. "
                            "Tu réponds en français (tolérance aux anglicismes quant/trading). "
                            "Tu disposes d'un pack de données WFO détaillé (métriques IS/OOS, fenêtres, classements, trials). "
                            "Réponds de façon lisible pour humain: structuré, factuel, utile. "
                            "Évite les réponses JSON minifiées; privilégie un texte clair avec sections et puces. "
                            "Quand une notion est complexe, ajoute une explication pédagogique simple (1-3 phrases). "
                            "N'invente pas de données; si info absente, dis-le explicitement. "
                            "Si un historique multi-tour est fourni, tiens compte des tours précédents."
                        )
                        followup_user_prompt = (
                            "Question utilisateur:\n"
                            f"{question_text}\n\n"
                            "Contexte d'analyse Expert (JSON):\n"
                            f"{json.dumps(followup_context, ensure_ascii=False, default=str)}\n\n"
                            "Consignes:\n"
                            "- Réponds directement à la question.\n"
                            "- Utilise prioritairement les données chiffrées du pack fourni.\n"
                            "- Si la question concerne un classement de fenêtres, renvoie un classement explicite window->valeur.\n"
                            "- Cite les éléments clés du run/fenêtres quand disponibles.\n"
                            "- Propose 1 à 3 actions concrètes si pertinent.\n"
                            "- Si le sujet est technique, ajoute un encadré \"Explication simple\"."
                        )

                        with st.spinner("Réponse Expert en cours..."):
                            try:
                                followup_answer = followup_gateway.generate(
                                    followup_system_prompt,
                                    followup_user_prompt,
                                    llm_cfg,
                                )
                                history = st.session_state.get("expert_followup_history", [])
                                if not isinstance(history, list):
                                    history = []
                                history.append(
                                    {
                                        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                                        "question": question_text,
                                        "answer": str(followup_answer or "").strip(),
                                    }
                                )
                                st.session_state["expert_followup_history"] = history[-50:]
                            except Exception as e:
                                st.error(f"Erreur lors de la question de suivi Expert: {e}")
            with clear_col:
                if st.button("Effacer l'historique Q/R", key="expert_followup_clear", width="stretch"):
                    st.session_state["expert_followup_history"] = []

            followup_history = st.session_state.get("expert_followup_history", [])
            if isinstance(followup_history, list) and followup_history:
                st.markdown("#### Conversation Expert")
                st.caption("Ordre d'affichage: du plus récent au plus ancien.")
                if followup_multi_turn:
                    visible_turns = list(reversed(followup_history[-int(followup_context_turns):]))
                else:
                    visible_turns = [followup_history[-1]]

                for idx, ex in enumerate(visible_turns, start=1):
                    if not isinstance(ex, dict):
                        continue
                    q_text = str(ex.get("question", "")).strip()
                    a_text = str(ex.get("answer", "")).strip()
                    a_display = _format_followup_answer_for_display(a_text)
                    created_at = str(ex.get("created_at", "")).strip()
                    st.caption(f"Tour {idx} • {created_at}" if created_at else f"Tour {idx}")
                    if q_text:
                        st.markdown(
                            "<div class='expert-followup-question-preview'>"
                            "<strong>Votre question</strong><br>"
                            f"{html.escape(q_text).replace(chr(10), '<br>')}"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                    if a_display:
                        st.markdown(
                            "<div class='expert-followup-answer-box'>"
                            "<strong>Réponse Expert</strong><br>"
                            f"{html.escape(a_display).replace(chr(10), '<br>')}"
                            "</div>",
                            unsafe_allow_html=True,
                        )

            download_payload = report_markdown or "Rapport Expert indisponible."
            st.download_button(
                "📥 Télécharger le rapport Expert (Markdown)",
                data=download_payload,
                file_name=f"expert_report_{expert_last.get('run_id', 'run')}.md",
                mime="text/markdown",
                key="expert_download_report"
            )

    with tab6:
        st.subheader("Detailed Results Data")
        st.write("Out-of-Sample Metrics:")
        st.dataframe(pd.DataFrame(results['out_of_sample_performance']))
        
        st.write("In-Sample Metrics:")
        st.dataframe(pd.DataFrame(results['in_sample_performance']))
        
        st.write("Full Results Object (JSON):")
        with st.expander("Show JSON"):
            # Exclude large dataframes for display
            clean_res = {k:v for k,v in results.items() if k not in ['window_results']}
            st.json(clean_res)

        if st.session_state.get("window_info_df") is not None:
            st.write("Window Info:")
            st.dataframe(st.session_state["window_info_df"], width="stretch")

        if st.session_state.get("all_trials_df") is not None:
            trials_df = st.session_state["all_trials_df"].copy()
            st.write(f"All Trials (rows brutes: {len(trials_df):,})")

            c_trials_1, c_trials_2, c_trials_3 = st.columns([1.6, 1.2, 1.2])
            with c_trials_1:
                window_options = []
                if "window" in trials_df.columns:
                    try:
                        window_options = sorted([int(w) for w in pd.Series(trials_df["window"]).dropna().unique().tolist()])
                    except Exception:
                        window_options = sorted(pd.Series(trials_df["window"]).dropna().unique().tolist())
                selected_windows = st.multiselect(
                    "Filtrer fenêtres",
                    options=window_options,
                    default=window_options,
                    key="all_trials_window_filter"
                )
            with c_trials_2:
                score_col = "combined_score" if "combined_score" in trials_df.columns else None
                min_score = None
                if score_col:
                    score_series = pd.to_numeric(trials_df[score_col], errors="coerce")
                    score_series = score_series.replace([np.inf, -np.inf], np.nan).dropna()
                    if not score_series.empty:
                        score_min = float(score_series.min())
                        score_max = float(score_series.max())
                        if not np.isfinite(score_min) or not np.isfinite(score_max):
                            score_min = None
                            score_max = None
                        if score_min is not None and score_max is not None:
                            if score_max < score_min:
                                score_min, score_max = score_max, score_min
                            # Sanitize persisted widget state (can contain -inf from previous runs).
                            current_min = st.session_state.get("all_trials_score_min", score_min)
                            try:
                                current_min = float(current_min)
                            except Exception:
                                current_min = score_min
                            if not np.isfinite(current_min):
                                current_min = score_min
                            current_min = min(max(current_min, score_min), score_max)
                            st.session_state["all_trials_score_min"] = current_min
                            step_val = max(0.001, abs(score_max - score_min) / 200.0)

                            min_score = st.number_input(
                                "Score min",
                                min_value=float(score_min),
                                max_value=float(score_max),
                                value=float(current_min),
                                step=float(step_val),
                                key="all_trials_score_min"
                            )
            with c_trials_3:
                rows_per_page = st.selectbox(
                    "Lignes/page",
                    options=[100, 250, 500, 1000, 2000],
                    index=1,
                    key="all_trials_page_size"
                )

            filtered_df = trials_df
            if selected_windows and "window" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["window"].isin(selected_windows)]
            if min_score is not None and "combined_score" in filtered_df.columns:
                filtered_df = filtered_df[pd.to_numeric(filtered_df["combined_score"], errors="coerce") >= float(min_score)]
            if "combined_score" in filtered_df.columns:
                filtered_df = filtered_df.sort_values("combined_score", ascending=False)

            total_filtered = len(filtered_df)
            max_page = max(1, int(np.ceil(total_filtered / rows_per_page)))
            page = st.number_input("Page", min_value=1, max_value=max_page, value=1, step=1, key="all_trials_page")
            start_idx = (int(page) - 1) * rows_per_page
            end_idx = start_idx + rows_per_page
            page_df = filtered_df.iloc[start_idx:end_idx]

            st.caption(
                f"Affichage {start_idx + 1:,}–{min(end_idx, total_filtered):,} / {total_filtered:,} "
                f"(filtré depuis {len(trials_df):,} lignes)."
            )
            st.dataframe(page_df, width="stretch")
    
    with tab7:
        st.subheader("🏆 Final Backtest Results")
        
        if 'final_portfolio' in st.session_state:
            pf = st.session_state['final_portfolio']
            params = st.session_state['final_params']
            best_score = st.session_state.get('final_params_score')
            best_window = st.session_state.get('final_params_window')
            best_is_metrics = st.session_state.get('final_params_is_metrics')
            best_oos_metrics = st.session_state.get('final_params_oos_metrics')
            best_is_score = st.session_state.get('final_params_is_score')
            best_oos_score = st.session_state.get('final_params_oos_score')
            final_params_source = str(st.session_state.get('final_params_source', 'best_window'))
            final_params_robust_summary = st.session_state.get('final_params_robust_summary') or {}
            macd_type_a = st.session_state.get('exit_macd_type_a')
            macd_type_b = st.session_state.get('exit_macd_type_b')
            exit_sar_enabled = st.session_state.get('exit_sar_enabled')
            exit_macd_enabled = st.session_state.get('exit_macd_enabled')
            
            if best_score is not None:
                st.markdown(f"**Best Optimization Score (combined_score):** `{best_score:.4f}`")
            if best_window is not None:
                st.markdown(f"**Best Window (WFO):** `{best_window}`")
            if best_is_score is not None:
                st.markdown(f"**IS Combined Score:** `{best_is_score:.4f}`")
            if best_oos_score is not None:
                st.markdown(f"**OOS Combined Score:** `{best_oos_score:.4f}`")
            st.markdown(
                f"**Parameter Source:** `{'robust_set' if final_params_source == 'robust_set' else 'best_window'}`"
            )
            if macd_type_a is not None or macd_type_b is not None:
                st.markdown(
                    f"**MACD Exit Types:** "
                    f"Type A = `{bool(macd_type_a)}`, "
                    f"Type B = `{bool(macd_type_b)}`"
                )
            if exit_macd_enabled is not None:
                st.markdown(f"**MACD Exit Enabled:** `{bool(exit_macd_enabled)}`")
            if exit_sar_enabled is not None:
                st.markdown(f"**PSAR Exit Enabled:** `{bool(exit_sar_enabled)}`")
            if final_params_source == "robust_set":
                st.markdown(f"**Used Parameters (Robust Set):** `{params}`")
                if isinstance(final_params_robust_summary, dict):
                    st.caption(
                        "Robust set: Top-N/fenêtre="
                        f"{final_params_robust_summary.get('top_n_per_window', 'n/a')}, "
                        "fenêtres utilisées="
                        f"{len(final_params_robust_summary.get('windows_used', []) or [])}."
                    )
            else:
                st.markdown(f"**Used Parameters (Best Window):** `{params}`")
            if best_is_metrics:
                st.markdown(
                    f"**IS (Window {best_is_metrics['window']}):** "
                    f"Return `{best_is_metrics['return']:.2f}%`, "
                    f"Sharpe `{best_is_metrics['sharpe']:.2f}`, "
                    f"Max DD `{best_is_metrics['max_drawdown']:.2f}%`, "
                    f"Win Rate `{best_is_metrics['win_rate']:.2f}%`, "
                    f"Trades `{best_is_metrics['n_trades']}`"
                )
            if best_oos_metrics:
                st.markdown(
                    f"**OOS (Window {best_oos_metrics['window']}):** "
                    f"Return `{best_oos_metrics['return']:.2f}%`, "
                    f"Sharpe `{best_oos_metrics['sharpe']:.2f}`, "
                    f"Max DD `{best_oos_metrics['max_drawdown']:.2f}%`, "
                    f"Win Rate `{best_oos_metrics['win_rate']:.2f}%`, "
                    f"Trades `{best_oos_metrics['n_trades']}`"
                )
            
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Return", f"{pf.total_return * 100:.2f}%")
            m2.metric("Sharpe Ratio", f"{pf.sharpe_ratio:.2f}")
            m3.metric("Max Drawdown", f"{pf.max_drawdown * 100:.2f}%")
            m4.metric("Win Rate", f"{pf.trades.win_rate * 100:.2f}%")
            
            st.markdown("#### Cumulative Returns")
            max_points = st.slider(
                "Max points to plot",
                min_value=1000,
                max_value=200000,
                value=20000,
                step=1000,
                key="max_plot_points",
                help="Réduit les séries volumineuses pour éviter les limites de taille des messages Streamlit."
            )
            # Avoid sending huge figures to the browser.
            try:
                value_series = pf.value() if callable(getattr(pf, "value", None)) else pf.value
                if value_series is None:
                    raise ValueError("Portfolio value series not available.")
                if not isinstance(value_series, pd.Series):
                    value_series = pd.Series(value_series)
                value_series = pd.to_numeric(value_series, errors="coerce").dropna()
                if value_series.empty:
                    raise ValueError("Portfolio value series is empty after cleaning.")
                value_series_plot = _downsample_series(value_series, max_points=max_points)

                fig_value = make_subplots(specs=[[{"secondary_y": True}]])
                fig_value.add_trace(
                    go.Scatter(x=value_series_plot.index, y=value_series_plot.values, mode="lines", name="Portfolio Value"),
                    secondary_y=False
                )

                # Overlay price on secondary axis if available
                price_series = None
                price_df = st.session_state.get('final_backtest_df')
                if price_df is None or price_df.empty:
                    price_df = df
                if price_df is not None and not price_df.empty:
                    if 'Close' in price_df.columns:
                        price_series = price_df['Close']
                    elif len(price_df.columns) > 0:
                        price_series = price_df.iloc[:, 0]
                if price_series is not None:
                    if not isinstance(price_series, pd.Series):
                        price_series = pd.Series(price_series)
                    price_series = pd.to_numeric(price_series, errors="coerce").dropna()

                    # Buy & Hold benchmark on the same capital base as Portfolio Value.
                    aligned_price = price_series.reindex(value_series.index).ffill().bfill().dropna()
                    common_index = value_series.index.intersection(aligned_price.index)
                    if len(common_index) > 1:
                        portfolio_common = value_series.loc[common_index]
                        price_common = aligned_price.loc[common_index]
                        initial_capital = float(portfolio_common.iloc[0])
                        initial_price = float(price_common.iloc[0])
                        if np.isfinite(initial_capital) and np.isfinite(initial_price) and initial_price != 0:
                            buy_hold_series = initial_capital * (price_common / initial_price)
                            buy_hold_series = _downsample_series(buy_hold_series, max_points=max_points)
                            fig_value.add_trace(
                                go.Scatter(
                                    x=buy_hold_series.index,
                                    y=buy_hold_series.values,
                                    mode="lines",
                                    name="Buy & Hold (same capital)",
                                    line=dict(color="#2CA02C", width=1.5, dash="dash")
                                ),
                                secondary_y=False
                            )

                    price_series = _downsample_series(price_series, max_points=max_points)
                    fig_value.add_trace(
                        go.Scatter(
                            x=price_series.index,
                            y=price_series.values,
                            mode="lines",
                            name="Asset Price",
                            line=dict(color="#FF7F0E", width=1)
                        ),
                        secondary_y=True
                    )

                fig_value.update_layout(height=400, template="plotly_dark", title="Portfolio Value vs Buy & Hold + Asset Price (Downsampled)")
                fig_value.update_yaxes(title_text="Portfolio Value", secondary_y=False)
                fig_value.update_yaxes(title_text="Asset Price", secondary_y=True)
                st.plotly_chart(fig_value, use_container_width=True)
            except Exception as e:
                st.warning(f"Plot skipped due to size or data issue: {e}")
            
            st.markdown("#### Trade Stats")
            st.dataframe(pf.trades.stats())
            trim_pct = st.slider(
                "Trim % for P&L metrics",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                key="pnl_trim_pct",
                help="Pourcentage tronqué/winsorisé sur chaque extrémité de la distribution."
            )
            pnl_metrics_df = _compute_trade_pnl_metrics(pd.DataFrame(pf.trades.records), trim=trim_pct / 100.0)
            if not pnl_metrics_df.empty:
                st.markdown("#### Average P&L per Trade (Multiple Methods)")
                st.dataframe(pnl_metrics_df, width="stretch")

            st.markdown("#### Performance by Time of Day and Day of Week")

            # Extract trade data
            trades_df = pd.DataFrame(pf.trades.records)
            had_trades = not trades_df.empty

            if had_trades:
                if len(trades_df) > 200000:
                    st.warning("Trade records are very large; displaying a sampled subset for charts.")
                    trades_df = trades_df.sample(200000, random_state=42).sort_index()
                if 'entry_ts' in trades_df.columns:
                    trades_df['entry_ts'] = pd.to_datetime(trades_df['entry_ts'])
                elif 'entry_idx' in trades_df.columns:
                    entry_index = pf.wrapper.index
                    try:
                        trades_df['entry_ts'] = pd.to_datetime(
                            entry_index.take(trades_df['entry_idx'].to_numpy())
                        )
                    except Exception:
                        st.warning("Unable to derive entry timestamps; skipping time-based charts.")
                        trades_df = pd.DataFrame()
                else:
                    st.warning("Trade records missing entry timestamps; skipping time-based charts.")
                    trades_df = pd.DataFrame()

            if not trades_df.empty:
                # Extract time components
                trades_df['day_of_week'] = trades_df['entry_ts'].dt.day_name()
                trades_df['hour_of_day'] = trades_df['entry_ts'].dt.hour

                # --- Heatmap of PnL by Day and Hour ---
                st.markdown("##### Profit & Loss Heatmap (by Entry Time)")
                
                pnl_by_time = trades_df.groupby(['day_of_week', 'hour_of_day'])['pnl'].sum().unstack(fill_value=0)
                
                # Order days of week correctly
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                pnl_by_time = pnl_by_time.reindex(day_order)

                fig_heatmap_time = px.imshow(
                    pnl_by_time,
                    labels=dict(x="Hour of Day", y="Day of Week", color="Total PnL"),
                    x=pnl_by_time.columns,
                    y=pnl_by_time.index,
                    aspect="auto",
                    color_continuous_scale="RdYlGn",
                    title="Total PnL by Day of Week and Hour of Day"
                )
                fig_heatmap_time.update_xaxes(title_text='Hour of Day')
                fig_heatmap_time.update_yaxes(title_text='Day of Week')
                st.plotly_chart(fig_heatmap_time, use_container_width=True)

                # --- Bar charts ---
                col_time1, col_time2 = st.columns(2)

                with col_time1:
                    st.markdown("##### Total PnL by Day of Week")
                    pnl_by_day = trades_df.groupby('day_of_week')['pnl'].sum().reindex(day_order)
                    fig_bar_day = px.bar(
                        pnl_by_day,
                        x=pnl_by_day.index,
                        y='pnl',
                        labels={'pnl': 'Total Profit & Loss'},
                        title="Total PnL per Day of Week"
                    )
                    st.plotly_chart(fig_bar_day, use_container_width=True)

                with col_time2:
                    st.markdown("##### Total PnL by Hour of Day")
                    pnl_by_hour = trades_df.groupby('hour_of_day')['pnl'].sum()
                    fig_bar_hour = px.bar(
                        pnl_by_hour,
                        x=pnl_by_hour.index,
                        y='pnl',
                        labels={'pnl': 'Total Profit & Loss'},
                        title="Total PnL per Hour of Day"
                    )
                    st.plotly_chart(fig_bar_hour, use_container_width=True)
            elif not had_trades:
                st.warning("No trades were made in this backtest, so no time-based analysis can be shown.")

            st.markdown("#### Rolling Performance Metrics")
            
            rolling_window = st.number_input("Rolling Window Size (periods)", min_value=1, value=30, step=1, key='rolling_window_size')

            if rolling_window:
                returns = pf.returns() if callable(getattr(pf, "returns", None)) else pf.returns
                returns = _downsample_series(returns, max_points=max(max_points, 5000))
                if pf.trades.records.size == 0:
                    st.warning("No trades were made, cannot calculate rolling performance.")
                elif len(returns) < rolling_window:
                    st.warning(f"Rolling window ({rolling_window}) is larger than the number of return periods ({len(returns)}). Please choose a smaller window.")
                else:
                    try:
                        # Calculate rolling metrics
                        if hasattr(pf, "rolling_returns"):
                            rolling_returns = pf.rolling_returns(window=rolling_window, annualize=False) * 100
                        else:
                            rolling_returns = ((1 + returns).rolling(window=rolling_window).apply(np.prod, raw=True) - 1) * 100

                        if hasattr(pf, "rolling_sharpe"):
                            rolling_sharpe = pf.rolling_sharpe(window=rolling_window)
                        else:
                            rolling_mean = returns.rolling(window=rolling_window).mean()
                            rolling_std = returns.rolling(window=rolling_window).std(ddof=0)
                            rolling_sharpe = rolling_mean.divide(rolling_std).multiply(np.sqrt(rolling_window))

                        # Drop NaNs which appear at the beginning of the series
                        rolling_returns = rolling_returns.dropna()
                        rolling_sharpe = rolling_sharpe.dropna()

                        if rolling_returns.empty or rolling_sharpe.empty:
                            st.warning("Not enough data to calculate rolling performance for the chosen window.")
                        else:
                            # Create figure with secondary y-axis
                            fig_rolling = make_subplots(specs=[[{"secondary_y": True}]])

                            # Add rolling returns trace
                            fig_rolling.add_trace(
                                go.Scatter(x=rolling_returns.index, y=rolling_returns, name="Rolling Returns (%)"),
                                secondary_y=False,
                            )

                            # Add rolling sharpe ratio trace
                            fig_rolling.add_trace(
                                go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, name="Rolling Sharpe Ratio"),
                                secondary_y=True,
                            )

                            # Add figure title
                            fig_rolling.update_layout(
                                title_text=f"{rolling_window}-Period Rolling Performance"
                            )

                            # Set y-axes titles
                            fig_rolling.update_yaxes(title_text="Rolling Returns (%)", secondary_y=False)
                            fig_rolling.update_yaxes(title_text="Rolling Sharpe Ratio", secondary_y=True)
                            st.plotly_chart(fig_rolling, use_container_width=True)

                    except Exception as e:
                        st.error(f"Could not generate rolling performance plots for window size {rolling_window}. Error: {e}")
        else:
            trades_df = st.session_state.get("final_trades_df")
            stats_df = st.session_state.get("final_trade_stats_df")
            if trades_df is not None or stats_df is not None:
                st.info("Loaded from results ZIP (portfolio object not available).")
                if stats_df is not None:
                    st.markdown("#### Trade Stats")
                    st.dataframe(stats_df)
                if trades_df is not None:
                    trim_pct = st.slider(
                        "Trim % for P&L metrics",
                        min_value=1,
                        max_value=20,
                        value=5,
                        step=1,
                        key="pnl_trim_pct_import",
                        help="Pourcentage tronqué/winsorisé sur chaque extrémité de la distribution."
                    )
                    pnl_metrics_df = _compute_trade_pnl_metrics(trades_df, trim=trim_pct / 100.0)
                    if not pnl_metrics_df.empty:
                        st.markdown("#### Average P&L per Trade (Multiple Methods)")
                        st.dataframe(pnl_metrics_df, width="stretch")
                    st.markdown("#### Trades")
                    st.dataframe(trades_df)
            else:
                st.info("Run the final backtest or load a results ZIP that includes final backtest data.")
        
        # Manual re-run using a selected WFO window
        if results.get('window_results'):
            st.markdown("#### Re-run Final Backtest by WFO Window")
            if df is None or df.empty:
                st.info("Price data (df) not available. Re-run optimization or include df.csv in the results ZIP.")
            else:
                window_options = [w.get('window_info', {}).get('window') for w in results['window_results']]
                window_options = [w for w in window_options if w is not None]
                if window_options:
                    with st.form("final_backtest_window_form"):
                        selected_window = st.selectbox("Select WFO Window", options=window_options, key='final_selected_window')
                        submitted = st.form_submit_button("Run Final Backtest (Selected Window)")

                    if submitted:
                        try:
                            selected_window_int = int(selected_window)
                        except Exception:
                            selected_window_int = selected_window
                        selected_entry = None
                        for window in results['window_results']:
                            if window.get('window_info', {}).get('window') == selected_window_int:
                                selected_entry = window
                                break

                        if selected_entry:
                            st.session_state.pop('final_portfolio', None)
                            selected_params = (selected_entry.get('best_params') or {}).copy()
                            int_params = {
                                'timeperiod', 'fenetre_lowest', 'longueur_mediane', 'Nb_bars_above', 'user_exit_sma_length',
                                'macd_fast_length', 'macd_slow_length', 'macd_signal_length'
                            }
                            for param in list(selected_params.keys()):
                                if param in int_params:
                                    try:
                                        selected_params[param] = int(round(float(selected_params[param])))
                                    except Exception:
                                        pass
                                elif isinstance(selected_params[param], float):
                                    selected_params[param] = round(selected_params[param], 2)

                            config_local = get_current_config()
                            final_start_date = st.session_state.get('final_start_date', config_local.get('start_date'))
                            final_end_date = st.session_state.get('final_end_date', config_local.get('end_date'))
                            final_file_path = st.session_state.get('final_file_path', config_local.get('file_path'))
                            with st.spinner("Loading data for final backtest range..."):
                                if config_local.get('from_file'):
                                    df_final = load_data(
                                        final_start_date,
                                        final_end_date,
                                        config_local.get('timeframe', DEFAULT_TIMEFRAME),
                                        from_file=True,
                                        file_path=final_file_path
                                    )
                                else:
                                    df_final = load_data(
                                        final_start_date,
                                        final_end_date,
                                        config_local.get('timeframe', DEFAULT_TIMEFRAME),
                                        from_file=False
                                    )
                            if df_final is None or df_final.empty:
                                st.error("No data loaded for the final backtest range.")
                            else:
                                selected_params['order_sizing_mode'] = config_local.get('order_sizing_mode', 'percent_equity')
                                selected_params['order_fixed_cash'] = float(config_local.get('order_fixed_cash', 10000.0))
                                selected_params['fees_pct'] = float(config_local.get('fees_pct', 0.0))
                                with st.spinner("Running Final Backtest on Full Dataset..."):
                                    try:
                                        strategy_adapter = resolve_strategy_adapter(
                                            strategy_mode=config_local.get("strategy_mode"),
                                            strategy_id=config_local.get("strategy_id"),
                                            config=config_local,
                                        )
                                        selected_portfolio = strategy_adapter.run_backtest(
                                            df_final,
                                            selected_params,
                                            config_local.get('timeframe', DEFAULT_TIMEFRAME),
                                            return_portfolio=True
                                        )
                                        st.session_state['final_backtest_df'] = df_final
                                        st.session_state['final_portfolio'] = selected_portfolio
                                        st.session_state['final_params'] = selected_params
                                        st.session_state['final_params_window'] = selected_window_int
                                        st.session_state['final_params_source'] = "best_window"
                                        st.session_state['final_params_robust_summary'] = results.get("robust_set_summary", {})
                                        st.success("Final Backtest Complete!")
                                        # Results panel is rendered above this form; rerun to show updated state immediately.
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error in final backtest: {e}")

elif not os.path.exists(DEFAULT_DATA_FILE):
    st.warning(f"⚠️ Default data file not found at: `{DEFAULT_DATA_FILE}`. Please configure the data source in the sidebar.")
else:
    st.info("👈 Click **Start Optimization** in the sidebar to run the backtest.")

# Keep the UI in sync with background WFO progress/completion without requiring user interaction.
if st.session_state.get('wfo_running'):
    live_thread = st.session_state.get('wfo_thread')
    if live_thread is not None and live_thread.is_alive():
        time.sleep(0.8)
    st.rerun()
