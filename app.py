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

# Plotly expects np.bool8 on older releases; alias for numpy>=2.0 compatibility.
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import from existing modules
from config import (
    DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_TIMEFRAME, DEFAULT_DATA_FILE, DEFAULT_PARAM_GRID,
    WFOSettings
)
from wfo import OptimizationInterrupted
from data_loading import load_data, get_csv_date_range
from strategy import run_backtest
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

def _build_results_payload():
    config_snapshot = get_current_config()
    results_snapshot = st.session_state.get("wfo_results")
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

    return {
        "selected_params": selected_params,
        "entry_params_enabled": enabled_entry,
        "exit_params_enabled": enabled_exit,
        "exit_params_fixed": fixed_exit,
        "exit_modules": exit_modules,
        "parameter_ranges": parameter_ranges,
        "entry_logic_summary": entry_logic,
        "exit_logic_summary": exit_logic,
    }

def _extract_final_backtest_for_expert(results, current_conf, df_source=None):
    summary = {
        "available": False,
        "source": "none",
        "run_final_backtest": False,
        "final_params": st.session_state.get("final_params"),
        "final_params_score": _to_jsonable(st.session_state.get("final_params_score")),
        "final_params_window": _to_jsonable(st.session_state.get("final_params_window")),
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

def _expert_prompt_templates_path():
    folder = os.path.join(os.path.dirname(__file__), "reports", "expert")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "prompt_templates.json")

def _load_expert_prompt_templates():
    path = _expert_prompt_templates_path()
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

def _save_expert_prompt_templates(templates):
    path = _expert_prompt_templates_path()
    payload = {
        "schema_version": "expert_prompt_templates.v1",
        "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "templates": templates if isinstance(templates, dict) else {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

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
    )

def _export_results_zip(data_snapshot_mode="manifest_only", df_max_rows=200000, full_package=False):
    if "wfo_results" not in st.session_state:
        st.error("No results available to export.")
        return None

    results = st.session_state["wfo_results"]
    df = st.session_state.get("df")

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

def _save_results_zip_to_disk(zip_buffer):
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"wfo_results_{timestamp}.zip"
    path = os.path.join(reports_dir, filename)
    with open(path, "wb") as f:
        f.write(zip_buffer.getvalue())
    return path

def _load_results_zip(zip_file):
    try:
        # Reset optional imported context to avoid stale cross-run usage.
        st.session_state.pop("expert_context_pack", None)
        with zipfile.ZipFile(zip_file) as zf:
            payload = {}
            if "results.json" in zf.namelist():
                payload = json.loads(zf.read("results.json").decode("utf-8"))
                if payload.get("wfo_results"):
                    st.session_state["wfo_results"] = payload["wfo_results"]
            if "audit_trace.json" in zf.namelist():
                traceability = json.loads(zf.read("audit_trace.json").decode("utf-8"))
                st.session_state["wfo_traceability"] = traceability
                if isinstance(traceability, dict) and isinstance(traceability.get("run"), dict):
                    st.session_state["wfo_run_metadata"] = traceability["run"]
            elif isinstance(payload, dict) and payload.get("traceability"):
                st.session_state["wfo_traceability"] = payload.get("traceability")
                if isinstance(payload["traceability"], dict) and isinstance(payload["traceability"].get("run"), dict):
                    st.session_state["wfo_run_metadata"] = payload["traceability"]["run"]
            if "df.parquet" in zf.namelist():
                try:
                    st.session_state["df"] = pd.read_parquet(io.BytesIO(zf.read("df.parquet")))
                except Exception as e:
                    st.warning(f"Impossible de lire df.parquet: {e}")
            elif "df.csv" in zf.namelist():
                df = pd.read_csv(io.BytesIO(zf.read("df.csv")))
                if "Open time" in df.columns:
                    df["Open time"] = pd.to_datetime(df["Open time"], errors="coerce")
                    df.set_index("Open time", inplace=True)
                st.session_state["df"] = df
            if "all_trials.csv" in zf.namelist():
                st.session_state["all_trials_df"] = pd.read_csv(io.BytesIO(zf.read("all_trials.csv")))
            if "window_info.csv" in zf.namelist():
                st.session_state["window_info_df"] = pd.read_csv(io.BytesIO(zf.read("window_info.csv")))

            if "expert_context_pack.json" in zf.namelist():
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

            # Optional: load backtest artifacts for display
            if "final_trades.csv" in zf.namelist():
                trades_df = pd.read_csv(io.BytesIO(zf.read("final_trades.csv")))
                st.session_state["final_trades_df"] = trades_df
            if "final_trade_stats.csv" in zf.namelist():
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
        st.session_state['start_date'] = min_date
        st.session_state['end_date'] = max_date
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

def load_best_params_into_inputs():
    final_params = st.session_state.get("final_params")
    if not final_params and 'wfo_results' in st.session_state:
        best_params, best_score, best_window, best_is, best_oos = _select_best_params_from_results(
            st.session_state['wfo_results'],
            get_current_config()
        )
        if best_params:
            st.session_state['final_params'] = best_params
            st.session_state['final_params_score'] = best_score
            st.session_state['final_params_window'] = best_window
            st.session_state['final_params_is_metrics'] = best_is
            st.session_state['final_params_oos_metrics'] = best_oos
            final_params = best_params
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
    if window_id is not None:
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
        'final_params_oos_metrics', 'final_params_is_score', 'final_params_oos_score'
    ]
    return {k: st.session_state[k] for k in keys if k in st.session_state}

def _restore_state_snapshot(snapshot):
    keys = [
        'wfo_results', 'df', 'final_backtest_df', 'final_portfolio', 'final_params',
        'final_params_score', 'final_params_window', 'final_params_is_metrics',
        'final_params_oos_metrics', 'final_params_is_score', 'final_params_oos_score'
    ]
    for key in keys:
        if key in st.session_state:
            st.session_state.pop(key)
    for key, value in snapshot.items():
        st.session_state[key] = value

def get_current_config():
    """Collects all sidebar widgets into a configuration dictionary."""
    config = {
        'start_date': start_date,
        'end_date': end_date,
        'timeframe': timeframe,
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

    # Use the single best parameter set across all windows (by combined_score).
    chosen_params, best_score, best_window, best_is_metrics, best_oos_metrics = _select_best_params_from_results(results, config)
    if not chosen_params:
        st.error("No valid parameters found for final backtest.")
        return

    # Define integer parameters that should be rounded
    int_params = {'timeperiod', 'fenetre_lowest', 'longueur_mediane', 'Nb_bars_above', 'user_exit_sma_length'}
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
            final_portfolio = run_backtest(
                df,
                chosen_params,
                config.get('timeframe', DEFAULT_TIMEFRAME),
                return_portfolio=True
            )
            st.session_state['final_portfolio'] = final_portfolio
            st.success("Final Backtest Complete!")
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
            "results_sha256": wfo_job_state.get("results_sha256")
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
    if not st.session_state.get('wfo_running'):
        if st.button(
            "🚀 Start WFO",
            type="primary",
            key="start_wfo_btn",
            width="stretch",
            help="Lance l'optimisation selon le mode choisi (WFO classique, grille précédente, NN, ou adaptatif continu)."
        ):
            if not selected_params:
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
                        "config_sha256": config_sha
                    }
                )

                def _wfo_worker():
                    results, df, elapsed = run_wfo(current_conf, control=control, job_state=job_state)
                    if control.should_stop():
                        job_state['status'] = 'stopped'
                        job_state['ended_at_utc'] = _utc_now_iso()
                    elif results is not None and df is not None:
                        job_state['status'] = 'completed'
                        run_meta_completed = {
                            "run_id": job_state.get("run_id"),
                            "status": "completed",
                            "started_at_utc": job_state.get("started_at_utc"),
                            "ended_at_utc": _utc_now_iso(),
                            "elapsed_seconds": elapsed,
                            "config_sha256": job_state.get("config_sha256")
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
            saved_path = _save_results_zip_to_disk(zip_buffer)
            st.session_state["results_zip_bytes"] = zip_buffer.getvalue()
            st.session_state["results_zip_path"] = saved_path
            st.sidebar.success(f"Saved: {saved_path}")

    if st.session_state.get("results_zip_bytes"):
        st.sidebar.download_button(
            label="⬇️ Download Results (ZIP)",
            data=st.session_state["results_zip_bytes"],
            file_name=os.path.basename(st.session_state.get("results_zip_path", "wfo_results.zip")),
            mime="application/zip",
            width="stretch",
            help="Télécharge l'archive des résultats en mémoire (JSON/CSV/trades selon disponibilité)."
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
        help="Exécute un backtest complet avec le meilleur jeu de paramètres sélectionné."
    ):
        run_final_backtest_logic()

# ==============================================================================
# RESULTS VISUALIZATION
# ==============================================================================

if 'wfo_results' in st.session_state:
    results = st.session_state['wfo_results']
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

            # Removed duplicate raw parameters table
        else:
            st.warning("No numeric parameters to visualize.")

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

        tpl_btn_1, tpl_btn_2, tpl_btn_3 = st.columns([1, 1, 1])
        with tpl_btn_1:
            if st.button("Sauvegarder template", key="expert_prompt_template_save", width="stretch"):
                target_name = str(new_tpl_name or "").strip()
                if not target_name:
                    st.warning("Renseigne un nom de template avant sauvegarde.")
                else:
                    templates[target_name] = {
                        "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
                        "mode": expert_mode,
                        "detail_level": expert_detail_level,
                        "system_prompt": str(st.session_state.get("expert_system_prompt_edit", "")),
                        "user_prompt": str(st.session_state.get("expert_user_prompt_edit", "")),
                    }
                    _save_expert_prompt_templates(templates)
                    st.success(f"Template `{target_name}` sauvegardé.")
        with tpl_btn_2:
            if st.button("Charger template", key="expert_prompt_template_load", width="stretch"):
                if selected_tpl_name == "(aucun)" or selected_tpl_name not in templates:
                    st.warning("Sélectionne un template valide à charger.")
                else:
                    selected_tpl = templates[selected_tpl_name]
                    st.session_state["expert_system_prompt_edit"] = str(selected_tpl.get("system_prompt", default_system_prompt))
                    st.session_state["expert_user_prompt_edit"] = str(selected_tpl.get("user_prompt", default_user_prompt))
                    st.success(f"Template `{selected_tpl_name}` chargé.")
                    st.rerun()
        with tpl_btn_3:
            if st.button("Supprimer template", key="expert_prompt_template_delete", width="stretch"):
                if selected_tpl_name == "(aucun)" or selected_tpl_name not in templates:
                    st.warning("Sélectionne un template valide à supprimer.")
                else:
                    templates.pop(selected_tpl_name, None)
                    _save_expert_prompt_templates(templates)
                    st.success(f"Template `{selected_tpl_name}` supprimé.")
                    st.rerun()

        c_prompt_a, c_prompt_b = st.columns([1.2, 1.2])
        with c_prompt_a:
            if st.button("Charger prompts auto", key="expert_reload_prompts"):
                st.session_state["expert_system_prompt_edit"] = default_system_prompt
                st.session_state["expert_user_prompt_edit"] = default_user_prompt
                st.rerun()
        with c_prompt_b:
            st.caption("Les prompts ci-dessous sont ceux utilisés pour l'appel API.")

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
                    user_prompt = str(st.session_state.get("expert_user_prompt_edit", "") or "").strip()
                    if not user_system_prompt or not user_prompt:
                        st.error("Les prompts Expert ne peuvent pas être vides.")
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
                                user_prompt_override=user_prompt,
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
                            "used_user_prompt": user_prompt,
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
                    if a_text:
                        st.markdown(
                            "<div class='expert-followup-answer-box'>"
                            "<strong>Réponse Expert</strong><br>"
                            f"{html.escape(a_text).replace(chr(10), '<br>')}"
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
                            int_params = {'timeperiod', 'fenetre_lowest', 'longueur_mediane', 'Nb_bars_above', 'user_exit_sma_length'}
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
                                        selected_portfolio = run_backtest(
                                            df_final,
                                            selected_params,
                                            config_local.get('timeframe', DEFAULT_TIMEFRAME),
                                            return_portfolio=True
                                        )
                                        st.session_state['final_backtest_df'] = df_final
                                        st.session_state['final_portfolio'] = selected_portfolio
                                        st.session_state['final_params'] = selected_params
                                        st.session_state['final_params_window'] = selected_window_int
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
