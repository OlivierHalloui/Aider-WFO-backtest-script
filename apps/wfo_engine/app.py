import logging
import sys
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
    WFOSettings, WFOE_UPLOAD_DIR,
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
    arrow_safe_df as _arrow_safe_df,
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
from ui.final_backtest_panel import (
    _select_best_params_from_results,
    _normalize_vote_value,
    _weighted_median,
    _build_robust_set_summary,
    _select_final_params_from_results,
    render_window_comparison_panel as _render_window_comparison_panel,
)
from ui.strategy_panel import render_strategy_panel
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
    build_order_semantics_report as _build_pine_order_semantics_report,
)
from pine_v3.llm_migration import (
    run_llm_spec_migration as _run_llm_spec_migration,
)
from pine_v3.catalog import (
    upsert_catalog_entry as _upsert_pine_catalog_entry,
    list_catalog_entries as _list_pine_catalog_entries,
    get_catalog_entry as _get_pine_catalog_entry,
    mark_catalog_entry_used as _mark_pine_catalog_entry_used,
    load_catalog_source_text as _load_pine_catalog_source_text,
)

# ---------------------------------------------------------------------------
# Logging — enable INFO output for WFO engine modules in the terminal.
# Each module gets a StreamHandler writing to stdout; propagation is disabled
# so Streamlit's root-logger capture doesn't interfere.
# ---------------------------------------------------------------------------
_wfoe_handler = logging.StreamHandler(sys.stdout)
_wfoe_handler.setLevel(logging.INFO)
_wfoe_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
))
for _mod in (
    "wfo",
    "adaptive_optimization",
    "main",
    "services.run_service",
    "ui.campaign_panel",
):
    _lg = logging.getLogger(_mod)
    _lg.setLevel(logging.INFO)
    if not _lg.handlers:
        _lg.addHandler(_wfoe_handler)
    _lg.propagate = False
del _wfoe_handler, _mod, _lg

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

# _select_best_params_from_results, _normalize_vote_value, _weighted_median,
# _build_robust_set_summary, _select_final_params_from_results
# -> imported from ui.final_backtest_panel

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
        "pine_llm_migration_report": st.session_state.get("pine_llm_migration_report"),
        "pine_catalog_last_entry": st.session_state.get("pine_catalog_last_entry"),
        "pine_artifacts_manifest": st.session_state.get("pine_artifacts_manifest"),
        "pine_beta_readiness_report": st.session_state.get("pine_beta_readiness_report"),
        "pine_execution_gate_report": st.session_state.get("pine_execution_gate_report"),
        "pine_order_semantics_report": st.session_state.get("pine_order_semantics_report"),
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

from metrics import (
    build_trials_dataframe_from_results as _build_trials_dataframe_from_results,
    build_window_info_dataframe as _build_window_info_dataframe,
    calc_pqs as _calc_pqs,
    calc_avg_pl as _calc_avg_pl,
)

# NOTE: `_to_jsonable` and `_safe_float_scalar` now come from
# `domain.serialization`.
# Expert AI helper functions -> imported from ui.expert_panel
# (see import block at top of file)
# Pine artifact functions -> imported from ui.pine_panel
# Export functions -> imported from ui.export_panel

def _export_results_zip(data_snapshot_mode="manifest_only", df_max_rows=200000, full_package=False):
    from ui.export_panel import _export_results_zip as _export_zip_impl
    return _export_zip_impl(
        data_snapshot_mode=data_snapshot_mode,
        df_max_rows=df_max_rows,
        full_package=full_package,
        build_results_payload=_build_results_payload,
        build_expert_context_pack_for_export=_build_expert_context_pack_for_export,
        compute_pine_order_semantics_report=_compute_pine_order_semantics_report,
        build_pine_beta_readiness_report=_build_pine_beta_readiness_report,
        build_pine_execution_gate_report=_build_pine_execution_gate_report,
        resolve_pine_source_for_artifacts=_resolve_pine_source_for_artifacts,
        resolve_pine_libraries_for_artifacts=_resolve_pine_libraries_for_artifacts,
        build_pine_generation_trace=_build_pine_generation_trace,
        build_pine_artifacts_manifest=_build_pine_artifacts_manifest,
        build_window_info_dataframe=_build_window_info_dataframe,
        build_trials_dataframe_from_results=_build_trials_dataframe_from_results,
        DEFAULT_STRATEGY_MODE=DEFAULT_STRATEGY_MODE,
    )

def _build_results_zip_filename(config):
    from ui.export_panel import _build_results_zip_filename as _impl
    return _impl(config, build_config_filename=_build_config_filename)

def _build_results_pdf_filename(config):
    from ui.export_panel import _build_results_pdf_filename as _impl
    return _impl(config, build_config_filename=_build_config_filename)

def _generate_wfo_pdf_report(results, config):
    from ui.export_panel import _generate_wfo_pdf_report as _impl
    from ui.expert_panel import _extract_final_backtest_for_expert
    return _impl(
        results, config,
        build_robust_set_summary=_build_robust_set_summary,
        extract_final_backtest_for_expert=_extract_final_backtest_for_expert,
        build_trials_dataframe_from_results=_build_trials_dataframe_from_results,
        build_window_info_dataframe=_build_window_info_dataframe,
    )



def _save_results_zip_to_disk(zip_buffer, config):
    from ui.export_panel import _save_results_zip_to_disk as _impl
    return _impl(zip_buffer, config, build_config_filename=_build_config_filename)

def _save_results_pdf_to_disk(pdf_buffer, config):
    from ui.export_panel import _save_results_pdf_to_disk as _impl
    return _impl(pdf_buffer, config, build_config_filename=_build_config_filename)


def _load_results_zip(zip_file):
    from ui.export_panel import _load_results_zip as _impl
    return _impl(
        zip_file,
        validate_strategy_spec_v1=_validate_strategy_spec_v1,
        apply_parity_reference_payload=_apply_parity_reference_payload,
        persist_pine_source_text=_persist_pine_source_text,
        persist_generated_strategy_text=_persist_generated_strategy_text,
        persist_pine_library_text=_persist_pine_library_text,
    )

@st.cache_data(show_spinner=False)
def _cached_csv_date_range(file_path: str, mtime: float):
    """Cached wrapper — result is reused as long as file path and mtime are unchanged."""
    return get_csv_date_range(file_path)


@st.cache_data(show_spinner=False)
def _cached_load_data_for_final(file_path: str, mtime: float, start_date, end_date, timeframe):
    """Cached loader for the final backtest — keyed by (file_path, mtime) so stale data is never used."""
    return load_data(start_date, end_date, timeframe, from_file=True, file_path=file_path)


def sync_dates_from_file(force=False):
    file_path = st.session_state.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return

    if not force and st.session_state.get('last_data_file_path') == file_path:
        return

    try:
        mtime = os.path.getmtime(file_path)
    except OSError:
        mtime = 0.0
    min_date, max_date = _cached_csv_date_range(file_path, mtime)
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
        target_dir = WFOE_UPLOAD_DIR
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


# Pine panel functions -> imported from ui.pine_panel
from ui.pine_panel import (
    _repo_root_dir,
    _read_text_file_with_fallback,
    _pine_imports_dir,
    _pine_generated_dir,
    _persist_pine_source_text,
    _persist_generated_strategy_text,
    _persist_uploaded_pine_file,
    _persist_pine_library_text,
    _persist_uploaded_pine_library_files,
    _normalize_token,
    _parse_pine_import_lines,
    _extract_library_decl_name,
    _extract_pine_library_functions,
    _extract_alias_function_calls,
    _analyze_provided_libraries,
    _validate_python_mapping_target,
    _resolve_pine_imports,
    _precheck_pine_script_text,
    _precheck_pine_script_file,
    _build_pine_compatibility_report,
    _build_pine_beta_readiness_report,
    _compute_pine_order_semantics_report,
    _get_pine_parity_thresholds_from_state,
    _get_pine_parity_detail_thresholds_from_state,
    _to_iso_utc,
    _extract_current_events_and_trades_for_parity,
    _extract_current_metrics_for_parity,
    _parse_parity_reference_payload_from_text,
    _apply_parity_reference_payload,
    _build_pine_artifacts_summary_for_expert,
    _resolve_pine_source_for_artifacts,
    _resolve_pine_libraries_for_artifacts,
    _build_pine_generation_trace,
    _build_pine_artifacts_manifest,
)

# Expert panel functions -> imported from ui.expert_panel
from ui.expert_panel import (
    EXPERT_MODEL_CATALOG,
    _get_expert_model_entries,
    _get_expert_provider_defaults,
    _build_expert_context_pack_for_export,
    _list_saved_expert_reports,
    _format_saved_expert_report_label,
    _load_saved_expert_report,
    _build_expert_input_data,
    _build_effective_expert_user_prompt,
    _load_expert_prompt_templates,
    _save_expert_prompt_templates,
    _resolve_template_prompts,
    _build_followup_context_pack,
    _format_followup_answer_for_display,
)

def load_best_params_into_inputs(window_id=None):
    from ui.final_backtest_panel import load_best_params_into_inputs as _load_best_params
    return _load_best_params(get_current_config=get_current_config, window_id=window_id)



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

    # --- App mode switcher ---
    st.radio(
        "Mode",
        options=["Single Run", "Campaign"],
        horizontal=True,
        key="app_mode",
        help="Single Run: optimize one config. Campaign: run multiple configs sequentially and compare results.",
    )

    has_final_params = 'final_params' in st.session_state or 'wfo_results' in st.session_state
    # Build window list for the selector (only when WFO results exist)
    _wfo_windows = []
    if 'wfo_results' in st.session_state:
        _wr = st.session_state['wfo_results']
        for _w in (_wr.get('window_results') or []):
            _wid = (_w.get('window_info') or {}).get('window')
            if _wid is not None:
                _wfo_windows.append(_wid)
    # Selector: "Best window (auto)" + one entry per window
    _window_options = ["Best window (auto)"] + [f"Fenêtre {w}" for w in _wfo_windows]
    _selected_label = st.sidebar.selectbox(
        "Source des paramètres",
        options=_window_options,
        index=0,
        key="load_params_window_selector",
        disabled=not has_final_params,
        help="Choisir la fenêtre WFO dont les best params seront chargés dans les inputs.",
    )
    # Resolve selected window_id (None = best window auto)
    _selected_window_id = None
    if _selected_label != "Best window (auto)" and _wfo_windows:
        _idx = _window_options.index(_selected_label) - 1  # offset for the "auto" entry
        if 0 <= _idx < len(_wfo_windows):
            _selected_window_id = _wfo_windows[_idx]

    st.sidebar.button(
        "📥 Load Params into Inputs",
        width="stretch",
        on_click=load_best_params_into_inputs,
        kwargs={"window_id": _selected_window_id},
        disabled=not has_final_params,
        help="Charge les paramètres de la fenêtre sélectionnée dans les champs Min/Max/Step."
    )
    if not has_final_params:
        st.sidebar.info("Lancez le WFO pour activer le chargement des paramètres.")

    # --- Stagewise report import ---
    st.sidebar.markdown("---")
    _stagewise_upload = st.sidebar.file_uploader(
        "📂 Importer rapport stagewise (JSON)",
        type=["json"],
        key="stagewise_report_uploader",
        help="Importe les best params d'une campagne stagewise_optimizer.py "
             "(stagewise_final_report.json) et les charge dans les inputs.",
    )
    if _stagewise_upload is not None:
        from ui.final_backtest_panel import load_stagewise_params_from_json
        _sw_file_id = getattr(_stagewise_upload, "file_id",
                              _stagewise_upload.name + str(_stagewise_upload.size))
        if st.session_state.get("_stagewise_file_id") != _sw_file_id:
            st.session_state["_stagewise_file_id"] = _sw_file_id
            try:
                _sw_report = json.load(_stagewise_upload)
                _ok = load_stagewise_params_from_json(_sw_report)
                if _ok:
                    _n_stages = _sw_report.get("n_stages", "?")
                    _direction = _sw_report.get("direction", "")
                    st.sidebar.success(
                        f"Params stagewise chargés ({_n_stages} runs, {_direction})."
                    )
                    st.rerun()
            except Exception as _sw_err:
                st.sidebar.error(f"Erreur import stagewise : {_sw_err}")

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
                    'pine_enforce_external_call_contract': 'pine_enforce_external_call_contract',
                    'pine_enforce_order_semantics': 'pine_enforce_order_semantics',
                    'pine_spec_parser_backend': 'pine_spec_parser_backend',
                    'pine_llm_provider': 'pine_llm_provider',
                    'pine_llm_model': 'pine_llm_model',
                    'pine_llm_base_url': 'pine_llm_base_url',
                    'pine_llm_temperature': 'pine_llm_temperature',
                    'pine_llm_max_tokens': 'pine_llm_max_tokens',
                    'pine_llm_timeout_s': 'pine_llm_timeout_s',
                    'pine_llm_retries': 'pine_llm_retries',
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
                    'exit_cross_sar_sma_enabled': 'exit_cross_sar_sma_enabled',
                    'exit_retour_bb_enabled': 'exit_retour_bb_enabled',
                    'exit_regline_enabled': 'exit_regline_enabled',
                    'exit_volat_down_enabled': 'exit_volat_down_enabled',
                    'use_roc_filter': 'use_roc_filter',
                    'use_t2_signal': 'use_t2_signal',
                    'use_divergence_bb': 'use_divergence_bb',
                    'macd_ma_type': 'macd_ma_type',
                    'strategy_direction': 'strategy_direction',
                    'pqs_n_ref': 'pqs_n_ref',
                    'selection_method': 'selection_method',
                    'svi_top_k': 'svi_top_k',
                    'svi_is2_fraction': 'svi_is2_fraction',
                    'cross_window_method': 'cross_window_method',
                    'optimize_exit_sar_enabled': 'optimize_exit_sar_enabled',
                    'optimize_exit_macd_enabled': 'optimize_exit_macd_enabled',
                    'optimize_exit_macd_type_a': 'optimize_exit_macd_type_a',
                    'optimize_exit_macd_type_b': 'optimize_exit_macd_type_b',
                    'optimize_use_roc_filter': 'optimize_use_roc_filter',
                    'optimize_use_t2_signal': 'optimize_use_t2_signal',
                    'optimize_use_divergence_bb': 'optimize_use_divergence_bb',
                    'optimize_exit_cross_sar_sma_enabled': 'optimize_exit_cross_sar_sma_enabled',
                    'optimize_exit_retour_bb_enabled': 'optimize_exit_retour_bb_enabled',
                    'optimize_exit_regline_enabled': 'optimize_exit_regline_enabled',
                    'optimize_exit_volat_down_enabled': 'optimize_exit_volat_down_enabled',
                    'nb_bars_under_bbw_mini': 'nb_bars_under_bbw_mini',
                    'nb_bars_entre_bb': 'nb_bars_entre_bb',
                    'depassement_sma_roc': 'depassement_sma_roc',
                    'roc_max_t1': 'roc_max_t1',
                    'nb_bars_left_pivot': 'nb_bars_left_pivot',
                    'nb_bars_right_pivot': 'nb_bars_right_pivot',
                    'nombre_periodes_reglin': 'nombre_periodes_reglin',
                    'i_bars_back': 'i_bars_back',
                    'seuil_overbought_bb': 'seuil_overbought_bb',
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
                    restored_fp = st.session_state.get('file_path', '')
                    if isinstance(restored_fp, str) and restored_fp and not os.path.exists(restored_fp):
                        # File no longer accessible (old /tmp upload, deleted, or moved).
                        # Reset to avoid a stale path being silently used.
                        _missing_name = os.path.basename(restored_fp)
                        st.session_state['file_path'] = ''
                        st.session_state.pop('uploaded_data_file_id', None)
                        st.session_state.pop('uploaded_data_file_path', None)
                        st.warning(
                            f"Le fichier de données `{_missing_name}` est introuvable. "
                            "Veuillez re-uploader le CSV ou corriger le chemin."
                        )
                    else:
                        sync_dates_from_file(force=True)
                        # sync_dates_from_file overwrites pending_start/end_date with the
                        # file's full date range.  Re-apply the config's specific dates so
                        # that the widgets show the saved values, not the file extremes.
                        if 'start_date' in loaded_config:
                            st.session_state['pending_start_date'] = str(loaded_config['start_date'])
                        if 'end_date' in loaded_config:
                            st.session_state['pending_end_date'] = str(loaded_config['end_date'])

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
            if "pine_enforce_external_call_contract" not in st.session_state:
                st.session_state["pine_enforce_external_call_contract"] = True
            st.checkbox(
                "Enforcer contrat des appels externes (strict)",
                key="pine_enforce_external_call_contract",
                help=(
                    "Si activé (recommandé), les runs Pine en mode strict sont bloqués "
                    "quand un appel `Alias.fonction(...)` n'est pas résolu/callable dans le mapping Python."
                ),
            )
            if "pine_enforce_order_semantics" not in st.session_state:
                st.session_state["pine_enforce_order_semantics"] = True
            st.checkbox(
                "Enforcer sémantique des ordres (strict)",
                key="pine_enforce_order_semantics",
                help=(
                    "Si activé (recommandé), les runs Pine en mode strict sont bloqués "
                    "quand des ordres limit/stop/trailing sont détectés dans la logique Pine."
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

        pending_pine_file_path = st.session_state.pop("pending_pine_file_path", None)
        if isinstance(pending_pine_file_path, str) and pending_pine_file_path.strip():
            st.session_state["pine_file_path"] = pending_pine_file_path.strip()

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
        if strategy_mode == "pine_imported":
            with st.expander("Catalogue stratégies Pine (P2.2)", expanded=False):
                st.caption(
                    "Catalogue local des stratégies Pine importées (versionnées par SHA1 source). "
                    "Permet recherche et rechargement rapide."
                )
                catalog_query = st.text_input(
                    "Recherche catalogue",
                    key="pine_catalog_query",
                    placeholder="strategy_id, nom, sha1...",
                )
                catalog_rows = _list_pine_catalog_entries(query=catalog_query, limit=300)
                st.caption(f"Entrées trouvées: {len(catalog_rows)}")
                if catalog_rows:
                    table = pd.DataFrame(
                        [
                            {
                                "entry_id": row.get("entry_id"),
                                "strategy_id": row.get("strategy_id"),
                                "strategy_name": row.get("strategy_name"),
                                "source_name": row.get("source_name"),
                                "sha1": str(row.get("source_sha1") or "")[:12],
                                "spec_valid": row.get("spec_valid"),
                                "compat_score": row.get("compatibility_score"),
                                "parser": row.get("parser_backend_used"),
                                "llm_used": row.get("llm_used"),
                                "updated_at_utc": row.get("updated_at_utc"),
                            }
                            for row in catalog_rows
                            if isinstance(row, dict)
                        ]
                    )
                    st.dataframe(table, width="stretch")

                    ids = [str(row.get("entry_id")) for row in catalog_rows if isinstance(row, dict)]
                    selected_id = st.selectbox(
                        "Entrée catalogue",
                        options=ids,
                        key="pine_catalog_selected_entry_id",
                    )
                    if st.button(
                        "Charger la stratégie depuis le catalogue",
                        key="pine_catalog_load_btn",
                        width="stretch",
                    ):
                        row = _get_pine_catalog_entry(selected_id)
                        if not isinstance(row, dict):
                            st.error("Entrée catalogue introuvable.")
                        else:
                            loaded_text = _load_pine_catalog_source_text(row)
                            loaded_path = str(row.get("source_path") or "").strip()
                            if isinstance(loaded_text, str) and loaded_text.strip():
                                restored_path = _persist_pine_source_text(
                                    loaded_text,
                                    source_name=str(row.get("source_name") or "catalog_strategy.pine.txt"),
                                )
                                if isinstance(restored_path, str) and restored_path.strip():
                                    st.session_state["pending_pine_file_path"] = restored_path
                                    st.session_state["pine_source_name"] = str(
                                        row.get("source_name") or os.path.basename(restored_path)
                                    )
                                    if str(row.get("strategy_id") or "").strip():
                                        st.session_state["pending_strategy_id"] = str(
                                            row.get("strategy_id") or ""
                                        ).strip()
                                    _mark_pine_catalog_entry_used(selected_id)
                                    st.success("Stratégie chargée depuis le catalogue.")
                                    st.rerun()
                            elif loaded_path and os.path.exists(loaded_path):
                                st.session_state["pending_pine_file_path"] = loaded_path
                                st.session_state["pine_source_name"] = str(
                                    row.get("source_name") or os.path.basename(loaded_path)
                                )
                                if str(row.get("strategy_id") or "").strip():
                                    st.session_state["pending_strategy_id"] = str(
                                        row.get("strategy_id") or ""
                                    ).strip()
                                _mark_pine_catalog_entry_used(selected_id)
                                st.success("Chemin source réutilisé depuis le catalogue.")
                                st.rerun()
                            else:
                                st.error(
                                    "Source introuvable pour cette entrée (ni snapshot ni chemin valide)."
                                )
                else:
                    st.info("Catalogue vide pour l'instant. Lance une pré-analyse valide pour alimenter le registre.")
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
                st.session_state.pop("pine_llm_migration_report", None)
                st.session_state.pop("pine_catalog_last_entry", None)
                st.session_state.pop("pine_llm_override_spec", None)
                st.session_state.pop("pine_llm_override_source_sha1", None)
                st.session_state.pop("pine_artifacts_manifest", None)
                st.session_state.pop("pine_beta_readiness_report", None)
                st.session_state.pop("pine_execution_gate_report", None)
                st.session_state.pop("pine_order_semantics_report", None)
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
                        llm_override_spec = st.session_state.get("pine_llm_override_spec")
                        if isinstance(llm_override_spec, dict):
                            llm_override_sha = str(st.session_state.get("pine_llm_override_source_sha1") or "").strip()
                            current_sha = str(
                                ((strategy_spec.get("source") or {}).get("source_sha1") or "")
                            ).strip()
                            llm_override_validation = _validate_strategy_spec_v1(llm_override_spec)
                            if (
                                llm_override_sha
                                and current_sha
                                and llm_override_sha == current_sha
                                and bool(llm_override_validation.get("valid", False))
                            ):
                                strategy_spec = llm_override_spec
                                strategy_spec_validation = llm_override_validation
                            elif llm_override_sha and current_sha and llm_override_sha != current_sha:
                                st.session_state.pop("pine_llm_override_spec", None)
                                st.session_state.pop("pine_llm_override_source_sha1", None)
                        st.session_state["pine_strategy_spec"] = _sanitize_for_json(strategy_spec)
                        st.session_state["pine_spec_sha256"] = _sha256_json(strategy_spec) if strategy_spec else None
                        st.session_state["pine_strategy_spec_validation"] = _sanitize_for_json(
                            strategy_spec_validation
                        )
                        st.session_state["pine_source_encoding"] = source_encoding
                        st.session_state["pine_source_text"] = pine_text
                        if bool(strategy_spec_validation.get("valid", False)):
                            try:
                                spec_source = strategy_spec.get("source") if isinstance(strategy_spec, dict) else {}
                                source_sha1 = (
                                    str((spec_source or {}).get("source_sha1") or "").strip()
                                    if isinstance(spec_source, dict)
                                    else ""
                                )
                                last_catalog_key = str(st.session_state.get("pine_catalog_last_upsert_key") or "")
                                current_catalog_key = (
                                    f"{str(st.session_state.get('strategy_id') or '')}:{source_sha1}"
                                    if source_sha1
                                    else ""
                                )
                                if current_catalog_key and current_catalog_key != last_catalog_key:
                                    catalog_entry = _upsert_pine_catalog_entry(
                                        strategy_spec=strategy_spec,
                                        strategy_spec_validation=strategy_spec_validation,
                                        precheck_report=pine_precheck_report,
                                        compatibility_report=pine_compatibility_report,
                                        generation_trace=st.session_state.get("pine_generation_trace"),
                                        source_text=pine_text,
                                        source_name=st.session_state.get("pine_source_name", ""),
                                        source_path=str(pine_file_path or ""),
                                        library_files=st.session_state.get("pine_library_files", []),
                                    )
                                    st.session_state["pine_catalog_last_upsert_key"] = current_catalog_key
                                    st.session_state["pine_catalog_last_entry"] = _sanitize_for_json(catalog_entry)
                            except Exception as e:
                                st.caption(f"Catalogue Pine: mise à jour ignorée ({e})")
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
                        st.session_state.pop("pine_order_semantics_report", None)
                        st.session_state.pop("pine_parity_report", None)
                        st.session_state.pop("pine_request_security_diagnostics", None)
                        st.session_state.pop("pine_mtf_parity_proof_report", None)
                        st.session_state.pop("pine_parity_reference_payload", None)
                        st.session_state.pop("pine_parity_reference_validation", None)
                        st.session_state.pop("pine_parity_reference_metrics", None)
                        st.session_state.pop("pine_parity_reference_text", None)
                        st.session_state.pop("pine_llm_migration_report", None)
                        st.session_state.pop("pine_llm_override_spec", None)
                        st.session_state.pop("pine_llm_override_source_sha1", None)
                elif status != "valid":
                    st.session_state.pop("pine_strategy_spec", None)
                    st.session_state.pop("pine_strategy_spec_validation", None)
                    st.session_state.pop("pine_codegen_report", None)
                    st.session_state.pop("pine_generated_module_path", None)
                    st.session_state.pop("pine_execution_gate_report", None)
                    st.session_state.pop("pine_order_semantics_report", None)
                    st.session_state.pop("pine_parity_report", None)
                    st.session_state.pop("pine_request_security_diagnostics", None)
                    st.session_state.pop("pine_mtf_parity_proof_report", None)
                    st.session_state.pop("pine_parity_reference_payload", None)
                    st.session_state.pop("pine_parity_reference_validation", None)
                    st.session_state.pop("pine_parity_reference_metrics", None)
                    st.session_state.pop("pine_parity_reference_text", None)
                    st.session_state.pop("pine_llm_migration_report", None)
                    st.session_state.pop("pine_llm_override_spec", None)
                    st.session_state.pop("pine_llm_override_source_sha1", None)
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

                order_semantics_report = _compute_pine_order_semantics_report(
                    strategy_spec=strategy_spec if isinstance(strategy_spec, dict) else {},
                    compat_mode=st.session_state.get("pine_compat_mode", "strict"),
                    enforce_order_semantics=bool(st.session_state.get("pine_enforce_order_semantics", True)),
                )
                if isinstance(order_semantics_report, dict) and order_semantics_report:
                    st.session_state["pine_order_semantics_report"] = order_semantics_report
                    c_ord_1, c_ord_2, c_ord_3 = st.columns(3)
                    c_ord_1.metric("Order Semantics", str(order_semantics_report.get("status", "n/a")))
                    c_ord_2.metric(
                        "Price Controls",
                        str(int(order_semantics_report.get("rules_with_price_controls", 0) or 0)),
                    )
                    c_ord_3.metric(
                        "Qty Controls",
                        str(int(order_semantics_report.get("rules_with_qty_controls", 0) or 0)),
                    )
                    if bool(st.session_state.get("pine_enforce_order_semantics", True)):
                        if bool(order_semantics_report.get("passed", False)):
                            st.success("Sémantique d'ordres compatible runtime (strict).")
                        else:
                            st.error("Sémantique d'ordres non compatible runtime: exécution strict bloquée.")
                    else:
                        if bool(order_semantics_report.get("passed", False)):
                            st.caption("Sémantique d'ordres validée (verrou strict désactivé).")
                        else:
                            st.warning("Sémantique d'ordres en échec, mais verrou strict désactivé.")
                    with st.expander("Rapport sémantique d'ordres Pine", expanded=False):
                        st.json(order_semantics_report)
                else:
                    st.session_state.pop("pine_order_semantics_report", None)

                with st.expander("Assistant LLM migration Pine -> spec (P2.1)", expanded=False):
                    if "pine_llm_provider" not in st.session_state:
                        st.session_state["pine_llm_provider"] = "openai"
                    if "pine_llm_model" not in st.session_state:
                        st.session_state["pine_llm_model"] = "gpt-5-mini"
                    if "pine_llm_base_url" not in st.session_state:
                        st.session_state["pine_llm_base_url"] = ""
                    if "pine_llm_temperature" not in st.session_state:
                        st.session_state["pine_llm_temperature"] = 0.2
                    if "pine_llm_max_tokens" not in st.session_state:
                        st.session_state["pine_llm_max_tokens"] = 4000
                    if "pine_llm_timeout_s" not in st.session_state:
                        st.session_state["pine_llm_timeout_s"] = 120
                    if "pine_llm_retries" not in st.session_state:
                        st.session_state["pine_llm_retries"] = 1

                    c_llm_1, c_llm_2, c_llm_3 = st.columns(3)
                    c_llm_1.selectbox(
                        "Provider",
                        options=["openai", "grok", "gemini"],
                        key="pine_llm_provider",
                        help="Fournisseur LLM utilisé pour proposer un brouillon de strategy_spec.",
                    )
                    c_llm_2.text_input(
                        "Model",
                        key="pine_llm_model",
                        help="Ex: gpt-5-mini, gpt-5.2, grok-4-1-fast-reasoning, gemini-3-flash-preview",
                    )
                    c_llm_3.text_input(
                        "Base URL (optionnel)",
                        key="pine_llm_base_url",
                        help="Laisser vide pour l'endpoint par défaut du provider.",
                    )

                    c_llm_4, c_llm_5, c_llm_6, c_llm_7 = st.columns(4)
                    c_llm_4.slider(
                        "Temperature",
                        min_value=0.0,
                        max_value=1.0,
                        step=0.1,
                        key="pine_llm_temperature",
                    )
                    c_llm_5.number_input("Max tokens", min_value=256, max_value=12000, step=256, key="pine_llm_max_tokens")
                    c_llm_6.number_input("Timeout (s)", min_value=10, max_value=600, step=10, key="pine_llm_timeout_s")
                    c_llm_7.number_input("Retries", min_value=0, max_value=5, step=1, key="pine_llm_retries")

                    st.text_input(
                        "API Key",
                        type="password",
                        key="pine_llm_api_key",
                        help="Clé non exportée dans la config/resultats. Utilisée uniquement pour l'appel en cours.",
                    )

                    llm_generate = st.button(
                        "Générer un brouillon LLM (revalidé)",
                        key="pine_llm_generate_btn",
                        width="stretch",
                        help=(
                            "Le brouillon LLM n'est jamais accepté sans validation stricte "
                            "`strategy_spec.v1`. En cas d'échec, fallback déterministe."
                        ),
                    )
                    if llm_generate:
                        pine_text_for_llm = st.session_state.get("pine_source_text")
                        if not isinstance(pine_text_for_llm, str) or not pine_text_for_llm.strip():
                            pine_source_path = str(st.session_state.get("pine_file_path") or "").strip()
                            if pine_source_path and os.path.exists(pine_source_path):
                                try:
                                    pine_text_for_llm, _ = _read_text_file_with_fallback(pine_source_path)
                                except Exception as e:
                                    pine_text_for_llm = ""
                                    st.error(f"Lecture du script Pine impossible: {e}")
                        if not isinstance(pine_text_for_llm, str) or not pine_text_for_llm.strip():
                            st.error("Source Pine indisponible: importe/charge d'abord une stratégie valide.")
                        else:
                            llm_cfg = LLMConfig(
                                provider=str(st.session_state.get("pine_llm_provider", "openai")),
                                model=str(st.session_state.get("pine_llm_model", "gpt-5-mini")),
                                api_key=str(st.session_state.get("pine_llm_api_key", "")),
                                base_url=str(st.session_state.get("pine_llm_base_url", "")),
                                temperature=float(st.session_state.get("pine_llm_temperature", 0.2)),
                                max_tokens=int(st.session_state.get("pine_llm_max_tokens", 4000)),
                                timeout_s=int(st.session_state.get("pine_llm_timeout_s", 120)),
                                retries=int(st.session_state.get("pine_llm_retries", 1)),
                            )
                            llm_report = _run_llm_spec_migration(
                                pine_text=pine_text_for_llm,
                                source_name=st.session_state.get("pine_source_name", ""),
                                strategy_id=st.session_state.get("strategy_id", ""),
                                precheck_report=st.session_state.get("pine_precheck_report"),
                                compatibility_report=st.session_state.get("pine_compatibility_report"),
                                parser_backend=st.session_state.get("pine_spec_parser_backend", "auto"),
                                llm_config=llm_cfg,
                            )
                            llm_report = _sanitize_for_json(llm_report)
                            st.session_state["pine_llm_migration_report"] = llm_report

                            trace = llm_report.get("trace") if isinstance(llm_report, dict) else {}
                            if isinstance(trace, dict):
                                current_trace = st.session_state.get("pine_generation_trace")
                                current_trace = current_trace if isinstance(current_trace, dict) else {}
                                current_trace["llm_used"] = True
                                current_trace["llm_migration"] = trace
                                current_trace["generated_at_utc"] = _utc_now_iso()
                                st.session_state["pine_generation_trace"] = _sanitize_for_json(current_trace)

                    llm_report_view = st.session_state.get("pine_llm_migration_report")
                    if isinstance(llm_report_view, dict):
                        c_r1, c_r2, c_r3 = st.columns(3)
                        c_r1.metric("LLM Status", str(llm_report_view.get("status", "n/a")))
                        cand_valid = (
                            ((llm_report_view.get("candidate_validation") or {}).get("valid"))
                            if isinstance(llm_report_view.get("candidate_validation"), dict)
                            else False
                        )
                        c_r2.metric("Candidate Valid", "yes" if bool(cand_valid) else "no")
                        acc_valid = (
                            ((llm_report_view.get("accepted_validation") or {}).get("valid"))
                            if isinstance(llm_report_view.get("accepted_validation"), dict)
                            else False
                        )
                        c_r3.metric("Accepted Valid", "yes" if bool(acc_valid) else "no")

                        for w in llm_report_view.get("warnings", []) or []:
                            st.caption(f"Avertissement: {w}")
                        for e in llm_report_view.get("errors", []) or []:
                            st.caption(f"Erreur: {e}")

                        if cand_valid:
                            if st.button(
                                "Appliquer le draft LLM validé",
                                key="pine_llm_apply_valid_spec_btn",
                                width="stretch",
                            ):
                                accepted_spec = llm_report_view.get("accepted_spec")
                                accepted_validation = llm_report_view.get("accepted_validation")
                                if isinstance(accepted_spec, dict) and isinstance(accepted_validation, dict) and bool(
                                    accepted_validation.get("valid", False)
                                ):
                                    st.session_state["pine_strategy_spec"] = _sanitize_for_json(accepted_spec)
                                    st.session_state["pine_spec_sha256"] = _sha256_json(accepted_spec) if accepted_spec else None
                                    st.session_state["pine_strategy_spec_validation"] = _sanitize_for_json(
                                        accepted_validation
                                    )
                                    st.session_state["pine_llm_override_spec"] = _sanitize_for_json(accepted_spec)
                                    st.session_state["pine_llm_override_source_sha1"] = (
                                        ((accepted_spec.get("source") or {}).get("source_sha1"))
                                        if isinstance(accepted_spec.get("source"), dict)
                                        else None
                                    )
                                    st.success("Draft LLM appliqué: strategy_spec.v1 mis à jour.")
                                else:
                                    st.error("Impossible d'appliquer: draft LLM non valide.")

                        with st.expander("Trace LLM migration", expanded=False):
                            if isinstance(llm_report_view.get("trace"), dict):
                                st.json(llm_report_view.get("trace"))

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
                    st.session_state.pop("pine_order_semantics_report", None)

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

                order_semantics_for_gate = _compute_pine_order_semantics_report(
                    strategy_spec=st.session_state.get("pine_strategy_spec"),
                    compat_mode=st.session_state.get("pine_compat_mode", "strict"),
                    enforce_order_semantics=bool(st.session_state.get("pine_enforce_order_semantics", True)),
                )
                if isinstance(order_semantics_for_gate, dict) and order_semantics_for_gate:
                    st.session_state["pine_order_semantics_report"] = order_semantics_for_gate
                else:
                    st.session_state.pop("pine_order_semantics_report", None)
                execution_gate_report = _build_pine_execution_gate_report(
                    strategy_mode=st.session_state.get("strategy_mode", DEFAULT_STRATEGY_MODE),
                    beta_readiness_report=st.session_state.get("pine_beta_readiness_report"),
                    parity_reference_payload=st.session_state.get("pine_parity_reference_payload"),
                    parity_reference_validation=st.session_state.get("pine_parity_reference_validation"),
                    parity_report=st.session_state.get("pine_parity_report"),
                    enforce_parity_when_reference=bool(st.session_state.get("pine_enforce_parity_gate", True)),
                    order_semantics_report=order_semantics_for_gate,
                    enforce_order_semantics=bool(st.session_state.get("pine_enforce_order_semantics", True)),
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

        st.markdown("**Sélection intra-fenêtre (Niveau 1)**")
        _sel_methods = ["snv", "svi", "raw_max"]
        _sel_labels  = {
            "snv":     "SNV — Stabilité Voisinage (KDTree)",
            "svi":     "SVI — Validation Interne IS₂",
            "raw_max": "Score brut max",
        }
        selection_method = st.selectbox(
            "Méthode sélection intra-fenêtre",
            options=_sel_methods,
            format_func=lambda x: _sel_labels[x],
            index=0,
            key='selection_method',
            help="SNV : plateau robuste par voisinage normalisé. SVI : meilleur sur IS₂. raw_max : pic absolu (risque overfit)."
        )
        if selection_method == "svi":
            _col_svi1, _col_svi2 = st.columns(2)
            _col_svi1.number_input(
                "SVI Top-K candidats", min_value=5, value=20, step=5,
                key='svi_top_k',
                help="Nombre de candidats IS-top évalués sur IS₂."
            )
            _col_svi2.slider(
                "SVI fraction IS₂", 0.10, 0.50, 0.30, 0.05,
                key='svi_is2_fraction',
                help="Part de IS réservée comme sous-période de validation."
            )

        st.markdown("**Sélection cross-fenêtres (Niveau 2)**")
        _cw_methods = ["best_is_oos", "best_oos", "robust_set", "weighted_oos"]
        _cw_labels  = {
            "best_is_oos":  "Best IS+OOS — fenêtre avec meilleure moyenne IS/OOS",
            "best_oos":     "Best OOS seul — fenêtre avec meilleur score OOS",
            "robust_set":   "Robust Set — vote pondéré top-N cross-fenêtres",
            "weighted_oos": "Médiane pondérée OOS — agrégation de toutes les fenêtres",
        }
        cross_window_method = st.selectbox(
            "Méthode sélection cross-fenêtres",
            options=_cw_methods,
            format_func=lambda x: _cw_labels[x],
            index=0,
            key='cross_window_method',
            help="Détermine quels paramètres sont utilisés pour le Final Backtest."
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
                value=5000,
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
        metric_options = ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate', 'avg_gain_per_trade', 'avg_loss_per_trade', 'avg_pl_per_trade', 'pqs']
        
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
    """Collects all widgets into a configuration dictionary.

    Strategy parameters (selected_params, config_params, exit flags) are read from
    st.session_state — they are now managed by render_strategy_panel() in the main panel.
    All other sidebar parameters (dates, timeframe, WFO engine settings) are still
    read from module-level variables set during the with st.sidebar: block.
    """
    # ---------------------------------------------------------------------------
    # Rebuild selected_params + config_params from session_state widget keys.
    # strategy_panel.py renders check_{key}, min_{key}, max_{key}, step_{key}
    # for every param in DEFAULT_PARAM_GRID.
    # ---------------------------------------------------------------------------
    selected_params = []
    config_params = {}
    # Dead-dimension guard: params whose controlling exit is disabled are never
    # optimized — exclude them regardless of their check_{param} state.
    _sar_on  = bool(st.session_state.get("exit_sar_enabled",  True))
    _macd_on = bool(st.session_state.get("exit_macd_enabled", True))
    _SAR_PARAMS  = {'sar_start', 'sar_increment', 'sar_maximum'}
    _MACD_PARAMS = {'macd_fast_length', 'macd_slow_length', 'macd_signal_length'}
    for param in DEFAULT_PARAM_GRID:
        if param in _SAR_PARAMS  and not _sar_on:
            continue
        if param in _MACD_PARAMS and not _macd_on:
            continue
        if st.session_state.get(f"check_{param}", True):
            selected_params.append(param)
            d_min, d_max, d_step = DEFAULT_PARAM_GRID[param]
            config_params[f'{param}_min'] = float(st.session_state.get(f"min_{param}", d_min))
            config_params[f'{param}_max'] = float(st.session_state.get(f"max_{param}", d_max))
            config_params[f'{param}_step'] = float(st.session_state.get(f"step_{param}", d_step))

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
    pine_order_semantics = st.session_state.get("pine_order_semantics_report")
    pine_order_semantics = pine_order_semantics if isinstance(pine_order_semantics, dict) else {}
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
        'pine_enforce_external_call_contract': bool(
            st.session_state.get("pine_enforce_external_call_contract", True)
        ),
        'pine_enforce_order_semantics': bool(
            st.session_state.get("pine_enforce_order_semantics", True)
        ),
        'pine_spec_parser_backend': st.session_state.get("pine_spec_parser_backend", "auto"),
        'pine_llm_provider': st.session_state.get("pine_llm_provider", "openai"),
        'pine_llm_model': st.session_state.get("pine_llm_model", "gpt-5-mini"),
        'pine_llm_base_url': st.session_state.get("pine_llm_base_url", ""),
        'pine_llm_temperature': float(st.session_state.get("pine_llm_temperature", 0.2)),
        'pine_llm_max_tokens': int(st.session_state.get("pine_llm_max_tokens", 4000)),
        'pine_llm_timeout_s': int(st.session_state.get("pine_llm_timeout_s", 120)),
        'pine_llm_retries': int(st.session_state.get("pine_llm_retries", 1)),
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
        'strategy_spec_sha256': st.session_state.get("pine_spec_sha256") if pine_spec else None,
        'strategy_spec_validation_errors': len(pine_spec_validation.get("errors", []) or []),
        'strategy_spec_parser_backend_used': (
            (pine_spec.get("transcription") or {}).get("parser_backend_used")
            if isinstance(pine_spec.get("transcription"), dict)
            else None
        ),
        'pine_llm_migration_status': (
            (st.session_state.get("pine_llm_migration_report") or {}).get("status")
            if isinstance(st.session_state.get("pine_llm_migration_report"), dict)
            else None
        ),
        'pine_beta_ready': bool(pine_beta.get("beta_ready", False)),
        'pine_beta_readiness_score': pine_beta.get("readiness_score"),
        'pine_enforce_parity_gate': bool(st.session_state.get("pine_enforce_parity_gate", True)),
        'pine_execution_gate_status': pine_execution_gate.get("status"),
        'pine_execution_gate_can_run': pine_execution_gate.get("can_run"),
        'pine_execution_gate_blockers_count': len(pine_execution_gate.get("blockers", []) or []),
        'pine_order_semantics_status': pine_order_semantics.get("status"),
        'pine_order_semantics_passed': pine_order_semantics.get("passed"),
        'pine_order_semantics_blockers_count': len(pine_order_semantics.get("blockers", []) or []),
        'pine_order_semantics_price_controls_count': int(pine_order_semantics.get("rules_with_price_controls", 0) or 0),
        'pine_order_semantics_qty_controls_count': int(pine_order_semantics.get("rules_with_qty_controls", 0) or 0),
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
        # Exit toggles — read from session_state (set by render_strategy_panel)
        'exit_sar_enabled': bool(st.session_state.get("exit_sar_enabled", True)),
        'exit_macd_enabled': bool(st.session_state.get("exit_macd_enabled", True)),
        'exit_macd_type_a': bool(st.session_state.get("exit_macd_type_a", True)),
        'exit_macd_type_b': bool(st.session_state.get("exit_macd_type_b", True)),
        'exit_cross_sar_sma_enabled': bool(st.session_state.get("exit_cross_sar_sma_enabled", True)),
        'exit_retour_bb_enabled': bool(st.session_state.get("exit_retour_bb_enabled", False)),
        'exit_regline_enabled': bool(st.session_state.get("exit_regline_enabled", False)),
        'exit_volat_down_enabled': bool(st.session_state.get("exit_volat_down_enabled", False)),
        'use_roc_filter': bool(st.session_state.get("use_roc_filter", True)),
        'use_t2_signal': bool(st.session_state.get("use_t2_signal", False)),
        'use_divergence_bb': bool(st.session_state.get("use_divergence_bb", True)),
        'use_divergence_bb_values': st.session_state.get("use_divergence_bb_values"),
        # Optimize flags: whether each boolean toggle is varied [True, False] in the grid
        'optimize_exit_sar_enabled': bool(st.session_state.get("optimize_exit_sar_enabled", True)),
        'optimize_exit_macd_enabled': bool(st.session_state.get("optimize_exit_macd_enabled", True)),
        'optimize_exit_macd_type_a': bool(st.session_state.get("optimize_exit_macd_type_a", True)),
        'optimize_exit_macd_type_b': bool(st.session_state.get("optimize_exit_macd_type_b", True)),
        'optimize_use_roc_filter': bool(st.session_state.get("optimize_use_roc_filter", False)),
        'optimize_use_t2_signal': bool(st.session_state.get("optimize_use_t2_signal", False)),
        'optimize_use_divergence_bb': bool(st.session_state.get("optimize_use_divergence_bb", False)),
        'optimize_exit_cross_sar_sma_enabled': bool(st.session_state.get("optimize_exit_cross_sar_sma_enabled", False)),
        'optimize_exit_retour_bb_enabled': bool(st.session_state.get("optimize_exit_retour_bb_enabled", False)),
        'optimize_exit_regline_enabled': bool(st.session_state.get("optimize_exit_regline_enabled", False)),
        'optimize_exit_volat_down_enabled': bool(st.session_state.get("optimize_exit_volat_down_enabled", False)),
        'pqs_n_ref': int(st.session_state.get("pqs_n_ref", 50)),
        'macd_ma_type': str(st.session_state.get("macd_ma_type", "sma")),
        'strategy_direction': str(st.session_state.get("strategy_direction", "long_only")),
        # Fixed strategy params (Pine V6 defaults — not in optimisation grid)
        'nb_bars_under_bbw_mini': int(st.session_state.get("nb_bars_under_bbw_mini", 4)),
        'nb_bars_entre_bb': int(st.session_state.get("nb_bars_entre_bb", 5)),
        'depassement_sma_roc': float(st.session_state.get("depassement_sma_roc", 0.01)),
        'roc_max_t1': float(st.session_state.get("roc_max_t1", 100.0)),
        'nb_bars_left_pivot': int(st.session_state.get("nb_bars_left_pivot", 2)),
        'nb_bars_right_pivot': int(st.session_state.get("nb_bars_right_pivot", 2)),
        'nombre_periodes_reglin': int(st.session_state.get("nombre_periodes_reglin", 15)),
        'i_bars_back': int(st.session_state.get("i_bars_back", 1)),
        'seuil_overbought_bb': float(st.session_state.get("seuil_overbought_bb", 0.85)),
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
        'selection_method': str(st.session_state.get('selection_method', 'snv')),
        'svi_top_k': int(st.session_state.get('svi_top_k', 20)),
        'svi_is2_fraction': float(st.session_state.get('svi_is2_fraction', 0.30)),
        'cross_window_method': str(st.session_state.get('cross_window_method', 'best_is_oos')),
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
        'adaptive_cycle_bars': int(st.session_state.get('adaptive_cycle_bars', 5000)),
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
    """Calculates the total number of parameter combinations.

    Mirrors get_param_grid() in main.py: numeric ranges + boolean toggle dims.
    """
    total = 1
    if not config.get('selected_params'):
        return 0

    # Dead-dimension guard: SAR/MACD numeric params excluded when exit disabled.
    _sar_on  = bool(config.get("exit_sar_enabled",  True))
    _macd_on = bool(config.get("exit_macd_enabled", True))
    _SAR_PARAMS  = {'sar_start', 'sar_increment', 'sar_maximum'}
    _MACD_PARAMS = {'macd_fast_length', 'macd_slow_length', 'macd_signal_length'}

    for param in config['selected_params']:
        if param in _SAR_PARAMS  and not _sar_on:
            continue
        if param in _MACD_PARAMS and not _macd_on:
            continue

        p_min  = config.get(f'{param}_min')
        p_max  = config.get(f'{param}_max')
        p_step = config.get(f'{param}_step')

        if p_step is None or p_step <= 0:
            continue

        # Robust count: same epsilon formula as get_param_grid in main.py
        count = int(np.floor((p_max - p_min + 1e-9) / p_step)) + 1
        total *= max(1, count)

    # Boolean toggle dimensions — ×2 only when toggle is ON and user chose to optimize it.
    def _opt(toggle_key, toggle_default, opt_key, opt_default):
        return bool(config.get(toggle_key, toggle_default)) and bool(config.get(opt_key, opt_default))

    if _opt('exit_sar_enabled', True, 'optimize_exit_sar_enabled', True):
        total *= 2
    if _macd_on:
        if bool(config.get('optimize_exit_macd_enabled', True)):
            total *= 2
        if _opt('exit_macd_type_a', True, 'optimize_exit_macd_type_a', True):
            total *= 2
        if _opt('exit_macd_type_b', True, 'optimize_exit_macd_type_b', True):
            total *= 2
    if _opt('use_roc_filter', True, 'optimize_use_roc_filter', False):
        total *= 2
    _t2_active = bool(config.get('use_t2_signal', False))
    if _t2_active:
        if bool(config.get('optimize_use_t2_signal', False)):
            total *= 2
        if _opt('use_divergence_bb', True, 'optimize_use_divergence_bb', False):
            total *= 2
    if _opt('exit_cross_sar_sma_enabled', True, 'optimize_exit_cross_sar_sma_enabled', False):
        total *= 2
    if _opt('exit_retour_bb_enabled', False, 'optimize_exit_retour_bb_enabled', False):
        total *= 2
    if _opt('exit_regline_enabled', False, 'optimize_exit_regline_enabled', False):
        total *= 2
    if _opt('exit_volat_down_enabled', False, 'optimize_exit_volat_down_enabled', False):
        total *= 2

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
    config_wfo_20260210_132530_01m_5s_bayes_05w_tr5000_classic_short.json
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

    direction_map = {
        'long_only':  'long',
        'short_only': 'short',
        'both':       'longshort',
    }
    direction_raw = str(config.get('strategy_direction', 'long_only')).lower()
    direction = direction_map.get(direction_raw, _normalize_filename_token(direction_raw, default='long', max_len=10))

    return f"config_wfo_{now_tag}_{period}_{timeframe}_{method}_{windows_label}_{trials_label}_{regime}_{direction}.json"

def run_final_backtest_logic(window_id=None):
    from ui.final_backtest_panel import run_final_backtest_logic as _run_final_backtest
    return _run_final_backtest(
        get_current_config=get_current_config,
        load_data=load_data,
        resolve_strategy_adapter=resolve_strategy_adapter,
        window_id=window_id,
    )


# ==============================================================================
# MAIN LOGIC
# ==============================================================================

def run_wfo(config, control=None, job_state=None):
    """Run WFO via the run service without direct UI calls."""
    import time as _time
    _ts = _time.strftime("%Y%m%d_%H%M%S")
    return run_optimization_job(
        config=config, control=control, job_state=job_state,
        run_ts=_ts, log_dir="reports/error_logs",
    )

# ---------------------------------------------------------------------------
# Campaign mode intercept — renders campaign panel and stops further execution
# ---------------------------------------------------------------------------
if st.session_state.get("app_mode", "Single Run") == "Campaign":
    from ui.campaign_panel import render_campaign_panel
    render_campaign_panel()
    st.stop()

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
        cycle_bars=current_conf.get("adaptive_cycle_bars", 5000),
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
    if (wfo_thread is not None and not wfo_thread.is_alive()
            and wfo_job_state is not None
            and not wfo_job_state.get('harvested', False)):
        wfo_job_state['harvested'] = True
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
            _log_p = wfo_job_state.get("error_log_path", "")
            st.session_state['wfo_notice'] = ("success", "Optimization finished.")
            st.session_state['wfo_error_log_path'] = _log_p
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
    order_semantics_for_launch = _compute_pine_order_semantics_report(
        strategy_spec=st.session_state.get("pine_strategy_spec"),
        compat_mode=st.session_state.get("pine_compat_mode", "strict"),
        enforce_order_semantics=bool(st.session_state.get("pine_enforce_order_semantics", True)),
    )
    if isinstance(order_semantics_for_launch, dict) and order_semantics_for_launch:
        st.session_state["pine_order_semantics_report"] = order_semantics_for_launch
    else:
        st.session_state.pop("pine_order_semantics_report", None)
    pine_gate_for_launch = _build_pine_execution_gate_report(
        strategy_mode=current_conf.get("strategy_mode", DEFAULT_STRATEGY_MODE),
        beta_readiness_report=st.session_state.get("pine_beta_readiness_report"),
        parity_reference_payload=st.session_state.get("pine_parity_reference_payload"),
        parity_reference_validation=st.session_state.get("pine_parity_reference_validation"),
        parity_report=st.session_state.get("pine_parity_report"),
        enforce_parity_when_reference=bool(st.session_state.get("pine_enforce_parity_gate", True)),
        order_semantics_report=order_semantics_for_launch,
        enforce_order_semantics=bool(st.session_state.get("pine_enforce_order_semantics", True)),
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
            elif not current_conf.get('selected_params'):
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
                    'results_sha256': None,
                    'harvested': False,
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
    _notice_log_p = st.session_state.pop('wfo_error_log_path', '')
    if notice_type == "success":
        st.success(notice_msg)
    elif notice_type == "warning":
        st.warning(notice_msg)
    else:
        st.error(notice_msg)
    if _notice_log_p:
        _notice_lp = os.path.abspath(_notice_log_p)
        if os.path.exists(_notice_lp):
            with open(_notice_lp, encoding="utf-8") as _f:
                _notice_log_data = _f.read()
            st.download_button(
                "📋 Télécharger journal erreurs",
                data=_notice_log_data,
                file_name=os.path.basename(_notice_lp),
                mime="application/json",
                key="dl_classic_error_log",
            )

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
                "`results_wfo_<YYYYMMDD_HHMMSS>_<periode>_<timeframe>_<methode>_<nb_fenetres>_<trials>_<regime>_<direction>.zip`."
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

# Final Backtest section — requires WFO results OR stagewise params loaded
_has_stagewise_params = (
    st.session_state.get('final_params_source') == 'stagewise'
    and 'final_params' in st.session_state
)
if 'wfo_results' in st.session_state or _has_stagewise_params:
    st.sidebar.divider()
    with st.sidebar.expander("🏆 Final Backtest", expanded=True):

        if _has_stagewise_params and 'wfo_results' not in st.session_state:
            st.info("Paramètres stagewise chargés — le backtest final utilisera les meilleurs params de la campagne.")

        # --- Robust Tests (Level 1) — only available with a WFO run ---
        robust_tests_enabled = False
        if 'wfo_results' in st.session_state:
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
            st.number_input(
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
            st.number_input(
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
            st.checkbox(
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

            st.divider()

        # --- PQS settings ---
        st.markdown("##### PQS — Profit Quality Score")
        st.number_input(
            "PQS n_ref (trades de référence)",
            min_value=1,
            max_value=500,
            value=int(st.session_state.get("pqs_n_ref", 50)),
            step=5,
            key="pqs_n_ref",
            help=(
                "Nombre de trades de référence pour le facteur de confiance PQS : √(n_trades / n_ref).\n\n"
                "• En dessous de n_ref trades → PQS pénalisé (évite les configs hyper-sélectives).\n"
                "• Au-dessus → PQS amplifié proportionnellement.\n"
                "Valeur recommandée : 50 (significativité statistique minimale)."
            ),
        )

        st.divider()

        # --- Date range, timeframe & data file ---
        st.markdown("##### Plage de données")
        default_final_start = st.session_state.get('opt_start_date', st.session_state.get('start_date', DEFAULT_START_DATE))
        default_final_end = st.session_state.get('opt_end_date', st.session_state.get('end_date', DEFAULT_END_DATE))
        st.text_input(
            "Final Start Date (YYYY-MM-DD)",
            value=default_final_start,
            key="final_start_date",
            help="Date de début du jeu de données utilisé pour le backtest final."
        )
        st.text_input(
            "Final End Date (YYYY-MM-DD)",
            value=default_final_end,
            key="final_end_date",
            help="Date de fin du jeu de données utilisé pour le backtest final."
        )
        st.text_input(
            "Final Data File Path",
            value=st.session_state.get('file_path', DEFAULT_DATA_FILE),
            key="final_file_path",
            help="Chemin du fichier de données pour le backtest final (si source locale)."
        )
        _tf_options = ['1s', '5s', '10s', '15s', '30s', '1m', '5m', '15m', '30m', '1h', '4h', '1d']
        _wfo_tf = st.session_state.get('timeframe', DEFAULT_TIMEFRAME)
        _final_tf_current = st.session_state.get('final_timeframe', _wfo_tf)
        _final_tf_idx = _tf_options.index(_final_tf_current) if _final_tf_current in _tf_options else \
                        (_tf_options.index(_wfo_tf) if _wfo_tf in _tf_options else 1)
        st.selectbox(
            "Timeframe du backtest final",
            options=_tf_options,
            index=_final_tf_idx,
            key="final_timeframe",
            help=(
                "Résolution temporelle utilisée pour le backtest final et comparatif. "
                "Peut différer du timeframe WFO — permet de tester les best params "
                "sur 10s, 30s, 1mn… sans relancer l'optimisation.\n\n"
                "⚠️ Le fichier de données doit contenir le timeframe sélectionné "
                "(ou une résolution plus fine pour le resampling)."
            ),
        )

        # --- Window selector for final backtest ---
        _fb_window_options = ["Meilleure fenêtre (auto)"] + [f"Fenêtre {w}" for w in _wfo_windows]
        _fb_selected_label = st.selectbox(
            "Fenêtre des paramètres",
            options=_fb_window_options,
            index=0,
            key="final_backtest_window_selector",
            disabled=not _wfo_windows,
            help=(
                "Sélectionner la fenêtre WFO dont les paramètres seront utilisés pour le final backtest. "
                "'Meilleure fenêtre (auto)' utilise le score combiné IS+OOS pour choisir automatiquement."
            ),
        )
        _fb_window_id = None
        if _fb_selected_label != "Meilleure fenêtre (auto)" and _wfo_windows:
            _fb_idx = _fb_window_options.index(_fb_selected_label) - 1
            if 0 <= _fb_idx < len(_wfo_windows):
                _fb_window_id = _wfo_windows[_fb_idx]

        if st.button(
            "🏆 Run Final Backtest",
            width="stretch",
            help=(
                "Exécute un backtest complet avec le jeu de paramètres final. "
                "Source: fenêtre sélectionnée ou best_window/robust_set selon options."
            )
        ):
            run_final_backtest_logic(window_id=_fb_window_id)

# ==============================================================================
# CACHED FIGURE BUILDERS
# Defined at module scope so @st.cache_data persists across reruns.
# Cache key = str(id(results)) — changes only when a new WFO run completes.
# Data arguments are prefixed with _ so Streamlit does not hash them.
# ==============================================================================

@st.cache_data(show_spinner=False)
def _cached_price_windows_chart(cache_key: str, price_col: str, _df, _window_results):
    """Price series + WFO train/test window overlays."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=_df.index, y=_df[price_col], mode='lines', name='Price',
        line=dict(color='#1f77b4', width=1)
    ))
    colors = {'train': 'rgba(0, 255, 0, 0.1)', 'test': 'rgba(255, 0, 0, 0.1)'}
    first_train_labeled = first_test_labeled = False
    out_of_range_windows = 0
    df_x_min = df_x_max = None
    try:
        if len(_df.index) > 0:
            df_x_min = pd.to_datetime(_df.index.min(), errors="coerce")
            df_x_max = pd.to_datetime(_df.index.max(), errors="coerce")
    except Exception:
        pass
    for i, window in enumerate(_window_results):
        info = window['window_info']
        if info['in_sample_start'] and info['in_sample_end']:
            kw = dict(x0=info['in_sample_start'], x1=info['in_sample_end'],
                      fillcolor=colors['train'], layer="below", line_width=0)
            if not first_train_labeled:
                kw["annotation_text"] = f"W{i+1} Train"
                first_train_labeled = True
            fig.add_vrect(**kw)
        if info['out_sample_start'] and info['out_sample_end']:
            kw = dict(x0=info['out_sample_start'], x1=info['out_sample_end'],
                      fillcolor=colors['test'], layer="below", line_width=0)
            if not first_test_labeled:
                kw["annotation_text"] = f"W{i+1} Test"
                first_test_labeled = True
            fig.add_vrect(**kw)
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
        fig.update_xaxes(range=[df_x_min, df_x_max])
    return fig, out_of_range_windows


@st.cache_data(show_spinner=False)
def _cached_is_oos_chart(cache_key: str, _oos_data, _is_data, metric1_name: str = 'sharpe_ratio'):
    """IS vs OOS return % (bars) + selected metric (lines) per window."""
    # Map config metric name → performance dict key + display label
    _METRIC_COL = {
        'sharpe_ratio':       ('sharpe',          'Sharpe'),
        'total_return':       ('return',           'Return %'),
        'max_drawdown':       ('max_drawdown',     'Max DD %'),
        'win_rate':           ('win_rate',         'Win Rate'),
        'avg_gain_per_trade': ('avg_gain_per_trade','Avg Gain/tr'),
        'avg_loss_per_trade': ('avg_loss_per_trade','Avg Loss/tr'),
        'avg_pl_per_trade':   ('avg_pl_per_trade', 'Avg P&L/tr'),
        'pqs':                ('pqs',              'PQS'),
        'calmar_ratio':       ('calmar_ratio',     'Calmar'),
        'sortino_ratio':      ('sortino_ratio',    'Sortino'),
    }
    metric_col, metric_label = _METRIC_COL.get(metric1_name, ('sharpe', 'Sharpe'))

    oos_df = pd.DataFrame(_oos_data)
    is_df  = pd.DataFrame(_is_data)
    if not oos_df.empty:
        oos_df['Window'] = oos_df['window'].astype(str)
    if not is_df.empty:
        is_df['Window'] = is_df['window'].astype(str)
    windows = sorted(
        (set(oos_df['Window'].tolist()) if not oos_df.empty else set()) |
        (set(is_df['Window'].tolist())  if not is_df.empty  else set()),
        key=lambda x: int(x)
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if not is_df.empty:
        _is_idx = is_df.set_index('Window').reindex(windows)
        fig.add_trace(go.Bar(
            x=windows, y=_is_idx['return'],
            name="IS Return %", marker_color='rgb(255, 127, 14)', opacity=0.7
        ), secondary_y=False)
        if metric_col in _is_idx.columns:
            fig.add_trace(go.Scatter(
                x=windows, y=_is_idx[metric_col],
                name=f"IS {metric_label}", mode='lines+markers',
                line=dict(color='rgb(214, 39, 40)')
            ), secondary_y=True)
    if not oos_df.empty:
        _oos_idx = oos_df.set_index('Window').reindex(windows)
        fig.add_trace(go.Bar(
            x=windows, y=_oos_idx['return'],
            name="OOS Return %", marker_color='rgb(55, 83, 109)'
        ), secondary_y=False)
        if metric_col in _oos_idx.columns:
            fig.add_trace(go.Scatter(
                x=windows, y=_oos_idx[metric_col],
                name=f"OOS {metric_label}", mode='lines+markers',
                line=dict(color='rgb(26, 118, 255)')
            ), secondary_y=True)

    # --- Best-window annotations (above chart) ---
    if not is_df.empty and metric_col in _is_idx.columns:
        _is_m = _is_idx[metric_col].dropna()
        if not _is_m.empty:
            best_is_win = _is_m.idxmax()
            best_is_val = float(_is_m.max())
            fig.add_annotation(
                x=best_is_win, xref="x", y=1.13, yref="paper",
                text=f"★ IS W{best_is_win} {metric_label}={best_is_val:.2f}",
                showarrow=False,
                font=dict(size=10, color="rgb(214, 39, 40)"),
                align="center",
                bgcolor="rgba(214, 39, 40, 0.12)",
                bordercolor="rgb(214, 39, 40)",
                borderpad=3, borderwidth=1,
            )
    if not oos_df.empty and metric_col in _oos_idx.columns:
        _oos_m = _oos_idx[metric_col].dropna()
        if not _oos_m.empty:
            best_oos_win = _oos_m.idxmax()
            best_oos_val = float(_oos_m.max())
            fig.add_annotation(
                x=best_oos_win, xref="x", y=1.04, yref="paper",
                text=f"★ OOS W{best_oos_win} {metric_label}={best_oos_val:.2f}",
                showarrow=False,
                font=dict(size=10, color="rgb(26, 118, 255)"),
                align="center",
                bgcolor="rgba(26, 118, 255, 0.12)",
                bordercolor="rgb(26, 118, 255)",
                borderpad=3, borderwidth=1,
            )

    fig.update_layout(height=490, template="plotly_dark", barmode="group",
                      title_text=f"Returns & {metric_label} per Window (IS vs OOS)",
                      margin=dict(t=95))
    fig.update_yaxes(title_text="Return %", secondary_y=False)
    fig.update_yaxes(title_text=metric_label, secondary_y=True)
    return fig


@st.cache_data(show_spinner=False)
def _cached_param_heatmap(cache_key: str, selected_params_key: tuple, _params_df):
    """Normalised parameter stability heatmap."""
    numeric_cols = list(_params_df.select_dtypes(include=[np.number]).columns)
    numeric_cols = [c for c in numeric_cols if c not in ['window', 'metric1_name', 'metric2_name']]
    if selected_params_key:
        numeric_cols = [c for c in numeric_cols if c in selected_params_key]
    if not numeric_cols:
        return None
    norm_df = _params_df[numeric_cols].copy()
    for col in norm_df.columns:
        col_min, col_max = norm_df[col].min(), norm_df[col].max()
        norm_df[col] = (norm_df[col] - col_min) / (col_max - col_min) if col_max != col_min else 0.5
    fig = px.imshow(norm_df.T,
                    labels=dict(x="Window", y="Parameter", color="Normalized Value"),
                    x=list(range(1, len(_params_df) + 1)),
                    aspect="auto", color_continuous_scale="Viridis")
    fig.update_layout(title="Parameter Evolution Across Windows (Normalized)", height=500)
    return fig


@st.cache_data(show_spinner=False)
def _cached_params_table(cache_key: str, _params_df):
    """Best parameters per window as an interactive Plotly table."""
    df_w = _params_df.copy()
    df_w['Window'] = list(range(1, len(_params_df) + 1))
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Window'] + list(_params_df.columns),
                    fill_color='#1e3a5f', align='left',
                    font=dict(size=12, color='#EAF2FF')),
        cells=dict(values=[df_w['Window']] + [df_w[c] for c in _params_df.columns],
                   fill_color=[['#18202A', '#1e2d40'] * (len(df_w) // 2 + 1)][:len(df_w)],
                   align='left',
                   font=dict(size=11, color='#EAF2FF'))
    )])
    fig.update_layout(title='Best Parameters Used for Backtesting in Each WFO Window',
                      title_font_size=16, title_font_color='#EAF2FF',
                      width=1200, height=400,
                      paper_bgcolor='#0F141B', plot_bgcolor='#0F141B')
    fig.add_annotation(
        text=("This table lists the optimal parameters selected during optimization for each "
              "Walk-Forward window.<br>These were used to generate the backtest results shown "
              "in the out-of-sample performance."),
        xref="paper", yref="paper", x=0.5, y=-0.15, showarrow=False,
        font=dict(size=12, color='#EAF2FF'), align="center", bgcolor="#18202A",
        bordercolor="#4DA3FF", borderwidth=1, borderpad=10
    )
    return fig


@st.cache_data(show_spinner=False)
def _cached_robust_support_chart(cache_key: str, _support_data):
    """Robust set support ratio bar chart."""
    support_df = pd.DataFrame(_support_data)
    if support_df.empty or not {"parameter", "support_ratio"}.issubset(support_df.columns):
        return None
    support_df = support_df.copy()
    support_df["support_ratio"] = pd.to_numeric(support_df["support_ratio"], errors="coerce")
    support_df = support_df.dropna(subset=["support_ratio"]).sort_values("support_ratio", ascending=False)
    if support_df.empty:
        return None
    fig = px.bar(support_df, x="parameter", y="support_ratio",
                 template="plotly_dark", title="Robust Set Support Ratio by Parameter",
                 labels={"support_ratio": "Support ratio pondéré", "parameter": "Paramètre"})
    fig.update_layout(height=320)
    return fig


@st.cache_data(show_spinner=False)
def _cached_winrate_hist(cache_key: str, _oos_df):
    """Win rate distribution histogram."""
    return px.histogram(_oos_df, x="win_rate", nbins=10,
                        title="Win Rate Distribution", template="plotly_dark")


@st.cache_data(show_spinner=False)
def _cached_drawdown_hist(cache_key: str, _oos_df):
    """Max drawdown distribution histogram."""
    return px.histogram(_oos_df, x="max_drawdown", nbins=10,
                        title="Max Drawdown Distribution", template="plotly_dark",
                        color_discrete_sequence=['red'])


# ==============================================================================
# STRATEGY CONFIGURATION PANEL (central panel — phases 1-8)
# Widgets set st.session_state keys read by get_current_config() on next rerun.
# ==============================================================================

render_strategy_panel()

# ==============================================================================
# STAGEWISE WFO PANEL
# ==============================================================================
with st.expander("🎯 Campagne WFO Stagewise", expanded=False):
    from ui.stagewise_panel import render_stagewise_panel
    render_stagewise_panel(
        get_current_config=get_current_config,
        load_data=load_data,
    )

# ==============================================================================
# STAGEWISE FINAL BACKTEST RESULTS
# Shown when a stagewise backtest ran successfully but no WFO campaign exists.
# ==============================================================================
if 'final_portfolio' in st.session_state and 'wfo_results' not in st.session_state:
    _sw_pf     = st.session_state['final_portfolio']
    _sw_params = st.session_state.get('final_params', {})
    _sw_src    = st.session_state.get('final_params_source', 'stagewise')
    st.divider()
    st.header("🏆 Backtest Final — Résultats")
    st.caption(f"Source des paramètres : `{_sw_src}`")

    _c1, _c2, _c3, _c4, _c5 = st.columns(5)
    _c1.metric("Total Return",  f"{_sw_pf.total_return * 100:.2f}%")
    _c2.metric("Sharpe Ratio",  f"{_sw_pf.sharpe_ratio:.3f}")
    _c3.metric("Mean P&L %",    f"{_calc_avg_pl(_sw_pf):+.4f}%")
    _c4.metric("Max Drawdown",  f"{_sw_pf.max_drawdown * 100:.2f}%")
    _c5.metric("Trades",        str(len(_sw_pf.trades)))

    _sw_tf = st.session_state.get('final_timeframe') or st.session_state.get('timeframe', DEFAULT_TIMEFRAME)
    st.caption(f"Timeframe : `{_sw_tf}`")

    with st.expander("Paramètres utilisés", expanded=False):
        st.json(_sw_params)

    st.markdown("#### Trade Stats")
    try:
        st.dataframe(_arrow_safe_df(_sw_pf.trades.stats()))
    except Exception as _e:
        st.warning(f"Trade stats non disponibles : {_e}")

    st.markdown("#### Courbe des returns (%)")
    try:
        _val = _sw_pf.value
        _pct = (_val / float(_val.iloc[0]) - 1.0) * 100.0
        _pct_ds = _downsample_series(_pct, max_points=5000)
        _fig_sw = go.Figure()
        _fig_sw.add_trace(go.Scatter(x=_pct_ds.index, y=_pct_ds.values,
                                     mode="lines", name="Portfolio"))
        _fig_sw.update_layout(height=300, template="plotly_dark",
                              yaxis_title="Return %", margin=dict(t=30, b=20))
        st.plotly_chart(_fig_sw, use_container_width=True)
    except Exception as _e:
        st.warning(f"Graphique non disponible : {_e}")

# ==============================================================================
# RESULTS VISUALIZATION
# ==============================================================================

if 'wfo_results' in st.session_state:
    results = st.session_state['wfo_results']
    if isinstance(results, dict):
        # Guard expensive derivations behind a results-identity token.
        # id(results) changes only when a new WFO run assigns a fresh dict to
        # session state, so all three builders are skipped on every plain rerender.
        _results_token = id(results)
        if st.session_state.get("_wfo_results_token") != _results_token:
            results["robust_set_summary"] = _build_robust_set_summary(results, current_conf)
            st.session_state["wfo_results"] = results

            all_trials_df = _build_trials_dataframe_from_results(results)
            if not all_trials_df.empty:
                st.session_state["all_trials_df"] = all_trials_df

            window_info_df = _build_window_info_dataframe(results)
            if not window_info_df.empty:
                st.session_state["window_info_df"] = window_info_df

            st.session_state["_wfo_results_token"] = _results_token

    df = st.session_state.get('df')
    traceability = results.get("traceability") or st.session_state.get("wfo_traceability")

    st.divider()
    if results.get("_source") == "stagewise":
        n_stg = results.get("_n_stages", "?")
        st.header(f"📊 Résultats WFO Stagewise ({n_stg} runs)")
        _sw_summary = results.get("_stages_summary", [])
        if _sw_summary:
            with st.expander("Résumé des runs stagewise", expanded=False):
                _sw_cols = st.columns(min(len(_sw_summary), 4))
                for _i, _s in enumerate(_sw_summary):
                    _col = _sw_cols[_i % len(_sw_cols)]
                    _icon = "✅" if _s.get("status") == "OK" else "❌"
                    _is_sh = _s.get("is_avg_sharpe")
                    _oos_sh = _s.get("oos_avg_sharpe")
                    _col.markdown(
                        f"**{_icon} Run {_s.get('stage')}**  \n"
                        f"{str(_s.get('name', '')).split('—')[-1].strip()}  \n"
                        f"IS `{f'{_is_sh:.2f}' if isinstance(_is_sh, float) else 'n/a'}`  "
                        f"OOS `{f'{_oos_sh:.2f}' if isinstance(_oos_sh, float) else 'n/a'}`"
                    )
    else:
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
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Avg Return", f"{oos_df['return'].mean():.2f}%")
        col2.metric("Avg Sharpe", f"{oos_df['sharpe'].mean():.2f}")
        col3.metric("Avg Mean P&L %", f"{oos_df['avg_pl_per_trade'].mean():+.4f}%" if 'avg_pl_per_trade' in oos_df.columns else "—")
        col4.metric("Avg Max Drawdown", f"{oos_df['max_drawdown'].mean():.2f}%")
        col5.metric("Avg Win Rate", f"{oos_df['win_rate'].mean():.2f}%")
        _avg_pqs = oos_df['pqs'].mean() if 'pqs' in oos_df.columns else float('nan')
        col6.metric("Avg PQS", f"{_avg_pqs:.4f}" if not (isinstance(_avg_pqs, float) and _avg_pqs != _avg_pqs) else "—")
    
    # Tabs for different views
    # Stable cache key: changes only when a new results dict is assigned to session state.
    _results_cache_key = str(id(results))

    # Sticky tab bar — injected once when results are displayed.
    # Targets the Streamlit tab list container and pins it below the top toolbar.
    st.markdown("""
        <style>
        div[data-testid="stTabs"] > div[role="tablist"] {
            position: sticky;
            top: 2.875rem;   /* height of Streamlit top toolbar */
            z-index: 999;
            background-color: #0e1117;
            padding-top: 4px;
            padding-bottom: 2px;
        }
        </style>
    """, unsafe_allow_html=True)

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
            price_col = 'Close' if 'Close' in df.columns else df.columns[0]
            _fig_price, _out_of_range = _cached_price_windows_chart(
                _results_cache_key, price_col, df, results['window_results']
            )
            st.plotly_chart(_fig_price, use_container_width=True)
            if _out_of_range > 0:
                st.info(
                    f"{_out_of_range} fenêtre(s) WFO sont hors de la plage de prix affichée. "
                    "Cela arrive si les résultats WFO importés et la série de prix active n'ont pas la même période."
                )
        
        # IS + OOS Performance per Window (shared scale)
        if results['out_of_sample_performance'] or results['in_sample_performance']:
            st.subheader("In-Sample vs Out-of-Sample Performance by Window")
            _metric1_name = current_conf.get('metric1_name', 'sharpe_ratio')
            _fig_perf = _cached_is_oos_chart(
                f"{_results_cache_key}_{_metric1_name}",
                results['out_of_sample_performance'],
                results['in_sample_performance'],
                metric1_name=_metric1_name,
            )
            st.plotly_chart(_fig_perf, use_container_width=True)

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
            _fig_heat = _cached_param_heatmap(
                _results_cache_key, tuple(selected_for_opt), params_df
            )
            if _fig_heat is not None:
                st.plotly_chart(_fig_heat, use_container_width=True)
            
            # =========================================================================
            # BEST PARAMETERS TABLE PER WFO WINDOW
            # =========================================================================
            st.subheader("Best Parameters Table per WFO Window")

            # Create a table showing the best parameters for each WFO window
            if not params_df.empty:
                _display_df = _arrow_safe_df(params_df.copy())
                _display_df.insert(0, "Window", range(1, len(_display_df) + 1))
                st.dataframe(_display_df, use_container_width=True, hide_index=True)

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

                _support_data = robust_summary.get("support_by_param", []) or []
                support_df = pd.DataFrame(_support_data)
                if not support_df.empty:
                    st.markdown("##### Support par paramètre")
                    st.dataframe(support_df, width="stretch")
                    _fig_support = _cached_robust_support_chart(_results_cache_key, _support_data)
                    if _fig_support is not None:
                        st.plotly_chart(_fig_support, use_container_width=True)

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
                st.plotly_chart(
                    _cached_winrate_hist(_results_cache_key, oos_df),
                    use_container_width=True,
                )
            with col_b:
                st.subheader("Drawdown Distribution")
                st.plotly_chart(
                    _cached_drawdown_hist(_results_cache_key, oos_df),
                    use_container_width=True,
                )

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
        st.dataframe(_arrow_safe_df(pd.DataFrame(results['out_of_sample_performance'])))

        st.write("In-Sample Metrics:")
        st.dataframe(_arrow_safe_df(pd.DataFrame(results['in_sample_performance'])))
        
        st.write("Full Results Object (JSON):")
        with st.expander("Show JSON"):
            # Exclude large dataframes for display
            clean_res = {k:v for k,v in results.items() if k not in ['window_results']}
            st.json(clean_res)

        if st.session_state.get("window_info_df") is not None:
            st.write("Window Info:")
            st.dataframe(_arrow_safe_df(st.session_state["window_info_df"]), width="stretch")

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
            st.dataframe(_arrow_safe_df(page_df), width="stretch")
    
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
            
            _wfo_tf    = st.session_state.get('timeframe', DEFAULT_TIMEFRAME)
            _final_tf  = st.session_state.get('final_timeframe') or _wfo_tf
            _tf_note   = f" *(backtest sur **{_final_tf}**)*" if _final_tf != _wfo_tf else ""
            st.markdown(
                f"**Timeframe WFO :** `{_wfo_tf}` — "
                f"**Timeframe backtest final :** `{_final_tf}`{_tf_note}"
            )
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
                _is_pqs = best_is_metrics.get('pqs')
                _is_pqs_str = f", PQS `{_is_pqs:.4f}`" if _is_pqs is not None else ""
                st.markdown(
                    f"**IS (Window {best_is_metrics['window']}):** "
                    f"Return `{best_is_metrics['return']:.2f}%`, "
                    f"Sharpe `{best_is_metrics['sharpe']:.2f}`, "
                    f"Max DD `{best_is_metrics['max_drawdown']:.2f}%`, "
                    f"Win Rate `{best_is_metrics['win_rate']:.2f}%`, "
                    f"Trades `{best_is_metrics['n_trades']}`"
                    f"{_is_pqs_str}"
                )
            if best_oos_metrics:
                _oos_pqs = best_oos_metrics.get('pqs')
                _oos_pqs_str = f", PQS `{_oos_pqs:.4f}`" if _oos_pqs is not None else ""
                st.markdown(
                    f"**OOS (Window {best_oos_metrics['window']}):** "
                    f"Return `{best_oos_metrics['return']:.2f}%`, "
                    f"Sharpe `{best_oos_metrics['sharpe']:.2f}`, "
                    f"Max DD `{best_oos_metrics['max_drawdown']:.2f}%`, "
                    f"Win Rate `{best_oos_metrics['win_rate']:.2f}%`, "
                    f"Trades `{best_oos_metrics['n_trades']}`"
                    f"{_oos_pqs_str}"
                )
            
            # Metrics
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Total Return", f"{pf.total_return * 100:.2f}%")
            m2.metric("Sharpe Ratio", f"{pf.sharpe_ratio:.2f}")
            m3.metric("Mean P&L %",   f"{_calc_avg_pl(pf):+.4f}%")
            m4.metric("Max Drawdown", f"{pf.max_drawdown * 100:.2f}%")
            m5.metric("Win Rate",     f"{pf.trades.win_rate * 100:.2f}%")
            _pqs_n_ref = int(st.session_state.get('pqs_n_ref', 50))
            _pqs_val = _calc_pqs(pf, n_ref=_pqs_n_ref)
            m6.metric("PQS", f"{_pqs_val:.4f}")
            
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

                # Compute portfolio % return from start
                pf_pct = (value_series / float(value_series.iloc[0]) - 1.0) * 100.0
                pf_pct_plot = _downsample_series(pf_pct, max_points=max_points)
                pf_final_pct = float(pf_pct.iloc[-1])

                # Gather price data
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

                # --- Graphique 1 : Prix de l'actif ---
                fig_price = go.Figure()
                if price_series is not None and not price_series.empty:
                    price_plot = _downsample_series(price_series, max_points=max_points)
                    fig_price.add_trace(go.Scatter(
                        x=price_plot.index, y=price_plot.values,
                        mode="lines", name="Asset Price",
                        line=dict(color="#FF7F0E", width=1)
                    ))
                fig_price.update_layout(
                    height=260, template="plotly_dark",
                    title="Prix de l'actif",
                    yaxis_title="Prix",
                    margin=dict(t=45, b=20)
                )
                st.plotly_chart(fig_price, use_container_width=True)

                # --- Graphique 2 : Returns % (portfolio vs B&H) ---
                fig_ret = go.Figure()
                fig_ret.add_trace(go.Scatter(
                    x=pf_pct_plot.index, y=pf_pct_plot.values,
                    mode="lines", name=f"Portfolio ({pf_final_pct:+.1f}%)",
                    line=dict(color="#1f77b4", width=1.5)
                ))
                # B&H % return
                if price_series is not None and not price_series.empty:
                    aligned_price = price_series.reindex(value_series.index).ffill().bfill().dropna()
                    common_index = value_series.index.intersection(aligned_price.index)
                    if len(common_index) > 1:
                        price_common = aligned_price.loc[common_index]
                        initial_price = float(price_common.iloc[0])
                        if np.isfinite(initial_price) and initial_price != 0:
                            bh_pct = (price_common / initial_price - 1.0) * 100.0
                            bh_final_pct = float(bh_pct.iloc[-1])
                            bh_pct_plot = _downsample_series(bh_pct, max_points=max_points)
                            fig_ret.add_trace(go.Scatter(
                                x=bh_pct_plot.index, y=bh_pct_plot.values,
                                mode="lines",
                                name=f"Buy & Hold ({bh_final_pct:+.1f}%)",
                                line=dict(color="#2CA02C", width=1.5, dash="dash")
                            ))
                            fig_ret.add_annotation(
                                x=bh_pct_plot.index[-1], y=bh_final_pct,
                                text=f"B&H {bh_final_pct:+.1f}%",
                                showarrow=True, arrowhead=2, arrowwidth=1,
                                arrowcolor="#2CA02C", ax=45, ay=0,
                                font=dict(size=9, color="#2CA02C"),
                                bgcolor="rgba(44,160,44,0.15)",
                                bordercolor="#2CA02C", borderpad=3, borderwidth=1,
                                xref="x", yref="y",
                            )
                fig_ret.add_annotation(
                    x=pf_pct_plot.index[-1], y=pf_final_pct,
                    text=f"Portfolio {pf_final_pct:+.1f}%",
                    showarrow=True, arrowhead=2, arrowwidth=1,
                    arrowcolor="#1f77b4", ax=45, ay=-20,
                    font=dict(size=9, color="#1f77b4"),
                    bgcolor="rgba(31,119,180,0.15)",
                    bordercolor="#1f77b4", borderpad=3, borderwidth=1,
                    xref="x", yref="y",
                )
                fig_ret.add_hline(y=0, line_dash="dot",
                                  line_color="rgba(255,255,255,0.25)", line_width=1)
                fig_ret.update_layout(
                    height=340, template="plotly_dark",
                    title="Returns (%)",
                    yaxis_title="Return %",
                    margin=dict(t=45, b=20)
                )
                st.plotly_chart(fig_ret, use_container_width=True)

            except Exception as e:
                st.warning(f"Plot skipped due to size or data issue: {e}")
            
            st.markdown("#### Trade Stats")
            st.dataframe(_arrow_safe_df(pf.trades.stats()))
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
                    st.dataframe(_arrow_safe_df(stats_df))
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
                    st.dataframe(_arrow_safe_df(trades_df))
            else:
                st.info("Run the final backtest or load a results ZIP that includes final backtest data.")
        
        # Comparative backtests — all windows on full date range
        _render_window_comparison_panel(
            get_current_config=get_current_config,
            load_data=load_data,
            resolve_strategy_adapter=resolve_strategy_adapter,
        )

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
