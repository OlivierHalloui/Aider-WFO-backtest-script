"""
Expert AI panel helper functions.

Extracted from app.py to keep the main Streamlit entry point lean.
All functions here are pure data-building / IO helpers consumed by the
Expert AI sidebar and follow-up Q/A workflow.
"""

import datetime
import json
import os
import re

import numpy as np
import pandas as pd
import streamlit as st

from domain.serialization import (
    sanitize_for_json as _sanitize_for_json,
    to_jsonable as _to_jsonable,
    safe_float_scalar as _safe_float_scalar,
    utc_now_iso as _utc_now_iso,
)
from expert import ExpertInputData, ExpertRunContext
from ui.data_utils import get_return_series as _get_return_series

# ---------------------------------------------------------------------------
# Lazy import helpers – avoid circular imports at module level.
# ---------------------------------------------------------------------------

def _get_build_trials_dataframe_from_results():
    """Return ``build_trials_dataframe_from_results`` from the metrics module."""
    from metrics import build_trials_dataframe_from_results
    return build_trials_dataframe_from_results


# ---------------------------------------------------------------------------
# Parameter help texts (duplicated from app.py constants to keep the module
# self-contained for the subset of functions that reference it).
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Expert model catalog
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Compact trials builder
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Pine artifacts summary  (NOTE: shared with Pine panel — included here for
# Expert diagnostics consumption)
# ---------------------------------------------------------------------------

def _build_pine_artifacts_summary_for_expert():
    """Build a compact Pine artifacts summary consumable by Expert diagnostics."""
    precheck = st.session_state.get("pine_precheck_report")
    compat = st.session_state.get("pine_compatibility_report")
    spec = st.session_state.get("pine_strategy_spec")
    spec_validation = st.session_state.get("pine_strategy_spec_validation")
    trace = st.session_state.get("pine_generation_trace")
    llm_migration_report = st.session_state.get("pine_llm_migration_report")
    codegen = st.session_state.get("pine_codegen_report")
    beta_readiness = st.session_state.get("pine_beta_readiness_report")
    execution_gate = st.session_state.get("pine_execution_gate_report")
    order_semantics = st.session_state.get("pine_order_semantics_report")
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
        and not isinstance(order_semantics, dict)
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
            "llm_migration_status": (
                llm_migration_report.get("status")
                if isinstance(llm_migration_report, dict)
                else None
            ),
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
            "order_semantics_status": (
                order_semantics.get("status") if isinstance(order_semantics, dict) else None
            ),
            "order_semantics_passed": (
                order_semantics.get("passed")
                if isinstance(order_semantics, dict) and "passed" in order_semantics
                else None
            ),
            "order_semantics_blockers_count": len((order_semantics.get("blockers") or []))
            if isinstance(order_semantics, dict)
            else 0,
            "order_semantics_price_controls_count": (
                int(order_semantics.get("rules_with_price_controls", 0))
                if isinstance(order_semantics, dict)
                else 0
            ),
            "order_semantics_qty_controls_count": (
                int(order_semantics.get("rules_with_qty_controls", 0))
                if isinstance(order_semantics, dict)
                else 0
            ),
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


# ---------------------------------------------------------------------------
# Strategy context builder
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Final backtest extractor
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Deterministic pre-diagnostic alerts
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Report directory / path helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Prompt template I/O
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Prompt resolution helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Saved report helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Expert input data builder
# ---------------------------------------------------------------------------

def _build_expert_input_data(results, current_conf):
    if not isinstance(results, dict):
        return None

    _build_trials_dataframe_from_results = _get_build_trials_dataframe_from_results()

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


# ---------------------------------------------------------------------------
# Follow-up Q/A helpers
# ---------------------------------------------------------------------------

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
    _build_trials_dataframe_from_results = _get_build_trials_dataframe_from_results()

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


# ---------------------------------------------------------------------------
# JSON / text parsing helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Serialization / export helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Expert model / provider helpers
# ---------------------------------------------------------------------------

def _get_expert_model_entries(provider):
    return EXPERT_MODEL_CATALOG.get(str(provider or "").lower(), EXPERT_MODEL_CATALOG["openai"])


def _get_expert_provider_defaults(provider):
    key = str(provider or "").lower()
    if key == "grok":
        return {"base_url": "https://api.x.ai/v1", "label": "Grok"}
    if key == "gemini":
        return {"base_url": "https://generativelanguage.googleapis.com/v1beta", "label": "Gemini"}
    return {"base_url": "https://api.openai.com/v1", "label": "OpenAI"}
