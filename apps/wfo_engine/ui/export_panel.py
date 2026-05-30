"""Export / ZIP / PDF functions extracted from app.py.

These are standalone copies of the export helpers.  They reference many
private helpers that still live in ``app.py`` (e.g. ``_build_results_payload``,
``_build_config_filename``, ``_build_expert_context_pack_for_export``, etc.).
The caller must supply or monkey-patch those dependencies when wiring
this module back into the Streamlit application.
"""

import datetime
import io
import json
import os
import hashlib
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pandas as pd
import streamlit as st

from config import DEFAULT_PARAM_GRID
from domain.serialization import (
    sanitize_for_json as _sanitize_for_json,
    to_jsonable as _to_jsonable,
)
from services.export_utils import (
    build_data_source_descriptor as _build_data_source_descriptor_base,
    build_replay_manifest as _build_replay_manifest_base,
)
from ui.data_utils import (
    downsample_series as _downsample_series,
    downsample_df as _downsample_df,
    compute_trade_pnl_metrics as _compute_trade_pnl_metrics,
)
from pine_v3.parity import (
    build_parity_reference_payload as _build_parity_reference_payload,
    validate_parity_reference_payload as _validate_parity_reference_payload,
)
from pine_v3.execution_gate import (
    build_execution_gate_report as _build_pine_execution_gate_report,
)
from pine_v3.runtime_adapter import (
    build_order_semantics_report as _build_pine_order_semantics_report,
)


# ---------------------------------------------------------------------------
# Thin wrappers (originally in app.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _export_results_zip(
    data_snapshot_mode="manifest_only",
    df_max_rows=200000,
    full_package=False,
    *,
    # Injected callables – the caller (app.py) must pass these
    build_results_payload,
    build_expert_context_pack_for_export,
    compute_pine_order_semantics_report,
    build_pine_beta_readiness_report,
    build_pine_execution_gate_report,
    resolve_pine_source_for_artifacts,
    resolve_pine_libraries_for_artifacts,
    build_pine_generation_trace,
    build_pine_artifacts_manifest,
    build_window_info_dataframe,
    build_trials_dataframe_from_results,
    DEFAULT_STRATEGY_MODE="atdmf",
):
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
    pine_order_semantics_report = compute_pine_order_semantics_report(
        strategy_spec=pine_strategy_spec,
        compat_mode=st.session_state.get("pine_compat_mode", "strict"),
        enforce_order_semantics=bool(st.session_state.get("pine_enforce_order_semantics", True)),
    )
    pine_parity_report = st.session_state.get("pine_parity_report")
    pine_request_security_diagnostics = st.session_state.get("pine_request_security_diagnostics")
    pine_mtf_parity_proof_report = st.session_state.get("pine_mtf_parity_proof_report")
    pine_parity_reference_payload = st.session_state.get("pine_parity_reference_payload")
    pine_parity_reference_validation = st.session_state.get("pine_parity_reference_validation")
    pine_parity_reference_metrics = st.session_state.get("pine_parity_reference_metrics")
    pine_beta_readiness_report = build_pine_beta_readiness_report(
        precheck_report=pine_precheck_report,
        compatibility_report=pine_compatibility_report,
        strategy_spec=pine_strategy_spec,
        strategy_spec_validation=pine_strategy_spec_validation,
        codegen_report=pine_codegen_report,
        generated_module_path=pine_generated_module_path,
    )
    pine_execution_gate_report = build_pine_execution_gate_report(
        strategy_mode=st.session_state.get("strategy_mode", DEFAULT_STRATEGY_MODE),
        beta_readiness_report=pine_beta_readiness_report,
        parity_reference_payload=pine_parity_reference_payload,
        parity_reference_validation=pine_parity_reference_validation,
        parity_report=pine_parity_report,
        enforce_parity_when_reference=bool(st.session_state.get("pine_enforce_parity_gate", True)),
        order_semantics_report=pine_order_semantics_report,
        enforce_order_semantics=bool(st.session_state.get("pine_enforce_order_semantics", True)),
    )
    pine_source_artifact = resolve_pine_source_for_artifacts()
    pine_library_artifacts = resolve_pine_libraries_for_artifacts()
    pine_generation_trace = build_pine_generation_trace(
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
    pine_artifacts_manifest = build_pine_artifacts_manifest(
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
        order_semantics_report=pine_order_semantics_report,
        parity_report=pine_parity_report,
        request_security_diagnostics=pine_request_security_diagnostics,
        mtf_parity_proof_report=pine_mtf_parity_proof_report,
        parity_reference_payload=pine_parity_reference_payload,
        parity_reference_validation=pine_parity_reference_validation,
    )
    # Batch all session-state mutations: collect into _updates / _pops, then apply
    # in two calls instead of 40+ individual Streamlit serialisation checkpoints.
    # NOTE: pine_library_paths/names depend on pine_library_files — resolved via
    #       local variable _pine_lib_files to avoid a read-after-write dependency.
    _updates: dict = {}
    _pops: list = []

    if isinstance(pine_source_artifact, dict) and pine_source_artifact.get("text"):
        _updates["pine_source_text"] = pine_source_artifact.get("text")
    else:
        _pops.append("pine_source_text")
    if isinstance(pine_library_artifacts, list):
        _pine_lib_files = [
            {
                "source_name": str(item.get("source_name") or ""),
                "path": str(item.get("path") or ""),
                "source_sha1": str(item.get("source_sha1") or ""),
                "size_bytes": int(len(str(item.get("text") or "").encode("utf-8", errors="ignore"))),
            }
            for item in pine_library_artifacts
            if isinstance(item, dict)
        ]
        _updates["pine_library_files"] = _pine_lib_files
        _updates["pine_library_paths"] = [str(item.get("path") or "") for item in _pine_lib_files]
        _updates["pine_library_names"] = [str(item.get("source_name") or "") for item in _pine_lib_files]
    if isinstance(pine_import_mapping, dict):
        _updates["pine_import_mapping"] = dict(pine_import_mapping)
    if isinstance(pine_codegen_report, dict) and pine_codegen_report:
        _updates["pine_codegen_report"] = pine_codegen_report
    else:
        _pops.append("pine_codegen_report")
    if isinstance(pine_generated_module_path, str) and pine_generated_module_path.strip():
        _updates["pine_generated_module_path"] = pine_generated_module_path.strip()
    else:
        _pops.append("pine_generated_module_path")
    if isinstance(pine_generation_trace, dict) and pine_generation_trace:
        _updates["pine_generation_trace"] = pine_generation_trace
    else:
        _pops.extend(["pine_generation_trace", "pine_llm_migration_report",
                       "pine_catalog_last_entry", "pine_llm_override_spec",
                       "pine_llm_override_source_sha1"])
    if isinstance(pine_beta_readiness_report, dict) and pine_beta_readiness_report:
        _updates["pine_beta_readiness_report"] = pine_beta_readiness_report
    else:
        _pops.append("pine_beta_readiness_report")
    if isinstance(pine_execution_gate_report, dict) and pine_execution_gate_report:
        _updates["pine_execution_gate_report"] = pine_execution_gate_report
    else:
        _pops.append("pine_execution_gate_report")
    if isinstance(pine_order_semantics_report, dict) and pine_order_semantics_report:
        _updates["pine_order_semantics_report"] = pine_order_semantics_report
    else:
        _pops.append("pine_order_semantics_report")
    if isinstance(pine_parity_report, dict) and pine_parity_report:
        _updates["pine_parity_report"] = pine_parity_report
    else:
        _pops.append("pine_parity_report")
    if isinstance(pine_request_security_diagnostics, dict) and pine_request_security_diagnostics:
        _updates["pine_request_security_diagnostics"] = pine_request_security_diagnostics
    else:
        _pops.append("pine_request_security_diagnostics")
    if isinstance(pine_mtf_parity_proof_report, dict) and pine_mtf_parity_proof_report:
        _updates["pine_mtf_parity_proof_report"] = pine_mtf_parity_proof_report
    else:
        _pops.append("pine_mtf_parity_proof_report")
    if isinstance(pine_parity_reference_payload, dict) and pine_parity_reference_payload:
        _updates["pine_parity_reference_payload"] = pine_parity_reference_payload
    else:
        _pops.append("pine_parity_reference_payload")
    if isinstance(pine_parity_reference_validation, dict) and pine_parity_reference_validation:
        _updates["pine_parity_reference_validation"] = pine_parity_reference_validation
    else:
        _pops.append("pine_parity_reference_validation")
    if isinstance(pine_parity_reference_metrics, dict) and pine_parity_reference_metrics:
        _updates["pine_parity_reference_metrics"] = pine_parity_reference_metrics
    else:
        _pops.extend(["pine_parity_reference_metrics", "pine_parity_reference_text"])
    if isinstance(pine_artifacts_manifest, dict) and pine_artifacts_manifest:
        _updates["pine_artifacts_manifest"] = pine_artifacts_manifest
    else:
        _pops.append("pine_artifacts_manifest")

    # Apply all mutations in two Streamlit operations instead of 40+
    for _k in _pops:
        st.session_state.pop(_k, None)
    st.session_state.update(_updates)

    payload = build_results_payload()
    config_snapshot = payload.get("config", {})
    zip_buffer = io.BytesIO()
    expert_context_pack = build_expert_context_pack_for_export(results, config_snapshot)

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
        if isinstance(pine_order_semantics_report, dict) and pine_order_semantics_report:
            zf.writestr(
                "pine_order_semantics_report.json",
                json.dumps(pine_order_semantics_report, indent=2, ensure_ascii=False),
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
        pine_llm_migration_report = st.session_state.get("pine_llm_migration_report")
        if isinstance(pine_llm_migration_report, dict) and pine_llm_migration_report:
            zf.writestr(
                "pine_llm_migration_report.json",
                json.dumps(pine_llm_migration_report, indent=2, ensure_ascii=False),
            )
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

        window_info_df = build_window_info_dataframe(results)
        if not window_info_df.empty:
            zf.writestr("window_info.csv", window_info_df.to_csv(index=False))

        trials_df = build_trials_dataframe_from_results(results)
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

        # Window comparison — IS params applied on the full period (if available)
        window_cmp_rows = st.session_state.get("window_comparison_results")
        if isinstance(window_cmp_rows, list) and window_cmp_rows:
            cmp_export = _sanitize_for_json([
                {
                    "window_id": r.get("window_id"),
                    "params": r.get("params", {}),
                    "metrics": r.get("metrics", {}),
                }
                for r in window_cmp_rows
            ])
            zf.writestr("window_comparison.json", json.dumps(cmp_export, indent=2))

            metrics_records = []
            for r in cmp_export:
                rec = {"window": r.get("window_id")}
                rec.update(r.get("metrics") or {})
                metrics_records.append(rec)
            metrics_df = pd.DataFrame(metrics_records)
            if not metrics_df.empty:
                zf.writestr("window_comparison_metrics.csv", metrics_df.to_csv(index=False))

            params_records = []
            for r in cmp_export:
                rec = {"window": r.get("window_id")}
                rec.update(r.get("params") or {})
                params_records.append(rec)
            params_df = pd.DataFrame(params_records)
            if not params_df.empty:
                zf.writestr("window_comparison_params.csv", params_df.to_csv(index=False))

        replay_manifest = _build_replay_manifest(
            payload,
            snapshot_mode_requested=snapshot_mode_requested,
            snapshot_mode_actual=snapshot_mode_actual,
            has_df_snapshot=has_df_snapshot,
            data_source=data_source_descriptor
        )
        zf.writestr("replay_manifest.json", json.dumps(replay_manifest, indent=2))

        # Include stagewise campaign report when results come from a stagewise run
        _sw_stages = (results or {}).get("_stages_summary")
        _sw_final = (results or {}).get("_final_best_params")
        if _sw_stages and _sw_final:
            _sw_report = {
                "source": "stagewise",
                "campaign_start": results.get("_campaign_start"),
                "campaign_end":   results.get("_campaign_end"),
                "n_stages":       results.get("_n_stages"),
                "final_best_params": _sw_final,
                "stages_summary":    _sw_stages,
            }
            zf.writestr(
                "stagewise_campaign_summary.json",
                json.dumps(_sw_report, indent=2, default=str),
            )

    zip_buffer.seek(0)
    return zip_buffer


def _build_results_zip_filename(config, *, build_config_filename):
    """
    Build ZIP filename using the same naming rule components as config export.
    Example:
    results_wfo_20260210_132530_01m_5s_bayes_05w_tr5000_classic_short.zip
    """
    cfg_name = build_config_filename(config)
    stem = cfg_name.removeprefix("config_wfo_").removesuffix(".json")
    return f"results_wfo_{stem}.zip"


def _build_results_pdf_filename(config, *, build_config_filename):
    """Build PDF filename aligned with config naming rule."""
    cfg_name = build_config_filename(config)
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


def _generate_wfo_pdf_report(
    results,
    config,
    *,
    # Injected callables
    build_robust_set_summary,
    extract_final_backtest_for_expert,
    build_trials_dataframe_from_results,
    build_window_info_dataframe,
):
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
        all_trials_df = build_trials_dataframe_from_results(results)
    window_info_df = st.session_state.get("window_info_df")
    if window_info_df is None or (isinstance(window_info_df, pd.DataFrame) and window_info_df.empty):
        window_info_df = build_window_info_dataframe(results)
    robust_summary = results.get("robust_set_summary")
    if not isinstance(robust_summary, dict) or not robust_summary:
        robust_summary = build_robust_set_summary(results, config)

    final_summary = extract_final_backtest_for_expert(results, config, df_source=st.session_state.get("df"))

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


def _save_results_zip_to_disk(zip_buffer, config, *, build_config_filename):
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    filename = _build_results_zip_filename(config, build_config_filename=build_config_filename)
    path = os.path.join(reports_dir, filename)
    path = os.path.abspath(path)
    with open(path, "wb") as f:
        f.write(zip_buffer.getvalue())
    return path


def _save_results_pdf_to_disk(pdf_buffer, config, *, build_config_filename):
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    filename = _build_results_pdf_filename(config, build_config_filename=build_config_filename)
    path = os.path.join(reports_dir, filename)
    path = os.path.abspath(path)
    with open(path, "wb") as f:
        f.write(pdf_buffer.getvalue())
    return path


def _load_results_zip(
    zip_file,
    *,
    # Injected callables
    validate_strategy_spec_v1,
    validate_parity_reference_payload_fn=None,
    build_parity_reference_payload_fn=None,
    apply_parity_reference_payload,
    persist_pine_source_text,
    persist_generated_strategy_text,
    persist_pine_library_text,
):
    if validate_parity_reference_payload_fn is None:
        validate_parity_reference_payload_fn = _validate_parity_reference_payload
    if build_parity_reference_payload_fn is None:
        build_parity_reference_payload_fn = _build_parity_reference_payload

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
                pine_llm_report = payload.get("pine_llm_migration_report")
                if isinstance(pine_llm_report, dict):
                    st.session_state["pine_llm_migration_report"] = pine_llm_report
                pine_catalog_entry = payload.get("pine_catalog_last_entry")
                if isinstance(pine_catalog_entry, dict):
                    st.session_state["pine_catalog_last_entry"] = pine_catalog_entry
                pine_manifest = payload.get("pine_artifacts_manifest")
                if isinstance(pine_manifest, dict):
                    st.session_state["pine_artifacts_manifest"] = pine_manifest
                pine_beta = payload.get("pine_beta_readiness_report")
                if isinstance(pine_beta, dict):
                    st.session_state["pine_beta_readiness_report"] = pine_beta
                pine_gate = payload.get("pine_execution_gate_report")
                if isinstance(pine_gate, dict):
                    st.session_state["pine_execution_gate_report"] = pine_gate
                pine_order_semantics = payload.get("pine_order_semantics_report")
                if isinstance(pine_order_semantics, dict):
                    st.session_state["pine_order_semantics_report"] = pine_order_semantics
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

            pine_llm_report, _ = _read_json_from_candidates(
                ["pine_llm_migration_report.json"],
                "pine_llm_migration_report",
            )
            if isinstance(pine_llm_report, dict):
                st.session_state["pine_llm_migration_report"] = pine_llm_report

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

            pine_order_semantics, _ = _read_json_from_candidates(
                ["pine_order_semantics_report.json"],
                "pine_order_semantics",
            )
            if isinstance(pine_order_semantics, dict):
                st.session_state["pine_order_semantics_report"] = pine_order_semantics

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
                loaded_validation = validate_parity_reference_payload_fn(
                    loaded_parity_payload,
                    allow_legacy=False,
                )
                normalized_payload = loaded_validation.get("normalized_payload")
                if isinstance(normalized_payload, dict) and normalized_payload:
                    apply_parity_reference_payload(normalized_payload, loaded_validation, update_text=True)
            elif isinstance(st.session_state.get("pine_parity_reference_metrics"), dict):
                legacy_payload = build_parity_reference_payload_fn(
                    st.session_state.get("pine_parity_reference_metrics"),
                    source={
                        "provider": "legacy_zip",
                        "strategy_id": str(st.session_state.get("strategy_id") or ""),
                    },
                )
                legacy_validation = validate_parity_reference_payload_fn(legacy_payload, allow_legacy=False)
                apply_parity_reference_payload(legacy_payload, legacy_validation, update_text=True)

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
                persisted_pine_path = persist_pine_source_text(pine_source_text, source_name=source_name)
                if isinstance(persisted_pine_path, str):
                    st.session_state["pine_file_path"] = persisted_pine_path

            generated_strategy_text, _, generated_strategy_file = _read_text_from_candidates(
                ["generated_strategy.py"],
                "generated_strategy",
            )
            if isinstance(generated_strategy_text, str) and generated_strategy_text.strip():
                generated_name = os.path.basename(str(generated_strategy_file or "generated_strategy.py"))
                persisted_generated_path = persist_generated_strategy_text(
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
                    persisted_path = persist_pine_library_text(lib_text, source_name=source_name or os.path.basename(zip_path))
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
                            validate_strategy_spec_v1(loaded_spec)
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
                            "Le SHA1 de la source Pine rechargee differe du precheck enregistre. "
                            "Verifie la coherence des artefacts."
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
