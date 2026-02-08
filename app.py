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
import subprocess

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
from main import get_param_grid, get_metrics_info, get_wfo_settings
from wfo import walk_forward_optimization, OptimizationInterrupted
from adaptive_optimization import adaptive_continuous_optimization
from data_loading import load_data, get_csv_date_range
from strategy import run_backtest

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

# ==============================================================================
# SIDEBAR CONFIGURATION
# ==============================================================================

def _downsample_series(series, max_points=20000):
    if series is None or len(series) <= max_points:
        return series
    step = max(1, len(series) // max_points)
    return series.iloc[::step]

def _downsample_df(df, max_rows=200000):
    if df is None or df.empty or len(df) <= max_rows:
        return df
    step = max(1, len(df) // max_rows)
    return df.iloc[::step]

def _get_return_series(trades_df):
    if trades_df is None or trades_df.empty:
        return None

    if "return" in trades_df.columns:
        return pd.to_numeric(trades_df["return"], errors="coerce")

    if "pnl" in trades_df.columns and "entry_value" in trades_df.columns:
        denom = pd.to_numeric(trades_df["entry_value"], errors="coerce")
        pnl = pd.to_numeric(trades_df["pnl"], errors="coerce")
        return pnl / denom.replace(0, np.nan)

    if "pnl" in trades_df.columns and "entry_price" in trades_df.columns and "size" in trades_df.columns:
        denom = pd.to_numeric(trades_df["entry_price"], errors="coerce") * pd.to_numeric(trades_df["size"], errors="coerce")
        pnl = pd.to_numeric(trades_df["pnl"], errors="coerce")
        return pnl / denom.replace(0, np.nan)

    return None

def _trimmed_mean(series, trim=0.05):
    if series is None:
        return None
    s = series.dropna()
    if s.empty:
        return None
    lower = s.quantile(trim)
    upper = s.quantile(1 - trim)
    return s[(s >= lower) & (s <= upper)].mean()

def _winsorized_mean(series, trim=0.05):
    if series is None:
        return None
    s = series.dropna()
    if s.empty:
        return None
    lower = s.quantile(trim)
    upper = s.quantile(1 - trim)
    return s.clip(lower=lower, upper=upper).mean()

def _compute_trade_pnl_metrics(trades_df, trim=0.05):
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    pnl = pd.to_numeric(trades_df.get("pnl"), errors="coerce") if "pnl" in trades_df.columns else None
    ret = _get_return_series(trades_df)

    rows = []

    def add_row(name, value=None, pct=None, value_std=None, pct_std=None, n_trades=None):
        rows.append({
            "Metric": name,
            "P&L Value": value,
            "P&L %": pct,
            "Std Value": value_std,
            "Std %": pct_std,
            "n trades": n_trades,
        })

    if pnl is not None:
        add_row("Mean P&L", pnl.mean(), None, pnl.std(ddof=0), None, pnl.dropna().shape[0])
        add_row("Median P&L", pnl.median(), None, pnl.std(ddof=0), None, pnl.dropna().shape[0])
        trimmed = pnl.dropna()
        if not trimmed.empty:
            lower = trimmed.quantile(trim)
            upper = trimmed.quantile(1 - trim)
            trimmed_vals = trimmed[(trimmed >= lower) & (trimmed <= upper)]
        else:
            trimmed_vals = trimmed
        add_row(
            f"Trimmed Mean P&L ({int(trim*100)}%)",
            _trimmed_mean(pnl, trim),
            None,
            trimmed_vals.std(ddof=0) if not trimmed_vals.empty else None,
            None,
            trimmed_vals.dropna().shape[0] if trimmed_vals is not None else None
        )
        wins_vals = pnl.dropna()
        if not wins_vals.empty:
            lower = wins_vals.quantile(trim)
            upper = wins_vals.quantile(1 - trim)
            wins_vals = wins_vals.clip(lower=lower, upper=upper)
        add_row(
            f"Winsorized Mean P&L ({int(trim*100)}%)",
            _winsorized_mean(pnl, trim),
            None,
            wins_vals.std(ddof=0) if not wins_vals.empty else None,
            None,
            wins_vals.dropna().shape[0] if wins_vals is not None else None
        )
        abs_pnl = pnl.abs()
        add_row("Mean |P&L|", abs_pnl.mean(), None, abs_pnl.std(ddof=0), None, abs_pnl.dropna().shape[0])

        if "size" in trades_df.columns:
            size = pd.to_numeric(trades_df["size"], errors="coerce")
            per_unit = pnl / size.replace(0, np.nan)
            add_row("Mean P&L per Unit", per_unit.mean(), None, per_unit.std(ddof=0), None, per_unit.dropna().shape[0])

    if ret is not None:
        add_row("Mean P&L %", None, ret.mean(), None, ret.std(ddof=0), ret.dropna().shape[0])
        add_row("Median P&L %", None, ret.median(), None, ret.std(ddof=0), ret.dropna().shape[0])
        trimmed_ret = ret.dropna()
        if not trimmed_ret.empty:
            lower = trimmed_ret.quantile(trim)
            upper = trimmed_ret.quantile(1 - trim)
            trimmed_ret_vals = trimmed_ret[(trimmed_ret >= lower) & (trimmed_ret <= upper)]
        else:
            trimmed_ret_vals = trimmed_ret
        add_row(
            f"Trimmed Mean P&L % ({int(trim*100)}%)",
            None,
            _trimmed_mean(ret, trim),
            None,
            trimmed_ret_vals.std(ddof=0) if not trimmed_ret_vals.empty else None,
            trimmed_ret_vals.dropna().shape[0] if trimmed_ret_vals is not None else None
        )
        wins_ret = ret.dropna()
        if not wins_ret.empty:
            lower = wins_ret.quantile(trim)
            upper = wins_ret.quantile(1 - trim)
            wins_ret = wins_ret.clip(lower=lower, upper=upper)
        add_row(
            f"Winsorized Mean P&L % ({int(trim*100)}%)",
            None,
            _winsorized_mean(ret, trim),
            None,
            wins_ret.std(ddof=0) if not wins_ret.empty else None,
            wins_ret.dropna().shape[0] if wins_ret is not None else None
        )
        abs_ret = ret.abs()
        add_row("Mean |P&L %|", None, abs_ret.mean(), None, abs_ret.std(ddof=0), abs_ret.dropna().shape[0])

        ret_clean = ret.dropna()
        ret_clean = ret_clean[ret_clean > -1]
        if not ret_clean.empty:
            log_mean = np.log1p(ret_clean).mean()
            geo_mean = np.expm1(log_mean)
            add_row("Geometric Mean P&L %", None, geo_mean, None, None, ret_clean.dropna().shape[0])

    if not rows:
        return pd.DataFrame()

    df_metrics = pd.DataFrame(rows)
    if "P&L Value" in df_metrics.columns:
        df_metrics["P&L Value"] = df_metrics["P&L Value"].apply(
            lambda x: f"{x:,.2f}" if pd.notna(x) else ""
        )
    if "P&L %" in df_metrics.columns:
        df_metrics["P&L %"] = df_metrics["P&L %"].apply(
            lambda x: f"{x * 100:.3f}%" if pd.notna(x) else ""
        )
    if "Std Value" in df_metrics.columns:
        df_metrics["Std Value"] = df_metrics["Std Value"].apply(
            lambda x: f"{x:,.2f}" if pd.notna(x) else ""
        )
    if "Std %" in df_metrics.columns:
        df_metrics["Std %"] = df_metrics["Std %"].apply(
            lambda x: f"{x * 100:.3f}%" if pd.notna(x) else ""
        )
    if "n trades" in df_metrics.columns:
        df_metrics["n trades"] = df_metrics["n trades"].apply(
            lambda x: f"{int(x)}" if pd.notna(x) else ""
        )
    return df_metrics

def _json_safe(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date)):
        return obj.isoformat()
    return str(obj)

def _sanitize_for_json(value):
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.ndarray, pd.Timestamp, datetime.datetime, datetime.date)):
        return _json_safe(value)
    return value

def _utc_now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def _sha256_json(value):
    try:
        normalized = _sanitize_for_json(value)
        serialized = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    except Exception:
        return None

def _safe_git_command(args):
    try:
        output = subprocess.check_output(
            ["git", *args],
            cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        return output or None
    except Exception:
        return None

def _get_git_traceability_info():
    status = _safe_git_command(["status", "--porcelain"])
    return {
        "branch": _safe_git_command(["branch", "--show-current"]),
        "commit": _safe_git_command(["rev-parse", "HEAD"]),
        "commit_short": _safe_git_command(["rev-parse", "--short", "HEAD"]),
        "remote_origin": _safe_git_command(["remote", "get-url", "origin"]),
        "working_tree_dirty": bool(status) if status is not None else None
    }

def _build_traceability_payload(config_snapshot=None, results_snapshot=None, run_metadata=None):
    # Centralized audit payload used in UI and exports.
    payload = {
        "generated_at_utc": _utc_now_iso(),
        "app_name": "ATDMF Strategy Walk-Forward Optimizer",
        "config_sha256": _sha256_json(config_snapshot) if config_snapshot is not None else None,
        "results_sha256": _sha256_json(results_snapshot) if results_snapshot is not None else None,
        "git": _get_git_traceability_info(),
        "run": run_metadata or {}
    }
    return _sanitize_for_json(payload)

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

def _export_results_zip(df_mode="none", df_max_rows=200000):
    if "wfo_results" not in st.session_state:
        st.error("No results available to export.")
        return None

    results = st.session_state["wfo_results"]
    df = st.session_state.get("df")

    payload = _build_results_payload()
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("results.json", json.dumps(payload, indent=2))
        zf.writestr("audit_trace.json", json.dumps(payload.get("traceability", {}), indent=2))

        if results.get("out_of_sample_performance"):
            oos_df = pd.DataFrame(results["out_of_sample_performance"])
            zf.writestr("out_of_sample_performance.csv", oos_df.to_csv(index=False))
        if results.get("in_sample_performance"):
            is_df = pd.DataFrame(results["in_sample_performance"])
            zf.writestr("in_sample_performance.csv", is_df.to_csv(index=False))
        if results.get("best_params"):
            params_df = pd.DataFrame(results["best_params"])
            zf.writestr("best_params.csv", params_df.to_csv(index=False))

        if df is not None and not df.empty and df_mode in ("full", "downsampled"):
            df_out = df.copy()
            if df_mode == "downsampled":
                df_out = _downsample_df(df_out, max_rows=df_max_rows)
            df_out.index.name = "Open time"
            zf.writestr("df.csv", df_out.to_csv())

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
            if "df.csv" in zf.namelist():
                df = pd.read_csv(io.BytesIO(zf.read("df.csv")))
                if "Open time" in df.columns:
                    df["Open time"] = pd.to_datetime(df["Open time"], errors="coerce")
                    df.set_index("Open time", inplace=True)
                st.session_state["df"] = df

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

with st.sidebar:
    st.header("⚙️ Configuration")

    has_final_params = 'final_params' in st.session_state or 'wfo_results' in st.session_state
    st.sidebar.button(
        "📥 Load Best Params into Inputs",
        use_container_width=True,
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
            file_path = st.text_input(
                "File Path",
                value=DEFAULT_DATA_FILE,
                key='file_path',
                on_change=sync_dates_from_file,
                help="Chemin du CSV OHLCV local. Les dates peuvent être synchronisées automatiquement avec le fichier."
            )
            if not os.path.exists(file_path):
                st.error("File not found! Please check the path.")
        else:
            file_path = DEFAULT_DATA_FILE

    # Helper to create param inputs
    def param_input(key, label, default_min, default_max, default_step):
        c1, c2, c3, c4 = st.columns([0.6, 1, 1, 1])
        
        # Checkbox state handled by st.session_state via key
        # Default value is True (checked) if not in state
        
        with c1:
            enabled = st.checkbox(
                key,
                value=True,
                key=f"check_{key}",
                help=PARAMETER_HELP.get(key, "Active/désactive ce paramètre dans l'optimisation.")
            )
        
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
        return enabled, min_val, max_val, step_val

    # --- Entry Parameters ---
    with st.expander("2. Entry Parameters", expanded=False):
        st.info("Configure the search space for entry parameters.")
        
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
        st.info("Configure the search space for exit parameters.")

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
        
        opt_methods = ['grid', 'bayesian', 'optuna']
        optimization_method = st.selectbox(
            "Optimization Method",
            options=opt_methods,
            index=0,
            key='optimization_method',
            help="`grid`: exhaustif, `bayesian/optuna`: recherche probabiliste plus efficace sur grands espaces."
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

        if optimization_regime == "prev_best_grid":
            st.caption("Window 1 uses full grid; next windows reuse the previous window best parameter values as grid.")

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
            st.caption("NN-guided mode: learns from previous windows and narrows the search space for the next one.")
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
            st.caption("Adaptive Continuous: no fixed WFO windows. The grid evolves cycle after cycle from historical trials.")
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
    """Run WFO without direct UI updates (safe for background thread)."""
    try:
        if job_state is not None:
            job_state['message'] = "Loading data..."

        if config['from_file']:
            df = load_data(
                config['start_date'],
                config['end_date'],
                config['timeframe'],
                from_file=True,
                file_path=config['file_path']
            )
        else:
            df = load_data(
                config['start_date'],
                config['end_date'],
                config['timeframe'],
                from_file=False
            )

        if df is None or df.empty:
            if job_state is not None:
                job_state['error'] = "No data found for the specified range/source."
            return None, None, None

        params_grid = get_param_grid(config)
        metrics_info = get_metrics_info(config)
        wfo_settings = get_wfo_settings(config)
        regime = str(getattr(wfo_settings, 'optimization_regime', 'classic')).lower()

        if job_state is not None:
            if regime == 'adaptive_continuous':
                job_state['message'] = "Starting adaptive continuous optimization..."
            else:
                job_state['message'] = (
                    f"Starting {config['optimization_method'].upper()} on {config['n_windows']} windows..."
                )

        start_time = time.time()

        def status_callback(msg):
            if job_state is None:
                return
            if isinstance(msg, str):
                job_state['message'] = msg
            elif isinstance(msg, dict) and msg.get('type') == 'stats':
                progress = msg.get('progress')
                if progress is not None:
                    try:
                        job_state['progress'] = min(max(float(progress), 0.0), 1.0)
                    except Exception:
                        pass
                else:
                    w = msg.get('window', 0)
                    job_state['progress'] = min(w / max(1, config['n_windows']), 1.0)

                if msg.get('message'):
                    job_state['message'] = msg.get('message')
                else:
                    w = msg.get('window', 0)
                    job_state['message'] = (
                        f"Window {min(w, config['n_windows'])}/{config['n_windows']}"
                    )

        if regime == 'adaptive_continuous':
            results = adaptive_continuous_optimization(
                df,
                param_grid=params_grid,
                metrics_info=metrics_info,
                timeframe=config['timeframe'],
                settings=wfo_settings,
                status_callback=status_callback,
                control=control
            )
        else:
            results = walk_forward_optimization(
                df,
                param_grid=params_grid,
                metrics_info=metrics_info,
                timeframe=config['timeframe'],
                settings=wfo_settings,
                status_callback=status_callback,
                control=control
            )

        elapsed = time.time() - start_time
        if job_state is not None:
            job_state['progress'] = 1.0
            job_state['message'] = "Optimization complete."
        return results, df, elapsed

    except OptimizationInterrupted:
        if job_state is not None:
            job_state['message'] = "Stop requested. Optimization interrupted."
        return None, None, None
    except Exception as e:
        if job_state is not None:
            job_state['error'] = str(e)
        return None, None, None

# --- Action Buttons ---
st.sidebar.divider()

# Live Combination Count
current_conf = get_current_config()
total_combos = calculate_combinations(current_conf)
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
            use_container_width=True,
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
        if st.button("🛑 Stop WFO", use_container_width=True, help="Demande un arrêt propre après l'essai en cours."):
            control = st.session_state.get('wfo_control')
            if control:
                control.request_stop()
            if st.session_state.get('wfo_job_state') is not None:
                st.session_state['wfo_job_state']['message'] = "Stop requested. Waiting for clean shutdown..."
            st.sidebar.warning("Stop requested. Optimization is shutting down...")
            st.rerun()

if st.session_state.get('wfo_running'):
    job_state = st.session_state.get('wfo_job_state', {})
    st.sidebar.progress(float(job_state.get('progress', 0.0)))
    st.sidebar.caption(job_state.get('message', "Running..."))

with col_save:
    # Save Config Button
    json_config = json.dumps(current_conf, indent=4)
    st.download_button(
        label="💾 Save Config",
        data=json_config,
        file_name="config.json",
        mime="application/json",
        use_container_width=True,
        help="Télécharge la configuration actuelle de tous les contrôles de la sidebar."
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
    df_export_mode = st.sidebar.selectbox(
        "df.csv export",
        options=["none", "downsampled", "full"],
        index=0,
        help="Inclut les données de prix dans le ZIP. Le mode downsampled réduit la taille."
    )
    df_max_rows = 200000
    if df_export_mode == "downsampled":
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
        use_container_width=True,
        help="Crée une archive ZIP des résultats dans le dossier `reports/`."
    ):
        zip_buffer = _export_results_zip(df_mode=df_export_mode, df_max_rows=df_max_rows)
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
            use_container_width=True,
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
        use_container_width=True,
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 OOS Performance", "🔍 Parameters", "📉 Drawdowns & Returns", "📋 Raw Data", "🏆 Final Backtest"])
    
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
            
            for i, window in enumerate(results['window_results']):
                info = window['window_info']
                # IS
                if info['in_sample_start'] and info['in_sample_end']:
                    fig.add_vrect(
                        x0=info['in_sample_start'], x1=info['in_sample_end'],
                        fillcolor=colors['train'], layer="below", line_width=0,
                        annotation_text=f"W{i+1} Train" if i==0 else None
                    )
                # OOS
                if info['out_sample_start'] and info['out_sample_end']:
                    fig.add_vrect(
                        x0=info['out_sample_start'], x1=info['out_sample_end'],
                        fillcolor=colors['test'], layer="below", line_width=0,
                        annotation_text=f"W{i+1} Test" if i==0 else None
                    )
                    
            fig.update_layout(height=500, template="plotly_dark", title_text="Market Data & WFO Windows")
            st.plotly_chart(fig, use_container_width=True)
        
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
    
    with tab5:
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
                value_series = _downsample_series(value_series, max_points=max_points)

                fig_value = make_subplots(specs=[[{"secondary_y": True}]])
                fig_value.add_trace(
                    go.Scatter(x=value_series.index, y=value_series.values, mode="lines", name="Portfolio Value"),
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

                fig_value.update_layout(height=400, template="plotly_dark", title="Portfolio Value + Asset Price (Downsampled)")
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
                st.dataframe(pnl_metrics_df, use_container_width=True)

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
                        st.dataframe(pnl_metrics_df, use_container_width=True)
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
