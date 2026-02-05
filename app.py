import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import json
import datetime
import io
import zipfile

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
    payload = {
        "exported_at": datetime.datetime.now().isoformat(),
        "config": get_current_config(),
        "wfo_results": st.session_state.get("wfo_results"),
        "has_final_portfolio": "final_portfolio" in st.session_state,
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
            if "results.json" in zf.namelist():
                payload = json.loads(zf.read("results.json").decode("utf-8"))
                if payload.get("wfo_results"):
                    st.session_state["wfo_results"] = payload["wfo_results"]
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

with st.sidebar:
    st.header("⚙️ Configuration")

    has_final_params = 'final_params' in st.session_state or 'wfo_results' in st.session_state
    st.sidebar.button(
        "📥 Load Best Params into Inputs",
        use_container_width=True,
        on_click=load_best_params_into_inputs if has_final_params else None,
        disabled=not has_final_params
    )
    if not has_final_params:
        st.sidebar.info("Run the final backtest to enable loading best parameters.")
    
    # --- File Uploader for Config ---
    uploaded_config = st.file_uploader("📂 Load Config (JSON)", type=['json'])
    
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
                    'parallel_backend': 'parallel_backend', 'max_workers': 'max_workers',
                    'use_numba': 'use_numba', 'metric1_name': 'metric1_name', 
                    'metric2_name': 'metric2_name', 'weight_metric1': 'weight_metric1',
                    'weight_metric2': 'weight_metric2', 'patience_level': 'patience_level',
                    'max_trials': 'max_trials', 'neighbor_count': 'neighbor_count',
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
    uploaded_results = st.file_uploader("📦 Load Results (ZIP)", type=['zip'])
    if uploaded_results is not None:
        _load_results_zip(uploaded_results)
    
    # --- Data Settings ---
    with st.expander("1. Data Configuration", expanded=True):
        # NOTE: Removed 'get_conf' usage for value=. The value argument is only used for initialization
        # when key is NOT in session_state. If key IS in session_state (e.g. from loader above), 
        # Streamlit ignores value=. This allows user edits to persist.
        
        start_date = st.text_input("Start Date (YYYY-MM-DD)", value=DEFAULT_START_DATE, key='start_date')
        end_date = st.text_input("End Date (YYYY-MM-DD)", value=DEFAULT_END_DATE, key='end_date')
        
        # Timeframe selection
        tf_options = ['1s', '5s', '15s', '30s', '1m', '5m', '15m', '30m', '1h', '4h', '1d']
        default_tf_idx = tf_options.index(DEFAULT_TIMEFRAME) if DEFAULT_TIMEFRAME in tf_options else 1
        timeframe = st.selectbox("Timeframe", options=tf_options, index=default_tf_idx, key='timeframe')
        
        # Data Source
        ds_options = ["Local File", "Binance API"]
        # Default index 0 (Local File) if not in state
        data_source = st.radio("Data Source", options=ds_options, index=0, key='data_source')
        
        if data_source == "Local File":
            file_path = st.text_input(
                "File Path",
                value=DEFAULT_DATA_FILE,
                key='file_path',
                on_change=sync_dates_from_file
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
            enabled = st.checkbox(key, value=True, key=f"check_{key}")
        
        with c2:
            min_val = st.number_input(f"Min", value=float(default_min), key=f"min_{key}", disabled=not enabled)
        with c3:
            max_val = st.number_input(f"Max", value=float(default_max), key=f"max_{key}", disabled=not enabled)
        with c4:
            step_val = st.number_input(f"Step", value=float(default_step), key=f"step_{key}", disabled=not enabled)
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
        n_windows = st.number_input("Number of Windows", min_value=1, value=1, help="Number of Walk-Forward windows", key='n_windows')
        train_size = st.slider("Train Size Ratio", 0.1, 0.9, 0.5, 0.05, help="Proportion of data used for optimization vs validation", key='train_size')
        anchored = st.checkbox("Anchored WFO", value=False, help="If checked, training window grows. If unchecked, it slides.", key='anchored')
        
        opt_methods = ['grid', 'bayesian', 'optuna']
        optimization_method = st.selectbox("Optimization Method", options=opt_methods, index=0, key='optimization_method')
        
        patience_levels = ['Low', 'Medium', 'High']
        patience_level = st.selectbox("Patience Level (Bayesian/Optuna)", options=patience_levels, index=1, key='patience_level')
        
        max_trials = st.number_input("Max Trials (Bayesian/Optuna)", min_value=10, value=200, step=10, key='max_trials')
        neighbor_count = st.number_input("Stability Neighbor Count", min_value=1, value=5, step=1, key='neighbor_count')
        
        backends = ['thread', 'dask', 'ray', 'pathos']
        parallel_backend = st.selectbox("Parallel Backend", options=backends, index=0, key='parallel_backend')
        
        max_workers = st.number_input("Max Workers", min_value=1, value=os.cpu_count() or 1, key='max_workers')
        use_numba = st.checkbox("Use Numba Acceleration", value=True, key='use_numba')

    # --- Metrics ---
    with st.expander("5. Performance Metrics", expanded=False):
        metric_options = ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate', 'avg_gain_per_trade', 'avg_loss_per_trade', 'avg_pl_per_trade']
        
        m1_idx = 0 # Default sharpe
        metric1 = st.selectbox("Primary Metric", options=metric_options, index=m1_idx, key='metric1_name')
        weight1 = st.number_input("Weight 1", value=1.0, key='weight_metric1')
        
        m2_idx = 1 # Default total_return
        metric2 = st.selectbox("Secondary Metric", options=metric_options, index=m2_idx, key='metric2_name')
        weight2 = st.number_input("Weight 2", value=0.0, key='weight_metric2')

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
            format_func=lambda v: sizing_options.get(v, v)
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
        'patience_level': patience_level,
        'max_trials': max_trials,
        'neighbor_count': neighbor_count,
        'parallel_backend': parallel_backend,
        'max_workers': max_workers,
        'use_numba': use_numba
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

def run_wfo(config, control=None):
    # --- Execution ---
    try:
        with st.status("Running Optimization...", expanded=True) as status:
            st.write("⏳ Loading Data...")
            
            # Load Data
            if config['from_file']:
                df = load_data(config['start_date'], config['end_date'], config['timeframe'], from_file=True, file_path=config['file_path'])
            else:
                df = load_data(config['start_date'], config['end_date'], config['timeframe'], from_file=False)
            
            if df is None or df.empty:
                status.update(label="Error: No data loaded.", state="error")
                st.error("No data found for the specified range/source.")
                return None, None

            st.write(f"✅ Loaded {len(df)} bars of data.")
            st.session_state['opt_start_date'] = config.get('start_date')
            st.session_state['opt_end_date'] = config.get('end_date')
            
            # Prepare WFO arguments
            params_grid = get_param_grid(config)
            metrics_info = get_metrics_info(config)
            wfo_settings = get_wfo_settings(config)
            
            st.write(f"⚙️ Parameter Space: {sum(len(v) for v in params_grid.values())} raw dimensions.")
            st.write(f"🚀 Starting {config['optimization_method'].upper()} optimization on {config['n_windows']} windows...")
            
            # Run WFO
            start_time = time.time()
            
            # Create a progress placeholder
            progress_bar = st.progress(0)
            
            # Define a simple callback to update status
            def status_callback(msg):
                if isinstance(msg, str):
                    pass
                elif isinstance(msg, dict) and msg.get('type') == 'stats':
                    w = msg.get('window', 0)
                    progress_bar.progress(min(w / max(1, config['n_windows']), 1.0))
            
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
            st.write(f"✅ Optimization completed in {elapsed:.2f} seconds.")
            status.update(label="Optimization Complete!", state="complete")
            
            return results, df

    except OptimizationInterrupted:
        st.warning("Optimization cancelled by user.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred during optimization: {str(e)}")
        # st.exception(e) # Uncomment for debug stack trace
        return None, None

# --- Action Buttons ---
st.sidebar.divider()

# Live Combination Count
current_conf = get_current_config()
total_combos = calculate_combinations(current_conf)
st.sidebar.info(f"📊 Total Parameter Combinations: **{total_combos:,}**")

col_run, col_save = st.sidebar.columns([1, 1])

with col_run:
    if st.button("🚀 Start WFO", type="primary", use_container_width=True):
        if not selected_params:
            st.error("Select params!")
        else:
            st.session_state['wfo_control'] = WFOControl()
            st.session_state['wfo_running'] = True
            results, df = run_wfo(current_conf, control=st.session_state['wfo_control'])
            st.session_state['wfo_running'] = False
            if results:
                st.session_state['wfo_results'] = results
                st.session_state['df'] = df
                st.success("Finished!")

# Cancel button for running optimization
if st.session_state.get('wfo_running'):
    if st.sidebar.button("🛑 Cancel WFO", use_container_width=True):
        control = st.session_state.get('wfo_control')
        if control:
            control.request_stop()
        st.sidebar.warning("Cancel requested. Stopping after current step...")

with col_save:
    # Save Config Button
    json_config = json.dumps(current_conf, indent=4)
    st.download_button(
        label="💾 Save Config",
        data=json_config,
        file_name="config.json",
        mime="application/json",
        use_container_width=True
    )

st.sidebar.divider()
st.sidebar.subheader("📤 Export Results")
if "wfo_results" in st.session_state:
    df_export_mode = st.sidebar.selectbox(
        "df.csv export",
        options=["none", "downsampled", "full"],
        index=0,
        help="Include price data in the ZIP. Downsampled reduces size."
    )
    df_max_rows = 200000
    if df_export_mode == "downsampled":
        df_max_rows = st.sidebar.slider(
            "Max rows for df.csv",
            min_value=10000,
            max_value=1000000,
            value=200000,
            step=10000,
            help="Approximate maximum rows to keep in df.csv."
        )
    if st.sidebar.button("💾 Save Results to Disk", use_container_width=True):
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
            use_container_width=True
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
        key="final_start_date"
    )
    final_end_date = st.sidebar.text_input(
        "Final End Date (YYYY-MM-DD)",
        value=default_final_end,
        key="final_end_date"
    )
    final_file_path = st.sidebar.text_input(
        "Final Data File Path",
        value=st.session_state.get('file_path', DEFAULT_DATA_FILE),
        key="final_file_path"
    )
    if st.sidebar.button("🏆 Run Final Backtest", use_container_width=True):
        run_final_backtest_logic()

# ==============================================================================
# RESULTS VISUALIZATION
# ==============================================================================

if 'wfo_results' in st.session_state:
    results = st.session_state['wfo_results']
    df = st.session_state.get('df')
    
    st.divider()
    st.header("📊 Optimization Results")
    
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
                help="Downsample large series to avoid Streamlit message size limits."
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
                help="Percent trimmed/winsorized from each tail."
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
                        help="Percent trimmed/winsorized from each tail."
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
                        selected_entry = None
                        for window in results['window_results']:
                            if window.get('window_info', {}).get('window') == selected_window:
                                selected_entry = window
                                break

                        if selected_entry:
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
                                        st.session_state['final_params_window'] = selected_window
                                        st.success("Final Backtest Complete!")
                                    except Exception as e:
                                        st.error(f"Error in final backtest: {e}")

elif not os.path.exists(DEFAULT_DATA_FILE):
    st.warning(f"⚠️ Default data file not found at: `{DEFAULT_DATA_FILE}`. Please configure the data source in the sidebar.")
else:
    st.info("👈 Click **Start Optimization** in the sidebar to run the backtest.")
