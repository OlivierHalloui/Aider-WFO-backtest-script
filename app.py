import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import json
import datetime
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
from data_loading import load_data
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

with st.sidebar:
    st.header("⚙️ Configuration")
    
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
                    'exit_sar_enabled': 'exit_sar_enabled'
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

                st.success(f"Loaded config: {uploaded_config.name}")
        except Exception as e:
            st.error(f"Error loading config: {e}")
    
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
            file_path = st.text_input("File Path", value=DEFAULT_DATA_FILE, key='file_path')
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

        exit_params = ['user_exit_sma_length', 'sar_start', 'sar_increment', 'sar_maximum']
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
    df = st.session_state['df']
    config = get_current_config()

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

    def select_best_params(wfo_results):
        best_params = None
        best_score = None
        best_window = None
        best_is_metrics = None
        best_oos_metrics = None

        is_map = {row.get('window'): row for row in wfo_results.get('in_sample_performance', [])}
        oos_map = {row.get('window'): row for row in wfo_results.get('out_of_sample_performance', [])}

        for window in wfo_results.get('window_results', []):
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

    # Use the single best parameter set across all windows (by combined_score).
    chosen_params, best_score, best_window, best_is_metrics, best_oos_metrics = select_best_params(results)
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
    st.session_state['final_params_is_score'] = combined_score(best_is_metrics)
    st.session_state['final_params_oos_score'] = combined_score(best_oos_metrics)
    
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

# Run Final Backtest Button (Conditional)
if 'wfo_results' in st.session_state:
    st.sidebar.divider()
    if st.sidebar.button("🏆 Run Final Backtest", use_container_width=True):
        run_final_backtest_logic()

# ==============================================================================
# RESULTS VISUALIZATION
# ==============================================================================

if 'wfo_results' in st.session_state:
    results = st.session_state['wfo_results']
    df = st.session_state['df']
    
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
            
            if best_score is not None:
                st.markdown(f"**Best Optimization Score (combined_score):** `{best_score:.4f}`")
            if best_window is not None:
                st.markdown(f"**Best Window (WFO):** `{best_window}`")
            if best_is_score is not None:
                st.markdown(f"**IS Combined Score:** `{best_is_score:.4f}`")
            if best_oos_score is not None:
                st.markdown(f"**OOS Combined Score:** `{best_oos_score:.4f}`")
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
            # VectorBT plot is a FigureWidget, convert to compatible format or use st.plotly_chart
            # pf.plot() returns a FigureWidget. st.plotly_chart handles it.
            st.plotly_chart(pf.plot(), use_container_width=True)
            
            st.markdown("#### Trade Stats")
            st.dataframe(pf.trades.stats())
            
        else:
            st.info("Click **Run Final Backtest** in the sidebar to generate results.")
        
        # Manual re-run using a selected WFO window
        if results.get('window_results'):
            st.markdown("#### Re-run Final Backtest by WFO Window")
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
                        with st.spinner("Running Final Backtest on Full Dataset..."):
                            try:
                                selected_portfolio = run_backtest(
                                    df,
                                    selected_params,
                                    config_local.get('timeframe', DEFAULT_TIMEFRAME),
                                    return_portfolio=True
                                )
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
