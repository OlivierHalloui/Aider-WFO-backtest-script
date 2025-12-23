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
import plotly.figure_factory as ff
import vectorbtpro as vbt

# Import from existing modules
from config import (
    DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_TIMEFRAME, DEFAULT_DATA_FILE, DEFAULT_PARAM_GRID,
    WFOSettings
)
from main import get_param_grid
from wfo_v2 import run_sota_wfo
from strategy_v2 import run_backtest_v2
from data_loading import load_data
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
                    'max_trials': 'max_trials'
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

    # --- Strategy Parameters ---
    with st.expander("2. Strategy Parameters", expanded=False):
        st.info("Configure the search space for each parameter.")
        
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

        # Generate inputs for all params in DEFAULT_PARAM_GRID
        config_params = {}
        selected_params = []
        
        for param, (d_min, d_max, d_step) in DEFAULT_PARAM_GRID.items():
            enabled, p_min, p_max, p_step = param_input(param, param, d_min, d_max, d_step)
            if enabled:
                selected_params.append(param)
                config_params[f'{param}_min'] = p_min
                config_params[f'{param}_max'] = p_max
                config_params[f'{param}_step'] = p_step

    # --- WFO Settings ---
    with st.expander("3. WFO Engine Settings", expanded=False):
        n_windows = st.number_input("Number of Windows", min_value=1, value=1, help="Number of Walk-Forward windows", key='n_windows')
        train_size = st.slider("Train Size Ratio", 0.1, 0.9, 0.5, 0.05, help="Proportion of data used for optimization vs validation", key='train_size')
        anchored = st.checkbox("Anchored WFO", value=False, help="If checked, training window grows. If unchecked, it slides.", key='anchored')
        
        opt_methods = ['grid', 'bayesian', 'optuna']
        optimization_method = st.selectbox("Optimization Method", options=opt_methods, index=0, key='optimization_method')
        
        patience_levels = ['Low', 'Medium', 'High']
        patience_level = st.selectbox("Patience Level (Bayesian/Optuna)", options=patience_levels, index=1, key='patience_level')
        
        max_trials = st.number_input("Max Trials (Bayesian/Optuna)", min_value=10, value=200, step=10, key='max_trials')
        
        backends = ['thread', 'dask', 'ray', 'pathos']
        parallel_backend = st.selectbox("Parallel Backend", options=backends, index=0, key='parallel_backend')
        
        max_workers = st.number_input("Max Workers", min_value=1, value=os.cpu_count() or 1, key='max_workers')
        use_numba = st.checkbox("Use Numba Acceleration", value=True, key='use_numba')

    # --- Metrics ---
    with st.expander("4. Performance Metrics", expanded=False):
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
        'n_windows': n_windows,
        'train_size': train_size,
        'anchored': anchored,
        'optimization_method': optimization_method,
        'patience_level': patience_level,
        'max_trials': max_trials,
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
    
    # Calculate average parameters
    params_df = pd.DataFrame(results['best_params'])
    avg_params = {}
    
    # Define integer parameters that should be rounded
    int_params = ['timeperiod', 'fenetre_lowest', 'longueur_mediane', 'Nb_bars_above', 'user_exit_sma_length']
    
    for param in params_df.columns:
        if param in ['window', 'metric1_name', 'metric2_name']:
            continue
        try:
            mean_val = params_df[param].mean()
            if param in int_params:
                avg_params[param] = int(round(mean_val))
            else:
                avg_params[param] = round(mean_val, 2)
        except:
            pass # Skip non-numeric
            
    st.session_state['final_params'] = avg_params
    
    with st.spinner("Running Final Backtest on Full Dataset..."):
        try:
            final_portfolio = run_backtest(df, avg_params, config['timeframe'], return_portfolio=True)
            st.session_state['final_portfolio'] = final_portfolio
            st.success("Final Backtest Complete!")
        except Exception as e:
            st.error(f"Error in final backtest: {e}")

def run_final_backtest_logic_v2():
    """Runs the final backtest using averaged parameters from SOTA WFO results."""
    if 'cv_results' not in st.session_state or 'df' not in st.session_state:
        st.error("No SOTA WFO results available to run final backtest.")
        return

    cv_results = st.session_state['cv_results']
    df = st.session_state['df']
    config = get_current_config()
    
    # Extract parameters from cv_results
    all_outputs = cv_results.out
    best_params_list = [output[1] for output in all_outputs]
    params_df = pd.DataFrame(best_params_list)
    
    avg_params = {}
    
    # Define integer parameters that should be rounded
    int_params = ['timeperiod', 'fenetre_lowest', 'longueur_mediane', 'Nb_bars_above', 'user_exit_sma_length']
    
    for param in params_df.columns:
        if param in ['window', 'metric1_name', 'metric2_name']:
            continue
        try:
            mean_val = params_df[param].mean()
            if param in int_params:
                avg_params[param] = int(round(mean_val))
            else:
                avg_params[param] = round(mean_val, 2)
        except:
            pass # Skip non-numeric
            
    st.session_state['final_params_v2'] = avg_params
    
    with st.spinner("Running Final Backtest on Full Dataset (SOTA Params)..."):
        try:
            final_portfolio = run_backtest_v2(df, avg_params, freq=config['timeframe'])
            st.session_state['final_portfolio_v2'] = final_portfolio
            st.success("Final Backtest Complete!")
        except Exception as e:
            st.error(f"Error in final backtest: {e}")

def create_robustness_radar(oos_df, params_df):
    """Calculates robustness metrics and generates a radar chart."""
    
    scores = {}
    
    # 1. Performance Consistency (lower std dev of returns is better)
    if oos_df['return'].mean() != 0 and oos_df['return'].std() > 0:
        perf_cv = oos_df['return'].std() / abs(oos_df['return'].mean())
        scores['Perf. Consistency'] = max(0, 1 - perf_cv)
    else:
        scores['Perf. Consistency'] = 0.5 # Neutral score

    # 2. Profitability (based on avg sharpe)
    # Normalize sharpe ratio. Assume a typical range of -1 to 3.
    sharpe_score = (oos_df['sharpe'].mean() - (-1)) / (3 - (-1))
    scores['Profitability'] = max(0, min(1, sharpe_score))

    # 3. Drawdown Robustness (lower avg drawdown is better)
    # Normalize drawdown. Assume a typical range of 5% to 50%.
    dd_score = 1 - ((oos_df['max_drawdown'].mean() - 5) / (50 - 5))
    scores['DD Robustness'] = max(0, min(1, dd_score))

    # 4. Parameter Stability
    numeric_params = params_df.select_dtypes(include=np.number).columns
    numeric_params = [c for c in numeric_params if 'window' not in c]
    if numeric_params:
        param_stabilities = []
        for col in numeric_params:
            if params_df[col].mean() != 0 and params_df[col].std() > 0:
                cv = params_df[col].std() / abs(params_df[col].mean())
                param_stabilities.append(max(0, 1 - cv))
            else:
                param_stabilities.append(0.5)
        scores['Param. Stability'] = np.mean(param_stabilities)
    else:
        scores['Param. Stability'] = 0.5 # Neutral if no numeric params

    # 5. Overfitting Meter (IS vs OOS performance)
    if 'wfo_results' in st.session_state and st.session_state['wfo_results']['in_sample_performance']:
        is_df = pd.DataFrame(st.session_state['wfo_results']['in_sample_performance'])
        is_return_avg = is_df['return'].mean()
        oos_return_avg = oos_df['return'].mean()
        if is_return_avg > 0:
            overfit_ratio = (is_return_avg - oos_return_avg) / is_return_avg
            scores['Overfit Resistance'] = max(0, 1 - overfit_ratio)
        else:
            scores['Overfit Resistance'] = 0.5 # Neutral if IS return is not positive
    else:
        scores['Overfit Resistance'] = 0.5

    # Calculate overall score
    overall_score = np.mean(list(scores.values()))
    
    # Create Radar Chart
    categories = list(scores.keys())
    values = list(scores.values())
    
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Robustness'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        template='plotly_dark',
        title='Strategy Robustness Dashboard'
    )
    
    return fig, overall_score



# ==============================================================================
# MAIN LOGIC
# ==============================================================================

# ==============================================================================

# MAIN LOGIC

# ==============================================================================



def run_wfo_v2(config):

    """

    Wrapper function to execute the SOTA WFO and handle results.

    """

    try:

        with st.status("Running State-of-the-Art Walk-Forward Optimization...", expanded=True) as status:

            st.write("⏳ Loading Data...")

            if config['from_file']:

                df = load_data(config['start_date'], config['end_date'], config['timeframe'], from_file=True, file_path=config['file_path'])

            else:

                df = load_data(config['start_date'], config['end_date'], config['timeframe'], from_file=False)

            

            if df is None or df.empty:

                status.update(label="Error: No data loaded.", state="error")

                st.error("No data found for the specified range/source.")

                return None, None

            st.write(f"✅ Loaded {len(df)} bars of data.")



            params_grid = get_param_grid(config)

            st.write(f"⚙️ Parameter Space: {sum(len(v) for v in params_grid.values())} raw dimensions.")

            st.write(f"🚀 Starting SOTA WFO on {config['n_windows']} windows...")
            
            # Note: The new WFO function prints its own progress to the console
            cv_results = run_sota_wfo(
                price_data=df,
                param_grid=params_grid,
                n_windows=config['n_windows'],
                train_size=config['train_size'],
                use_anchored=config['anchored'],
                metric=config['metric1_name'],
                optimization_method=config['optimization_method'],
                max_trials=config['max_trials']
            )
            
            status.update(label="SOTA WFO Complete!", state="complete")

            return cv_results, df



    except Exception as e:

        st.error(f"An error occurred during SOTA WFO: {str(e)}")

        # st.exception(e) # Uncomment for debug stack trace

        return None, None



# --- Action Buttons ---

st.sidebar.divider()

st.sidebar.markdown("### 🚀 SOTA WFO Execution")



# Live Combination Count

current_conf = get_current_config()

total_combos = calculate_combinations(current_conf)

st.sidebar.info(f"📊 Grid Search Combinations: **{total_combos:,}**")



col_run, col_save = st.sidebar.columns([1, 1])



with col_run:

    if st.button("🚀 Start SOTA WFO", type="primary", use_container_width=True):

        if not selected_params:

            st.error("Select at least one parameter to optimize!")

        else:

            cv_results, df = run_wfo_v2(current_conf)

            if cv_results:

                st.session_state['cv_results'] = cv_results

                st.session_state['df'] = df

                st.success("SOTA WFO Finished!")



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



# ==============================================================================

# RESULTS VISUALIZATION (V2)

# ==============================================================================



if 'cv_results' in st.session_state:



    cv_results = st.session_state['cv_results']



    price_df = st.session_state['df']



    



    # --- Proactive Fix: Manually parse results for robustness ---



    # The decorated WFO function returns a tuple: (oos_pf, best_params, in_sample_perf)



    # We manually unpack these results from the `out` attribute of the CrossValidator object.



    # This is more robust than relying on helper methods like .get_oos_portfolios().



    all_outputs = cv_results.out



    oos_portfolios = [output[0] for output in all_outputs]



    best_params_list = [output[1] for output in all_outputs]







    # Create the stitched portfolio and the DataFrame of best parameters
    stitched_pf = vbt.Portfolio.row_stack(oos_portfolios)
    best_params_per_fold = pd.DataFrame(best_params_list)



    best_params_per_fold.index.name = 'Split'



    



    st.divider()



    st.header("📊 SOTA WFO Results")



    



    # --- Main Stitched Portfolio Stats ---



    st.subheader("Stitched Out-of-Sample Performance")



    stats = stitched_pf.stats()



    col1, col2, col3, col4 = st.columns(4)



    col1.metric("Total Return", f"{stats['Total Return [%]']:.2f}%")



    col2.metric("Sharpe Ratio", f"{stats['Sharpe Ratio']:.2f}")



    col3.metric("Max Drawdown", f"{stats['Max Drawdown [%]']:.2f}%")



    col4.metric("Win Rate", f"{stats['Win Rate [%]']:.2f}%")







    # --- Tabs for different views ---



    tab1, tab2, tab3 = st.tabs(["📈 Overall Performance", "🔍 Window Analysis", "📋 Trade Details"])







    with tab1:

        st.subheader("Stitched Equity Curve & Drawdowns")

        st.plotly_chart(stitched_pf.plot(subplots=['cum_returns', 'drawdowns']), use_container_width=True)

        st.subheader("Full Period Performance Stats")

        # Convert stats to string to avoid PyArrow serialization errors with Timedelta objects
        st.dataframe(stats.astype(str))







    with tab2:



        st.subheader("Out-of-Sample Window Performance")







        # Create a dataframe of performance metrics for each OOS window



        window_metrics = []



        for i, pf in enumerate(oos_portfolios):



            s = pf.stats()



            metrics = {



                'Window': i,



                'Return %': s['Total Return [%]'],



                'Sharpe Ratio': s['Sharpe Ratio'],



                'Max DD %': s['Max Drawdown [%]'],



                'Win Rate %': s['Win Rate [%]'],



                '# Trades': s['Total Trades'],



            }



            window_metrics.append(metrics)



        window_metrics_df = pd.DataFrame(window_metrics)







        st.dataframe(window_metrics_df)







        st.subheader("Best Parameters per Window")



        st.dataframe(best_params_per_fold)



        



        st.subheader("Parameter Stability")



        # Check if there are any parameters to plot



        if not best_params_per_fold.empty:



            fig_line = go.Figure()



            for param_name in best_params_per_fold.columns:



                # Ensure data is numeric before plotting



                if pd.api.types.is_numeric_dtype(best_params_per_fold[param_name]):



                    fig_line.add_trace(go.Scatter(



                        x=best_params_per_fold.index,



                        y=best_params_per_fold[param_name],



                        mode='lines+markers',



                        name=param_name



                    ))



            fig_line.update_layout(



                title="Optimal Parameter Evolution Across Windows",



                xaxis_title="Window",



                yaxis_title="Parameter Value",



                template="plotly_dark",



                height=500



            )



            st.plotly_chart(fig_line, use_container_width=True)



        else:



            st.warning("No parameter data to display.")







    with tab3:

        st.subheader("All Stitched Trades")

        st.dataframe(stitched_pf.trades.records_readable.astype(str))

    # ==============================================================================
    # FINAL BACKTEST (WHOLE DATASET)
    # ==============================================================================
    st.divider()
    st.header("🏁 Final Backtest (Whole Dataset)")
    st.info("Run a single backtest on the entire dataset using the average best parameters found during WFO.")
    
    if st.button("▶️ Run Final Backtest", type="primary"):
        run_final_backtest_logic_v2()

    if 'final_portfolio_v2' in st.session_state:
        fpf = st.session_state['final_portfolio_v2']
        fparams = st.session_state.get('final_params_v2', {})
        
        st.success("Final Backtest Completed!")
        st.write("### 🧠 Average Parameters Used:")
        st.json(fparams)
        
        fstats = fpf.stats()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final Return", f"{fstats['Total Return [%]']:.2f}%")
        c2.metric("Final Sharpe", f"{fstats['Sharpe Ratio']:.2f}")
        c3.metric("Final Max DD", f"{fstats['Max Drawdown [%]']:.2f}%")
        c4.metric("Final Win Rate", f"{fstats['Win Rate [%]']:.2f}%")
        
        st.subheader("Final Equity Curve")
        st.plotly_chart(fpf.plot(subplots=['cum_returns', 'drawdowns']), use_container_width=True)

elif not os.path.exists(DEFAULT_DATA_FILE):

    st.warning(f"⚠️ Default data file not found at: `{DEFAULT_DATA_FILE}`. Please configure the data source in the sidebar.")

else:

    st.info("👈 Click **Start SOTA WFO** in the sidebar to run the new backtest.")
