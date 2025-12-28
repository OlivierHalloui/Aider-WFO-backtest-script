# Import necessary libraries for main script

import pandas as pd
import numpy as np
import datetime
import os
import traceback
import time
import sys
from typing import Optional, Dict, Callable, Any
from config import (
    DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_TIMEFRAME, DEFAULT_DATA_FILE,
    DEFAULT_PARAM_GRID, WFOSettings
)
from data_loading import get_dates, load_data
from strategy import run_backtest
from wfo import walk_forward_optimization, OptimizationInterrupted
from visualization import (
    visualize_wfo_results, create_parameter_performance_map, visualize_robustness_metrics,
    integrate_report_generation
)

# ======================================================================
# MAIN PROGRAM
# ======================================================================

def build_default_param_grid(selected_params: Optional[list] = None) -> Dict[str, list]:
    if not selected_params:
        selected_params = list(DEFAULT_PARAM_GRID.keys())
    return {param: list(DEFAULT_PARAM_GRID[param]) for param in selected_params if param in DEFAULT_PARAM_GRID}

def get_param_grid(config):
    """
    Get parameter grid from config dict.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    dict
        Parameter grid for optimization
    """
    # Define parameter options
    param_options = {
        'timeperiod': "Période pour les Bandes de Bollinger",
        'StDev': "Nombre d'écarts-types pour les Bandes de Bollinger",
        'coeff_medianeBBW': "Coefficient utilisé dans le signal horizontal",
        'coef_mediane': "Coefficient pour le signal écart BB borné",
        'fenetre_lowest': "Fenêtre pour le signal lowest",
        'seuil_lowest': "Seuil pour le signal lowest",
        'longueur_mediane': "Fenêtre médiane du signal BB borné",
        'Nb_bars_above': "Nbre de barres pour la validation du signal BB borné",
        'user_exit_sma_length': "Longueur SMA sortie",
        'sar_start': "Parabolic SAR start",
        'sar_increment': "Parabolic SAR increment",
        'sar_maximum': "Parabolic SAR maximum"
    }
    
    # ATDMF Strategy default parameters (Hardcoded Source of Truth)
    strategy_params = {
        'timeperiod': 20,
        'StDev': 2.0,
        'matype': 0,
        'coeff_medianeBBW': 1.1,
        'coef_mediane': 1.0,
        'Nb_bars_above' : 5,
        'fenetre_lowest': 30,
        'seuil_lowest': 3.5,
        'longueur_mediane': 100,
        'user_exit_sma_length': 20,
        'sar_start': 0.02,
        'sar_increment': 0.02,
        'sar_maximum': 0.2
    }
    
    # Default parameter ranges
    default_ranges = {
        'timeperiod': (10, 30, 5),
        'StDev': (1, 2.5, 0.5),
        'coeff_medianeBBW': (0.8, 1.6, 0.4),
        'coef_mediane': (1, 1.5, 0.5),
        'fenetre_lowest': (30, 60, 10),
        'seuil_lowest': (1.0, 3.5, 0.5),
        'longueur_mediane': (50, 150, 50),
        'Nb_bars_above': (2, 6, 2),
        'user_exit_sma_length': (10, 30, 10),
        'sar_start': (0.02, 0.05, 0.01),
        'sar_increment': (0.02, 0.05, 0.01),
        'sar_maximum': (0.1, 0.3, 0.05)
    }
    
    # Get selected parameters from config
    selected_params = config.get('selected_params', [])
    if not selected_params:
         # Fallback if list is empty (shouldn't happen usually if app.py works right, but good for safety)
         selected_params = list(default_ranges.keys())

    param_grid = {}
    
    # 1. Add Selected Parameters (Ranges)
    for param in selected_params:
        if param in default_ranges:
            default_min, default_max, default_step = default_ranges[param]
            min_val = config.get(f'{param}_min', default_min)
            max_val = config.get(f'{param}_max', default_max)
            step = config.get(f'{param}_step', default_step)
            if param in ['timeperiod', 'fenetre_lowest', 'user_exit_sma_length']:
                param_grid[param] = list(range(int(min_val), int(max_val) + 1, int(step)))
            else:
                decimals = max(0, int(round(-np.log10(step)))) if step > 0 else 0
                values = np.arange(min_val, max_val + step, step)
                param_grid[param] = list(np.round(values, decimals))
    
    # 2. Add Unselected Parameters (Fixed Defaults)
    # Iterate over all known strategy parameters. If not in param_grid, add default as single value.
    for param, default_val in strategy_params.items():
        if param not in param_grid and param in default_ranges: # Only consider params that are optimizable (in ranges)
             param_grid[param] = [default_val]

    # Exit toggles (fixed)
    param_grid['exit_sar_enabled'] = [bool(config.get('exit_sar_enabled', True))]

    return param_grid


def display_default_parameters():
    """
    Display all default indicator parameters and their values.
    """
    print("\n=== Default Indicator Parameters ===")
    
    # ATDMF Strategy parameters
    strategy_params = {
        # Bollinger Bands parameters
        'timeperiod': 20,
        'StDev': 2.0,
        'matype': 0,  # Simple Moving Average
        
        # Custom signal parameters
        'coeff_medianeBBW': 1.1,
        'coef_mediane': 1.0,
        'Nb_bars_above' : 5,
        'fenetre_lowest': 30,
        'seuil_lowest': 3.5,
        'longueur_mediane': 100,
        
        # Exit parameters
        'user_exit_sma_length': 20
    }
    
    # Parameter descriptions
    param_descriptions = {
        'timeperiod': 'Bollinger Bands period (lookback window)',
        'StDev': 'Number of standard deviations for Bollinger Bands',
        'matype': 'Moving average type (0=SMA, 1=EMA, 2=WMA, 3=DEMA, etc.)',
        'coeff_medianeBBW': 'Coefficient for Bollinger Bandwidth median in horizontal signal',
        'coef_mediane': 'Coefficient for median in ecart bollinger signal',
        'Nb_bars_above': 'Nb of bars "coef_mediane" crossed under',
        'fenetre_lowest': 'Window size for lowest BBW signal',
        'seuil_lowest': 'Threshold for lowest BBW signal',
        'longueur_mediane': 'Window size for median calculation in ecart bollinger signal',
        'user_exit_sma_length': 'Moving average period for exit signals'
    }
    
    # Display parameters with descriptions and values
    max_param_len = max(len(param) for param in strategy_params.keys())
    max_desc_len = max(len(desc) for desc in param_descriptions.values())
    
    print(f"{'Parameter':<{max_param_len+2}} | {'Description':<{max_desc_len+2}} | {'Default Value'}")
    print("-" * (max_param_len+2 + max_desc_len+2 + 20))
    
    for param, value in strategy_params.items():
        if param in param_descriptions:
            print(f"{param:<{max_param_len+2}} | {param_descriptions[param]:<{max_desc_len+2}} | {value}")
        else:
            print(f"{param:<{max_param_len+2}} | {'No description available':<{max_desc_len+2}} | {value}")



def get_metrics_info(config):
    """
    Get metrics information from config dict.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    dict
        Metrics information for optimization
    """
    # Define available metrics
    available_metrics = ['max_drawdown', 'sharpe_ratio', 'total_return', 'avg_gain_per_trade', 'avg_loss_per_trade', 'win_rate','avg_pl_per_trade']
    
    metric1_name = config.get('metric1_name', 'sharpe_ratio')
    metric2_name = config.get('metric2_name', 'total_return')
    
    weight_metric1 = config.get('weight_metric1', 1.0)
    weight_metric2 = config.get('weight_metric2', 0.0)
    
    return {
        'metric1_name': metric1_name,
        'metric2_name': metric2_name,
        'weight_metric1': weight_metric1,
        'weight_metric2': weight_metric2
    }

def get_wfo_settings(config):
    """
    Get Walk-Forward Optimization settings from config dict.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    WFOSettings
        WFO settings object
    """
    settings = WFOSettings()
    
    settings.n_windows = config.get('n_windows', 1)
    settings.train_size = config.get('train_size', 0.5)
    settings.anchored = config.get('anchored', False)
    settings.optimization_method = config.get('optimization_method', 'bayesian')
    settings.parallel_backend = config.get('parallel_backend', 'dask')
    settings.max_workers = config.get('max_workers', os.cpu_count() or 1)
    settings.use_numba = config.get('use_numba', True)
    settings.patience_level = config.get('patience_level', 'Medium')
    settings.max_trials = config.get('max_trials', 200)
    settings.neighbor_count = config.get('neighbor_count', 5)
    settings.exit_sar_enabled = config.get('exit_sar_enabled', True)
    
    return settings

def run_optimization(config, status_callback: Optional[Callable[[Any], None]] = None, control: Optional[Any] = None):
    """
    Run the optimization process using config dict.
    """
    def log(message: str):
        print(message)
        if status_callback:
            status_callback(message)

    log("Preparing date range...")
    start_date, end_date = get_dates(config)
    
    log("\nLoading data...")
    timeframe = config.get('timeframe', DEFAULT_TIMEFRAME)
    
    from_file = config.get('from_file', True)
    file_path = config.get('file_path', DEFAULT_DATA_FILE)
    
    if from_file:
        df = load_data(start_date, end_date, timeframe, from_file=True, file_path=file_path)
    else:
        df = load_data(start_date, end_date, timeframe, from_file=False)
    
    log(f"Loaded {len(df)} bars of data from {start_date} to {end_date}")
    
    display_default_parameters()

    param_grid = get_param_grid(config)
    log(f"Parameter grid: {param_grid}")
    
    metrics_info = get_metrics_info(config)
    log(
        f"Metrics: {metrics_info['metric1_name']} (weight: {metrics_info['weight_metric1']:.2f}), "
        f"{metrics_info['metric2_name']} (weight: {metrics_info['weight_metric2']:.2f})"
    )
    
    settings = get_wfo_settings(config)
    
    log("Starting walk-forward optimization...")
    wfo_results = walk_forward_optimization(
        df, 
        param_grid=param_grid,
        metrics_info=metrics_info,
        timeframe=timeframe,
        settings=settings,
        status_callback=status_callback,
        control=control
    )
    log("Walk-forward optimization completed.")
    
    results_dir = "WFO_Results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if wfo_results['out_of_sample_performance']:
        oos_df = pd.DataFrame(wfo_results['out_of_sample_performance'])
        oos_df.to_csv(f"{results_dir}/oos_performance_{timestamp}.csv", index=False)
    
    params_df = pd.DataFrame(wfo_results['best_params'])
    params_df.to_csv(f"{results_dir}/best_parameters_{timestamp}.csv", index=False)
    
    with open(f"{results_dir}/wfo_settings_{timestamp}.txt", 'w') as f:
        for key, value in wfo_results['settings'].items():
            f.write(f"{key}: {value}\n")
    
    log(f"\nResults saved to {results_dir} directory")
    
    visualize = config.get('visualize', True)
    if visualize:
        try:
            from vectorbtpro.data.base import OHLCV
            df = OHLCV.from_df(df)
        except (ImportError, ValueError) as e:
            log(f"Note: Could not register DataFrame with OHLCV: {e}")
       
        visualize_wfo_results(wfo_results, df)
        create_parameter_performance_map(wfo_results)
        visualize_robustness_metrics(wfo_results)

    if config.get('generate_report', True):
        log("Generating PDF report...")
        integrate_report_generation(wfo_results, df, strategy_name="ATDMF Strategy")
    else:
        log("Skipping PDF report generation as requested.")

    run_final = config.get('run_final', False)
    if run_final:
        avg_params = {}
        for param in param_grid.keys():
            if param in ['timeperiod', 'fenetre_lowest', 'user_exit_sma_length']:
                avg_params[param] = int(round(params_df[param].mean()))
            else:
                avg_params[param] = round(params_df[param].mean(), 2)
        
        log(f"Running final backtest with parameters: {avg_params}")
        final_portfolio = run_backtest(df, avg_params, timeframe)
        
        avg_pl_per_trade = 0
        if len(final_portfolio.trades) > 0:
            avg_pl_per_trade = (final_portfolio.total_return*100) / len(final_portfolio.trades)
        
        log("\n=== Final Backtest Results ===")
        log(f"Total Return: {final_portfolio.total_return * 100:.2f}%")
        log(f"Sharpe Ratio: {final_portfolio.sharpe_ratio:.2f}")
        log(f"Max Drawdown: {final_portfolio.max_drawdown * 100:.2f}%")
        log(f"Win Rate: {final_portfolio.trades.win_rate *100:.2f}%")
        log(f"Average P&L per Trade: {avg_pl_per_trade:.2f}%")
        log(f"Number of Trades: {len(final_portfolio.trades)}")
        log(f"Calmar Ratio: {final_portfolio.calmar_ratio:.2f}")
        log(f"Sortino Ratio: {final_portfolio.sortino_ratio:.2f}")
        
        final_portfolio.plot().show()

def main():
    """Main function to run the WFO process."""
    print("=== ATDMF Strategy Walk-Forward Optimization ===")
    print("This script performs Walk-Forward Optimization on the ATDMF strategy.")
    
    if '--config' in sys.argv:
        import json
        try:
            config_idx = sys.argv.index('--config') + 1
            config_path = sys.argv[config_idx]
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Loading configuration from {config_path}...")
            run_optimization(config)
        except Exception as e:
            print(f"Error loading config: {e}")
            traceback.print_exc()
    elif '--no-gui' in sys.argv:
        # Console mode
        config = {
            'start_date': DEFAULT_START_DATE,
            'end_date': DEFAULT_END_DATE,
            'timeframe': DEFAULT_TIMEFRAME,
            'from_file': True,
            'file_path': DEFAULT_DATA_FILE,
            'optimization_method': 'grid',
            'n_windows': 1,
            'train_size': 0.5,
            'visualize': False,
            'generate_report': False
        }
        run_optimization(config)
    else:
        # GUI mode
        print("Launching Streamlit GUI...")
        import subprocess
        try:
            subprocess.run(["streamlit", "run", "app.py"], check=True)
        except FileNotFoundError:
            print("Error: 'streamlit' command not found. Please install it using 'pip install streamlit'.")
        except KeyboardInterrupt:
            print("\nStreamlit GUI stopped.")

if __name__ == "__main__":
    main()
