# Import necessary libraries for main script
import pandas as pd
import numpy as np
import datetime
import os
from typing import Optional, Dict
from config import (
    DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_TIMEFRAME, DEFAULT_DATA_FILE,
    DEFAULT_PARAM_GRID, WFOSettings
)
from data_loading import get_dates, load_data
from strategy import run_backtest
from wfo import walk_forward_optimization
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

def get_param_grid():
    """
    Get parameter grid from user input.
    
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
        'user_exit_sma_length': "Longueur SMA sortie"
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
        'user_exit_sma_length': (10, 30, 10)
    }
    
    # Get user input for parameters to optimize
    if input("\nUse default parameter grid? (y/n) [default: y]: ").lower() != 'n':
        return build_default_param_grid()

    print("\nSelect parameters to optimize:")
    for i, (param, desc) in enumerate(param_options.items(), 1):
        print(f"{i}. {param} - {desc}")
        min_val, max_val, step = default_ranges[param]
        print(f"   Default range: {min_val} to {max_val}, step: {step}")
    
    selected = input("Enter parameter numbers separated by commas (e.g., 1,2): ")
    selected_indices = [int(idx) for idx in selected.split(',') if idx.strip().isdigit()]
    
    if not selected_indices:
        print("No valid parameters selected. Using default grid.")
        return build_default_param_grid()

    param_keys = list(param_options.keys())
    selected_params = [param_keys[idx-1] for idx in selected_indices if 1 <= idx <= len(param_keys)]
    
    param_grid = {}
    for param in selected_params:
        if param in default_ranges:
            default_min, default_max, default_step = default_ranges[param]
            print(f"\nCustomize range for {param}:")
            try:
                min_val = float(input(f"Minimum value [default: {default_min}]: ") or default_min)
                max_val = float(input(f"Maximum value [default: {default_max}]: ") or default_max)
                step = float(input(f"Step size [default: {default_step}]: ") or default_step)
            except ValueError:
                print(f"Invalid input. Using default values for {param}.")
                min_val, max_val, step = default_min, default_max, default_step
            if param in ['timeperiod', 'fenetre_lowest', 'user_exit_sma_length']:
                param_grid[param] = list(range(int(min_val), int(max_val) + 1, int(step)))
            else:
                param_grid[param] = list(np.round(np.arange(min_val, max_val + step, step), 1))
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

def get_metrics_info():
    """
    Get metrics information from user input.
    
    Returns:
    --------
    dict
        Metrics information for optimization
    """
    # Define available metrics
    available_metrics = ['max_drawdown', 'sharpe_ratio', 'total_return', 'avg_gain_per_trade', 'avg_loss_per_trade', 'win_rate','avg_pl_per_trade']
    
    print("\nSelect metrics to optimize:")
    for i, metric in enumerate(available_metrics, 1):
        print(f"{i}. {metric}")
    
    try:
        metric1_idx = int(input("Enter number for first metric: ")) - 1
        metric2_idx = int(input("Enter number for second metric: ")) - 1
        
        if not (0 <= metric1_idx < len(available_metrics) and 0 <= metric2_idx < len(available_metrics)):
            raise ValueError
            
        metric1_name = available_metrics[metric1_idx]
        metric2_name = available_metrics[metric2_idx]
        
        weight_metric1 = float(input(f"Enter weight for {metric1_name} (0-1): "))
        weight_metric1 = max(0, min(1, weight_metric1))  # Clamp between 0 and 1
        weight_metric2 = 1 - weight_metric1
    except (ValueError, IndexError):
        print("Invalid metrics selected. Using total_return and max_drawdown as defaults.")
        metric1_name = 'total_return'
        metric2_name = 'max_drawdown'
        weight_metric1 = 1.0
        weight_metric2 = 0.0
    
    return {
        'metric1_name': metric1_name,
        'metric2_name': metric2_name,
        'weight_metric1': weight_metric1,
        'weight_metric2': weight_metric2
    }

def get_wfo_settings():
    """
    Get Walk-Forward Optimization settings from user input.
    
    Returns:
    --------
    WFOSettings
        WFO settings object
    """
    settings = WFOSettings()
    
    print("\nConfigure Walk-Forward Optimization settings:")
    
    # Number of windows
    try:
        settings.n_windows = int(input("Number of windows to divide data into [default: 1]: ") or 1)
    except ValueError:
        settings.n_windows = 1
    
    # Training size
    try:
        settings.train_size = float(input("Training size as proportion (0-1) [default: 0.5]: ") or 0.5)
        settings.train_size = max(0.1, min(0.9, settings.train_size))  # Clamp between 0.1 and 0.9
    except ValueError:
        settings.train_size = 0.5
    
    # Anchored or unanchored
    anchored_input = input("Use anchored WFO (fixed start date) or unanchored (rolling window)? (a/u) [default: u]: ").lower()
    settings.anchored = anchored_input == 'a'
    
    # Parallelization backend
    print("\nSelect parallelization backend:")
    print("1. threadpool - Best for Numba functions")
    print("2. dask - Best for CPU-bound tasks (default)")
    print("3. ray - Best for distributed computing")
    print("4. pathos - Alternative for multiprocessing")
    
    backend_choice = input("Enter choice (1-4) [default: 2]: ") or "2"
    backends = {
        "1": "threadpool",
        "2": "dask",
        "3": "ray",
        "4": "pathos"
    }
    settings.parallel_backend = backends.get(backend_choice, "threadpool")

    # Max workers
    try:
        max_workers_input = input(f"Max workers [default: all cores ({os.cpu_count() or 1})]: ").strip()
        if max_workers_input:
            settings.max_workers = max(1, int(max_workers_input))
        else:
            settings.max_workers = os.cpu_count() or 1
    except ValueError:
        settings.max_workers = os.cpu_count() or 1

    # Numba acceleration
    numba_choice = input("Use Numba acceleration? (y/n) [default: y]: ").lower()
    settings.use_numba = numba_choice != 'n'
    
    return settings

def main():
    """Main function to run the WFO process."""
    print("=== ATDMF Strategy Walk-Forward Optimization ===")
    print("This script performs Walk-Forward Optimization on the ATDMF strategy.")
    print("It demonstrates how to properly cross-validate trading strategies to avoid overfitting.")
    
    # Get dates
    start_date, end_date = get_dates()
    
    # Load data
    print("\nLoading data...")
    timeframe = '5s'
    
    # Ask user whether to load from file or fetch from Binance
    from_file = input("\nLoad data from file? (y/n) [default: y]: ").lower() != 'n'
    
    if from_file:
        file_path = input("Enter path to CSV file [default: use built-in dataset]: ") or DEFAULT_DATA_FILE
        df = load_data(start_date, end_date, timeframe, from_file=True, file_path=file_path)
    else:
        df = load_data(start_date, end_date, timeframe, from_file=False)
    
    print(f"Loaded {len(df)} bars of data from {start_date} to {end_date}")
    
    # Display default parameters
    display_default_parameters()

    # Get parameter grid
    param_grid = get_param_grid()
    print(f"Parameter grid: {param_grid}")
    
    # Get metrics information
    metrics_info = get_metrics_info()
    print(f"Metrics: {metrics_info['metric1_name']} (weight: {metrics_info['weight_metric1']:.2f}), "
          f"{metrics_info['metric2_name']} (weight: {metrics_info['weight_metric2']:.2f})")
    
    # Get WFO settings
    settings = get_wfo_settings()
    
    # Run Walk-Forward Optimization
    wfo_results = walk_forward_optimization(
        df, 
        param_grid=param_grid,
        metrics_info=metrics_info,
        timeframe=timeframe,
        settings=settings
    )
    
    # Save results to CSV
    results_dir = "WFO_Results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save out-of-sample metrics if available
    if wfo_results['out_of_sample_performance']:
        oos_df = pd.DataFrame(wfo_results['out_of_sample_performance'])
        oos_df.to_csv(f"{results_dir}/oos_performance_{timestamp}.csv", index=False)
    
    # Save best parameters for each window
    params_df = pd.DataFrame(wfo_results['best_params'])
    params_df.to_csv(f"{results_dir}/best_parameters_{timestamp}.csv", index=False)
    
    # Save WFO settings
    with open(f"{results_dir}/wfo_settings_{timestamp}.txt", 'w') as f:
        for key, value in wfo_results['settings'].items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nResults saved to {results_dir} directory")
    
    # Visualize results
    visualize = input("\nVisualize results? (y/n) [default: y]: ").lower() != 'n'
    if visualize:
        # Register DataFrame with OHLC accessor if needed
        try:
            from vectorbtpro.data.base import OHLCV
            df = OHLCV.from_df(df)
        except (ImportError, ValueError) as e:
            # If registration fails, continue with regular DataFrame
            print(f"Note: Could not register DataFrame with OHLCV: {e}")
       
       # Core WFO visualizations
        visualize_wfo_results(wfo_results, df)
        
        
        # Parameter performance analysis
        create_parameter_performance_map(wfo_results)
        
        # Robustness analysis
        visualize_robustness_metrics(wfo_results)

    # Generate comprehensive PDF report
    integrate_report_generation(wfo_results, df, strategy_name="ATDMF Strategy")

    # Ask if user wants to run a final backtest with best parameters
    run_final = input("\nRun a final backtest with averaged best parameters? (y/n): ")
    if run_final.lower() == 'y':
        # Average the parameters from all windows
        avg_params = {}
        for param in param_grid.keys():
            if param in ['timeperiod', 'fenetre_lowest', 'user_exit_sma_length']:
                avg_params[param] = int(round(params_df[param].mean()))
            else:
                avg_params[param] = round(params_df[param].mean(), 2)
        
        print(f"Running final backtest with parameters: {avg_params}")
        final_portfolio = run_backtest(df, avg_params, timeframe)
        
        # Calculate average P&L per trade
        avg_pl_per_trade = 0
        if len(final_portfolio.trades) > 0:
            avg_pl_per_trade = (final_portfolio.total_return*100) / len(final_portfolio.trades)
        
        print("\n=== Final Backtest Results ===")
        print(f"Total Return: {final_portfolio.total_return * 100:.2f}%")
        print(f"Sharpe Ratio: {final_portfolio.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {final_portfolio.max_drawdown * 100:.2f}%")
        print(f"Win Rate: {final_portfolio.trades.win_rate *100:.2f}%")
        print(f"Average P&L per Trade: {avg_pl_per_trade:.2f}%")
        print(f"Number of Trades: {len(final_portfolio.trades)}")
        print(f"Calmar Ratio: {final_portfolio.calmar_ratio:.2f}")
        print(f"Sortino Ratio: {final_portfolio.sortino_ratio:.2f}")
        
        # Plot performance
        final_portfolio.plot().show()

if __name__ == "__main__":
    main()
