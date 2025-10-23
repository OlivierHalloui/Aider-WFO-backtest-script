# Import necessary libraries for WFO
import pandas as pd
import numpy as np
import vectorbtpro as vbt
from itertools import product
from tqdm import tqdm
import time
from datetime import timedelta
from strategy import run_backtest
from config import WFOSettings
from skopt import gp_minimize
from skopt.space import Integer, Real
import optuna

# ======================================================================
# WALK-FORWARD OPTIMIZATION FRAMEWORK
# ======================================================================

def optimize_parameters(in_sample_df, param_grid, metrics_info, timeframe='5s', settings=None):
    """
    Optimize parameters using selected optimization method on in-sample data.
    
    Parameters:
    -----------
    in_sample_df : pandas.DataFrame
        In-sample OHLCV data
    param_grid : dict
        Parameter grid with bounds (min, max) for each parameter
    metrics_info : dict
        Metrics information dictionary
    timeframe : str, optional
        Timeframe of the data
    settings : WFOSettings, optional
        WFO settings
        
    Returns:
    --------
    pandas.DataFrame
        Sorted optimization results
    """
    if settings is None:
        settings = WFOSettings()
    
    method = settings.optimization_method.lower()
    
    if method == "grid":
        # Original grid search implementation
        param_dicts = []
        param_keys = list(param_grid.keys())
        
        for values in product(*param_grid.values()):
            param_dict = dict(zip(param_keys, values))
            param_dict.update(metrics_info)
            param_dicts.append(param_dict)
        
        results = []
        for param_dict in tqdm(param_dicts, desc="Optimizing parameters"):
            score = run_backtest(in_sample_df, param_dict, timeframe, return_portfolio=False)
            result = param_dict.copy()
            result['combined_score'] = score
            results.append(result)
        
        results_df = pd.DataFrame(results)
        sorted_results = results_df.sort_values('combined_score', ascending=False)
        
    elif method == "bayesian":
        # Bayesian Optimization implementation
        space = [
            Integer(param_grid['timeperiod'][0], param_grid['timeperiod'][1], name='timeperiod'),
            Real(param_grid['StDev'][0], param_grid['StDev'][1], name='StDev'),
            Real(param_grid['coeff_medianeBBW'][0], param_grid['coeff_medianeBBW'][1], name='coeff_medianeBBW'),
            Real(param_grid['coef_mediane'][0], param_grid['coef_mediane'][1], name='coef_mediane'),
            Integer(param_grid['fenetre_lowest'][0], param_grid['fenetre_lowest'][1], name='fenetre_lowest'),
            Real(param_grid['seuil_lowest'][0], param_grid['seuil_lowest'][1], name='seuil_lowest'),
            Integer(param_grid['longueur_mediane'][0], param_grid['longueur_mediane'][1], name='longueur_mediane'),
            Integer(param_grid['Nb_bars_above'][0], param_grid['Nb_bars_above'][1], name='Nb_bars_above'),
            Integer(param_grid['user_exit_sma_length'][0], param_grid['user_exit_sma_length'][1], name='user_exit_sma_length')
        ]
        
        def objective(params):
            param_dict = dict(zip([dim.name for dim in space], params))
            param_dict.update(metrics_info)
            score = run_backtest(in_sample_df, param_dict, timeframe, return_portfolio=False)
            return -score  # Minimize negative score
        
        n_calls = min(200, np.prod([len(values) if isinstance(values, list) else 1 for values in param_grid.values()]))
        res = gp_minimize(objective, space, n_calls=n_calls, random_state=42, n_initial_points=20)
        
        results = []
        for i, (params, score) in enumerate(zip(res.x_iters, res.func_vals)):
            param_dict = dict(zip([dim.name for dim in space], params))
            param_dict.update(metrics_info)
            param_dict['combined_score'] = -score
            results.append(param_dict)
        
        results_df = pd.DataFrame(results)
        sorted_results = results_df.sort_values('combined_score', ascending=False)
        
    elif method == "optuna":
        # Optuna (TPE) implementation
        def objective(trial):
            params = {
                'timeperiod': trial.suggest_int('timeperiod', param_grid['timeperiod'][0], param_grid['timeperiod'][1]),
                'StDev': trial.suggest_float('StDev', param_grid['StDev'][0], param_grid['StDev'][1]),
                'coeff_medianeBBW': trial.suggest_float('coeff_medianeBBW', param_grid['coeff_medianeBBW'][0], param_grid['coeff_medianeBBW'][1]),
                'coef_mediane': trial.suggest_float('coef_mediane', param_grid['coef_mediane'][0], param_grid['coef_mediane'][1]),
                'fenetre_lowest': trial.suggest_int('fenetre_lowest', param_grid['fenetre_lowest'][0], param_grid['fenetre_lowest'][1]),
                'seuil_lowest': trial.suggest_float('seuil_lowest', param_grid['seuil_lowest'][0], param_grid['seuil_lowest'][1]),
                'longueur_mediane': trial.suggest_int('longueur_mediane', param_grid['longueur_mediane'][0], param_grid['longueur_mediane'][1]),
                'Nb_bars_above': trial.suggest_int('Nb_bars_above', param_grid['Nb_bars_above'][0], param_grid['Nb_bars_above'][1]),
                'user_exit_sma_length': trial.suggest_int('user_exit_sma_length', param_grid['user_exit_sma_length'][0], param_grid['user_exit_sma_length'][1])
            }
            params.update(metrics_info)
            return run_backtest(in_sample_df, params, timeframe, return_portfolio=False)
        
        study = optuna.create_study(direction='maximize')
        n_trials = min(200, np.prod([len(values) if isinstance(values, list) else 1 for values in param_grid.values()]))
        study.optimize(objective, n_trials=n_trials)
        
        results = []
        for trial in study.trials:
            param_dict = trial.params.copy()
            param_dict.update(metrics_info)
            param_dict['combined_score'] = trial.value
            results.append(param_dict)
        
        results_df = pd.DataFrame(results)
        sorted_results = results_df.sort_values('combined_score', ascending=False)
        
    else:
        raise ValueError(f"Unsupported optimization method: {method}. Choose 'grid', 'bayesian', or 'optuna'.")
    
    return sorted_results

def walk_forward_optimization(df, param_grid=None, metrics_info=None, timeframe='5s', settings=None):
    """
    Performs Walk-Forward Optimization on the given data with timing measurements.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        OHLCV DataFrame
    param_grid : dict, optional
        Parameter grid with bounds (min, max) for each parameter
    metrics_info : dict, optional
        Metrics information dictionary
    timeframe : str, optional
        Timeframe of the data
    settings : WFOSettings, optional
        WFO settings
        
    Returns:
    --------
    dict
        Dictionary with WFO results
    """
    # Start timing
    start_time = time.time()
    window_times = []
    optimization_times = []
    
    if settings is None:
        settings = WFOSettings()
        
    if param_grid is None:
        param_grid = {
            'timeperiod': [10, 15, 20, 25, 30],
            'StDev': [0.5, 1.0, 1.5, 2.0, 2.5]
        }
        
    if metrics_info is None:
        metrics_info = {
            'metric1_name': settings.optimization_metric,
            'metric2_name': settings.secondary_metric,
            'weight_metric1': settings.metric_weights[0],
            'weight_metric2': settings.metric_weights[1]
        }
    
    # Calculate the size of each window
    total_rows = len(df)
    window_size = total_rows // settings.n_windows
    
    # Store WFO results
    wfo_results = {
        'window_results': [],
        'in_sample_performance': [],
        'out_of_sample_performance': [],
        'best_params': [],
        'settings': {
            'n_windows': settings.n_windows,
            'train_size': settings.train_size,
            'anchored': settings.anchored,
            'optimization_metric': settings.optimization_metric,
            'secondary_metric': settings.secondary_metric,
            'metric_weights': settings.metric_weights,
            'parallel_backend': settings.parallel_backend,
            'use_numba': settings.use_numba,
            'optimization_method': settings.optimization_method
        }
    }
    
    # Calculate total parameter combinations for reporting
    param_combinations = np.prod([len(values) for values in param_grid.values()])
    
    print(f"Starting Walk-Forward Optimization with {settings.n_windows} windows, {settings.train_size*100}% training size")
    print(f"WFO Type: {'Anchored' if settings.anchored else 'Unanchored'}")
    print(f"Primary Metric: {settings.optimization_metric} (weight: {settings.metric_weights[0]})")
    print(f"Secondary Metric: {settings.secondary_metric} (weight: {settings.metric_weights[1]})")
    print(f"Parallelization Backend: {settings.parallel_backend}")
    print(f"Numba Acceleration: {'Enabled' if settings.use_numba else 'Disabled'}")
    print(f"Optimization Method: {settings.optimization_method}")
    print(f"Parameter Combinations: {param_combinations}")
    
    # Loop through each window
    for i in range(settings.n_windows):
        window_start_time = time.time()
        
        start_idx = i * window_size
        end_idx = start_idx + window_size if i < settings.n_windows - 1 else total_rows
        
        window_df = df.iloc[start_idx:end_idx].copy()
        
        # For anchored WFO, always start from the first data point
        if settings.anchored:
            in_sample_start_idx = 0
        else:
            in_sample_start_idx = start_idx
            
        # Calculate in-sample end index
        in_sample_end_idx = start_idx + int(window_size * settings.train_size)
        
        # Create in-sample and out-of-sample DataFrames
        if settings.anchored:
            in_sample_df = df.iloc[in_sample_start_idx:in_sample_end_idx].copy()
        else:
            in_sample_df = window_df.iloc[:int(window_size * settings.train_size)].copy()
            
        out_sample_df = window_df.iloc[int(window_size * settings.train_size):].copy()
        
        window_dates = {
            'window': i + 1,
            'start_date': window_df.index[0],
            'end_date': window_df.index[-1],
            'in_sample_start': in_sample_df.index[0],
            'in_sample_end': in_sample_df.index[-1],
            'out_sample_start': out_sample_df.index[0] if len(out_sample_df) > 0 else None,
            'out_sample_end': out_sample_df.index[-1] if len(out_sample_df) > 0 else None
        }
        
        print(f"\nWindow {i+1}/{settings.n_windows}: {window_dates['start_date']} to {window_dates['end_date']}")
        print(f"In-Sample: {window_dates['in_sample_start']} to {window_dates['in_sample_end']}")
        if len(out_sample_df) > 0:
            print(f"Out-of-Sample: {window_dates['out_sample_start']} to {window_dates['out_sample_end']}")
        
        # Optimize parameters on in-sample data
        print(f"Optimizing parameters on in-sample data ({len(in_sample_df)} bars)...")
        
        # Time the optimization process
        optimization_start = time.time()
        optimization_results = optimize_parameters(
            in_sample_df, param_grid, metrics_info, timeframe, settings
        )
        optimization_time = time.time() - optimization_start
        optimization_times.append(optimization_time)
        
        # Get best parameters
        best_params = optimization_results.iloc[0].drop(['combined_score', metrics_info['metric1_name'], metrics_info['metric2_name']], errors='ignore').to_dict()
        
        print(f"Best parameters found: {best_params}")
        print(f"Score: {optimization_results.iloc[0]['combined_score']:.4f}")
        print(f"Optimization time: {timedelta(seconds=int(optimization_time))}")
        
        # Test best parameters on in-sample data
        print(f"Testing best parameters on in-sample data ({len(in_sample_df)} bars)...")
        in_sample_portfolio = run_backtest(in_sample_df, best_params, timeframe)
        
        in_sample_metrics = {
            'window': i + 1,
            'return': in_sample_portfolio.total_return * 100,
            'sharpe': in_sample_portfolio.sharpe_ratio,
            'max_drawdown': in_sample_portfolio.max_drawdown * 100,
            'win_rate': in_sample_portfolio.trades.win_rate,
            'calmar_ratio': in_sample_portfolio.calmar_ratio if in_sample_portfolio.max_drawdown > 0 else np.nan,
            'sortino_ratio': in_sample_portfolio.sortino_ratio,
            'n_trades': len(in_sample_portfolio.trades)
        }
        
        print(f"In-Sample Performance:")
        print(f"Return: {in_sample_metrics['return']:.2f}%")
        print(f"Sharpe Ratio: {in_sample_metrics['sharpe']:.2f}")
        print(f"Max Drawdown: {in_sample_metrics['max_drawdown']:.2f}%")
        print(f"Win Rate: {in_sample_metrics['win_rate']:.2f}%")
        print(f"Number of Trades: {in_sample_metrics['n_trades']}")
        
        wfo_results['in_sample_performance'].append(in_sample_metrics)
        
        # Test on out-of-sample data if available
        if len(out_sample_df) > 0:
            print(f"Testing best parameters on out-of-sample data ({len(out_sample_df)} bars)...")
            out_sample_portfolio = run_backtest(out_sample_df, best_params, timeframe)
            
            # Calculate performance metrics
            # try:
            #     trades_stats = out_sample_portfolio.trades.stats()
            #     win_rate = trades_stats['win_rate']
            # except:
            #     win_rate = 0.0
                
            out_sample_metrics = {
                'window': i + 1,
                'return': out_sample_portfolio.total_return * 100,
                'sharpe': out_sample_portfolio.sharpe_ratio,
                'max_drawdown': out_sample_portfolio.max_drawdown * 100,
                'win_rate': out_sample_portfolio.trades.win_rate,  #* 100,
                'calmar_ratio': out_sample_portfolio.calmar_ratio if out_sample_portfolio.max_drawdown > 0 else np.nan,
                'sortino_ratio': out_sample_portfolio.sortino_ratio,
                'n_trades': len(out_sample_portfolio.trades)
            }
            
            print(f"Out-of-Sample Performance:")
            print(f"Return: {out_sample_metrics['return']:.2f}%")
            print(f"Sharpe Ratio: {out_sample_metrics['sharpe']:.2f}")
            print(f"Max Drawdown: {out_sample_metrics['max_drawdown']:.2f}%")
            print(f"Win Rate: {out_sample_metrics['win_rate']:.2f}%")
            print(f"Number of Trades: {out_sample_metrics['n_trades']}")
            
            wfo_results['out_of_sample_performance'].append(out_sample_metrics)
        
        # Store window results
        window_result = {
            'window_info': window_dates,
            'optimization_results': optimization_results.head(5).to_dict('records'),
            'best_params': best_params
        }
        
        wfo_results['window_results'].append(window_result)
        wfo_results['best_params'].append(best_params)
        
        # Record window processing time
        window_time = time.time() - window_start_time
        window_times.append(window_time)
        print(f"Window processing time: {timedelta(seconds=int(window_time))}")
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Calculate aggregate in-sample performance
    if wfo_results['in_sample_performance']:
        is_df = pd.DataFrame(wfo_results['in_sample_performance'])
        
        print("\n=== Aggregate In-Sample Performance ===")
        print(f"Average Return: {is_df['return'].mean():.2f}%")
        print(f"Average Sharpe Ratio: {is_df['sharpe'].mean():.2f}")
        print(f"Average Max Drawdown: {is_df['max_drawdown'].mean():.2f}%")
        print(f"Average Win Rate: {is_df['win_rate'].mean():.2f}%")
        print(f"Average Calmar Ratio: {is_df['calmar_ratio'].mean():.2f}")
        print(f"Average Sortino Ratio: {is_df['sortino_ratio'].mean():.2f}")
        print(f"Total Trades: {is_df['n_trades'].sum()}")
        print(f"Cumulative Return: {((1 + is_df['return']/100).prod() - 1) * 100:.2f}%")
    
    # Calculate aggregate out-of-sample performance if available
    if wfo_results['out_of_sample_performance']:
        oos_df = pd.DataFrame(wfo_results['out_of_sample_performance'])
        
        print("\n=== Aggregate Out-of-Sample Performance ===")
        print(f"Average Return: {oos_df['return'].mean():.2f}%")
        print(f"Average Sharpe Ratio: {oos_df['sharpe'].mean():.2f}")
        print(f"Average Max Drawdown: {oos_df['max_drawdown'].mean():.2f}%")
        print(f"Average Win Rate: {oos_df['win_rate'].mean():.2f}%")
        print(f"Average Calmar Ratio: {oos_df['calmar_ratio'].mean():.2f}")
        print(f"Average Sortino Ratio: {oos_df['sortino_ratio'].mean():.2f}")
        print(f"Total Trades: {oos_df['n_trades'].sum()}")
        print(f"Cumulative Return: {((1 + oos_df['return']/100).prod() - 1) * 100:.2f}%")

        # Check for consistency in parameter selection
        params_df = pd.DataFrame(wfo_results['best_params'])
        print("\n=== Parameter Consistency Analysis ===")
        for param in param_grid.keys():
            print(f"{param}: {params_df[param].value_counts().to_dict()}")
        
        # Calculate parameter stability
        param_stability = {}
        for param in param_grid.keys():
            param_values = params_df[param].values
            param_stability[param] = 1.0 - (np.std(param_values) / np.mean(param_values)) if np.mean(param_values) > 0 else 0.0
            
        print("\n=== Parameter Stability (higher is better) ===")
        for param, stability in param_stability.items():
            print(f"{param}: {stability:.4f}")
    
    # Add timing information to results
    wfo_results['timing'] = {
        'total_time': total_time,
        'window_times': window_times,
        'optimization_times': optimization_times,
        'avg_window_time': np.mean(window_times),
        'avg_optimization_time': np.mean(optimization_times),
        'backend': settings.parallel_backend,
        'use_numba': settings.use_numba,
        'param_combinations': param_combinations
    }
    
    # Print timing summary
    print("\n=== Performance Timing Summary ===")
    print(f"Backend: {settings.parallel_backend}")
    print(f"Numba: {'Enabled' if settings.use_numba else 'Disabled'}")
    print(f"Optimization Method: {settings.optimization_method}")
    print(f"Total processing time: {timedelta(seconds=int(total_time))}")
    print(f"Average window time: {timedelta(seconds=int(np.mean(window_times)))}")
    print(f"Average optimization time: {timedelta(seconds=int(np.mean(optimization_times)))}")
    print(f"Parameter combinations per window: {param_combinations}")
    print(f"Processing speed: {param_combinations * settings.n_windows / total_time:.2f} combinations/second")
    
    return wfo_results
