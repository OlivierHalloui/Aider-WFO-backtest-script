# Import necessary libraries for WFO
import pandas as pd
import numpy as np
import vectorbtpro as vbt
from itertools import product
from tqdm import tqdm
import time
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from strategy import run_backtest
from config import WFOSettings
import optuna

try:
    import dask
    from dask.distributed import Client, as_completed as dask_as_completed
except ImportError:
    dask = None
try:
    import ray
except ImportError:
    ray = None

class OptimizationInterrupted(Exception):
    """Raised when the optimization process is interrupted by the user."""
    pass


backtest_cache = {}
backtest_cache_lock = threading.Lock()

# ======================================================================
# WALK-FORWARD OPTIMIZATION FRAMEWORK
# ======================================================================

def optimize_parameters(in_sample_df, param_grid, metrics_info, timeframe='5s', settings=None, control=None):
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
    tuple[pandas.DataFrame, int]
        Sorted optimization results and number of evaluations performed
    """
    if settings is None:
        settings = WFOSettings()
    
    # Validate and initialize parallel backend
    if settings.parallel_backend.lower() == 'dask':
        if dask is None:
            raise ImportError("Dask not installed. Install with 'pip install dask distributed'.")
        client = Client(processes=False, threads_per_worker=1, n_workers=settings.max_workers or 1)
        executor_class = 'dask'
    elif settings.parallel_backend.lower() == 'ray':
        if ray is None:
            raise ImportError("Ray not installed. Install with 'pip install ray'.")
        ray.init(num_cpus=settings.max_workers or 1)
        executor_class = 'ray'
    else:
        executor_class = 'thread'  # Default to threads
    
    # Validate inputs
    if not isinstance(param_grid, dict) or not param_grid:
        raise ValueError("param_grid must be a non-empty dict.")
    for key, bounds in param_grid.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) < 2:
            raise ValueError(f"param_grid['{key}'] must be a list/tuple with at least min and max.")
        if not all(isinstance(b, (int, float)) for b in bounds[:2]):
            raise ValueError(f"param_grid['{key}'] bounds must be numeric.")
    if not isinstance(in_sample_df, pd.DataFrame) or in_sample_df.empty:
        raise ValueError("in_sample_df must be a non-empty pandas DataFrame.")
    if not hasattr(settings, 'optimization_method') or settings.optimization_method.lower() not in ['grid', 'bayesian', 'optuna']:
        raise ValueError("settings.optimization_method must be 'grid', 'bayesian', or 'optuna'.")
    
    method = settings.optimization_method.lower()
    data_signature = (
        in_sample_df.index[0] if len(in_sample_df) > 0 else None,
        in_sample_df.index[-1] if len(in_sample_df) > 0 else None,
        len(in_sample_df)
    )
    
    def evaluate_params(param_dict):
        """(Change 6) Evaluate run_backtest with caching, logging, and interruption control."""
        if control:
            control.wait_if_paused()
            if control.should_stop():
                raise OptimizationInterrupted()
        cache_key = (data_signature, tuple(sorted(param_dict.items())))
        with backtest_cache_lock:
            if cache_key in backtest_cache:
                return backtest_cache[cache_key]
        start = time.time()
        score = run_backtest(in_sample_df, param_dict, timeframe, return_portfolio=False)
        elapsed = time.time() - start
        print(f"Evaluation time: {elapsed:.2f}s for params: {param_dict}")
        with backtest_cache_lock:
            backtest_cache[cache_key] = score
        return score
    
    if method == "grid":
        param_dicts = []
        param_keys = list(param_grid.keys())
        
        for values in product(*param_grid.values()):
            param_dict = dict(zip(param_keys, values))
            param_dict.update(metrics_info)
            param_dicts.append(param_dict)
        
        # Batch param_dicts if chunk_size is set
        if hasattr(settings, 'chunk_size') and settings.chunk_size > 0:
            param_dicts = [param_dicts[i:i + settings.chunk_size] for i in range(0, len(param_dicts), settings.chunk_size)]
        
        results = []

        max_workers = settings.max_workers or 1
        if max_workers > 1 and executor_class != 'thread':
            if executor_class == 'dask':
                futures = [client.submit(evaluate_params, pdict) for pdict in param_dicts]
                for future in tqdm(dask_as_completed(futures), total=len(param_dicts), desc="Optimizing parameters"):
                    base = future_to_params[future].copy()  # Note: Adjust future_to_params to use futures as keys
                    base['combined_score'] = future.result()
                    results.append(base)
            elif executor_class == 'ray':
                @ray.remote
                def remote_evaluate(pdict):
                    return evaluate_params(pdict)
                futures = [remote_evaluate.remote(pdict) for pdict in param_dicts]
                for future in tqdm(ray.get(futures), total=len(param_dicts), desc="Optimizing parameters"):
                    # Assuming futures are in order; adjust if needed
                    base = param_dicts[len(results)].copy()
                    base['combined_score'] = future
                    results.append(base)
        else:
            # Fallback to threads or sequential
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_params = {executor.submit(evaluate_params, pdict): pdict for pdict in param_dicts}
                try:
                    for future in tqdm(as_completed(future_to_params), total=len(param_dicts), desc="Optimizing parameters"):
                        base = future_to_params[future].copy()
                        base['combined_score'] = future.result()
                        results.append(base)
                except OptimizationInterrupted:
                    for future in future_to_params:
                        future.cancel()
                    raise
        
        results_df = pd.DataFrame(results)
        sorted_results = results_df.sort_values('combined_score', ascending=False)
        evaluation_count = len(results_df)
        
        if results_df.empty:
            raise ValueError("No optimization results found. Check param_grid or data.")
        
    elif method == "bayesian":
        def extract_bounds(bounds):
            if isinstance(bounds, (list, tuple)):
                if len(bounds) == 3 and all(isinstance(b, (int, float)) for b in bounds[:2]):
                    return bounds
                if len(bounds) >= 2 and all(isinstance(b, (int, float)) for b in bounds[:2]):
                    step = max(abs(bounds[1] - bounds[0]), 1)
                    return bounds[0], bounds[-1], step
            raise ValueError(f"Unsupported bounds format for Bayesian optimization: {bounds}")
        
        continuous_ranges = []
        for bounds in param_grid.values():
            min_val, max_val, _ = extract_bounds(bounds)
            continuous_ranges.append(max(max_val - min_val, 1e-3))
        estimated_space_size = min(1000, max(1, int(np.prod(continuous_ranges))))
        total_combinations = estimated_space_size
        
        n_calls = min(500, max(100, total_combinations))
        n_initial_points = min(50, max(10, int(0.1 * n_calls)))
        patience = max(1, int(0.2 * n_calls))
        
        class NoImprovementStopper:
            def __init__(self, patience_steps):
                self.patience_steps = patience_steps
                self.best_value = np.inf
                self.no_improve_steps = 0
            
            def __call__(self, study, trial):
                value = trial.value
                if value < self.best_value - 1e-9:
                    self.best_value = value
                    self.no_improve_steps = 0
                else:
                    self.no_improve_steps += 1
                if self.no_improve_steps >= self.patience_steps:
                    study.stop()
        
        int_params = {'timeperiod', 'fenetre_lowest', 'longueur_mediane', 'Nb_bars_above', 'user_exit_sma_length'}
        sampler = optuna.samplers.TPESampler(
            multivariate=True,
            constant_liar=True,
            n_startup_trials=n_initial_points,
            seed=getattr(settings, 'random_state', 42)
        )
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=n_initial_points)
        study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
        n_jobs = min(4, max(1, settings.max_workers or 1))
        
        if executor_class in ['dask', 'ray']:
            n_jobs = 1  # Let backend handle parallelism
        
        def objective(trial):
            if control:
                control.wait_if_paused()
                if control.should_stop():
                    raise OptimizationInterrupted()
            params = {}
            for param_name, bounds in param_grid.items():
                min_val, max_val, step = extract_bounds(bounds)
                if param_name in int_params:
                    params[param_name] = trial.suggest_int(param_name, int(min_val), int(max_val), step=int(max(step, 1)))
                else:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        float(min_val),
                        float(max_val)
                    )
            params.update(metrics_info)
            score = evaluate_params(params)
            noisy_value = -score + np.random.normal(0, 1e-6)
            return noisy_value
        
        early_stopper = NoImprovementStopper(patience)
        try:
            study.optimize(
                objective,
                n_trials=n_calls,
                n_jobs=n_jobs,
                callbacks=[early_stopper]
            )
        except OptimizationInterrupted:
            raise
        
        results = []
        for trial in study.trials:
            param_dict = trial.params.copy()
            param_dict.update(metrics_info)
            param_dict['combined_score'] = -trial.value
            results.append(param_dict)
        
        if not results:
            raise ValueError("Bayesian optimization produced no results. Check bounds or trials.")
        
        results_df = pd.DataFrame(results)
        sorted_results = results_df.sort_values('combined_score', ascending=False)
        evaluation_count = len(study.trials)
        
        if results_df.empty:
            raise ValueError("No optimization results found. Check param_grid or data.")
        
    elif method == "optuna":
        # Optuna (TPE) implementation - dynamic based on param_grid
        def objective(trial):
            if control:
                control.wait_if_paused()
                if control.should_stop():
                    raise OptimizationInterrupted()
            params = {}
            for param_name, bounds in param_grid.items():
                if param_name in ['timeperiod', 'fenetre_lowest', 'longueur_mediane', 'Nb_bars_above', 'user_exit_sma_length']:
                    params[param_name] = trial.suggest_int(param_name, bounds[0], bounds[1])
                else:
                    params[param_name] = trial.suggest_float(param_name, bounds[0], bounds[1])
            params.update(metrics_info)
            return evaluate_params(params)
        
        study = optuna.create_study(direction='maximize')
        n_trials = min(200, np.prod([len(values) if isinstance(values, list) else 1 for values in param_grid.values()]))
        try:
            study.optimize(objective, n_trials=n_trials)
        except OptimizationInterrupted:
            raise
        
        results = []
        for trial in study.trials:
            param_dict = trial.params.copy()
            param_dict.update(metrics_info)
            param_dict['combined_score'] = trial.value
            results.append(param_dict)
        
        if not results:
            raise ValueError("Optuna optimization produced no results. Check bounds or trials.")
        
        results_df = pd.DataFrame(results)
        sorted_results = results_df.sort_values('combined_score', ascending=False)
        evaluation_count = len(study.trials)
        
        if results_df.empty:
            raise ValueError("No optimization results found. Check param_grid or data.")
        
    else:
        raise ValueError(f"Unsupported optimization method: {method}. Choose 'grid', 'bayesian', or 'optuna'.")
    
    return sorted_results, evaluation_count

def walk_forward_optimization(df, param_grid=None, metrics_info=None, timeframe='5s', settings=None, status_callback=None, control=None):
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
    
    # Validate inputs
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("df must be a non-empty pandas DataFrame.")
    if settings.n_windows < 1:
        raise ValueError("settings.n_windows must be at least 1.")
    if not (0 < settings.train_size <= 1):
        raise ValueError("settings.train_size must be between 0 and 1.")
        
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
    
    def log(message: str):
        print(message)
        if status_callback:
            status_callback(message)
    
    def report_stats(payload: dict):
        if status_callback:
            status_callback(payload)
    
    log(f"Starting Walk-Forward Optimization with {settings.n_windows} windows, {settings.train_size*100}% training size")
    log(f"WFO Type: {'Anchored' if settings.anchored else 'Unanchored'}")
    log(f"Primary Metric: {settings.optimization_metric} (weight: {settings.metric_weights[0]})")
    log(f"Secondary Metric: {settings.secondary_metric} (weight: {settings.metric_weights[1]})")
    log(f"Parallelization Backend: {settings.parallel_backend}")
    log(f"Numba Acceleration: {'Enabled' if settings.use_numba else 'Disabled'}")
    log(f"Optimization Method: {settings.optimization_method}")
    log(f"Parameter Combinations: {param_combinations}")
    
    # Loop through each window
    for i in range(settings.n_windows):
        if control:
            control.wait_if_paused(log)
            if control.should_stop():
                log("Stop requested before processing the next window. Exiting.")
                raise OptimizationInterrupted()

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
        
        log(f"\nWindow {i+1}/{settings.n_windows}: {window_dates['start_date']} to {window_dates['end_date']}")
        log(f"In-Sample: {window_dates['in_sample_start']} to {window_dates['in_sample_end']}")
        if len(out_sample_df) > 0:
            log(f"Out-of-Sample: {window_dates['out_sample_start']} to {window_dates['out_sample_end']}")
        
        # Optimize parameters on in-sample data
        log(f"Optimizing parameters on in-sample data ({len(in_sample_df)} bars)...")
        
        # Time the optimization process
        optimization_start = time.time()
        try:
            optimization_results, eval_count = optimize_parameters(
                in_sample_df, param_grid, metrics_info, timeframe, settings, control=control
            )
        except OptimizationInterrupted:
            log("Optimization interrupted during parameter search.")
            raise
        optimization_time = time.time() - optimization_start
        optimization_times.append(optimization_time)
        
        # Get best parameters
        best_params = optimization_results.iloc[0].drop(['combined_score', metrics_info['metric1_name'], metrics_info['metric2_name']], errors='ignore').to_dict()
        
        log(f"Best parameters found: {best_params}")
        log(f"Score: {optimization_results.iloc[0]['combined_score']:.4f}")
        log(f"Optimization time: {timedelta(seconds=int(optimization_time))}")
        
        # Test best parameters on in-sample data
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
        
        wfo_results['in_sample_performance'].append(in_sample_metrics)
        
        # Test on out-of-sample data if available
        out_sample_metrics = None
        if len(out_sample_df) > 0:
            log(f"Testing best parameters on out-of-sample data ({len(out_sample_df)} bars)...")
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
            
            log(f"Out-of-Sample Performance:")
            log(f"Return: {out_sample_metrics['return']:.2f}%")
            log(f"Sharpe Ratio: {out_sample_metrics['sharpe']:.2f}")
            log(f"Max Drawdown: {out_sample_metrics['max_drawdown']:.2f}%")
            log(f"Win Rate: {out_sample_metrics['win_rate']:.2f}%")
            log(f"Number of Trades: {out_sample_metrics['n_trades']}")
            
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
        log(f"Window processing time: {timedelta(seconds=int(window_time))}")
        
        # Progress metrics for GUI
        combinations_tested = max(1, eval_count)
        combos_per_sec = combinations_tested / optimization_time if optimization_time > 0 else 0.0
        windows_completed = i + 1
        remaining_windows = settings.n_windows - windows_completed
        avg_window_time = np.mean(window_times)
        eta_seconds = avg_window_time * remaining_windows if avg_window_time and remaining_windows > 0 else 0.0
        
        report_payload = {
            'type': 'stats',
            'speed': combos_per_sec,
            'eta': eta_seconds,
            'window': windows_completed,
            'evaluations': combinations_tested,
            'window_metrics': {
                'in_sample': in_sample_metrics,
                'out_sample': out_sample_metrics
            }
        }
        report_stats(report_payload)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Calculate aggregate in-sample performance
    if wfo_results['in_sample_performance']:
        is_df = pd.DataFrame(wfo_results['in_sample_performance'])
        
        log("\n=== Aggregate In-Sample Performance ===")
        log(f"Average Return: {is_df['return'].mean():.2f}%")
        log(f"Average Sharpe Ratio: {is_df['sharpe'].mean():.2f}")
        log(f"Average Max Drawdown: {is_df['max_drawdown'].mean():.2f}%")
        log(f"Average Win Rate: {is_df['win_rate'].mean():.2f}%")
        log(f"Average Calmar Ratio: {is_df['calmar_ratio'].mean():.2f}")
        log(f"Average Sortino Ratio: {is_df['sortino_ratio'].mean():.2f}")
        log(f"Total Trades: {is_df['n_trades'].sum()}")
        log(f"Cumulative Return: {((1 + is_df['return']/100).prod() - 1) * 100:.2f}%")
    
    # Calculate aggregate out-of-sample performance if available
    if wfo_results['out_of_sample_performance']:
        oos_df = pd.DataFrame(wfo_results['out_of_sample_performance'])
        
        log("\n=== Aggregate Out-of-Sample Performance ===")
        log(f"Average Return: {oos_df['return'].mean():.2f}%")
        log(f"Average Sharpe Ratio: {oos_df['sharpe'].mean():.2f}")
        log(f"Average Max Drawdown: {oos_df['max_drawdown'].mean():.2f}%")
        log(f"Average Win Rate: {oos_df['win_rate'].mean():.2f}%")
        log(f"Average Calmar Ratio: {oos_df['calmar_ratio'].mean():.2f}")
        log(f"Average Sortino Ratio: {oos_df['sortino_ratio'].mean():.2f}")
        log(f"Total Trades: {oos_df['n_trades'].sum()}")
        log(f"Cumulative Return: {((1 + oos_df['return']/100).prod() - 1) * 100:.2f}%")

        # Check for consistency in parameter selection
        params_df = pd.DataFrame(wfo_results['best_params'])
        log("\n=== Parameter Consistency Analysis ===")
        for param in param_grid.keys():
            log(f"{param}: {params_df[param].value_counts().to_dict()}")
        
        # Calculate parameter stability
        param_stability = {}
        for param in param_grid.keys():
            param_values = params_df[param].values
            param_stability[param] = 1.0 - (np.std(param_values) / np.mean(param_values)) if np.mean(param_values) > 0 else 0.0
            
        log("\n=== Parameter Stability (higher is better) ===")
        for param, stability in param_stability.items():
            log(f"{param}: {stability:.4f}")
    
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
    log("\n=== Performance Timing Summary ===")
    log(f"Backend: {settings.parallel_backend}")
    log(f"Numba: {'Enabled' if settings.use_numba else 'Disabled'}")
    log(f"Optimization Method: {settings.optimization_method}")
    log(f"Total processing time: {timedelta(seconds=int(total_time))}")
    log(f"Average window time: {timedelta(seconds=int(np.mean(window_times)))}")
    log(f"Average optimization time: {timedelta(seconds=int(np.mean(optimization_times)))}")
    log(f"Parameter combinations per window: {param_combinations}")
    processing_speed = (param_combinations * settings.n_windows / total_time) if total_time > 0 else 0.0
    log(f"Processing speed: {processing_speed:.2f} combinations/second")
    
    return wfo_results
