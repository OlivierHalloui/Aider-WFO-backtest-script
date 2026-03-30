# Import necessary libraries for WFO
import logging
import pandas as pd
import numpy as np
import vectorbtpro as vbt
from itertools import product, islice
import time
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from scipy.spatial import KDTree
from config import WFOSettings
from strategy_adapters import resolve_strategy_adapter
from strategy import clear_window_indicator_cache
from metrics import trade_stat, calc_avg_pl, calc_pqs, safe_float, _get_trades_stats
from neural_search import NeuralSearchGuide, _match_prev_value_to_candidates, _build_prev_best_grid, _safe_float
import optuna

logger = logging.getLogger(__name__)

class OptimizationInterrupted(Exception):
    """Raised when the optimization process is interrupted by the user."""
    pass


# ======================================================================
# GRID EVALUATION HELPERS
# ======================================================================

def _normalize_scores(raw, n_expected):
    """Flatten *raw* backtest output to a list of *n_expected* floats."""
    if np.isscalar(raw):
        return [float(raw)] * n_expected
    if hasattr(raw, 'values'):
        arr = np.asarray(raw.values, dtype=float).reshape(-1)
    else:
        arr = np.asarray(raw, dtype=float).reshape(-1)
    return list(arr)


def _eval_chunk_vectorized(combos, param_keys, metrics_info,
                           in_sample_df, strategy_adapter, timeframe):
    """Attempt a single vectorized backtest for *combos*.

    Returns a list of floats (one per combo) or raises on failure.
    """
    transposed = list(zip(*combos))
    vparams = {key: np.array(transposed[k]) for k, key in enumerate(param_keys)}
    vparams.update(metrics_info)
    raw = strategy_adapter.run_backtest(
        in_sample_df, vparams, timeframe=timeframe, return_portfolio=False
    )
    scores = _normalize_scores(raw, len(combos))
    if len(scores) != len(combos):
        raise ValueError(
            f"Score/combo length mismatch: {len(scores)} scores for {len(combos)} combos"
        )
    return scores


def _eval_chunk_bisect(combos, param_keys, metrics_info,
                       in_sample_df, strategy_adapter, timeframe,
                       evaluate_params_fn, depth=0):
    """Evaluate *combos* vectorized; bisect recursively on failure.

    For a chunk of N combos where only K fail (K << N), this reduces
    sequential per-combo fallback from O(N) to O(K · log N) instead of
    O(N). For K=1, N=1000: ~10 recursive calls vs 1000 sequential calls.

    Returns a flat list of float scores aligned with *combos*.
    """
    if not combos:
        return []

    # --- Happy path: try vectorized on the full (sub-)chunk ---
    try:
        return _eval_chunk_vectorized(
            combos, param_keys, metrics_info,
            in_sample_df, strategy_adapter, timeframe,
        )
    except Exception as e:
        # --- Base case: single combo must be evaluated sequentially ---
        if len(combos) == 1:
            combo_params = dict(zip(param_keys, combos[0]))
            combo_params.update(metrics_info)
            try:
                raw = evaluate_params_fn(combo_params)
                scores = _normalize_scores(raw, 1)
                return [scores[0] if scores else float('nan')]
            except Exception as e2:
                logger.debug("Single-combo eval failed: %s — params: %s", e2,
                             combo_params)
                return [float('nan')]

        # --- Recursive bisection: split and retry each half ---
        if depth == 0:
            logger.warning(
                "Vectorized chunk (%d combos) failed — bisecting to isolate bad combos. "
                "Error: %s", len(combos), e,
            )
        mid = len(combos) // 2
        left  = _eval_chunk_bisect(combos[:mid], param_keys, metrics_info,
                                   in_sample_df, strategy_adapter, timeframe,
                                   evaluate_params_fn, depth + 1)
        right = _eval_chunk_bisect(combos[mid:], param_keys, metrics_info,
                                   in_sample_df, strategy_adapter, timeframe,
                                   evaluate_params_fn, depth + 1)
        return left + right


# ======================================================================
# WALK-FORWARD OPTIMIZATION FRAMEWORK
# ======================================================================

def optimize_parameters(
    in_sample_df,
    param_grid,
    metrics_info,
    timeframe='5s',
    settings=None,
    control=None,
    strategy_adapter=None,
    cache=None,
    cache_lock=None,
):
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
    cache : dict, optional
        Backtest result cache scoped to the current WFO run
    cache_lock : threading.Lock, optional
        Lock protecting *cache*

    Returns:
    --------
    tuple[pandas.DataFrame, int]
        Sorted optimization results and number of evaluations performed
    """
    if settings is None:
        settings = WFOSettings()
    if strategy_adapter is None:
        strategy_adapter = resolve_strategy_adapter(strategy_mode='native_atdmf')
    if cache is None:
        cache = {}
    if cache_lock is None:
        cache_lock = threading.Lock()

    method = settings.optimization_method.lower()

    # Validate inputs
    if not isinstance(param_grid, dict) or not param_grid:
        raise ValueError("param_grid must be a non-empty dict.")
    for key, bounds in param_grid.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) < 1: # Modified check
             raise ValueError(f"param_grid['{key}'] must be a list/tuple.")
    if not isinstance(in_sample_df, pd.DataFrame) or in_sample_df.empty:
        raise ValueError("in_sample_df must be a non-empty pandas DataFrame.")
    if not hasattr(settings, 'optimization_method') or settings.optimization_method.lower() not in ['grid', 'bayesian', 'optuna']:
        raise ValueError("settings.optimization_method must be 'grid', 'bayesian', or 'optuna'.")

    data_signature = (
        in_sample_df.index[0] if len(in_sample_df) > 0 else None,
        in_sample_df.index[-1] if len(in_sample_df) > 0 else None,
        len(in_sample_df)
    )

    def evaluate_params(param_dict):
        """Evaluate run_backtest with caching, logging, and interruption control."""
        if control:
            control.wait_if_paused()
            if control.should_stop():
                raise OptimizationInterrupted()
        cache_key = (data_signature, tuple(sorted(param_dict.items())))
        with cache_lock:
            if cache_key in cache:
                return cache[cache_key]
        score = strategy_adapter.run_backtest(
            in_sample_df, param_dict, timeframe=timeframe, return_portfolio=False
        )
        with cache_lock:
            cache[cache_key] = score
        return score

    def expand_param_values(bounds):
        if isinstance(bounds, (list, tuple)):
            if len(bounds) == 1:
                return [bounds[0]]
            if len(bounds) == 3 and all(isinstance(b, (int, float)) for b in bounds[:2]):
                min_val, max_val, step = bounds
                if step == 0:
                    return [min_val]
                count = int(np.floor((max_val - min_val) / step)) + 1
                values = [min_val + step * i for i in range(max(count, 1))]
                return values
            if len(bounds) >= 2 and all(isinstance(b, (int, float)) for b in bounds):
                return list(bounds)
        raise ValueError(f"Unsupported bounds format: {bounds}")

    def split_params(grid):
        tunable = {}
        fixed = {}
        for key, bounds in grid.items():
            values = expand_param_values(bounds)
            if len(values) == 1:
                fixed[key] = values[0]
            else:
                tunable[key] = values
        return tunable, fixed

    tunable_grid, fixed_params = split_params(param_grid)

    if method == "grid":
        logger.info("Using Vectorized Grid Search...")

        # 1. Expand the grid into lists of values
        param_keys = list(param_grid.keys())
        param_values_list = []
        for key in param_keys:
            values = expand_param_values(param_grid[key])
            param_values_list.append(values)

        if not param_values_list or any(len(v) == 0 for v in param_values_list):
            raise ValueError("No parameter combinations generated.")

        # 2. Iterate Cartesian product lazily in chunks to avoid huge peak memory.
        total_combos = int(np.prod([len(v) for v in param_values_list]))
        if total_combos <= 0:
            raise ValueError("No parameter combinations generated.")
        logger.info("Generating %d parameter combinations...", total_combos)

        # Chunking Logic
        chunk_size = getattr(settings, 'batch_size', 1000) # Default to 1000 if not set
        if chunk_size <= 0: chunk_size = 1000

        result_chunks = []
        combos_iter = product(*param_values_list)
        chunk_start = 0

        # Process in chunks
        while True:
            if control:
                control.wait_if_paused()
                if control.should_stop():
                    raise OptimizationInterrupted()

            chunk_combos = list(islice(combos_iter, chunk_size))
            if not chunk_combos:
                break

            # Evaluate chunk — vectorized with bisection fallback on failure.
            # _eval_chunk_bisect tries vectorized first; if it fails, splits the
            # chunk in half and recurses until the bad combo(s) are isolated and
            # evaluated sequentially. This preserves vectorization for all good
            # combos instead of falling back to O(N) sequential evaluation.
            score_values = _eval_chunk_bisect(
                chunk_combos, param_keys, metrics_info,
                in_sample_df, strategy_adapter, timeframe,
                evaluate_params,
            )

            chunk_df = pd.DataFrame(chunk_combos, columns=param_keys)
            chunk_df['combined_score'] = score_values
            result_chunks.append(chunk_df)

            nan_count = sum(1 for s in score_values if not np.isfinite(float(s) if s is not None else float('nan')))
            if nan_count:
                logger.warning(
                    "Chunk %d-%d: %d/%d combos returned non-finite score.",
                    chunk_start, chunk_start + len(chunk_combos), nan_count, len(chunk_combos),
                )

            chunk_start += len(chunk_combos)

        # 4. Construct Results DataFrame
        if not result_chunks:
            raise ValueError("No optimization results generated in grid mode.")
        results_df = pd.concat(result_chunks, ignore_index=True)

        # Add metrics info columns (constant)
        for k, v in metrics_info.items():
            results_df[k] = v

        sorted_results = results_df.sort_values('combined_score', ascending=False)
        evaluation_count = len(results_df)

    elif method == "bayesian":
        # Bayesian optimization (Iterative)
        if tunable_grid:
            total_combinations = int(np.prod([len(values) for values in tunable_grid.values()]))
        else:
            total_combinations = 1

        # Use user-defined max_trials, capped by total search space size
        max_trials = getattr(settings, 'max_trials', 200)
        n_calls = max(1, min(max_trials, total_combinations))
        n_initial_points = min(50, n_calls, max(1, int(0.1 * n_calls)))

        # Patience logic
        patience_level = getattr(settings, 'patience_level', 'Medium')
        patience_factor = 0.2 # Medium default
        if patience_level == 'Low': patience_factor = 0.1
        elif patience_level == 'High': patience_factor = 0.4

        patience = max(1, int(patience_factor * n_calls))

        class NoImprovementStopper:
            def __init__(self, patience_steps):
                self.patience_steps = patience_steps
                self.best_value = None
                self.no_improve_steps = 0

            def __call__(self, study, trial):
                value = trial.value
                if value is None:
                    return
                if self.best_value is None or value < self.best_value - 1e-9:
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

        # For Bayesian, we stick to sequential or simple parallel if needed,
        # but since we optimized the core, even single threaded is faster.
        # Vectorizing Bayesian is hard because it's sequential by nature.
        n_jobs = 1

        def objective(trial):
            if control:
                control.wait_if_paused()
                if control.should_stop():
                    raise OptimizationInterrupted()
            params = {}
            for param_name, values in tunable_grid.items():
                if param_name in int_params:
                    params[param_name] = int(trial.suggest_categorical(param_name, values))
                else:
                    params[param_name] = float(trial.suggest_categorical(param_name, values))
            params.update(fixed_params)
            params.update(metrics_info)
            score = evaluate_params(params)

            # Handle NaN/None scores gracefully
            if score is None or np.isnan(score) or np.isinf(score):
                return float('inf') # Return worst possible value for minimization

            return -score

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
            if trial.value is None:
                continue
            param_dict = trial.params.copy()

            # Re-inject fixed parameters
            for param_name, value in fixed_params.items():
                param_dict[param_name] = value

            param_dict.update(metrics_info)
            param_dict['combined_score'] = -trial.value
            results.append(param_dict)

        if not results:
            raise ValueError("Bayesian optimization produced no results. Check bounds or trials.")

        results_df = pd.DataFrame(results)
        sorted_results = results_df.sort_values('combined_score', ascending=False)
        evaluation_count = len(study.trials)

    elif method == "optuna":
        # Optuna (TPE) implementation - dynamic based on param_grid

        def objective(trial):
            if control:
                control.wait_if_paused()
                if control.should_stop():
                    raise OptimizationInterrupted()
            params = {}
            for param_name, values in tunable_grid.items():
                if param_name in ['timeperiod', 'fenetre_lowest', 'longueur_mediane', 'Nb_bars_above', 'user_exit_sma_length']:
                    params[param_name] = int(trial.suggest_categorical(param_name, values))
                else:
                    params[param_name] = float(trial.suggest_categorical(param_name, values))
            params.update(fixed_params)

            params.update(metrics_info)
            score = evaluate_params(params)

            # Handle NaN/None scores gracefully
            if score is None or np.isnan(score) or np.isinf(score):
                return float('-inf') # Return worst possible value for maximization

            return score

        study = optuna.create_study(direction='maximize')

        # Calculate total combinations roughly
        total_combos = int(np.prod([len(values) for values in tunable_grid.values()])) if tunable_grid else 1
        max_trials = getattr(settings, 'max_trials', 200)
        n_trials = max(1, min(max_trials, total_combos))

        try:
            study.optimize(objective, n_trials=n_trials)
        except OptimizationInterrupted:
            raise

        results = []
        for trial in study.trials:
            param_dict = trial.params.copy()

            # Re-inject fixed parameters
            for param_name, value in fixed_params.items():
                param_dict[param_name] = value

            param_dict.update(metrics_info)
            param_dict['combined_score'] = trial.value
            results.append(param_dict)

        if not results:
            raise ValueError("Optuna optimization produced no results. Check bounds or trials.")

        results_df = pd.DataFrame(results)
        sorted_results = results_df.sort_values('combined_score', ascending=False)
        evaluation_count = len(study.trials)

    else:
        raise ValueError(f"Unsupported optimization method: {method}. Choose 'grid', 'bayesian', or 'optuna'.")

    return sorted_results, evaluation_count

def get_stable_best_params(optimization_results, param_grid, neighbor_count=5, score_col='combined_score'):
    """Select a robust best row using neighborhood-averaged scores in normalized param space."""
    if optimization_results is None or optimization_results.empty:
        raise ValueError("optimization_results must be a non-empty DataFrame.")
    if score_col not in optimization_results.columns:
        raise ValueError(f"'{score_col}' not found in optimization_results.")

    param_cols = [k for k in param_grid.keys() if k in optimization_results.columns]
    numeric_cols = [c for c in param_cols if pd.api.types.is_numeric_dtype(optimization_results[c])]
    if not numeric_cols:
        return optimization_results.iloc[0], None

    values = optimization_results[numeric_cols].astype(float).to_numpy()
    mins = np.nanmin(values, axis=0)
    maxs = np.nanmax(values, axis=0)
    scales = np.where(maxs - mins == 0, 1.0, maxs - mins)
    norm_vals = (values - mins) / scales

    scores = pd.to_numeric(optimization_results[score_col], errors='coerce').to_numpy()
    k = max(1, min(neighbor_count, len(optimization_results)))

    tree = KDTree(norm_vals)
    _, indices = tree.query(norm_vals, k=k)
    smoothed = np.nanmean(scores[indices], axis=1)

    if np.all(np.isnan(smoothed)):
        return optimization_results.iloc[0], None

    best_pos = int(np.nanargmax(smoothed))
    best_row = optimization_results.iloc[best_pos]
    return best_row, smoothed[best_pos]

def walk_forward_optimization(
    df,
    param_grid=None,
    metrics_info=None,
    timeframe='5s',
    settings=None,
    status_callback=None,
    control=None,
    strategy_adapter=None,
):
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
    if strategy_adapter is None:
        strategy_adapter = resolve_strategy_adapter(strategy_mode='native_atdmf')

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
    optimization_regime = getattr(settings, 'optimization_regime', 'classic')
    regime_key = str(optimization_regime).lower()
    use_nn_guided = regime_key == 'nn_guided'
    use_prev_best_grid = regime_key == 'prev_best_grid'
    nn_guide = NeuralSearchGuide(param_grid, settings) if use_nn_guided else None
    prev_window_best_params = None

    # Run-scoped backtest cache — LRU-bounded to prevent unbounded memory growth.
    # At ~1 KB per entry (param tuple key + float score), 5000 entries ≈ 5 MB max.
    class _BoundedCache:
        """Thread-safe dict with a hard eviction limit (FIFO when full)."""
        def __init__(self, maxsize=5000):
            self._d = {}
            self._maxsize = maxsize
            self._lock = threading.Lock()

        def get(self, key, default=None):
            with self._lock:
                return self._d.get(key, default)

        def __getitem__(self, key):
            with self._lock:
                return self._d[key]

        def __setitem__(self, key, value):
            with self._lock:
                if key in self._d:
                    return
                if len(self._d) >= self._maxsize:
                    # Evict oldest inserted key (Python 3.7+ dict preserves insertion order)
                    self._d.pop(next(iter(self._d)))
                self._d[key] = value

        def __contains__(self, key):
            with self._lock:
                return key in self._d

        def __len__(self):
            with self._lock:
                return len(self._d)

    backtest_cache = _BoundedCache(maxsize=5000)
    backtest_cache_lock = threading.Lock()  # kept for API compatibility with optimize_parameters

    wfo_results = {
        'window_results': [],
        'in_sample_performance': [],
        'out_of_sample_performance': [],
        'best_params': [],
        'nn_guidance': [],
        'prev_best_guidance': [],
        'settings': {
            'n_windows': settings.n_windows,
            'train_size': settings.train_size,
            'anchored': settings.anchored,
            'optimization_metric': settings.optimization_metric,
            'secondary_metric': settings.secondary_metric,
            'metric_weights': settings.metric_weights,
            'parallel_backend': settings.parallel_backend,
            'use_numba': settings.use_numba,
            'optimization_method': settings.optimization_method,
            'optimization_regime': optimization_regime,
            'neighbor_count': getattr(settings, 'neighbor_count', 5),
            'nn_min_samples': int(getattr(settings, 'nn_min_samples', 500)),
            'nn_candidate_pool_size': int(getattr(settings, 'nn_candidate_pool_size', 3000)),
            'nn_top_k': int(getattr(settings, 'nn_top_k', 250)),
            'nn_exploration_ratio': float(getattr(settings, 'nn_exploration_ratio', 0.15)),
            'nn_hidden_size': int(getattr(settings, 'nn_hidden_size', 32)),
            'nn_epochs': int(getattr(settings, 'nn_epochs', 60)),
            'nn_learning_rate': float(getattr(settings, 'nn_learning_rate', 0.01)),
            'nn_l2': float(getattr(settings, 'nn_l2', 1e-4)),
            'exit_sar_enabled': getattr(settings, 'exit_sar_enabled', True),
            'exit_macd_enabled': getattr(settings, 'exit_macd_enabled', True),
            'exit_macd_type_a': getattr(settings, 'exit_macd_type_a', True),
            'exit_macd_type_b': getattr(settings, 'exit_macd_type_b', True),
            'exit_cross_sar_sma_enabled': getattr(settings, 'exit_cross_sar_sma_enabled', True),
            'exit_retour_bb_enabled': getattr(settings, 'exit_retour_bb_enabled', False),
            'exit_regline_enabled': getattr(settings, 'exit_regline_enabled', False),
            'exit_volat_down_enabled': getattr(settings, 'exit_volat_down_enabled', False),
            'use_t2_signal': getattr(settings, 'use_t2_signal', False),
            'macd_ma_type': getattr(settings, 'macd_ma_type', 'sma'),
        }
    }

    # Calculate total parameter combinations for reporting
    param_combinations = np.prod([len(values) for values in param_grid.values()])

    def log(message: str):
        logger.info(message)
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
    log(f"Optimization Regime: {optimization_regime}")
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

        window_df = df.iloc[start_idx:end_idx]

        # For anchored WFO, always start from the first data point
        if settings.anchored:
            in_sample_start_idx = 0
        else:
            in_sample_start_idx = start_idx

        # Calculate in-sample end index
        in_sample_end_idx = start_idx + int(window_size * settings.train_size)

        # Create in-sample and out-of-sample DataFrames
        if settings.anchored:
            in_sample_df = df.iloc[in_sample_start_idx:in_sample_end_idx]
        else:
            in_sample_df = window_df.iloc[:int(window_size * settings.train_size)]

        out_sample_df = window_df.iloc[int(window_size * settings.train_size):]

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

        # Build the search grid for this window.
        window_param_grid = param_grid
        nn_window_info = {
            'enabled': False,
            'trained': False,
            'baseline_combinations': int(np.prod([len(v) for v in param_grid.values()])),
            'guided_combinations': int(np.prod([len(v) for v in param_grid.values()]))
        }
        if use_prev_best_grid:
            if prev_window_best_params is None:
                log("Previous-best-grid mode: first window uses full baseline grid.")
            else:
                window_param_grid, prev_info = _build_prev_best_grid(param_grid, prev_window_best_params)
                log(
                    "Previous-best-grid mode: "
                    f"{prev_info['guided_combinations']} combos "
                    f"(baseline {prev_info['baseline_combinations']})"
                )
                wfo_results['prev_best_guidance'].append({
                    'window': i + 1,
                    **prev_info
                })
        elif nn_guide is not None:
            window_param_grid, nn_window_info = nn_guide.build_guided_grid(param_grid)
            if nn_window_info.get('enabled'):
                log(
                    "NN-guided grid: "
                    f"{nn_window_info['guided_combinations']} combos "
                    f"(baseline {nn_window_info['baseline_combinations']})"
                )
            else:
                log("NN-guided grid not active yet (insufficient cumulative trials).")

        # Clear per-window indicator cache so new window data is used
        clear_window_indicator_cache()

        # Optimize parameters on in-sample data
        log(f"Optimizing parameters on in-sample data ({len(in_sample_df)} bars)...")

        # Time the optimization process
        optimization_start = time.time()
        try:
            optimization_results, eval_count = optimize_parameters(
                in_sample_df,
                window_param_grid,
                metrics_info,
                timeframe,
                settings,
                control=control,
                strategy_adapter=strategy_adapter,
                cache=backtest_cache,
                cache_lock=backtest_cache_lock,
            )
        except OptimizationInterrupted:
            log("Optimization interrupted during parameter search.")
            raise
        optimization_time = time.time() - optimization_start
        optimization_times.append(optimization_time)

        # Get best parameters using stability selection
        neighbor_count = getattr(settings, 'neighbor_count', 5)
        best_row, stable_score = get_stable_best_params(
            optimization_results,
            window_param_grid,
            neighbor_count=neighbor_count
        )
        best_params = best_row.drop(
            ['combined_score', metrics_info['metric1_name'], metrics_info['metric2_name']],
            errors='ignore'
        ).to_dict()

        log(f"Best parameters found: {best_params}")
        log(f"Score: {best_row['combined_score']:.4f}")
        if stable_score is not None:
            log(f"Stable score (neighbor avg): {stable_score:.4f}")
        log(f"Optimization time: {timedelta(seconds=int(optimization_time))}")

        # Test best parameters on in-sample data
        in_sample_portfolio = strategy_adapter.run_backtest(
            in_sample_df, best_params, timeframe=timeframe, return_portfolio=True
        )

        _is_stats = _get_trades_stats(in_sample_portfolio.trades)
        in_sample_metrics = {
            'window': i + 1,
            'return': in_sample_portfolio.total_return * 100,
            'sharpe': in_sample_portfolio.sharpe_ratio,
            'max_drawdown': in_sample_portfolio.max_drawdown * 100,
            'win_rate': in_sample_portfolio.trades.win_rate,
            'avg_gain_per_trade': trade_stat(in_sample_portfolio.trades, 'avg_winning_trade', _stats_cache=_is_stats),
            'avg_loss_per_trade': trade_stat(in_sample_portfolio.trades, 'avg_losing_trade', _stats_cache=_is_stats),
            'avg_pl_per_trade': calc_avg_pl(in_sample_portfolio),
            'calmar_ratio': in_sample_portfolio.calmar_ratio if in_sample_portfolio.max_drawdown > 0 else np.nan,
            'sortino_ratio': in_sample_portfolio.sortino_ratio,
            'pqs': calc_pqs(in_sample_portfolio),
            'n_trades': len(in_sample_portfolio.trades)
        }

        wfo_results['in_sample_performance'].append(in_sample_metrics)

        # Test on out-of-sample data if available
        out_sample_metrics = None
        if len(out_sample_df) > 0:
            log(f"Testing best parameters on out-of-sample data ({len(out_sample_df)} bars)...")
            out_sample_portfolio = strategy_adapter.run_backtest(
                out_sample_df, best_params, timeframe=timeframe, return_portfolio=True
            )

            _oos_stats = _get_trades_stats(out_sample_portfolio.trades)
            out_sample_metrics = {
                'window': i + 1,
                'return': out_sample_portfolio.total_return * 100,
                'sharpe': out_sample_portfolio.sharpe_ratio,
                'max_drawdown': out_sample_portfolio.max_drawdown * 100,
                'win_rate': out_sample_portfolio.trades.win_rate,  #* 100,
                'avg_gain_per_trade': trade_stat(out_sample_portfolio.trades, 'avg_winning_trade', _stats_cache=_oos_stats),
                'avg_loss_per_trade': trade_stat(out_sample_portfolio.trades, 'avg_losing_trade', _stats_cache=_oos_stats),
                'avg_pl_per_trade': calc_avg_pl(out_sample_portfolio),
                'calmar_ratio': out_sample_portfolio.calmar_ratio if out_sample_portfolio.max_drawdown > 0 else np.nan,
                'sortino_ratio': out_sample_portfolio.sortino_ratio,
                'pqs': calc_pqs(out_sample_portfolio),
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
            # Keep all in-sample trial rows for post-run statistics and replayability.
            'optimization_trials': optimization_results.to_dict('records'),
            'optimization_trials_count': int(len(optimization_results)),
            'evaluations': int(eval_count),
            'best_params': best_params
        }

        wfo_results['window_results'].append(window_result)
        wfo_results['best_params'].append(best_params)
        prev_window_best_params = best_params.copy()

        if nn_guide is not None:
            nn_guide.last_best_params = best_params.copy()
            nn_guide.update(optimization_results, window_param_grid)
            latest_weights = nn_guide.get_parameter_weights()
            nn_window_info['trained_after_window'] = bool(nn_guide.trained)
            if latest_weights:
                nn_window_info['parameter_weights'] = latest_weights
            wfo_results['nn_guidance'].append({
                'window': i + 1,
                **nn_window_info
            })


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

        # Calculate parameter stability (numeric only)
        param_stability = {}
        for param in param_grid.keys():
            series = pd.to_numeric(params_df[param], errors='coerce')
            series = series.dropna()
            if series.empty:
                continue
            mean_val = series.mean()
            if mean_val > 0:
                param_stability[param] = 1.0 - (series.std() / mean_val)
            else:
                param_stability[param] = 0.0

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
