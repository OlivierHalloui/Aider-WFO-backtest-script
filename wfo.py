# Import necessary libraries for WFO
import pandas as pd
import numpy as np
import vectorbtpro as vbt
from itertools import product, islice
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


def _safe_float(value, default=0.0):
    try:
        if isinstance(value, (bool, np.bool_)):
            return 1.0 if value else 0.0
        if isinstance(value, (np.integer, int, np.floating, float)):
            value = float(value)
            if np.isnan(value) or np.isinf(value):
                return float(default)
            return value
        return float(default)
    except Exception:
        return float(default)


class NeuralSearchGuide:
    """Lightweight numpy MLP used to guide WFO search space from past windows."""

    def __init__(self, base_param_grid, settings):
        self.param_keys = list(base_param_grid.keys())
        self.category_maps = {key: {} for key in self.param_keys}
        self.records = []
        self.targets = []
        self.max_records = int(getattr(settings, 'nn_max_records', 200000))
        self.min_samples = int(getattr(settings, 'nn_min_samples', 500))
        self.candidate_pool_size = int(getattr(settings, 'nn_candidate_pool_size', 3000))
        self.top_k = int(getattr(settings, 'nn_top_k', 250))
        self.exploration_ratio = float(getattr(settings, 'nn_exploration_ratio', 0.15))
        self.hidden_size = int(getattr(settings, 'nn_hidden_size', 32))
        self.epochs = int(getattr(settings, 'nn_epochs', 60))
        self.learning_rate = float(getattr(settings, 'nn_learning_rate', 0.01))
        self.l2 = float(getattr(settings, 'nn_l2', 1e-4))
        self.random_state = int(getattr(settings, 'random_state', 42))
        self.rng = np.random.default_rng(self.random_state)
        self.last_best_params = None

        self.x_mean = None
        self.x_std = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.trained = False

    def _encode_value(self, key, value):
        if isinstance(value, (bool, np.bool_)):
            return 1.0 if value else 0.0
        if isinstance(value, (np.integer, int, np.floating, float)):
            return _safe_float(value, default=0.0)
        map_key = str(value)
        cat_map = self.category_maps.setdefault(key, {})
        if map_key not in cat_map:
            cat_map[map_key] = len(cat_map)
        return float(cat_map[map_key])

    def _encode_rows(self, rows):
        encoded = []
        for row in rows:
            vector = [self._encode_value(key, row.get(key)) for key in self.param_keys]
            encoded.append(vector)
        return np.asarray(encoded, dtype=float)

    def _init_model(self, in_dim):
        scale1 = np.sqrt(2.0 / max(1, in_dim))
        scale2 = np.sqrt(2.0 / max(1, self.hidden_size))
        self.W1 = self.rng.normal(0.0, scale1, size=(in_dim, self.hidden_size))
        self.b1 = np.zeros((1, self.hidden_size), dtype=float)
        self.W2 = self.rng.normal(0.0, scale2, size=(self.hidden_size, 1))
        self.b2 = np.zeros((1, 1), dtype=float)

    def _fit(self):
        if len(self.targets) < self.min_samples:
            self.trained = False
            return

        X = self._encode_rows(self.records)
        y = np.asarray(self.targets, dtype=float).reshape(-1, 1)
        if X.ndim != 2 or X.shape[0] < self.min_samples:
            self.trained = False
            return

        self.x_mean = X.mean(axis=0, keepdims=True)
        self.x_std = X.std(axis=0, keepdims=True)
        self.x_std = np.where(self.x_std < 1e-9, 1.0, self.x_std)
        Xn = (X - self.x_mean) / self.x_std

        self.y_mean = float(np.nanmean(y))
        y_std = float(np.nanstd(y))
        self.y_std = 1.0 if y_std < 1e-9 else y_std
        yn = (y - self.y_mean) / self.y_std

        in_dim = Xn.shape[1]
        if self.W1 is None or self.W1.shape[0] != in_dim:
            self._init_model(in_dim)

        n = Xn.shape[0]
        lr = max(1e-5, self.learning_rate)
        l2 = max(0.0, self.l2)

        for _ in range(max(1, self.epochs)):
            h = np.tanh(Xn @ self.W1 + self.b1)
            pred = h @ self.W2 + self.b2
            err = pred - yn

            d_pred = (2.0 / n) * err
            grad_W2 = h.T @ d_pred + l2 * self.W2
            grad_b2 = np.sum(d_pred, axis=0, keepdims=True)

            d_h = d_pred @ self.W2.T
            d_z = d_h * (1.0 - h ** 2)
            grad_W1 = Xn.T @ d_z + l2 * self.W1
            grad_b1 = np.sum(d_z, axis=0, keepdims=True)

            self.W2 -= lr * grad_W2
            self.b2 -= lr * grad_b2
            self.W1 -= lr * grad_W1
            self.b1 -= lr * grad_b1

        self.trained = True

    def predict_scores(self, rows):
        if not self.trained or self.W1 is None:
            return None
        X = self._encode_rows(rows)
        Xn = (X - self.x_mean) / self.x_std
        h = np.tanh(Xn @ self.W1 + self.b1)
        pred = h @ self.W2 + self.b2
        return pred.reshape(-1) * self.y_std + self.y_mean

    def update(self, optimization_results, param_grid):
        if optimization_results is None or optimization_results.empty:
            return
        if 'combined_score' not in optimization_results.columns:
            return

        valid_rows = optimization_results.replace([np.inf, -np.inf], np.nan).dropna(subset=['combined_score'])
        if valid_rows.empty:
            return

        param_cols = [key for key in self.param_keys if key in valid_rows.columns]
        if not param_cols:
            return

        for _, row in valid_rows.iterrows():
            row_dict = {key: row.get(key) for key in param_cols}
            score = _safe_float(row.get('combined_score'), default=np.nan)
            if np.isnan(score) or np.isinf(score):
                continue
            self.records.append(row_dict)
            self.targets.append(score)

        if len(self.targets) > self.max_records:
            self.records = self.records[-self.max_records:]
            self.targets = self.targets[-self.max_records:]

        self._fit()

    def get_parameter_weights(self):
        if not self.trained or self.W1 is None or self.W2 is None:
            return {}
        # Input saliency proxy from absolute path weights through first hidden layer.
        raw = np.abs(self.W1) @ np.abs(self.W2).reshape(-1, 1)
        raw = raw.reshape(-1)
        total = float(np.sum(raw))
        if total <= 0:
            return {key: 0.0 for key in self.param_keys}
        return {key: float(raw[idx] / total) for idx, key in enumerate(self.param_keys)}

    def build_guided_grid(self, base_param_grid):
        """Create a reduced parameter grid guided by NN scores plus exploration safeguards."""
        baseline_combos = int(np.prod([len(v) for v in base_param_grid.values()]))
        if not self.trained:
            return base_param_grid, {
                'enabled': False,
                'trained': False,
                'reason': 'insufficient_samples',
                'baseline_combinations': baseline_combos,
                'guided_combinations': baseline_combos
            }

        sample_count = max(self.top_k * 2, self.candidate_pool_size)
        sample_count = max(500, sample_count)
        sampled_rows = []
        for _ in range(sample_count):
            candidate = {}
            for key, values in base_param_grid.items():
                if not values:
                    continue
                idx = int(self.rng.integers(0, len(values)))
                candidate[key] = values[idx]
            sampled_rows.append(candidate)

        predicted = self.predict_scores(sampled_rows)
        if predicted is None or len(predicted) == 0:
            return base_param_grid, {
                'enabled': False,
                'trained': True,
                'reason': 'prediction_failed',
                'baseline_combinations': baseline_combos,
                'guided_combinations': baseline_combos
            }

        top_k = max(1, min(self.top_k, len(sampled_rows)))
        top_idx = np.argpartition(predicted, -top_k)[-top_k:]
        top_rows = [sampled_rows[i] for i in top_idx]

        guided_grid = {}
        for key, base_values in base_param_grid.items():
            base_values = list(base_values)
            if len(base_values) <= 1:
                guided_grid[key] = base_values
                continue

            selected_values = []
            seen = set()
            for row in top_rows:
                value = row.get(key)
                if value in seen:
                    continue
                seen.add(value)
                selected_values.append(value)

            if self.last_best_params and key in self.last_best_params and self.last_best_params[key] in base_values:
                best_value = self.last_best_params[key]
                if best_value not in seen:
                    seen.add(best_value)
                    selected_values.append(best_value)

            min_keep = max(2, int(np.ceil(len(base_values) * 0.2)))
            while len(selected_values) < min_keep:
                for value in base_values:
                    if value not in seen:
                        seen.add(value)
                        selected_values.append(value)
                        break
                else:
                    break

            exploration_count = max(1, int(np.ceil(len(base_values) * self.exploration_ratio)))
            remaining = [value for value in base_values if value not in seen]
            if remaining:
                self.rng.shuffle(remaining)
                for value in remaining[:exploration_count]:
                    seen.add(value)
                    selected_values.append(value)

            # Preserve original order for reproducibility.
            guided_values = [value for value in base_values if value in seen]
            guided_grid[key] = guided_values if guided_values else base_values

        guided_combos = int(np.prod([len(v) for v in guided_grid.values()]))
        weights = self.get_parameter_weights()
        return guided_grid, {
            'enabled': True,
            'trained': True,
            'baseline_combinations': baseline_combos,
            'guided_combinations': guided_combos,
            'reduction_ratio': 1.0 - (guided_combos / baseline_combos) if baseline_combos > 0 else 0.0,
            'parameter_weights': weights
        }


def _match_prev_value_to_candidates(prev_value, candidates):
    if candidates is None:
        return None
    for candidate in candidates:
        if candidate == prev_value:
            return candidate
    # Fallbacks for bool/num casting mismatches
    if isinstance(prev_value, (bool, np.bool_)):
        prev_bool = bool(prev_value)
        for candidate in candidates:
            if isinstance(candidate, (bool, np.bool_)) and bool(candidate) == prev_bool:
                return candidate
        return None

    if isinstance(prev_value, (int, float, np.integer, np.floating)):
        prev_num = float(prev_value)
        for candidate in candidates:
            if isinstance(candidate, (int, float, np.integer, np.floating)) and np.isclose(float(candidate), prev_num):
                return candidate
    return None


def _build_prev_best_grid(base_param_grid, prev_best_params):
    """Build a grid centered on previous-window best values when they still exist in domains."""
    guided_grid = {}
    fixed_count = 0
    for key, values in base_param_grid.items():
        base_values = list(values)
        prev_value = prev_best_params.get(key) if isinstance(prev_best_params, dict) else None
        matched = _match_prev_value_to_candidates(prev_value, base_values)
        if matched is not None:
            guided_grid[key] = [matched]
            fixed_count += 1
        else:
            guided_grid[key] = base_values

    baseline = int(np.prod([len(v) for v in base_param_grid.values()])) if base_param_grid else 0
    guided = int(np.prod([len(v) for v in guided_grid.values()])) if guided_grid else 0
    return guided_grid, {
        'enabled': True,
        'trained': True,
        'fixed_params': fixed_count,
        'baseline_combinations': baseline,
        'guided_combinations': guided,
        'reduction_ratio': 1.0 - (guided / baseline) if baseline > 0 else 0.0
    }

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
    parallel_backend = getattr(settings, 'parallel_backend', 'thread').lower()
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
        with backtest_cache_lock:
            if cache_key in backtest_cache:
                return backtest_cache[cache_key]
        # start = time.time()
        score = run_backtest(in_sample_df, param_dict, timeframe, return_portfolio=False)
        # elapsed = time.time() - start
        # print(f"Evaluation time: {elapsed:.2f}s for params: {param_dict}")
        with backtest_cache_lock:
            backtest_cache[cache_key] = score
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
        print("Using Vectorized Grid Search...")

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
        print(f"Generating {total_combos} parameter combinations...")

        # Chunking Logic
        chunk_size = getattr(settings, 'batch_size', 1000) # Default to 1000 if not set
        if chunk_size <= 0: chunk_size = 1000

        result_chunks = []
        combos_iter = product(*param_values_list)
        chunk_start = 0
        fallback_used = False

        # Process in chunks
        while True:
            if control:
                control.wait_if_paused()
                if control.should_stop():
                    raise OptimizationInterrupted()

            chunk_combos = list(islice(combos_iter, chunk_size))
            if not chunk_combos:
                break

            # Transpose chunk
            transposed_chunk = list(zip(*chunk_combos))

            vectorized_params = {}
            for k_idx, key in enumerate(param_keys):
                vectorized_params[key] = np.array(transposed_chunk[k_idx])

            # Add metrics info
            vectorized_params.update(metrics_info)

            try:
                # Run backtest for this chunk
                chunk_scores = run_backtest(in_sample_df, vectorized_params, timeframe, return_portfolio=False)

                # Normalize result to one score per combination.
                if np.isscalar(chunk_scores):
                    score_values = [float(chunk_scores)] * len(chunk_combos)
                elif hasattr(chunk_scores, 'values'):
                    score_values = list(np.asarray(chunk_scores.values).reshape(-1))
                else:
                    score_values = list(np.asarray(chunk_scores).reshape(-1))

                if len(score_values) != len(chunk_combos):
                    raise ValueError(
                        f"Chunk score size mismatch: got {len(score_values)} scores for "
                        f"{len(chunk_combos)} combinations."
                    )

                chunk_df = pd.DataFrame(chunk_combos, columns=param_keys)
                chunk_df['combined_score'] = score_values
                result_chunks.append(chunk_df)

            except Exception as e:
                if not fallback_used:
                    print(
                        "Vectorized grid chunk failed; falling back to per-combination "
                        "evaluation for robustness."
                    )
                    fallback_used = True
                print(f"Error during vectorized backtest chunk {chunk_start}-{chunk_start+len(chunk_combos)}: {e}")

                score_values = []
                for combo in chunk_combos:
                    combo_params = dict(zip(param_keys, combo))
                    combo_params.update(metrics_info)
                    combo_score = evaluate_params(combo_params)
                    if np.isscalar(combo_score):
                        score_values.append(float(combo_score))
                    elif hasattr(combo_score, 'values'):
                        arr = np.asarray(combo_score.values).reshape(-1)
                        score_values.append(float(arr[0]) if len(arr) else float('nan'))
                    else:
                        arr = np.asarray(combo_score).reshape(-1)
                        score_values.append(float(arr[0]) if len(arr) else float('nan'))

                chunk_df = pd.DataFrame(chunk_combos, columns=param_keys)
                chunk_df['combined_score'] = score_values
                result_chunks.append(chunk_df)

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
    smoothed = np.full(len(optimization_results), np.nan, dtype=float)

    for i in range(len(optimization_results)):
        dists = np.linalg.norm(norm_vals - norm_vals[i], axis=1)
        idx = np.argpartition(dists, k - 1)[:k]
        smoothed[i] = np.nanmean(scores[idx])

    if np.all(np.isnan(smoothed)):
        return optimization_results.iloc[0], None

    best_pos = int(np.nanargmax(smoothed))
    best_row = optimization_results.iloc[best_pos]
    return best_row, smoothed[best_pos]

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

    def calc_avg_pl(port):
        total_ret = port.total_return * 100
        n_trades = port.trades.count()
        if n_trades is None or n_trades == 0:
            return 0.0
        avg_pl = total_ret / n_trades
        if hasattr(avg_pl, 'replace'):
            avg_pl = avg_pl.replace([np.inf, -np.inf], 0).fillna(0)
        else:
            if np.isinf(avg_pl) or np.isnan(avg_pl):
                avg_pl = 0.0
        return avg_pl

    def trade_stat(trades, attr_name, default=0.0):
        value = getattr(trades, attr_name, None)
        if value is not None:
            return value
        try:
            stats = trades.stats()
        except Exception:
            return default
        keys = [
            attr_name,
            attr_name.replace('_', ' '),
            attr_name.replace('_', ' ').title(),
            attr_name.replace('_', ' ').capitalize(),
        ]
        for key in keys:
            try:
                value = stats.get(key) if hasattr(stats, 'get') else stats[key]
            except Exception:
                value = None
            if value is not None:
                return value
        return default
    
    # Store WFO results
    optimization_regime = getattr(settings, 'optimization_regime', 'classic')
    regime_key = str(optimization_regime).lower()
    use_nn_guided = regime_key == 'nn_guided'
    use_prev_best_grid = regime_key == 'prev_best_grid'
    nn_guide = NeuralSearchGuide(param_grid, settings) if use_nn_guided else None
    prev_window_best_params = None

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
            'exit_macd_type_b': getattr(settings, 'exit_macd_type_b', True)
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

        # Optimize parameters on in-sample data
        log(f"Optimizing parameters on in-sample data ({len(in_sample_df)} bars)...")
        
        # Time the optimization process
        optimization_start = time.time()
        try:
            optimization_results, eval_count = optimize_parameters(
                in_sample_df, window_param_grid, metrics_info, timeframe, settings, control=control
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
        in_sample_portfolio = run_backtest(in_sample_df, best_params, timeframe)
        
        in_sample_metrics = {
            'window': i + 1,
            'return': in_sample_portfolio.total_return * 100,
            'sharpe': in_sample_portfolio.sharpe_ratio,
            'max_drawdown': in_sample_portfolio.max_drawdown * 100,
            'win_rate': in_sample_portfolio.trades.win_rate,
            'avg_gain_per_trade': trade_stat(in_sample_portfolio.trades, 'avg_winning_trade'),
            'avg_loss_per_trade': trade_stat(in_sample_portfolio.trades, 'avg_losing_trade'),
            'avg_pl_per_trade': calc_avg_pl(in_sample_portfolio),
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
                'avg_gain_per_trade': trade_stat(out_sample_portfolio.trades, 'avg_winning_trade'),
                'avg_loss_per_trade': trade_stat(out_sample_portfolio.trades, 'avg_losing_trade'),
                'avg_pl_per_trade': calc_avg_pl(out_sample_portfolio),
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
