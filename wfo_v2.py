# wfo_v2.py

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from scipy.ndimage import uniform_filter
import optuna

# Import from the refactored strategy file
from strategy_v2 import ATDMF_Factory, WARMUP_PERIOD

def find_stable_params(perf_grid, param_grid):
    """
    Finds the most stable parameter combination from a performance grid.
    Smoothing the performance landscape to avoid overfitting to outliers.
    """
    # 1. Handle single combination
    if np.size(perf_grid) <= 1:
        return {k: v[0] for k, v in param_grid.items()}

    # 2. Reshape the performance series into a multi-dimensional grid
    # Identify parameters that actually have multiple values
    multi_params = [k for k, v in param_grid.items() if len(v) > 1]
    shape = [len(param_grid[k]) for k in multi_params]
    
    # Values are sorted by MultiIndex, so we can reshape
    perf_array = perf_grid.values.reshape(shape)

    # 3. Smooth the performance grid using a uniform filter.
    # The size of the filter is 20% of the dimension's size, with a minimum of 1.
    filter_size = [max(1, int(s * 0.2)) for s in shape]
    smoothed_perf = uniform_filter(perf_array, size=filter_size)

    # 4. Find the index of the maximum value in the *smoothed* grid
    if np.all(np.isnan(smoothed_perf)):
         # Fallback: if all performance is NaN, return the first combination
         # Also ensure we add single-value params later
         best_params = {}
         for k in multi_params:
             best_params[k] = param_grid[k][0]
         for k, v in param_grid.items():
            if k not in best_params:
                best_params[k] = v[0]
         return best_params

    max_idx_flat = np.nanargmax(smoothed_perf)
    best_indices = np.unravel_index(max_idx_flat, shape)

    # 5. Map the indices back to the actual parameter values
    best_params = {}
    for i, key in enumerate(multi_params):
        best_params[key] = param_grid[key][best_indices[i]]
        
    # Add back single-value params
    for k, v in param_grid.items():
        if k not in best_params:
            best_params[k] = v[0]

    return best_params

def run_sota_wfo(
    price_data,
    param_grid,
    n_windows=10,
    train_size=0.8,
    use_anchored=False,
    metric='sharpe_ratio',
    optimization_method='grid',
    max_trials=100,
    **kwargs
):
    """
    Performs a State-of-the-Art Walk-Forward Optimization.
    
    Implements:
    - Explicit Warmup Handling (Lookback Buffer)
    - Robust Parameter Selection (Smoothing)
    - Stitched Equity Curve Generation (Concatenation)
    - Multiple Optimization Engines (Grid, Bayesian/Optuna)
    """
    if price_data.index.freq is None:
        freq = pd.infer_freq(price_data.index)
    else:
        freq = price_data.index.freq
        
    # 1. Define Splitter
    splitter_cls = vbt.Splitter.from_expanding if use_anchored else vbt.Splitter.from_n_rolling
    splitter = splitter_cls(
        price_data.index,
        n=n_windows,
        split=train_size,
        set_labels=["train", "test"]
    )
    
    oos_portfolios = []
    best_params_list = []
    in_sample_perfs = []
    last_test_end_idx = -1
    
    # 2. Optimization Loop
    for i, split_masks in enumerate(splitter.get_iter_split_masks()):
        # Handle split_masks being a DataFrame (common with set_labels)
        if isinstance(split_masks, pd.DataFrame):
            train_mask = split_masks.iloc[:, 0].values
            test_mask = split_masks.iloc[:, 1].values
        elif hasattr(split_masks, 'values'):
             # Fallback for other pandas-like objects
             vals = split_masks.values
             if vals.ndim == 2:
                  train_mask = vals[:, 0]
                  test_mask = vals[:, 1]
             else:
                  train_mask = np.array(split_masks[0])
                  test_mask = np.array(split_masks[1])
        else:
             # Tuple or list
             train_mask = np.array(split_masks[0])
             test_mask = np.array(split_masks[1])
        
        # --- OVERLAP PROTECTION (STITCHING) ---
        # Ensure test sets are strictly contiguous and non-overlapping
        test_indices = np.where(test_mask)[0]
        if len(test_indices) > 0:
            if last_test_end_idx != -1:
                # If current test starts before or at last test end, trim it
                test_mask[test_indices[test_indices <= last_test_end_idx]] = False
            
            # Update last_test_end_idx from the updated mask
            new_test_indices = np.where(test_mask)[0]
            if len(new_test_indices) > 0:
                last_test_end_idx = new_test_indices[-1]
        
        # --- WARMUP PROTECTION ---
        # Ensure we don't trade during the first WARMUP_PERIOD bars of the entire dataset
        # because indicators will be NaN or unreliable.
        train_mask[:WARMUP_PERIOD] = False
        test_mask[:WARMUP_PERIOD] = False
        
        print(f"Processing Window {i+1}/{n_windows}...")
        
        # --- A. IN-SAMPLE OPTIMIZATION ---
        if optimization_method == 'grid':
            # Grid Search using VectorBT's built-in parameter broadcasting
            train_ind = ATDMF_Factory.run(
                close=price_data['Close'],
                high=price_data['High'],
                low=price_data['Low'],
                open=price_data['Open'],
                **param_grid,
                param_product=True,
                vbt_slice=train_mask
            )
            train_pf = vbt.Portfolio.from_signals(
                close=price_data['Close'].iloc[train_mask],
                entries=train_ind.entries,
                exits=train_ind.exits,
                freq=freq,
                init_cash=10000,
                fees=0.001
            )
            in_sample_perf = train_pf.deep_getattr(metric)
            # Find best stable params from the grid
            best_stable_params = find_stable_params(in_sample_perf, param_grid)
            
        elif optimization_method in ['bayesian', 'optuna']:
            # Optuna Optimization (TPE)
            def objective(trial):
                trial_params = {}
                for p_name, p_vals in param_grid.items():
                    if len(p_vals) == 1:
                        trial_params[p_name] = p_vals[0]
                    elif isinstance(p_vals[0], int):
                        trial_params[p_name] = trial.suggest_int(p_name, min(p_vals), max(p_vals))
                    else:
                        trial_params[p_name] = trial.suggest_float(p_name, min(p_vals), max(p_vals))
                
                # Evaluate on train slice
                t_ind = ATDMF_Factory.run(
                    close=price_data['Close'],
                    high=price_data['High'],
                    low=price_data['Low'],
                    open=price_data['Open'],
                    **trial_params,
                    vbt_slice=train_mask
                )
                t_pf = vbt.Portfolio.from_signals(
                    close=price_data['Close'].iloc[train_mask],
                    entries=t_ind.entries,
                    exits=t_ind.exits,
                    freq=freq,
                    init_cash=10000,
                    fees=0.001
                )
                res = t_pf.deep_getattr(metric)
                if np.isnan(res) or np.isinf(res):
                    return float('-inf')
                return res

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=max_trials)
            best_stable_params = study.best_params.copy()
            
            # Add back single-value params
            for k, v in param_grid.items():
                if k not in best_stable_params:
                    best_stable_params[k] = v[0]
            
            in_sample_perf = pd.Series([study.best_value], index=[0]) # Placeholder for IS perf
            
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
            
        # --- B. OUT-OF-SAMPLE TEST ---
        oos_ind = ATDMF_Factory.run(
            close=price_data['Close'],
            high=price_data['High'],
            low=price_data['Low'],
            open=price_data['Open'],
            **best_stable_params,
            # No vbt_slice here, we'll slice output manually to be 100% sure of index
        )
        
        # Explicitly slice the results to match the test mask index
        # This ensures the resulting Portfolio has the correct (contiguous) index
        oos_pf = vbt.Portfolio.from_signals(
            close=price_data['Close'].iloc[test_mask],
            entries=oos_ind.entries.iloc[test_mask],
            exits=oos_ind.exits.iloc[test_mask],
            freq=freq,
            init_cash=10000,
            fees=0.001
        )
        
        oos_portfolios.append(oos_pf)
        best_params_list.append(best_stable_params)
        in_sample_perfs.append(in_sample_perf)

    # 3. Return results in a GUI-compatible wrapper
    class MockCVResults:
        def __init__(self, out_list):
            self.out = out_list
            
    return MockCVResults(list(zip(oos_portfolios, best_params_list, in_sample_perfs)))
