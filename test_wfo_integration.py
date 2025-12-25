
import pandas as pd
import numpy as np
import sys
import os

# Ensure we can import from current directory
sys.path.append(os.getcwd())

from main import get_param_grid, get_metrics_info, get_wfo_settings
from wfo import walk_forward_optimization, WFOSettings
import vectorbtpro as vbt

# Mock Data Generation
def generate_mock_data(n=1000):
    index = pd.date_range("2025-01-01", periods=n, freq="5s")
    close = np.random.uniform(100, 200, size=n)
    # Make sure close is Series
    return pd.DataFrame({'Close': close, 'Open': close, 'High': close+1, 'Low': close-1}, index=index)

def test_integration():
    print("Generating mock data...")
    df = generate_mock_data(100) # Small dataset for speed
    
    # Mock Config: Unselect 'coeff_medianeBBW' to trigger default injection
    config = {
        'selected_params': ['timeperiod'], # Only timeperiod selected
        'timeperiod_min': 10,
        'timeperiod_max': 12,
        'timeperiod_step': 1,
        # 'coeff_medianeBBW' is NOT selected
        'metric1_name': 'sharpe_ratio',
        'metric2_name': 'total_return',
        'weight_metric1': 1.0,
        'weight_metric2': 0.0,
        'n_windows': 1,
        'train_size': 0.5,
        'optimization_method': 'grid',
        'parallel_backend': 'thread', # Use thread for simplicity
        'max_trials': 5 # Low trials for bayesian/optuna
    }
    
    print("\n--- Testing get_param_grid ---")
    param_grid = get_param_grid(config)
    print("Param Grid Keys:", list(param_grid.keys()))
    
    if 'coeff_medianeBBW' not in param_grid:
        print("FAIL: coeff_medianeBBW missing from param_grid")
        return
    if param_grid['coeff_medianeBBW'] != [1.1]: # Default value
        print(f"FAIL: coeff_medianeBBW value mismatch. Expected [1.1], got {param_grid['coeff_medianeBBW']}")
        return
    print("PASS: get_param_grid correctly populated default.")
    
    metrics_info = get_metrics_info(config)
    
    # Test 1: Grid Search
    print("\n--- Testing WFO (Grid) ---")
    settings = get_wfo_settings(config)
    settings.optimization_method = 'grid'
    
    try:
        walk_forward_optimization(df, param_grid, metrics_info, timeframe='5s', settings=settings)
        print("PASS: Grid Search ran successfully.")
    except Exception as e:
        print(f"FAIL: Grid Search raised exception: {e}")
        import traceback
        traceback.print_exc()
        
    # Test 2: Bayesian
    print("\n--- Testing WFO (Bayesian) ---")
    settings.optimization_method = 'bayesian'
    settings.max_trials = 5
    try:
        walk_forward_optimization(df, param_grid, metrics_info, timeframe='5s', settings=settings)
        print("PASS: Bayesian ran successfully.")
    except Exception as e:
        print(f"FAIL: Bayesian raised exception: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Optuna
    print("\n--- Testing WFO (Optuna) ---")
    settings.optimization_method = 'optuna'
    settings.max_trials = 5
    try:
        walk_forward_optimization(df, param_grid, metrics_info, timeframe='5s', settings=settings)
        print("PASS: Optuna ran successfully.")
    except Exception as e:
        print(f"FAIL: Optuna raised exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_integration()
