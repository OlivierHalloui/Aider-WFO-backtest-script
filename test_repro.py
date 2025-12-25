
import sys
import os
import pandas as pd
import numpy as np
from main import get_param_grid

# Mock config
config = {
    'selected_params': ['timeperiod'], # Only timeperiod selected
    'timeperiod_min': 10,
    'timeperiod_max': 10,
    'timeperiod_step': 1
}

print("Testing get_param_grid...")
grid = get_param_grid(config)
print("Grid keys:", grid.keys())
print("Nb_bars_above in grid:", 'Nb_bars_above' in grid)
if 'Nb_bars_above' in grid:
    print("Nb_bars_above value:", grid['Nb_bars_above'])

# Verify wfo.py extract_bounds
from wfo import optimize_parameters, WFOSettings

# Mock Settings
settings = WFOSettings()
settings.optimization_method = 'bayesian'

print("\nTesting wfo logic (mock)...")
try:
    # We can't easily run optimize_parameters without data, but we can check the bayesian logic block if we extract it or dry run
    # Let's just check if extract_bounds works for [5] by importing it if possible, or defining it similarly
    
    bounds = grid['Nb_bars_above']
    
    def extract_bounds(bounds):
        if isinstance(bounds, (list, tuple)):
            if len(bounds) == 1 and isinstance(bounds[0], (int, float)):
                return bounds[0], bounds[0], 1
            if len(bounds) == 3 and all(isinstance(b, (int, float)) for b in bounds[:2]):
                return bounds
            if len(bounds) >= 2 and all(isinstance(b, (int, float)) for b in bounds[:2]):
                step = max(abs(bounds[1] - bounds[0]), 1)
                return bounds[0], bounds[-1], step
        raise ValueError(f"Unsupported bounds format: {bounds}")
        
    print(f"Extract bounds for {bounds}: {extract_bounds(bounds)}")
    
except Exception as e:
    print(f"Error: {e}")
