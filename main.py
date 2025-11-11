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
    
    # Get selected parameters from config
    selected_params = config.get('selected_params', list(DEFAULT_PARAM_GRID.keys()))
    
    param_grid = {}
    for param in selected_params:
        if param in default_ranges:
            default_min, default_max, default_step = default_ranges[param]
            min_val = config.get(f'{param}_min', default_min)
            max_val = config.get(f'{param}_max', default_max)
            step = config.get(f'{param}_step', default_step)
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
    
    return settings

def run_optimization(config):
    """
    Run the optimization process using config dict.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    """
    # Get dates
    start_date, end_date = get_dates(config)
    
    # Load data
    print("\nLoading data...")
    timeframe = config.get('timeframe', DEFAULT_TIMEFRAME)
    
    from_file = config.get('from_file', True)
    file_path = config.get('file_path', DEFAULT_DATA_FILE)
    
    if from_file:
        df = load_data(start_date, end_date, timeframe, from_file=True, file_path=file_path)
    else:
        df = load_data(start_date, end_date, timeframe, from_file=False)
    
    print(f"Loaded {len(df)} bars of data from {start_date} to {end_date}")
    
    # Display default parameters
    display_default_parameters()

    # Get parameter grid
    param_grid = get_param_grid(config)
    print(f"Parameter grid: {param_grid}")
    
    # Get metrics information
    metrics_info = get_metrics_info(config)
    print(f"Metrics: {metrics_info['metric1_name']} (weight: {metrics_info['weight_metric1']:.2f}), "
          f"{metrics_info['metric2_name']} (weight: {metrics_info['weight_metric2']:.2f})")
    
    # Get WFO settings
    settings = get_wfo_settings(config)
    
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
    visualize = config.get('visualize', True)
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
    run_final = config.get('run_final', False)
    if run_final:
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

# GUI Class
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import threading
import sys
import queue

class ConfigGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ATDMF Strategy WFO Optimizer")
        self.geometry("800x600")
        
        # Default config
        self.config = {
            'start_date': DEFAULT_START_DATE,
            'end_date': DEFAULT_END_DATE,
            'timeframe': DEFAULT_TIMEFRAME,
            'from_file': True,
            'file_path': DEFAULT_DATA_FILE,
            'selected_params': list(DEFAULT_PARAM_GRID.keys()),
            'metric1_name': 'sharpe_ratio',
            'metric2_name': 'total_return',
            'weight_metric1': 1.0,
            'weight_metric2': 0.0,
            'n_windows': 1,
            'train_size': 0.5,
            'anchored': False,
            'optimization_method': 'bayesian',
            'parallel_backend': 'dask',
            'max_workers': os.cpu_count() or 1,
            'use_numba': True,
            'visualize': True,
            'run_final': False
        }
        
        # Add ranges for params
        for param in DEFAULT_PARAM_GRID.keys():
            self.config[f'{param}_min'] = DEFAULT_PARAM_GRID[param][0]
            self.config[f'{param}_max'] = DEFAULT_PARAM_GRID[param][1]
            self.config[f'{param}_step'] = DEFAULT_PARAM_GRID[param][2]
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)
        
        # Create tabs
        self.create_dates_tab()
        self.create_parameters_tab()
        self.create_metrics_tab()
        self.create_wfo_settings_tab()
        self.create_run_tab()
        
        # Status queue for threading
        self.status_queue = queue.Queue()
        self.after(100, self.check_status_queue)
    
    def create_dates_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Dates & Data")
        
        ttk.Label(tab, text="Start Date (YYYY-MM-DD):").grid(row=0, column=0, sticky='w')
        self.start_date_entry = ttk.Entry(tab)
        self.start_date_entry.insert(0, self.config['start_date'])
        self.start_date_entry.grid(row=0, column=1)
        
        ttk.Label(tab, text="End Date (YYYY-MM-DD):").grid(row=1, column=0, sticky='w')
        self.end_date_entry = ttk.Entry(tab)
        self.end_date_entry.insert(0, self.config['end_date'])
        self.end_date_entry.grid(row=1, column=1)
        
        ttk.Label(tab, text="Timeframe:").grid(row=2, column=0, sticky='w')
        self.timeframe_combo = ttk.Combobox(tab, values=['1s', '5s', '1m', '5m', '1h'])
        self.timeframe_combo.set(self.config['timeframe'])
        self.timeframe_combo.grid(row=2, column=1)
        
        ttk.Label(tab, text="Data Source:").grid(row=3, column=0, sticky='w')
        self.data_source_var = tk.BooleanVar(value=self.config['from_file'])
        ttk.Radiobutton(tab, text="From File", variable=self.data_source_var, value=True).grid(row=3, column=1, sticky='w')
        ttk.Radiobutton(tab, text="From Binance", variable=self.data_source_var, value=False).grid(row=4, column=1, sticky='w')
        
        ttk.Label(tab, text="File Path:").grid(row=5, column=0, sticky='w')
        self.file_path_entry = ttk.Entry(tab)
        self.file_path_entry.insert(0, self.config['file_path'])
        self.file_path_entry.grid(row=5, column=1)
        ttk.Button(tab, text="Browse", command=self.browse_file).grid(row=5, column=2)
    
    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, filename)
    
    def create_parameters_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Parameters")
        
        self.param_vars = {}
        self.param_entries = {}
        row = 0
        for param in DEFAULT_PARAM_GRID.keys():
            var = tk.BooleanVar(value=param in self.config['selected_params'])
            self.param_vars[param] = var
            ttk.Checkbutton(tab, text=param, variable=var).grid(row=row, column=0, sticky='w')
            
            ttk.Label(tab, text="Min:").grid(row=row, column=1)
            min_entry = ttk.Entry(tab)
            min_entry.insert(0, self.config[f'{param}_min'])
            min_entry.grid(row=row, column=2)
            
            ttk.Label(tab, text="Max:").grid(row=row, column=3)
            max_entry = ttk.Entry(tab)
            max_entry.insert(0, self.config[f'{param}_max'])
            max_entry.grid(row=row, column=4)
            
            ttk.Label(tab, text="Step:").grid(row=row, column=5)
            step_entry = ttk.Entry(tab)
            step_entry.insert(0, self.config[f'{param}_step'])
            step_entry.grid(row=row, column=6)
            
            self.param_entries[param] = (min_entry, max_entry, step_entry)
            row += 1
        
        ttk.Button(tab, text="Reset to Defaults", command=self.reset_params).grid(row=row, column=0, columnspan=7)
    
    def reset_params(self):
        for param in DEFAULT_PARAM_GRID.keys():
            self.param_vars[param].set(True)
            min_val, max_val, step_val = DEFAULT_PARAM_GRID[param]
            self.param_entries[param][0].delete(0, tk.END)
            self.param_entries[param][0].insert(0, min_val)
            self.param_entries[param][1].delete(0, tk.END)
            self.param_entries[param][1].insert(0, max_val)
            self.param_entries[param][2].delete(0, tk.END)
            self.param_entries[param][2].insert(0, step_val)
    
    def create_metrics_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Metrics")
        
        ttk.Label(tab, text="Primary Metric:").grid(row=0, column=0, sticky='w')
        self.metric1_combo = ttk.Combobox(tab, values=['max_drawdown', 'sharpe_ratio', 'total_return', 'avg_gain_per_trade', 'avg_loss_per_trade', 'win_rate', 'avg_pl_per_trade'])
        self.metric1_combo.set(self.config['metric1_name'])
        self.metric1_combo.grid(row=0, column=1)
        
        ttk.Label(tab, text="Weight:").grid(row=0, column=2)
        self.weight1_entry = ttk.Entry(tab)
        self.weight1_entry.insert(0, self.config['weight_metric1'])
        self.weight1_entry.grid(row=0, column=3)
        
        ttk.Label(tab, text="Secondary Metric:").grid(row=1, column=0, sticky='w')
        self.metric2_combo = ttk.Combobox(tab, values=['max_drawdown', 'sharpe_ratio', 'total_return', 'avg_gain_per_trade', 'avg_loss_per_trade', 'win_rate', 'avg_pl_per_trade'])
        self.metric2_combo.set(self.config['metric2_name'])
        self.metric2_combo.grid(row=1, column=1)
        
        ttk.Label(tab, text="Weight:").grid(row=1, column=2)
        self.weight2_entry = ttk.Entry(tab)
        self.weight2_entry.insert(0, self.config['weight_metric2'])
        self.weight2_entry.grid(row=1, column=3)
    
    def create_wfo_settings_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="WFO Settings")
        
        ttk.Label(tab, text="Number of Windows:").grid(row=0, column=0, sticky='w')
        self.n_windows_entry = ttk.Entry(tab)
        self.n_windows_entry.insert(0, self.config['n_windows'])
        self.n_windows_entry.grid(row=0, column=1)
        
        ttk.Label(tab, text="Training Size (0-1):").grid(row=1, column=0, sticky='w')
        self.train_size_entry = ttk.Entry(tab)
        self.train_size_entry.insert(0, self.config['train_size'])
        self.train_size_entry.grid(row=1, column=1)
        
        self.anchored_var = tk.BooleanVar(value=self.config['anchored'])
        ttk.Checkbutton(tab, text="Anchored WFO", variable=self.anchored_var).grid(row=2, column=0, columnspan=2, sticky='w')
        
        ttk.Label(tab, text="Optimization Method:").grid(row=3, column=0, sticky='w')
        self.opt_method_combo = ttk.Combobox(tab, values=['grid', 'bayesian', 'optuna'])
        self.opt_method_combo.set(self.config['optimization_method'])
        self.opt_method_combo.grid(row=3, column=1)
        
        ttk.Label(tab, text="Parallel Backend:").grid(row=4, column=0, sticky='w')
        self.backend_combo = ttk.Combobox(tab, values=['dask', 'ray', 'pathos', 'threadpool'])
        self.backend_combo.set(self.config['parallel_backend'])
        self.backend_combo.grid(row=4, column=1)
        
        ttk.Label(tab, text="Max Workers:").grid(row=5, column=0, sticky='w')
        self.max_workers_entry = ttk.Entry(tab)
        self.max_workers_entry.insert(0, self.config['max_workers'])
        self.max_workers_entry.grid(row=5, column=1)
        
        self.numba_var = tk.BooleanVar(value=self.config['use_numba'])
        ttk.Checkbutton(tab, text="Use Numba Acceleration", variable=self.numba_var).grid(row=6, column=0, columnspan=2, sticky='w')
    
    def create_run_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Run")
        
        self.summary_label = ttk.Label(tab, text="Configuration Summary: Ready")
        self.summary_label.pack(pady=10)
        
        self.visualize_var = tk.BooleanVar(value=self.config['visualize'])
        ttk.Checkbutton(tab, text="Visualize Results", variable=self.visualize_var).pack(anchor='w')
        
        self.run_final_var = tk.BooleanVar(value=self.config['run_final'])
        ttk.Checkbutton(tab, text="Run Final Backtest", variable=self.run_final_var).pack(anchor='w')
        
        ttk.Button(tab, text="Save Config", command=self.save_config).pack(pady=5)
        ttk.Button(tab, text="Load Config", command=self.load_config).pack(pady=5)
        
        ttk.Button(tab, text="Run Optimization", command=self.run_optimization_thread).pack(pady=10)
        
        self.status_text = tk.Text(tab, height=10, state='disabled')
        self.status_text.pack(fill='both', expand=True)
    
    def collect_config(self):
        try:
            config = {}
            # Dates
            config['start_date'] = self.start_date_entry.get()
            config['end_date'] = self.end_date_entry.get()
            pd.to_datetime(config['start_date'])  # Validate
            pd.to_datetime(config['end_date'])  # Validate
            config['timeframe'] = self.timeframe_combo.get()
            config['from_file'] = self.data_source_var.get()
            config['file_path'] = self.file_path_entry.get()
            
            # Parameters
            selected_params = [p for p, v in self.param_vars.items() if v.get()]
            config['selected_params'] = selected_params
            for param in selected_params:
                min_val = float(self.param_entries[param][0].get())
                max_val = float(self.param_entries[param][1].get())
                step_val = float(self.param_entries[param][2].get())
                config[f'{param}_min'] = min_val
                config[f'{param}_max'] = max_val
                config[f'{param}_step'] = step_val
            
            # Metrics
            config['metric1_name'] = self.metric1_combo.get()
            config['metric2_name'] = self.metric2_combo.get()
            config['weight_metric1'] = float(self.weight1_entry.get())
            config['weight_metric2'] = float(self.weight2_entry.get())
            
            # WFO Settings
            config['n_windows'] = int(self.n_windows_entry.get())
            config['train_size'] = float(self.train_size_entry.get())
            config['anchored'] = self.anchored_var.get()
            config['optimization_method'] = self.opt_method_combo.get()
            config['parallel_backend'] = self.backend_combo.get()
            config['max_workers'] = int(self.max_workers_entry.get())
            config['use_numba'] = self.numba_var.get()
            
            # Run options
            config['visualize'] = self.visualize_var.get()
            config['run_final'] = self.run_final_var.get()
            
            self.config = config
            self.summary_label.config(text="Configuration Summary: Valid")
            return True
        except Exception as e:
            messagebox.showerror("Validation Error", f"Invalid input: {str(e)}")
            self.summary_label.config(text="Configuration Summary: Invalid")
            return False
    
    def run_optimization_thread(self):
        if not self.collect_config():
            return
        thread = threading.Thread(target=self.run_optimization_worker)
        thread.start()
    
    def run_optimization_worker(self):
        try:
            self.status_queue.put("Starting optimization...")
            run_optimization(self.config)
            self.status_queue.put("Optimization completed successfully!")
        except Exception as e:
            self.status_queue.put(f"Error: {str(e)}")
    
    def check_status_queue(self):
        try:
            while True:
                msg = self.status_queue.get_nowait()
                self.status_text.config(state='normal')
                self.status_text.insert(tk.END, msg + '\n')
                self.status_text.config(state='disabled')
                self.status_text.see(tk.END)
        except queue.Empty:
            pass
        self.after(100, self.check_status_queue)
    
    def save_config(self):
        if not self.collect_config():
            return
        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if filename:
            with open(filename, 'w') as f:
                json.dump(self.config, f, indent=4)
    
    def load_config(self):
        filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if filename:
            with open(filename, 'r') as f:
                self.config = json.load(f)
            # Update widgets (simplified, in practice update each widget)
            self.start_date_entry.delete(0, tk.END)
            self.start_date_entry.insert(0, self.config['start_date'])
            # ... similarly for others

def main():
    """Main function to run the WFO process."""
    print("=== ATDMF Strategy Walk-Forward Optimization ===")
    print("This script performs Walk-Forward Optimization on the ATDMF strategy.")
    print("It demonstrates how to properly cross-validate trading strategies to avoid overfitting.")
    
    if '--no-gui' in sys.argv:
        # Console mode
        config = {}
        start_date, end_date = get_dates(config)
        # ... rest of original main() with inputs
        # For brevity, assume original code here, but since it's refactored, use run_optimization with manual config
        config = {
            'start_date': input(f"Enter start date (YYYY-MM-DD) [default: {DEFAULT_START_DATE}]: ") or DEFAULT_START_DATE,
            'end_date': input(f"Enter end date (YYYY-MM-DD) [default: {DEFAULT_END_DATE}]: ") or DEFAULT_END_DATE,
            # ... add all inputs manually
        }
        run_optimization(config)
    else:
        # GUI mode
        app = ConfigGUI()
        app.mainloop()

if __name__ == "__main__":
    main()
