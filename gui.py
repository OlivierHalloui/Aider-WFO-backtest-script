import os
import json
import queue
import threading
import traceback
import time

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from config import (
    DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_TIMEFRAME, DEFAULT_DATA_FILE, DEFAULT_PARAM_GRID
)
from wfo import OptimizationInterrupted
from main import run_optimization


class OptimizationControl:
    """Manage cooperative stop/pause behavior for long-running optimizations."""

    def __init__(self):
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()

    def request_stop(self):
        self.stop_event.set()
        self.pause_event.clear()

    def should_stop(self) -> bool:
        return self.stop_event.is_set()

    def request_pause(self):
        self.pause_event.set()

    def resume(self):
        self.pause_event.clear()

    def wait_if_paused(self, logger=None):
        notified = False
        while self.pause_event.is_set() and not self.stop_event.is_set():
            if logger and not notified:
                logger("Optimization paused. Waiting to resume...")
                notified = True
            time.sleep(0.25)

class ConfigGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ATDMF Strategy WFO Optimizer")
        self.geometry("800x600")
        self.timeframe_options = ['1s', '5s', '1m', '5m', '1h']
        self.int_params = {'timeperiod', 'fenetre_lowest', 'longueur_mediane', 'Nb_bars_above', 'user_exit_sma_length'}
        
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
            'run_final': False,
            'generate_report': True
        }
        self.performance_var = tk.StringVar(value="Speed: N/A | ETA: N/A")
        self.window_progress_var = tk.StringVar(value="Window Progress: 0/0")
        self.generate_report_var = tk.BooleanVar(value=self.config.get('generate_report', True))
        
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
        self.is_running = False
        self.current_thread = None
        self.control = None
        self.stop_requested = False
        self.apply_config_to_widgets()

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
        self.timeframe_combo = ttk.Combobox(tab, values=self.timeframe_options, state='readonly')
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
        self.combo_var = tk.StringVar(value="Combinations: calculating...")
        self.combo_update_job = None
        row = 0
        for param in DEFAULT_PARAM_GRID.keys():
            var = tk.BooleanVar(value=param in self.config['selected_params'])
            self.param_vars[param] = var
            ttk.Checkbutton(tab, text=param, variable=var, command=self.schedule_combo_update).grid(row=row, column=0, sticky='w')
            
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
            for entry in (min_entry, max_entry, step_entry):
                entry.bind("<KeyRelease>", lambda e: self.schedule_combo_update())
                entry.bind("<FocusOut>", lambda e: self.schedule_combo_update())
            row += 1
        
        ttk.Button(tab, text="Reset to Defaults", command=self.reset_params).grid(row=row, column=0, columnspan=7)
        row += 1
        ttk.Label(tab, textvariable=self.combo_var).grid(row=row, column=0, columnspan=7, sticky='w', pady=(5, 0))
        self.schedule_combo_update(immediate=True)
    
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
        self.schedule_combo_update(immediate=True)

    def schedule_combo_update(self, immediate=False):
        if hasattr(self, 'combo_update_job') and self.combo_update_job:
            self.after_cancel(self.combo_update_job)
            self.combo_update_job = None
        if immediate:
            self.update_combo_label()
        else:
            self.combo_update_job = self.after(300, self.update_combo_label)

    def update_combo_label(self):
        try:
            total = self._calculate_combination_count()
            self.combo_var.set(f"Combinations: {int(total)}")
        except Exception as err:
            self.combo_var.set(f"Combinations: N/A ({err})")

    def _calculate_combination_count(self):
        selected_params = [p for p, v in self.param_vars.items() if v.get()]
        if not selected_params:
            return 0
        total = 1
        for param in selected_params:
            min_entry, max_entry, step_entry = self.param_entries[param]
            min_val = float(min_entry.get())
            max_val = float(max_entry.get())
            step_val = float(step_entry.get())
            if step_val <= 0:
                raise ValueError("step must be > 0")
            if max_val < min_val:
                raise ValueError("max < min")
            if param in self.int_params:
                min_i = int(min_val)
                max_i = int(max_val)
                step_i = max(1, int(round(step_val)))
                count = ((max_i - min_i) // step_i) + 1
            else:
                count = int(np.floor((max_val - min_val) / step_val + 1e-9)) + 1
            if count <= 0:
                raise ValueError("no values")
            total *= count
        return total
    
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

        ttk.Checkbutton(tab, text="Generate PDF Report", variable=self.generate_report_var).pack(anchor='w')
        
        ttk.Button(tab, text="Save Config", command=self.save_config).pack(pady=5)
        ttk.Button(tab, text="Load Config", command=self.load_config).pack(pady=5)
        
        self.run_button = ttk.Button(tab, text="Run Optimization", command=self.run_optimization_thread)
        self.run_button.pack(pady=10)

        self.stop_button = ttk.Button(tab, text="Stop", command=self.stop_optimization, state='disabled')
        self.stop_button.pack(pady=5)

        ttk.Label(tab, textvariable=self.performance_var).pack(anchor='w', padx=5)
        ttk.Label(tab, textvariable=self.window_progress_var).pack(anchor='w', padx=5, pady=(0, 5))
        
        self.status_text = tk.Text(tab, height=10, state='disabled')
        self.status_text.pack(fill='both', expand=True)

    def _set_entry_value(self, entry_widget, value):
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, str(value))

    def _ensure_config_defaults(self, config):
        config.setdefault('selected_params', list(DEFAULT_PARAM_GRID.keys()))
        for param, defaults in DEFAULT_PARAM_GRID.items():
            config.setdefault(f'{param}_min', defaults[0])
            config.setdefault(f'{param}_max', defaults[1])
            config.setdefault(f'{param}_step', defaults[2])
        return config

    def apply_config_to_widgets(self):
        config = self._ensure_config_defaults(self.config.copy())
        self.config = config

        # Dates & data settings
        self._set_entry_value(self.start_date_entry, config.get('start_date', DEFAULT_START_DATE))
        self._set_entry_value(self.end_date_entry, config.get('end_date', DEFAULT_END_DATE))
        self.timeframe_combo.set(config.get('timeframe', DEFAULT_TIMEFRAME))
        self.data_source_var.set(config.get('from_file', True))
        self._set_entry_value(self.file_path_entry, config.get('file_path', DEFAULT_DATA_FILE))

        # Parameters
        selected_params = config.get('selected_params', list(DEFAULT_PARAM_GRID.keys()))
        for param in DEFAULT_PARAM_GRID.keys():
            self.param_vars[param].set(param in selected_params)
            min_entry, max_entry, step_entry = self.param_entries[param]
            self._set_entry_value(min_entry, config.get(f'{param}_min', DEFAULT_PARAM_GRID[param][0]))
            self._set_entry_value(max_entry, config.get(f'{param}_max', DEFAULT_PARAM_GRID[param][1]))
            self._set_entry_value(step_entry, config.get(f'{param}_step', DEFAULT_PARAM_GRID[param][2]))

        # Metrics
        self.metric1_combo.set(config.get('metric1_name', 'sharpe_ratio'))
        self.metric2_combo.set(config.get('metric2_name', 'total_return'))
        self._set_entry_value(self.weight1_entry, config.get('weight_metric1', 1.0))
        self._set_entry_value(self.weight2_entry, config.get('weight_metric2', 0.0))

        # WFO settings
        self._set_entry_value(self.n_windows_entry, config.get('n_windows', 1))
        self._set_entry_value(self.train_size_entry, config.get('train_size', 0.5))
        self.anchored_var.set(config.get('anchored', False))
        self.opt_method_combo.set(config.get('optimization_method', 'bayesian'))
        self.backend_combo.set(config.get('parallel_backend', 'dask'))
        self._set_entry_value(self.max_workers_entry, config.get('max_workers', os.cpu_count() or 1))
        self.numba_var.set(config.get('use_numba', True))

        # Run options
        self.visualize_var.set(config.get('visualize', True))
        self.run_final_var.set(config.get('run_final', False))
        self.generate_report_var.set(config.get('generate_report', True))
        total_windows = config.get('n_windows', 0)
        self.performance_var.set("Speed: N/A | ETA: N/A")
        self.window_progress_var.set(f"Window Progress: 0/{total_windows}")

    def collect_config(self):
        try:
            config = {}
            # Dates
            config['start_date'] = self.start_date_entry.get()
            config['end_date'] = self.end_date_entry.get()
            pd.to_datetime(config['start_date'])  # Validate
            pd.to_datetime(config['end_date'])  # Validate
            timeframe_value = self.timeframe_combo.get()
            if timeframe_value not in self.timeframe_options:
                raise ValueError(f"Invalid timeframe selected: {timeframe_value}")
            config['timeframe'] = timeframe_value
            config['from_file'] = self.data_source_var.get()
            file_path_value = self.file_path_entry.get().strip()
            if config['from_file']:
                if not file_path_value:
                    raise ValueError("File path is required when 'From File' is selected.")
                if not os.path.isfile(file_path_value):
                    raise FileNotFoundError(f"Data file not found: {file_path_value}")
            config['file_path'] = file_path_value
            
            # Parameters
            selected_params = [p for p, v in self.param_vars.items() if v.get()]
            if not selected_params:
                raise ValueError("At least one parameter must be selected for optimization.")
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
            config['generate_report'] = self.generate_report_var.get()
            
            self.config = config
            self.summary_label.config(text="Configuration Summary: Valid")
            return True
        except Exception as e:
            messagebox.showerror("Validation Error", f"Invalid input: {str(e)}")
            self.summary_label.config(text="Configuration Summary: Invalid")
            return False
    
    def run_optimization_thread(self):
        if self.is_running:
            messagebox.showinfo("Run In Progress", "An optimization is already running. Please wait for it to finish.")
            return
        if not self.collect_config():
            return
        self.is_running = True
        self.stop_requested = False
        self.run_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.control = OptimizationControl()
        self.performance_var.set("Speed: N/A | ETA: N/A")
        total_windows = self.config.get('n_windows', 0)
        self.window_progress_var.set(f"Window Progress: 0/{total_windows}")
        self.summary_label.config(text="Configuration Summary: Running...")
        thread = threading.Thread(target=self.run_optimization_worker, daemon=True)
        self.current_thread = thread
        thread.start()
    
    def run_optimization_worker(self):
        success = True
        try:
            self.status_queue.put("Starting optimization...")
            run_optimization(self.config, status_callback=self.status_queue.put, control=self.control)
            self.status_queue.put("Optimization completed successfully!")
        except OptimizationInterrupted:
            success = False
            self.status_queue.put("Optimization stopped by user.")
        except Exception as e:
            success = False
            self.status_queue.put(f"Error: {str(e)}")
            self.status_queue.put(traceback.format_exc())
        finally:
            self.after(0, lambda: self.on_run_complete(success))
    
    def on_run_complete(self, success: bool):
        self.is_running = False
        self.current_thread = None
        self.run_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.control = None
        if self.stop_requested and not success:
            self.summary_label.config(text="Configuration Summary: Stopped")
        elif success:
            self.summary_label.config(text="Configuration Summary: Completed")
        else:
            self.summary_label.config(text="Configuration Summary: Error")
        self.stop_requested = False
    
    def check_status_queue(self):
        try:
            while True:
                msg = self.status_queue.get_nowait()
                if isinstance(msg, dict):
                    self.handle_status_payload(msg)
                else:
                    self.append_status_text(str(msg))
        except queue.Empty:
            pass
        self.after(100, self.check_status_queue)

    def append_status_text(self, text: str):
        self.status_text.config(state='normal')
        self.status_text.insert(tk.END, text + '\n')
        self.status_text.config(state='disabled')
        self.status_text.see(tk.END)

    def handle_status_payload(self, payload: dict):
        if payload.get('type') != 'stats':
            return
        speed = payload.get('speed', 0.0)
        eta = payload.get('eta', 0.0)
        window = payload.get('window', 0)
        total_windows = self.config.get('n_windows', 0)
        self.performance_var.set(f"Speed: {speed:.2f} combos/s | ETA: {self.format_duration(eta)}")
        self.window_progress_var.set(f"Window Progress: {window}/{total_windows}")

        window_metrics = payload.get('window_metrics') or {}
        in_sample = window_metrics.get('in_sample')
        out_sample = window_metrics.get('out_sample')
        lines = []
        if in_sample:
            lines.append(
                f"Window {window} In-Sample -> Return: {in_sample.get('return', 0):.2f}% | "
                f"Sharpe: {in_sample.get('sharpe', 0):.2f} | Trades: {in_sample.get('n_trades', 0)}"
            )
        if out_sample:
            lines.append(
                f"Window {window} OOS -> Return: {out_sample.get('return', 0):.2f}% | "
                f"Sharpe: {out_sample.get('sharpe', 0):.2f} | Trades: {out_sample.get('n_trades', 0)}"
            )
        if lines:
            self.append_status_text("\n".join(lines))

    def format_duration(self, seconds: float) -> str:
        seconds = int(max(0, round(seconds)))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours:d}h {minutes:02d}m {secs:02d}s"
        if minutes:
            return f"{minutes:d}m {secs:02d}s"
        return f"{secs:d}s"

    def stop_optimization(self):
        if not self.is_running or not self.control:
            return
        self.stop_requested = True
        self.control.request_stop()
        self.summary_label.config(text="Configuration Summary: Stopping...")
        self.append_status_text("Stop requested... waiting for current window to finish.")
        self.stop_button.config(state='disabled')
    
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
            try:
                with open(filename, 'r') as f:
                    loaded_config = json.load(f)
            except (OSError, json.JSONDecodeError) as err:
                messagebox.showerror("Load Error", f"Could not load configuration: {err}")
                return

            merged_config = self.config.copy()
            merged_config.update(loaded_config)
            self.config = self._ensure_config_defaults(merged_config)
            self.apply_config_to_widgets()
            self.summary_label.config(text=f"Configuration Summary: Loaded {os.path.basename(filename)}")
