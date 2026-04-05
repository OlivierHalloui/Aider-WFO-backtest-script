# Import necessary libraries for main script

import logging
import pandas as pd
import numpy as np
import datetime
import os
import traceback
import time
import sys
from typing import Optional, Dict, Callable, Any

logger = logging.getLogger(__name__)
from config import (
    DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_TIMEFRAME, DEFAULT_DATA_FILE,
    DEFAULT_PARAM_GRID, DEFAULT_STRATEGY_MODE, DEFAULT_STRATEGY_ID, WFOSettings
)
from data_loading import get_dates, load_data
from strategy_adapters import resolve_strategy_adapter
from wfo import walk_forward_optimization, OptimizationInterrupted
from adaptive_optimization import adaptive_continuous_optimization
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
        'user_exit_sma_length': "Longueur SMA sortie",
        'sar_start': "Parabolic SAR start",
        'sar_increment': "Parabolic SAR increment",
        'sar_maximum': "Parabolic SAR maximum",
        'macd_fast_length': "MACD fast length",
        'macd_slow_length': "MACD slow length",
        'macd_signal_length': "MACD signal length"
    }
    
    # ATDMF Strategy author's expert-chosen default parameters (not derived from ranges)
    strategy_params = {
        'timeperiod': 20,
        'StDev': 2.0,
        'matype': 0,
        'coeff_medianeBBW': 1.1,
        'coef_mediane': 1.0,
        'Nb_bars_above' : 5,
        'fenetre_lowest': 30,
        'seuil_lowest': 3.5,
        'longueur_mediane': 100,
        'nb_bars_under_bbw_mini': 4,
        'nb_bars_entre_bb': 5,
        'user_exit_sma_length': 20,
        'sar_start': 0.02,
        'sar_increment': 0.02,
        'sar_maximum': 0.2,
        'macd_fast_length': 12,
        'macd_slow_length': 26,
        'macd_signal_length': 9
    }
    
    # Use the canonical parameter ranges from config module (single source of truth)
    default_ranges = DEFAULT_PARAM_GRID

    # Get selected parameters from config
    selected_params = config.get('selected_params', [])
    if not selected_params:
         # Fallback if list is empty (shouldn't happen usually if app.py works right, but good for safety)
         selected_params = list(default_ranges.keys())

    param_grid = {}
    
    # 1. Add Selected Parameters (Ranges)
    for param in selected_params:
        if param in default_ranges:
            default_min, default_max, default_step = default_ranges[param]
            min_val = config.get(f'{param}_min', default_min)
            max_val = config.get(f'{param}_max', default_max)
            step = config.get(f'{param}_step', default_step)
            if param in ['timeperiod', 'fenetre_lowest', 'user_exit_sma_length', 'macd_fast_length', 'macd_slow_length', 'macd_signal_length']:
                param_grid[param] = list(range(int(min_val), int(max_val) + 1, int(step)))
            else:
                decimals = max(0, int(round(-np.log10(step)))) if step > 0 else 0
                # Use count-based expansion to avoid IEEE 754 floating-point drift
                # (np.arange(min, max+step, step) can include an extra value when
                # min==max because 0.8+0.3 = 1.0999… < 1.1, so numpy includes it).
                count = max(1, int(np.floor((max_val - min_val) / step + 1e-9)) + 1)
                values = [min_val + step * i for i in range(count)]
                param_grid[param] = list(np.round(values, decimals))
    
    # 1b. Collapse SAR/MACD params to fixed defaults when their exit is disabled.
    # A param in selected_params has no effect on the objective when its exit is off —
    # optimizing it would add dead dimensions to the search space.
    _sar_active  = bool(config.get('exit_sar_enabled', True)) or bool(config.get('exit_cross_sar_sma_enabled', True))
    _macd_active = bool(config.get('exit_macd_enabled', True))

    _SAR_DEFAULTS  = {'sar_start': 0.02, 'sar_increment': 0.02, 'sar_maximum': 0.2}
    _MACD_DEFAULTS = {'macd_fast_length': 12, 'macd_slow_length': 26, 'macd_signal_length': 9}

    if not _sar_active:
        for _p, _default in _SAR_DEFAULTS.items():
            if _p in param_grid:
                logger.debug(
                    "Collapsing '%s' to fixed default %s (exit_sar_enabled=False, "
                    "exit_cross_sar_sma_enabled=False)", _p, _default
                )
                param_grid[_p] = [strategy_params.get(_p, _default)]

    if not _macd_active:
        for _p, _default in _MACD_DEFAULTS.items():
            if _p in param_grid:
                logger.debug(
                    "Collapsing '%s' to fixed default %s (exit_macd_enabled=False)", _p, _default
                )
                param_grid[_p] = [strategy_params.get(_p, _default)]

    # 2. Add Unselected Parameters (Fixed Defaults)
    # Iterate over all known strategy parameters. If not in param_grid, add default as single value.
    for param, default_val in strategy_params.items():
        if param not in param_grid and param in default_ranges: # Only consider params that are optimizable (in ranges)
             param_grid[param] = [default_val]

    strategy_mode = str(config.get('strategy_mode', 'native_atdmf')).lower()
    if strategy_mode == 'pine_imported':
        # V3 block 1 runtime (strategy_test) does not use SAR/MACD exits.
        # Keep fixed values to avoid useless cartesian expansion.
        param_grid['exit_sar_enabled'] = [False]
        param_grid['exit_macd_enabled'] = [False]
        param_grid['exit_macd_type_a'] = [False]
        param_grid['exit_macd_type_b'] = [False]
    else:
        # Exit toggles (include with/without when checked)
        exit_sar_checked = bool(config.get('exit_sar_enabled', True))
        exit_macd_checked = bool(config.get('exit_macd_enabled', True))
        exit_macd_type_a_checked = bool(config.get('exit_macd_type_a', True))
        exit_macd_type_b_checked = bool(config.get('exit_macd_type_b', True))

        _sar_opt = bool(config.get('optimize_exit_sar_enabled', True))
        param_grid['exit_sar_enabled'] = [True, False] if (exit_sar_checked and _sar_opt) else [exit_sar_checked]
        if exit_macd_checked:
            _macd_opt = bool(config.get('optimize_exit_macd_enabled', True))
            param_grid['exit_macd_enabled'] = [True, False] if _macd_opt else [True]
            _ta_opt = bool(config.get('optimize_exit_macd_type_a', True))
            param_grid['exit_macd_type_a'] = [True, False] if (exit_macd_type_a_checked and _ta_opt) else [exit_macd_type_a_checked]
            _tb_opt = bool(config.get('optimize_exit_macd_type_b', True))
            param_grid['exit_macd_type_b'] = [True, False] if (exit_macd_type_b_checked and _tb_opt) else [exit_macd_type_b_checked]
        else:
            param_grid['exit_macd_enabled'] = [False]
            param_grid['exit_macd_type_a'] = [False]
            param_grid['exit_macd_type_b'] = [False]

    # Fixed execution settings (not optimized)
    param_grid['order_sizing_mode'] = [config.get('order_sizing_mode', 'percent_equity')]
    param_grid['order_fixed_cash'] = [float(config.get('order_fixed_cash', 10000.0))]
    param_grid['fees_pct'] = [float(config.get('fees_pct', 0.0))]
    param_grid['pqs_n_ref'] = [int(config.get('pqs_n_ref', 50))]
    # Entry filter toggles: [True, False] when enabled AND user opted to optimize, else fixed.
    _roc_enabled = bool(config.get('use_roc_filter', True))
    _roc_opt = bool(config.get('optimize_use_roc_filter', False))
    param_grid['use_roc_filter'] = [True, False] if (_roc_enabled and _roc_opt) else [_roc_enabled]
    _t2_active = bool(config.get('use_t2_signal', False))
    _t2_opt = bool(config.get('optimize_use_t2_signal', False))
    param_grid['use_t2_signal'] = [True, False] if (_t2_active and _t2_opt) else [_t2_active]
    # use_divergence_bb: only meaningful when T2 is active.
    if _t2_active:
        _div_bb_enabled = bool(config.get('use_divergence_bb', True))
        _div_bb_opt = bool(config.get('optimize_use_divergence_bb', False))
        param_grid['use_divergence_bb'] = [True, False] if (_div_bb_enabled and _div_bb_opt) else [_div_bb_enabled]
    else:
        param_grid['use_divergence_bb'] = [False]
    # Phase 5 exit toggles: [True, False] when enabled AND user opted to optimize, else fixed.
    _cross_sar_sma = bool(config.get('exit_cross_sar_sma_enabled', True))
    _cross_opt = bool(config.get('optimize_exit_cross_sar_sma_enabled', False))
    param_grid['exit_cross_sar_sma_enabled'] = [True, False] if (_cross_sar_sma and _cross_opt) else [_cross_sar_sma]
    _retour_bb = bool(config.get('exit_retour_bb_enabled', False))
    _retour_opt = bool(config.get('optimize_exit_retour_bb_enabled', False))
    param_grid['exit_retour_bb_enabled'] = [True, False] if (_retour_bb and _retour_opt) else [_retour_bb]
    _regline = bool(config.get('exit_regline_enabled', False))
    _regline_opt = bool(config.get('optimize_exit_regline_enabled', False))
    param_grid['exit_regline_enabled'] = [True, False] if (_regline and _regline_opt) else [_regline]
    _volat_down = bool(config.get('exit_volat_down_enabled', False))
    _volat_opt = bool(config.get('optimize_exit_volat_down_enabled', False))
    param_grid['exit_volat_down_enabled'] = [True, False] if (_volat_down and _volat_opt) else [_volat_down]
    param_grid['macd_ma_type'] = [str(config.get('macd_ma_type', 'sma'))]
    # Strategy direction — fixed per run (optimize long and short separately)
    param_grid['strategy_direction'] = [str(config.get('strategy_direction', 'long_only'))]
    # Fixed T0/T1 strategy params (Pine V6 defaults — not in optimisation grid)
    param_grid['depassement_sma_roc'] = [float(config.get('depassement_sma_roc', 0.01))]
    param_grid['roc_max_t1'] = [float(config.get('roc_max_t1', 100.0))]
    param_grid['nb_bars_left_pivot'] = [int(config.get('nb_bars_left_pivot', 2))]
    param_grid['nb_bars_right_pivot'] = [int(config.get('nb_bars_right_pivot', 2))]
    param_grid['nombre_periodes_reglin'] = [int(config.get('nombre_periodes_reglin', 15))]
    param_grid['i_bars_back'] = [int(config.get('i_bars_back', 1))]
    param_grid['seuil_overbought_bb'] = [float(config.get('seuil_overbought_bb', 0.85))]

    return param_grid


def display_default_parameters():
    """
    Display all default indicator parameters and their values.
    """
    logger.info("=== Default Indicator Parameters ===")

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
    
    logger.debug(f"{'Parameter':<{max_param_len+2}} | {'Description':<{max_desc_len+2}} | {'Default Value'}")
    logger.debug("-" * (max_param_len+2 + max_desc_len+2 + 20))

    for param, value in strategy_params.items():
        if param in param_descriptions:
            logger.debug(f"{param:<{max_param_len+2}} | {param_descriptions[param]:<{max_desc_len+2}} | {value}")
        else:
            logger.debug(f"{param:<{max_param_len+2}} | {'No description available':<{max_desc_len+2}} | {value}")



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
    settings.optimization_regime = config.get('optimization_regime', 'classic')
    settings.parallel_backend = config.get('parallel_backend', 'dask')
    settings.max_workers = config.get('max_workers', os.cpu_count() or 1)
    settings.use_numba = config.get('use_numba', True)
    settings.patience_level = config.get('patience_level', 'Medium')
    settings.max_trials = config.get('max_trials', 200)
    settings.neighbor_count = config.get('neighbor_count', 5)
    settings.nn_min_samples = config.get('nn_min_samples', 500)
    settings.nn_candidate_pool_size = config.get('nn_candidate_pool_size', 3000)
    settings.nn_top_k = config.get('nn_top_k', 250)
    settings.nn_exploration_ratio = config.get('nn_exploration_ratio', 0.15)
    settings.nn_hidden_size = config.get('nn_hidden_size', 32)
    settings.nn_epochs = config.get('nn_epochs', 60)
    settings.nn_learning_rate = config.get('nn_learning_rate', 0.01)
    settings.nn_l2 = config.get('nn_l2', 1e-4)
    settings.adaptive_train_bars = config.get('adaptive_train_bars', 5000)
    settings.adaptive_cycle_bars = config.get('adaptive_cycle_bars', 5000)
    settings.adaptive_trials_per_cycle = config.get('adaptive_trials_per_cycle', 150)
    settings.adaptive_candidate_pool_size = config.get('adaptive_candidate_pool_size', 3000)
    settings.adaptive_keep_ratio = config.get('adaptive_keep_ratio', 0.40)
    settings.adaptive_exploration_ratio = config.get('adaptive_exploration_ratio', 0.20)
    settings.adaptive_min_values_per_param = config.get('adaptive_min_values_per_param', 2)
    settings.adaptive_decay = config.get('adaptive_decay', 0.98)
    settings.adaptive_ucb_beta = config.get('adaptive_ucb_beta', 0.75)
    settings.adaptive_warmup_trials = config.get('adaptive_warmup_trials', 300)
    settings.adaptive_max_cycles = config.get('adaptive_max_cycles', 0)
    settings.adaptive_oos_weight = config.get('adaptive_oos_weight', 2.0)
    settings.use_t2_signal = bool(config.get('use_t2_signal', False))
    settings.pqs_n_ref = int(config.get('pqs_n_ref', 50))
    settings.exit_sar_enabled = config.get('exit_sar_enabled', True)
    settings.exit_macd_enabled = config.get('exit_macd_enabled', True)
    settings.exit_macd_type_a = config.get('exit_macd_type_a', True)
    settings.exit_macd_type_b = config.get('exit_macd_type_b', True)
    settings.exit_cross_sar_sma_enabled = config.get('exit_cross_sar_sma_enabled', True)
    settings.exit_retour_bb_enabled = config.get('exit_retour_bb_enabled', False)
    settings.exit_regline_enabled = config.get('exit_regline_enabled', False)
    settings.exit_volat_down_enabled = config.get('exit_volat_down_enabled', False)
    settings.macd_ma_type = str(config.get('macd_ma_type', 'sma'))

    return settings

def run_optimization(config, status_callback: Optional[Callable[[Any], None]] = None, control: Optional[Any] = None):
    """
    Run the optimization process using config dict.
    """
    def log(message: str):
        logger.info(message)
        if status_callback:
            status_callback(message)

    strategy_mode = str(config.get("strategy_mode", DEFAULT_STRATEGY_MODE)).lower()
    strategy_id = str(config.get("strategy_id", DEFAULT_STRATEGY_ID))
    strategy_adapter = resolve_strategy_adapter(
        strategy_mode=strategy_mode,
        strategy_id=strategy_id,
        config=config,
    )

    log("Preparing date range...")
    start_date, end_date = get_dates(config)
    
    log("\nLoading data...")
    timeframe = config.get('timeframe', DEFAULT_TIMEFRAME)
    
    from_file = config.get('from_file', True)
    file_path = config.get('file_path', DEFAULT_DATA_FILE)
    
    if from_file:
        df = load_data(start_date, end_date, timeframe, from_file=True, file_path=file_path)
    else:
        df = load_data(start_date, end_date, timeframe, from_file=False)
    
    log(f"Loaded {len(df)} bars of data from {start_date} to {end_date}")
    
    display_default_parameters()

    param_grid = get_param_grid(config)
    log(f"Parameter grid: {param_grid}")
    
    metrics_info = get_metrics_info(config)
    log(
        f"Metrics: {metrics_info['metric1_name']} (weight: {metrics_info['weight_metric1']:.2f}), "
        f"{metrics_info['metric2_name']} (weight: {metrics_info['weight_metric2']:.2f})"
    )
    
    settings = get_wfo_settings(config)
    
    if str(settings.optimization_regime).lower() == 'adaptive_continuous':
        log("Starting adaptive continuous optimization...")
        wfo_results = adaptive_continuous_optimization(
            df,
            param_grid=param_grid,
            metrics_info=metrics_info,
            timeframe=timeframe,
            settings=settings,
            status_callback=status_callback,
            control=control,
            strategy_adapter=strategy_adapter,
        )
        log("Adaptive continuous optimization completed.")
    else:
        log("Starting walk-forward optimization...")
        wfo_results = walk_forward_optimization(
            df,
            param_grid=param_grid,
            metrics_info=metrics_info,
            timeframe=timeframe,
            settings=settings,
            status_callback=status_callback,
            control=control,
            strategy_adapter=strategy_adapter,
        )
        log("Walk-forward optimization completed.")
    
    results_dir = "WFO_Results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if wfo_results['out_of_sample_performance']:
        oos_df = pd.DataFrame(wfo_results['out_of_sample_performance'])
        oos_df.to_csv(f"{results_dir}/oos_performance_{timestamp}.csv", index=False)
    
    params_df = pd.DataFrame(wfo_results['best_params'])
    params_df.to_csv(f"{results_dir}/best_parameters_{timestamp}.csv", index=False)
    
    with open(f"{results_dir}/wfo_settings_{timestamp}.txt", 'w') as f:
        for key, value in wfo_results['settings'].items():
            f.write(f"{key}: {value}\n")
    
    log(f"\nResults saved to {results_dir} directory")
    
    visualize = config.get('visualize', True)
    if visualize:
        try:
            from vectorbtpro.data.base import OHLCV
            df = OHLCV.from_df(df)
        except (ImportError, ValueError) as e:
            log(f"Note: Could not register DataFrame with OHLCV: {e}")
       
        visualize_wfo_results(wfo_results, df)
        create_parameter_performance_map(wfo_results)
        visualize_robustness_metrics(wfo_results)

    if config.get('generate_report', True):
        log("Generating PDF report...")
        integrate_report_generation(wfo_results, df, strategy_name="ATDMF Strategy")
    else:
        log("Skipping PDF report generation as requested.")

    run_final = config.get('run_final', False)
    if run_final:
        avg_params = {}
        for param in param_grid.keys():
            if param in ['timeperiod', 'fenetre_lowest', 'user_exit_sma_length']:
                avg_params[param] = int(round(params_df[param].mean()))
            else:
                avg_params[param] = round(params_df[param].mean(), 2)
        
        log(f"Running final backtest with parameters: {avg_params}")
        final_portfolio = strategy_adapter.run_backtest(
            df,
            avg_params,
            timeframe=timeframe,
            return_portfolio=True,
        )
        
        avg_pl_per_trade = 0
        if len(final_portfolio.trades) > 0:
            avg_pl_per_trade = (final_portfolio.total_return*100) / len(final_portfolio.trades)
        
        log("\n=== Final Backtest Results ===")
        log(f"Total Return: {final_portfolio.total_return * 100:.2f}%")
        log(f"Sharpe Ratio: {final_portfolio.sharpe_ratio:.2f}")
        log(f"Max Drawdown: {final_portfolio.max_drawdown * 100:.2f}%")
        log(f"Win Rate: {final_portfolio.trades.win_rate *100:.2f}%")
        log(f"Average P&L per Trade: {avg_pl_per_trade:.2f}%")
        log(f"Number of Trades: {len(final_portfolio.trades)}")
        log(f"Calmar Ratio: {final_portfolio.calmar_ratio:.2f}")
        log(f"Sortino Ratio: {final_portfolio.sortino_ratio:.2f}")
        
        final_portfolio.plot().show()

def main():
    """Main function to run the WFO process."""
    logger.info("=== ATDMF Strategy Walk-Forward Optimization ===")
    logger.info("This script performs Walk-Forward Optimization on the ATDMF strategy.")
    
    if '--config' in sys.argv:
        import json
        try:
            config_idx = sys.argv.index('--config') + 1
            config_path = sys.argv[config_idx]
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info("Loading configuration from %s...", config_path)
            run_optimization(config)
        except Exception as e:
            logger.warning("Error loading config: %s", e)
            traceback.print_exc()
    elif '--no-gui' in sys.argv:
        # Console mode
        config = {
            'start_date': DEFAULT_START_DATE,
            'end_date': DEFAULT_END_DATE,
            'timeframe': DEFAULT_TIMEFRAME,
            'strategy_mode': DEFAULT_STRATEGY_MODE,
            'strategy_id': DEFAULT_STRATEGY_ID,
            'from_file': True,
            'file_path': DEFAULT_DATA_FILE,
            'optimization_method': 'grid',
            'n_windows': 1,
            'train_size': 0.5,
            'visualize': False,
            'generate_report': False
        }
        run_optimization(config)
    else:
        # GUI mode
        logger.info("Launching Streamlit GUI...")
        import subprocess
        try:
            subprocess.run(["streamlit", "run", "app.py"], check=True)
        except FileNotFoundError:
            logger.warning("Error: 'streamlit' command not found. Please install it using 'pip install streamlit'.")
        except KeyboardInterrupt:
            logger.info("Streamlit GUI stopped.")

if __name__ == "__main__":
    main()
