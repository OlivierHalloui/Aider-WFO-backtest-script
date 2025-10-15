// Import necessary libraries
import pandas as pd
import numpy as np
import vectorbtpro as vbt
from vectorbtpro.indicators.factory import IndicatorFactory
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from matplotlib.backends.backend_pdf import PdfPages
import talib
import datetime
import time
import os
from numba import njit
from itertools import product
from tqdm import tqdm
import time
from datetime import timedelta
import platform
import subprocess
import seaborn as sns
import io
from PIL import Image
import logging
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_START_DATE = "2025-01-19"
DEFAULT_END_DATE = "2025-01-31"
DEFAULT_TIMEFRAME = '5S'
DEFAULT_FILE_PATH = '/home/olivier/Downloads/ATDMF_strategy_V5_long/ATDMF_strategy_long_BTCFDUSD05S/Data/Binance_BTCUSDT_OHLCV_B_2025-01-19_2025-01-31_1s/Binance_BTCUSDT_OHLCV_B_2025-01-19_2025-01-31_5S.csv'

# ======================================================================
# CONFIGURATION CLASSES
# ======================================================================

class WFOSettings:
    """Configuration class for Walk-Forward Optimization settings."""
    def __init__(self):
        self.n_windows: int = 1
        self.train_size: float = 0.5
        self.anchored: bool = False
        self.optimization_metric: str = "sharpe_ratio"
        self.secondary_metric: str = "total_return"
        self.metric_weights: Tuple[float, float] = (1.0, 0.0)
        self.parallel_backend: str = "dask"
        self.use_numba: bool = True
        self.chunk_size: Optional[str] = "auto"

class StrategyConfig:
    """Configuration class for strategy parameters."""
    def __init__(self):
        self.timeperiod: int = 20
        self.StDev: float = 1.3
        self.matype: int = 0
        self.coeff_medianeBBW: float = 1.1
        self.coef_mediane: float = 1.0
        self.Nb_bars_above: int = 5
        self.fenetre_lowest: int = 30
        self.seuil_lowest: float = 3.5
        self.user_exit_sma_length: int = 20

# ======================================================================
# NUMBA-OPTIMIZED INDICATOR FUNCTIONS
# ======================================================================

@njit(cache=True)
def bbands_1d_nb(close: np.ndarray, window: int = 20, alpha: float = 2.0, ddof: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numba-optimized Bollinger Bands calculation."""
    n = len(close)
    upper_band = np.empty(n, dtype=np.float64)
    middle_band = np.empty(n, dtype=np.float64)
    lower_band = np.empty(n, dtype=np.float64)
    middle_band = vbt.indicators.nb.ma_1d_nb(close, window)
    std = vbt.indicators.nb.msd_1d_nb(close, window, ddof=ddof)
    for i in range(n):
        if np.isnan(middle_band[i]) or np.isnan(std[i]):
            upper_band[i] = np.nan
            lower_band[i] = np.nan
        else:
            upper_band[i] = middle_band[i] + alpha * std[i]
            lower_band[i] = middle_band[i] - alpha * std[i]
    return upper_band, middle_band, lower_band

@njit
def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Numba-optimized rolling mean."""
    result = np.full(len(arr), np.nan)
    cumsum = 0.0
    for i in range(len(arr)):
        if i >= window:
            cumsum -= arr[i - window]
        cumsum += arr[i]
        if i >= window - 1:
            result[i] = cumsum / window
    return result

@njit
def rolling_median(arr: np.ndarray, window: int) -> np.ndarray:
    """Numba-optimized rolling median."""
    result = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        result[i] = np.median(arr[i - window + 1:i + 1])
    return result

@njit
def rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    """Numba-optimized rolling min."""
    result = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        result[i] = np.min(arr[i - window + 1:i + 1])
    return result

@njit
def compute_bars_since_below(ecart_borne1: np.ndarray, mediane: np.ndarray, coef: float, Nb_bars_above: int) -> np.ndarray:
    """Compute bars since below signal."""
    bars_since_below = np.empty(len(ecart_borne1))
    counter = np.inf
    for i in range(len(ecart_borne1)):
        if np.isnan(mediane[i]) or np.isnan(ecart_borne1[i]):
            bars_since_below[i] = np.nan
            continue
        if ecart_borne1[i] < mediane[i] / coef:
            counter = 0
        else:
            counter += 1
        bars_since_below[i] = counter
    return bars_since_below >= Nb_bars_above

@njit
def ecart_bollinger_borne_signal_nb(prix: np.ndarray, upper_band: np.ndarray, lower_band: np.ndarray, timeperiod: int, longueur_mediane: int, coef_mediane: float, Nb_bars_above: int) -> np.ndarray:
    """Ecart Bollinger Borne signal."""
    ecart = upper_band - lower_band
    sma = vbt.indicators.nb.ma_1d_nb(prix, timeperiod)
    ecart_borne1 = ecart / sma
    mediane = rolling_median(ecart_borne1, longueur_mediane)
    return compute_bars_since_below(ecart_borne1, mediane, coef_mediane, Nb_bars_above)

@njit
def bollinger_horizontal_signal_nb(upper_band: np.ndarray, lower_band: np.ndarray, middle_band: np.ndarray, coeff_medianeBBW: float) -> np.ndarray:
    """Bollinger Horizontal signal."""
    BBW = (upper_band - lower_band) / middle_band
    MMBBW = rolling_mean(BBW, 5)
    medianeBBW = rolling_median(BBW, 200)
    seuil = medianeBBW / coeff_medianeBBW
    signal = np.zeros(len(BBW), dtype=np.int32)
    for i in range(len(BBW)):
        if np.isnan(seuil[i]):
            continue
        if BBW[i] < seuil[i] or MMBBW[i] < seuil[i]:
            signal[i] = 1
    return signal

@njit
def cross_bbw_low_signal_nb(upper_band: np.ndarray, lower_band: np.ndarray, middle_band: np.ndarray, fenetre_lowest: int, seuil_lowest: float) -> np.ndarray:
    """Cross BBW Low signal."""
    largeur_bb = (upper_band - lower_band) / middle_band
    bbw_lowest = rolling_min(largeur_bb, fenetre_lowest)
    signal = np.full(len(largeur_bb), False)
    for i in range(len(largeur_bb)):
        if np.isnan(bbw_lowest[i]) or np.isnan(largeur_bb[i]):
            continue
        signal[i] = largeur_bb[i] <= (bbw_lowest[i] * seuil_lowest)
    return signal

@njit(cache=True)
def calculate_exit_sma_nb(close: np.ndarray, user_exit_sma_length: int) -> np.ndarray:
    """Numba-optimized SMA exit signal."""
    close_1d = close if len(close.shape) == 1 else close.flatten()
    n = len(close_1d)
    sma = vbt.indicators.nb.ma_1d_nb(close_1d, user_exit_sma_length)
    signal = np.zeros(n, dtype=np.bool_)
    for i in range(1, n):
        if (not np.isnan(close_1d[i-1]) and not np.isnan(sma[i-1]) and 
            not np.isnan(close_1d[i]) and not np.isnan(sma[i]) and 
            close_1d[i-1] > sma[i-1] and close_1d[i] < sma[i]):
            signal[i] = True
    return signal

# ======================================================================
# INDICATOR FACTORIES
# ======================================================================

EcartBollingerBorne = IndicatorFactory(
    input_names=['prix', 'upper_band', 'lower_band'],
    param_names=['timeperiod', 'longueur_mediane', 'coef_mediane', 'Nb_bars_above'],
    output_names=['signal']
).with_custom_func(
    ecart_bollinger_borne_signal_nb,
    input_shapes=dict(prix='1d', upper_band='1d', lower_band='1d'),
    param_defaults=dict(timeperiod=20, longueur_mediane=100, coef_mediane=1.0, Nb_bars_above=5),
    output_shapes=dict(signal='1d')
)

BollingerHorizontal = IndicatorFactory(
    input_names=['upper_band', 'lower_band', 'middle_band'],
    param_names=['coeff_medianeBBW'],
    output_names=['signal']
).with_custom_func(
    bollinger_horizontal_signal_nb,
    input_shapes=dict(upper_band='1d', lower_band='1d', middle_band='1d'),
    param_defaults=dict(coeff_medianeBBW=1.1),
    output_shapes=dict(signal='1d')
)

CrossBBWLowSignal = IndicatorFactory(
    input_names=['upper_band', 'lower_band', 'middle_band'],
    param_names=['fenetre_lowest', 'seuil_lowest'],
    output_names=['signal']
).with_custom_func(
    cross_bbw_low_signal_nb,
    input_shapes=dict(upper_band='1d', lower_band='1d', middle_band='1d'),
    param_defaults=dict(fenetre_lowest=30, seuil_lowest=3.5),
    output_shapes=dict(signal='1d')
)

# ======================================================================
# DATA LOADING AND PREPROCESSING
# ======================================================================

def get_dates() -> Tuple[str, str]:
    """Prompt for date range input with defaults."""
    start_date = input(f"Enter start date (YYYY-MM-DD) [default: {DEFAULT_START_DATE}]: ") or DEFAULT_START_DATE
    end_date = input(f"Enter end date (YYYY-MM-DD) [default: {DEFAULT_END_DATE}]: ") or DEFAULT_END_DATE
    return start_date, end_date

def load_data(start_date: str, end_date: str, timeframe: str = DEFAULT_TIMEFRAME, from_file: bool = True, file_path: Optional[str] = None) -> pd.DataFrame:
    """Load OHLCV data."""
    try:
        if from_file and file_path:
            df = pd.read_csv(file_path)
            df['Open time'] = pd.to_datetime(df['Open time'])
            df.set_index('Open time', inplace=True)
            df = df.resample(timeframe).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            }).dropna()
            logger.info(f"Loaded data from file: {len(df)} bars")
        else:
            base_timeframe = '1s'
            df_1s = vbt.BinanceData.fetch(
                ["BTCUSDT"], 
                start=start_date, 
                end=end_date,
                timeframe=base_timeframe
            )
            df = df_1s.resample(timeframe).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            }).dropna()
            folder_name = f"Binance_BTCUSDT_OHLCV_B_{start_date}_{end_date}_{timeframe}"
            folder_path = f"./Data/{folder_name}"
            os.makedirs(folder_path, exist_ok=True)
            file_path = f"{folder_path}/Binance_BTCUSDT_OHLCV_B_{start_date}_{end_date}_{timeframe}.csv"
            df.to_csv(file_path)
            logger.info(f"Fetched and saved data: {len(df)} bars")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

# ======================================================================
# STRATEGY IMPLEMENTATION
# ======================================================================

class TradingStrategy(ABC):
    """Abstract base class for trading strategies."""
    @abstractmethod
    def create_signal_generators(self, df: pd.DataFrame, **params) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def create_entry_exit_conditions(self, df: pd.DataFrame, signals: Dict[str, np.ndarray], upper_band: np.ndarray, middle_band: np.ndarray) -> Tuple[pd.Series, pd.Series]:
        pass

class ATDMFStrategy(TradingStrategy):
    """ATDMF trading strategy implementation."""
    def create_signal_generators(self, df: pd.DataFrame, **params) -> Dict[str, np.ndarray]:
        """Create signal generators."""
        close_prices = df['Close'].values
        use_numba = params.get('use_numba', True)
        
        if use_numba:
            close_2d = vbt.to_2d_array(close_prices)
            try:
                upper_band, middle_band, lower_band = vbt.indicators.nb.bbands_nb(
                    close_2d, 
                    window=params['timeperiod'], 
                    wtype=params['matype'], 
                    alpha=params['StDev'],
                    minp=None,
                    adjust=False,
                    ddof=0
                )
                if len(close_prices.shape) == 1:
                    upper_band = upper_band[:, 0]
                    middle_band = middle_band[:, 0]
                    lower_band = lower_band[:, 0]
            except Exception as e:
                logger.warning(f"VectorBT bbands_nb failed: {e}, falling back to TA-Lib")
                upper_band, middle_band, lower_band = talib.BBANDS(
                    close_prices, timeperiod=params['timeperiod'], 
                    nbdevup=params['StDev'], nbdevdn=params['StDev'], 
                    matype=params['matype']
                )
            
            signals = {
                'upper_band': upper_band,
                'middle_band': middle_band,
                'lower_band': lower_band,
                'ecart_bollinger_signal': ecart_bollinger_borne_signal_nb(
                    close_prices, upper_band, lower_band, 
                    params['timeperiod'], 100, params['coef_mediane'], params['Nb_bars_above']
                ),
                'bollinger_horizontal_signal': bollinger_horizontal_signal_nb(
                    upper_band, lower_band, middle_band, params['coeff_medianeBBW']
                ),
                'bbw_lowest_signal': cross_bbw_low_signal_nb(
                    upper_band, lower_band, middle_band, params['fenetre_lowest'], params['seuil_lowest']
                ),
                'sma_exit_signal': calculate_exit_sma_nb(close_prices, params['user_exit_sma_length'])
            }
        else:
            upper_band, middle_band, lower_band = talib.BBANDS(
                close_prices, timeperiod=params['timeperiod'], 
                nbdevup=params['StDev'], nbdevdn=params['StDev'], 
                matype=params['matype']
            )
            ecart = upper_band - lower_band
            ecart_borne1 = ecart / talib.SMA(close_prices, timeperiod=params['timeperiod'])
            mediane = pd.Series(ecart_borne1).rolling(100).median() / 1
            signals = {
                'upper_band': upper_band,
                'middle_band': middle_band,
                'lower_band': lower_band,
                'ecart_bollinger_signal': (pd.Series(ecart_borne1).shift(6) < mediane).values,
                'bollinger_horizontal_signal': np.where(
                    (pd.Series((upper_band - lower_band) / middle_band).rolling(window=5).mean() < 
                     pd.Series((upper_band - lower_band) / middle_band).rolling(window=200).median() / params['coeff_medianeBBW']) |
                    (pd.Series((upper_band - lower_band) / middle_band) < 
                     pd.Series((upper_band - lower_band) / middle_band).rolling(window=200).median() / params['coeff_medianeBBW']), 
                    True, False
                ),
                'bbw_lowest_signal': (upper_band - lower_band) < (talib.MIN(upper_band - lower_band, timeperiod=params['fenetre_lowest']) * params['seuil_lowest']),
                'sma_exit_signal': (pd.Series(close_prices).shift(1) > pd.Series(talib.SMA(close_prices, params['user_exit_sma_length'])).shift(1)) & 
                                   (pd.Series(close_prices) < pd.Series(talib.SMA(close_prices, params['user_exit_sma_length'])))
            }
        return signals

    def create_entry_exit_conditions(self, df: pd.DataFrame, signals: Dict[str, np.ndarray], upper_band: np.ndarray, middle_band: np.ndarray) -> Tuple[pd.Series, pd.Series]:
        """Create entry and exit conditions."""
        cond_df = pd.DataFrame({
            'bbw_lowest_signal': signals['bbw_lowest_signal'],
            'ecart_bollinger_signal': signals['ecart_bollinger_signal'],
            'bollinger_horizontal_signal': signals['bollinger_horizontal_signal'],
            'sma_exit_signal': signals['sma_exit_signal'],
            'close': df['Close'].values,
            'high': df['High'].values,
            'upper_band': upper_band,
            'middle_band': middle_band
        }, index=df.index)
        
        entry_condition = (
            (cond_df['bbw_lowest_signal']) &
            (cond_df['close'] > cond_df['upper_band'])
        )
        
        exit_condition = (cond_df['sma_exit_signal'])
        return entry_condition, exit_condition

def run_backtest(df: pd.DataFrame, params: Dict[str, Any], timeframe: str = DEFAULT_TIMEFRAME, return_portfolio: bool = True) -> Any:
    """Run a backtest."""
    try:
        strategy = ATDMFStrategy()
        signals = strategy.create_signal_generators(df, **params)
        entry_condition, exit_condition = strategy.create_entry_exit_conditions(df, signals, signals['upper_band'], signals['middle_band'])
        
        portfolio = vbt.Portfolio.from_signals(
            close=df['Close'],
            entries=entry_condition,
            exits=exit_condition,
            max_size=10000,
            init_cash=10000,
            fees=0.0,
            freq=timeframe
        )
        
        if return_portfolio:
            return portfolio
        else:
            metric1_name = params.get('metric1_name', 'sharpe_ratio')
            metric1 = getattr(portfolio, metric1_name.replace('_', ''), None)
            if metric1_name == 'max_drawdown':
                metric1 = portfolio.max_drawdown * 100 * -1
            elif metric1_name == 'total_return':
                metric1 = portfolio.total_return * 100
            elif metric1_name == 'avg_pl_per_trade' and len(portfolio.trades) > 0:
                metric1 = (portfolio.total_return * 100) / len(portfolio.trades)
            else:
                metric1 = metric1 or 0
            return metric1
    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        raise

# ======================================================================
# WALK-FORWARD OPTIMIZATION FRAMEWORK
# ======================================================================

class WFOOptimizer:
    """Walk-Forward Optimization optimizer."""
    def __init__(self, settings: WFOSettings):
        self.settings = settings
    
    def optimize_parameters(self, in_sample_df: pd.DataFrame, param_grid: Dict[str, List], metrics_info: Dict[str, Any], timeframe: str) -> pd.DataFrame:
        """Optimize parameters."""
        param_dicts = [dict(zip(param_grid.keys(), values)) | metrics_info for values in product(*param_grid.values())]
        
        @vbt.parameterized(execute_kwargs=dict(show_progress=True, engine=self.settings.parallel_backend, chunk_len=self.settings.chunk_size))
        def run_parameterized_backtest(df: pd.DataFrame, param_dict: Dict[str, Any], timeframe: str) -> float:
            return run_backtest(df, param_dict, timeframe, return_portfolio=False)
        
        results = [param_dict | {'combined_score': run_parameterized_backtest(in_sample_df, param_dict, timeframe)} for param_dict in tqdm(param_dicts, desc="Optimizing parameters")]
        return pd.DataFrame(results).sort_values('combined_score', ascending=False)
    
    def walk_forward_optimization(self, df: pd.DataFrame, param_grid: Optional[Dict[str, List]] = None, metrics_info: Optional[Dict[str, Any]] = None, timeframe: str = DEFAULT_TIMEFRAME) -> Dict[str, Any]:
        """Perform WFO."""
        start_time = time.time()
        window_times = []
        optimization_times = []
        
        if param_grid is None:
            param_grid = {'timeperiod': [10, 15, 20, 25, 30], 'StDev': [0.5, 1.0, 1.5, 2.0, 2.5]}
        if metrics_info is None:
            metrics_info = {'metric1_name': self.settings.optimization_metric, 'metric2_name': self.settings.secondary_metric, 'weight_metric1': self.settings.metric_weights[0], 'weight_metric2': self.settings.metric_weights[1]}
        
        total_rows = len(df)
        window_size = total_rows // self.settings.n_windows
        wfo_results = {'window_results': [], 'out_of_sample_performance': [], 'best_params': [], 'settings': vars(self.settings)}
        
        logger.info(f"Starting WFO with {self.settings.n_windows} windows")
        
        for i in range(self.settings.n_windows):
            window_start_time = time.time()
            start_idx = i * window_size
            end_idx = start_idx + window_size if i < self.settings.n_windows - 1 else total_rows
            
            window_df = df.iloc[start_idx:end_idx].copy()
            in_sample_start_idx = 0 if self.settings.anchored else start_idx
            in_sample_end_idx = start_idx + int(window_size * self.settings.train_size)
            
            in_sample_df = df.iloc[in_sample_start_idx:in_sample_end_idx].copy() if self.settings.anchored else window_df.iloc[:int(window_size * self.settings.train_size)].copy()
            out_sample_df = window_df.iloc[int(window_size * self.settings.train_size):].copy()
            
            optimization_start = time.time()
            optimization_results = self.optimize_parameters(in_sample_df, param_grid, metrics_info, timeframe)
            optimization_time = time.time() - optimization_start
            optimization_times.append(optimization_time)
            
            best_params = optimization_results.iloc[0].drop(['combined_score', metrics_info['metric1_name'], metrics_info['metric2_name']], errors='ignore').to_dict()
            
            if len(out_sample_df) > 0:
                out_sample_portfolio = run_backtest(out_sample_df, best_params, timeframe)
                out_sample_metrics = {
                    'window': i + 1,
                    'return': out_sample_portfolio.total_return * 100,
                    'sharpe': out_sample_portfolio.sharpe_ratio,
                    'max_drawdown': out_sample_portfolio.max_drawdown * 100,
                    'win_rate': out_sample_portfolio.trades.win_rate,
                    'calmar_ratio': out_sample_portfolio.calmar_ratio if out_sample_portfolio.max_drawdown > 0 else np.nan,
                    'sortino_ratio': out_sample_portfolio.sortino_ratio,
                    'n_trades': len(out_sample_portfolio.trades)
                }
                wfo_results['out_of_sample_performance'].append(out_sample_metrics)
            
            wfo_results['window_results'].append({'window_info': {'window': i + 1, 'start_date': window_df.index[0], 'end_date': window_df.index[-1]}, 'optimization_results': optimization_results.head(5).to_dict('records'), 'best_params': best_params})
            wfo_results['best_params'].append(best_params)
            window_times.append(time.time() - window_start_time)
        
        total_time = time.time() - start_time
        wfo_results['timing'] = {'total_time': total_time, 'window_times': window_times, 'optimization_times': optimization_times, 'avg_window_time': np.mean(window_times), 'avg_optimization_time': np.mean(optimization_times), 'backend': self.settings.parallel_backend, 'use_numba': self.settings.use_numba, 'param_combinations': np.prod([len(values) for values in param_grid.values()])}
        
        logger.info(f"WFO completed in {timedelta(seconds=int(total_time))}")
        return wfo_results

# ======================================================================
# VISUALIZATION
# ======================================================================

def visualize_wfo_results(wfo_results: Dict[str, Any], df: pd.DataFrame) -> None:
    """Visualize WFO results."""
    if not wfo_results['out_of_sample_performance']:
        logger.warning("No out-of-sample results to visualize")
        return
    
    sns.set(style="whitegrid")
    oos_df = pd.DataFrame(wfo_results['out_of_sample_performance'])
    params_df = pd.DataFrame(wfo_results['best_params'])
    
    # Price chart with windows
    plt.figure(figsize=(14, 10))
    price_series = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
    plt.plot(df.index, price_series, label='Price', color='#1f77b4')
    
    for window_idx, window_result in enumerate(wfo_results['window_results']):
        window_info = window_result['window_info']
        plt.axvline(x=pd.to_datetime(window_info['start_date']), color='black', linestyle='--', alpha=0.5)
        if window_info.get('in_sample_start') and window_info.get('in_sample_end'):
            plt.axvspan(pd.to_datetime(window_info['in_sample_start']), pd.to_datetime(window_info['in_sample_end']), alpha=0.15, color='green', label='In-Sample' if window_idx == 0 else "")
        if window_info.get('out_sample_start') and window_info.get('out_sample_end'):
            plt.axvspan(pd.to_datetime(window_info['out_sample_start']), pd.to_datetime(window_info['out_sample_end']), alpha=0.15, color='red', label='Out-of-Sample' if window_idx == 0 else "")
    
    plt.title('Price Chart with WFO Windows', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.figtext(0.5, 0.01, "Chart shows price with in-sample (green) and out-of-sample (red) periods.", ha="center", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()
    
    # In-sample vs out-of-sample comparison
    is_metrics = [{'window': i + 1, 'performance': window_result['optimization_results'][0]['combined_score'], 'type': 'In-Sample'} for i, window_result in enumerate(wfo_results['window_results']) if window_result['optimization_results']]
    oos_metrics = [{'window': metric['window'], 'performance': metric['return'] if 'return' in metric else metric['sharpe'], 'type': 'Out-of-Sample'} for metric in oos_df.to_dict('records')]
    comparison_df = pd.DataFrame(is_metrics + oos_metrics)
    
    if not comparison_df.empty:
        plt.figure(figsize=(14, 10))
        sns.barplot(x='window', y='performance', hue='type', data=comparison_df, palette={'In-Sample': 'skyblue', 'Out-of-Sample': 'salmon'})
        plt.title('In-Sample vs Out-of-Sample Performance', fontsize=16)
        plt.xlabel('Window', fontsize=14)
        plt.ylabel('Performance', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # Parameter stability
    numeric_params = params_df.select_dtypes(include=['number']).columns
    if len(numeric_params) > 0:
        plt.figure(figsize=(14, 10))
        norm_params_df = pd.DataFrame()
        for param in numeric_params:
            if params_df[param].nunique() <= 1:
                continue
            min_val = params_df[param].min()
            max_val = params_df[param].max()
            if max_val > min_val:
                norm_params_df[param] = (params_df[param] - min_val) / (max_val - min_val)
            else:
                norm_params_df[param] = params_df[param] / params_df[param]
        
        x = list(range(1, len(params_df) + 1))
        markers = ['o', 's', 'd', '^', 'v']
        for i, param in enumerate(norm_params_df.columns):
            marker = markers[i % len(markers)]
            plt.plot(x, norm_params_df[param], marker=marker, label=param, linewidth=2, markersize=8)
            stability = 1.0 - (np.std(params_df[param]) / np.mean(params_df[param])) if np.mean(params_df[param]) > 0 else 0.0
            plt.annotate(f"Stability: {stability:.2f}", xy=(x[-1], norm_params_df[param].iloc[-1]), xytext=(x[-1] + 0.1, norm_params_df[param].iloc[-1]), fontsize=9)
        
        plt.title('Parameter Stability Across Windows', fontsize=16)
        plt.xlabel('Window', fontsize=14)
        plt.ylabel('Normalized Parameter Value', fontsize=14)
        plt.xticks(x)
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, 1.05)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

# ======================================================================
# UTILITY FUNCTIONS FOR MAIN
# ======================================================================

def display_default_parameters() -> None:
    """Display default parameters for strategy and WFO settings."""
    strategy_config = StrategyConfig()
    wfo_settings = WFOSettings()
    print("Default Strategy Parameters:")
    for attr, value in vars(strategy_config).items():
        print(f"  {attr}: {value}")
    print("\nDefault WFO Settings:")
    for attr, value in vars(wfo_settings).items():
        print(f"  {attr}: {value}")

def get_param_grid() -> Dict[str, List]:
    """Get parameter grid for optimization."""
    # Default parameter grid; can be modified or prompted
    return {
        'timeperiod': [10, 15, 20, 25, 30],
        'StDev': [0.5, 1.0, 1.5, 2.0, 2.5],
        'coeff_medianeBBW': [1.0, 1.1, 1.2],
        'coef_mediane': [0.8, 1.0, 1.2],
        'Nb_bars_above': [3, 5, 7],
        'fenetre_lowest': [20, 30, 40],
        'seuil_lowest': [3.0, 3.5, 4.0],
        'user_exit_sma_length': [15, 20, 25]
    }

def get_metrics_info() -> Dict[str, Any]:
    """Get metrics information for optimization."""
    # Default metrics; can be modified
    return {
        'metric1_name': 'sharpe_ratio',
        'metric2_name': 'total_return',
        'weight_metric1': 1.0,
        'weight_metric2': 0.0
    }

def get_wfo_settings() -> WFOSettings:
    """Get WFO settings."""
    # Return default settings; can be modified or prompted
    return WFOSettings()

# ======================================================================
# MAIN PROGRAM
# ======================================================================

def main() -> None:
    """Main function."""
    logger.info("Starting ATDMF WFO")
    
    start_date, end_date = get_dates()
    df = load_data(start_date, end_date, from_file=True, file_path=DEFAULT_FILE_PATH)
    
    display_default_parameters()
    param_grid = get_param_grid()
    metrics_info = get_metrics_info()
    settings = get_wfo_settings()
    
    optimizer = WFOOptimizer(settings)
    wfo_results = optimizer.walk_forward_optimization(df, param_grid, metrics_info)
    
    visualize_wfo_results(wfo_results, df)
    
    # Save results
    results_dir = "WFO_Results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if wfo_results['out_of_sample_performance']:
        pd.DataFrame(wfo_results['out_of_sample_performance']).to_csv(f"{results_dir}/oos_performance_{timestamp}.csv", index=False)
    pd.DataFrame(wfo_results['best_params']).to_csv(f"{results_dir}/best_parameters_{timestamp}.csv", index=False)
    
    logger.info("WFO completed")

if __name__ == "__main__":
    main()
