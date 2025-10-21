# Import necessary libraries
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

DEFAULT_START_DATE = "2025-01-19"
DEFAULT_END_DATE = "2025-01-31"
DEFAULT_TIMEFRAME = '5S'
DEFAULT_DATA_FILE = (
    "/home/olivier/Downloads/ATDMF_strategy_V5_long/ATDMF_strategy_long_"
    "BTCFDUSD05S/Data/Binance_BTCUSDT_OHLCV_B_2025-01-19_2025-01-31_1s/"
    "Binance_BTCUSDT_OHLCV_B_2025-01-19_2025-01-31_5S.csv"
)
# Default parameter grid for optimization
# DEFAULT_PARAM_GRID = {
#     'timeperiod': [10, 15, 20, 25, 30],
#     'StDev': [0.5, 1.0, 1.5, 2.0, 2.5],
#     'coeff_medianeBBW': [1.0, 1.1, 1.2],
#     'coef_mediane': [0.8, 1.0, 1.2],
#     'Nb_bars_above': [3, 5, 7],
#     'fenetre_lowest': [20, 30, 40],
#     'seuil_lowest': [3.0, 3.5, 4.0],
#     'user_exit_sma_length': [15, 20, 25]
# }

# Expanded parameter grid for more extensive search
DEFAULT_PARAM_GRID = {
    'timeperiod': [10, 15, 20, 25, 30],
    'StDev': [0.5, 1.0, 1.5, 2.0, 2.5],
    'coeff_medianeBBW': [0.8, 1.0, 1.2, 1.4, 1.6],
    'coef_mediane': [0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
    'fenetre_lowest': [30, 40, 50, 60, 70, 80],
    'seuil_lowest': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
    'longueur_mediane': [50.0, 75.0, 100.0, 125.0, 150.0],
    'Nb_bars_above': [2.0, 4.0, 6.0, 8.0, 10.0],
    'user_exit_sma_length': [15, 20, 25]
}

# ======================================================================
# CONFIGURATION SETTINGS
# ======================================================================

# Walk-Forward Optimization settings
class WFOSettings:
    def __init__(self):
        self.n_windows = 1               # Number of windows to divide data into
        self.train_size = 0.5            # Proportion of window for training
        self.anchored = False            # Whether to use anchored (fixed start date) WFO
        self.optimization_metric = "sharpe_ratio"  # Main metric to optimize
        self.secondary_metric = "total_return"     # Secondary metric to optimize
        self.metric_weights = (1.0, 0.0) # Weights for primary and secondary metrics
        self.parallel_backend = "dask"   # Parallelization backend ('dask', 'ray', 'pathos', 'threadpool')
        self.use_numba = True            # Whether to use Numba for accelerated computations
        self.chunk_size = "auto"         # Chunk size for parallelization

# ======================================================================
# NUMBA-OPTIMIZED INDICATOR FUNCTIONS
# ======================================================================

@njit(cache=True)
def bbands_1d_nb(close, window=20, alpha=2.0, ddof=0):
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
def rolling_mean(arr, window):
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
def rolling_median(arr, window):
    result = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        result[i] = np.median(arr[i - window + 1:i + 1])
    return result

@njit
def rolling_min(arr, window):
    result = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        result[i] = np.min(arr[i - window + 1:i + 1])
    return result

@njit
def compute_bars_since_below(ecart_borne1, mediane, coef, Nb_bars_above):
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
def ecart_bollinger_borne_signal_nb(prix, upper_band, lower_band, timeperiod, longueur_mediane, coef_mediane, Nb_bars_above):
    ecart = upper_band - lower_band
    sma = vbt.indicators.nb.ma_1d_nb(prix, timeperiod)
    ecart_borne1 = ecart / sma
    mediane = rolling_median(ecart_borne1, longueur_mediane)
    return compute_bars_since_below(ecart_borne1, mediane, coef_mediane, Nb_bars_above)

@njit
def bollinger_horizontal_signal_nb(upper_band, lower_band, middle_band, coeff_medianeBBW):
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
def cross_bbw_low_signal_nb(upper_band, lower_band, middle_band, fenetre_lowest, seuil_lowest):
    largeur_bb = (upper_band - lower_band) / middle_band
    bbw_lowest = rolling_min(largeur_bb, fenetre_lowest)
    signal = np.full(len(largeur_bb), False)
    for i in range(len(largeur_bb)):
        if np.isnan(bbw_lowest[i]) or np.isnan(largeur_bb[i]):
            continue
        signal[i] = largeur_bb[i] <= (bbw_lowest[i] * seuil_lowest)
    return signal


@njit(cache=True)
def calculate_exit_sma_nb(close, user_exit_sma_length):
    """
    Numba-optimized implementation of the SMA exit signal.
    
    Returns a boolean array where True indicates an exit signal.
    """
    # Ensure input is 1D
    close_1d = close if len(close.shape) == 1 else close.flatten()
    
    n = len(close_1d)
    
    # Use VectorBT's built-in moving average function or calculate manually
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

def get_dates():
    """Prompt for date range input with defaults."""
    default_start = "2025-01-19"
    default_end = "2025-01-31"
    
    start_date = input(f"Enter start date (YYYY-MM-DD) [default: {default_start}]: ") or default_start
    end_date = input(f"Enter end date (YYYY-MM-DD) [default: {default_end}]: ") or default_end
    
    return start_date, end_date

def load_data(start_date, end_date, timeframe='5S', from_file=True, file_path=None):
    """
    Load OHLCV data for the specified period, either from Binance API or from a file.
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    timeframe : str, optional
        Timeframe to use
    from_file : bool, optional
        Whether to load from file or fetch from Binance
    file_path : str, optional
        Path to the data file
        
    Returns:
    --------
    pandas.DataFrame
        OHLCV data
    """
    file_path = '/home/olivier/Downloads/ATDMF_strategy_V5_long/ATDMF_strategy_long_BTCFDUSD05S/Data/Binance_BTCUSDT_OHLCV_B_2025-01-19_2025-01-31_1s/Binance_BTCUSDT_OHLCV_B_2025-01-19_2025-01-31_5S.csv'
    if from_file and file_path:
        # Load from file
        df = pd.read_csv(file_path)
        df['Open time'] = pd.to_datetime(df['Open time'])
        df.set_index('Open time', inplace=True)
        df = df.resample(timeframe).agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last'
        }).dropna()
        print(df)

    else:
        # Fetch from Binance
        base_timeframe = '1s'  # Fetch at 1s resolution
        df_1s = vbt.BinanceData.fetch(
            ["BTCUSDT"], 
            start=start_date, 
            end=end_date,
            timeframe=base_timeframe
        )
        
        # Resample to desired timeframe
        df = df_1s.resample(timeframe).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna()
        # Save data for later use
        folder_name = f"Binance_BTCUSDT_OHLCV_B_{start_date}_{end_date}_{timeframe}"
        folder_path = f"./Data/{folder_name}"
        os.makedirs(folder_path, exist_ok=True)
        file_path = f"{folder_path}/Binance_BTCUSDT_OHLCV_B_{start_date}_{end_date}_{timeframe}.csv"
        df.to_csv(file_path)
        
    return df

# ======================================================================
# STRATEGY IMPLEMENTATION
# ======================================================================

def create_signal_generators(df, **params):
    """
    Create signal generators for the ATDMF strategy using VectorBT's built-in Numba-optimized
    indicator functions or TA-Lib implementations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        OHLCV DataFrame with 'Open', 'High', 'Low', 'Close'
    **params : dict
        Strategy parameters
        
    Returns:
    --------
    dict
        Dictionary of signal generators
    """
    # Extract parameters with defaults
    timeperiod = params.get('timeperiod', 20)
    StDev = params.get('StDev', 1.3)
    matype = params.get('matype', 0)
    coeff_medianeBBW = params.get('coeff_medianeBBW', 1.1)
    coef_mediane = params.get('coef_mediane', 1.0)
    Nb_bars_above = params.get('Nb_bars_above', 5)
    fenetre_lowest = params.get('fenetre_lowest', 30)
    seuil_lowest = params.get('seuil_lowest', 3.5)
    user_exit_sma_length = params.get('user_exit_sma_length', 20)
    use_numba = params.get('use_numba', True)
    
    Prix = df['Close'].values
    Open = df['Open'].values
    High = df['High'].values
    Low = df['Low'].values
    
    # Generate signals
    if use_numba:
        # Use VectorBT's built-in Numba-accelerated implementation
        # Convert to 2D array for VectorBT functions if needed
        Prix_2d = vbt.to_2d_array(Prix)
        
        try:
            # Try using VectorBT's built-in bbands_nb function
            upper_band, middle_band, lower_band = vbt.indicators.nb.bbands_nb(
                Prix_2d, 
                window=timeperiod, 
                wtype=matype, 
                alpha=StDev,
                minp=None,
                adjust=False,
                ddof=0
            )
            
            # Convert back to 1D if input was 1D
            if len(Prix.shape) == 1:
                upper_band = upper_band[:, 0]
                middle_band = middle_band[:, 0]
                lower_band = lower_band[:, 0]
                
        except Exception as e:
            print(f"Error using VectorBT's bbands_nb function: {e}")
            print("Falling back to TA-Lib implementation")
            # Fallback to TA-Lib
            upper_band, middle_band, lower_band = talib.BBANDS(
                Prix, timeperiod=timeperiod, 
                nbdevup=StDev, nbdevdn=StDev, 
                matype=matype
            )
        
        # Calculate signals using Numba functions
        nb_bars_above_signal = ecart_bollinger_borne_signal_nb(
            Prix, upper_band, lower_band, 
            timeperiod=timeperiod, 
            longueur_mediane=100, 
            coef_mediane=coef_mediane,
            Nb_bars_above=Nb_bars_above
        )
        
        bollinger_horizontal_signal = bollinger_horizontal_signal_nb(
            upper_band, lower_band, middle_band,
            coeff_medianeBBW=coeff_medianeBBW
        )
        
        cross_bbw_low_signal = cross_bbw_low_signal_nb(
            upper_band, lower_band, middle_band,
            fenetre_lowest=fenetre_lowest,
            seuil_lowest=seuil_lowest
        )
        
        sma_exit_signal = calculate_exit_sma_nb(Prix, user_exit_sma_length)
        
    else:
        # Standard TA-Lib implementation
        upper_band, middle_band, lower_band = talib.BBANDS(
            Prix, timeperiod=timeperiod, 
            nbdevup=StDev, nbdevdn=StDev, 
            matype=matype
        )
        
        # Ecart_Bollinger_borne signal
        ecart = upper_band - lower_band
        ecart_borne1 = ecart / talib.SMA(Prix, timeperiod=timeperiod)
        mediane = pd.Series(ecart_borne1).rolling(100).median() / coef_mediane
        nb_bars_above_signal = (pd.Series(ecart_borne1).shift(6) < mediane).values
        
        # Bollinger Horizontal signal
        BBW = (upper_band - lower_band) / middle_band
        MMBBW = pd.Series(BBW).rolling(window=5).mean()
        medianeBBW = pd.Series(BBW).rolling(window=200).median()
        seuilBBW = BBW < (medianeBBW / coeff_medianeBBW)
        seuilMMBBW = MMBBW < (medianeBBW / coeff_medianeBBW)
        bollinger_horizontal_signal = np.where(seuilBBW | seuilMMBBW, True, False)
        
        # BBW Lowest signal
        largeur_bb = (upper_band - lower_band) / middle_band
        bbw_lowest = pd.Series(largeur_bb).rolling(window=fenetre_lowest, min_periods=fenetre_lowest).min()
        cross_bbw_low_signal = largeur_bb < (bbw_lowest * seuil_lowest)
        
        # SMA Exit signal
        sma = talib.SMA(Prix, user_exit_sma_length)
        sma_exit_signal = (pd.Series(Prix).shift(1) > pd.Series(sma).shift(1)) & (pd.Series(Prix) < pd.Series(sma))
        sma_exit_signal = sma_exit_signal.values
    
    return {
        'upper_band': upper_band,
        'middle_band': middle_band,
        'lower_band': lower_band,
        'nb_bars_above_signal': nb_bars_above_signal,
        'bollinger_horizontal_signal': bollinger_horizontal_signal,
        'cross_bbw_low_signal': cross_bbw_low_signal,
        'sma_exit_signal': sma_exit_signal
    }

def create_entry_exit_conditions(df, signals, upper_band, middle_band):
    """
    Create entry and exit conditions for the ATDMF strategy.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        OHLCV DataFrame
    signals : dict
        Signal generators from create_signal_generators
    upper_band : array_like
        Upper Bollinger Band values
    middle_band : array_like
        Middle Bollinger Band values
        
    Returns:
    --------
    tuple
        (entry_condition, exit_condition)
    """
    # Extract price data
    Prix = df['Close'].values
    High = df['High'].values
    
    # Create a DataFrame for conditions
    cond_df = pd.DataFrame({
        'cross_bbw_low_signal': signals['cross_bbw_low_signal'],
        'nb_bars_above_signal': signals['nb_bars_above_signal'],
        'bollinger_horizontal_signal': signals['bollinger_horizontal_signal'],
        'sma_exit_signal': signals['sma_exit_signal'],
        'close': Prix,
        'high': High,
        'upper_band': upper_band,
        'middle_band': middle_band
    }, index=df.index)
    
    # Entry condition matches V1B logic
    entry_condition = (
    cond_df['nb_bars_above_signal'].astype(bool) &
    cond_df['cross_bbw_low_signal'].astype(bool) &
    cond_df['bollinger_horizontal_signal'].astype(bool) &
    (cond_df['close'] > cond_df['upper_band']) &
    (cond_df['close'].shift(1) < cond_df['upper_band'].shift(1))
    ).fillna(False)

    # Exit condition
    exit_condition = (
        (cond_df['sma_exit_signal']) # | (cond_df['close'] < cond_df['middle_band'])
    )
    return entry_condition, exit_condition

def run_backtest(df, params, timeframe='5S', return_portfolio=True):
    """
    Run a backtest with the ATDMF strategy using given parameters.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        OHLCV DataFrame
    params : dict
        Strategy parameters
    timeframe : str, optional
        Timeframe of the data
    return_portfolio : bool, optional
        Whether to return the portfolio object or a performance metric
        
    Returns:
    --------
    object
        Portfolio object or performance metric
    """
    # Generate signals
    signals = create_signal_generators(df, **params)
    
    # Create entry and exit conditions
    entry_condition, exit_condition = create_entry_exit_conditions(
        df, signals, signals['upper_band'], signals['middle_band']
    )
    
    # Create portfolio
    portfolio = vbt.Portfolio.from_signals(
        close=df['Close'],
        entries=entry_condition,
        exits=exit_condition,
        max_size = 10000,
        init_cash=10000,
        fees=0.0,
        freq=timeframe
    )
    
    if return_portfolio:
        return portfolio
    else:
        # Calculate performance metrics
        metric1_name = params.get('metric1_name', 'sharpe_ratio')
        metric2_name = params.get('metric2_name', 'total_return')
        weight_metric1 = params.get('weight_metric1', 1.0)
        weight_metric2 = params.get('weight_metric2', 0.0)
        
        # Calculate metrics
        if metric1_name == 'max_drawdown':
            metric1 = portfolio.max_drawdown * 100 * -1  # Invert so higher is better
        elif metric1_name == 'sharpe_ratio':
            metric1 = portfolio.sharpe_ratio
        elif metric1_name == 'total_return':
            metric1 = portfolio.total_return * 100
        elif metric1_name == 'avg_gain_per_trade':
            metric1 = portfolio.trades.avg_winning_trade
        elif metric1_name == 'avg_loss_per_trade':
            metric1 = portfolio.trades.avg_losing_trade * -1  # Invert for optimization
        elif metric1_name == 'win_rate':
            metric1 = portfolio.trades.win_rate
        elif metric1_name == 'avg_pl_per_trade':
            # Average P&L per trade - total return divided by number of trades
            if len(portfolio.trades) > 0:
                metric1 = (portfolio.total_return*100) / len(portfolio.trades)
            else:
                metric1 = 0  # Default value if no trades

        if metric2_name == 'max_drawdown':
            metric2 = portfolio.max_drawdown * 100 * -1
        elif metric2_name == 'sharpe_ratio':
            metric2 = portfolio.sharpe_ratio
        elif metric2_name == 'total_return':
            metric2 = portfolio.total_return * 100
        elif metric2_name == 'avg_gain_per_trade':
            metric2 = portfolio.trades.avg_winning_trade
        elif metric2_name == 'avg_loss_per_trade':
            metric2 = portfolio.trades.avg_losing_trade * -1
        elif metric2_name == 'win_rate':
            metric2 = portfolio.trades.win_rate
        elif metric2_name == 'avg_pl_per_trade':
            if len(portfolio.trades) > 0:
                metric2 = (portfolio.total_return*100) / len(portfolio.trades)
            else:
                metric2 = 0        
        # Combined metric
        combined_metric = (weight_metric1 * metric1 + weight_metric2 * metric2) / (weight_metric1 + weight_metric2)
        return combined_metric

# ======================================================================
# WALK-FORWARD OPTIMIZATION FRAMEWORK
# ======================================================================

def optimize_parameters(in_sample_df, param_grid, metrics_info, timeframe='5S', settings=None):
    """
    Optimize parameters using grid search on in-sample data.
    
    Parameters:
    -----------
    in_sample_df : pandas.DataFrame
        In-sample OHLCV data
    param_grid : dict
        Parameter grid to search
    metrics_info : dict
        Metrics information dictionary
    timeframe : str, optional
        Timeframe of the data
    settings : WFOSettings, optional
        WFO settings
        
    Returns:
    --------
    pandas.DataFrame
        Sorted optimization results
    """
    if settings is None:
        settings = WFOSettings()
    
    # Create parameter combinations with Param for vectorbt's parameterized decorator
    param_dicts = []
    param_keys = list(param_grid.keys())
    
    for values in product(*param_grid.values()):
        param_dict = dict(zip(param_keys, values))
        param_dict.update(metrics_info)
        param_dicts.append(param_dict)
    
    # Use vectorbt's parameterized decorator for optimization
    @vbt.parameterized(
        execute_kwargs=dict(
            show_progress=True,
            engine=settings.parallel_backend,
            chunk_len=settings.chunk_size
        )
    )
    def run_parameterized_backtest(df, param_dict, timeframe='5S'):
        params = param_dict.copy()
        return run_backtest(df, params, timeframe, return_portfolio=False)
    
    # Run parameterized backtest
    results = []
    for param_dict in tqdm(param_dicts, desc="Optimizing parameters"):
        score = run_parameterized_backtest(in_sample_df, param_dict, timeframe)
        result = param_dict.copy()
        result['combined_score'] = score
        results.append(result)
    
    # Convert to DataFrame and sort
    results_df = pd.DataFrame(results)
    sorted_results = results_df.sort_values('combined_score', ascending=False)
    
    return sorted_results


def walk_forward_optimization(df, param_grid=None, metrics_info=None, timeframe='5S', settings=None):
    """
    Performs Walk-Forward Optimization on the given data with timing measurements.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        OHLCV DataFrame
    param_grid : dict, optional
        Parameter grid to search
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
        
    if param_grid is None:
        param_grid = {
            'timeperiod': [10, 15, 20, 25, 30],
            'StDev': [0.5, 1.0, 1.5, 2.0, 2.5],
            'coeff_medianeBBW': [0.8, 1.0, 1.2, 1.4, 1.6],
            'coef_mediane': [0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
            'fenetre_lowest': [30, 40, 50, 60, 70, 80],
            'seuil_lowest': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            'longueur_mediane': [50.0, 75.0, 100.0, 125.0, 150.0],
            'Nb_bars_above': [2.0, 4.0, 6.0, 8.0, 10.0]
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
            'use_numba': settings.use_numba
        }
    }
    
    # Calculate total parameter combinations for reporting
    param_combinations = np.prod([len(values) for values in param_grid.values()])
    
    print(f"Starting Walk-Forward Optimization with {settings.n_windows} windows, {settings.train_size*100}% training size")
    print(f"WFO Type: {'Anchored' if settings.anchored else 'Unanchored'}")
    print(f"Primary Metric: {settings.optimization_metric} (weight: {settings.metric_weights[0]})")
    print(f"Secondary Metric: {settings.secondary_metric} (weight: {settings.metric_weights[1]})")
    print(f"Parallelization Backend: {settings.parallel_backend}")
    print(f"Numba Acceleration: {'Enabled' if settings.use_numba else 'Disabled'}")
    print(f"Parameter Combinations: {param_combinations}")
    
    # Loop through each window
    for i in range(settings.n_windows):
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
        
        print(f"\nWindow {i+1}/{settings.n_windows}: {window_dates['start_date']} to {window_dates['end_date']}")
        print(f"In-Sample: {window_dates['in_sample_start']} to {window_dates['in_sample_end']}")
        if len(out_sample_df) > 0:
            print(f"Out-of-Sample: {window_dates['out_sample_start']} to {window_dates['out_sample_end']}")
        
        # Optimize parameters on in-sample data
        print(f"Optimizing parameters on in-sample data ({len(in_sample_df)} bars)...")
        
        # Time the optimization process
        optimization_start = time.time()
        optimization_results = optimize_parameters(
            in_sample_df, param_grid, metrics_info, timeframe, settings
        )
        optimization_time = time.time() - optimization_start
        optimization_times.append(optimization_time)
        
        # Get best parameters
        best_params = optimization_results.iloc[0].drop(['combined_score', metrics_info['metric1_name'], metrics_info['metric2_name']], errors='ignore').to_dict()
        
        print(f"Best parameters found: {best_params}")
        print(f"Score: {optimization_results.iloc[0]['combined_score']:.4f}")
        print(f"Optimization time: {timedelta(seconds=int(optimization_time))}")
        
        # Test on out-of-sample data if available
        if len(out_sample_df) > 0:
            print(f"Testing best parameters on out-of-sample data ({len(out_sample_df)} bars)...")
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
            
            print(f"Out-of-Sample Performance:")
            print(f"Return: {out_sample_metrics['return']:.2f}%")
            print(f"Sharpe Ratio: {out_sample_metrics['sharpe']:.2f}")
            print(f"Max Drawdown: {out_sample_metrics['max_drawdown']:.2f}%")
            print(f"Win Rate: {out_sample_metrics['win_rate']:.2f}%")
            print(f"Number of Trades: {out_sample_metrics['n_trades']}")
            
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
        print(f"Window processing time: {timedelta(seconds=int(window_time))}")
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Calculate aggregate out-of-sample performance if available
    if wfo_results['out_of_sample_performance']:
        oos_df = pd.DataFrame(wfo_results['out_of_sample_performance'])
        
        print("\n=== Aggregate Out-of-Sample Performance ===")
        print(f"Average Return: {oos_df['return'].mean():.2f}%")
        print(f"Average Sharpe Ratio: {oos_df['sharpe'].mean():.2f}")
        print(f"Average Max Drawdown: {oos_df['max_drawdown'].mean():.2f}%")
        print(f"Average Win Rate: {oos_df['win_rate'].mean():.2f}%")
        print(f"Average Calmar Ratio: {oos_df['calmar_ratio'].mean():.2f}")
        print(f"Average Sortino Ratio: {oos_df['sortino_ratio'].mean():.2f}")
        print(f"Total Trades: {oos_df['n_trades'].sum()}")
        print(f"Cumulative Return: {((1 + oos_df['return']/100).prod() - 1) * 100:.2f}%")

        # Check for consistency in parameter selection
        params_df = pd.DataFrame(wfo_results['best_params'])
        print("\n=== Parameter Consistency Analysis ===")
        for param in param_grid.keys():
            print(f"{param}: {params_df[param].value_counts().to_dict()}")
        
        # Calculate parameter stability
        param_stability = {}
        for param in param_grid.keys():
            param_values = params_df[param].values
            param_stability[param] = 1.0 - (np.std(param_values) / np.mean(param_values)) if np.mean(param_values) > 0 else 0.0
            
        print("\n=== Parameter Stability (higher is better) ===")
        for param, stability in param_stability.items():
            print(f"{param}: {stability:.4f}")
    
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
    print("\n=== Performance Timing Summary ===")
    print(f"Backend: {settings.parallel_backend}")
    print(f"Numba: {'Enabled' if settings.use_numba else 'Disabled'}")
    print(f"Total processing time: {timedelta(seconds=int(total_time))}")
    print(f"Average window time: {timedelta(seconds=int(np.mean(window_times)))}")
    print(f"Average optimization time: {timedelta(seconds=int(np.mean(optimization_times)))}")
    print(f"Parameter combinations per window: {param_combinations}")
    print(f"Processing speed: {param_combinations * settings.n_windows / total_time:.2f} combinations/second")
    
    return wfo_results

# ======================================================================
# VISUALIZATION
# ======================================================================

def visualize_wfo_results(wfo_results, df):
    """
    Enhanced visualization of Walk-Forward Optimization results with explicit legends
    and descriptions for each chart.
    
    Parameters:
    -----------
    wfo_results : dict
        WFO results from walk_forward_optimization
    df : pandas.DataFrame
        Original OHLCV DataFrame
    """
    if not wfo_results['out_of_sample_performance']:
        print("No out-of-sample results to visualize")
        return
    
    # Set Seaborn style
    sns.set(style="whitegrid", font_scale=1.1)
    
    # Convert performance metrics to DataFrame
    oos_df = pd.DataFrame(wfo_results['out_of_sample_performance'])
    params_df = pd.DataFrame(wfo_results['best_params'])
    
    # =========================================================================
    # 1. MAIN OVERVIEW PLOT - PRICE WITH WINDOW BOUNDARIES
    # =========================================================================
    
    # Create a figure showing price with vertical lines indicating window boundaries
    plt.figure(figsize=(14, 10))
    
    # Plot the close price
    if 'Close' in df.columns:
        price_series = df['Close']
    elif 'close' in df.columns:
        price_series = df['close']
    else:
        price_series = df.iloc[:, 0]  # As a last resort
    
    # Plot price series
    plt.plot(df.index, price_series, label='Price', color='#1f77b4')
    
    # Add vertical lines for window boundaries
    window_colors = {
        'train': 'green',
        'test': 'red'
    }
    
    # Add vertical lines for window boundaries
    for window_idx, window_result in enumerate(wfo_results['window_results']):
        window_info = window_result['window_info']
        
        # Add vertical line at window start
        plt.axvline(x=pd.to_datetime(window_info['start_date']), 
                   color='black', linestyle='--', alpha=0.5)
        
        # Mark in-sample and out-of-sample regions
        if window_info['in_sample_start'] and window_info['in_sample_end']:
            plt.axvspan(
                pd.to_datetime(window_info['in_sample_start']),
                pd.to_datetime(window_info['in_sample_end']),
                alpha=0.15, color=window_colors['train'], 
                label=f'In-Sample {window_idx+1}' if window_idx == 0 else ""
            )
        
        if window_info['out_sample_start'] and window_info['out_sample_end']:
            plt.axvspan(
                pd.to_datetime(window_info['out_sample_start']),
                pd.to_datetime(window_info['out_sample_end']),
                alpha=0.15, color=window_colors['test'], 
                label=f'Out-of-Sample {window_idx+1}' if window_idx == 0 else ""
            )
    
    # Customize the plot
    plt.title('Price Chart with Walk-Forward Optimization Windows', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Add description as text in the plot
    plt.figtext(0.5, 0.01, 
               "This chart shows the price series with in-sample (green) and out-of-sample (red) periods highlighted.\n"
               "Each vertical line represents the boundary between consecutive WFO windows.",
               ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()
    
    # =========================================================================
    # 2. IN-SAMPLE VS OUT-OF-SAMPLE PERFORMANCE COMPARISON
    # =========================================================================
    
    # Create a comparison of key metrics between in-sample and out-of-sample periods
    # This helps assess overfitting
    
    # Extract performance metrics for both in-sample and out-of-sample
    is_metrics = []
    for window_idx, window_result in enumerate(wfo_results['window_results']):
        # Get best in-sample result from optimization results
        if window_result['optimization_results'] and len(window_result['optimization_results']) > 0:
            best_result = window_result['optimization_results'][0]
            if 'combined_score' in best_result:
                is_metrics.append({
                    'window': window_idx + 1,
                    'performance': best_result['combined_score'],
                    'type': 'In-Sample'
                })
    
    # Get out-of-sample metrics
    oos_metrics = []
    for metric in oos_df.to_dict('records'):
        oos_metrics.append({
            'window': metric['window'],
            'performance': metric['return'] if 'return' in metric else (
                          metric['sharpe'] if 'sharpe' in metric else 0),
            'type': 'Out-of-Sample'
        })
    
    # Combine metrics
    comparison_df = pd.DataFrame(is_metrics + oos_metrics)
    
    if not comparison_df.empty:
        plt.figure(figsize=(14, 10))
        
        # Create grouped bar chart
        sns.barplot(
            x='window', 
            y='performance', 
            hue='type',
            data=comparison_df,
            palette={'In-Sample': 'skyblue', 'Out-of-Sample': 'salmon'}
        )
        
        plt.title('In-Sample vs Out-of-Sample Performance by Window', fontsize=16)
        plt.xlabel('Window', fontsize=14)
        plt.ylabel('Performance', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add description
        plt.figtext(0.5, 0.01, 
                   "This chart compares in-sample (optimization) performance with out-of-sample (validation) results.\n"
                   "Large differences suggest potential overfitting. Similar performance indicates strategy robustness.",
                   ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()
    
    # =========================================================================
    # 3. PARAMETER STABILITY ANALYSIS
    # =========================================================================
    
    # This visualization shows how parameters change across windows
    # Stable parameters indicate robust strategies
    
    # Filter only numeric parameters
    numeric_params = params_df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_params) > 0:
        # Create a parameter stability plot
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Get normalized parameter values for comparison across different scales
        norm_params_df = pd.DataFrame()
        for param in numeric_params:
            # Skip if all values are identical (stability = 1.0)
            if params_df[param].nunique() <= 1:
                continue
                
            # Normalize to 0-1 range for plotting on same scale
            min_val = params_df[param].min()
            max_val = params_df[param].max()
            
            if max_val > min_val:  # Avoid division by zero
                norm_params_df[param] = (params_df[param] - min_val) / (max_val - min_val)
            else:
                norm_params_df[param] = params_df[param] / params_df[param]
        
        # Create x-axis (window numbers)
        x = list(range(1, len(params_df) + 1))
        
        # Plot each parameter
        markers = ['o', 's', 'd', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'D', '|', '_']
        for i, param in enumerate(norm_params_df.columns):
            marker = markers[i % len(markers)]
            plt.plot(x, norm_params_df[param], marker=marker, label=param, linewidth=2, markersize=8)
            
            # Calculate stability metric (1 - coefficient of variation)
            stability = 1.0 - (np.std(params_df[param]) / np.mean(params_df[param])) if np.mean(params_df[param]) > 0 else 0.0
            
            # Annotate with stability value
            plt.annotate(f"Stability: {stability:.2f}", 
                        xy=(x[-1], norm_params_df[param].iloc[-1]),
                        xytext=(x[-1] + 0.1, norm_params_df[param].iloc[-1]),
                        fontsize=9)
        
        # Add parameter absolute value table
        param_table = ''
        for i, window in enumerate(x):
            param_table += f'Window {window}: '
            param_values = []
            for param in numeric_params:
                if param in norm_params_df.columns:
                    param_values.append(f"{param}={params_df[param].iloc[i]:.2f}")
            param_table += ', '.join(param_values) + '\n'
        
        plt.figtext(0.5, 0.01, param_table, ha="center", fontsize=9, 
                   bbox={"facecolor":"white", "alpha":0.8, "pad":5})
        
        # Customize the plot
        plt.title('Parameter Stability Across Windows (Normalized Values)', fontsize=16)
        plt.xlabel('Window', fontsize=14)
        plt.ylabel('Normalized Parameter Value', fontsize=14)
        plt.xticks(x)
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, 1.05)  # Give some margin above and below
        plt.legend(loc='best')
        
        # Add description
        plt.figtext(0.5, 0.15, 
                   "This chart shows how optimized parameters change across windows (normalized to 0-1 scale).\n"
                   "Stable parameters (less variation) indicate more robust strategies.\n"
                   "Stability score ranges from 0 (unstable) to 1 (perfectly stable).",
                   ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        plt.tight_layout(rect=[0, 0.25, 1, 0.97])
        plt.show()
    
    # =========================================================================
    # 4. OUT-OF-SAMPLE PERFORMANCE METRICS DASHBOARD
    # =========================================================================
    
    # Create a comprehensive dashboard of OOS performance metrics
    if not oos_df.empty:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Returns by Window (%)', 'Sharpe Ratio by Window', 
                          'Maximum Drawdown by Window (%)', 'Win Rate by Window (%)'),
            shared_xaxes=True,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. Returns plot
        if 'return' in oos_df.columns:
            window_nums = list(range(1, len(oos_df) + 1))
            
            # Bar chart with returns
            fig.add_trace(
                go.Bar(
                    x=window_nums,
                    y=oos_df['return'],
                    name='Return (%)',
                    marker_color='rgb(55, 83, 109)',
                    text=oos_df['return'].round(2).astype(str) + '%',
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            # Add benchmark line (average return)
            fig.add_trace(
                go.Scatter(
                    x=window_nums,
                    y=[oos_df['return'].mean()] * len(window_nums),
                    mode='lines',
                    name=f'Avg Return: {oos_df["return"].mean():.2f}%',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=1
            )
        
        # 2. Sharpe ratio plot
        if 'sharpe' in oos_df.columns:
            fig.add_trace(
                go.Bar(
                    x=window_nums,
                    y=oos_df['sharpe'],
                    name='Sharpe Ratio',
                    marker_color='rgb(26, 118, 255)',
                    text=oos_df['sharpe'].round(2).astype(str),
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            # Add benchmark line (average Sharpe)
            fig.add_trace(
                go.Scatter(
                    x=window_nums,
                    y=[oos_df['sharpe'].mean()] * len(window_nums),
                    mode='lines',
                    name=f'Avg Sharpe: {oos_df["sharpe"].mean():.2f}',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=2
            )
        
        # 3. Max drawdown plot
        if 'max_drawdown' in oos_df.columns:
            fig.add_trace(
                go.Bar(
                    x=window_nums,
                    y=oos_df['max_drawdown'],
                    name='Max Drawdown (%)',
                    marker_color='rgb(204, 0, 0)',
                    text=oos_df['max_drawdown'].round(2).astype(str) + '%',
                    textposition='auto'
                ),
                row=2, col=1
            )
            
            # Add benchmark line (average max drawdown)
            fig.add_trace(
                go.Scatter(
                    x=window_nums,
                    y=[oos_df['max_drawdown'].mean()] * len(window_nums),
                    mode='lines',
                    name=f'Avg Drawdown: {oos_df["max_drawdown"].mean():.2f}%',
                    line=dict(color='black', dash='dash')
                ),
                row=2, col=1
            )
        
        # 4. Win rate plot
        if 'win_rate' in oos_df.columns:
            fig.add_trace(
                go.Bar(
                    x=window_nums,
                    y=oos_df['win_rate'],
                    name='Win Rate (%)',
                    marker_color='rgb(60, 179, 113)',
                    text=oos_df['win_rate'].round(2).astype(str) + '%',
                    textposition='auto'
                ),
                row=2, col=2
            )
            
            # Add benchmark line (average win rate)
            fig.add_trace(
                go.Scatter(
                    x=window_nums,
                    y=[oos_df['win_rate'].mean()] * len(window_nums),
                    mode='lines',
                    name=f'Avg Win Rate: {oos_df["win_rate"].mean():.2f}%',
                    line=dict(color='black', dash='dash')
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text='Out-of-Sample Performance Metrics Dashboard',
            title_font_size=20,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=800,
            width=1200,
            annotations=[
                dict(
                    text="This dashboard shows key performance metrics across all out-of-sample periods.<br>Consistent results across windows indicate strategy robustness.",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.15,
                    font=dict(size=14)
                )
            ]
        )
        
        # Update axes
        fig.update_xaxes(title_text='Window', row=2, col=1)
        fig.update_xaxes(title_text='Window', row=2, col=2)
        fig.update_yaxes(title_text='Return (%)', row=1, col=1)
        fig.update_yaxes(title_text='Sharpe Ratio', row=1, col=2)
        fig.update_yaxes(title_text='Max Drawdown (%)', row=2, col=1)
        fig.update_yaxes(title_text='Win Rate (%)', row=2, col=2)
        
        fig.show()
    
    # =========================================================================
    # 5. PARAMETER IMPACT HEATMAP
    # =========================================================================
    
    # This visualization shows how different parameters impact performance metrics
    # Helps identify which parameters are most important for the strategy
    
    if len(numeric_params) >= 2 and len(oos_df) >= 3:
        # Compute correlation between parameters and metrics
        correlation_data = []
        
        for param in numeric_params:
            for metric in ['return', 'sharpe', 'max_drawdown', 'win_rate']:
                if metric in oos_df.columns and not oos_df[metric].isna().all():
                    # Skip if all parameter values are identical
                    if params_df[param].nunique() <= 1:
                        continue
                        
                    # Get valid values for correlation calculation
                    param_values = params_df[param].values
                    metric_values = oos_df[metric].values
                    valid_mask = ~np.isnan(param_values) & ~np.isnan(metric_values)
                    
                    if sum(valid_mask) >= 3:  # Need at least 3 valid points for meaningful correlation
                        try:
                            corr = np.corrcoef(param_values[valid_mask], metric_values[valid_mask])[0, 1]
                            if not np.isnan(corr) and not np.isinf(corr):
                                correlation_data.append({
                                    'Parameter': param,
                                    'Metric': metric,
                                    'Correlation': corr
                                })
                        except Exception as e:
                            print(f"Could not calculate correlation for {param} vs {metric}: {e}")
        
        # Create a correlation heatmap if we have data
        if correlation_data:
            corr_df = pd.DataFrame(correlation_data)
            # Pivot to create a matrix suitable for heatmap
            pivot_df = corr_df.pivot(index='Parameter', columns='Metric', values='Correlation')
            
            plt.figure(figsize=(14, 10))
            heatmap = sns.heatmap(
                pivot_df,
                cmap="coolwarm",
                annot=True,
                fmt=".2f",
                linewidths=0.5,
                center=0,
                vmin=-1,
                vmax=1,
                cbar_kws={"shrink": .8, "label": "Correlation Coefficient"}
            )
            
            plt.title('Parameter Impact on Performance Metrics', fontsize=16)
            plt.ylabel('Parameter', fontsize=14)
            plt.xlabel('Performance Metric', fontsize=14)
            
            # Add description
            plt.figtext(0.5, 0.01, 
                       "This heatmap shows correlations between optimized parameters and out-of-sample performance metrics.\n"
                       "Positive correlations (blue) indicate that increasing the parameter tends to improve the metric.\n"
                       "Negative correlations (red) indicate that decreasing the parameter tends to improve the metric.",
                       ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
            
            plt.tight_layout(rect=[0, 0.09, 1, 0.97])
            plt.show()
    
    # =========================================================================
    # 6. TRADES ANALYSIS VISUALIZATION
    # =========================================================================
    
    # This visualization compares trade metrics across windows
    if 'n_trades' in oos_df.columns and not oos_df['n_trades'].isna().all():
        fig, ax1 = plt.subplots(figsize=(14, 10))
        
        # X-axis: window numbers
        window_labels = [f"{i+1}" for i in range(len(oos_df))]
        x = np.arange(len(window_labels))
        
        # Plot number of trades as bars
        bars = ax1.bar(x - 0.2, oos_df['n_trades'], width=0.4, color='skyblue', label='Number of Trades')
        ax1.set_xlabel('Window', fontsize=14)
        ax1.set_ylabel('Number of Trades', fontsize=14, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Add trade count labels above bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)
        
        # Plot win rate on the same graph with secondary y-axis
        ax2 = ax1.twinx()
        if 'win_rate' in oos_df.columns and not oos_df['win_rate'].isna().all():
            bars2 = ax2.bar(x + 0.2, oos_df['win_rate'], width=0.4, color='salmon', label='Win Rate (%)')
            ax2.set_ylabel('Win Rate (%)', fontsize=14, color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Add win rate labels above bars
            for i, bar in enumerate(bars2):
                height = bar.get_height()
                ax2.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=9)
        
        # Add a horizontal line for average win rate
        if 'win_rate' in oos_df.columns:
            avg_win_rate = oos_df['win_rate'].mean()
            ax2.axhline(y=avg_win_rate, color='red', linestyle='dashed', alpha=0.8, 
                       label=f'Avg Win Rate: {avg_win_rate:.2f}%')
        
        # Set x-ticks at bar positions
        ax1.set_xticks(x)
        ax1.set_xticklabels(window_labels)
        
        # Title and grid
        plt.title('Trade Metrics by Window', fontsize=16)
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Create a combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Add description
        plt.figtext(0.5, 0.01, 
                   "This chart compares the number of trades (blue) and win rate (red) across each window.\n"
                   "Consistent trade frequency and win rates indicate stable strategy performance.",
                   ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        plt.tight_layout(rect=[0, 0.07, 1, 0.97])
        plt.show()
    
    # =========================================================================
    # 7. DISTRIBUTION OF RETURNS (RISK ANALYSIS)
    # =========================================================================
    
    # This visualization shows the distribution of returns and key risk metrics
    if 'return' in oos_df.columns and not oos_df['return'].isna().all():
        plt.figure(figsize=(14, 10))
        
        # Create distribution plot with kernel density estimation
        sns.histplot(oos_df['return'].values, kde=True, stat="density", 
                   color='skyblue', bins=min(10, len(oos_df)))
        
        # Mark important statistics
        mean_return = oos_df['return'].mean()
        median_return = oos_df['return'].median()
        std_return = oos_df['return'].std()
        
        # Add vertical lines for mean, median
        plt.axvline(mean_return, color='red', linestyle='dashed', linewidth=2, 
                  label=f'Mean: {mean_return:.2f}%')
        plt.axvline(median_return, color='green', linestyle='dashed', linewidth=2, 
                   label=f'Median: {median_return:.2f}%')
        
        # Add vertical lines for mean ± 1 std dev (68% confidence interval)
        plt.axvline(mean_return + std_return, color='purple', linestyle='dotted', 
                  label=f'Mean + 1σ: {mean_return + std_return:.2f}%')
        plt.axvline(mean_return - std_return, color='purple', linestyle='dotted', 
                  label=f'Mean - 1σ: {mean_return - std_return:.2f}%')
        
        # Add horizontal line at y=0 to emphasize negative returns
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Calculate key metrics for the text box
        negative_returns = (oos_df['return'] < 0).mean() * 100
        positive_returns = (oos_df['return'] > 0).mean() * 100
        skewness = oos_df['return'].skew()
        kurtosis = oos_df['return'].kurtosis()
        
        # Add text box with metrics
        metrics_text = (
            f"Distribution Metrics:\n"
            f"Mean Return: {mean_return:.2f}%\n"
            f"Median Return: {median_return:.2f}%\n"
            f"Standard Deviation: {std_return:.2f}%\n"
            f"Negative Windows: {negative_returns:.1f}%\n"
            f"Positive Windows: {positive_returns:.1f}%\n"
            f"Skewness: {skewness:.2f}\n"
            f"Kurtosis: {kurtosis:.2f}"
        )
        
        # Add the metrics textbox
        plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                    backgroundcolor='white', alpha=0.8,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                    verticalalignment='top')
        
        # Customize the plot
        plt.title('Distribution of Out-of-Sample Returns', fontsize=16)
        plt.xlabel('Return (%)', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
        # Add description
        plt.figtext(0.5, 0.01, 
                   "This chart shows the distribution of out-of-sample returns across all windows.\n"
                   "A good strategy should have a distribution skewed to the right (positive returns)\n"
                   "with a majority of returns above zero.",
                   ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        plt.tight_layout(rect=[0, 0.07, 1, 0.97])
        plt.show()

def create_parameter_performance_map(wfo_results, top_n_params=5):
    """
    Creates a visual map showing which parameter combinations performed best
    across different windows, helping identify robust parameter sets.
    
    Parameters:
    -----------
    wfo_results : dict
        WFO results from walk_forward_optimization
    top_n_params : int, optional
        Number of top parameter combinations to analyze per window
        
    Returns:
    --------
    None
    """
    if not wfo_results['window_results']:
        print("No window results to analyze")
        return
    
    # Extract top parameter combinations for each window
    all_top_params = []
    
    for window_idx, window_result in enumerate(wfo_results['window_results']):
        if 'optimization_results' in window_result and window_result['optimization_results']:
            # Get top N parameter sets
            for i, param_set in enumerate(window_result['optimization_results'][:top_n_params]):
                # Skip if no combined_score
                if 'combined_score' not in param_set:
                    continue
                
                # Extract parameters (excluding scores and metrics)
                param_dict = {k: v for k, v in param_set.items() 
                             if k not in ['combined_score', 'metric1_name', 'metric2_name', 
                                         'weight_metric1', 'weight_metric2']}
                
                # Create a parameter key (string representation of parameters)
                param_key = ', '.join([f"{k}={v}" for k, v in sorted(param_dict.items())])
                
                all_top_params.append({
                    'window': window_idx + 1,
                    'rank': i + 1,
                    'param_key': param_key,
                    'score': param_set['combined_score'],
                    **param_dict  # Include individual parameters
                })
    
    # Convert to DataFrame
    if not all_top_params:
        print("No parameter data available for visualization")
        return
        
    params_df = pd.DataFrame(all_top_params)
    
    # =========================================================================
    # 1. PARAMETER FREQUENCY ANALYSIS
    # =========================================================================
    
    # Count frequency of each parameter combination
    param_counts = params_df['param_key'].value_counts().reset_index()
    param_counts.columns = ['Parameter Combination', 'Frequency']
    
    # Get the top N most frequent parameter combinations
    top_params = param_counts.head(min(10, len(param_counts)))
    
    plt.figure(figsize=(14, 10))
    bars = plt.barh(top_params['Parameter Combination'], top_params['Frequency'], color='skyblue')
    
    # Add count labels to bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f"{width}", va='center')
    
    plt.title('Most Frequent Top-Performing Parameter Combinations Across Windows', fontsize=16)
    plt.xlabel('Frequency (Number of Windows)', fontsize=14)
    plt.ylabel('Parameter Combination', fontsize=14)
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add description
    plt.figtext(0.5, 0.01, 
               "This chart shows which parameter combinations appeared most frequently in the top-performing results across windows.\n"
               "Parameter combinations that consistently perform well indicate more robust strategy settings.",
               ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.97])
    plt.show()
    
    # =========================================================================
    # 2. PARAMETER PERFORMANCE HEATMAP
    # =========================================================================
    
    # If we have any numeric parameters with multiple values, create 2D heatmaps
    numeric_params = params_df.select_dtypes(include=['number']).columns.tolist()
    # Filter out window, rank, and score columns
    numeric_params = [p for p in numeric_params if p not in ['window', 'rank', 'score']]
    
    if len(numeric_params) >= 2:
        # Get the two parameters with the most unique values
        param_unique_counts = [(param, params_df[param].nunique()) for param in numeric_params]
        param_unique_counts.sort(key=lambda x: x[1], reverse=True)
        
        # Select the top two parameters for the heatmap
        if len(param_unique_counts) >= 2:
            param_x, _ = param_unique_counts[0]
            param_y, _ = param_unique_counts[1]
            
            # Create a pivot table of average scores for parameter combinations
            pivot_data = params_df.pivot_table(
                values='score', 
                index=param_y,
                columns=param_x,
                aggfunc='mean'
            )
            
            plt.figure(figsize=(14, 10))
            heatmap = sns.heatmap(
                pivot_data,
                annot=True,
                fmt=".2f",
                cmap='viridis',
                linewidths=0.5,
                cbar_kws={"shrink": .8, "label": "Average Performance Score"}
            )
            
            plt.title(f'Parameter Performance Map: {param_y} vs {param_x}', fontsize=16)
            plt.xlabel(param_x, fontsize=14)
            plt.ylabel(param_y, fontsize=14)
            
            # Add description
            plt.figtext(0.5, 0.01, 
                       f"This heatmap visualizes how different combinations of {param_x} and {param_y} impact performance.\n"
                       "Darker colors indicate better performance. This helps identify optimal parameter regions across all windows.",
                       ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
            
            plt.tight_layout(rect=[0, 0.07, 1, 0.97])
            plt.show()
    
    # =========================================================================
    # 3. PARAMETER RANKING VISUALIZATION
    # =========================================================================
    
    # Create visualization of parameter ranks across windows
    # This shows if certain parameters consistently rank high
    
    # Get top 3 parameter combinations for each window
    top3_params = params_df[params_df['rank'] <= 3].copy()
    
    if len(top3_params) > 0:
        plt.figure(figsize=(14, 10))
        
        # Create a rank plot by window
        sns.scatterplot(
            data=top3_params,
            x='window',
            y='rank',
            hue='param_key',
            size='score',
            sizes=(100, 400),
            alpha=0.7,
            palette='viridis'
        )
        
        # Customize the plot
        plt.title('Top Parameter Rankings by Window', fontsize=16)
        plt.xlabel('Window', fontsize=14)
        plt.ylabel('Rank (1 = Best)', fontsize=14)
        plt.yticks([1, 2, 3])
        plt.grid(True, alpha=0.3)
        
        # Reverse y-axis so that rank 1 is at the top
        plt.gca().invert_yaxis()
        
        # If there are too many parameter combinations, limit the legend
        if len(top3_params['param_key'].unique()) > 10:
            # Get the most frequent parameter combinations for the legend
            top_params = top3_params['param_key'].value_counts().head(10).index.tolist()
            handles, labels = plt.gca().get_legend_handles_labels()
            
            # Create a filtered legend
            new_handles = []
            new_labels = []
            for handle, label in zip(handles, labels):
                if label in top_params or 'score' in label:
                    new_handles.append(handle)
                    new_labels.append(label)
            
            plt.legend(new_handles, new_labels, title='Parameter Combinations', 
                     loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        else:
            plt.legend(title='Parameter Combinations', loc='upper center', 
                     bbox_to_anchor=(0.5, -0.15), ncol=2)
        
        # Add description
        plt.figtext(0.5, 0.01, 
                   "This chart shows which parameter combinations ranked in the top 3 for each window.\n"
                   "Parameter combinations that appear multiple times demonstrate consistent performance.\n"
                   "Marker size indicates performance score (larger = better).",
                   ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        plt.tight_layout(rect=[0, 0.2, 1, 0.97])
        plt.show()

def visualize_robustness_metrics(wfo_results):
    """
    Creates visualizations specifically focused on strategy robustness metrics
    derived from Walk-Forward Optimization results.
    
    Parameters:
    -----------
    wfo_results : dict
        WFO results from walk_forward_optimization
        
    Returns:
    --------
    None
    """
    if not wfo_results['out_of_sample_performance']:
        print("No out-of-sample results to analyze robustness")
        return
    
    # Extract out-of-sample performance data
    oos_df = pd.DataFrame(wfo_results['out_of_sample_performance'])
    
    # Extract parameter data
    params_df = pd.DataFrame(wfo_results['best_params'])
    
    # Define robustness metrics
    robustness_metrics = {}
    
    # 1. Performance Consistency: % of positive OOS periods
    if 'return' in oos_df.columns:
        positive_periods = (oos_df['return'] > 0).mean() * 100
        robustness_metrics['Positive Periods (%)'] = positive_periods
    
    # 2. Performance Stability: coefficient of variation of returns
    #    Lower values indicate more stable returns
    if 'return' in oos_df.columns and not oos_df['return'].isna().all():
        # Calculate coefficient of variation (std/mean) for non-zero mean
        mean_return = oos_df['return'].mean()
        if abs(mean_return) > 1e-6:  # Avoid division by zero or tiny numbers
            cv_return = oos_df['return'].std() / abs(mean_return)
            # Convert to stability (1 - normalized CV)
            # Limit to range [0, 1] by using 1 / (1 + CV)
            return_stability = 1 / (1 + cv_return)
            robustness_metrics['Return Stability (0-1)'] = return_stability
    
    # 3. Sharpe Consistency: coefficient of variation of Sharpe ratios
    if 'sharpe' in oos_df.columns and not oos_df['sharpe'].isna().all():
        # Only for positive mean Sharpe
        mean_sharpe = oos_df['sharpe'].mean()
        if mean_sharpe > 1e-6:
            cv_sharpe = oos_df['sharpe'].std() / mean_sharpe
            sharpe_stability = 1 / (1 + cv_sharpe)
            robustness_metrics['Sharpe Stability (0-1)'] = sharpe_stability
    
    # 4. Win Rate Consistency
    if 'win_rate' in oos_df.columns and not oos_df['win_rate'].isna().all():
        mean_win_rate = oos_df['win_rate'].mean()
        if mean_win_rate > 1e-6:
            cv_win_rate = oos_df['win_rate'].std() / mean_win_rate
            win_rate_stability = 1 / (1 + cv_win_rate)
            robustness_metrics['Win Rate Stability (0-1)'] = win_rate_stability
    
    # 5. Parameter Stability: average of 1 - CV for each parameter
    param_stability = {}
    numeric_params = params_df.select_dtypes(include=['number']).columns.tolist()
    
    for param in numeric_params:
        mean_value = params_df[param].mean()
        if abs(mean_value) > 1e-6:  # Avoid division by zero
            cv = params_df[param].std() / abs(mean_value)
            stability = 1 / (1 + cv)
            param_stability[param] = stability
    
    if param_stability:
        robustness_metrics['Avg Parameter Stability (0-1)'] = np.mean(list(param_stability.values()))
        # Also store individual parameter stability
        for param, stability in param_stability.items():
            robustness_metrics[f'{param} Stability'] = stability
    
    # 6. OOS vs IS Performance Ratio 
    # If we have both IS and OOS metrics, calculate ratio
    is_metrics = []
    for window_result in wfo_results['window_results']:
        if window_result['optimization_results'] and len(window_result['optimization_results']) > 0:
            best_result = window_result['optimization_results'][0]
            if 'combined_score' in best_result:
                is_metrics.append(best_result['combined_score'])
    
    if is_metrics and 'return' in oos_df.columns:
        is_avg = np.mean(is_metrics)
        oos_avg = oos_df['return'].mean()
        
        if abs(is_avg) > 1e-6:  # Avoid division by zero
            oos_is_ratio = oos_avg / is_avg
            # Normalize to 0-1 scale: 1 means OOS = IS (perfect), 0 means completely different
            oos_is_consistency = 1 - min(1, abs(1 - oos_is_ratio))
            robustness_metrics['OOS/IS Consistency (0-1)'] = oos_is_consistency
    
    # 7. Calculate overall robustness score (average of all metrics)
    # Filter to only include 0-1 scaled metrics
    scaled_metrics = {k: v for k, v in robustness_metrics.items() if '(0-1)' in k}
    if scaled_metrics:
        robustness_metrics['Overall Robustness Score (0-1)'] = np.mean(list(scaled_metrics.values()))
    
    # =========================================================================
    # VISUALIZATION: ROBUSTNESS DASHBOARD
    # =========================================================================
    
    plt.figure(figsize=(14, 10))
    
    # Filter metrics to only include those on 0-1 scale for the radar chart
    radar_metrics = {k.replace(' (0-1)', ''): v for k, v in robustness_metrics.items() if '(0-1)' in k}
    
    if len(radar_metrics) >= 3:
        # Create radar chart (spider plot) for robustness metrics
        # Prepare data for radar chart
        categories = list(radar_metrics.keys())
        values = list(radar_metrics.values())
        
        # Close the plot by appending the first value at the end
        categories = categories + [categories[0]]
        values = values + [values[0]]
        
        # Calculate angle for each category
        N = len(categories) - 1  # Excluding the repeated first element
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create radar plot
        ax = plt.subplot(2, 2, 1, polar=True)
        
        # Draw the chart
        plt.polar(angles, values, marker='o', linestyle='-', linewidth=2, label='Robustness Metrics')
        
        # Fill the area
        plt.fill(angles, values, alpha=0.25)
        
        # Set category labels
        plt.xticks(angles[:-1], categories[:-1])
        
        # Set radial limits
        plt.ylim(0, 1)
        
        # Add radial grid lines at 0.2, 0.4, 0.6, 0.8
        plt.yticks([0.2, 0.4, 0.6, 0.8], ['0.2', '0.4', '0.6', '0.8'], color='grey', size=8)
        
        # Draw y-axis circles
        for ytick in [0.2, 0.4, 0.6, 0.8]:
            ax.add_artist(plt.Circle((0, 0), ytick, fill=False, color='grey', linestyle='--', alpha=0.4))
        
        # Title for this subplot
        plt.title('Strategy Robustness Metrics (1.0 = Ideal)', fontsize=14, pad=20)
    
    # Create bar chart with all robustness metrics
    ax2 = plt.subplot(2, 2, 2)
    
    metrics_to_plot = {k: v for k, v in robustness_metrics.items() 
                     if not k.startswith('Avg') and '(0-1)' in k}
    
    # Sort metrics by value
    sorted_metrics = dict(sorted(metrics_to_plot.items(), key=lambda item: item[1], reverse=True))
    
    # Plot bar chart
    bars = plt.barh(
        [k.replace(' (0-1)', '') for k in sorted_metrics.keys()], 
        list(sorted_metrics.values()),
        color='skyblue'
    )
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_x_pos = width + 0.01
        plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
               f'{width:.2f}', va='center')
    
    plt.xlim(0, 1.1)
    plt.xlabel('Score (0-1 Scale)', fontsize=12)
    plt.ylabel('Metric', fontsize=12)
    plt.title('Robustness Metrics Comparison', fontsize=14)
    plt.grid(True, axis='x', alpha=0.3)
    
    # Create parameter stability bar chart
    param_stability_items = {k: v for k, v in robustness_metrics.items() 
                           if 'Stability' in k and not k.startswith('Avg') and '(0-1)' not in k}
    
    if param_stability_items:
        ax3 = plt.subplot(2, 2, 3)
        
        # Sort parameters by stability
        sorted_param_stability = dict(sorted(param_stability_items.items(), 
                                           key=lambda item: item[1], reverse=True))
        
        # Plot bar chart
        bars = plt.barh(
            list(sorted_param_stability.keys()), 
            list(sorted_param_stability.values()),
            color='lightgreen'
        )
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_x_pos = width + 0.01
            plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                   f'{width:.2f}', va='center')
        
        plt.xlim(0, 1.1)
        plt.xlabel('Stability Score (0-1 Scale)', fontsize=12)
        plt.ylabel('Parameter', fontsize=12)
        plt.title('Parameter Stability Analysis', fontsize=14)
        plt.grid(True, axis='x', alpha=0.3)
    
    # Create text summary of robustness analysis
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')  # Turn off axis
    
    # Prepare summary text
    summary_text = "Robustness Analysis Summary:\n\n"
    
    if 'Overall Robustness Score (0-1)' in robustness_metrics:
        score = robustness_metrics['Overall Robustness Score (0-1)']
        summary_text += f"Overall Robustness: {score:.2f}/1.00\n\n"
        
        # Add interpretation
        if score >= 0.8:
            summary_text += "Interpretation: Excellent robustness - highly consistent across all metrics.\n"
        elif score >= 0.6:
            summary_text += "Interpretation: Good robustness - consistent performance with minor variations.\n"
        elif score >= 0.4:
            summary_text += "Interpretation: Moderate robustness - some inconsistencies but generally acceptable.\n"
        elif score >= 0.2:
            summary_text += "Interpretation: Low robustness - significant inconsistencies across metrics.\n"
        else:
            summary_text += "Interpretation: Poor robustness - extremely inconsistent performance.\n"
    
    # Add details about positive periods
    if 'Positive Periods (%)' in robustness_metrics:
        pos_periods = robustness_metrics['Positive Periods (%)']
        summary_text += f"\nPositive Periods: {pos_periods:.1f}% of out-of-sample windows\n"
    
    # Add OOS/IS comparison if available
    if 'OOS/IS Consistency (0-1)' in robustness_metrics:
        oos_is = robustness_metrics['OOS/IS Consistency (0-1)']
        summary_text += f"OOS/IS Consistency: {oos_is:.2f}/1.00\n"
        
        if oos_is >= 0.8:
            summary_text += "    (Very small performance drop from in-sample to out-of-sample)\n"
        elif oos_is >= 0.5:
            summary_text += "    (Moderate performance drop from in-sample to out-of-sample)\n"
        else:
            summary_text += "    (Significant performance drop from in-sample to out-of-sample)\n"
    
    # Add parameter stability info
    if 'Avg Parameter Stability (0-1)' in robustness_metrics:
        param_stab = robustness_metrics['Avg Parameter Stability (0-1)']
        summary_text += f"\nParameter Stability: {param_stab:.2f}/1.00\n"
        
        if param_stab >= 0.8:
            summary_text += "    (Highly stable parameters across windows - strategy is robust)\n"
        elif param_stab >= 0.5:
            summary_text += "    (Moderately stable parameters - acceptable robustness)\n"
        else:
            summary_text += "    (Unstable parameters - may indicate curve-fitting)\n"
    
    # Add the text to the plot
    plt.text(0, 1.0, summary_text, fontsize=11, va='top', linespacing=1.5)
    
    # Add overall title
    plt.suptitle('Strategy Robustness Analysis Dashboard', fontsize=18, y=0.98)
    
    # Add description at the bottom
    plt.figtext(0.5, 0.01, 
               "This dashboard provides a comprehensive analysis of strategy robustness based on WFO results.\n"
               "Higher scores (closer to 1.0) indicate better robustness across all metrics.\n"
               "A truly robust strategy should perform consistently across different market conditions.",
               ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.subplots_adjust(top=0.9)
    plt.show()

def generate_wfo_report_pdf(wfo_results, df, output_path=None, strategy_name="ATDMF Strategy"):
    """
    Génère un rapport PDF complet contenant tous les résultats et graphiques
    de l'analyse Walk-Forward Optimization.
    
    Parameters:
    -----------
    wfo_results : dict
        Résultats du walk_forward_optimization
    df : pandas.DataFrame
        DataFrame OHLCV original
    output_path : str, optional
        Chemin où sauvegarder le PDF. Par défaut, "{strategy_name}_WFO_Report_{date}.pdf"
    strategy_name : str, optional
        Nom de la stratégie pour le titre du rapport
        
    Returns:
    --------
    str
        Chemin vers le fichier PDF généré
    """
    
    # Configurer le style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (14, 10)
    plt.rcParams['figure.dpi'] = 100
    
    # Créer le chemin du fichier de sortie s'il n'est pas spécifié
    if output_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = 'WFO_Reports'
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/{strategy_name.replace(' ', '_')}_WFO_Report_{timestamp}.pdf"
    
    # Extraire les données de performance
    if wfo_results['out_of_sample_performance']:
        oos_df = pd.DataFrame(wfo_results['out_of_sample_performance'])
    else:
        oos_df = pd.DataFrame()
    
    params_df = pd.DataFrame(wfo_results['best_params'])
    
    # Créer le PDF
    with PdfPages(output_path) as pdf:
        
        # =====================================================================
        # PAGE DE TITRE
        # =====================================================================
        plt.figure(figsize=(14, 10))
        plt.axis('off')
        
        # Titre du rapport
        plt.text(0.5, 0.8, f"Rapport d'Analyse Walk-Forward Optimization", 
                fontsize=15, ha='center')
        plt.text(0.5, 0.7, f"{strategy_name}", fontsize=24, ha='center')
        
        # Date et informations de base
        today = datetime.datetime.now().strftime("%d/%m/%Y")
        plt.text(0.5, 0.6, f"Généré le {today}", fontsize=16, ha='center')
        
        # Statistiques de base
        if 'settings' in wfo_results:
            settings = wfo_results['settings']
            info_text = (
                f"Nombre de fenêtres: {settings['n_windows']}\n"
                f"Type de WFO: {'Ancrée' if settings['anchored'] else 'Non-ancrée'}\n"
                f"Proportion d'entraînement: {settings['train_size']*100:.0f}%\n"
                f"Métrique primaire: {settings['optimization_metric']}\n"
                f"Métrique secondaire: {settings['secondary_metric']}\n"
            )
            plt.text(0.5, 0.45, info_text, fontsize=14, ha='center', linespacing=1.5)
        
        # Icône ou logo (optionnel)
        # Vous pourriez ajouter un logo ici si nécessaire
        
        # Ajouter la première page
        pdf.savefig()
        plt.close()
        
        # =====================================================================
        # RÉSUMÉ DES PERFORMANCES
        # =====================================================================
        plt.figure(figsize=(14, 10))
        plt.axis('off')
        
        plt.text(0.5, 0.95, "Résumé des Performances", fontsize=24, ha='center')
        
        # Tableau des performances OOS
        if not oos_df.empty:
            # Calculer les statistiques agrégées
            summary = {
                "Rendement moyen (%)": oos_df['return'].mean() if 'return' in oos_df.columns else np.nan,
                "Rendement cumulatif (%)": ((1 + oos_df['return']/100).prod() - 1) * 100 if 'return' in oos_df.columns else np.nan, 
                "Ratio de Sharpe moyen": oos_df['sharpe'].mean() if 'sharpe' in oos_df.columns else np.nan,
                "Drawdown maximum moyen (%)": oos_df['max_drawdown'].mean() if 'max_drawdown' in oos_df.columns else np.nan,
                "Taux de réussite moyen (%)": oos_df['win_rate'].mean() if 'win_rate' in oos_df.columns else np.nan,
                "Ratio de Calmar moyen": oos_df['calmar_ratio'].mean() if 'calmar_ratio' in oos_df.columns else np.nan,
                "Ratio de Sortino moyen": oos_df['sortino_ratio'].mean() if 'sortino_ratio' in oos_df.columns else np.nan,
                "Nombre total de trades": oos_df['n_trades'].sum() if 'n_trades' in oos_df.columns else np.nan,
                "% de périodes positives": (oos_df['return'] > 0).mean() * 100 if 'return' in oos_df.columns else np.nan
            }
            
            # Créer un tableau pour les résultats résumés
            table_data = []
            for metric, value in summary.items():
                if not np.isnan(value):
                    if "(%)" in metric:
                        formatted_value = f"{value:.2f}%"
                    elif "Nombre" in metric:
                        formatted_value = f"{int(value)}"
                    else:
                        formatted_value = f"{value:.4f}"
                    table_data.append([metric, formatted_value])
            
            if table_data:
                table = plt.table(cellText=table_data, 
                                 colLabels=["Métrique", "Valeur"], 
                                 loc='center', 
                                 cellLoc='center', 
                                 bbox=[0.2, 0.6, 0.6, 0.25])
                table.auto_set_font_size(False)
                table.set_fontsize(12)
                table.scale(1, 1.5)
                
                # Ajouter un titre au tableau
                plt.text(0.5, 0.87, "Métriques Agrégées Out-of-Sample", fontsize=16, ha='center')
        
        # Tableau des paramètres optimaux agrégés
        if not params_df.empty:
            # Calculer les paramètres optimaux agrégés (moyenne)
            aggregated_params = {}
            for param in params_df.columns:
                if pd.api.types.is_numeric_dtype(params_df[param]):
                    if param in ['timeperiod', 'fenetre_lowest', 'user_exit_sma_length']:
                        aggregated_params[param] = int(round(params_df[param].mean()))
                    else:
                        aggregated_params[param] = round(params_df[param].mean(), 2)
            
            # Créer un tableau pour les paramètres agrégés
            param_data = [[param, value] for param, value in aggregated_params.items()]
            
            if param_data:
                param_table = plt.table(cellText=param_data, 
                                      colLabels=["Paramètre", "Valeur Moyenne"], 
                                      loc='center', 
                                      cellLoc='center', 
                                      bbox=[0.2, 0.25, 0.6, 0.25])
                param_table.auto_set_font_size(False)
                param_table.set_fontsize(12)
                param_table.scale(1, 1.5)
                
                # Ajouter un titre au tableau
                plt.text(0.5, 0.52, "Paramètres Optimaux Agrégés", fontsize=16, ha='center')
        
        # Ajouter la page de résumé
        pdf.savefig()
        plt.close()
        
        # =====================================================================
        # GRAPHIQUE DES PRIX AVEC FENÊTRES WFO
        # =====================================================================
        
        # Fonction utilitaire pour capturer les figures matplotlib
        def save_figure_to_pdf(pdf):
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            pdf.savefig()
            plt.close()
            
        # Fonction utilitaire pour capturer les figures plotly
        def save_plotly_to_pdf(fig, pdf):
            buf = io.BytesIO()
            pio.write_image(fig, buf, format='png', width=900, height=600)
            buf.seek(0)
            img = Image.open(buf)
            
            plt.figure(figsize=(14, 10))
            plt.imshow(np.array(img))
            plt.axis('off')
            pdf.savefig()
            plt.close()
        
        # Graphique des prix avec fenêtres WFO
        plt.figure(figsize=(14, 10))
        
        # Plot the close price
        if 'Close' in df.columns:
            price_series = df['Close']
        elif 'close' in df.columns:
            price_series = df['close']
        else:
            price_series = df.iloc[:, 0]  # En dernier recours
        
        # Tracer la série de prix
        plt.plot(df.index, price_series, label='Prix', color='#1f77b4')
        
        # Ajouter des lignes verticales pour les limites des fenêtres
        window_colors = {
            'train': 'green',
            'test': 'red'
        }
        
        # Ajouter des lignes verticales pour les limites des fenêtres
        for window_idx, window_result in enumerate(wfo_results['window_results']):
            window_info = window_result['window_info']
            
            # Ajouter une ligne verticale au début de la fenêtre
            plt.axvline(x=pd.to_datetime(window_info['start_date']), 
                       color='black', linestyle='--', alpha=0.5)
            
            # Marquer les régions in-sample et out-of-sample
            if window_info['in_sample_start'] and window_info['in_sample_end']:
                plt.axvspan(
                    pd.to_datetime(window_info['in_sample_start']),
                    pd.to_datetime(window_info['in_sample_end']),
                    alpha=0.15, color=window_colors['train'], 
                    label=f'In-Sample {window_idx+1}' if window_idx == 0 else ""
                )
            
            if window_info['out_sample_start'] and window_info['out_sample_end']:
                plt.axvspan(
                    pd.to_datetime(window_info['out_sample_start']),
                    pd.to_datetime(window_info['out_sample_end']),
                    alpha=0.15, color=window_colors['test'], 
                    label=f'Out-of-Sample {window_idx+1}' if window_idx == 0 else ""
                )
        
        # Personnaliser le graphique
        plt.title('Graphique des Prix avec Fenêtres Walk-Forward Optimization', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Prix', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        save_figure_to_pdf(pdf)
        
        # =====================================================================
        # COMPARAISON IN-SAMPLE VS OUT-OF-SAMPLE
        # =====================================================================
        
        # Extraire les métriques de performance pour in-sample et out-of-sample
        is_metrics = []
        for window_idx, window_result in enumerate(wfo_results['window_results']):
            # Obtenir le meilleur résultat in-sample des résultats d'optimisation
            if 'optimization_results' in window_result and window_result['optimization_results']:
                best_result = window_result['optimization_results'][0]
                if 'combined_score' in best_result:
                    is_metrics.append({
                        'window': window_idx + 1,
                        'performance': best_result['combined_score'],
                        'type': 'In-Sample'
                    })
        
        # Obtenir les métriques out-of-sample
        oos_metrics = []
        for metric in oos_df.to_dict('records'):
            if 'window' in metric:
                oos_metrics.append({
                    'window': metric['window'],
                    'performance': metric['return'] if 'return' in metric else (
                                  metric['sharpe'] if 'sharpe' in metric else 0),
                    'type': 'Out-of-Sample'
                })
        
        # Combiner les métriques
        comparison_df = pd.DataFrame(is_metrics + oos_metrics)
        
        if not comparison_df.empty:
            plt.figure(figsize=(14, 10))
            
            # Créer un graphique à barres groupées
            ax = sns.barplot(
                x='window', 
                y='performance', 
                hue='type',
                data=comparison_df,
                palette={'In-Sample': 'skyblue', 'Out-of-Sample': 'salmon'}
            )
            
            # Ajouter des étiquettes de valeur sur les barres
            for i, p in enumerate(ax.patches):
                height = p.get_height()
                ax.text(p.get_x() + p.get_width()/2., height + 0.1,
                       f'{height:.2f}', ha="center")
            
            plt.title('Comparaison In-Sample vs Out-of-Sample par Fenêtre', fontsize=16)
            plt.xlabel('Fenêtre', fontsize=14)
            plt.ylabel('Performance', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(title='Type')
            
            save_figure_to_pdf(pdf)
        
        # =====================================================================
        # ANALYSE DE STABILITÉ DES PARAMÈTRES
        # =====================================================================
        
        # Cette visualisation montre comment les paramètres évoluent à travers les fenêtres
        
        # Filtrer uniquement les paramètres numériques
        numeric_params = params_df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_params) > 0:
            # Créer un graphique de stabilité des paramètres
            plt.figure(figsize=(14, 10))
            
            # Obtenir les valeurs de paramètres normalisées pour comparaison à travers différentes échelles
            norm_params_df = pd.DataFrame()
            for param in numeric_params:
                # Ignorer si toutes les valeurs sont identiques (stabilité = 1.0)
                if params_df[param].nunique() <= 1:
                    continue
                    
                # Normaliser à l'échelle 0-1 pour tracer sur la même échelle
                min_val = params_df[param].min()
                max_val = params_df[param].max()
                
                if max_val > min_val:  # Éviter la division par zéro
                    norm_params_df[param] = (params_df[param] - min_val) / (max_val - min_val)
                else:
                    norm_params_df[param] = params_df[param] / params_df[param]
            
            # Créer l'axe des x (numéros de fenêtre)
            x = list(range(1, len(params_df) + 1))
            
            # Tracer chaque paramètre
            markers = ['o', 's', 'd', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'D', '|', '_']
            for i, param in enumerate(norm_params_df.columns):
                marker = markers[i % len(markers)]
                plt.plot(x, norm_params_df[param], marker=marker, label=param, linewidth=2, markersize=8)
                
                # Calculer la métrique de stabilité (1 - coefficient de variation)
                stability = 1.0 - (np.std(params_df[param]) / np.mean(params_df[param])) if np.mean(params_df[param]) > 0 else 0.0
                
                # Annoter avec la valeur de stabilité
                plt.annotate(f"Stabilité: {stability:.2f}", 
                            xy=(x[-1], norm_params_df[param].iloc[-1]),
                            xytext=(x[-1] + 0.1, norm_params_df[param].iloc[-1]),
                            fontsize=9)
            
            # Ajouter un tableau de valeurs absolues des paramètres
            param_table = ''
            for i, window in enumerate(x):
                param_table += f'Fenêtre {window}: '
                param_values = []
                for param in numeric_params:
                    if param in norm_params_df.columns:
                        param_values.append(f"{param}={params_df[param].iloc[i]:.2f}")
                param_table += ', '.join(param_values) + '\n'
            
            plt.figtext(0.5, 0.01, param_table, ha="center", fontsize=9, 
                       bbox={"facecolor":"white", "alpha":0.8, "pad":5})
            
            # Personnaliser le graphique
            plt.title('Stabilité des Paramètres à Travers les Fenêtres (Valeurs Normalisées)', fontsize=16)
            plt.xlabel('Fenêtre', fontsize=14)
            plt.ylabel('Valeur Normalisée du Paramètre', fontsize=14)
            plt.xticks(x)
            plt.grid(True, alpha=0.3)
            plt.ylim(-0.05, 1.05)  # Donner une marge au-dessus et en-dessous
            plt.legend(loc='best')
            
            save_figure_to_pdf(pdf)
        
        # =====================================================================
        # DASHBOARD DES MÉTRIQUES DE PERFORMANCE OUT-OF-SAMPLE
        # =====================================================================
        
        if not oos_df.empty:
            # Créer des graphiques séparés pour chaque métrique clé
            
            # 1. Graphique des rendements
            if 'return' in oos_df.columns:
                plt.figure(figsize=(14, 10))
                window_nums = list(range(1, len(oos_df) + 1))
                
                bars = plt.bar(window_nums, oos_df['return'], color='royalblue')
                
                # Ajouter des étiquettes de valeur sur les barres
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}%', ha='center', va='bottom')
                
                # Ajouter une ligne de référence pour la moyenne
                plt.axhline(y=oos_df['return'].mean(), color='red', linestyle='--', 
                          label=f'Moyenne: {oos_df["return"].mean():.2f}%')
                
                plt.title('Rendements par Fenêtre (%)', fontsize=16)
                plt.xlabel('Fenêtre', fontsize=14)
                plt.ylabel('Rendement (%)', fontsize=14)
                plt.xticks(window_nums)
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                save_figure_to_pdf(pdf)
            
            # 2. Graphique du ratio de Sharpe
            if 'sharpe' in oos_df.columns:
                plt.figure(figsize=(14, 10))
                
                bars = plt.bar(window_nums, oos_df['sharpe'], color='green')
                
                # Ajouter des étiquettes de valeur sur les barres
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom')
                
                # Ajouter une ligne de référence pour la moyenne
                plt.axhline(y=oos_df['sharpe'].mean(), color='red', linestyle='--', 
                          label=f'Moyenne: {oos_df["sharpe"].mean():.2f}')
                
                plt.title('Ratio de Sharpe par Fenêtre', fontsize=16)
                plt.xlabel('Fenêtre', fontsize=14)
                plt.ylabel('Ratio de Sharpe', fontsize=14)
                plt.xticks(window_nums)
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                save_figure_to_pdf(pdf)
            
            # 3. Graphique du Drawdown Maximum
            if 'max_drawdown' in oos_df.columns:
                plt.figure(figsize=(14, 10))
                
                bars = plt.bar(window_nums, oos_df['max_drawdown'], color='firebrick')
                
                # Ajouter des étiquettes de valeur sur les barres
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}%', ha='center', va='bottom')
                
                # Ajouter une ligne de référence pour la moyenne
                plt.axhline(y=oos_df['max_drawdown'].mean(), color='black', linestyle='--', 
                          label=f'Moyenne: {oos_df["max_drawdown"].mean():.2f}%')
                
                plt.title('Drawdown Maximum par Fenêtre (%)', fontsize=16)
                plt.xlabel('Fenêtre', fontsize=14)
                plt.ylabel('Drawdown Maximum (%)', fontsize=14)
                plt.xticks(window_nums)
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                save_figure_to_pdf(pdf)
            
            # 4. Graphique du Taux de Réussite et Nombre de Trades
            if 'win_rate' in oos_df.columns and 'n_trades' in oos_df.columns:
                plt.figure(figsize=(14, 10))
                
                fig, ax1 = plt.subplots(figsize=(14, 10))
                
                # Tracer le nombre de trades comme des barres
                bars = ax1.bar(np.array(window_nums) - 0.2, oos_df['n_trades'], 
                             width=0.4, color='skyblue', label='Nombre de Trades')
                ax1.set_xlabel('Fenêtre', fontsize=14)
                ax1.set_ylabel('Nombre de Trades', fontsize=14, color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                
                # Ajouter les étiquettes du nombre de trades au-dessus des barres
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax1.annotate(f'{int(height)}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points de décalage vertical
                               textcoords="offset points",
                               ha='center', va='bottom',
                               fontsize=9)
                
                # Tracer le taux de réussite sur le même graphique avec un axe y secondaire
                ax2 = ax1.twinx()
                bars2 = ax2.bar(np.array(window_nums) + 0.2, oos_df['win_rate'], 
                              width=0.4, color='salmon', label='Taux de Réussite (%)')
                ax2.set_ylabel('Taux de Réussite (%)', fontsize=14, color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                
                # Ajouter les étiquettes du taux de réussite au-dessus des barres
                for i, bar in enumerate(bars2):
                    height = bar.get_height()
                    ax2.annotate(f'{height:.1f}%',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points de décalage vertical
                               textcoords="offset points",
                               ha='center', va='bottom',
                               fontsize=9)
                
                # Ajouter une ligne horizontale pour le taux de réussite moyen
                avg_win_rate = oos_df['win_rate'].mean()
                ax2.axhline(y=avg_win_rate, color='red', linestyle='dashed', alpha=0.8, 
                           label=f'Taux moyen: {avg_win_rate:.2f}%')
                
                # Définir les ticks x aux positions des barres
                ax1.set_xticks(window_nums)
                
                # Titre et grille
                plt.title('Métriques de Trading par Fenêtre', fontsize=16)
                ax1.grid(True, axis='y', alpha=0.3)
                
                # Créer une légende combinée
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                
                save_figure_to_pdf(pdf)
        
        # =====================================================================
        # ANALYSE DES IMPACTS DES PARAMÈTRES
        # =====================================================================
        
        # Cette visualisation montre comment différents paramètres impactent les métriques de performance
        
        if len(numeric_params) >= 2 and len(oos_df) >= 3:
            # Calculer la corrélation entre les paramètres et les métriques
            correlation_data = []
            
            for param in numeric_params:
                for metric in ['return', 'sharpe', 'max_drawdown', 'win_rate']:
                    if metric in oos_df.columns and not oos_df[metric].isna().all():
                        # Ignorer si toutes les valeurs de paramètres sont identiques
                        if params_df[param].nunique() <= 1:
                            continue
                            
                        # Obtenir des valeurs valides pour le calcul de corrélation
                        param_values = params_df[param].values
                        metric_values = oos_df[metric].values
                        valid_mask = ~np.isnan(param_values) & ~np.isnan(metric_values)
                        
                        if sum(valid_mask) >= 3:  # Besoin d'au moins 3 points valides pour une corrélation significative
                            try:
                                corr = np.corrcoef(param_values[valid_mask], metric_values[valid_mask])[0, 1]
                                if not np.isnan(corr) and not np.isinf(corr):
                                    correlation_data.append({
                                        'Paramètre': param,
                                        'Métrique': metric,
                                        'Corrélation': corr
                                    })
                            except Exception as e:
                                print(f"Impossible de calculer la corrélation pour {param} vs {metric}: {e}")
            
            # Créer une heatmap de corrélation si nous avons des données
            if correlation_data:
                corr_df = pd.DataFrame(correlation_data)
                # Pivoter pour créer une matrice adaptée à la heatmap
                pivot_df = corr_df.pivot(index='Paramètre', columns='Métrique', values='Corrélation')
                
                plt.figure(figsize=(14, 10))
                heatmap = sns.heatmap(
                    pivot_df,
                    cmap="coolwarm",
                    annot=True,
                    fmt=".2f",
                    linewidths=0.5,
                    center=0,
                    vmin=-1,
                    vmax=1,
                    cbar_kws={"shrink": .8, "label": "Coefficient de Corrélation"}
                )
                
                plt.title('Impact des Paramètres sur les Métriques de Performance', fontsize=16)
                plt.ylabel('Paramètre', fontsize=14)
                plt.xlabel('Métrique de Performance', fontsize=14)
                
                save_figure_to_pdf(pdf)
        
        # =====================================================================
        # DISTRIBUTION DES RENDEMENTS (ANALYSE DE RISQUE)
        # =====================================================================
        
        # Cette visualisation montre la distribution des rendements et les métriques de risque clés
        if 'return' in oos_df.columns and not oos_df['return'].isna().all():
            plt.figure(figsize=(14, 10))
            
            # Créer un graphique de distribution avec estimation de densité par noyau
            sns.histplot(oos_df['return'].values, kde=True, stat="density", 
                       color='skyblue', bins=min(10, len(oos_df)))
            
            # Marquer les statistiques importantes
            mean_return = oos_df['return'].mean()
            median_return = oos_df['return'].median()
            std_return = oos_df['return'].std()
            
            # Ajouter des lignes verticales pour la moyenne, la médiane
            plt.axvline(mean_return, color='red', linestyle='dashed', linewidth=2, 
                      label=f'Moyenne: {mean_return:.2f}%')
            plt.axvline(median_return, color='green', linestyle='dashed', linewidth=2, 
                       label=f'Médiane: {median_return:.2f}%')
            
            # Ajouter des lignes verticales pour moyenne ± 1 écart-type (intervalle de confiance 68%)
            plt.axvline(mean_return + std_return, color='purple', linestyle='dotted', 
                      label=f'Moyenne + 1σ: {mean_return + std_return:.2f}%')
            plt.axvline(mean_return - std_return, color='purple', linestyle='dotted', 
                      label=f'Moyenne - 1σ: {mean_return - std_return:.2f}%')
            
            # Ajouter une ligne horizontale à y=0 pour souligner les rendements négatifs
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Calculer les métriques clés pour le texte
            negative_returns = (oos_df['return'] < 0).mean() * 100
            positive_returns = (oos_df['return'] > 0).mean() * 100
            skewness = oos_df['return'].skew()
            kurtosis = oos_df['return'].kurtosis()
            
            # Ajouter une zone de texte avec les métriques
            metrics_text = (
                f"Métriques de Distribution:\n"
                f"Rendement Moyen: {mean_return:.2f}%\n"
                f"Rendement Médian: {median_return:.2f}%\n"
                f"Écart-Type: {std_return:.2f}%\n"
                f"Fenêtres Négatives: {negative_returns:.1f}%\n"
                f"Fenêtres Positives: {positive_returns:.1f}%\n"
                f"Asymétrie: {skewness:.2f}\n"
                f"Kurtosis: {kurtosis:.2f}"
            )
            
            # Ajouter la zone de texte des métriques
            plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                        backgroundcolor='white', alpha=0.8,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                        verticalalignment='top')
            
            # Personnaliser le graphique
            plt.title('Distribution des Rendements Out-of-Sample', fontsize=16)
            plt.xlabel('Rendement (%)', fontsize=14)
            plt.ylabel('Densité', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper right')
            
            save_figure_to_pdf(pdf)
            
        # =====================================================================
        # ANALYSE DE ROBUSTESSE DE LA STRATÉGIE
        # =====================================================================
        
        # Calculer les métriques de robustesse
        robustness_metrics = {}
        
        # 1. Cohérence de Performance: % de périodes OOS positives
        if 'return' in oos_df.columns:
            positive_periods = (oos_df['return'] > 0).mean() * 100
            robustness_metrics['Périodes Positives (%)'] = positive_periods
        
        # 2. Stabilité de Performance: coefficient de variation des rendements
        if 'return' in oos_df.columns and not oos_df['return'].isna().all():
            mean_return = oos_df['return'].mean()
            if abs(mean_return) > 1e-6:  # Éviter division par zéro ou nombres minuscules
                cv_return = oos_df['return'].std() / abs(mean_return)
                # Convertir en stabilité (1 - CV normalisé)
                # Limiter à la plage [0, 1] en utilisant 1 / (1 + CV)
                return_stability = 1 / (1 + cv_return)
                robustness_metrics['Stabilité des Rendements (0-1)'] = return_stability
        
        # 3. Cohérence de Sharpe: coefficient de variation des ratios de Sharpe
        if 'sharpe' in oos_df.columns and not oos_df['sharpe'].isna().all():
            # Seulement pour la moyenne de Sharpe positive
            mean_sharpe = oos_df['sharpe'].mean()
            if mean_sharpe > 1e-6:
                cv_sharpe = oos_df['sharpe'].std() / mean_sharpe
                sharpe_stability = 1 / (1 + cv_sharpe)
                robustness_metrics['Stabilité de Sharpe (0-1)'] = sharpe_stability
        
        # 4. Cohérence du Taux de Réussite
        if 'win_rate' in oos_df.columns and not oos_df['win_rate'].isna().all():
            mean_win_rate = oos_df['win_rate'].mean()
            if mean_win_rate > 1e-6:
                cv_win_rate = oos_df['win_rate'].std() / mean_win_rate
                win_rate_stability = 1 / (1 + cv_win_rate)
                robustness_metrics['Stabilité du Taux de Réussite (0-1)'] = win_rate_stability
        
        # 5. Stabilité des Paramètres: moyenne de 1 - CV pour chaque paramètre
        param_stability = {}
        numeric_params = params_df.select_dtypes(include=['number']).columns.tolist()
        
        for param in numeric_params:
            mean_value = params_df[param].mean()
            if abs(mean_value) > 1e-6:  # Éviter division par zéro
                cv = params_df[param].std() / abs(mean_value)
                stability = 1 / (1 + cv)
                param_stability[param] = stability
        
        if param_stability:
            robustness_metrics['Stabilité Moy. des Paramètres (0-1)'] = np.mean(list(param_stability.values()))
            # Stocker également la stabilité individuelle des paramètres
            for param, stability in param_stability.items():
                robustness_metrics[f'Stabilité {param}'] = stability
        
        # 6. Ratio Performance OOS vs IS
        # Si nous avons les métriques IS et OOS, calculer le ratio
        is_metrics = []
        for window_result in wfo_results['window_results']:
            if window_result['optimization_results'] and len(window_result['optimization_results']) > 0:
                best_result = window_result['optimization_results'][0]
                if 'combined_score' in best_result:
                    is_metrics.append(best_result['combined_score'])
        
        if is_metrics and 'return' in oos_df.columns:
            is_avg = np.mean(is_metrics)
            oos_avg = oos_df['return'].mean()
            
            if abs(is_avg) > 1e-6:  # Éviter division par zéro
                oos_is_ratio = oos_avg / is_avg
                # Normaliser à l'échelle 0-1: 1 signifie OOS = IS (parfait), 0 signifie complètement différent
                oos_is_consistency = 1 - min(1, abs(1 - oos_is_ratio))
                robustness_metrics['Cohérence OOS/IS (0-1)'] = oos_is_consistency
        
        # 7. Calculer le score de robustesse global (moyenne de toutes les métriques)
        # Filtrer pour inclure uniquement les métriques à l'échelle 0-1
        scaled_metrics = {k: v for k, v in robustness_metrics.items() if '(0-1)' in k}
        if scaled_metrics:
            robustness_metrics['Score de Robustesse Global (0-1)'] = np.mean(list(scaled_metrics.values()))
        
        # Créer une page résumant les métriques de robustesse
        if robustness_metrics:
            plt.figure(figsize=(14, 10))
            plt.axis('off')
            
            plt.text(0.5, 0.95, "Analyse de Robustesse de la Stratégie", fontsize=14, ha='center')
            
            # Filtrer les métriques pour le graphique radar
            radar_metrics = {k.replace(' (0-1)', ''): v for k, v in robustness_metrics.items() if '(0-1)' in k}
            
            # Si nous avons suffisamment de métriques pour un graphique radar
            if len(radar_metrics) >= 3:
                # Créer un graphique radar (spider plot) pour les métriques de robustesse
                ax1 = plt.subplot2grid((2, 2), (0, 0), polar=True)
                
                # Préparer les données pour le graphique radar
                categories = list(radar_metrics.keys())
                values = list(radar_metrics.values())
                
                # Fermer le tracé en ajoutant la première valeur à la fin
                categories = categories + [categories[0]]
                values = values + [values[0]]
                
                # Calculer l'angle pour chaque catégorie
                N = len(categories) - 1  # Excluant le premier élément répété
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # Fermer la boucle
                
                # Tracer le graphique
                ax1.plot(angles, values, marker='o', linestyle='-', linewidth=2, label='Métriques de Robustesse')
                
                # Remplir la zone
                ax1.fill(angles, values, alpha=0.25)
                
                # Définir les étiquettes de catégorie
                ax1.set_xticks(angles[:-1])
                ax1.set_xticklabels(categories[:-1])
                
                # Définir les limites radiales
                ax1.set_ylim(0, 1)
                
                # Ajouter des lignes de grille radiales à 0.2, 0.4, 0.6, 0.8
                plt.yticks([0.2, 0.4, 0.6, 0.8], ['0.2', '0.4', '0.6', '0.8'], color='grey', size=8)
                
                # Dessiner des cercles d'axe y
                for ytick in [0.2, 0.4, 0.6, 0.8]:
                    ax1.add_artist(plt.Circle((0, 0), ytick, fill=False, color='grey', linestyle='--', alpha=0.4))
                
                # Titre pour ce sous-graphique
                ax1.set_title('Métriques de Robustesse de la Stratégie (1.0 = Idéal)', fontsize=14, pad=20)
            
            # Créer un graphique à barres avec toutes les métriques de robustesse
            ax2 = plt.subplot2grid((2, 2), (0, 1))
            
            metrics_to_plot = {k: v for k, v in robustness_metrics.items() 
                             if not k.startswith('Stabilité Moy.') and '(0-1)' in k}
            
            # Trier les métriques par valeur
            sorted_metrics = dict(sorted(metrics_to_plot.items(), key=lambda item: item[1], reverse=True))
            
            # Tracer le graphique à barres
            if sorted_metrics:
                bars = ax2.barh(
                    [k.replace(' (0-1)', '') for k in sorted_metrics.keys()], 
                    list(sorted_metrics.values()),
                    color='skyblue'
                )
                
                # Ajouter des étiquettes de valeur
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    label_x_pos = width + 0.01
                    ax2.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                           f'{width:.2f}', va='center')
                
                ax2.set_xlim(0, 1.1)
                ax2.set_xlabel('Score (échelle 0-1)', fontsize=12)
                ax2.set_ylabel('Métrique', fontsize=12)
                ax2.set_title('Comparaison des Métriques de Robustesse', fontsize=14)
                ax2.grid(True, axis='x', alpha=0.3)
            
            # Créer un graphique à barres de stabilité des paramètres
            param_stability_items = {k: v for k, v in robustness_metrics.items() 
                                   if 'Stabilité' in k and not k.startswith('Stabilité Moy.') and '(0-1)' not in k}
            
            if param_stability_items:
                ax3 = plt.subplot2grid((2, 2), (1, 0))
                
                # Trier les paramètres par stabilité
                sorted_param_stability = dict(sorted(param_stability_items.items(), 
                                                   key=lambda item: item[1], reverse=True))
                
                # Tracer un graphique à barres
                bars = ax3.barh(
                    list(sorted_param_stability.keys()), 
                    list(sorted_param_stability.values()),
                    color='lightgreen'
                )
                
                # Ajouter des étiquettes de valeur
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    label_x_pos = width + 0.01
                    ax3.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                           f'{width:.2f}', va='center')
                
                ax3.set_xlim(0, 1.1)
                ax3.set_xlabel('Score de Stabilité (échelle 0-1)', fontsize=12)
                ax3.set_ylabel('Paramètre', fontsize=12)
                ax3.set_title('Analyse de Stabilité des Paramètres', fontsize=14)
                ax3.grid(True, axis='x', alpha=0.3)
            
            # Créer un résumé textuel de l'analyse de robustesse
            ax4 = plt.subplot2grid((2, 2), (1, 1))
            ax4.axis('off')  # Désactiver l'axe
            
            # Préparer le texte de résumé
            summary_text = "Résumé de l'Analyse de Robustesse:\n\n"
            
            if 'Score de Robustesse Global (0-1)' in robustness_metrics:
                score = robustness_metrics['Score de Robustesse Global (0-1)']
                summary_text += f"Robustesse Globale: {score:.2f}/1.00\n\n"
                
                # Ajouter une interprétation
                if score >= 0.8:
                    summary_text += "Interprétation: Excellente robustesse - très cohérente sur toutes les métriques.\n"
                elif score >= 0.6:
                    summary_text += "Interprétation: Bonne robustesse - performance cohérente avec des variations mineures.\n"
                elif score >= 0.4:
                    summary_text += "Interprétation: Robustesse modérée - quelques incohérences mais généralement acceptable.\n"
                elif score >= 0.2:
                    summary_text += "Interprétation: Faible robustesse - incohérences significatives entre les métriques.\n"
                else:
                    summary_text += "Interprétation: Mauvaise robustesse - performance extrêmement incohérente.\n"
            
            # Ajouter des détails sur les périodes positives
            if 'Périodes Positives (%)' in robustness_metrics:
                pos_periods = robustness_metrics['Périodes Positives (%)']
                summary_text += f"\nPériodes positives: {pos_periods:.1f}% des fenêtres out-of-sample\n"
            
            # Ajouter une comparaison OOS/IS si disponible
            if 'Cohérence OOS/IS (0-1)' in robustness_metrics:
                oos_is = robustness_metrics['Cohérence OOS/IS (0-1)']
                summary_text += f"Cohérence OOS/IS: {oos_is:.2f}/1.00\n"
                
                if oos_is >= 0.8:
                    summary_text += "    (Très faible baisse de performance de in-sample à out-of-sample)\n"
                elif oos_is >= 0.5:
                    summary_text += "    (Baisse de performance modérée de in-sample à out-of-sample)\n"
                else:
                    summary_text += "    (Baisse de performance significative de in-sample à out-of-sample)\n"
            
            # Ajouter des informations sur la stabilité des paramètres
            if 'Stabilité Moy. des Paramètres (0-1)' in robustness_metrics:
                param_stab = robustness_metrics['Stabilité Moy. des Paramètres (0-1)']
                summary_text += f"\nStabilité des Paramètres: {param_stab:.2f}/1.00\n"
                
                if param_stab >= 0.8:
                    summary_text += "    (Paramètres très stables à travers les fenêtres - stratégie robuste)\n"
                elif param_stab >= 0.5:
                    summary_text += "    (Paramètres modérément stables - robustesse acceptable)\n"
                else:
                    summary_text += "    (Paramètres instables - peut indiquer un surajustement)\n"
            
            # Ajouter le texte au graphique
            ax4.text(0, 1.0, summary_text, fontsize=11, va='top', linespacing=1.5)
            
            save_figure_to_pdf(pdf)
            
        # =====================================================================
        # DERNIÈRE PAGE - CONCLUSIONS
        # =====================================================================
        
        plt.figure(figsize=(14, 10))
        plt.axis('off')
        
        plt.text(0.5, 0.95, "Conclusions et Recommandations", fontsize=24, ha='center')
        
        # Préparer le texte de conclusion
        conclusion_text = ""
        
        # Juger de la qualité globale de la stratégie
        if 'return' in oos_df.columns and 'sharpe' in oos_df.columns:
            avg_return = oos_df['return'].mean()
            avg_sharpe = oos_df['sharpe'].mean()
            
            conclusion_text += f"Performance Globale:\n"
            
            if avg_return > 0 and avg_sharpe > 1:
                conclusion_text += "✅ La stratégie a démontré une performance positive dans l'ensemble avec un rendement moyen de "
                conclusion_text += f"{avg_return:.2f}% et un ratio de Sharpe moyen de {avg_sharpe:.2f}.\n\n"
            elif avg_return > 0:
                conclusion_text += "⚠️ La stratégie a montré un rendement positif moyen de "
                conclusion_text += f"{avg_return:.2f}%, mais avec un ratio de Sharpe moyen de seulement {avg_sharpe:.2f}.\n\n"
            else:
                conclusion_text += "❌ La stratégie n'a pas démontré une performance positive, avec un rendement moyen de "
                conclusion_text += f"{avg_return:.2f}% et un ratio de Sharpe moyen de {avg_sharpe:.2f}.\n\n"
        
        # Évaluer la robustesse
        if 'Score de Robustesse Global (0-1)' in robustness_metrics:
            robustness_score = robustness_metrics['Score de Robustesse Global (0-1)']
            
            conclusion_text += f"Robustesse de la Stratégie:\n"
            
            if robustness_score >= 0.7:
                conclusion_text += "✅ La stratégie démontre une robustesse élevée avec un score de "
                conclusion_text += f"{robustness_score:.2f}/1.00, ce qui suggère qu'elle est susceptible de bien se comporter dans des conditions de marché futures.\n\n"
            elif robustness_score >= 0.4:
                conclusion_text += "⚠️ La stratégie présente une robustesse modérée avec un score de "
                conclusion_text += f"{robustness_score:.2f}/1.00. Une certaine prudence est recommandée lors de son déploiement en temps réel.\n\n"
            else:
                conclusion_text += "❌ La stratégie ne démontre pas une robustesse suffisante, avec un score de seulement "
                conclusion_text += f"{robustness_score:.2f}/1.00. Un travail supplémentaire est nécessaire pour améliorer sa fiabilité.\n\n"
        
        # Évaluer la stabilité des paramètres
        if 'Stabilité Moy. des Paramètres (0-1)' in robustness_metrics:
            param_stability = robustness_metrics['Stabilité Moy. des Paramètres (0-1)']
            
            conclusion_text += f"Paramètres Optimaux:\n"
            
            if param_stability >= 0.7:
                conclusion_text += "✅ Les paramètres optimaux sont restés hautement cohérents à travers les fenêtres "
                conclusion_text += f"(stabilité: {param_stability:.2f}/1.00), ce qui suggère que la stratégie n'est pas suroptimisée.\n\n"
            elif param_stability >= 0.4:
                conclusion_text += "⚠️ Une certaine variation dans les paramètres optimaux est observée "
                conclusion_text += f"(stabilité: {param_stability:.2f}/1.00). Envisagez d'utiliser les valeurs moyennes des paramètres.\n\n"
            else:
                conclusion_text += "❌ Les paramètres optimaux varient significativement entre les fenêtres "
                conclusion_text += f"(stabilité: {param_stability:.2f}/1.00), ce qui suggère un possible surajustement. "
                conclusion_text += "Une approche plus robuste de sélection des paramètres est recommandée.\n\n"
        
        # Recommandations finales
        conclusion_text += "Recommandations:\n\n"
        
        # Décider des recommandations basées sur les métriques
        if not oos_df.empty and 'return' in oos_df.columns and 'sharpe' in oos_df.columns:
            avg_return = oos_df['return'].mean()
            avg_sharpe = oos_df['sharpe'].mean()
            positive_rate = (oos_df['return'] > 0).mean()
            
            # Construire les recommandations
            recommendations = []
            
            if avg_return > 0 and avg_sharpe > 1 and positive_rate >= 0.6:
                recommendations.append("✅ La stratégie peut être considérée pour un déploiement en production avec les paramètres optimaux agrégés.")
            elif avg_return > 0 and avg_sharpe > 0.5:
                recommendations.append("⚠️ La stratégie pourrait être utilisée avec prudence, potentiellement avec une taille de position réduite.")
            else:
                recommendations.append("❌ Il est recommandé d'améliorer davantage la stratégie avant de la déployer en production.")
            
            # Recommandations sur les paramètres
            if 'Stabilité Moy. des Paramètres (0-1)' in robustness_metrics:
                param_stability = robustness_metrics['Stabilité Moy. des Paramètres (0-1)']
                if param_stability < 0.5:
                    recommendations.append("⚠️ Étant donné la variabilité des paramètres optimaux, envisagez d'utiliser une stratégie d'adaptation dynamique des paramètres.")
            
            # Recommandations sur le risque
            if 'max_drawdown' in oos_df.columns:
                avg_drawdown = oos_df['max_drawdown'].mean()
                if avg_drawdown > 20:
                    recommendations.append(f"⚠️ Le drawdown moyen de {avg_drawdown:.2f}% est élevé. Envisagez d'ajouter des mécanismes de gestion des risques.")
            
            # Ajouter toutes les recommandations au texte de conclusion
            for recommendation in recommendations:
                conclusion_text += f"• {recommendation}\n"
        else:
            conclusion_text += "• Données insuffisantes pour formuler des recommandations spécifiques."
        
        # Ajouter une note finale
        conclusion_text += "\nNote Finale:\n"
        conclusion_text += "L'analyse Walk-Forward Optimization fournit un aperçu précieux de la performance attendue en conditions réelles, "
        conclusion_text += "mais ne garantit pas les résultats futurs. Surveillez constamment la performance de la stratégie et ajustez si nécessaire."
        
        # Ajouter le texte de conclusion au graphique
        plt.text(0.1, 0.85, conclusion_text, fontsize=12, va='top', linespacing=1.8)
        
        save_figure_to_pdf(pdf)
    
    print(f"Rapport PDF généré avec succès: {output_path}")
    return output_path

def integrate_report_generation(wfo_results, df, strategy_name="ATDMF Strategy"):
    """
    Fonction wrapper pour intégrer la génération de rapport dans le workflow principal.
    
    Parameters:
    -----------
    wfo_results : dict
        Résultats du walk_forward_optimization
    df : pandas.DataFrame
        DataFrame OHLCV original
    strategy_name : str, optional
        Nom de la stratégie pour le titre du rapport
        
    Returns:
    --------
    None
    """
    # Demander à l'utilisateur s'il souhaite générer un rapport PDF
    generate_report = input("\nGénérer un rapport PDF complet? (o/n) [default: o]: ").lower()
    
    if generate_report != 'n':
        print("\nGénération du rapport PDF en cours...")
        try:
            report_path = generate_wfo_report_pdf(wfo_results, df, strategy_name=strategy_name)
            print(f"Rapport généré avec succès: {report_path}")
            
            # Option pour ouvrir automatiquement le PDF
            open_report = input("Ouvrir le rapport PDF maintenant? (o/n) [default: o]: ").lower()
            if open_report != 'n':
                
                if platform.system() == 'Darwin':  # macOS
                    subprocess.call(('open', report_path))
                elif platform.system() == 'Windows':  # Windows
                    os.startfile(report_path)
                else:  # Linux
                    subprocess.call(('xdg-open', report_path))
                    
        except Exception as e:
            print(f"Erreur lors de la génération du rapport PDF: {e}")
            print("Vérifiez que vous avez installé toutes les dépendances nécessaires:")
            print("  - matplotlib")
            print("  - pandas")
            print("  - numpy")
            print("  - seaborn")
            print("  - plotly")
            print("  - Pillow (PIL)")
    else:
        print("Génération du rapport PDF ignorée.")

# ======================================================================
# MAIN PROGRAM
# ======================================================================

def get_param_grid():
    """
    Get parameter grid from user input.
    
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
        'timeperiod': (10, 30, 1),
        'StDev': (0.5, 2.5, 0.1),
        'coeff_medianeBBW': (0.8, 1.6, 0.1),
        'coef_mediane': (0.5, 1.5, 0.1),
        'fenetre_lowest': (30, 50, 5),
        'seuil_lowest': (1, 3.5, 0.2),
        'longueur_mediane': (50, 150, 10),
        'Nb_bars_above': (2, 10, 1),
        'user_exit_sma_length': (10, 30, 2)
    }
    
    # Get user input for parameters to optimize
    print("\nSelect parameters to optimize:")
    for i, (param, desc) in enumerate(param_options.items(), 1):
        print(f"{i}. {param} - {desc}")
        # Display default values
        min_val, max_val, step = default_ranges[param]
        print(f"   Default range: {min_val} to {max_val}, step: {step}")
    
    selected = input("Enter parameter numbers separated by commas (e.g., 1,2): ")
    selected_indices = [int(idx) for idx in selected.split(',') if idx.strip().isdigit()]
    
    if not selected_indices:
        print("No valid parameters selected. Using timeperiod and StDev as defaults.")
        selected_params = ['timeperiod', 'StDev','coeff_medianeBBW', 'coef_mediane', 'fenetre_lowest', 'seuil_lowest', 'longueur_mediane', 'Nb_bars_above', 'user_exit_sma_length']
    else:
        param_keys = list(param_options.keys())
        selected_params = [param_keys[idx-1] for idx in selected_indices if 1 <= idx <= len(param_keys)]
    
    # Build parameter grid
    param_grid = {}
    for param in selected_params:
        if param in default_ranges:
            default_min, default_max, default_step = default_ranges[param]
            
            # Allow user to customize ranges
            print(f"\nCustomize range for {param}:")
            try:
                min_val = float(input(f"Minimum value [default: {default_min}]: ") or default_min)
                max_val = float(input(f"Maximum value [default: {default_max}]: ") or default_max)
                step = float(input(f"Step size [default: {default_step}]: ") or default_step)
            except ValueError:
                print(f"Invalid input. Using default values for {param}.")
                min_val, max_val, step = default_min, default_max, default_step
            
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

def get_metrics_info():
    """
    Get metrics information from user input.
    
    Returns:
    --------
    dict
        Metrics information for optimization
    """
    # Define available metrics
    available_metrics = ['max_drawdown', 'sharpe_ratio', 'total_return', 'avg_gain_per_trade', 'avg_loss_per_trade', 'win_rate','avg_pl_per_trade']
    
    print("\nSelect metrics to optimize:")
    for i, metric in enumerate(available_metrics, 1):
        print(f"{i}. {metric}")
    
    try:
        metric1_idx = int(input("Enter number for first metric: ")) - 1
        metric2_idx = int(input("Enter number for second metric: ")) - 1
        
        if not (0 <= metric1_idx < len(available_metrics) and 0 <= metric2_idx < len(available_metrics)):
            raise ValueError
            
        metric1_name = available_metrics[metric1_idx]
        metric2_name = available_metrics[metric2_idx]
        
        weight_metric1 = float(input(f"Enter weight for {metric1_name} (0-1): "))
        weight_metric1 = max(0, min(1, weight_metric1))  # Clamp between 0 and 1
        weight_metric2 = 1 - weight_metric1
    except (ValueError, IndexError):
        print("Invalid metrics selected. Using total_return and max_drawdown as defaults.")
        metric1_name = 'total_return'
        metric2_name = 'max_drawdown'
        weight_metric1 = 1.0
        weight_metric2 = 0.0
    
    return {
        'metric1_name': metric1_name,
        'metric2_name': metric2_name,
        'weight_metric1': weight_metric1,
        'weight_metric2': weight_metric2
    }

def get_wfo_settings():
    """
    Get Walk-Forward Optimization settings from user input.
    
    Returns:
    --------
    WFOSettings
        WFO settings object
    """
    settings = WFOSettings()
    
    print("\nConfigure Walk-Forward Optimization settings:")
    
    # Number of windows
    try:
        settings.n_windows = int(input("Number of windows to divide data into [default: 1]: ") or 1)
    except ValueError:
        settings.n_windows = 1
    
    # Training size
    try:
        settings.train_size = float(input("Training size as proportion (0-1) [default: 0.5]: ") or 0.5)
        settings.train_size = max(0.1, min(0.9, settings.train_size))  # Clamp between 0.1 and 0.9
    except ValueError:
        settings.train_size = 0.5
    
    # Anchored or unanchored
    anchored_input = input("Use anchored WFO (fixed start date) or unanchored (rolling window)? (a/u) [default: u]: ").lower()
    settings.anchored = anchored_input == 'a'
    
    # Parallelization backend
    print("\nSelect parallelization backend:")
    print("1. threadpool - Best for Numba functions")
    print("2. dask - Best for CPU-bound tasks")
    print("3. ray - Best for distributed computing")
    print("4. pathos - Alternative for multiprocessing")
    
    backend_choice = input("Enter choice (1-4) [default: 1]: ") or "1"
    backends = {
        "1": "threadpool",
        "2": "dask",
        "3": "ray",
        "4": "pathos"
    }
    settings.parallel_backend = backends.get(backend_choice, "threadpool")
    
    # Numba acceleration
    numba_choice = input("Use Numba acceleration? (y/n) [default: y]: ").lower()
    settings.use_numba = numba_choice != 'n'
    
    return settings

def main():
    """Main function to run the WFO process."""
    print("=== ATDMF Strategy Walk-Forward Optimization ===")
    print("This script performs Walk-Forward Optimization on the ATDMF strategy.")
    print("It demonstrates how to properly cross-validate trading strategies to avoid overfitting.")
    
    # Get dates
    start_date, end_date = get_dates()
    
    # Load data
    print("\nLoading data...")
    timeframe = '5S'
    
    # Ask user whether to load from file or fetch from Binance
    from_file = input("\nLoad data from file? (y/n) [default: y]: ").lower() != 'n'
    
    if from_file:
        file_path = input("Enter path to CSV file [default: ./Data/latest/data.csv]: ") or "./Data/latest/data.csv"
        df = load_data(start_date, end_date, timeframe, from_file=True, file_path=file_path)
    else:
        df = load_data(start_date, end_date, timeframe, from_file=False)
    
    print(f"Loaded {len(df)} bars of data from {start_date} to {end_date}")
    
    # Display default parameters
    display_default_parameters()

    # Get parameter grid
    param_grid = get_param_grid()
    print(f"Parameter grid: {param_grid}")
    
    # Get metrics information
    metrics_info = get_metrics_info()
    print(f"Metrics: {metrics_info['metric1_name']} (weight: {metrics_info['weight_metric1']:.2f}), "
          f"{metrics_info['metric2_name']} (weight: {metrics_info['weight_metric2']:.2f})")
    
    # Get WFO settings
    settings = get_wfo_settings()
    
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
    visualize = input("\nVisualize results? (y/n) [default: y]: ").lower() != 'n'
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
    run_final = input("\nRun a final backtest with averaged best parameters? (y/n): ")
    if run_final.lower() == 'y':
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

if __name__ == "__main__":
    main()

