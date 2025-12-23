# strategy_v2.py

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from vectorbtpro.indicators import talib
from numba import njit

# Define a constant for the warmup period. This should be the longest lookback
# period required by any indicator in the strategy.
# The `mediane_bbw` uses a rolling window of 200, which is the largest.
# A safe buffer is added.
WARMUP_PERIOD = 250

# ======================================================================
# 1. Custom Numba Functions for Speed
# ======================================================================

@njit
def ecart_bollinger_borne_signal_nb(prix, upper_band, lower_band, timeperiod, longueur_mediane, coef_mediane, Nb_bars_above):
    mediane_prix = np.full_like(prix, np.nan)
    for i in range(longueur_mediane, len(prix)):
        mediane_prix[i] = np.median(prix[i-longueur_mediane:i])

    max_ecart = np.abs(upper_band - lower_band)
    mediane_ecart = np.full_like(max_ecart, np.nan)
    for i in range(longueur_mediane, len(max_ecart)):
        mediane_ecart[i] = np.median(max_ecart[i-longueur_mediane:i])

    bars_since_below = np.zeros_like(prix)
    for i in range(1, len(prix)):
        if prix[i-1] < mediane_prix[i-1]:
            bars_since_below[i] = 0
        else:
            bars_since_below[i] = bars_since_below[i-1] + 1
            
    signal = (max_ecart > mediane_ecart * coef_mediane) & (bars_since_below > Nb_bars_above)
    return signal

@njit
def cross_bbw_low_signal_nb(upper_band, lower_band, middle_band, fenetre_lowest, seuil_lowest):
    bbw = (upper_band - lower_band) / middle_band
    lowest_bbw = np.full_like(bbw, np.nan)
    for i in range(fenetre_lowest, len(bbw)):
        lowest_bbw[i] = np.min(bbw[i-fenetre_lowest:i])
    
    signal = bbw < lowest_bbw * seuil_lowest
    return signal

@njit
def bollinger_horizontal_signal_nb(bbw, mmbbw, mediane_bbw, coeff_medianeBBW):
    return (bbw < mediane_bbw * coeff_medianeBBW) & (mmbbw < mediane_bbw)

def np_shift(arr, n=1):
    """
    Shifts a numpy array or pandas object by n.
    Fills with NaN.
    """
    if hasattr(arr, 'shift'):
        return arr.shift(n)
    
    # Numpy array handling
    out = np.empty_like(arr)
    if n > 0:
        out[:n] = np.nan
        out[n:] = arr[:-n]
    elif n < 0:
        out[n:] = np.nan
        out[:n] = arr[-n:]
    else:
        out[:] = arr
    return out

# ======================================================================
# 2. Indicator Factory Definition
# ======================================================================

def atdmf_apply_func(close, high, low, open, timeperiod, StDev, matype, coeff_medianeBBW, coef_mediane, Nb_bars_above, fenetre_lowest, seuil_lowest, longueur_mediane, user_exit_sma_length, **kwargs):
    """The core logic for the ATDMF strategy, designed to be used with vbt.IndicatorFactory."""
    # --- Indicator Calculations ---
    
    # 1. Bollinger Bands
    bbands = talib('BBANDS').run(
        close, timeperiod=timeperiod, nbdevup=StDev, nbdevdn=StDev, matype=matype
    )
    
    # 2. Custom Bollinger signals
    bbw = (bbands.upperband - bbands.lowerband) / bbands.middleband
    mmbbw = bbw.rolling(5).mean()
    mediane_bbw = bbw.rolling(200).median()

    # --- Signal Generation using Numba functions ---
    
    # Crossover signal
    prev_close = np_shift(close, 1)
    prev_upper = np_shift(bbands.upperband, 1)
    crossover_signal = (close > bbands.upperband) & (prev_close < prev_upper)

    # Final Entry Signal
    entries = (
        ecart_bollinger_borne_signal_nb(
            close, 
            bbands.upperband.values, 
            bbands.lowerband.values, 
            timeperiod, 
            longueur_mediane, coef_mediane, Nb_bars_above
        ) &
        cross_bbw_low_signal_nb(
            bbands.upperband.values, 
            bbands.lowerband.values, 
            bbands.middleband.values, 
            fenetre_lowest, seuil_lowest
        ) &
        bollinger_horizontal_signal_nb(
            bbw.values, 
            mmbbw.values, 
            mediane_bbw.values, 
            coeff_medianeBBW
        ) &
        crossover_signal.values
    )

    # Final Exit Signal
    sma_exit = talib('SMA').run(close, timeperiod=user_exit_sma_length)
    exits = close < sma_exit.real

    return entries, exits

# Encapsulate the entire strategy logic into a VectorBT Pro IndicatorFactory.
# This makes it reusable, easily pluggable into other vbt components, and handles
# broadcasting of parameters automatically.
ATDMF_Factory = vbt.IndicatorFactory(
    class_name='ATDMF',
    input_names=['close', 'high', 'low', 'open'], # Future-proofing
    param_names=[
        'timeperiod', 'StDev', 'matype', 
        'coeff_medianeBBW', 'coef_mediane', 'Nb_bars_above', 
        'fenetre_lowest', 'seuil_lowest', 
        'longueur_mediane', 'user_exit_sma_length'
    ],
    output_names=['entries', 'exits']
).with_apply_func(
    atdmf_apply_func
)

# ======================================================================
# 3. Simplified Backtesting Function
# ======================================================================

def run_backtest_v2(price_data, params, freq='5s'):
    """
    Runs a backtest using the new ATDMF_Factory.
    
    This function is now a simple wrapper that:
    1. Runs the indicator factory to get entry/exit signals.
    2. Runs the portfolio simulation based on those signals.
    
    Args:
        price_data (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close'.
        params (dict): Dictionary of parameters for the strategy.
        freq (str): Frequency of the data for portfolio calculations.
    """
    # Run the factory to get entries and exits.
    # vbt automatically handles vectorized parameters (like lists or arrays).
    indicator = ATDMF_Factory.run(
        close=price_data['Close'],
        high=price_data['High'],
        low=price_data['Low'],
        open=price_data['Open'],
        **params
    )
    
    # Create the portfolio from the generated signals.
    portfolio = vbt.Portfolio.from_signals(
        close=price_data['Close'],
        entries=indicator.entries,
        exits=indicator.exits,
        freq=freq,
        init_cash=10000,
        fees=0.001
    )
    
    return portfolio
