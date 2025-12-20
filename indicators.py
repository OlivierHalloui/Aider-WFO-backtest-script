# Import necessary libraries for indicators
import numpy as np
from numba import njit, prange
import vectorbtpro as vbt

# ======================================================================
# NUMBA-OPTIMIZED INDICATOR FUNCTIONS
# ======================================================================

@njit(cache=True)
def rolling_mean(arr, window):
    result = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        if i >= window - 1:
            result[i] = np.mean(arr[i - window + 1:i + 1])
    return result

@njit(cache=True)
def rolling_median(arr, window):
    result = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        if i >= window - 1:
            result[i] = np.median(arr[i - window + 1:i + 1])
    return result

@njit(cache=True)
def rolling_min(arr, window):
    result = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        if i >= window - 1:
            result[i] = np.min(arr[i - window + 1:i + 1])
    return result

@njit(cache=True)
def compute_bars_since_below(ecart_borne1, mediane, coef, Nb_bars_above):
    bars_since_below = np.empty(len(ecart_borne1))
    counter = 1000000 # Large number instead of inf for int array safety if needed
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

@njit(cache=True)
def ecart_bollinger_borne_signal_nb(prix, upper_band, lower_band, timeperiod, longueur_mediane, coef_mediane, Nb_bars_above):
    ecart = upper_band - lower_band
    # Note: using vbt.indicators.nb.ma_1d_nb directly inside
    sma = vbt.indicators.nb.ma_1d_nb(prix, timeperiod)
    ecart_borne1 = ecart / sma
    mediane = rolling_median(ecart_borne1, longueur_mediane)
    return compute_bars_since_below(ecart_borne1, mediane, coef_mediane, Nb_bars_above)

@njit(cache=True)
def bollinger_horizontal_signal_nb(upper_band, lower_band, middle_band, coeff_medianeBBW):
    BBW = (upper_band - lower_band) / middle_band
    MMBBW = rolling_mean(BBW, 5)
    medianeBBW = rolling_median(BBW, 200)
    seuil = medianeBBW / coeff_medianeBBW
    signal = np.zeros(len(BBW), dtype=np.bool_)
    for i in range(len(BBW)):
        if np.isnan(seuil[i]):
            continue
        if BBW[i] < seuil[i] or MMBBW[i] < seuil[i]:
            signal[i] = True
    return signal

@njit(cache=True)
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
    """
    n = len(close)
    sma = vbt.indicators.nb.ma_1d_nb(close, user_exit_sma_length)
    
    signal = np.zeros(n, dtype=np.bool_)
    for i in range(1, n):
        if (not np.isnan(close[i-1]) and not np.isnan(sma[i-1]) and 
            not np.isnan(close[i]) and not np.isnan(sma[i]) and 
            close[i-1] > sma[i-1] and close[i] < sma[i]):
            signal[i] = True   
    return signal

# ======================================================================
# INDICATOR FACTORIES (Vectorized)
# ======================================================================

# Using with_apply_func to allow VBT to handle broadcasting of parameters automatically.
# We explicitly set input names to match what the strategy will pass.

EcartBollingerBorne = vbt.IF(
    class_name='EcartBollingerBorne',
    input_names=['prix', 'upper_band', 'lower_band'],
    param_names=['timeperiod', 'longueur_mediane', 'coef_mediane', 'Nb_bars_above'],
    output_names=['signal']
).with_apply_func(
    ecart_bollinger_borne_signal_nb,
    takes_1d=True, 
    timeperiod=20, 
    longueur_mediane=100, 
    coef_mediane=1.0, 
    Nb_bars_above=5
)

BollingerHorizontal = vbt.IF(
    class_name='BollingerHorizontal',
    input_names=['upper_band', 'lower_band', 'middle_band'],
    param_names=['coeff_medianeBBW'],
    output_names=['signal']
).with_apply_func(
    bollinger_horizontal_signal_nb,
    takes_1d=True,
    coeff_medianeBBW=1.1
)

CrossBBWLowSignal = vbt.IF(
    class_name='CrossBBWLowSignal',
    input_names=['upper_band', 'lower_band', 'middle_band'],
    param_names=['fenetre_lowest', 'seuil_lowest'],
    output_names=['signal']
).with_apply_func(
    cross_bbw_low_signal_nb,
    takes_1d=True,
    fenetre_lowest=30, 
    seuil_lowest=3.5
)

SMAExit = vbt.IF(
    class_name='SMAExit',
    input_names=['close'],
    param_names=['user_exit_sma_length'],
    output_names=['signal']
).with_apply_func(
    calculate_exit_sma_nb,
    takes_1d=True,
    user_exit_sma_length=20
)