# Import necessary libraries for indicators
import numpy as np
from numba import njit, prange
from vectorbtpro.indicators.factory import IndicatorFactory

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

@njit(parallel=True, cache=True)
def rolling_mean(arr, window):
    result = np.full(len(arr), np.nan)
    for i in prange(window - 1, len(arr)):
        result[i] = np.mean(arr[i - window + 1:i + 1])
    return result

@njit(parallel=True, cache=True)
def rolling_median(arr, window):
    result = np.full(len(arr), np.nan)
    for i in prange(window - 1, len(arr)):
        result[i] = np.median(arr[i - window + 1:i + 1])
    return result

@njit(parallel=True, cache=True)
def rolling_min(arr, window):
    result = np.full(len(arr), np.nan)
    for i in prange(window - 1, len(arr)):
        result[i] = np.min(arr[i - window + 1:i + 1])
    return result

@njit(parallel=True, cache=True)
def compute_bars_since_below(ecart_borne1, mediane, coef, Nb_bars_above):
    bars_since_below = np.empty(len(ecart_borne1))
    counter = np.inf
    for i in prange(len(ecart_borne1)):
        if np.isnan(mediane[i]) or np.isnan(ecart_borne1[i]):
            bars_since_below[i] = np.nan
            continue
        if ecart_borne1[i] < mediane[i] / coef:
            counter = 0
        else:
            counter += 1
        bars_since_below[i] = counter
    return bars_since_below >= Nb_bars_above

@njit(parallel=True, cache=True)
def ecart_bollinger_borne_signal_nb(prix, upper_band, lower_band, timeperiod, longueur_mediane, coef_mediane, Nb_bars_above):
    ecart = upper_band - lower_band
    sma = vbt.indicators.nb.ma_1d_nb(prix, timeperiod)
    ecart_borne1 = ecart / sma
    mediane = rolling_median(ecart_borne1, longueur_mediane)
    return compute_bars_since_below(ecart_borne1, mediane, coef_mediane, Nb_bars_above)

@njit(parallel=True, cache=True)
def bollinger_horizontal_signal_nb(upper_band, lower_band, middle_band, coeff_medianeBBW):
    BBW = (upper_band - lower_band) / middle_band
    MMBBW = rolling_mean(BBW, 5)
    medianeBBW = rolling_median(BBW, 200)
    seuil = medianeBBW / coeff_medianeBBW
    signal = np.zeros(len(BBW), dtype=np.int32)
    for i in prange(len(BBW)):
        if np.isnan(seuil[i]):
            continue
        if BBW[i] < seuil[i] or MMBBW[i] < seuil[i]:
            signal[i] = 1
    return signal

@njit(parallel=True, cache=True)
def cross_bbw_low_signal_nb(upper_band, lower_band, middle_band, fenetre_lowest, seuil_lowest):
    largeur_bb = (upper_band - lower_band) / middle_band
    bbw_lowest = rolling_min(largeur_bb, fenetre_lowest)
    signal = np.full(len(largeur_bb), False)
    for i in prange(len(largeur_bb)):
        if np.isnan(bbw_lowest[i]) or np.isnan(largeur_bb[i]):
            continue
        signal[i] = largeur_bb[i] <= (bbw_lowest[i] * seuil_lowest)
    return signal

@njit(parallel=True, cache=True)
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
    for i in prange(1, n):
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
