# Import necessary libraries for indicators
import numpy as np
from numba import njit, prange
import vectorbtpro as vbt

# ======================================================================
# NUMBA-OPTIMIZED INDICATOR FUNCTIONS
# ======================================================================

# Using vbt.generic.nb optimized functions where available
# They are typically O(N) instead of O(N*W)

@njit(cache=True)
def rolling_mean(arr, window):
    # vbt.generic.nb.rolling_mean_1d_nb(a, window, minp)
    # We use min_periods=window to match previous behavior (nan until full window)
    return vbt.generic.nb.rolling_mean_1d_nb(arr, window, window)

@njit(cache=True)
def rolling_median(arr, window):
    # VectorBT Pro doesn't expose a public rolling_median_1d_nb in generic.nb yet.
    # We keep the custom implementation.
    # Standard naive implementation O(N * W * log W) or O(N * W)
    result = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        if i >= window - 1:
            # We use nanmedian to be safe, or just median if we are sure no nans in slice
            # But the slice might contain nans if input has nans.
            # vbt usually handles nans.
            window_slice = arr[i - window + 1:i + 1]
            if np.isnan(window_slice).all():
                result[i] = np.nan
            else:
                result[i] = np.nanmedian(window_slice)
    return result

@njit(cache=True)
def rolling_min(arr, window):
    # vbt.generic.nb.rolling_min_1d_nb(a, window, minp)
    return vbt.generic.nb.rolling_min_1d_nb(arr, window, window)

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
def bollinger_horizontal_signal_nb(bbw, mmbbw, mediane_bbw, coeff_medianeBBW):
    signal = np.zeros(len(bbw), dtype=np.bool_)
    for i in range(len(bbw)):
        if np.isnan(mediane_bbw[i]):
            continue
        seuil = mediane_bbw[i] / coeff_medianeBBW
        if np.isnan(seuil):
            continue
        if (not np.isnan(bbw[i]) and bbw[i] < seuil) or (not np.isnan(mmbbw[i]) and mmbbw[i] < seuil):
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

@njit(cache=True)
def ema_nb(arr, length):
    if length <= 0:
        return arr
    n = len(arr)
    out = np.empty(n)
    if n == 0:
        return out
    alpha = 2.0 / (length + 1.0)
    out[0] = arr[0]
    for i in range(1, n):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out

@njit(cache=True)
def macd_exit_signal_nb(close, fast_length, slow_length, signal_length, use_type_a, use_type_b):
    n = len(close)
    signal = np.zeros(n, dtype=np.bool_)
    if n < 2:
        return signal

    ema_fast = ema_nb(close, fast_length)
    ema_slow = ema_nb(close, slow_length)
    macd = ema_fast - ema_slow
    macd_signal = ema_nb(macd, signal_length)

    for i in range(1, n):
        cross_under = macd[i - 1] > macd_signal[i - 1] and macd[i] < macd_signal[i]
        type_a = cross_under and macd_signal[i] < macd_signal[i - 1]
        type_b = cross_under
        signal[i] = (use_type_a and type_a) or (use_type_b and type_b)

    return signal

@njit(cache=True)
def parabolic_sar_nb(high, low, sar_start, sar_increment, sar_maximum):
    n = len(high)
    sar = np.full(n, np.nan)
    if n == 0:
        return sar

    # Initialize trend based on early price action
    trend = 1
    if n > 1 and high[1] < high[0]:
        trend = -1

    if trend == 1:
        ep = high[0]
        sar[0] = low[0]
    else:
        ep = low[0]
        sar[0] = high[0]

    af = sar_start

    for i in range(1, n):
        prev_sar = sar[i - 1]
        sar_i = prev_sar + af * (ep - prev_sar)

        if trend == 1:
            if i >= 2:
                sar_i = min(sar_i, low[i - 1], low[i - 2])
            else:
                sar_i = min(sar_i, low[i - 1])

            if low[i] < sar_i:
                trend = -1
                sar_i = ep
                ep = low[i]
                af = sar_start
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + sar_increment, sar_maximum)
        else:
            if i >= 2:
                sar_i = max(sar_i, high[i - 1], high[i - 2])
            else:
                sar_i = max(sar_i, high[i - 1])

            if high[i] > sar_i:
                trend = 1
                sar_i = ep
                ep = high[i]
                af = sar_start
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + sar_increment, sar_maximum)

        sar[i] = sar_i

    return sar

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
    input_names=['bbw', 'mmbbw', 'mediane_bbw'],
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

ParabolicSAR = vbt.IF(
    class_name='ParabolicSAR',
    input_names=['high', 'low'],
    param_names=['sar_start', 'sar_increment', 'sar_maximum'],
    output_names=['sar']
).with_apply_func(
    parabolic_sar_nb,
    takes_1d=True,
    sar_start=0.02,
    sar_increment=0.02,
    sar_maximum=0.2
)

MACDExit = vbt.IF(
    class_name='MACDExit',
    input_names=['close'],
    param_names=['fast_length', 'slow_length', 'signal_length', 'use_type_a', 'use_type_b'],
    output_names=['signal']
).with_apply_func(
    macd_exit_signal_nb,
    takes_1d=True,
    fast_length=12,
    slow_length=26,
    signal_length=9,
    use_type_a=True,
    use_type_b=True
)
