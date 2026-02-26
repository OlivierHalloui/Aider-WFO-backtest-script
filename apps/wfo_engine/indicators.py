# Import necessary libraries for indicators
import numpy as np
from numba import njit, prange
import vectorbtpro as vbt

# ----------------------------------------------------------------------
# VectorBT settings guard
# ----------------------------------------------------------------------
# Some environments load vectorbtpro with a partially initialized settings
# object (missing "indicators"), which raises KeyError during vbt.IF(...)
# factory creation at import time.
def _ensure_vbt_indicators_settings():
    try:
        settings = getattr(vbt, "settings", None)
        if settings is None:
            return
        try:
            has_key = "indicators" in settings
        except Exception:
            has_key = False
        if not has_key:
            try:
                settings["indicators"] = {}
            except Exception:
                pass
    except Exception:
        # Keep import resilient; downstream code will surface real failures.
        pass

def _safe_if(*args, **kwargs):
    try:
        return vbt.IF(*args, **kwargs)
    except KeyError as e:
        if str(e).strip("'\"") == "indicators":
            _ensure_vbt_indicators_settings()
            return vbt.IF(*args, **kwargs)
        raise

_ensure_vbt_indicators_settings()

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
    # O(n * w) using np.partition for O(w) median selection per bar,
    # down from O(n * w * log w) with nanmedian (full sort).
    # np.partition finds the k-th smallest element without a full sort.
    # VBT passes window as float64; cast to int — np.partition and range() require int.
    w = int(window)
    result = np.full(len(arr), np.nan)
    half = w // 2
    for i in range(w - 1, len(arr)):
        window_slice = arr[i - w + 1:i + 1]
        if np.isnan(window_slice).all():
            continue
        partitioned = np.partition(window_slice, half)
        if w % 2 == 1:
            result[i] = partitioned[half]
        else:
            result[i] = (partitioned[half - 1] + partitioned[half]) / 2.0
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
    # Pine V6 uses the v80 destructuring bug: ta.bb() -> [middle, upper, lower] but
    # the old library assigned upperBand=middle, lowerBand=upper.
    # So Pine's ecart = upperBand - lowerBand = middle - upper → NEGATIVE.
    # Python must replicate this signed behaviour so that compute_bars_since_below
    # fires on WIDE bands (not narrow), matching Pine semantics for nb_bars_above_signal.
    ecart = lower_band - upper_band  # negative, like Pine's (middle - upper)
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
def cross_sar_sma_exit_nb(sma, sar):
    """Cross SAR/SMA exit: SAR crosses above SMA (crossunder of SMA below SAR).
    Signal = SMA[i-1] > SAR[i-1] AND SMA[i] < SAR[i] (SAR passes above SMA)."""
    n = len(sma)
    signal = np.zeros(n, dtype=np.bool_)
    for i in range(1, n):
        if (not np.isnan(sma[i-1]) and not np.isnan(sar[i-1]) and
                not np.isnan(sma[i]) and not np.isnan(sar[i]) and
                sma[i-1] >= sar[i-1] and sma[i] < sar[i]):
            signal[i] = True
    return signal


@njit(cache=True)
def pivot_low_nb(series, left, right):
    """Detect pivot lows: series[i-left..i-1] all > series[i] and series[i+1..i+right] all > series[i].
    Returns boolean array with True at pivot low bar (confirmed after 'right' bars)."""
    n = len(series)
    signal = np.zeros(n, dtype=np.bool_)
    for i in range(left, n - right):
        if np.isnan(series[i]):
            continue
        is_pivot = True
        for j in range(1, left + 1):
            if np.isnan(series[i - j]) or series[i - j] <= series[i]:
                is_pivot = False
                break
        if not is_pivot:
            continue
        for j in range(1, right + 1):
            if np.isnan(series[i + j]) or series[i + j] <= series[i]:
                is_pivot = False
                break
        if is_pivot:
            # Signal fires at confirmation bar (i + right)
            signal[i + right] = True
    return signal


@njit(cache=True)
def linreg_exit_nb(close, length, offset):
    """Linear regression exit: crossunder(close, linreg(close, length, offset)).
    linreg = linear regression forecast value at bar - offset.

    O(n) incremental algorithm: sum_x and sum_x2 are mathematical constants that
    depend only on `length` (not on which window).  sum_y and sum_qy are maintained
    as sliding accumulators — O(1) update per bar vs the previous O(length) inner loop.

      sum_x  = 0+1+...+(L-1)     = L*(L-1)/2         (constant)
      sum_x2 = 0²+1²+...+(L-1)² = L*(L-1)*(2L-1)/6  (constant)
      sum_xy = sum_qy - start * sum_y    (sum_qy uses absolute bar indices)
    """
    # VBT passes length and offset as float64; cast to int for indices and range().
    # Keep L as float64 for the regression arithmetic.
    _length = int(length)
    _offset = int(offset)
    n = len(close)
    signal = np.zeros(n, dtype=np.bool_)
    if n < _length + _offset:
        return signal

    L = float(_length)
    sum_x_c = L * (L - 1.0) / 2.0
    sum_x2_c = L * (L - 1.0) * (2.0 * L - 1.0) / 6.0
    denom_c = L * sum_x2_c - sum_x_c * sum_x_c

    linreg = np.full(n, np.nan)
    first_i = _length - 1 + _offset

    # Initialise sliding accumulators for the first window (bars 0.._length-1)
    sum_y = 0.0
    sum_qy = 0.0   # Σ j * close[j] using absolute bar index j
    nan_count = 0
    for j in range(_length):
        v = close[j]
        if np.isnan(v):
            nan_count += 1
        else:
            sum_y += v
            sum_qy += float(j) * v

    if nan_count == 0 and denom_c != 0.0:
        sum_xy = sum_qy  # start0 == 0, so sum_qy - 0 * sum_y == sum_qy
        slope = (L * sum_xy - sum_x_c * sum_y) / denom_c
        intercept = (sum_y - slope * sum_x_c) / L
        linreg[first_i] = intercept + slope * (L - 1.0)

    # Slide window one bar at a time — O(1) per bar
    for i in range(first_i + 1, n):
        start = i - _offset - _length + 1
        old_start = start - 1    # bar dropping out of the back of the window
        new_end = i - _offset    # bar entering the front of the window

        old_val = close[old_start]
        if np.isnan(old_val):
            nan_count -= 1
        else:
            sum_y -= old_val
            sum_qy -= float(old_start) * old_val

        new_val = close[new_end]
        if np.isnan(new_val):
            nan_count += 1
        else:
            sum_y += new_val
            sum_qy += float(new_end) * new_val

        if nan_count > 0 or denom_c == 0.0:
            continue

        sum_xy = sum_qy - float(start) * sum_y
        slope = (L * sum_xy - sum_x_c * sum_y) / denom_c
        intercept = (sum_y - slope * sum_x_c) / L
        linreg[i] = intercept + slope * (L - 1.0)

    # Crossunder detection: close crosses below linreg
    for i in range(1, n):
        if (not np.isnan(close[i-1]) and not np.isnan(linreg[i-1]) and
                not np.isnan(close[i]) and not np.isnan(linreg[i]) and
                close[i-1] >= linreg[i-1] and close[i] < linreg[i]):
            signal[i] = True
    return signal


@njit(cache=True)
def volat_down_exit_nb(close, upper, lower, seuil_overbought):
    """Volatility down / %BB exit: crossunder of %BB below seuil_overbought.
    %BB (BBR) = (close - lower) / (upper - lower)
    Signal = BBR[i-1] >= seuil AND BBR[i] < seuil."""
    n = len(close)
    signal = np.zeros(n, dtype=np.bool_)
    for i in range(1, n):
        band_width_prev = upper[i-1] - lower[i-1]
        band_width = upper[i] - lower[i]
        if band_width_prev == 0.0 or band_width == 0.0:
            continue
        if (np.isnan(close[i-1]) or np.isnan(upper[i-1]) or np.isnan(lower[i-1]) or
                np.isnan(close[i]) or np.isnan(upper[i]) or np.isnan(lower[i])):
            continue
        bbr_prev = (close[i-1] - lower[i-1]) / band_width_prev
        bbr = (close[i] - lower[i]) / band_width
        if bbr_prev >= seuil_overbought and bbr < seuil_overbought:
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
def macd_exit_signal_nb(close, fast_length, slow_length, signal_length, use_type_a, use_type_b, use_sma=False):
    n = len(close)
    signal = np.zeros(n, dtype=np.bool_)
    if n < 2:
        return signal

    if use_sma:
        ma_fast = vbt.indicators.nb.ma_1d_nb(close, fast_length)
        ma_slow = vbt.indicators.nb.ma_1d_nb(close, slow_length)
        macd = ma_fast - ma_slow
        macd_signal = vbt.indicators.nb.ma_1d_nb(macd, signal_length)
    else:
        ma_fast = ema_nb(close, fast_length)
        ma_slow = ema_nb(close, slow_length)
        macd = ma_fast - ma_slow
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

@njit(cache=True)
def depassement_roc_long_nb(close, open_, high, low, depass_sma_roc, roc_max):
    """RoC filter for T1: candle is bullish, RoC is between depass*avg and roc_max*avg.
    RoC = (high - low) / high * 100
    moy_RoC = SMA(RoC, 20)
    signal = (close > open) AND (RoC >= depass * moy_RoC) AND (RoC <= roc_max * moy_RoC)
    """
    n = len(close)
    signal = np.zeros(n, dtype=np.bool_)
    if n < 2:
        return signal

    roc = np.empty(n)
    for i in range(n):
        if high[i] == 0.0 or np.isnan(high[i]) or np.isnan(low[i]):
            roc[i] = np.nan
        else:
            roc[i] = (high[i] - low[i]) / high[i] * 100.0

    moy_roc = vbt.indicators.nb.ma_1d_nb(roc, 20)

    for i in range(n):
        if np.isnan(close[i]) or np.isnan(open_[i]) or np.isnan(moy_roc[i]) or np.isnan(roc[i]):
            continue
        if moy_roc[i] == 0.0:
            continue
        if (close[i] > open_[i] and
                roc[i] >= depass_sma_roc * moy_roc[i] and
                roc[i] <= roc_max * moy_roc[i]):
            signal[i] = True

    return signal


@njit(cache=True)
def nb_bars_under_bbw_signal_nb(cross_bbw_low, nb_bars_mini):
    """Rolling count: True when cross_bbw_low has been True for at least nb_bars_mini consecutive bars."""
    n = len(cross_bbw_low)
    signal = np.zeros(n, dtype=np.bool_)
    counter = 0
    for i in range(n):
        if cross_bbw_low[i]:
            counter += 1
        else:
            counter = 0
        if counter >= nb_bars_mini:
            signal[i] = True
    return signal


@njit(cache=True)
def bbandcross_barssince_signal_nb(close, upper, lower, nb_bars_entre):
    """True when barssince(crossover(close, upper)) >= nb_bars_entre
    AND barssince(crossunder(close, lower)) >= nb_bars_entre.
    This ensures the price hasn't recently crossed the BB bands."""
    n = len(close)
    signal = np.zeros(n, dtype=np.bool_)
    if n < 2:
        return signal

    barssince_cross_upper = nb_bars_entre  # start high so signal is True initially
    barssince_cross_lower = nb_bars_entre

    for i in range(1, n):
        if np.isnan(close[i]) or np.isnan(close[i - 1]):
            continue
        if np.isnan(upper[i]) or np.isnan(upper[i - 1]):
            continue
        if np.isnan(lower[i]) or np.isnan(lower[i - 1]):
            continue

        # crossover(close, upper): close crosses above upper
        if close[i - 1] <= upper[i - 1] and close[i] > upper[i]:
            barssince_cross_upper = 0
        else:
            barssince_cross_upper += 1

        # crossunder(close, lower): close crosses below lower
        if close[i - 1] >= lower[i - 1] and close[i] < lower[i]:
            barssince_cross_lower = 0
        else:
            barssince_cross_lower += 1

        signal[i] = (barssince_cross_upper >= nb_bars_entre) and (barssince_cross_lower >= nb_bars_entre)

    return signal


# ======================================================================
# INDICATOR FACTORIES (Vectorized)
# ======================================================================

# Using with_apply_func to allow VBT to handle broadcasting of parameters automatically.
# We explicitly set input names to match what the strategy will pass.

EcartBollingerBorne = _safe_if(
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

BollingerHorizontal = _safe_if(
    class_name='BollingerHorizontal',
    input_names=['bbw', 'mmbbw', 'mediane_bbw'],
    param_names=['coeff_medianeBBW'],
    output_names=['signal']
).with_apply_func(
    bollinger_horizontal_signal_nb,
    takes_1d=True,
    coeff_medianeBBW=1.1
)

CrossBBWLowSignal = _safe_if(
    class_name='CrossBBWLowSignal',
    input_names=['upper_band', 'lower_band', 'middle_band'],
    param_names=['fenetre_lowest', 'seuil_lowest'],
    output_names=['signal']
).with_apply_func(
    cross_bbw_low_signal_nb,
    takes_1d=True,
    fenetre_lowest=80,
    seuil_lowest=3.5
)

NbBarsUnderBBW = _safe_if(
    class_name='NbBarsUnderBBW',
    input_names=['cross_bbw_low'],
    param_names=['nb_bars_mini'],
    output_names=['signal']
).with_apply_func(
    nb_bars_under_bbw_signal_nb,
    takes_1d=True,
    nb_bars_mini=4
)

BBandCrossBarssince = _safe_if(
    class_name='BBandCrossBarssince',
    input_names=['close', 'upper', 'lower'],
    param_names=['nb_bars_entre'],
    output_names=['signal']
).with_apply_func(
    bbandcross_barssince_signal_nb,
    takes_1d=True,
    nb_bars_entre=5
)

DepassementRoCLong = _safe_if(
    class_name='DepassementRoCLong',
    input_names=['close', 'open_', 'high', 'low'],
    param_names=['depass_sma_roc', 'roc_max'],
    output_names=['signal']
).with_apply_func(
    depassement_roc_long_nb,
    takes_1d=True,
    depass_sma_roc=0.01,
    roc_max=100.0
)

SMAExit = _safe_if(
    class_name='SMAExit',
    input_names=['close'],
    param_names=['user_exit_sma_length'],
    output_names=['signal']
).with_apply_func(
    calculate_exit_sma_nb,
    takes_1d=True,
    user_exit_sma_length=20
)

ParabolicSAR = _safe_if(
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

MACDExit = _safe_if(
    class_name='MACDExit',
    input_names=['close'],
    param_names=['fast_length', 'slow_length', 'signal_length', 'use_type_a', 'use_type_b', 'use_sma'],
    output_names=['signal']
).with_apply_func(
    macd_exit_signal_nb,
    takes_1d=True,
    fast_length=9,
    slow_length=19,
    signal_length=6,
    use_type_a=True,
    use_type_b=True,
    use_sma=True
)

CrossSARSMAExit = _safe_if(
    class_name='CrossSARSMAExit',
    input_names=['sma', 'sar'],
    param_names=[],
    output_names=['signal']
).with_apply_func(
    cross_sar_sma_exit_nb,
    takes_1d=True
)

PivotLow = _safe_if(
    class_name='PivotLow',
    input_names=['series'],
    param_names=['left', 'right'],
    output_names=['signal']
).with_apply_func(
    pivot_low_nb,
    takes_1d=True,
    left=2,
    right=2
)

LinregExit = _safe_if(
    class_name='LinregExit',
    input_names=['close'],
    param_names=['length', 'offset'],
    output_names=['signal']
).with_apply_func(
    linreg_exit_nb,
    takes_1d=True,
    length=15,
    offset=1
)

VolatDownExit = _safe_if(
    class_name='VolatDownExit',
    input_names=['close', 'upper', 'lower'],
    param_names=['seuil_overbought'],
    output_names=['signal']
).with_apply_func(
    volat_down_exit_nb,
    takes_1d=True,
    seuil_overbought=0.85
)
