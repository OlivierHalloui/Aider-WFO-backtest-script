# Import necessary libraries for strategy
import pandas as pd
import numpy as np
import vectorbtpro as vbt
from metrics import trade_stat, calc_pqs
from indicators import (
    EcartBollingerBorne, BollingerHorizontal,
    CrossBBWLowSignal, NbBarsUnderBBW, BBandCrossBarssince,
    DepassementRoCLong, DepassementRoCShort,
    SMAExit, SMAShortExit, ParabolicSAR,
    MACDExit, MACDShortExit,
    CrossSARSMAExit, CrossSARSMAShortExit,
    PivotLow, PivotHigh, LinregExit, VolatDownExit,
)

# ======================================================================
# Utility helpers
# ======================================================================

def _bool_fill(s) -> "pd.Series":
    """Replace NaN with False without triggering pandas FutureWarning.

    pandas ≥ 2.1 warns when silently downcasting object-dtype arrays via
    .fillna(). Using np.where on the raw array bypasses the pandas
    downcasting path entirely.  Preserves 2D shape (DataFrame) for the
    vectorized (multi-combination) path.
    """
    vals = np.where(pd.isna(s), False, np.asarray(s, dtype=bool))
    if hasattr(s, 'columns'):
        return pd.DataFrame(vals, index=s.index, columns=s.columns)
    return pd.Series(vals, index=s.index if hasattr(s, 'index') else None)


def _flag_is_vectorized(flag) -> bool:
    """True when a boolean flag is a multi-element array (vectorized run)."""
    return (
        hasattr(flag, "__len__")
        and not isinstance(flag, (str, bytes, dict))
        and np.asarray(flag).size > 1
    )


def _flag_array(flag) -> np.ndarray:
    """Return a 1-D boolean array for a vectorized flag."""
    return np.asarray(flag, dtype=bool).reshape(-1)


def _apply_enable_mask(signal, flag, vector_len):
    """Mask a vectorized (2-D) signal by an enable-flag array, or zero it when
    scalar-disabled.  Mirrors the SAR/MACD exit pattern for the remaining
    enable flags (cross_sar_sma, retour_bb, regline, volat_down)."""
    if _flag_is_vectorized(flag):
        mask = pd.DataFrame(
            np.tile(_flag_array(flag), (len(signal), 1)),
            index=signal.index,
            columns=signal.columns,
        )
        return signal & mask
    if not bool(flag):
        return signal & ~signal
    return signal



# ======================================================================
# Per-window indicator cache
# ======================================================================
# Bollinger Bands and derived rolling statistics (bbw, mmbbw, mediane_bbw) are
# expensive to compute and depend only on (timeperiod, StDev, matype) — not on
# the other tunable parameters.  In sequential (non-vectorised) mode, many trials
# share the same Bollinger Band settings.  Caching here gives a high hit rate:
#   7 timeperiod × 5 StDev × 1 matype → ~35 distinct entries vs 1000+ trials.
#
# Thread safety: this cache is intentionally module-level and works correctly for
# the current single-threaded window loop.  If window-level parallelism is ever
# added, replace with threading.local().
_WINDOW_INDICATOR_CACHE: dict = {}


def clear_window_indicator_cache() -> None:
    """Reset the per-window indicator cache.  Call once at the start of each WFO window."""
    _WINDOW_INDICATOR_CACHE.clear()


# ======================================================================
# STRATEGY IMPLEMENTATION
# ======================================================================

def create_signal_generators(df, **params):
    """
    Create signal generators for the ATDMF strategy using VectorBT's built-in 
    indicator factories or custom vectorized indicators.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        OHLCV DataFrame with 'Open', 'High', 'Low', 'Close'
    **params : dict
        Strategy parameters. Can be scalars or lists/arrays for vectorization.
        
    Returns:
    --------
    dict
        Dictionary of signal generators (VBT objects or DataFrames)
    """
    # Parameter defaults for extraction, broadcast, and scalarization.
    _PARAM_DEFAULTS = {
        'timeperiod': 12, 'StDev': 1.3, 'matype': 0,
        'coeff_medianeBBW': 1.2, 'coef_mediane': 0.89,
        'Nb_bars_above': 1, 'fenetre_lowest': 80, 'seuil_lowest': 3.5,
        'longueur_mediane': 100, 'nb_bars_under_bbw_mini': 4,
        'nb_bars_entre_bb': 5,
        'depassement_sma_roc': 0.01, 'roc_max_t1': 100.0,
        'use_roc_filter': True,
        # Keep T2 optional by default in native mode to avoid ultra-sparse entries
        # on short WFO windows; Pine-parity runs can still force True explicitly.
        'use_t2_signal': False, 'use_divergence_bb': True,
        'user_exit_sma_length': 14,
        'sar_start': 0.02, 'sar_increment': 0.02, 'sar_maximum': 0.2,
        'exit_sar_enabled': True, 'macd_fast_length': 9,
        'macd_slow_length': 19, 'macd_signal_length': 6,
        'macd_ma_type': 'sma',
        'exit_macd_enabled': True, 'exit_macd_type_a': True, 'exit_macd_type_b': True,
        'exit_cross_sar_sma_enabled': True,
        'exit_retour_bb_enabled': False, 'nb_bars_left_pivot': 2, 'nb_bars_right_pivot': 2,
        'exit_regline_enabled': False, 'nombre_periodes_reglin': 15, 'i_bars_back': 1,
        'exit_volat_down_enabled': False, 'seuil_overbought_bb': 0.85,
        'strategy_direction': 'long_only',
    }

    def is_array_like(val):
        return hasattr(val, "__len__") and not isinstance(val, (str, bytes, dict))

    # Detect vectorized length from array-like parameters.
    vector_len = 1
    lengths = []
    for name in _PARAM_DEFAULTS:
        val = params.get(name, _PARAM_DEFAULTS[name])
        if is_array_like(val):
            try:
                lengths.append(len(val))
            except Exception:
                pass

    non_scalar_lengths = sorted(set(l for l in lengths if l > 1))
    if len(non_scalar_lengths) > 1:
        raise ValueError(f"Inconsistent vectorized parameter lengths: {non_scalar_lengths}")
    if non_scalar_lengths:
        vector_len = non_scalar_lengths[0]

    def broadcast(val, length):
        if is_array_like(val):
            arr = np.asarray(val).reshape(-1)
            if len(arr) == 0:
                raise ValueError("Empty parameter array is not allowed.")
            if len(arr) == 1:
                return np.full(length, arr[0])
            if len(arr) != length:
                raise ValueError(f"Parameter length mismatch: expected {length}, got {len(arr)}.")
            return arr
        return np.full(length, val)

    def scalarize(val):
        if not is_array_like(val):
            return val
        arr = np.asarray(val).reshape(-1)
        if len(arr) == 0:
            raise ValueError("Empty parameter array is not allowed.")
        out = arr[0]
        return out.item() if isinstance(out, np.generic) else out

    # Extract, broadcast or scalarize all parameters in a single loop.
    p = {}
    for name, default in _PARAM_DEFAULTS.items():
        val = params.get(name, default)
        p[name] = broadcast(val, vector_len) if vector_len > 1 else scalarize(val)

    # Unpack for readability in downstream code.
    timeperiod = p['timeperiod']
    StDev = p['StDev']
    matype = p['matype']
    coeff_medianeBBW = p['coeff_medianeBBW']
    coef_mediane = p['coef_mediane']
    Nb_bars_above = p['Nb_bars_above']
    fenetre_lowest = p['fenetre_lowest']
    seuil_lowest = p['seuil_lowest']
    longueur_mediane = p['longueur_mediane']
    nb_bars_under_bbw_mini = p['nb_bars_under_bbw_mini']
    nb_bars_entre_bb = p['nb_bars_entre_bb']
    depassement_sma_roc = p['depassement_sma_roc']
    roc_max_t1 = p['roc_max_t1']
    use_roc_filter = p['use_roc_filter']
    use_t2_signal = p['use_t2_signal']
    use_divergence_bb = p['use_divergence_bb']
    user_exit_sma_length = p['user_exit_sma_length']
    sar_start = p['sar_start']
    sar_increment = p['sar_increment']
    sar_maximum = p['sar_maximum']
    exit_sar_enabled = p['exit_sar_enabled']
    macd_fast_length = p['macd_fast_length']
    macd_slow_length = p['macd_slow_length']
    macd_signal_length = p['macd_signal_length']
    macd_ma_type = p['macd_ma_type']
    exit_macd_enabled = p['exit_macd_enabled']
    exit_macd_type_a = p['exit_macd_type_a']
    exit_macd_type_b = p['exit_macd_type_b']
    exit_cross_sar_sma_enabled = p['exit_cross_sar_sma_enabled']
    exit_retour_bb_enabled = p['exit_retour_bb_enabled']
    nb_bars_left_pivot = p['nb_bars_left_pivot']
    nb_bars_right_pivot = p['nb_bars_right_pivot']
    exit_regline_enabled = p['exit_regline_enabled']
    nombre_periodes_reglin = p['nombre_periodes_reglin']
    i_bars_back = p['i_bars_back']
    exit_volat_down_enabled = p['exit_volat_down_enabled']
    seuil_overbought_bb = p['seuil_overbought_bb']

    # Keep OHLC inputs as 1D series and let vectorbt broadcast with parameter arrays.
    # Pre-expanding price columns together with vectorized params can create cartesian
    # products (N inputs x N params) and break shape alignment.
    close_price = df['Close']
    open_price  = df['Open']
    high_price  = df['High']
    low_price   = df['Low']

    _sd_raw = p.get('strategy_direction', 'long_only')
    if is_array_like(_sd_raw) and np.asarray(_sd_raw).size > 1:
        _sd_vals = np.asarray(_sd_raw).reshape(-1)
        if len(set(_sd_vals.tolist())) > 1:
            raise ValueError(
                f"strategy_direction must be uniform across combinations, got {list(_sd_vals)}")
    strategy_direction = str(scalarize(_sd_raw))

    def normalize_columns(obj):
        """Ensure deterministic unique columns to avoid many-to-many label alignment."""
        if hasattr(obj, "columns"):
            out = obj.copy()
            out.columns = pd.RangeIndex(len(out.columns))
            return out
        return obj
    
    # 1. Bollinger Bands (Vectorized via TA-Lib wrapper in VBT)
    # This handles scalar or array parameters for timeperiod/StDev
    # 1 input -> N params -> N outputs. per_column=False (default)
    #
    # Per-window cache: bbands and derived rolling stats depend only on
    # (timeperiod, StDev, matype).  In sequential mode, many trials share the same
    # Bollinger Band settings — cache the result to avoid recomputation.
    # Cache key includes a data signature so cached arrays from the IS period are
    # never reused for OOS or final-backtest DataFrames (different index/length).
    _df_sig = (len(df), df.index[0] if len(df) > 0 else None)
    _bb_key = (
        int(timeperiod) if np.isscalar(timeperiod) else None,
        round(float(StDev), 6) if np.isscalar(StDev) else None,
        int(matype) if np.isscalar(matype) else None,
        _df_sig,
    )
    _bb_cached = _WINDOW_INDICATOR_CACHE.get(_bb_key) if None not in _bb_key else None
    if _bb_cached is not None:
        upper_band, lower_band, middle_band, bbw, mmbbw, mediane_bbw = _bb_cached
    else:
        bbands = vbt.talib("BBANDS").run(
            close_price,
            timeperiod=timeperiod,
            nbdevup=StDev,
            nbdevdn=StDev,
            matype=matype,
            skipna=True
        )
        upper_band = normalize_columns(bbands.upperband)
        lower_band = normalize_columns(bbands.lowerband)
        middle_band = normalize_columns(bbands.middleband)
        bbw = (upper_band - lower_band) / middle_band
        mmbbw = bbw.rolling(window=5).mean()
        mediane_bbw = bbw.rolling(window=200).median()
        if None not in _bb_key:
            _WINDOW_INDICATOR_CACHE[_bb_key] = (
                upper_band, lower_band, middle_band, bbw, mmbbw, mediane_bbw
            )

    def align_input_to_columns(base_series, reference_obj):
        if not hasattr(reference_obj, "columns"):
            return base_series
        cols = reference_obj.columns
        aligned = pd.concat([base_series] * len(cols), axis=1)
        aligned.columns = pd.RangeIndex(len(cols))
        return aligned

    close_price_aligned = align_input_to_columns(close_price, upper_band)
    high_price_aligned = align_input_to_columns(high_price, upper_band)
    low_price_aligned = align_input_to_columns(low_price, upper_band)

    # 2. Custom Indicators (Vectorized via indicators.py factories)

    # Ecart Bollinger Borne
    # Needs: price, upper, lower, timeperiod, longueur_mediane, coef_mediane, Nb_bars_above
    # N inputs -> N params -> N outputs (1-to-1). per_column=True
    nb_bars_above_ind = EcartBollingerBorne.run(
        prix=close_price_aligned,
        upper_band=upper_band,
        lower_band=lower_band,
        timeperiod=timeperiod,
        longueur_mediane=longueur_mediane,
        coef_mediane=coef_mediane,
        Nb_bars_above=Nb_bars_above,
        per_column=True
    )

    # Bollinger Horizontal
    # Needs: bbw, mmbbw, mediane_bbw, coeff_medianeBBW
    # bbw / mmbbw / mediane_bbw are read from cache above (or freshly computed).
    
    # Pass pre-calculated stats to indicator logic
    bollinger_horizontal_ind = BollingerHorizontal.run(
        bbw=bbw,
        mmbbw=mmbbw,
        mediane_bbw=mediane_bbw,
        coeff_medianeBBW=coeff_medianeBBW,
        per_column=True
    )
    
    # Cross BBW Low
    # Needs: upper, lower, middle, fenetre_lowest, seuil_lowest
    # N inputs -> N params -> N outputs. per_column=True
    cross_bbw_low_ind = CrossBBWLowSignal.run(
        upper_band=upper_band,
        lower_band=lower_band,
        middle_band=middle_band,
        fenetre_lowest=fenetre_lowest,
        seuil_lowest=seuil_lowest,
        per_column=True
    )

    # Nb bars under BBW (consecutive bars where cross_bbw_low is True)
    cross_bbw_low_signal = normalize_columns(cross_bbw_low_ind.signal)
    nb_bars_under_bbw_ind = NbBarsUnderBBW.run(
        cross_bbw_low=cross_bbw_low_signal.astype(float),
        nb_bars_mini=nb_bars_under_bbw_mini,
        per_column=True
    )

    # BBand cross barssince (no recent crossover/crossunder of BB bands)
    bbandcross_barssince_ind = BBandCrossBarssince.run(
        close=close_price_aligned,
        upper=upper_band,
        lower=lower_band,
        nb_bars_entre=nb_bars_entre_bb,
        per_column=True
    )

    # DepassementRoC — T1 filter
    open_price = df['Open']
    open_price_aligned = align_input_to_columns(open_price, upper_band)
    depassement_roc_ind = DepassementRoCLong.run(
        close=close_price_aligned,
        open_=open_price_aligned,
        high=high_price_aligned,
        low=low_price_aligned,
        depass_sma_roc=depassement_sma_roc,
        roc_max=roc_max_t1,
        per_column=True
    )

    # SMA Exit — cached by (user_exit_sma_length, data); sma_series co-computed here
    _sma_exit_key = (
        "sma",
        int(user_exit_sma_length) if np.isscalar(user_exit_sma_length) else None,
        _df_sig,
    )
    _sma_exit_cached = _WINDOW_INDICATOR_CACHE.get(_sma_exit_key) if None not in _sma_exit_key else None
    if _sma_exit_cached is not None:
        _sma_exit_signal, sma_series = _sma_exit_cached
    else:
        sma_exit_ind = SMAExit.run(
            close=close_price_aligned,
            user_exit_sma_length=user_exit_sma_length,
            per_column=True
        )
        _sma_exit_signal = normalize_columns(sma_exit_ind.signal)
        if is_array_like(user_exit_sma_length) and np.asarray(user_exit_sma_length).size > 1 \
                and hasattr(close_price_aligned, 'columns'):
            # Vectorized: one SMA per combination (per-column window).
            _lens = np.asarray(user_exit_sma_length, dtype=int).reshape(-1)
            if close_price_aligned.shape[1] != len(_lens):
                raise ValueError(
                    f"user_exit_sma_length has {len(_lens)} values but close has "
                    f"{close_price_aligned.shape[1]} columns")
            sma_series = pd.concat(
                [close_price_aligned.iloc[:, i].rolling(window=int(_lens[i]), min_periods=int(_lens[i])).mean()
                 for i in range(len(_lens))], axis=1)
            sma_series.columns = close_price_aligned.columns
        elif is_array_like(user_exit_sma_length) and np.asarray(user_exit_sma_length).size > 1:
            raise ValueError(
                "user_exit_sma_length is vectorized but close_price_aligned is not 2-D")
        else:
            _sma_len = int(scalarize(user_exit_sma_length)) if is_array_like(user_exit_sma_length) else int(user_exit_sma_length)
            sma_series = close_price_aligned.rolling(window=_sma_len, min_periods=_sma_len).mean()
        if None not in _sma_exit_key:
            _WINDOW_INDICATOR_CACHE[_sma_exit_key] = (_sma_exit_signal, sma_series)

    use_sma = (macd_ma_type == 'sma') if isinstance(macd_ma_type, str) else True
    # MACD Exit — cached by (fast, slow, signal, type_a, type_b, use_sma, data)
    _macd_exit_key = (
        "macd",
        int(macd_fast_length) if np.isscalar(macd_fast_length) else None,
        int(macd_slow_length) if np.isscalar(macd_slow_length) else None,
        int(macd_signal_length) if np.isscalar(macd_signal_length) else None,
        bool(exit_macd_type_a) if np.isscalar(exit_macd_type_a) else None,
        bool(exit_macd_type_b) if np.isscalar(exit_macd_type_b) else None,
        bool(use_sma),
        _df_sig,
    )
    _macd_cached = _WINDOW_INDICATOR_CACHE.get(_macd_exit_key) if None not in _macd_exit_key else None
    if _macd_cached is not None:
        macd_exit_signal = _macd_cached
    else:
        _macd_ind = MACDExit.run(
            close=close_price_aligned,
            fast_length=macd_fast_length,
            slow_length=macd_slow_length,
            signal_length=macd_signal_length,
            use_type_a=exit_macd_type_a,
            use_type_b=exit_macd_type_b,
            use_sma=use_sma,
            per_column=True
        )
        macd_exit_signal = normalize_columns(_macd_ind.signal)
        if None not in _macd_exit_key:
            _WINDOW_INDICATOR_CACHE[_macd_exit_key] = macd_exit_signal

    # Parabolic SAR — cached by (sar_start, sar_increment, sar_maximum, data)
    _sar_key = (
        "sar",
        round(float(sar_start), 6) if np.isscalar(sar_start) else None,
        round(float(sar_increment), 6) if np.isscalar(sar_increment) else None,
        round(float(sar_maximum), 6) if np.isscalar(sar_maximum) else None,
        _df_sig,
    )
    _sar_cached = _WINDOW_INDICATOR_CACHE.get(_sar_key) if None not in _sar_key else None
    if _sar_cached is not None:
        sar_signal = _sar_cached
    else:
        psar_ind = ParabolicSAR.run(
            high=high_price_aligned,
            low=low_price_aligned,
            sar_start=sar_start,
            sar_increment=sar_increment,
            sar_maximum=sar_maximum,
            per_column=True
        )
        sar_signal = normalize_columns(psar_ind.sar)
        if None not in _sar_key:
            _WINDOW_INDICATOR_CACHE[_sar_key] = sar_signal
    prev_sar = sar_signal.shift(1)
    prev_close = close_price_aligned.shift(1)
    if hasattr(sar_signal, "columns"):
        sar_exit_arr = (
            prev_close.to_numpy() > prev_sar.to_numpy()
        ) & (
            sar_signal.to_numpy() > close_price_aligned.to_numpy()
        )
        sar_exit_signal = _bool_fill(pd.DataFrame(
            sar_exit_arr,
            index=sar_signal.index,
            columns=sar_signal.columns
        ))
    else:
        sar_exit_signal = _bool_fill(
            (prev_close > prev_sar) &
            (sar_signal > close_price_aligned)
        )

    # macd_exit_signal set in MACD cache block above
    sar_exit_signal = _apply_enable_mask(sar_exit_signal, exit_sar_enabled, vector_len)
    macd_exit_signal = _apply_enable_mask(macd_exit_signal, exit_macd_enabled, vector_len)

    # --- Additional exit signals (Phase 5) ---

    # 1. Cross SAR/SMA exit: SAR crosses above SMA
    # sma_series already computed and cached alongside SMAExit above.
    # NOTE: CrossSARSMAExit has no VBT params (param_names=[]), so per_column=True
    # is not valid.  With takes_1d=True, VBT already iterates over columns
    # automatically when inputs are DataFrames.
    cross_sar_sma_exit_ind = CrossSARSMAExit.run(
        sma=sma_series,
        sar=sar_signal,
    )
    cross_sar_sma_signal = normalize_columns(cross_sar_sma_exit_ind.signal).astype(bool)
    cross_sar_sma_signal = _apply_enable_mask(cross_sar_sma_signal, exit_cross_sar_sma_enabled, vector_len)

    # 2. Retour BB exit: pivot low on lower band
    retour_bb_signal = None
    _retour_vec = vector_len > 1 and _flag_is_vectorized(exit_retour_bb_enabled)
    if _retour_vec or bool(exit_retour_bb_enabled):
        retour_bb_ind = PivotLow.run(
            series=lower_band,
            left=nb_bars_left_pivot,
            right=nb_bars_right_pivot,
            per_column=True
        )
        retour_bb_signal = normalize_columns(retour_bb_ind.signal).astype(bool)
        if _retour_vec:
            retour_bb_signal = _apply_enable_mask(retour_bb_signal, exit_retour_bb_enabled, vector_len)

    # 3. Regline exit: crossunder(close, linreg) — cached by (length, offset, data)
    regline_signal = None
    _regline_vec = vector_len > 1 and _flag_is_vectorized(exit_regline_enabled)
    if _regline_vec or bool(exit_regline_enabled):
        _linreg_key = (
            "linreg",
            int(nombre_periodes_reglin) if np.isscalar(nombre_periodes_reglin) else None,
            int(i_bars_back) if np.isscalar(i_bars_back) else None,
            _df_sig,
        )
        _linreg_cached = _WINDOW_INDICATOR_CACHE.get(_linreg_key) if None not in _linreg_key else None
        if _linreg_cached is not None:
            regline_signal = _linreg_cached
        else:
            regline_ind = LinregExit.run(
                close=close_price_aligned,
                length=nombre_periodes_reglin,
                offset=i_bars_back,
                per_column=True
            )
            regline_signal = normalize_columns(regline_ind.signal).astype(bool)
            if None not in _linreg_key:
                _WINDOW_INDICATOR_CACHE[_linreg_key] = regline_signal
        if _regline_vec:
            regline_signal = _apply_enable_mask(regline_signal, exit_regline_enabled, vector_len)

    # 4. Volat down exit: crossunder(%BB, seuil_overbought)
    volat_down_signal = None
    _volat_vec = vector_len > 1 and _flag_is_vectorized(exit_volat_down_enabled)
    if _volat_vec or bool(exit_volat_down_enabled):
        volat_down_ind = VolatDownExit.run(
            close=close_price_aligned,
            upper=upper_band,
            lower=lower_band,
            seuil_overbought=seuil_overbought_bb,
            per_column=True
        )
        volat_down_signal = normalize_columns(volat_down_ind.signal).astype(bool)
        if _volat_vec:
            volat_down_signal = _apply_enable_mask(volat_down_signal, exit_volat_down_enabled, vector_len)

    # ── SHORT SIGNALS (generated only when direction requires it) ──────────
    depassement_roc_short_signal = None
    sma_short_exit_signal        = None
    macd_short_exit_signal_out   = None
    cross_sar_sma_short_signal   = None
    sar_short_exit_signal        = None
    retour_bb_short_exit_signal  = None
    regline_short_signal         = None
    volat_up_exit_signal         = None

    if strategy_direction in ('short_only', 'both'):
        # --- T1 short entry: bearish RoC filter ---
        depassement_roc_short_ind = DepassementRoCShort.run(
            close=close_price_aligned,
            open_=open_price,
            high=high_price_aligned,
            low=low_price_aligned,
            depass_sma_roc=depassement_sma_roc,
            roc_max=roc_max_t1,
            per_column=True
        )
        depassement_roc_short_signal = normalize_columns(
            depassement_roc_short_ind.signal).astype(bool)

        # --- SMA short exit: close crosses ABOVE SMA ---
        # SMA exit has no separate enable flag in long mode (always active).
        # Mirror: always active for short mode too.
        sma_short_ind = SMAShortExit.run(
            close=close_price_aligned,
            user_exit_sma_length=user_exit_sma_length,
            per_column=True
        )
        sma_short_exit_signal = normalize_columns(sma_short_ind.signal).astype(bool)

        # --- SAR short exit: SAR crosses BELOW close (price recovers) ---
        # sar_signal contains raw SAR values (not boolean)
        close_price_aligned_s = close_price_aligned
        prev_sar_s = sar_signal.shift(1)
        prev_close_s = close_price_aligned_s.shift(1)
        if hasattr(sar_signal, "columns"):
            sar_short_arr = (
                prev_close_s.to_numpy() < prev_sar_s.to_numpy()
            ) & (
                close_price_aligned_s.to_numpy() > sar_signal.to_numpy()
            )
            sar_short_exit_signal = _bool_fill(pd.DataFrame(
                sar_short_arr,
                index=sar_signal.index,
                columns=sar_signal.columns
            ))
        else:
            sar_short_exit_signal = _bool_fill(
                (prev_close_s < prev_sar_s) &
                (close_price_aligned_s > sar_signal)
            )
        sar_short_exit_signal = _apply_enable_mask(sar_short_exit_signal, exit_sar_enabled, vector_len)

        # --- MACD short exit: MACD crosses ABOVE signal line ---
        _macd_short_vec = vector_len > 1 and _flag_is_vectorized(exit_macd_enabled)
        if _macd_short_vec or bool(exit_macd_enabled):
            _macd_short_key = (
                "macd_short",
                int(macd_fast_length) if np.isscalar(macd_fast_length) else None,
                int(macd_slow_length) if np.isscalar(macd_slow_length) else None,
                int(macd_signal_length) if np.isscalar(macd_signal_length) else None,
                bool(exit_macd_type_a) if np.isscalar(exit_macd_type_a) else None,
                bool(exit_macd_type_b) if np.isscalar(exit_macd_type_b) else None,
                bool(use_sma),
                _df_sig,
            )
            _macd_short_cached = (
                _WINDOW_INDICATOR_CACHE.get(_macd_short_key)
                if None not in _macd_short_key else None
            )
            if _macd_short_cached is not None:
                macd_short_exit_signal_out = _macd_short_cached
            else:
                _macd_short_ind = MACDShortExit.run(
                    close=close_price_aligned,
                    fast_length=macd_fast_length,
                    slow_length=macd_slow_length,
                    signal_length=macd_signal_length,
                    use_type_a=exit_macd_type_a,
                    use_type_b=exit_macd_type_b,
                    use_sma=use_sma,
                    per_column=True
                )
                macd_short_exit_signal_out = normalize_columns(_macd_short_ind.signal)
                if None not in _macd_short_key:
                    _WINDOW_INDICATOR_CACHE[_macd_short_key] = macd_short_exit_signal_out
        if _macd_short_vec:
            macd_short_exit_signal_out = _apply_enable_mask(macd_short_exit_signal_out, exit_macd_enabled, vector_len)

        # --- Cross SAR/SMA short exit: SAR crosses BELOW SMA ---
        _css_short_vec = vector_len > 1 and _flag_is_vectorized(exit_cross_sar_sma_enabled)
        if _css_short_vec or bool(exit_cross_sar_sma_enabled):
            cross_sar_sma_short_ind = CrossSARSMAShortExit.run(
                sma=sma_series,
                sar=sar_signal,
            )
            cross_sar_sma_short_signal = normalize_columns(
                cross_sar_sma_short_ind.signal).astype(bool)
            if _css_short_vec:
                cross_sar_sma_short_signal = _apply_enable_mask(cross_sar_sma_short_signal, exit_cross_sar_sma_enabled, vector_len)

        # --- Retour BB short exit: pivot HIGH on upper band ---
        _retour_short_vec = vector_len > 1 and _flag_is_vectorized(exit_retour_bb_enabled)
        if _retour_short_vec or bool(exit_retour_bb_enabled):
            pivot_high_ind = PivotHigh.run(
                series=upper_band,
                left=nb_bars_left_pivot,
                right=nb_bars_right_pivot,
                per_column=True
            )
            retour_bb_short_exit_signal = normalize_columns(
                pivot_high_ind.signal).astype(bool)
            if _retour_short_vec:
                retour_bb_short_exit_signal = _apply_enable_mask(retour_bb_short_exit_signal, exit_retour_bb_enabled, vector_len)

        # --- Regline short exit: close crosses ABOVE linreg (crossover) ---
        # Mirror of regline long exit (crossunder). Reuses cached linreg signal series.
        _regline_short_vec = vector_len > 1 and _flag_is_vectorized(exit_regline_enabled)
        if (_regline_short_vec or bool(exit_regline_enabled)) and regline_signal is not None:
            # linreg_exit_nb detects crossunder (long). For short we need crossover.
            # Reconstruct the crossover from close and the cached regline series.
            # We compute it inline to avoid duplicating the expensive linreg calculation.
            _linreg_key_s = (
                "linreg",
                int(nombre_periodes_reglin) if np.isscalar(nombre_periodes_reglin) else None,
                int(i_bars_back) if np.isscalar(i_bars_back) else None,
                _df_sig,
            )
            # Reuse the same LinregExit indicator but detect crossover instead of crossunder.
            # We build the crossover signal directly from close and the linreg value series.
            # Since linreg_exit_nb only returns the crossunder boolean, we need the raw
            # linreg values — compute them via a second LinregExit run on a reversed series
            # is too expensive; instead we build an inline pandas crossover:
            _cl_aligned = close_price_aligned
            # shift(1) gives previous bar; crossover = prev_close < prev_linreg AND close > linreg
            # We approximate linreg via the regline_signal to avoid recomputing:
            # regline_signal[t]=True means close crossed UNDER linreg at t (long exit).
            # For short exit we need: close crosses OVER linreg. This is NOT simply ~regline_signal.
            # We use LinregExit re-run on -close to get the crossover (algebraic mirror):
            _linreg_short_key = (
                "linreg_short",
                int(nombre_periodes_reglin) if np.isscalar(nombre_periodes_reglin) else None,
                int(i_bars_back) if np.isscalar(i_bars_back) else None,
                _df_sig,
            )
            _linreg_short_cached = (
                _WINDOW_INDICATOR_CACHE.get(_linreg_short_key)
                if None not in _linreg_short_key else None
            )
            if _linreg_short_cached is not None:
                regline_short_signal = _linreg_short_cached
            else:
                # Run LinregExit on negated close — crossunder(-close, linreg(-close))
                # is equivalent to crossover(close, linreg(close)).
                _neg_close = -close_price_aligned
                _linreg_short_ind = LinregExit.run(
                    close=_neg_close,
                    length=nombre_periodes_reglin,
                    offset=i_bars_back,
                    per_column=True
                )
                regline_short_signal = normalize_columns(_linreg_short_ind.signal).astype(bool)
                if None not in _linreg_short_key:
                    _WINDOW_INDICATOR_CACHE[_linreg_short_key] = regline_short_signal
            if _regline_short_vec:
                regline_short_signal = _apply_enable_mask(regline_short_signal, exit_regline_enabled, vector_len)
        else:
            regline_short_signal = None

        # --- Volat up short exit: %BB crosses ABOVE (1 - seuil_overbought) ---
        # Mirror of volat_down exit (long): price was in lower %BB zone, recovers upward.
        # Signal = BBR[i-1] < (1-seuil) AND BBR[i] >= (1-seuil).
        _volat_short_vec = vector_len > 1 and _flag_is_vectorized(exit_volat_down_enabled)
        if _volat_short_vec or bool(exit_volat_down_enabled):
            seuil_oversold = 1.0 - seuil_overbought_bb
            # Run VolatDownExit on mirrored %BB: %BB_short = (upper-close)/(upper-lower)
            # which equals (1 - %BB_long). crossunder(%BB_short, 1-seuil) == crossover(%BB_long, seuil-1+1=seuil)
            # Simpler: pass seuil_oversold as threshold to VolatDownExit on the mirrored series.
            _seuil_key = round(float(seuil_overbought_bb), 6) if np.isscalar(seuil_overbought_bb) else None
            _vup_key = ("volat_up", _seuil_key, _df_sig)
            _vup_cached = _WINDOW_INDICATOR_CACHE.get(_vup_key) if None not in _vup_key else None
            if _vup_cached is not None:
                volat_up_exit_signal = _vup_cached
            else:
                # %BB_short = (upper - close) / (upper - lower) = 1 - %BB_long.
                # crossunder(%BB_short, seuil_oversold) == crossover(%BB_long, 1-seuil_oversold)
                # Reuse VolatDownExit factory on the flipped series.
                _upper_a = upper_band
                _lower_a = lower_band
                _mirrored_close = _upper_a + _lower_a - close_price_aligned
                _vup_ind = VolatDownExit.run(
                    close=_mirrored_close,
                    upper=_upper_a,
                    lower=_lower_a,
                    seuil_overbought=seuil_oversold,
                    per_column=True
                )
                volat_up_exit_signal = normalize_columns(_vup_ind.signal).astype(bool)
                if None not in _vup_key:
                    _WINDOW_INDICATOR_CACHE[_vup_key] = volat_up_exit_signal
            if _volat_short_vec:
                volat_up_exit_signal = _apply_enable_mask(volat_up_exit_signal, exit_volat_down_enabled, vector_len)
        else:
            volat_up_exit_signal = None

    return {
        'upper_band': upper_band,
        'lower_band': lower_band,
        'middle_band': middle_band,
        'nb_bars_above_signal': normalize_columns(nb_bars_above_ind.signal),
        'bollinger_horizontal_signal': normalize_columns(bollinger_horizontal_ind.signal).astype(bool),
        'cross_bbw_low_signal': cross_bbw_low_signal,
        'nb_bars_under_bbw_signal': normalize_columns(nb_bars_under_bbw_ind.signal).astype(bool),
        'bbandcross_barssince_signal': normalize_columns(bbandcross_barssince_ind.signal).astype(bool),
        'depassement_roc_signal': normalize_columns(depassement_roc_ind.signal).astype(bool),
        'use_roc_filter': use_roc_filter,
        'use_t2_signal': use_t2_signal,
        'use_divergence_bb': use_divergence_bb,
        'strategy_direction': strategy_direction,
        # Long exits
        'sma_exit_signal': _sma_exit_signal,
        'sar_exit_signal': sar_exit_signal,
        'macd_exit_signal': macd_exit_signal.astype(bool),
        'cross_sar_sma_exit_signal': cross_sar_sma_signal,
        'retour_bb_exit_signal': retour_bb_signal,
        'regline_exit_signal': regline_signal,
        'volat_down_exit_signal': volat_down_signal,
        # Short entry
        'depassement_roc_short_signal': depassement_roc_short_signal,
        # Short exits
        'sma_short_exit_signal': sma_short_exit_signal,
        'sar_short_exit_signal': sar_short_exit_signal,
        'macd_short_exit_signal': macd_short_exit_signal_out,
        'cross_sar_sma_short_exit_signal': cross_sar_sma_short_signal,
        'retour_bb_short_exit_signal': retour_bb_short_exit_signal,
        'regline_short_exit_signal': regline_short_signal,
        'volat_up_exit_signal': volat_up_exit_signal,
    }

def create_entry_exit_conditions(df, signals):
    """
    Create entry and exit conditions for the ATDMF strategy.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        OHLCV DataFrame
    signals : dict
        Signal generators from create_signal_generators
        
    Returns:
    --------
    tuple
        (entry_condition, exit_condition)
    """
    close = df['Close']
    
    # VBT objects support logical operations (&, |) and comparison operators (>, <)
    # They automatically align and broadcast.
    
    upper_band = signals['upper_band']
    # middle_band = signals['middle_band'] # Not used in entry currently
    
    # Entry Condition
    # 1. All custom signals must be True
    # 2. Close > Upper Band
    # 3. Previous Close < Previous Upper Band (Crossover)
    
    # Shifted values for crossover check
    # We can use vbt.fshift or .shift() on the wrapper/series
    # If upper_band is a VBT object, .shift(1) works.
    
    prev_close = close.shift(1)
    prev_upper = upper_band.shift(1)
    
    # Combine signals
    # Note: Ensure we are working with booleans
    # Use DataFrame methods with explicit axis=0 to broadcast Series across columns
    # close > upper_band  => upper_band.lt(close, axis=0)
    # prev_close < prev_upper => prev_upper.gt(prev_close, axis=0)
    
    # T0: all sub-signals must be True
    T0 = (
        signals['cross_bbw_low_signal'] &
        signals['nb_bars_under_bbw_signal'] &
        signals['bbandcross_barssince_signal'] &
        signals['bollinger_horizontal_signal'] &
        signals['nb_bars_above_signal']
    )

    # Crossover: close crosses above upper band
    crossover_upper = upper_band.lt(close, axis=0) & prev_upper.ge(prev_close, axis=0)

    # T1: T0 (current or previous bar) + crossover + optional RoC filter
    _t0_cross = (T0 | _bool_fill(T0.shift(1))) & crossover_upper
    _roc = signals.get('use_roc_filter', True)
    _roc_vec = _flag_is_vectorized(_roc)
    if _roc_vec:
        _roc_mask = np.tile(_flag_array(_roc), (len(_t0_cross), 1))
        _dep = np.asarray(signals['depassement_roc_signal'], dtype=bool)
        if _dep.ndim == 1:
            _dep2d = np.broadcast_to(_dep.reshape(-1, 1), _roc_mask.shape)
        else:
            _dep2d = _dep
        T1 = pd.DataFrame(
            np.asarray(_t0_cross, dtype=bool) & np.where(_roc_mask, _dep2d, True),
            index=_t0_cross.index, columns=_t0_cross.columns,
        )
    else:
        T1 = _t0_cross & signals['depassement_roc_signal'] if bool(_roc) else _t0_cross

    # T2: stop-buy order placed at close of T1 bar, filled on next bar if High breaks out
    # Setup bar (T1, t-1): divergence_BB must be true on setup bar
    # Trigger bar (T2, t):  High[t] > High[t-1]  → order fills at High[t-1] + mintick
    # divergence_BB = upper band expanding AND lower band expanding (bands diverging)
    high = df['High']
    lower_band = signals['lower_band']
    divergence_BB = (upper_band.diff() > 0) & (lower_band.diff() < 0)

    _MINTICK = 0.01  # BTCUSDT mintick (syminfo.mintick in Pine V6)

    use_t2 = signals.get('use_t2_signal', True)
    use_div_bb = signals.get('use_divergence_bb', True)
    _t2_vec = _flag_is_vectorized(use_t2)
    _div_vec = _flag_is_vectorized(use_div_bb)

    if _t2_vec or _div_vec:
        # Vectorized T2 path: compute both branches for all combos, then select.
        _div_on_t1 = _bool_fill(divergence_BB.shift(1)) | _bool_fill(divergence_BB.shift(2))
        _bo = np.asarray(high.gt(high.shift(1), axis=0), dtype=bool).reshape(-1, 1) \
            & np.asarray(_bool_fill(T1.shift(1)), dtype=bool)  # (time, combos)
        _div = np.asarray(_div_on_t1, dtype=bool)
        _idx, _cols = T1.index, T1.columns
        if _div_vec:
            _t2_arr = np.where(
                np.tile(_flag_array(use_div_bb), (len(T1), 1)), _bo & _div, _bo)
        else:
            _t2_arr = (_bo & _div) if bool(use_div_bb) else _bo
        if _t2_vec:
            _t2_mask = np.tile(_flag_array(use_t2), (len(T1), 1))
            _entry_arr = np.where(_t2_mask, _t2_arr, np.asarray(T1, dtype=bool))
            _price = np.broadcast_to(
                (high.shift(1) + _MINTICK).to_numpy().reshape(-1, 1), _t2_mask.shape)
            _close_2d = np.broadcast_to(
                np.asarray(close, dtype=float).reshape(-1, 1), _t2_mask.shape)
            entry_condition = pd.DataFrame(_entry_arr, index=_idx, columns=_cols)
            t2_entry_price = pd.DataFrame(
                np.where(_t2_mask, _price, _close_2d), index=_idx, columns=_cols)
        else:
            # use_t2 scalar + use_div_bb vectorized: honor scalar use_t2 semantics.
            if bool(use_t2):
                entry_condition = pd.DataFrame(_t2_arr, index=_idx, columns=_cols)
                t2_entry_price = pd.DataFrame(
                    np.broadcast_to(
                        (high.shift(1) + _MINTICK).to_numpy().reshape(-1, 1),
                        _t2_arr.shape), index=_idx, columns=_cols)
            else:
                entry_condition = _bool_fill(T1)
                t2_entry_price = None
    elif bool(use_t2):
        # divergence_BB evaluated on the T1 (setup) bar, i.e. shift(1) relative to trigger bar
        _div_on_t1 = _bool_fill(divergence_BB.shift(1)) | _bool_fill(divergence_BB.shift(2))
        _breakout = high.gt(high.shift(1), axis=0) & _bool_fill(T1.shift(1))
        T2 = (_breakout & _div_on_t1) if bool(use_div_bb) else _breakout
        entry_condition = _bool_fill(T2)
        # Entry price = High of T1 bar + mintick (stop-buy fill price)
        t2_entry_price = high.shift(1) + _MINTICK
    else:
        entry_condition = _bool_fill(T1)
        t2_entry_price = None

    # Exit Condition — combine all enabled exits
    exit_condition = (
        signals['sma_exit_signal'] |
        signals['sar_exit_signal'] |
        signals['macd_exit_signal'] |
        signals['cross_sar_sma_exit_signal']
    )
    # Optional exits (None when disabled)
    for key in ('retour_bb_exit_signal', 'regline_exit_signal', 'volat_down_exit_signal'):
        sig = signals.get(key)
        if sig is not None:
            exit_condition = exit_condition | sig
    exit_condition = _bool_fill(exit_condition)

    # ── SHORT SIGNALS ──────────────────────────────────────────────────────
    strategy_direction = signals.get('strategy_direction', 'long_only')
    short_entry_condition = None
    short_exit_condition  = None
    t2_short_entry_price  = None

    if strategy_direction in ('short_only', 'both'):
        lower_band = signals['lower_band']
        prev_lower = lower_band.shift(1)

        # T1 short: close crosses BELOW lower band
        crossunder_lower = lower_band.gt(close, axis=0) & prev_lower.le(prev_close, axis=0)
        _t0_cross_short = (T0 | _bool_fill(T0.shift(1))) & crossunder_lower
        _dep_short = signals.get('depassement_roc_short_signal')
        if _roc_vec:
            _roc_mask_s = np.tile(_flag_array(_roc), (len(_t0_cross_short), 1))
            if _dep_short is not None:
                _dep_s = np.asarray(_dep_short, dtype=bool)
                if _dep_s.ndim == 1:
                    _dep_s2d = np.broadcast_to(_dep_s.reshape(-1, 1), _roc_mask_s.shape)
                else:
                    _dep_s2d = _dep_s
                _t1s_arr = np.asarray(_t0_cross_short, dtype=bool) & np.where(_roc_mask_s, _dep_s2d, True)
            else:
                _t1s_arr = np.asarray(_t0_cross_short, dtype=bool)
            T1_short = pd.DataFrame(_t1s_arr, index=_t0_cross_short.index, columns=_t0_cross_short.columns)
        else:
            T1_short = (_t0_cross_short & _dep_short) if (bool(_roc) and _dep_short is not None) else _t0_cross_short

        # T2 short (optional): Low[t] < Low[t-1] after T1_short bar
        if _t2_vec or _div_vec:
            low = df['Low']
            _bo_s = np.asarray(low.lt(low.shift(1), axis=0), dtype=bool).reshape(-1, 1) \
                & np.asarray(_bool_fill(T1_short.shift(1)), dtype=bool)  # (time, combos)
            _div_s = np.asarray(_div_on_t1, dtype=bool)
            _idx_s, _cols_s = T1_short.index, T1_short.columns
            if _div_vec:
                _t2s_arr = np.where(
                    np.tile(_flag_array(use_div_bb), (len(T1_short), 1)), _bo_s & _div_s, _bo_s)
            else:
                _t2s_arr = (_bo_s & _div_s) if bool(use_div_bb) else _bo_s
            if _t2_vec:
                _t2_mask_s = np.tile(_flag_array(use_t2), (len(T1_short), 1))
                _sentry_arr = np.where(_t2_mask_s, _t2s_arr, np.asarray(T1_short, dtype=bool))
                _price_s = np.broadcast_to(
                    (low.shift(1) - _MINTICK).to_numpy().reshape(-1, 1), _t2_mask_s.shape)
                _close2d_s = np.broadcast_to(
                    np.asarray(close, dtype=float).reshape(-1, 1), _t2_mask_s.shape)
                short_entry_condition = pd.DataFrame(
                    _sentry_arr, index=_idx_s, columns=_cols_s)
                t2_short_entry_price = pd.DataFrame(
                    np.where(_t2_mask_s, _price_s, _close2d_s), index=_idx_s, columns=_cols_s)
            else:
                # use_t2 scalar + use_div_bb vectorized: honor scalar use_t2 semantics.
                if bool(use_t2):
                    short_entry_condition = pd.DataFrame(
                        _t2s_arr, index=_idx_s, columns=_cols_s)
                    t2_short_entry_price = pd.DataFrame(
                        np.broadcast_to(
                            (low.shift(1) - _MINTICK).to_numpy().reshape(-1, 1),
                            _t2s_arr.shape), index=_idx_s, columns=_cols_s)
                else:
                    short_entry_condition = _bool_fill(T1_short)
                    t2_short_entry_price = None
        elif bool(use_t2):
            low = df['Low']
            _breakout_short = low.lt(low.shift(1), axis=0) & _bool_fill(T1_short.shift(1))
            T2_short = (_breakout_short & _div_on_t1) if bool(use_div_bb) else _breakout_short
            short_entry_condition = _bool_fill(T2_short)
            t2_short_entry_price  = low.shift(1) - _MINTICK   # stop-sell fill
        else:
            short_entry_condition = _bool_fill(T1_short)

        # Short exit: combination of all enabled short exit signals
        _sma_s   = signals.get('sma_short_exit_signal')
        _sar_s   = signals.get('sar_short_exit_signal')
        _macd_s  = signals.get('macd_short_exit_signal')
        _xss_s   = signals.get('cross_sar_sma_short_exit_signal')

        # Start from a zero baseline with the SAME shape as T1_short (1-D scalar,
        # 2-D vectorized) so downstream ORs align column-wise.
        _false_base = _bool_fill(T1_short & ~T1_short)   # all-False
        short_exit_condition = _false_base
        if _sma_s is not None:
            short_exit_condition = short_exit_condition | _sma_s
        if _sar_s is not None:
            short_exit_condition = short_exit_condition | _sar_s
        if _macd_s is not None:
            short_exit_condition = short_exit_condition | _macd_s.astype(bool)
        if _xss_s is not None:
            short_exit_condition = short_exit_condition | _xss_s
        for key in ('retour_bb_short_exit_signal', 'regline_short_exit_signal',
                    'volat_up_exit_signal'):
            sig = signals.get(key)
            if sig is not None:
                short_exit_condition = short_exit_condition | sig
        short_exit_condition = _bool_fill(short_exit_condition)

    return entry_condition, exit_condition, t2_entry_price, \
           short_entry_condition, short_exit_condition, t2_short_entry_price


def run_backtest(df, params, timeframe='5s', return_portfolio=True, trade_start=0):
    """
    Run a backtest with the ATDMF strategy using given parameters.
    Supports both scalar parameters (single backtest) and lists/arrays (vectorized backtest).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        OHLCV DataFrame
    params : dict
        Strategy parameters
    timeframe : str, optional
        Timeframe of the data
    return_portfolio : bool, optional
        Whether to return the portfolio object or calculate metrics.
        If vectorized (multiple params) and return_portfolio=False, returns a DataFrame of metrics.
        
    Returns:
    --------
    object
        Portfolio object or performance metric(s)
    """
    # Generate signals (vectorized)
    try:
        signals = create_signal_generators(df, **params)
    except KeyError as e:
        raise ValueError(f"Missing parameter in create_signal_generators: {e}. Params keys: {list(params.keys())}")
    except Exception as e:
        raise RuntimeError(f"Error in create_signal_generators: {e}")
    
    # Create entry and exit conditions (vectorized)
    (entry_condition, exit_condition, t2_entry_price,
     short_entry_condition, short_exit_condition, t2_short_entry_price) = \
        create_entry_exit_conditions(df, signals)

    def _scalar_param(value, default):
        if hasattr(value, "__len__") and not isinstance(value, (str, bytes, dict)):
            arr = np.asarray(value).reshape(-1)
            if len(arr) == 0:
                return default
            value = arr[0]
        if isinstance(value, np.generic):
            return value.item()
        return value if value is not None else default

    order_sizing_mode = str(_scalar_param(params.get('order_sizing_mode', 'percent_equity'), 'percent_equity'))
    order_fixed_cash = float(_scalar_param(params.get('order_fixed_cash', 10000.0), 10000.0))
    fees_pct = float(_scalar_param(params.get('fees_pct', 0.0), 0.0))
    fees = fees_pct / 100.0
    slippage_bps = float(_scalar_param(params.get('slippage_bps', 0.0), 0.0))
    slippage = slippage_bps / 10000.0  # VBT expects a price fraction per fill

    # Create portfolio (vectorized)
    # from_signals automatically handles multi-column boolean dataframes
    if order_sizing_mode == 'fixed_cash':
        size = order_fixed_cash
        size_type = 'value'
    else:
        size = 1.0
        size_type = 'percent'

    # T2 mode: override execution price to High[t1] + mintick (stop-buy fill).
    # Avoid copying the full Close Series — allocate a numpy array and patch
    # only the T2 bars (typically <1% of rows), then wrap back as Series.
    if t2_entry_price is not None:
        mask = entry_condition.values if hasattr(entry_condition, 'values') else np.asarray(entry_condition)
        t2_vals = t2_entry_price.values if hasattr(t2_entry_price, 'values') else np.asarray(t2_entry_price)
        if mask.ndim == 2:
            # Vectorized: build a per-combination execution price matrix.
            _close_2d = np.broadcast_to(df['Close'].values.reshape(-1, 1), mask.shape)
            exec_price = pd.DataFrame(
                np.where(mask, t2_vals, _close_2d),
                index=df['Close'].index, columns=entry_condition.columns)
        else:
            close_vals = df['Close'].values.copy()  # numpy copy — no pandas overhead
            close_vals[mask] = t2_vals[mask]
            exec_price = pd.Series(close_vals, index=df['Close'].index)
    else:
        exec_price = df['Close']

    _direction = str(_scalar_param(params.get('strategy_direction', 'long_only'), 'long_only')).lower()

    # In short_only mode, suppress long entries to avoid accidental longs
    _entries = entry_condition if _direction in ('long_only', 'both') else \
        _bool_fill(entry_condition * False)
    _exits   = exit_condition  if _direction in ('long_only', 'both') else \
        _bool_fill(exit_condition * False)

    pf_kwargs = dict(
        close=df['Close'],
        entries=_entries,
        exits=_exits,
        price=exec_price,
        size=size,
        size_type=size_type,
        init_cash=order_fixed_cash,
        fees=fees,
        slippage=slippage,
        freq=timeframe,
    )

    if short_entry_condition is not None and short_exit_condition is not None:
        # Short execution price: patch exec_price on T2-short bars (Low[t1] - mintick).
        # VBT from_signals() has no 'short_price' arg — both long and short use 'price'.
        if t2_short_entry_price is not None:
            # exec_price may already be a patched Series (T2 long) or df['Close'].
            # Work on a numpy copy to avoid mutating the original.
            mask_s = short_entry_condition.values \
                if hasattr(short_entry_condition, 'values') else np.asarray(short_entry_condition)
            t2s_vals = t2_short_entry_price.values \
                if hasattr(t2_short_entry_price, 'values') else np.asarray(t2_short_entry_price)
            if mask_s.ndim == 2:
                if hasattr(exec_price, 'ndim') and exec_price.ndim == 2:
                    exec_vals = exec_price.values.copy()
                else:
                    exec_vals = np.broadcast_to(
                        np.asarray(exec_price, dtype=float).reshape(-1, 1), mask_s.shape).copy()
                exec_vals[mask_s] = t2s_vals[mask_s]
                pf_kwargs['price'] = pd.DataFrame(
                    exec_vals, index=df['Close'].index, columns=short_entry_condition.columns)
            else:
                exec_vals = exec_price.values.copy() \
                    if hasattr(exec_price, 'values') else np.array(exec_price, dtype=float)
                exec_vals[mask_s] = t2s_vals[mask_s]
                pf_kwargs['price'] = pd.Series(exec_vals, index=df['Close'].index)
        pf_kwargs['short_entries'] = short_entry_condition
        pf_kwargs['short_exits']   = short_exit_condition

    # Indicators see history; execution begins flat at the scoring boundary.
    if trade_start:
        if not 0 <= trade_start < len(df):
            raise ValueError("trade_start must point to a bar in df")
        for key, value in pf_kwargs.items():
            if isinstance(value, (pd.Series, pd.DataFrame)):
                pf_kwargs[key] = value.iloc[trade_start:]
    portfolio = vbt.Portfolio.from_signals(**pf_kwargs)
    
    if return_portfolio:
        return portfolio
    else:
        # Calculate performance metrics
        # If this is a vectorized run, we calculate metrics for all columns
        
        metric1_name = params.get('metric1_name', 'sharpe_ratio')
        metric2_name = params.get('metric2_name', 'total_return')
        weight_metric1 = params.get('weight_metric1', 1.0)
        weight_metric2 = params.get('weight_metric2', 0.0)
        
        # Helper to get metric safely
        def get_metric(port, name):
            """Compute a metric value for scalar or vectorized portfolios."""
            if name == 'max_drawdown':
                return port.max_drawdown * 100 * -1
            elif name == 'sharpe_ratio':
                from metrics import _sharpe_from_returns
                return _sharpe_from_returns(port, port.sharpe_ratio)
            elif name == 'total_return':
                return port.total_return * 100
            elif name == 'avg_gain_per_trade':
                return trade_stat(port.trades, 'avg_winning_trade')
            elif name == 'avg_loss_per_trade':
                return trade_stat(port.trades, 'avg_losing_trade') * -1
            elif name == 'win_rate':
                return port.trades.win_rate
            elif name == 'avg_pl_per_trade':
                # Custom calculation
                # For vectorized portfolio, this returns a Series
                total_ret = port.total_return * 100
                n_trades = port.trades.count()
                # Handle division by zero or no trades safely
                # n_trades may be scalar or a pandas Series (vectorized run).
                if hasattr(n_trades, 'replace'):
                    safe_trades = n_trades.replace(0, np.nan)
                    avg_pl = total_ret / safe_trades
                    return avg_pl.replace([np.inf, -np.inf], 0).fillna(0)

                try:
                    if float(n_trades) == 0.0:
                        return 0.0
                    avg_pl = total_ret / n_trades
                    if np.isinf(avg_pl) or np.isnan(avg_pl):
                        return 0.0
                    return avg_pl
                except Exception:
                    return 0.0
            elif name == 'pqs':
                _n_ref = int(_scalar_param(params.get('pqs_n_ref', 50), 50))
                return calc_pqs(port, n_ref=_n_ref)
            return 0.0

        m1 = get_metric(portfolio, metric1_name)
        m2 = get_metric(portfolio, metric2_name)
        
        combined_metric = (weight_metric1 * m1 + weight_metric2 * m2) / (weight_metric1 + weight_metric2)
        
        # If result is a Series (multiple params), return it. If scalar, return scalar.
        return combined_metric
