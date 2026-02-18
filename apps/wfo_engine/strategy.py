# Import necessary libraries for strategy
import pandas as pd
import numpy as np
import vectorbtpro as vbt
from metrics import trade_stat
from indicators import (
    EcartBollingerBorne, BollingerHorizontal,
    CrossBBWLowSignal, NbBarsUnderBBW, BBandCrossBarssince,
    DepassementRoCLong,
    SMAExit, ParabolicSAR, MACDExit,
    CrossSARSMAExit, PivotLow, LinregExit, VolatDownExit
)

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
        'use_t2_signal': True,
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
    use_t2_signal = p['use_t2_signal']
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
    high_price = df['High']
    low_price = df['Low']

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
    bbands = vbt.talib("BBANDS").run(
        close_price, 
        timeperiod=timeperiod, 
        nbdevup=StDev, 
        nbdevdn=StDev, 
        matype=matype,
        skipna=True # Good practice
    )
    upper_band = normalize_columns(bbands.upperband)
    lower_band = normalize_columns(bbands.lowerband)
    middle_band = normalize_columns(bbands.middleband)

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
    # Optimization: Calculate rolling stats using efficient Pandas/VBT backend
    # instead of Numba loop for median
    bbw = (upper_band - lower_band) / middle_band
    
    # Using vbt accessor for rolling if available, or pandas
    # bbands outputs are vbt-wrapped pandas objects
    # We explicitly access .vbt to ensure we get VBT functionality if needed, or just standard pandas
    # Standard pandas rolling is efficient enough compared to custom Numba loop
    mmbbw = bbw.rolling(window=5).mean()
    mediane_bbw = bbw.rolling(window=200).median()
    
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

    # SMA Exit
    # Needs: close, length
    # 1 input -> N params -> N outputs. per_column=False (default)
    sma_exit_ind = SMAExit.run(
        close=close_price_aligned,
        user_exit_sma_length=user_exit_sma_length,
        per_column=True
    )

    use_sma = (macd_ma_type == 'sma') if isinstance(macd_ma_type, str) else True
    macd_exit_ind = MACDExit.run(
        close=close_price_aligned,
        fast_length=macd_fast_length,
        slow_length=macd_slow_length,
        signal_length=macd_signal_length,
        use_type_a=exit_macd_type_a,
        use_type_b=exit_macd_type_b,
        use_sma=use_sma,
        per_column=True
    )

    # Parabolic SAR
    psar_ind = ParabolicSAR.run(
        high=high_price_aligned,
        low=low_price_aligned,
        sar_start=sar_start,
        sar_increment=sar_increment,
        sar_maximum=sar_maximum,
        per_column=True
    )

    sar_signal = normalize_columns(psar_ind.sar)
    prev_sar = sar_signal.shift(1)
    prev_close = close_price_aligned.shift(1)
    if hasattr(sar_signal, "columns"):
        sar_exit_arr = (
            prev_close.to_numpy() > prev_sar.to_numpy()
        ) & (
            sar_signal.to_numpy() > close_price_aligned.to_numpy()
        )
        sar_exit_signal = pd.DataFrame(
            sar_exit_arr,
            index=sar_signal.index,
            columns=sar_signal.columns
        ).fillna(False).astype(bool)
    else:
        sar_exit_signal = (
            (prev_close > prev_sar) &
            (sar_signal > close_price_aligned)
        ).fillna(False).astype(bool)

    macd_exit_signal = normalize_columns(macd_exit_ind.signal)
    if vector_len > 1 and is_array_like(exit_sar_enabled):
        exit_mask = pd.DataFrame(
            np.tile(np.asarray(exit_sar_enabled, dtype=bool), (len(sar_exit_signal), 1)),
            index=sar_exit_signal.index,
            columns=sar_exit_signal.columns
        )
        sar_exit_signal = sar_exit_signal & exit_mask
    else:
        if not bool(exit_sar_enabled):
            sar_exit_signal[:] = False

    if vector_len > 1 and is_array_like(exit_macd_enabled):
        exit_mask = pd.DataFrame(
            np.tile(np.asarray(exit_macd_enabled, dtype=bool), (len(macd_exit_signal), 1)),
            index=macd_exit_signal.index,
            columns=macd_exit_signal.columns
        )
        macd_exit_signal = macd_exit_signal & exit_mask
    else:
        if not bool(exit_macd_enabled):
            macd_exit_signal[:] = False

    # --- Additional exit signals (Phase 5) ---

    # 1. Cross SAR/SMA exit: SAR crosses above SMA
    # Use the same SMA length as the SMA exit for consistency
    _sma_len = int(scalarize(user_exit_sma_length)) if is_array_like(user_exit_sma_length) else int(user_exit_sma_length)
    sma_series = close_price_aligned.rolling(window=_sma_len, min_periods=_sma_len).mean()
    # NOTE: CrossSARSMAExit has no VBT params (param_names=[]), so per_column=True
    # is not valid.  With takes_1d=True, VBT already iterates over columns
    # automatically when inputs are DataFrames.
    cross_sar_sma_exit_ind = CrossSARSMAExit.run(
        sma=sma_series,
        sar=sar_signal,
    )
    cross_sar_sma_signal = normalize_columns(cross_sar_sma_exit_ind.signal).astype(bool)
    if not bool(exit_cross_sar_sma_enabled):
        cross_sar_sma_signal[:] = False

    # 2. Retour BB exit: pivot low on lower band
    retour_bb_signal = None
    if bool(exit_retour_bb_enabled):
        retour_bb_ind = PivotLow.run(
            series=lower_band,
            left=nb_bars_left_pivot,
            right=nb_bars_right_pivot,
            per_column=True
        )
        retour_bb_signal = normalize_columns(retour_bb_ind.signal).astype(bool)

    # 3. Regline exit: crossunder(close, linreg)
    regline_signal = None
    if bool(exit_regline_enabled):
        regline_ind = LinregExit.run(
            close=close_price_aligned,
            length=nombre_periodes_reglin,
            offset=i_bars_back,
            per_column=True
        )
        regline_signal = normalize_columns(regline_ind.signal).astype(bool)

    # 4. Volat down exit: crossunder(%BB, seuil_overbought)
    volat_down_signal = None
    if bool(exit_volat_down_enabled):
        volat_down_ind = VolatDownExit.run(
            close=close_price_aligned,
            upper=upper_band,
            lower=lower_band,
            seuil_overbought=seuil_overbought_bb,
            per_column=True
        )
        volat_down_signal = normalize_columns(volat_down_ind.signal).astype(bool)

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
        'use_t2_signal': use_t2_signal,
        'sma_exit_signal': normalize_columns(sma_exit_ind.signal),
        'sar_exit_signal': sar_exit_signal,
        'macd_exit_signal': macd_exit_signal.astype(bool),
        'cross_sar_sma_exit_signal': cross_sar_sma_signal,
        'retour_bb_exit_signal': retour_bb_signal,
        'regline_exit_signal': regline_signal,
        'volat_down_exit_signal': volat_down_signal,
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

    # T1: T0 (current or previous bar) + crossover + RoC filter
    T1 = (T0 | T0.shift(1).fillna(False)) & crossover_upper & signals['depassement_roc_signal']

    # T2: high breakout + T1 on previous bar + divergence_BB
    # divergence_BB = upper expanding AND lower expanding (bands diverging)
    high = df['High']
    lower_band = signals['lower_band']
    divergence_BB = (upper_band.diff() > 0) & (lower_band.diff() < 0)

    use_t2 = signals.get('use_t2_signal', True)
    if use_t2:
        T2 = (
            high.gt(high.shift(1), axis=0) &
            T1.shift(1).fillna(False) &
            (divergence_BB | divergence_BB.shift(1).fillna(False))
        )
        entry_condition = T2.fillna(False).astype(bool)
    else:
        entry_condition = T1.fillna(False).astype(bool)
    
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
    exit_condition = exit_condition.fillna(False).astype(bool)
    
    return entry_condition, exit_condition

def run_backtest(df, params, timeframe='5s', return_portfolio=True):
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
    entry_condition, exit_condition = create_entry_exit_conditions(df, signals)

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

    # Create portfolio (vectorized)
    # from_signals automatically handles multi-column boolean dataframes
    if order_sizing_mode == 'fixed_cash':
        size = order_fixed_cash
        size_type = 'value'
    else:
        size = 1.0
        size_type = 'percent'

    portfolio = vbt.Portfolio.from_signals(
        close=df['Close'],
        entries=entry_condition,
        exits=exit_condition,
        size=size,
        size_type=size_type,
        init_cash=10000,
        fees=fees,
        freq=timeframe
    )
    
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
                return port.sharpe_ratio
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
            return 0.0

        m1 = get_metric(portfolio, metric1_name)
        m2 = get_metric(portfolio, metric2_name)
        
        combined_metric = (weight_metric1 * m1 + weight_metric2 * m2) / (weight_metric1 + weight_metric2)
        
        # If result is a Series (multiple params), return it. If scalar, return scalar.
        return combined_metric
