# Import necessary libraries for strategy
import pandas as pd
import numpy as np
import vectorbtpro as vbt
from indicators import (
    EcartBollingerBorne, BollingerHorizontal,
    CrossBBWLowSignal, SMAExit
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
    # Extract parameters
    timeperiod = params.get('timeperiod', 20)
    StDev = params.get('StDev', 2.0)
    matype = params.get('matype', 0)
    coeff_medianeBBW = params.get('coeff_medianeBBW', 1.1)
    coef_mediane = params.get('coef_mediane', 1.0)
    Nb_bars_above = params.get('Nb_bars_above', 5)
    fenetre_lowest = params.get('fenetre_lowest', 30)
    seuil_lowest = params.get('seuil_lowest', 3.5)
    user_exit_sma_length = params.get('user_exit_sma_length', 20)
    
    # Vectorization handling
    # Check if any parameter is an array/list to determine if we are in a vectorized run
    vector_len = 1
    all_params = [timeperiod, StDev, matype, coeff_medianeBBW, coef_mediane, 
                  Nb_bars_above, fenetre_lowest, seuil_lowest, user_exit_sma_length]
    
    for p in all_params:
        if hasattr(p, '__len__') and not isinstance(p, str):
            vector_len = len(p)
            break
            
    def broadcast(val, length):
        if hasattr(val, '__len__') and not isinstance(val, str):
            return val
        return np.full(length, val)
        
    if vector_len > 1:
        timeperiod = broadcast(timeperiod, vector_len)
        StDev = broadcast(StDev, vector_len)
        matype = broadcast(matype, vector_len)
        coeff_medianeBBW = broadcast(coeff_medianeBBW, vector_len)
        coef_mediane = broadcast(coef_mediane, vector_len)
        Nb_bars_above = broadcast(Nb_bars_above, vector_len)
        fenetre_lowest = broadcast(fenetre_lowest, vector_len)
        seuil_lowest = broadcast(seuil_lowest, vector_len)
        user_exit_sma_length = broadcast(user_exit_sma_length, vector_len)

    # We use 'Close' for most calculations
    # VBT's run methods accept Series or DataFrame. 
    # If we pass the full DataFrame, we might need to specify the column or rely on broadcasting if it's 1D.
    # To be safe and explicit, we pass the specific Series.
    close_price = df['Close']
    
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
    
    # 2. Custom Indicators (Vectorized via indicators.py factories)
    
    # Ecart Bollinger Borne
    # Needs: price, upper, lower, timeperiod, longueur_mediane, coef_mediane, Nb_bars_above
    # N inputs -> N params -> N outputs (1-to-1). per_column=True
    longueur_mediane = params.get('longueur_mediane', 100)
    nb_bars_above_ind = EcartBollingerBorne.run(
        prix=close_price,
        upper_band=bbands.upperband,
        lower_band=bbands.lowerband,
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
    bbw = (bbands.upperband - bbands.lowerband) / bbands.middleband
    
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
        upper_band=bbands.upperband,
        lower_band=bbands.lowerband,
        middle_band=bbands.middleband,
        fenetre_lowest=fenetre_lowest,
        seuil_lowest=seuil_lowest,
        per_column=True
    )
    
    # SMA Exit
    # Needs: close, length
    # 1 input -> N params -> N outputs. per_column=False (default)
    sma_exit_ind = SMAExit.run(
        close=close_price,
        user_exit_sma_length=user_exit_sma_length
    )
    
    def clean_cols(obj):
        """Helper to replace complex MultiIndex columns with simple RangeIndex for alignment."""
        if hasattr(obj, 'columns'):
            obj = obj.copy()
            obj.columns = pd.RangeIndex(len(obj.columns))
        return obj

    return {
        'upper_band': clean_cols(bbands.upperband),
        'lower_band': clean_cols(bbands.lowerband),
        'middle_band': clean_cols(bbands.middleband),
        'nb_bars_above_signal': clean_cols(nb_bars_above_ind.signal),
        'bollinger_horizontal_signal': clean_cols(bollinger_horizontal_ind.signal).astype(bool),
        'cross_bbw_low_signal': clean_cols(cross_bbw_low_ind.signal),
        'sma_exit_signal': clean_cols(sma_exit_ind.signal)
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
    
    entry_condition = (
        signals['nb_bars_above_signal'] & 
        signals['cross_bbw_low_signal'] & 
        signals['bollinger_horizontal_signal'] & 
        upper_band.lt(close, axis=0) & 
        prev_upper.gt(prev_close, axis=0)
    ).fillna(False).astype(bool)
    
    # Exit Condition
    exit_condition = signals['sma_exit_signal'].fillna(False).astype(bool)
    
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
    
    # Create portfolio (vectorized)
    # from_signals automatically handles multi-column boolean dataframes
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
        # Calculate performance metrics
        # If this is a vectorized run, we calculate metrics for all columns
        
        metric1_name = params.get('metric1_name', 'sharpe_ratio')
        metric2_name = params.get('metric2_name', 'total_return')
        weight_metric1 = params.get('weight_metric1', 1.0)
        weight_metric2 = params.get('weight_metric2', 0.0)
        
        # Helper to get metric safely
        def get_metric(port, name):
            if name == 'max_drawdown':
                return port.max_drawdown * 100 * -1
            elif name == 'sharpe_ratio':
                return port.sharpe_ratio
            elif name == 'total_return':
                return port.total_return * 100
            elif name == 'avg_gain_per_trade':
                return port.trades.avg_winning_trade
            elif name == 'avg_loss_per_trade':
                return port.trades.avg_losing_trade * -1
            elif name == 'win_rate':
                return port.trades.win_rate
            elif name == 'avg_pl_per_trade':
                # Custom calculation
                # For vectorized portfolio, this returns a Series
                total_ret = port.total_return * 100
                n_trades = port.trades.count()
                # Handle division by zero or no trades safely
                # n_trades can be 0.
                if n_trades == 0:
                    return 0.0
                avg_pl = total_ret / n_trades
                
                if hasattr(avg_pl, 'replace'):
                    avg_pl = avg_pl.replace([np.inf, -np.inf], 0).fillna(0)
                else:
                    if np.isinf(avg_pl) or np.isnan(avg_pl):
                        avg_pl = 0.0
                return avg_pl
            return 0.0

        m1 = get_metric(portfolio, metric1_name)
        m2 = get_metric(portfolio, metric2_name)
        
        combined_metric = (weight_metric1 * m1 + weight_metric2 * m2) / (weight_metric1 + weight_metric2)
        
        # If result is a Series (multiple params), return it. If scalar, return scalar.
        return combined_metric