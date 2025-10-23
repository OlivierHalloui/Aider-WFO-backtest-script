# Import necessary libraries for strategy
import pandas as pd
import numpy as np
import vectorbtpro as vbt
import talib
from .indicators import (
    bbands_1d_nb, ecart_bollinger_borne_signal_nb, bollinger_horizontal_signal_nb,
    cross_bbw_low_signal_nb, calculate_exit_sma_nb
)

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

def run_backtest(df, params, timeframe='5s', return_portfolio=True):
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
