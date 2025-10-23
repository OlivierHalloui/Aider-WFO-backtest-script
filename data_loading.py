# Import necessary libraries for data loading
import pandas as pd
import vectorbtpro as vbt
import os
from config import DEFAULT_DATA_FILE

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

def load_data(start_date, end_date, timeframe='5s', from_file=True, file_path=None):
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
    if from_file:
        file_path = file_path or DEFAULT_DATA_FILE
        dtypes = {
            'Open time': 'str',  # Will convert to datetime later
            'Open': 'float64',
            'High': 'float64',
            'Low': 'float64',
            'Close': 'float64',
            'Volume': 'float64' if 'Volume' in pd.read_csv(file_path, nrows=1).columns else None
        }
        df = pd.read_csv(file_path, dtype=dtypes)
        df['Open time'] = pd.to_datetime(df['Open time'], errors='coerce')
        df.set_index('Open time', inplace=True)
        # Resample efficiently
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
