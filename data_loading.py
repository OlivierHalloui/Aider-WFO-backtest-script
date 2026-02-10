# Import necessary libraries for data loading
import pandas as pd
import vectorbtpro as vbt
import os
from pathlib import Path
from config import DEFAULT_DATA_FILE, DEFAULT_START_DATE, DEFAULT_END_DATE

# ======================================================================
# DATA LOADING AND PREPROCESSING
# ======================================================================

def get_dates(config):
    """Prompt for date range input with defaults."""
    start_date = config.get('start_date', DEFAULT_START_DATE)
    end_date = config.get('end_date', DEFAULT_END_DATE)

    return start_date, end_date

def _normalize_date_range(start_date, end_date):
    """Normalize string/ts bounds into pandas timestamps suitable for index filtering."""
    start_ts = pd.to_datetime(start_date, errors='coerce') if start_date else None
    end_ts = pd.to_datetime(end_date, errors='coerce') if end_date else None

    if start_ts is not None and pd.isna(start_ts):
        start_ts = None
    if end_ts is not None and pd.isna(end_ts):
        end_ts = None

    # If user provided date-only (YYYY-MM-DD), include the full end day.
    if end_ts is not None and isinstance(end_date, str) and len(end_date) <= 10:
        end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    return start_ts, end_ts

def _apply_date_filter(df, start_date, end_date):
    """Filter a datetime-indexed frame between inclusive start/end bounds."""
    if df is None or df.empty:
        return df

    start_ts, end_ts = _normalize_date_range(start_date, end_date)
    index_tz = getattr(df.index, "tz", None)

    if start_ts is not None:
        if index_tz is not None:
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize(index_tz)
            else:
                start_ts = start_ts.tz_convert(index_tz)
        elif start_ts.tzinfo is not None:
            start_ts = start_ts.tz_localize(None)

    if end_ts is not None:
        if index_tz is not None:
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize(index_tz)
            else:
                end_ts = end_ts.tz_convert(index_tz)
        elif end_ts.tzinfo is not None:
            end_ts = end_ts.tz_localize(None)

    if start_ts is not None:
        df = df[df.index >= start_ts]
    if end_ts is not None:
        df = df[df.index <= end_ts]
    return df

def get_csv_date_range(file_path):
    """Read a CSV and return min/max date strings detected in its timestamp column."""
    if not file_path or not os.path.exists(file_path):
        return None, None

    try:
        try:
            dates_df = pd.read_csv(file_path, usecols=['Open time'])
            date_col = 'Open time'
        except ValueError:
            dates_df = pd.read_csv(file_path, nrows=1)
            if dates_df.columns.empty:
                return None, None
            date_col = dates_df.columns[0]
            dates_df = pd.read_csv(file_path, usecols=[date_col])

        series = pd.to_datetime(dates_df[date_col], errors='coerce')
        series = series.dropna()
        if series.empty:
            return None, None

        min_date = series.min().date().isoformat()
        max_date = series.max().date().isoformat()
        return min_date, max_date
    except Exception:
        return None, None

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
        df = _apply_date_filter(df, start_date, end_date)
        print(df)

    else:
        # Fetch from Binance
        base_timeframe = '1s'  # Fetch at 1s resolution
        data_obj = vbt.BinanceData.fetch(
            ["BTCUSDT"], 
            start=start_date, 
            end=end_date,
            timeframe=base_timeframe
        )
        # Extract DataFrame from the wrapper object
        df_1s = data_obj.get()
        
        # Resample to desired timeframe
        df = df_1s.resample(timeframe).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna()
        df = _apply_date_filter(df, start_date, end_date)
        # Save data for later use
        folder_name = f"Binance_BTCUSDT_OHLCV_B_{start_date}_{end_date}_{timeframe}"
        base_dir = Path(__file__).resolve().parent
        folder_path = base_dir / "Data" / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        file_path = folder_path / f"Binance_BTCUSDT_OHLCV_B_{start_date}_{end_date}_{timeframe}.csv"
        df.to_csv(file_path)
        print(f"Saved Binance data to {file_path}")
        
    return df
