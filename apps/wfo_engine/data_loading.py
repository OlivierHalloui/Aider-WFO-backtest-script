# Import necessary libraries for data loading
import logging
import os
from pathlib import Path

import pandas as pd
import vectorbtpro as vbt

from config import DEFAULT_DATA_FILE, DEFAULT_START_DATE, DEFAULT_END_DATE

logger = logging.getLogger(__name__)

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

def _to_pandas_freq(timeframe: str) -> str:
    """Shared frequency adapter for data loading and Pine V3 MTF (pine_v3.mtf).

    Pine distinguishes 'm' (minute) from 'M' (calendar month); pandas uses
    'M' or 'ME' (depending on its version) for month-end. Fixed-length units
    are converted to seconds or whole calendar days before reaching pandas.
    Unknown frequencies are passed through for pandas to validate.
    """
    tf = str(timeframe).strip()
    if tf.endswith('M'):
        try:
            n = int(tf[:-1])
        except ValueError:
            pass
        else:
            # pandas < 2.2 supports 'M'; newer pandas uses 'ME'. Never
            # lowercase Pine's monthly suffix into a one-minute frequency.
            try:
                pd.tseries.frequencies.to_offset('ME')
                return f'{n}ME'
            except ValueError:
                return f'{n}M'

    tf_lower = tf.lower()
    for suffix, factor in {'s': 1, 'm': 60, 'h': 3600}.items():
        if tf_lower.endswith(suffix):
            try:
                n = int(tf_lower[:-1])
            except ValueError:
                break
            total_s = n * factor
            if total_s >= 86400 and total_s % 86400 == 0:
                return f'{total_s // 86400}D'
            return f'{total_s}s'

    for suffix, pandas_suffix in (('d', 'D'), ('w', 'W')):
        if tf_lower.endswith(suffix):
            try:
                return f'{int(tf_lower[:-1])}{pandas_suffix}'
            except ValueError:
                break
    return timeframe   # fallback: pass through unchanged


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

def _read_csv_period(file_path, usecols, start_date, end_date, warmup_bars):
    """Read selected CSV rows while preserving its header; fallback on bad dates."""
    if isinstance(warmup_bars, bool) or not isinstance(warmup_bars, int) or warmup_bars < 0:
        raise ValueError("warmup_bars must be a non-negative integer")
    read_kwargs = dict(usecols=usecols, dtype={'Open time': 'str'})
    try:
        start_ts, end_ts = _normalize_date_range(start_date, end_date)
        if start_ts is None or end_ts is None:
            raise ValueError("both valid date bounds are required")
        # Scan timestamps only: OHLCV data outside the period is never loaded.
        dates = pd.to_datetime(
            pd.read_csv(file_path, usecols=['Open time'], dtype={'Open time': 'str'})['Open time'],
            errors='raise',
        )
        if dates.empty or dates.isna().any() or not dates.is_monotonic_increasing:
            raise ValueError("CSV timestamps are empty, invalid or unsorted")
        if dates.dt.tz is not None:
            start_ts = start_ts.tz_localize(dates.dt.tz) if start_ts.tzinfo is None else start_ts.tz_convert(dates.dt.tz)
            end_ts = end_ts.tz_localize(dates.dt.tz) if end_ts.tzinfo is None else end_ts.tz_convert(dates.dt.tz)
        elif start_ts.tzinfo is not None:
            start_ts = start_ts.tz_localize(None)
            end_ts = end_ts.tz_localize(None)
        first = int(dates.searchsorted(start_ts, side='left'))
        last = int(dates.searchsorted(end_ts, side='right'))
        if first >= last:
            raise ValueError("requested period is not in CSV")
        row_start = max(0, first - warmup_bars)
        return pd.read_csv(
            file_path, skiprows=range(1, row_start + 1), nrows=last - row_start, **read_kwargs,
        ), dates.iloc[row_start] if row_start < first else None
    except (ValueError, TypeError, OverflowError, KeyError, OSError) as exc:
        logger.warning("CSV filtered read failed (%s); falling back to full read", exc)
        return pd.read_csv(file_path, **read_kwargs), None

def load_data(start_date, end_date, timeframe='5s', from_file=True, file_path=None, warmup_bars: int = 0):
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
    warmup_bars : int, optional
        Number of source CSV rows preceding start_date (file source only).
        
    Returns:
    --------
    pandas.DataFrame
        OHLCV data
    """
    if from_file:
        file_path = file_path or DEFAULT_DATA_FILE
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        header_cols = pd.read_csv(file_path, nrows=0).columns.tolist()
        required_cols = ['Open time', 'Open', 'High', 'Low', 'Close']
        missing = [c for c in required_cols if c not in header_cols]
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        usecols = required_cols + (['Volume'] if 'Volume' in header_cols else [])

        # R10 — don't use dtype= so pandas doesn't raise ValueError on 'N/A' or
        # other non-numeric strings; coerce to numeric after loading instead.
        df, warmup_start = _read_csv_period(file_path, usecols, start_date, end_date, warmup_bars)
        numeric_cols = ['Open', 'High', 'Low', 'Close'] + (['Volume'] if 'Volume' in usecols else [])
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        nan_count = df[['Open', 'High', 'Low', 'Close']].isna().sum().sum()
        if nan_count > 0:
            logger.warning("CSV: %d non-numeric values coerced to NaN in OHLC columns", nan_count)
        df['Open time'] = pd.to_datetime(df['Open time'], errors='coerce')
        df.set_index('Open time', inplace=True)
        df = df[~df.index.isna()]
        # Resample efficiently — convert to pandas-compatible frequency first
        df = df.resample(_to_pandas_freq(timeframe)).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna()
        df = _apply_date_filter(df, warmup_start if warmup_start is not None else start_date, end_date)

    else:
        # Fetch from Binance
        # api.binance.com may be geo-blocked; use api1.binance.com as fallback.
        base_timeframe = '1s'  # Fetch at 1s resolution
        try:
            data_obj = vbt.BinanceData.fetch(
                ["BTCUSDT"],
                start=start_date,
                end=end_date,
                timeframe=base_timeframe,
                client_config=dict(base_endpoint='1'),
            )
        except Exception as exc:
            logger.error("Binance fetch failed: %s", exc, exc_info=True)
            raise RuntimeError(
                f"Impossible de charger les données Binance. "
                f"Vérifier la connectivité et le geo-block (api1.binance.com). "
                f"Erreur: {exc}"
            ) from exc
        # Extract DataFrame from the VBT symbol_dict wrapper — R11: guard missing key
        df_1s = data_obj.data.get('BTCUSDT')
        if df_1s is None or df_1s.empty:
            raise RuntimeError(
                "Binance fetch returned no data for BTCUSDT. "
                "Check date range, connectivity, and geo-block."
            )

        # Save raw 1s data immediately — before any resample that could fail
        base_dir = Path(__file__).resolve().parent
        raw_folder = base_dir / "Data" / f"Binance_BTCUSDT_OHLCV_B_{start_date}_{end_date}_1s"
        raw_folder.mkdir(parents=True, exist_ok=True)
        raw_path = raw_folder / f"Binance_BTCUSDT_OHLCV_B_{start_date}_{end_date}_1s.csv"
        df_1s.to_csv(raw_path)

        # Resample to desired timeframe
        df = df_1s.resample(_to_pandas_freq(timeframe)).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna()
        df = _apply_date_filter(df, start_date, end_date)
        # Save resampled data
        folder_name = f"Binance_BTCUSDT_OHLCV_B_{start_date}_{end_date}_{timeframe}"
        folder_path = base_dir / "Data" / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        file_path = folder_path / f"Binance_BTCUSDT_OHLCV_B_{start_date}_{end_date}_{timeframe}.csv"
        df.to_csv(file_path)
        
    return df
