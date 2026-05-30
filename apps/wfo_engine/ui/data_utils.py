"""Data and trade-table presentation helpers for the Streamlit UI."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _coerce_arrow_value(v):
    """Convert a single cell value to a type PyArrow can serialize.

    Priority:
      1. None / NaN-like         → keep as-is (Arrow handles None)
      2. bool (before int!)      → bool
      3. numpy integer           → int
      4. numpy floating          → float  (NaN/Inf → None)
      5. numpy bool_             → bool
      6. pd.Timedelta / Timestamp → str
      7. Any remaining non-native → str
    """
    if v is None:
        return v
    # numpy bool must come before numpy integer (bool_ is a subclass of integer)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(v, (pd.Timedelta, pd.Timestamp)):
        return str(v)
    # Plain Python scalars are fine (bytes decoded to avoid Arrow binary inference)
    if isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, bytes):
        return v.decode('utf-8', errors='replace')
    # Anything else (complex numpy types, custom objects, …) → str
    return str(v)


def arrow_safe_df(obj):
    """Return a DataFrame safe for Arrow/Streamlit serialization.

    Normalises every object-dtype column so that PyArrow can serialize it:
    numpy scalars → Python native, Timedelta/Timestamp → str, unknowns → str.

    When *obj* is a pd.Series (e.g. pf.trades.stats()), it is reshaped into a
    two-column DataFrame ["Metric", "Value"] first.
    """
    if isinstance(obj, pd.Series):
        df = obj.reset_index()
        df.columns = ["Metric", "Value"]
        # Value is always mixed (float, str, int, timedelta…) — force to homogeneous str
        # so PyArrow never has to infer a common type across incompatible Python scalars.
        df["Value"] = df["Value"].apply(
            lambda v: str(_coerce_arrow_value(v)) if v is not None else None
        )
        return df
    else:
        df = obj.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(_coerce_arrow_value)
    return df


def downsample_series(series, max_points: int = 20000):
    if series is None or len(series) <= max_points:
        return series
    step = max(1, len(series) // max_points)
    return series.iloc[::step]


def downsample_df(df, max_rows: int = 200000):
    if df is None or df.empty or len(df) <= max_rows:
        return df
    step = max(1, len(df) // max_rows)
    return df.iloc[::step]


def get_return_series(trades_df):
    if trades_df is None or trades_df.empty:
        return None

    if "return" in trades_df.columns:
        return pd.to_numeric(trades_df["return"], errors="coerce")

    if "pnl" in trades_df.columns and "entry_value" in trades_df.columns:
        denom = pd.to_numeric(trades_df["entry_value"], errors="coerce")
        pnl = pd.to_numeric(trades_df["pnl"], errors="coerce")
        return pnl / denom.replace(0, np.nan)

    if "pnl" in trades_df.columns and "entry_price" in trades_df.columns and "size" in trades_df.columns:
        denom = pd.to_numeric(trades_df["entry_price"], errors="coerce") * pd.to_numeric(trades_df["size"], errors="coerce")
        pnl = pd.to_numeric(trades_df["pnl"], errors="coerce")
        return pnl / denom.replace(0, np.nan)

    return None


def trimmed_mean(series, trim: float = 0.05):
    if series is None:
        return None
    s = series.dropna()
    if s.empty:
        return None
    lower = s.quantile(trim)
    upper = s.quantile(1 - trim)
    return s[(s >= lower) & (s <= upper)].mean()


def winsorized_mean(series, trim: float = 0.05):
    if series is None:
        return None
    s = series.dropna()
    if s.empty:
        return None
    lower = s.quantile(trim)
    upper = s.quantile(1 - trim)
    return s.clip(lower=lower, upper=upper).mean()


def compute_trade_pnl_metrics(trades_df, trim: float = 0.05) -> pd.DataFrame:
    """Compute P&L % summaries per trade for the trades table.

    Only percentage-based, directional metrics are kept — absolute $ values
    are sizing-dependent and not comparable across backtests.

    Retained metrics:
      - Mean P&L %         : arithmetic mean (primary estimator)
      - Median P&L %       : robust to outliers
      - Trimmed Mean P&L % : arithmetic mean after removing top/bottom trim %
      - Geometric Mean P&L %: compounding-correct estimator
    """
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    ret = get_return_series(trades_df)
    if ret is None:
        return pd.DataFrame()

    rows = []

    def _add(name, pct, pct_std, n):
        rows.append({"Metric": name, "P&L %": pct, "Std %": pct_std, "N trades": n})

    n_all = int(ret.dropna().shape[0])

    # Mean
    _add("Mean P&L %", ret.mean(), ret.std(ddof=0), n_all)

    # Median
    _add("Median P&L %", ret.median(), ret.std(ddof=0), n_all)

    # Trimmed mean
    trimmed = ret.dropna()
    if not trimmed.empty:
        lo, hi = trimmed.quantile(trim), trimmed.quantile(1 - trim)
        trimmed_vals = trimmed[(trimmed >= lo) & (trimmed <= hi)]
        _add(
            f"Trimmed Mean P&L % ({int(trim * 100)}%)",
            trimmed_mean(ret, trim),
            trimmed_vals.std(ddof=0) if not trimmed_vals.empty else None,
            int(trimmed_vals.shape[0]),
        )

    # Geometric mean (compounding-correct, excludes total-loss trades)
    ret_clean = ret.dropna()
    ret_clean = ret_clean[ret_clean > -1]
    if not ret_clean.empty:
        geo = np.expm1(np.log1p(ret_clean).mean())
        _add("Geometric Mean P&L %", geo, None, int(ret_clean.shape[0]))

    if not rows:
        return pd.DataFrame()

    df_metrics = pd.DataFrame(rows)
    df_metrics["P&L %"] = df_metrics["P&L %"].apply(
        lambda x: f"{x * 100:.4f}%" if pd.notna(x) else ""
    )
    df_metrics["Std %"] = df_metrics["Std %"].apply(
        lambda x: f"{x * 100:.4f}%" if pd.notna(x) else ""
    )
    return df_metrics
