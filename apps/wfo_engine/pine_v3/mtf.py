"""MTF helpers for Pine V3 runtime (`request.security`-style alignment)."""

from __future__ import annotations

from typing import Callable, Literal

import pandas as pd
import vectorbtpro as vbt


TimingMode = Literal["opening", "closing"]
GapsMode = Literal["on", "off"]


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Downsample OHLCV using trading-standard aggregation rules."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input OHLCV dataframe is empty or invalid.")
    if "Open" not in df.columns or "High" not in df.columns or "Low" not in df.columns or "Close" not in df.columns:
        raise ValueError("OHLCV dataframe must contain Open/High/Low/Close columns.")

    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    return df.resample(timeframe).agg(agg).dropna(subset=["Open", "High", "Low", "Close"])


def _to_ffill(gaps: GapsMode | str | bool | None) -> bool:
    """
    Convert Pine gaps mode to forward-fill behavior.

    Pine-style interpretation:
    - gaps='off' => propagate latest known value => ffill=True
    - gaps='on'  => keep sparse updates only       => ffill=False
    """
    if isinstance(gaps, bool):
        return bool(gaps)
    mode = str(gaps or "off").strip().lower()
    if mode in {"on", "gaps_on"}:
        return False
    return True


def realign_series_to_base(
    series: pd.Series | pd.DataFrame,
    base_index: pd.Index,
    *,
    timing: TimingMode = "closing",
    gaps: GapsMode | str | bool | None = "off",
    source_freq: str | None = None,
    target_freq: str | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Align higher-timeframe series onto base index without lookahead bias.

    - `timing='closing'`: use latest value available at bar close (high/low/close/close-based indicators)
    - `timing='opening'`: value known at bar open (open/open-based indicators)
    """
    if not isinstance(base_index, pd.Index) or len(base_index) == 0:
        raise ValueError("base_index must be a non-empty pandas index.")
    if not isinstance(series, (pd.Series, pd.DataFrame)) or len(series) == 0:
        raise ValueError("series must be a non-empty pandas Series/DataFrame.")

    src = series.sort_index()
    target_index = pd.DatetimeIndex(base_index)
    src_freq = source_freq if isinstance(source_freq, str) and source_freq.strip() else False
    tgt_freq = target_freq if isinstance(target_freq, str) and target_freq.strip() else False
    resampler = vbt.Resampler(
        source_index=src.index,
        target_index=target_index,
        source_freq=src_freq,
        target_freq=tgt_freq,
    )

    ffill = _to_ffill(gaps)
    timing_norm = str(timing).strip().lower()
    if timing_norm == "opening":
        aligned = src.vbt.realign_opening(resampler, ffill=ffill)
    else:
        aligned = src.vbt.realign_closing(resampler, ffill=ffill)
    return aligned


def request_security_series(
    base_ohlcv: pd.DataFrame,
    *,
    timeframe: str,
    expr_fn: Callable[[pd.DataFrame], pd.Series | pd.DataFrame],
    timing: TimingMode = "closing",
    gaps: GapsMode | str | bool | None = "off",
    base_freq: str | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Pine-style `request.security` helper for one expression on a higher timeframe.

    Workflow:
    1. Downsample base OHLCV to `timeframe`
    2. Evaluate `expr_fn` on HTF data
    3. Realign result back to base index with opening/closing semantics
    """
    htf_ohlcv = resample_ohlcv(base_ohlcv, timeframe=timeframe)
    expr = expr_fn(htf_ohlcv)
    if not isinstance(expr, (pd.Series, pd.DataFrame)):
        raise ValueError("expr_fn must return a pandas Series or DataFrame.")
    return realign_series_to_base(
        expr,
        base_index=base_ohlcv.index,
        timing=timing,
        gaps=gaps,
        source_freq=timeframe,
        target_freq=base_freq,
    )


def request_security_signal(
    base_ohlcv: pd.DataFrame,
    *,
    timeframe: str,
    signal_fn: Callable[[pd.DataFrame], pd.Series | pd.DataFrame],
    timing: TimingMode = "closing",
    gaps: GapsMode | str | bool | None = "on",
    base_freq: str | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Convenience wrapper for boolean/event signals (sparse by default).

    Uses `gaps='on'` by default to avoid repeated signals after upsampling.
    """
    out = request_security_series(
        base_ohlcv=base_ohlcv,
        timeframe=timeframe,
        expr_fn=signal_fn,
        timing=timing,
        gaps=gaps,
        base_freq=base_freq,
    )
    if isinstance(out, pd.Series):
        return out.fillna(False).astype(bool)
    return out.fillna(False).astype(bool)
