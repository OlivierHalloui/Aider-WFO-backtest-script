"""Data and trade-table presentation helpers for the Streamlit UI."""

from __future__ import annotations

import numpy as np
import pandas as pd


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
    """Compute robust P&L summaries (value and percent) for the trades table."""
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    pnl = pd.to_numeric(trades_df.get("pnl"), errors="coerce") if "pnl" in trades_df.columns else None
    ret = get_return_series(trades_df)

    rows = []

    def add_row(name, value=None, pct=None, value_std=None, pct_std=None, n_trades=None):
        rows.append(
            {
                "Metric": name,
                "P&L Value": value,
                "P&L %": pct,
                "Std Value": value_std,
                "Std %": pct_std,
                "n trades": n_trades,
            }
        )

    if pnl is not None:
        add_row("Mean P&L", pnl.mean(), None, pnl.std(ddof=0), None, pnl.dropna().shape[0])
        add_row("Median P&L", pnl.median(), None, pnl.std(ddof=0), None, pnl.dropna().shape[0])
        trimmed = pnl.dropna()
        if not trimmed.empty:
            lower = trimmed.quantile(trim)
            upper = trimmed.quantile(1 - trim)
            trimmed_vals = trimmed[(trimmed >= lower) & (trimmed <= upper)]
        else:
            trimmed_vals = trimmed
        add_row(
            f"Trimmed Mean P&L ({int(trim*100)}%)",
            trimmed_mean(pnl, trim),
            None,
            trimmed_vals.std(ddof=0) if not trimmed_vals.empty else None,
            None,
            trimmed_vals.dropna().shape[0] if trimmed_vals is not None else None,
        )
        wins_vals = pnl.dropna()
        if not wins_vals.empty:
            lower = wins_vals.quantile(trim)
            upper = wins_vals.quantile(1 - trim)
            wins_vals = wins_vals.clip(lower=lower, upper=upper)
        add_row(
            f"Winsorized Mean P&L ({int(trim*100)}%)",
            winsorized_mean(pnl, trim),
            None,
            wins_vals.std(ddof=0) if not wins_vals.empty else None,
            None,
            wins_vals.dropna().shape[0] if wins_vals is not None else None,
        )
        abs_pnl = pnl.abs()
        add_row("Mean |P&L|", abs_pnl.mean(), None, abs_pnl.std(ddof=0), None, abs_pnl.dropna().shape[0])

        if "size" in trades_df.columns:
            size = pd.to_numeric(trades_df["size"], errors="coerce")
            per_unit = pnl / size.replace(0, np.nan)
            add_row("Mean P&L per Unit", per_unit.mean(), None, per_unit.std(ddof=0), None, per_unit.dropna().shape[0])

    if ret is not None:
        add_row("Mean P&L %", None, ret.mean(), None, ret.std(ddof=0), ret.dropna().shape[0])
        add_row("Median P&L %", None, ret.median(), None, ret.std(ddof=0), ret.dropna().shape[0])
        trimmed_ret = ret.dropna()
        if not trimmed_ret.empty:
            lower = trimmed_ret.quantile(trim)
            upper = trimmed_ret.quantile(1 - trim)
            trimmed_ret_vals = trimmed_ret[(trimmed_ret >= lower) & (trimmed_ret <= upper)]
        else:
            trimmed_ret_vals = trimmed_ret
        add_row(
            f"Trimmed Mean P&L % ({int(trim*100)}%)",
            None,
            trimmed_mean(ret, trim),
            None,
            trimmed_ret_vals.std(ddof=0) if not trimmed_ret_vals.empty else None,
            trimmed_ret_vals.dropna().shape[0] if trimmed_ret_vals is not None else None,
        )
        wins_ret = ret.dropna()
        if not wins_ret.empty:
            lower = wins_ret.quantile(trim)
            upper = wins_ret.quantile(1 - trim)
            wins_ret = wins_ret.clip(lower=lower, upper=upper)
        add_row(
            f"Winsorized Mean P&L % ({int(trim*100)}%)",
            None,
            winsorized_mean(ret, trim),
            None,
            wins_ret.std(ddof=0) if not wins_ret.empty else None,
            wins_ret.dropna().shape[0] if wins_ret is not None else None,
        )
        abs_ret = ret.abs()
        add_row("Mean |P&L %|", None, abs_ret.mean(), None, abs_ret.std(ddof=0), abs_ret.dropna().shape[0])

        ret_clean = ret.dropna()
        ret_clean = ret_clean[ret_clean > -1]
        if not ret_clean.empty:
            log_mean = np.log1p(ret_clean).mean()
            geo_mean = np.expm1(log_mean)
            add_row("Geometric Mean P&L %", None, geo_mean, None, None, ret_clean.dropna().shape[0])

    if not rows:
        return pd.DataFrame()

    df_metrics = pd.DataFrame(rows)
    if "P&L Value" in df_metrics.columns:
        df_metrics["P&L Value"] = df_metrics["P&L Value"].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
    if "P&L %" in df_metrics.columns:
        df_metrics["P&L %"] = df_metrics["P&L %"].apply(lambda x: f"{x * 100:.3f}%" if pd.notna(x) else "")
    if "Std Value" in df_metrics.columns:
        df_metrics["Std Value"] = df_metrics["Std Value"].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
    if "Std %" in df_metrics.columns:
        df_metrics["Std %"] = df_metrics["Std %"].apply(lambda x: f"{x * 100:.3f}%" if pd.notna(x) else "")
    if "n trades" in df_metrics.columns:
        df_metrics["n trades"] = df_metrics["n trades"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "")
    return df_metrics
