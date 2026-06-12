"""Canonical metric helpers shared across WFO and adaptive optimization engines.

This module consolidates duplicated utility functions that were previously
defined in both wfo.py and adaptive_optimization.py.  Downstream modules
should import from here; the original private copies will be deprecated.
"""

import numpy as np
import pandas as pd


def safe_float(value, default=0.0):
    """Convert *value* to a plain Python float, returning *default* on failure.

    Handles numpy scalars, booleans, NaN and Inf gracefully.
    """
    try:
        if isinstance(value, (bool, np.bool_)):
            return 1.0 if value else 0.0
        if isinstance(value, (np.integer, int, np.floating, float)):
            value = float(value)
            if np.isnan(value) or np.isinf(value):
                return float(default)
            return value
        return float(default)
    except (ValueError, TypeError, OverflowError):
        return float(default)


def to_scalar_score(score):
    """Reduce *score* to a single finite float.

    If *score* is already a scalar it is returned directly (NaN/Inf → ``np.nan``).
    For array-like inputs the finite mean is returned.
    """
    if score is None:
        return np.nan
    if np.isscalar(score):
        try:
            out = float(score)
            if np.isnan(out) or np.isinf(out):
                return np.nan
            return out
        except Exception:
            return np.nan
    try:
        if hasattr(score, "values"):
            arr = np.asarray(score.values, dtype=float)
        else:
            arr = np.asarray(score, dtype=float)
        if arr.size == 0:
            return np.nan
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan
        return float(np.mean(arr))
    except Exception:
        return np.nan


def python_scalar(value):
    """Convert a numpy scalar to the corresponding native Python type."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _get_trades_stats(trades) -> dict:
    """Compute ``trades.stats()`` once and return the result dict.

    Callers that extract multiple statistics from the same trades object should
    call this once, then pass the result as ``_stats_cache`` to every
    ``trade_stat()`` call, avoiding repeated dict rebuilds.
    """
    try:
        return trades.stats() or {}
    except Exception:
        return {}


def trade_stat(trades, attr_name, default=0.0, _stats_cache=None):
    """Retrieve a named statistic from a VectorBT ``Trades`` object.

    Tries direct attribute access first, then falls back to the full
    ``trades.stats()`` dict with several key-casing variants.

    Parameters
    ----------
    _stats_cache : dict, optional
        Pre-built result of ``trades.stats()``.  When provided the
        ``trades.stats()`` call is skipped entirely.
    """
    value = getattr(trades, attr_name, None)
    if value is not None:
        return value
    if _stats_cache is None:
        _stats_cache = _get_trades_stats(trades)
    keys = [
        attr_name,
        attr_name.replace("_", " "),
        attr_name.replace("_", " ").title(),
        attr_name.replace("_", " ").capitalize(),
    ]
    for key in keys:
        try:
            value = _stats_cache.get(key) if hasattr(_stats_cache, "get") else _stats_cache[key]
        except Exception:
            value = None
        if value is not None:
            return value
    return default


def calc_avg_pl(portfolio):
    """Return average P/L per trade for scalar or vectorized portfolios."""
    try:
        total_ret = portfolio.total_return * 100
        n_trades = portfolio.trades.count()
        if n_trades is None:
            return 0.0

        # Vectorized (Series) path
        if hasattr(n_trades, "replace"):
            safe_trades = n_trades.replace(0, np.nan)
            avg_pl = total_ret / safe_trades
            avg_pl = avg_pl.replace([np.inf, -np.inf], 0).fillna(0)
            return avg_pl

        # Scalar path
        if float(n_trades) == 0.0:
            return 0.0
        avg_pl = total_ret / n_trades
        if hasattr(avg_pl, "replace"):
            avg_pl = avg_pl.replace([np.inf, -np.inf], 0).fillna(0)
        else:
            if np.isinf(avg_pl) or np.isnan(avg_pl):
                avg_pl = 0.0
        return avg_pl
    except Exception:
        return 0.0


def calc_pqs(portfolio, n_ref: int = 50) -> float:
    """Profit Quality Score with statistical confidence weighting.

    Formula:
        PQS = (Return% / |MaxDD%|) × AvgP&L/tr% × √(n_trades / n_ref)

    The √(n_trades / n_ref) factor rewards strategies with sufficient trade
    frequency and penalises those with very few trades even if AvgP&L is high,
    preventing the optimiser from converging on hyper-selective configs.

    Parameters
    ----------
    portfolio : VectorBT portfolio object
    n_ref : int
        Reference trade count for the confidence factor (default 50).
        Factor = 1.0 at exactly n_ref trades; < 1.0 below, > 1.0 above.

    Returns 0.0 when MaxDD or n_trades is zero.
    """
    try:
        ret = float(to_scalar_score(getattr(portfolio, "total_return", 0.0) * 100))
        dd = float(to_scalar_score(getattr(portfolio, "max_drawdown", 0.0) * 100))
        abs_dd = abs(dd)
        if abs_dd == 0.0:
            return 0.0
        avg_pl = float(to_scalar_score(calc_avg_pl(portfolio)))
        try:
            n_trades = int(len(portfolio.trades))
        except Exception:
            n_trades = 0
        if n_trades == 0:
            return 0.0
        confidence = np.sqrt(max(n_trades, 0) / max(n_ref, 1))
        return (ret / abs_dd) * avg_pl * confidence
    except Exception:
        return 0.0


def portfolio_metrics(portfolio, window_id, n_ref: int = 50):
    """Build a summary dict of key performance metrics for *portfolio*.

    Parameters
    ----------
    portfolio : VectorBT portfolio object
    window_id : str or int
        Identifier for the WFO window / adaptive cycle.
    """
    try:
        n_trades = len(portfolio.trades)
    except Exception:
        n_trades = 0
    return {
        "window": window_id,
        "return": float(to_scalar_score(getattr(portfolio, "total_return", 0.0) * 100)),
        "sharpe": float(to_scalar_score(getattr(portfolio, "sharpe_ratio", 0.0))),
        "max_drawdown": float(to_scalar_score(getattr(portfolio, "max_drawdown", 0.0) * 100)),
        "win_rate": float(to_scalar_score(getattr(getattr(portfolio, "trades", object()), "win_rate", 0.0))),
        "avg_gain_per_trade": float(to_scalar_score(trade_stat(portfolio.trades, "avg_winning_trade"))),
        "avg_loss_per_trade": float(to_scalar_score(trade_stat(portfolio.trades, "avg_losing_trade"))),
        "avg_pl_per_trade": float(to_scalar_score(calc_avg_pl(portfolio))),
        "calmar_ratio": float(to_scalar_score(getattr(portfolio, "calmar_ratio", 0.0))),
        "sortino_ratio": float(to_scalar_score(getattr(portfolio, "sortino_ratio", 0.0))),
        "pqs": calc_pqs(portfolio, n_ref=n_ref),
        "n_trades": int(n_trades),
    }


# ---------------------------------------------------------------------------
# Results DataFrame builders (pure data, no UI dependencies)
# ---------------------------------------------------------------------------

def build_trials_dataframe_from_results(results):
    """Build a flat DataFrame of all optimization trials across WFO windows."""
    rows = []
    if not isinstance(results, dict):
        return pd.DataFrame()
    for window_result in results.get("window_results", []):
        if not isinstance(window_result, dict):
            continue
        info = window_result.get("window_info", {}) or {}
        window_id = info.get("window")
        trials = window_result.get("optimization_trials") or []
        trial_source = "optimization_trials"
        if not isinstance(trials, list) or len(trials) == 0:
            trials = window_result.get("optimization_results") or []
            trial_source = "optimization_results"
        for idx, trial in enumerate(trials):
            if not isinstance(trial, dict):
                continue
            row = dict(trial)
            row["window"] = window_id
            row["trial_rank_in_window"] = idx + 1
            row["trial_source"] = trial_source
            row["in_sample_start"] = info.get("in_sample_start")
            row["in_sample_end"] = info.get("in_sample_end")
            row["out_sample_start"] = info.get("out_sample_start")
            row["out_sample_end"] = info.get("out_sample_end")
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_window_info_dataframe(results):
    """Build a summary DataFrame with one row per WFO window."""
    rows = []
    if not isinstance(results, dict):
        return pd.DataFrame()
    for window_result in results.get("window_results", []):
        if not isinstance(window_result, dict):
            continue
        info = window_result.get("window_info", {}) or {}
        row = dict(info)
        row["optimization_trials_count"] = int(window_result.get("optimization_trials_count", 0))
        row["evaluations"] = int(window_result.get("evaluations", 0))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
