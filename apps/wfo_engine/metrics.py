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
        stats = trades.stats()
        # NB: never use `stats or {}` — pandas raises on the truth value of a
        # Series, and the exception silently collapsed this to an empty dict,
        # making every statistic lookup fall back to its default (e.g. 0.0).
        return {} if stats is None else stats
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
        # VectorBT marks percentage statistics with a [%] suffix
        f"{attr_name.replace('_', ' ').title()} [%]",
        f"{attr_name.replace('_', ' ').capitalize()} [%]",
    ]
    for key in keys:
        try:
            value = _stats_cache.get(key) if hasattr(_stats_cache, "get") else _stats_cache[key]
        except Exception:
            value = None
        if value is not None:
            return value
    return default


def _trade_returns(trades):
    """Per-trade returns of *trades* as a clean numeric array (1-D or 2-D)."""
    import numpy as np

    rets = getattr(trades, "returns", None)
    if rets is None:
        return None
    if callable(rets):
        rets = rets()
    rets = getattr(rets, "values", rets)
    try:
        rets = np.asarray(rets, dtype=float)
    except Exception:
        return None
    return rets if rets.size else None


def calc_avg_pl(portfolio):
    """Mean P&L per trade **in %** = mean of the per-trade returns.

    This is the average of each trade's percentage return (``trades.returns``),
    NOT ``total_return / n_trades``: the total return is compounded across
    trades, so dividing it by the trade count does not give a per-trade mean
    and silently mixes the % return with the raw P&L scale.
    """
    try:
        rets = _trade_returns(portfolio.trades)
        if rets is None:
            return 0.0
        if rets.ndim > 1:
            import pandas as pd

            avg = np.nanmean(rets, axis=0) * 100.0
            if avg.size == 1:
                return float(avg[0])
            return pd.Series(avg)
        rets = rets[np.isfinite(rets)]
        if rets.size == 0:
            return 0.0
        return float(rets.mean() * 100.0)
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


def _get_returns(portfolio):
    """Extract the per-bar returns of *portfolio* as a clean 1-D finite array.

    VectorBT exposes ``returns`` as a property or a method depending on the
    build; NaN bars (leading/interior) are dropped. Returns None when no usable
    return series is available (e.g. mocked portfolios in tests).
    """
    try:
        import numpy as np

        rets = getattr(portfolio, "returns", None)
        if callable(rets):
            rets = rets()  # defensive: some VectorBT builds expose returns() as a method
        rets = np.asarray(rets, dtype=float)
        if rets.ndim > 1:
            rets = rets[:, 0]  # first column; never pool columns (they are distinct assets)
        rets = rets.ravel()
        rets = rets[np.isfinite(rets)]
        return rets if rets.size > 1 else None
    except Exception:
        return None


def _sharpe_from_returns(portfolio, fallback):
    """Non-annualised Sharpe = mean / std of the window's per-bar returns.

    VectorBT's ``sharpe_ratio`` annualises with the bar frequency: on 5s data
    that multiplies by ``sqrt(periods/year)`` ~ 2500 and turns a raw ratio of
    0.05 into absurd figures (33-125). The raw mean/std is the honest,
    window-comparable number. Falls back to the portfolio's own ratio when
    returns are unavailable (e.g. mocked portfolios in tests).
    """
    try:
        rets = _get_returns(portfolio)
        if rets is None:
            return fallback
        std = rets.std(ddof=1)  # sample std (ddof=1), the usual Sharpe convention
        if std > 0:
            return float(rets.mean() / std)
        return float("nan")  # zero variance: undefined, do NOT leak the annualised fallback
    except Exception:
        return fallback


def _sortino_from_returns(portfolio, fallback):
    """Non-annualised Sortino = mean / downside deviation of per-bar returns.

    Same de-annualisation rationale as `_sharpe_from_returns`; the downside
    deviation uses the root-mean-square of negative returns (target 0).
    """
    try:
        import numpy as np

        rets = _get_returns(portfolio)
        if rets is None:
            return fallback
        downside = np.minimum(rets, 0.0)
        ds_std = float(np.sqrt(np.mean(downside ** 2)))
        if ds_std > 0:
            return float(rets.mean() / ds_std)
        return float("nan")  # no downside: undefined, avoid the annualised fallback
    except Exception:
        return fallback


def _calmar_from_returns(portfolio, fallback):
    """Calmar = total return / max drawdown of the window (no annualisation)."""
    try:
        import numpy as np

        rets = _get_returns(portfolio)
        if rets is None:
            return fallback
        cum = np.cumprod(1.0 + rets)
        peak = np.maximum.accumulate(cum)
        max_dd = float(np.max((peak - cum) / peak))
        total_ret = float(cum[-1] - 1.0)
        if max_dd > 0:
            return float(total_ret / max_dd)
        return float("nan")  # no drawdown: undefined, avoid the annualised fallback
    except Exception:
        return fallback


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
        "sharpe": float(to_scalar_score(_sharpe_from_returns(
            portfolio, getattr(portfolio, "sharpe_ratio", 0.0)))),
        "max_drawdown": float(to_scalar_score(getattr(portfolio, "max_drawdown", 0.0) * 100)),
        "win_rate": float(to_scalar_score(getattr(getattr(portfolio, "trades", object()), "win_rate", 0.0))),
        "avg_gain_per_trade": float(to_scalar_score(trade_stat(portfolio.trades, "avg_winning_trade"))),
        "avg_loss_per_trade": float(to_scalar_score(trade_stat(portfolio.trades, "avg_losing_trade"))),
        "avg_pl_per_trade": float(to_scalar_score(calc_avg_pl(portfolio))),
        "calmar_ratio": float(to_scalar_score(_calmar_from_returns(portfolio, getattr(portfolio, "calmar_ratio", 0.0)))),
        "sortino_ratio": float(to_scalar_score(_sortino_from_returns(portfolio, getattr(portfolio, "sortino_ratio", 0.0)))),
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
