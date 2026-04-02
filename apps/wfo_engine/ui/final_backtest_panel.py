"""Final backtest and parameter selection functions extracted from app.py.

These are standalone copies of the param-selection and final-backtest helpers.
Some functions reference private helpers that still live in ``app.py``
(e.g. ``get_current_config``, ``resolve_strategy_adapter``, ``load_data``).
The caller must supply or monkey-patch those dependencies when wiring
this module back into the Streamlit application.
"""

import os

import numpy as np
import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from config import DEFAULT_PARAM_GRID, DEFAULT_TIMEFRAME
from metrics import portfolio_metrics
from domain.serialization import (
    sanitize_for_json as _sanitize_for_json,
    to_jsonable as _to_jsonable,
)
from ui.data_utils import arrow_safe_df as _arrow_safe_df


# ---------------------------------------------------------------------------
# Metric resolution — single source of truth for all metric name → dict key
# mappings. Used by _select_best_params_from_results and the window
# comparison score builder. Add new metrics here only.
# ---------------------------------------------------------------------------

def _resolve_metric_value(row: dict, name: str):
    """Return the numeric value for *name* from a performance metrics row.

    Returns None when the row is empty or the key is absent.
    Metrics where lower is better (max_drawdown, avg_loss_per_trade) are
    negated so that callers can always maximise the returned value.
    """
    if not row:
        return None
    if name == 'max_drawdown':
        v = row.get('max_drawdown')
        return None if v is None else -v
    if name == 'sharpe_ratio':
        return row.get('sharpe')
    if name == 'total_return':
        return row.get('return')
    if name == 'win_rate':
        return row.get('win_rate')
    if name == 'avg_gain_per_trade':
        return row.get('avg_gain_per_trade')
    if name == 'avg_loss_per_trade':
        v = row.get('avg_loss_per_trade')
        return None if v is None else -v
    if name == 'avg_pl_per_trade':
        return row.get('avg_pl_per_trade')
    if name == 'pqs':
        return row.get('pqs')
    if name == 'calmar_ratio':
        return row.get('calmar_ratio')
    if name == 'sortino_ratio':
        return row.get('sortino_ratio')
    return None


# ---------------------------------------------------------------------------
# Parameter selection helpers
# ---------------------------------------------------------------------------

def _select_best_params_from_results(results, config):
    metric1_name = config.get('metric1_name', 'sharpe_ratio')
    metric2_name = config.get('metric2_name', 'total_return')
    weight_metric1 = float(config.get('weight_metric1', 1.0))
    weight_metric2 = float(config.get('weight_metric2', 0.0))

    def combined_score(row):
        total_weight = weight_metric1 + weight_metric2
        if total_weight == 0:
            return None
        m1 = _resolve_metric_value(row, metric1_name)
        m2 = _resolve_metric_value(row, metric2_name)
        if weight_metric1 != 0 and m1 is None:
            return None
        if weight_metric2 != 0 and m2 is None:
            return None
        if m1 is None:
            m1 = 0.0
        if m2 is None:
            m2 = 0.0
        return (weight_metric1 * m1 + weight_metric2 * m2) / total_weight

    best_params = None
    best_score = None
    best_window = None
    best_is_metrics = None
    best_oos_metrics = None

    is_map = {row.get('window'): row for row in results.get('in_sample_performance', [])}
    oos_map = {row.get('window'): row for row in results.get('out_of_sample_performance', [])}

    for window in results.get('window_results', []):
        window_id = window.get('window_info', {}).get('window')
        if window_id is None:
            continue
        is_row = is_map.get(window_id)
        oos_row = oos_map.get(window_id)
        is_score = combined_score(is_row)
        oos_score = combined_score(oos_row)

        if is_score is None and oos_score is None:
            continue
        if is_score is None:
            window_score = oos_score
        elif oos_score is None:
            window_score = is_score
        else:
            window_score = (is_score + oos_score) / 2

        if best_score is None or window_score > best_score:
            best_score = window_score
            best_params = (window.get('best_params') or {}).copy()
            best_window = window_id
            best_is_metrics = is_row
            best_oos_metrics = oos_row

    return best_params, best_score, best_window, best_is_metrics, best_oos_metrics


def _get_params_for_window(results, window_id):
    """Return (best_params, is_metrics, oos_metrics) for a specific WFO window."""
    is_map  = {row.get('window'): row for row in results.get('in_sample_performance',  [])}
    oos_map = {row.get('window'): row for row in results.get('out_of_sample_performance', [])}
    for window in results.get('window_results', []):
        wid = window.get('window_info', {}).get('window')
        if wid == window_id:
            return (
                (window.get('best_params') or {}).copy(),
                is_map.get(window_id),
                oos_map.get(window_id),
            )
    return None, None, None


def _normalize_vote_value(value):
    """Normalize heterogeneous trial values for robust voting/aggregation."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return ""
        s_l = s.lower()
        if s_l in {"true", "1", "yes", "y"}:
            return True
        if s_l in {"false", "0", "no", "n"}:
            return False
        try:
            f = float(s)
            if np.isfinite(f):
                if abs(f - round(f)) < 1e-9:
                    return int(round(f))
                return float(f)
        except Exception:
            return s
        return s
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if not np.isfinite(f):
            return None
        if abs(f - round(f)) < 1e-9:
            return int(round(f))
        return f
    return value


def _weighted_median(values, weights):
    """Compute weighted median for numeric vectors."""
    if len(values) == 0:
        return None
    vals = np.asarray(values, dtype=float)
    wts = np.asarray(weights, dtype=float)
    mask = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
    vals = vals[mask]
    wts = wts[mask]
    if vals.size == 0:
        return None
    order = np.argsort(vals)
    vals = vals[order]
    wts = wts[order]
    cumsum = np.cumsum(wts)
    cutoff = 0.5 * float(np.sum(wts))
    idx = int(np.searchsorted(cumsum, cutoff, side="left"))
    idx = max(0, min(idx, len(vals) - 1))
    return float(vals[idx])


def _build_robust_set_summary(results, config):
    """
    Build level-1 robust set summary:
    Top-N in each window + weighted vote/median across windows.
    """
    enabled = bool(config.get("robust_tests_enabled", False)) if isinstance(config, dict) else False
    top_n = max(1, int(config.get("robust_top_n_per_window", 20) or 20)) if isinstance(config, dict) else 20
    min_windows = max(1, int(config.get("robust_min_windows", 3) or 3)) if isinstance(config, dict) else 3
    use_for_final = bool(config.get("robust_use_for_final_backtest", False)) if isinstance(config, dict) else False
    selected_params = [str(p) for p in (config.get("selected_params") or [])] if isinstance(config, dict) else []

    summary = {
        "schema_version": "robust_set.v1",
        "enabled": enabled,
        "use_for_final_backtest": use_for_final,
        "method": "top_n_weighted_vote",
        "top_n_per_window": int(top_n),
        "min_windows_required": int(min_windows),
        "status": "disabled" if not enabled else "pending",
        "windows_total": int(len(results.get("window_results", []) or [])) if isinstance(results, dict) else 0,
        "windows_used": [],
        "candidates_total": 0,
        "robust_params": {},
        "support_by_param": [],
        "notes": [],
    }

    if not enabled:
        summary["notes"].append("Robust set desactive dans la configuration.")
        return summary
    if not isinstance(results, dict):
        summary["status"] = "invalid_input"
        summary["notes"].append("Resultats WFO invalides.")
        return summary

    pool_rows = []
    windows_with_trials = []
    score_col = "combined_score"

    for wr in results.get("window_results", []) or []:
        if not isinstance(wr, dict):
            continue
        info = wr.get("window_info", {}) or {}
        window_id = info.get("window")
        trials = wr.get("optimization_trials") or wr.get("optimization_results") or []
        if not isinstance(trials, list) or not trials:
            continue
        tdf = pd.DataFrame(trials)
        if tdf.empty or score_col not in tdf.columns:
            continue
        tdf[score_col] = pd.to_numeric(tdf[score_col], errors="coerce")
        tdf = tdf.replace([np.inf, -np.inf], np.nan).dropna(subset=[score_col])
        if tdf.empty:
            continue
        tdf = tdf.sort_values(score_col, ascending=False).head(top_n).reset_index(drop=True)
        windows_with_trials.append(window_id)
        for rank_idx, (_, row) in enumerate(tdf.iterrows(), start=1):
            item = row.to_dict()
            # Higher weight for higher rank in each window.
            item["_weight"] = float(top_n - rank_idx + 1)
            item["_window"] = window_id
            item["_rank"] = rank_idx
            pool_rows.append(item)

    if not pool_rows:
        summary["status"] = "no_trials"
        summary["notes"].append("Aucun trial exploitable trouve dans les fenetres WFO.")
        return summary

    pool_df = pd.DataFrame(pool_rows)
    windows_used = sorted(list({w for w in windows_with_trials if w is not None}))
    summary["windows_used"] = windows_used
    summary["candidates_total"] = int(len(pool_df))
    if len(windows_used) < min_windows:
        summary["status"] = "insufficient_windows"
        summary["notes"].append(
            f"Fenetres exploitables insuffisantes ({len(windows_used)}/{min_windows})."
        )
    else:
        summary["status"] = "ok"

    strategy_param_names = set(DEFAULT_PARAM_GRID.keys()) | {
        "exit_sar_enabled",
        "exit_macd_enabled",
        "exit_macd_type_a",
        "exit_macd_type_b",
        "order_sizing_mode",
        "order_fixed_cash",
        "fees_pct",
    }
    candidate_cols = [c for c in pool_df.columns if c in strategy_param_names]
    if selected_params:
        ordered = [p for p in selected_params if p in candidate_cols]
        extra = [p for p in candidate_cols if p not in ordered]
        candidate_cols = ordered + extra

    support_rows = []
    robust_params = {}
    for param in candidate_cols:
        sub = pool_df[[param, "_weight", "_window"]].copy()
        sub = sub.dropna(subset=[param])
        if sub.empty:
            continue
        sub["norm_value"] = sub[param].map(_normalize_vote_value)
        sub = sub.dropna(subset=["norm_value"])
        if sub.empty:
            continue

        numeric_vals = pd.to_numeric(sub["norm_value"], errors="coerce")
        is_numeric = numeric_vals.notna().all() and sub["norm_value"].map(lambda x: isinstance(x, (int, float, np.integer, np.floating))).all()

        if is_numeric:
            values = numeric_vals.astype(float).to_numpy()
            weights = pd.to_numeric(sub["_weight"], errors="coerce").fillna(0).to_numpy(dtype=float)
            med = _weighted_median(values, weights)
            observed_unique = np.unique(values[np.isfinite(values)])
            if med is None or observed_unique.size == 0:
                continue
            # Keep value on observed grid to avoid out-of-domain drift.
            chosen = float(observed_unique[int(np.argmin(np.abs(observed_unique - med)))])
            if abs(chosen - round(chosen)) < 1e-9:
                chosen = int(round(chosen))
            chosen_token = _normalize_vote_value(chosen)
        else:
            weight_by_value = (
                sub.groupby("norm_value")["_weight"]
                .sum()
                .sort_values(ascending=False)
            )
            if weight_by_value.empty:
                continue
            chosen_token = _normalize_vote_value(weight_by_value.index[0])
            chosen = chosen_token

        total_weight = float(pd.to_numeric(sub["_weight"], errors="coerce").fillna(0).sum())
        selected_mask = sub["norm_value"].map(_normalize_vote_value) == chosen_token
        selected_weight = float(pd.to_numeric(sub.loc[selected_mask, "_weight"], errors="coerce").fillna(0).sum())
        support_ratio = (selected_weight / total_weight) if total_weight > 0 else 0.0
        windows_hit = sorted(list({w for w in sub.loc[selected_mask, "_window"].tolist() if w is not None}))

        robust_params[param] = _to_jsonable(chosen)
        support_rows.append(
            {
                "parameter": param,
                "selected_value": _to_jsonable(chosen),
                "support_ratio": float(support_ratio),
                "weighted_support": float(selected_weight),
                "weighted_total": float(total_weight),
                "windows_covered": int(len(windows_hit)),
                "distinct_values": int(sub["norm_value"].nunique(dropna=True)),
            }
        )

    summary["robust_params"] = _sanitize_for_json(robust_params)
    summary["support_by_param"] = _sanitize_for_json(support_rows)
    if summary["status"] == "ok" and not robust_params:
        summary["status"] = "insufficient_data"
        summary["notes"].append("Impossible de construire des parametres robustes a partir des Top-N.")
    return summary


def _select_final_params_from_results(results, config):
    """Select final params using classic best-window or robust-set mode with safe fallback."""
    best_params, best_score, best_window, best_is_metrics, best_oos_metrics = _select_best_params_from_results(results, config)
    robust_summary = _build_robust_set_summary(results, config)
    use_robust = bool(config.get("robust_tests_enabled", False)) and bool(
        config.get("robust_use_for_final_backtest", False)
    )
    robust_params = robust_summary.get("robust_params") if isinstance(robust_summary, dict) else None

    if use_robust and isinstance(robust_params, dict) and robust_params and robust_summary.get("status") == "ok":
        return (
            robust_params.copy(),
            None,
            None,
            None,
            None,
            "robust_set",
            robust_summary,
        )
    return (
        best_params,
        best_score,
        best_window,
        best_is_metrics,
        best_oos_metrics,
        "best_window",
        robust_summary,
    )


# ---------------------------------------------------------------------------
# Load best params into sidebar inputs
# ---------------------------------------------------------------------------

def load_best_params_into_inputs(*, get_current_config, window_id=None):
    """Load best/robust params from WFO results into sidebar input widgets.

    Parameters
    ----------
    window_id : int or None
        If None (default) the global best window (or robust set) is used.
        If set, the best params of that specific WFO window are loaded.
    """
    final_params = None
    final_source = None
    loaded_window = None

    if 'wfo_results' in st.session_state:
        results = st.session_state['wfo_results']

        if window_id is not None:
            # Load from a specific window chosen by the user
            wp, is_m, oos_m = _get_params_for_window(results, window_id)
            if wp:
                final_params = wp
                final_source = f"window_{window_id}"
                loaded_window = window_id
                st.session_state['final_params']            = wp
                st.session_state['final_params_score']      = None
                st.session_state['final_params_window']     = window_id
                st.session_state['final_params_is_metrics'] = is_m
                st.session_state['final_params_oos_metrics']= oos_m
                st.session_state['final_params_source']     = final_source
            else:
                st.sidebar.error(f"Aucun paramètre trouvé pour la fenêtre {window_id}.")
                return
        else:
            # Default: global best window / robust set
            best_params, best_score, best_window, best_is, best_oos, final_source, robust_summary = (
                _select_final_params_from_results(results, get_current_config())
            )
            if best_params:
                st.session_state['final_params']              = best_params
                st.session_state['final_params_score']        = best_score
                st.session_state['final_params_window']       = best_window
                st.session_state['final_params_is_metrics']   = best_is
                st.session_state['final_params_oos_metrics']  = best_oos
                st.session_state['final_params_source']       = final_source
                st.session_state['final_params_robust_summary'] = robust_summary
                final_params = best_params
                loaded_window = best_window
                final_source = final_source or "best_window"

    if final_params is None:
        final_params  = st.session_state.get("final_params")
        final_source  = st.session_state.get("final_params_source")
        loaded_window = st.session_state.get("final_params_window")
    if not final_params:
        st.sidebar.error("No final parameters available.")
        return

    int_params = {
        'timeperiod', 'fenetre_lowest', 'longueur_mediane', 'Nb_bars_above', 'user_exit_sma_length',
        'macd_fast_length', 'macd_slow_length', 'macd_signal_length',
        'nb_bars_under_bbw_mini', 'nb_bars_entre_bb',
    }
    for param in DEFAULT_PARAM_GRID:
        st.session_state[f"check_{param}"] = False

    for param, value in final_params.items():
        if param not in DEFAULT_PARAM_GRID:
            continue
        try:
            if param in int_params:
                value = int(round(float(value)))
            else:
                value = float(value)
        except Exception:
            continue

        st.session_state[f"check_{param}"] = True
        st.session_state[f"min_{param}"] = value
        st.session_state[f"max_{param}"] = value
        # Use the canonical grid step so expanding the range later gives natural increments
        _d_min, _d_max, _d_step = DEFAULT_PARAM_GRID[param]
        st.session_state[f"step_{param}"] = _d_step

    if str(final_source or "").lower() == "robust_set":
        st.sidebar.success("Paramètres robust-set chargés dans les inputs.")
    elif loaded_window is not None:
        st.sidebar.success(f"Paramètres de la fenêtre {loaded_window} chargés.")


# ---------------------------------------------------------------------------
# Run final backtest
# ---------------------------------------------------------------------------

def run_final_backtest_logic(*, get_current_config, load_data, resolve_strategy_adapter):
    """Runs the final backtest using averaged parameters."""
    if 'wfo_results' not in st.session_state or 'df' not in st.session_state:
        st.error("No WFO results available to run final backtest.")
        return

    results = st.session_state['wfo_results']
    config = get_current_config()
    final_start_date = st.session_state.get('final_start_date', config.get('start_date'))
    final_end_date = st.session_state.get('final_end_date', config.get('end_date'))
    final_file_path = st.session_state.get('final_file_path', config.get('file_path'))

    with st.spinner("Loading data for final backtest..."):
        if config.get('from_file'):
            df = load_data(
                final_start_date,
                final_end_date,
                config.get('timeframe', DEFAULT_TIMEFRAME),
                from_file=True,
                file_path=final_file_path
            )
        else:
            df = load_data(
                final_start_date,
                final_end_date,
                config.get('timeframe', DEFAULT_TIMEFRAME),
                from_file=False
            )

    if df is None or df.empty:
        st.error("No data loaded for the final backtest range.")
        return
    st.session_state['final_backtest_df'] = df

    # Use either classic best-window params or robust-set params (if enabled).
    (
        chosen_params,
        best_score,
        best_window,
        best_is_metrics,
        best_oos_metrics,
        final_source,
        robust_summary,
    ) = _select_final_params_from_results(results, config)
    if not chosen_params:
        st.error("No valid parameters found for final backtest.")
        return
    if (
        bool(config.get("robust_tests_enabled", False))
        and bool(config.get("robust_use_for_final_backtest", False))
        and str(final_source or "").lower() != "robust_set"
    ):
        st.warning(
            "Robust Set demande mais non applicable sur ce run. "
            "Fallback automatique vers la selection classique best_window."
        )

    # Define integer parameters that should be rounded
    int_params = {
        'timeperiod', 'fenetre_lowest', 'longueur_mediane', 'Nb_bars_above', 'user_exit_sma_length',
        'macd_fast_length', 'macd_slow_length', 'macd_signal_length',
        'nb_bars_under_bbw_mini', 'nb_bars_entre_bb',
    }
    for param in list(chosen_params.keys()):
        if param in int_params:
            try:
                chosen_params[param] = int(round(float(chosen_params[param])))
            except Exception:
                pass
        elif isinstance(chosen_params[param], float):
            chosen_params[param] = round(chosen_params[param], 2)

    st.session_state['final_params'] = chosen_params
    st.session_state['final_params_score'] = best_score
    st.session_state['final_params_window'] = best_window
    st.session_state['final_params_is_metrics'] = best_is_metrics
    st.session_state['final_params_oos_metrics'] = best_oos_metrics
    st.session_state['final_params_source'] = final_source
    st.session_state['final_params_robust_summary'] = robust_summary

    def _combined_score_local(row):
        _m1_name = config.get('metric1_name', 'sharpe_ratio')
        _m2_name = config.get('metric2_name', 'total_return')
        _w1 = float(config.get('weight_metric1', 1.0))
        _w2 = float(config.get('weight_metric2', 0.0))
        total_weight = _w1 + _w2
        if total_weight == 0:
            return None
        m1 = _resolve_metric_value(row, _m1_name)
        m2 = _resolve_metric_value(row, _m2_name)
        if _w1 != 0 and m1 is None:
            return None
        if _w2 != 0 and m2 is None:
            return None
        return (_w1 * (m1 or 0.0) + _w2 * (m2 or 0.0)) / total_weight

    st.session_state['final_params_is_score'] = _combined_score_local(best_is_metrics)
    st.session_state['final_params_oos_score'] = _combined_score_local(best_oos_metrics)

    # Inject execution settings into params for the final backtest
    chosen_params['order_sizing_mode'] = config.get('order_sizing_mode', 'percent_equity')
    chosen_params['order_fixed_cash'] = float(config.get('order_fixed_cash', 10000.0))
    chosen_params['fees_pct'] = float(config.get('fees_pct', 0.0))

    with st.spinner("Running Final Backtest on Full Dataset..."):
        try:
            strategy_adapter = resolve_strategy_adapter(
                strategy_mode=config.get("strategy_mode"),
                strategy_id=config.get("strategy_id"),
                config=config,
            )
            final_portfolio = strategy_adapter.run_backtest(
                df,
                chosen_params,
                config.get('timeframe', DEFAULT_TIMEFRAME),
                return_portfolio=True
            )
            st.session_state['final_portfolio'] = final_portfolio
            st.success("Final Backtest Complete!")
            if str(final_source or "").lower() == "robust_set":
                st.info("Final backtest executed with robust-set parameters (Top-N vote multi-fenetres).")
        except Exception as e:
            st.error(f"Error in final backtest: {e}")


# ---------------------------------------------------------------------------
# Window comparison panel
# ---------------------------------------------------------------------------

# Metrics shown in the comparison table (display label → metrics dict key)
_CMP_METRICS = {
    "PQS": "pqs",
    "Sharpe": "sharpe",
    "Return %": "return",
    "Max DD %": "max_drawdown",
    "Win Rate %": "win_rate",
    "Avg P&L/tr %": "avg_pl_per_trade",
    "Calmar": "calmar_ratio",
    "Sortino": "sortino_ratio",
    "N trades": "n_trades",
}
# Metrics where lower is better (used to invert colour scale)
_LOWER_IS_BETTER = {"max_drawdown"}

_INT_PARAMS = {
    'timeperiod', 'fenetre_lowest', 'longueur_mediane', 'Nb_bars_above',
    'user_exit_sma_length', 'macd_fast_length', 'macd_slow_length',
    'macd_signal_length', 'nb_bars_under_bbw_mini', 'nb_bars_entre_bb',
}


def _load_full_df(config, load_data):
    """Load the full date-range dataframe for window comparison backtests."""
    final_start = st.session_state.get('final_start_date', config.get('start_date'))
    final_end   = st.session_state.get('final_end_date',   config.get('end_date'))
    file_path   = st.session_state.get('final_file_path',  config.get('file_path'))
    tf          = config.get('timeframe', DEFAULT_TIMEFRAME)

    if config.get('from_file'):
        try:
            mtime = os.path.getmtime(file_path) if file_path and os.path.exists(file_path) else 0.0
        except OSError:
            mtime = 0.0
        return load_data(final_start, final_end, tf, from_file=True, file_path=file_path)
    return load_data(final_start, final_end, tf, from_file=False)


def _cast_params(params: dict) -> dict:
    """Return a copy of params with integer fields correctly cast."""
    p = params.copy()
    for k in list(p.keys()):
        if k in _INT_PARAMS:
            try:
                p[k] = int(round(float(p[k])))
            except Exception:
                pass
    return p


def _build_cmp_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Build a display DataFrame from comparison result rows."""
    records = []
    for r in rows:
        m = r.get("metrics", {})
        rec = {"Fenêtre": r["window_id"]}
        for label, key in _CMP_METRICS.items():
            val = m.get(key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                rec[label] = np.nan
            else:
                rec[label] = round(float(val), 4)
        records.append(rec)
    return pd.DataFrame(records).set_index("Fenêtre")


def _style_cmp_dataframe(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Apply green/red gradient per column; invert for max-drawdown."""
    def _col_bg(s: pd.Series) -> list[str]:
        invert = s.name in _LOWER_IS_BETTER
        finite = s.dropna()
        if finite.empty or finite.max() == finite.min():
            return [""] * len(s)
        lo, hi = float(finite.min()), float(finite.max())
        styles = []
        for v in s:
            if pd.isna(v):
                styles.append("")
                continue
            norm = (float(v) - lo) / (hi - lo)   # 0 = worst, 1 = best
            if invert:
                norm = 1.0 - norm
            r_ = int(255 * (1 - norm))
            g_ = int(200 * norm)
            styles.append(f"background-color: rgba({r_},{g_},80,0.35)")
        return styles

    return df.style.apply(_col_bg, axis=0).format(
        {col: "{:.4f}" for col in df.columns if col != "N trades"},
        na_rep="—",
    ).format({"N trades": "{:.0f}"}, na_rep="—")


def _render_best_window_chart(pf, df_full: pd.DataFrame, window_id, ref_label: str) -> None:
    """Render equity curve + Buy&Hold + price for *pf* (best window portfolio)."""
    _wid_label = "Robust Set" if str(window_id) == "Robust" else f"Fenêtre {window_id}"
    st.markdown(f"#### {_wid_label} — meilleure sur *{ref_label}*")

    try:
        value_series = pf.value() if callable(getattr(pf, "value", None)) else pf.value
        if value_series is None or (hasattr(value_series, "empty") and value_series.empty):
            st.info("Série de valeur portfolio indisponible.")
            return
        value_series = pd.to_numeric(pd.Series(value_series), errors="coerce").dropna()
    except Exception as e:
        st.warning(f"Impossible d'extraire la valeur portfolio : {e}")
        return

    # Downsample to keep the figure light (max 30 000 points)
    MAX_PTS = 30_000

    def _ds(s: pd.Series) -> pd.Series:
        if len(s) <= MAX_PTS:
            return s
        step = max(1, len(s) // MAX_PTS)
        return s.iloc[::step]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=_ds(value_series).index, y=_ds(value_series).values,
                   mode="lines", name="Valeur portfolio",
                   line=dict(color="#1f77b4", width=1.5)),
        secondary_y=False,
    )

    # Buy & Hold on same capital
    if df_full is not None and not df_full.empty:
        price = df_full["Close"] if "Close" in df_full.columns else df_full.iloc[:, 0]
        price = pd.to_numeric(price, errors="coerce").dropna()
        aligned = price.reindex(value_series.index).ffill().bfill().dropna()
        common = value_series.index.intersection(aligned.index)
        if len(common) > 1:
            pf_c, pr_c = value_series.loc[common], aligned.loc[common]
            init_cap, init_pr = float(pf_c.iloc[0]), float(pr_c.iloc[0])
            if np.isfinite(init_cap) and np.isfinite(init_pr) and init_pr != 0:
                bh = init_cap * (pr_c / init_pr)
                bh_final_val = float(bh.iloc[-1])
                bh_return_pct = (bh_final_val / init_cap - 1.0) * 100.0
                bh_ds = _ds(bh)
                fig.add_trace(
                    go.Scatter(x=bh_ds.index, y=bh_ds.values,
                               mode="lines",
                               name=f"Buy & Hold — final: {bh_final_val:,.0f} ({bh_return_pct:+.1f}%)",
                               line=dict(color="#2ca02c", width=1.5, dash="dash")),
                    secondary_y=False,
                )
                fig.add_annotation(
                    x=bh_ds.index[-1], y=bh_final_val,
                    text=f"B&H {bh_return_pct:+.1f}%<br>{bh_final_val:,.0f}",
                    showarrow=True, arrowhead=2, arrowwidth=1,
                    arrowcolor="#2ca02c",
                    ax=40, ay=-30,
                    font=dict(size=9, color="#2ca02c"),
                    bgcolor="rgba(44,160,44,0.15)",
                    bordercolor="#2ca02c", borderpad=3, borderwidth=1,
                    xref="x", yref="y",
                )
        fig.add_trace(
            go.Scatter(x=_ds(price).index, y=_ds(price).values,
                       mode="lines", name="Prix asset",
                       line=dict(color="#ff7f0e", width=1)),
            secondary_y=True,
        )

    fig.update_layout(height=420, template="plotly_dark",
                      title=f"{_wid_label} — Valeur portfolio vs Buy&Hold + Prix")
    fig.update_yaxes(title_text="Valeur portfolio", secondary_y=False)
    fig.update_yaxes(title_text="Prix", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # Trade stats summary
    try:
        st.markdown("##### Statistiques des trades")
        st.dataframe(_arrow_safe_df(pf.trades.stats()))
    except Exception:
        pass


def render_window_comparison_panel(*, get_current_config, load_data, resolve_strategy_adapter):
    """Run the best params of every WFO window on the full date range and compare results.

    Renders:
    - a button to trigger all backtests,
    - a reference-metric selector,
    - a styled comparison table (one row per window),
    - equity-curve chart for the best window.
    """
    results = st.session_state.get('wfo_results')
    if not results or not results.get('window_results'):
        return

    st.markdown("---")
    st.markdown("#### Backtests comparatifs — paramètres IS de chaque fenêtre sur la période complète")
    st.caption(
        "Pour chaque fenêtre WFO, les meilleurs paramètres trouvés en IS sont appliqués "
        "sur toute la période de backtest définie ci-dessus. Le tableau compare les résultats."
    )

    ref_label = st.selectbox(
        "Métrique de référence (meilleure fenêtre)",
        options=list(_CMP_METRICS.keys()),
        index=0,
        key="window_cmp_ref_metric",
    )
    ref_key = _CMP_METRICS[ref_label]
    higher_is_better = ref_key not in _LOWER_IS_BETTER

    run_btn = st.button("▶ Lancer la comparaison de toutes les fenêtres", key="run_window_comparison")

    # Cache invalidation token — changes when wfo_results is replaced
    token = id(results)
    cached_token = st.session_state.get("_window_cmp_token")
    cached_rows  = st.session_state.get("window_comparison_results")

    need_run = run_btn or (cached_rows is None) or (cached_token != token)

    if run_btn or (cached_rows is not None and cached_token == token):
        # Results already cached for this wfo_results object — skip re-run unless button pressed
        if not run_btn and cached_token == token:
            need_run = False

    if need_run and run_btn:
        config = get_current_config()

        with st.spinner("Chargement des données pour la période complète…"):
            df_full = _load_full_df(config, load_data)

        if df_full is None or df_full.empty:
            st.error("Données non disponibles pour la période sélectionnée.")
            return

        try:
            adapter = resolve_strategy_adapter(
                strategy_mode=config.get("strategy_mode"),
                strategy_id=config.get("strategy_id"),
                config=config,
            )
        except Exception as e:
            st.error(f"Erreur lors de la résolution de l'adaptateur stratégie : {e}")
            return

        window_results = results.get('window_results', [])
        rows: list[dict] = []
        best_score: float | None = None
        best_pf = None
        best_window_id: int | None = None

        progress = st.progress(0, text="Backtest en cours…")
        n = len(window_results)

        for i, window in enumerate(window_results):
            window_id = window.get('window_info', {}).get('window', i + 1)
            raw_params = (window.get('best_params') or {}).copy()
            params = _cast_params(raw_params)
            params['order_sizing_mode'] = config.get('order_sizing_mode', 'percent_equity')
            params['order_fixed_cash']  = float(config.get('order_fixed_cash', 10000.0))
            params['fees_pct']          = float(config.get('fees_pct', 0.0))

            try:
                pf = adapter.run_backtest(
                    df_full, params,
                    config.get('timeframe', DEFAULT_TIMEFRAME),
                    return_portfolio=True,
                )
                metrics = portfolio_metrics(pf, window_id)
            except Exception as e:
                metrics = {k: np.nan for k in ("sharpe", "return", "max_drawdown",
                                               "win_rate", "avg_pl_per_trade",
                                               "calmar_ratio", "sortino_ratio", "n_trades")}
                metrics["window"] = window_id
                metrics["error"] = str(e)
                pf = None

            rows.append({"window_id": window_id, "params": raw_params, "metrics": metrics})

            # Track best according to selected metric (evaluated at run time)
            score_val = metrics.get(ref_key)
            if score_val is not None and not (isinstance(score_val, float) and np.isnan(score_val)):
                score = float(score_val) * (1 if higher_is_better else -1)
                if best_score is None or score > best_score:
                    best_score = score
                    best_pf = pf
                    best_window_id = window_id

            progress.progress((i + 1) / n, text=f"Fenêtre {window_id}/{n}…")

        progress.empty()

        # ── Robust Set row (if enabled and valid) ────────────────────────
        robust_summary = _build_robust_set_summary(results, config)
        if robust_summary.get("status") == "ok" and robust_summary.get("robust_params"):
            robust_raw = dict(robust_summary["robust_params"])
            robust_params = _cast_params(robust_raw)
            robust_params['order_sizing_mode'] = config.get('order_sizing_mode', 'percent_equity')
            robust_params['order_fixed_cash']  = float(config.get('order_fixed_cash', 10000.0))
            robust_params['fees_pct']          = float(config.get('fees_pct', 0.0))
            try:
                with st.spinner("Backtest Robust Set…"):
                    pf_robust = adapter.run_backtest(
                        df_full, robust_params,
                        config.get('timeframe', DEFAULT_TIMEFRAME),
                        return_portfolio=True,
                    )
                metrics_robust = portfolio_metrics(pf_robust, "Robust")
            except Exception as e:
                metrics_robust = {k: np.nan for k in ("sharpe", "return", "max_drawdown",
                                                       "win_rate", "avg_pl_per_trade",
                                                       "calmar_ratio", "sortino_ratio", "n_trades")}
                metrics_robust["window"] = "Robust"
                metrics_robust["error"] = str(e)
                pf_robust = None
            rows.append({"window_id": "Robust", "params": robust_raw, "metrics": metrics_robust})
            score_val = metrics_robust.get(ref_key)
            if score_val is not None and not (isinstance(score_val, float) and np.isnan(score_val)):
                score = float(score_val) * (1 if higher_is_better else -1)
                if best_score is None or score > best_score:
                    best_score = score
                    best_pf = pf_robust
                    best_window_id = "Robust"

        st.session_state["window_comparison_results"] = rows
        st.session_state["_window_cmp_token"]         = token
        st.session_state["_window_cmp_best_pf"]       = best_pf
        st.session_state["_window_cmp_best_id"]       = best_window_id
        st.session_state["_window_cmp_df_full"]       = df_full
        cached_rows  = rows
        cached_token = token   # sync local var so the render block fires immediately

    # ── Render table ────────────────────────────────────────────────────────
    if cached_rows and cached_token == token:
        df_cmp = _build_cmp_dataframe(cached_rows)

        # Identify best row for current metric selection (may differ from run-time choice)
        best_id_display: int | None = None
        best_score_display: float | None = None
        for r in cached_rows:
            val = r.get("metrics", {}).get(ref_key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            score = float(val) * (1 if higher_is_better else -1)
            if best_score_display is None or score > best_score_display:
                best_score_display = score
                best_id_display = r["window_id"]

        if best_id_display is not None:
            label_best = "Robust Set" if best_id_display == "Robust" else f"Fenêtre {best_id_display}"
            st.success(
                f"**Meilleur résultat sur *{ref_label}* :** "
                f"{label_best} "
                f"({ref_label} = {df_cmp.loc[best_id_display, ref_label]:.4f})"
            )

        # Highlight best row (gold) and Robust row (blue permanent border)
        def _highlight_best(row):
            styles = []
            for _ in row:
                is_best   = row.name == best_id_display
                is_robust = row.name == "Robust"
                if is_robust and is_best:
                    styles.append("font-weight: bold; border: 2px solid #f0c040; "
                                  "background-color: rgba(85,153,255,0.15);")
                elif is_robust:
                    styles.append("font-weight: bold; border: 2px solid #5599ff; "
                                  "background-color: rgba(85,153,255,0.10);")
                elif is_best:
                    styles.append("font-weight: bold; border: 2px solid #f0c040;")
                else:
                    styles.append("")
            return styles

        styler = _style_cmp_dataframe(df_cmp).apply(_highlight_best, axis=1)
        st.dataframe(styler, use_container_width=True)

        # Params table
        with st.expander("Paramètres par fenêtre (+ Robust Set)", expanded=False):
            param_records = []
            for r in cached_rows:
                label = "Robust Set" if r["window_id"] == "Robust" else f"W{r['window_id']}"
                rec = {"Fenêtre": label}
                rec.update({k: v for k, v in r.get("params", {}).items()
                             if k in DEFAULT_PARAM_GRID})
                param_records.append(rec)
            if param_records:
                st.dataframe(_arrow_safe_df(pd.DataFrame(param_records).set_index("Fenêtre")), use_container_width=True)

        # ── Charts for best window ────────────────────────────────────────
        best_pf  = st.session_state.get("_window_cmp_best_pf")
        best_id  = st.session_state.get("_window_cmp_best_id")
        df_full  = st.session_state.get("_window_cmp_df_full")

        # If reference metric changed after the run, re-find best portfolio by re-running
        if best_id != best_id_display and best_id_display is not None:
            config = get_current_config()
            # Retrieve params for best_id_display from cached rows
            best_row = next((r for r in cached_rows if r["window_id"] == best_id_display), None)
            if best_row is not None and df_full is not None and not df_full.empty:
                params = _cast_params(best_row["params"])
                params['order_sizing_mode'] = config.get('order_sizing_mode', 'percent_equity')
                params['order_fixed_cash']  = float(config.get('order_fixed_cash', 10000.0))
                params['fees_pct']          = float(config.get('fees_pct', 0.0))
                try:
                    adapter = resolve_strategy_adapter(
                        strategy_mode=config.get("strategy_mode"),
                        strategy_id=config.get("strategy_id"),
                        config=config,
                    )
                    with st.spinner(f"Génération du graphique — fenêtre {best_id_display}…"):
                        best_pf = adapter.run_backtest(
                            df_full, params,
                            config.get('timeframe', DEFAULT_TIMEFRAME),
                            return_portfolio=True,
                        )
                    st.session_state["_window_cmp_best_pf"] = best_pf
                    st.session_state["_window_cmp_best_id"] = best_id_display
                except Exception as e:
                    st.warning(f"Impossible de générer le graphique : {e}")
                    best_pf = None

        if best_pf is not None and best_id_display is not None:
            _render_best_window_chart(best_pf, df_full, best_id_display, ref_label)
