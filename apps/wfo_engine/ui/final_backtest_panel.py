"""Final backtest and parameter selection functions extracted from app.py.

These are standalone copies of the param-selection and final-backtest helpers.
Some functions reference private helpers that still live in ``app.py``
(e.g. ``get_current_config``, ``resolve_strategy_adapter``, ``load_data``).
The caller must supply or monkey-patch those dependencies when wiring
this module back into the Streamlit application.
"""

import numpy as np
import pandas as pd
import streamlit as st

from config import DEFAULT_PARAM_GRID, DEFAULT_TIMEFRAME
from domain.serialization import (
    sanitize_for_json as _sanitize_for_json,
    to_jsonable as _to_jsonable,
)


# ---------------------------------------------------------------------------
# Parameter selection helpers
# ---------------------------------------------------------------------------

def _select_best_params_from_results(results, config):
    metric1_name = config.get('metric1_name', 'sharpe_ratio')
    metric2_name = config.get('metric2_name', 'total_return')
    weight_metric1 = float(config.get('weight_metric1', 1.0))
    weight_metric2 = float(config.get('weight_metric2', 0.0))

    def get_metric_value(row, name):
        if not row:
            return None
        if name == 'max_drawdown':
            value = row.get('max_drawdown')
            return None if value is None else -value
        if name == 'sharpe_ratio':
            return row.get('sharpe')
        if name == 'total_return':
            return row.get('return')
        if name == 'win_rate':
            return row.get('win_rate')
        if name == 'avg_gain_per_trade':
            return row.get('avg_gain_per_trade')
        if name == 'avg_loss_per_trade':
            value = row.get('avg_loss_per_trade')
            return None if value is None else -value
        if name == 'avg_pl_per_trade':
            return row.get('avg_pl_per_trade')
        return None

    def combined_score(row):
        total_weight = weight_metric1 + weight_metric2
        if total_weight == 0:
            return None
        m1 = get_metric_value(row, metric1_name)
        m2 = get_metric_value(row, metric2_name)
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

def load_best_params_into_inputs(*, get_current_config):
    """Load best/robust params from WFO results into sidebar input widgets."""
    final_params = None
    final_source = None
    if 'wfo_results' in st.session_state:
        best_params, best_score, best_window, best_is, best_oos, final_source, robust_summary = _select_final_params_from_results(
            st.session_state['wfo_results'],
            get_current_config()
        )
        if best_params:
            st.session_state['final_params'] = best_params
            st.session_state['final_params_score'] = best_score
            st.session_state['final_params_window'] = best_window
            st.session_state['final_params_is_metrics'] = best_is
            st.session_state['final_params_oos_metrics'] = best_oos
            st.session_state['final_params_source'] = final_source
            st.session_state['final_params_robust_summary'] = robust_summary
            final_params = best_params
            final_source = final_source or "best_window"
    if final_params is None:
        final_params = st.session_state.get("final_params")
        final_source = st.session_state.get("final_params_source")
    if not final_params:
        st.sidebar.error("No final parameters available.")
        return

    int_params = {'timeperiod', 'fenetre_lowest', 'longueur_mediane', 'Nb_bars_above', 'user_exit_sma_length',
                  'macd_fast_length', 'macd_slow_length', 'macd_signal_length'}
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
        st.session_state[f"step_{param}"] = 1 if param in int_params else 0.01

    window_id = st.session_state.get('final_params_window')
    if str(final_source or "").lower() == "robust_set":
        st.sidebar.success("Loaded robust-set parameters into input ranges.")
    elif window_id is not None:
        st.sidebar.success(f"Loaded best parameters from window {window_id}.")
    else:
        st.sidebar.success("Loaded best parameters.")


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
        'macd_fast_length', 'macd_slow_length', 'macd_signal_length'
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
        metric1_name = config.get('metric1_name', 'sharpe_ratio')
        metric2_name = config.get('metric2_name', 'total_return')
        weight_metric1 = float(config.get('weight_metric1', 1.0))
        weight_metric2 = float(config.get('weight_metric2', 0.0))

        def get_metric_value(row, name):
            if not row:
                return None
            if name == 'max_drawdown':
                value = row.get('max_drawdown')
                return None if value is None else -value
            if name == 'sharpe_ratio':
                return row.get('sharpe')
            if name == 'total_return':
                return row.get('return')
            if name == 'win_rate':
                return row.get('win_rate')
            if name == 'avg_gain_per_trade':
                return row.get('avg_gain_per_trade')
            if name == 'avg_loss_per_trade':
                value = row.get('avg_loss_per_trade')
                return None if value is None else -value
            if name == 'avg_pl_per_trade':
                return row.get('avg_pl_per_trade')
            return None

        total_weight = weight_metric1 + weight_metric2
        if total_weight == 0:
            return None
        m1 = get_metric_value(row, metric1_name)
        m2 = get_metric_value(row, metric2_name)
        if weight_metric1 != 0 and m1 is None:
            return None
        if weight_metric2 != 0 and m2 is None:
            return None
        if m1 is None:
            m1 = 0.0
        if m2 is None:
            m2 = 0.0
        return (weight_metric1 * m1 + weight_metric2 * m2) / total_weight

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
