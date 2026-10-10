"""Local, deterministic quantitative indicators for the « Analyse quant » panel.

This module implements the §5.0 (manifest + integrity check) and §5.1
(indicators Q1–Q8 + pre-verdict) blocks of the *Analyse quant* feature.

Design rules (from the cahier des charges, §5):

- **Pure and reproducible**: no network, no Streamlit, no VectorBT import.
  Every function is a pure function of its inputs and must be re-runnable.
- **Never ``0`` for a missing value**: every unavailable metric is ``None``
  (JSON ``null``) with an explicit ``raison``.  ``NaN``/``Infinity`` never leak.
- **Source + formula**: each value is annotated with the field and formula used
  to produce it (``source`` / ``formula`` keys) so the expert can audit it.
- **Deterministic pre-verdict**: ``compute_pre_verdict`` produces the local
  GO/WATCH/NO_GO *before* any A2A call, using the calculable definitions of §5.3.

The entry point is :func:`compute_quant_indicators`, which returns a single dict
compatible with the ``quant_analysis`` UI flow and the ZIP export.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from config import DEFAULT_PARAM_GRID
from domain.serialization import sanitize_for_json, sha256_json, utc_now_iso

# ---------------------------------------------------------------------------
# Versioning — used in the cache key and the exported manifest.
# ---------------------------------------------------------------------------

INDICATOR_VERSION = "1.0.1"
SCHEMA_VERSION = "quant_indicators.v1"
ENGINE_VERSION = "wfo_engine"
METRICS_VERSION = "metrics.v1"

# Unit conventions (frozen enums — §5.1 / §5.2)
UNIT_RETURN = "return_pct"       # total_return expressed in %
UNIT_WIN_RATE = "win_rate_pct"   # win rate expressed in %
UNIT_SHARPE = "sharpe_per_bar"   # non-annualised Sharpe (mean/std per bar)

# Verdict vocabulary (§5.2)
VERDICTS = ("GO", "WATCH", "NO_GO")
VERDICT_SCOPE = "exploratoire"

# Neighbourhood thresholds (§5.3)
NEIGH_DIST_MAX = 0.10       # d <= 0.10 defines a "voisin"
NEIGH_PERF_RATIO = 0.80     # score >= 80% of winner defines "voisin performant"
NEIGH_ISOLATED_LT = 2       # |V_perf| < 2  -> "gagnant isole"
DEGRAD_MEDIAN_RATIO = 0.50  # median(V) < 50% of winner -> "degradation marquee"
CONJ_WINDOWS_MIN = 2        # conjunction must hold in >= 2 windows

# Verdict thresholds (§5.3 table) — configurable defaults.
QUANT_VERDICT_MIN_WINDOWS_GO = 5
QUANT_VERDICT_MIN_WINDOWS_WATCH = 3
QUANT_VERDICT_MIN_TRADES_GO = 100
QUANT_VERDICT_MIN_TRADES_WATCH = 30
QUANT_VERDICT_EROSION_GO_PCT = 30.0
QUANT_VERDICT_EROSION_WATCH_PCT = 50.0
QUANT_VERDICT_EROSION_RECURRENT_FRAC = 0.50
QUANT_VERDICT_BOUND_FRAC = 0.80          # "systematiquement en borne" fraction
QUANT_VERDICT_NEIGH_MIN = 5              # "5 configurations voisines performantes"
QUANT_VERDICT_CONCENTRATION_TRADE_FRAC = 0.50
QUANT_VERDICT_CONCENTRATION_WINDOW_FRAC = 0.60
QUANT_VERDICT_MAJORITY_FRAC = 0.50       # "> 50%"

# Robust z-score alternative (§5.3)
ZSCORE_CONSISTENCY = 1.4826
ZSCORE_IQR_DIVISOR = 1.349
ZSCORE_MIN_SAMPLE = 5
ALT_VPERF_Z_MIN = 0.0
ALT_DEGRAD_Z_MAX = -0.5


# ---------------------------------------------------------------------------
# Scalar / coercion helpers
# ---------------------------------------------------------------------------

def _fnum(value: Any) -> Optional[float]:
    """Coerce *value* to a finite float, or None when absent/non-finite.

    Never raises: non-coercible values become None.  This is the single place
    that implements the « null + raison, jamais 0 » policy.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return 1.0 if value else 0.0
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _avail(value: Optional[float], raison: Optional[str] = None) -> dict:
    """Wrap a metric value with its availability status (§5.3 policy)."""
    if value is None:
        return {
            "value": None,
            "available": False,
            "raison": raison or "indisponible",
        }
    return {"value": value, "available": True, "raison": None}


def _median_of(values: Sequence[Optional[float]]) -> Optional[float]:
    finite = [v for v in values if v is not None]
    if not finite:
        return None
    return float(np.median(np.asarray(finite, dtype=float)))


def _mean_of(values: Sequence[Optional[float]]) -> Optional[float]:
    finite = [v for v in values if v is not None]
    if not finite:
        return None
    return float(np.mean(np.asarray(finite, dtype=float)))


def _compounded_return_pct(values: Sequence[Optional[float]]) -> Optional[float]:
    """Compounded (portfolio-equivalent) total return in % from period returns."""
    finite = [v for v in values if v is not None]
    if not finite:
        return None
    try:
        total = 1.0
        for v in finite:
            total *= 1.0 + v / 100.0
        return (total - 1.0) * 100.0
    except (OverflowError, ValueError):
        return None


def _hash_dataframe(df: Optional[pd.DataFrame]) -> Optional[str]:
    """Deterministic content hash of a DataFrame (canonical CSV serialisation)."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    try:
        raw = df.to_csv(index=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Returns helpers
# ---------------------------------------------------------------------------

def _sharpe_from_returns(rets: Any) -> Optional[float]:
    """Non-annualised Sharpe = mean / std (ddof=1) of per-bar returns.

    Mirrors ``metrics._sharpe_from_returns`` but operates directly on a returns
    array/Series instead of a VectorBT portfolio.  Returns None (indisponible)
    when the series is missing, too short, or has zero variance — never a
    fallback from a different provenance (§5.1 Q7).
    """
    if rets is None:
        return None
    try:
        arr = np.asarray(rets, dtype=float)
    except (TypeError, ValueError):
        return None
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return None
    std = float(arr.std(ddof=1))
    if not math.isfinite(std) or std <= 0.0:
        return None
    mean = float(arr.mean())
    if not math.isfinite(mean):
        return None
    return mean / std


def _canonical_index_bytes(idx: Any) -> bytes:
    """Canonical bytes for a pandas index (values + dtype), order-sensitive."""
    try:
        dtype_name = str(getattr(idx, "dtype", np.asarray(idx).dtype))
        vals = list(idx)
        s = dtype_name + "::" + ",".join(str(v) for v in vals)
        return s.encode("utf-8")
    except Exception:
        return b""


def _hash_returns(rets: Any) -> Optional[str]:
    """Byte-level hash of a returns series (order + content + index sensitive)."""
    if rets is None:
        return None
    try:
        if isinstance(rets, pd.Series):
            arr = np.asarray(rets.values, dtype=np.float64).ravel()
            payload = _canonical_index_bytes(rets.index) + b"|" + np.ascontiguousarray(arr, dtype=np.float64).tobytes()
            return hashlib.sha256(payload).hexdigest()
        arr = np.asarray(rets, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Config / settings extraction
# ---------------------------------------------------------------------------

def _get_run_settings(wfo_results: Mapping[str, Any]) -> dict:
    settings = wfo_results.get("settings") if isinstance(wfo_results, Mapping) else None
    return settings if isinstance(settings, Mapping) else {}


def resolve_run_id(wfo_results) -> Optional[str]:
    """Read the existing run identity, including the production traceability path."""
    if not isinstance(wfo_results, Mapping):
        return None
    run_id = wfo_results.get("run_id")
    if isinstance(run_id, str) and run_id.strip():
        return run_id
    traceability = wfo_results.get("traceability")
    run = traceability.get("run") if isinstance(traceability, Mapping) else None
    run_id = run.get("run_id") if isinstance(run, Mapping) else None
    return run_id if isinstance(run_id, str) and run_id.strip() else None


def _merge_config(wfo_results: Mapping[str, Any], config: Optional[Mapping[str, Any]]) -> dict:
    """Merge run settings (authoritative for run-level facts) with UI config."""
    merged = dict(_get_run_settings(wfo_results))
    if isinstance(config, Mapping):
        merged.update({k: v for k, v in config.items()})
    return merged


def _resolve_cost_convention(merged: Mapping[str, Any]) -> dict:
    """Resolve the cost convention with an explicit provenance status.

    Distinguishes (§5.0 / Q8.1, F6):
      - ``unknown`` : neither component declared, or a declared component is
        ``None`` (a declared-but-unknown value does NOT prove zero fees)
      - ``invalid`` : a declared component is non-finite / non-numeric
      - ``known``   : declared components are numeric (0 or > 0); ``gross`` is
        reserved for an *explicit* numeric zero of known provenance

    Returns ``{status, convention, fees_pct, slippage_bps, cost_per_side_pct,
    costs_included}``.  ``costs_included`` is None unless status == "known".
    """
    fees = merged.get("fees_pct")
    slippage = merged.get("slippage_bps")
    fees_present = "fees_pct" in merged
    slippage_present = "slippage_bps" in merged

    # declared-but-None is "unknown", declared non-numeric is "invalid"
    fees_none = fees_present and fees is None
    slippage_none = slippage_present and slippage is None
    fees_invalid = fees_present and not fees_none and _fnum(fees) is None
    slippage_invalid = slippage_present and not slippage_none and _fnum(slippage) is None

    if fees_invalid or slippage_invalid:
        return {
            "status": "invalid", "convention": "invalid",
            "fees_pct": None, "slippage_bps": None,
            "cost_per_side_pct": None, "costs_included": None,
        }
    if fees_none or slippage_none:
        return {
            "status": "unknown", "convention": "unknown",
            "fees_pct": None, "slippage_bps": None,
            "cost_per_side_pct": None, "costs_included": None,
        }
    if not fees_present and not slippage_present:
        return {
            "status": "unknown", "convention": "unknown",
            "fees_pct": None, "slippage_bps": None,
            "cost_per_side_pct": None, "costs_included": None,
        }

    # both non-None numeric (or one present numeric, the other absent -> UI default 0)
    fees_pct = _fnum(fees) if fees_present else 0.0
    slippage_bps = _fnum(slippage) if slippage_present else 0.0
    cost_per_side_pct = (fees_pct or 0.0) + ((slippage_bps or 0.0) / 100.0)
    costs_included = cost_per_side_pct > 0.0
    return {
        "status": "known",
        "convention": "net_of_costs" if costs_included else "gross",
        "fees_pct": fees_pct,
        "slippage_bps": slippage_bps,
        "cost_per_side_pct": cost_per_side_pct,
        "costs_included": costs_included,
    }


def _get_windows(wfo_results: Mapping[str, Any]) -> list:
    wrs = wfo_results.get("window_results") if isinstance(wfo_results, Mapping) else None
    return wrs if isinstance(wrs, list) else []


def _get_window_id(window: Mapping[str, Any], fallback: int) -> Any:
    info = window.get("window_info") if isinstance(window, Mapping) else None
    wid = (info or {}).get("window") if isinstance(info, Mapping) else None
    return wid if wid is not None else fallback


# ---------------------------------------------------------------------------
# Trial extraction
# ---------------------------------------------------------------------------

def _window_trials(window: Mapping[str, Any]) -> list:
    """Return the list of trial records for one window (all trials preferred)."""
    if not isinstance(window, Mapping):
        return []
    trials = window.get("optimization_trials")
    if not isinstance(trials, list) or not trials:
        trials = window.get("optimization_results")
    return trials if isinstance(trials, list) else []


def _trials_dataframe(window: Mapping[str, Any]) -> pd.DataFrame:
    trials = _window_trials(window)
    if not trials:
        return pd.DataFrame()
    try:
        return pd.DataFrame(trials)
    except Exception:
        return pd.DataFrame()


def _score_column(df: pd.DataFrame) -> Optional[str]:
    for col in ("combined_score", "score", "objective"):
        if col in df.columns:
            return col
    return None


def _authoritative_params(
    wfo_results: Mapping[str, Any],
    param_grid: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> list:
    """Authoritative parameter names (for signatures/dedup), never inferred from
    arbitrary trial columns (a metric column such as ``return`` must never be
    treated as a parameter).  Sources: param_grid keys, config selected_params,
    and the union of best_params keys across windows (the params actually used).
    """
    names: set = set()
    if isinstance(param_grid, Mapping):
        names.update(param_grid.keys())
    if isinstance(config, Mapping):
        sel = config.get("selected_params")
        if isinstance(sel, (list, tuple)):
            names.update(sel)
    for w in _get_windows(wfo_results):
        bp = w.get("best_params") if isinstance(w, Mapping) else None
        if isinstance(bp, Mapping):
            names.update(bp.keys())
    return sorted(names)


def _signature_columns(df: pd.DataFrame, authoritative: Sequence[str]) -> list:
    """The authoritative params actually present as columns of *df*."""
    return [p for p in authoritative if p in df.columns]


def _coerce_param_value(v: Any) -> Any:
    """Normalise a parameter value for equality / distance computations."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if not math.isfinite(f):
            return None
        if abs(f - round(f)) < 1e-9:
            return int(round(f))
        return f
    return v


# ---------------------------------------------------------------------------
# Parameter ranges (authoritative param names only — F7)
# ---------------------------------------------------------------------------

def _resolve_param_ranges(
    wfo_results: Mapping[str, Any],
    all_trials: Optional[pd.DataFrame],
    param_grid: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Resolve each *authoritative* parameter's (min, max) range + categorical flag.

    The parameter names come ONLY from an authoritative source (``param_grid``
    keys or ``config['selected_params']``).  Trial/score/metadata columns are
    **never** treated as parameters (F7).  Priority per parameter:
      1. explicit ``param_grid`` spec (``(min, max)`` or iterable of values)
      2. ``config['min_<p>']`` / ``config['max_<p>']``
      3. ``DEFAULT_PARAM_GRID`` (config.py)
      4. observed min/max across best_params/trials (authoritative params only)

    Returns ``{param: {"min": m, "max": M, "categorical": bool}}``.  Fixed
    (zero-range) numeric dimensions are dropped so ``p`` excludes them; a
    categorical param always gets a nominal {0,1} range.
    """
    cfg = config if isinstance(config, Mapping) else {}
    if isinstance(param_grid, Mapping) and param_grid:
        param_names = list(param_grid.keys())
    else:
        sel = cfg.get("selected_params")
        param_names = list(sel) if isinstance(sel, (list, tuple)) else []

    ranges: dict = {}

    def _register(param: str, lo: Any, hi: Any, categorical: bool):
        flo = _fnum(lo)
        fhi = _fnum(hi)
        if categorical:
            ranges[param] = {"min": 0.0, "max": 1.0, "categorical": True}
            return
        if flo is None or fhi is None:
            return
        if flo == fhi:
            return  # fixed numeric dimension -> excluded (zero range)
        lo_n, hi_n = min(flo, fhi), max(flo, fhi)
        ranges[param] = {"min": lo_n, "max": hi_n, "categorical": False}

    for param in param_names:
        spec = param_grid.get(param) if isinstance(param_grid, Mapping) else None
        if spec is not None:
            if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
                vals = list(spec)
                if vals and all(isinstance(v, (bool, np.bool_)) for v in vals):
                    _register(param, 0, 1, True)
                    continue
                nums = [_fnum(v) for v in vals]
                nums = [n for n in nums if n is not None]
                if nums:
                    _register(param, min(nums), max(nums), False)
                continue
            if isinstance(spec, tuple) and len(spec) >= 2:
                _register(param, spec[0], spec[1], False)
                continue

        lo = cfg.get(f"min_{param}")
        hi = cfg.get(f"max_{param}")
        if lo is not None and hi is not None:
            _register(param, lo, hi, False)
            continue

        dg = DEFAULT_PARAM_GRID.get(param)
        if dg is not None:
            _register(param, dg[0], dg[1], False)
            continue

        # observed fallback (authoritative param only)
        vals: list = []
        categorical = False
        for window in _get_windows(wfo_results):
            bp = window.get("best_params") if isinstance(window, Mapping) else None
            if isinstance(bp, Mapping) and param in bp:
                cv = _coerce_param_value(bp[param])
                if isinstance(cv, (bool, str)):
                    categorical = True
                else:
                    f = _fnum(cv)
                    if f is not None:
                        vals.append(f)
            for trial in _window_trials(window):
                if isinstance(trial, Mapping) and param in trial:
                    cv = _coerce_param_value(trial[param])
                    if isinstance(cv, (bool, str)):
                        categorical = True
                    else:
                        f = _fnum(cv)
                        if f is not None:
                            vals.append(f)
        if categorical:
            _register(param, 0, 1, True)
        elif vals:
            _register(param, min(vals), max(vals), False)

    return ranges


# ---------------------------------------------------------------------------
# 5.0 Manifest
# ---------------------------------------------------------------------------

def build_run_manifest(
    wfo_results: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
    *,
    all_trials: Optional[pd.DataFrame] = None,
    final_trades: Optional[pd.DataFrame] = None,
    per_bar_returns: Optional[Mapping[str, Any]] = None,
    oos_trades: Optional[pd.DataFrame] = None,
    param_grid: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Build the run manifest (§5.0): scope, provenance, conventions."""
    merged = _merge_config(wfo_results, config)
    settings = _get_run_settings(wfo_results)
    windows = _get_windows(wfo_results)

    n_windows = _fnum(settings.get("n_windows", merged.get("n_windows")))
    anchored = bool(settings.get("anchored", merged.get("anchored", False)))
    optimization_method = settings.get("optimization_method", merged.get("optimization_method"))
    selection_method = settings.get("selection_method", merged.get("selection_method", "snv"))
    regime = settings.get("optimization_regime", merged.get("optimization_regime", "classic"))
    mode = "anchored" if anchored else "rolling"

    window_dates = []
    for i, window in enumerate(windows):
        info = (window.get("window_info") if isinstance(window, Mapping) else None) or {}
        window_dates.append({
            "window": _get_window_id(window, i + 1),
            "start": info.get("start_date"),
            "end": info.get("end_date"),
            "in_sample_start": info.get("in_sample_start"),
            "in_sample_end": info.get("in_sample_end"),
            "out_sample_start": info.get("out_sample_start"),
            "out_sample_end": info.get("out_sample_end"),
        })

    final_period = {
        "start": merged.get("final_start_date"),
        "end": merged.get("final_end_date"),
    }

    # costs convention (§5.1 Q8): distinguish unknown / invalid / zero
    costs = _resolve_cost_convention(merged)

    has_per_bar = bool(per_bar_returns) if isinstance(per_bar_returns, Mapping) else False
    returns_source = "per_bar_and_per_trade" if has_per_bar else "per_trade_only"

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "indicator_version": INDICATOR_VERSION,
        "run_id": resolve_run_id(wfo_results),
        "input_digest": _compute_input_digest(
            wfo_results, all_trials, final_trades, per_bar_returns, config, oos_trades, param_grid
        ),
        "input_digest_replay": compute_replay_input_digest(wfo_results, config),
        "engine_version": ENGINE_VERSION,
        "metrics_version": METRICS_VERSION,
        "seed": merged.get("seed", settings.get("seed")),
        "optimization_method": optimization_method,
        "optimization_regime": regime,
        "selection_method": selection_method,
        "mode": mode,
        "n_windows": n_windows,
        "n_windows_computed": len(windows),
        "window_dates": window_dates,
        "embargo": merged.get("embargo"),
        "purge": merged.get("purge"),
        "timeframe": merged.get("timeframe"),
        "final_period": final_period,
        "returns_source": returns_source,
        "returns_granularity": "per_bar" if has_per_bar else "per_trade",
        "units": {
            "return": UNIT_RETURN,
            "win_rate": UNIT_WIN_RATE,
            "sharpe": UNIT_SHARPE,
        },
        "costs": {
            "included": costs["costs_included"],
            "fees_pct": costs["fees_pct"],
            "slippage_bps": costs["slippage_bps"],
            "convention": costs["convention"],
            "status": costs["status"],
        },
    }
    return sanitize_for_json(manifest)


def compute_replay_input_digest(wfo_results, config=None) -> Optional[str]:
    """Hash only the shared replay inputs: run results and config.

    This associates archived facts with an imported run, not with missing raw
    evidence. It is a content fingerprint, not an authenticity certificate.
    """
    if not isinstance(wfo_results, Mapping) or not wfo_results:
        return None
    return sha256_json({
        "wfo_results": wfo_results,
        "config": config if isinstance(config, Mapping) else {},
    })


def _compute_input_digest(
    wfo_results: Mapping[str, Any],
    all_trials: Optional[pd.DataFrame],
    final_trades: Optional[pd.DataFrame],
    per_bar_returns: Optional[Mapping[str, Any]],
    config: Optional[Mapping[str, Any]] = None,
    oos_trades: Optional[pd.DataFrame] = None,
    param_grid: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Deterministic hash of the run artifacts + config + grid (F1).

    Hashes the **content** of the results, trials, trades (Final and OOS), the
    config and the parameter grid so that any change to a score, a trade P&L, a
    setting or a grid range invalidates the digest (and therefore the
    cache/export key).
    """
    payload: dict = {"indicator_version": INDICATOR_VERSION}
    if isinstance(wfo_results, Mapping):
        payload["wfo_results"] = wfo_results
    if isinstance(all_trials, pd.DataFrame) and not all_trials.empty:
        payload["all_trials"] = _hash_dataframe(all_trials)
    if isinstance(final_trades, pd.DataFrame) and not final_trades.empty:
        payload["final_trades"] = _hash_dataframe(final_trades)
    if isinstance(oos_trades, pd.DataFrame) and not oos_trades.empty:
        payload["oos_trades"] = _hash_dataframe(oos_trades)
    if isinstance(param_grid, Mapping):
        payload["param_grid"] = param_grid
    if isinstance(per_bar_returns, Mapping):
        sigs = {}
        for scope, series in per_bar_returns.items():
            if isinstance(series, Mapping):
                sigs[str(scope)] = {str(k): _hash_returns(v) for k, v in series.items()}
            else:
                sigs[str(scope)] = _hash_returns(series)
        payload["per_bar_returns"] = sigs
    if isinstance(config, Mapping):
        payload["config"] = config
    return sha256_json(payload)


# ---------------------------------------------------------------------------
# 5.0 Contrôle d'intégrité (blocking)
# ---------------------------------------------------------------------------

def check_run_integrity(
    wfo_results: Mapping[str, Any],
    *,
    all_trials: Optional[pd.DataFrame] = None,
    final_trades: Optional[pd.DataFrame] = None,
    per_bar_returns: Optional[Mapping[str, Any]] = None,
    oos_trades: Optional[pd.DataFrame] = None,
    config: Optional[Mapping[str, Any]] = None,
    param_grid: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Blocking integrity check (§5.0).

    Returns ``{status, ok, causes, checks}``.  When ``ok`` is False the run is
    ``non_evaluable`` and **no A2A call may be issued**.
    """
    causes: list = []
    checks: dict = {}

    def _fail(key: str, raison: str):
        checks[key] = {"ok": False, "raison": raison}
        causes.append(raison)

    def _pass(key: str, extra: Optional[dict] = None):
        checks[key] = {"ok": True, "raison": None, **(extra or {})}

    if not isinstance(wfo_results, Mapping) or not wfo_results:
        _fail("run_present", "wfo_results absent ou vide")
        return {"status": "non_evaluable", "ok": False, "causes": causes, "checks": checks}
    _pass("run_present")

    windows = _get_windows(wfo_results)
    if not windows:
        _fail("windows_present", "aucune fenêtre WFO dans wfo_results")
    else:
        _pass("windows_present")

    # declared n_windows must be present, finite, a positive integer, and
    # exactly equal to the computed windows — no truncation (P0)
    settings = _get_run_settings(wfo_results)
    raw_nw = settings.get("n_windows")
    nw = _fnum(raw_nw)
    if nw is None or nw < 1 or nw != int(nw):
        _fail(
            "n_windows_coherence",
            f"settings.n_windows invalide ({raw_nw!r}) — attendu un entier fini positif",
        )
    elif int(nw) != len(windows):
        _fail(
            "n_windows_coherence",
            f"settings.n_windows ({int(nw)}) != fenêtres calculées ({len(windows)})",
        )
    else:
        _pass("n_windows_coherence", {"n_windows": int(nw)})

    # IS/OOS pairing, non-empty, uniqueness, match to computed windows
    is_perf = wfo_results.get("in_sample_performance") or []
    oos_perf = wfo_results.get("out_of_sample_performance") or []
    is_perf = [r for r in is_perf if isinstance(r, Mapping)]
    oos_perf = [r for r in oos_perf if isinstance(r, Mapping)]
    is_windows = [r.get("window") for r in is_perf]
    oos_windows = [r.get("window") for r in oos_perf]
    expected_windows = [_get_window_id(w, i + 1) for i, w in enumerate(windows)]

    if not is_perf and not oos_perf:
        _fail("is_oos_paired", "listes IS et OOS toutes deux vides")
    elif set(is_windows) != set(oos_windows):
        _fail(
            "is_oos_paired",
            f"fenêtres IS ({sorted(map(str, is_windows))}) non appariées aux fenêtres OOS "
            f"({sorted(map(str, oos_windows))})",
        )
    elif len(is_windows) != len(set(is_windows)) or len(oos_windows) != len(set(oos_windows)):
        _fail("is_oos_paired", "identifiants de fenêtre dupliqués (IS ou OOS)")
    elif set(is_windows) != set(expected_windows):
        _fail(
            "is_oos_paired",
            f"fenêtres de performance ({sorted(map(str, is_windows))}) non conformes "
            f"aux fenêtres calculées ({sorted(map(str, expected_windows))})",
        )
    else:
        _pass("is_oos_paired", {"n_windows": len(expected_windows)})

    # best_params per window
    bp_missing = sum(
        1 for w in windows if not isinstance(w, Mapping) or not isinstance(w.get("best_params"), Mapping)
    )
    if bp_missing:
        _fail("best_params_present", f"{bp_missing} fenêtre(s) sans best_params")
    else:
        _pass("best_params_present")

    # finite values in IS/OOS metrics (blocking)
    finite_violations = 0
    for rows in (is_perf, oos_perf):
        for r in rows:
            for key in ("return", "sharpe"):
                v = r.get(key)
                if v is not None and _fnum(v) is None:
                    finite_violations += 1
    if finite_violations:
        _fail("finite_values", f"{finite_violations} valeur(s) IS/OOS non finie(s)")
    else:
        _pass("finite_values")

    # required IS/OOS fields present + valid (blocking): `return` is the core
    # metric; `n_trades` must be a finite non-negative integer (P0)
    missing_required = 0
    for rows in (is_perf, oos_perf):
        for r in rows:
            if r.get("return") is None:
                missing_required += 1
            nt_raw = r.get("n_trades")
            nt = _fnum(nt_raw)
            if nt is None or nt < 0 or nt != int(nt):
                missing_required += 1
    if missing_required:
        _fail("required_fields", f"{missing_required} champ(s) obligatoire(s) IS/OOS manquant(s) ou invalide(s) (return / n_trades)")
    else:
        _pass("required_fields")

    # cardinality coherence (blocking): declared optimization_trials_count and
    # evaluations must be finite non-negative integers equal to the trial rows
    # (one row per evaluation in the engine) (P0)
    cardinality_violations = 0
    for window in windows:
        trials = _window_trials(window)
        for key in ("optimization_trials_count", "evaluations"):
            raw = (window or {}).get(key)
            if raw is None:
                continue
            val = _fnum(raw)
            if val is None or val < 0 or val != int(val):
                cardinality_violations += 1  # non-finite / negative / non-integer
            elif int(val) != len(trials):
                cardinality_violations += 1  # mismatch with trial rows
    if cardinality_violations:
        _fail("cardinality", f"{cardinality_violations} incohérence(s) de cardinalité (trials_count / evaluations vs lignes)")
    else:
        _pass("cardinality")

    # trial-score finiteness — reported as a diagnostic, not blocking (NaN/Inf
    # scores are the pruned/failed trials already counted by Q1).
    n_nonfinite_scores = 0
    for window in windows:
        df = _trials_dataframe(window)
        sc = _score_column(df)
        if df.empty or sc is None:
            continue
        s = pd.to_numeric(df[sc], errors="coerce")
        n_nonfinite_scores += int(s.apply(lambda x: 1 if (x is not None and not math.isfinite(float(x))) else 0).sum())
    checks["trial_scores_finite"] = {
        "ok": True, "raison": None, "n_non_finite": n_nonfinite_scores,
    }

    # trades available
    n_final_trades = len(final_trades) if isinstance(final_trades, pd.DataFrame) else None
    if n_final_trades is None:
        _fail("final_trades", "final_trades indisponible (aucun trade de backtest final)")
    elif n_final_trades == 0:
        _fail("final_trades", "final_trades vide (0 trade)")
    else:
        _pass("final_trades", {"n_trades": n_final_trades})

    # unique identifier
    run_id = resolve_run_id(wfo_results)
    if run_id is None:
        digest = _compute_input_digest(wfo_results, all_trials, final_trades, per_bar_returns, config, oos_trades, param_grid)
        if digest:
            _pass("run_id", {"fallback_digest": True})
        else:
            _fail("run_id", "identifiant de run indéterminable")
    else:
        _pass("run_id")

    # trials coherence
    if isinstance(all_trials, pd.DataFrame) and not all_trials.empty:
        sc = _score_column(all_trials)
        valid = 0
        if sc is not None:
            valid = int(pd.to_numeric(all_trials[sc], errors="coerce").dropna().shape[0])
        if valid == 0:
            _fail("trials", "aucun essai à score valide dans all_trials")
        else:
            _pass("trials", {"n_trials": int(len(all_trials)), "n_valid": valid})
    else:
        total_trials = sum(len(_window_trials(w)) for w in windows)
        if total_trials == 0:
            _fail("trials", "aucun trial d'optimisation présent")
        else:
            _pass("trials", {"n_trials": total_trials})

    # costs provenance known (§5.0 / Q8.1) — unknown/invalid convention blocking
    costs = _resolve_cost_convention(_merge_config(wfo_results, config))
    if costs["status"] == "unknown":
        _fail("costs_applied", "provenance des coûts inconnue (fees_pct/slippage_bps absents)")
    elif costs["status"] == "invalid":
        _fail("costs_applied", "composante de coût invalide (fees_pct/slippage_bps non numérique)")
    else:
        _pass("costs_applied", {
            "fees_pct": costs["fees_pct"],
            "slippage_bps": costs["slippage_bps"],
            "costs_included": costs["costs_included"],
            "convention": costs["convention"],
        })

    # param-level pairing — verify the engine-side params fingerprint (§5.0).
    # The IS/OOS metric rows now carry a ``params_sha`` (sha256 of the selected
    # params actually used, emitted by wfo.py / adaptive_optimization.py).
    #   - keys inconsistent          -> blocking
    #   - fingerprint mismatch       -> blocking
    #   - fingerprint matches        -> verified (ok=True)
    #   - fingerprint absent (legacy)-> non-blocking "inferred_only"
    best_keys = [
        set((w.get("best_params") or {}).keys()) for w in windows if isinstance(w, Mapping)
    ]
    consistent = all(k == best_keys[0] for k in best_keys) if best_keys else True
    if not consistent:
        _fail("params_matched", "clés de best_params incohérentes entre fenêtres")
        checks["params_matched"] = {
            "ok": False, "blocking": True, "raison": "clés de best_params incohérentes entre fenêtres",
            "verified": "inconsistent", "formal_verification": False,
        }
    else:
        is_by_window = {r.get("window"): r for r in is_perf}
        oos_by_window = {r.get("window"): r for r in oos_perf}
        all_windows = set(is_by_window) | set(oos_by_window)
        n_expected_sides = 3 * len(all_windows)  # IS + OOS + window-level
        n_present = 0
        n_mismatch = 0
        for window in windows:
            wid = _get_window_id(window, 0)
            bp = window.get("best_params") if isinstance(window, Mapping) else None
            if not isinstance(bp, Mapping):
                continue
            expected = sha256_json(bp)
            # window-level fingerprint (window_results[].params_sha)
            window_fp = window.get("params_sha") if isinstance(window, Mapping) else None
            if window_fp is not None:
                n_present += 1
                if window_fp != expected:
                    n_mismatch += 1
            for side_map in (is_by_window, oos_by_window):
                row = side_map.get(wid)
                if not isinstance(row, Mapping):
                    continue
                fp = row.get("params_sha")
                if fp is None:
                    continue  # this side carries no fingerprint
                n_present += 1
                if fp != expected:
                    n_mismatch += 1
        if n_mismatch:
            _fail(
                "params_matched",
                f"empreinte de paramètres incohérente ({n_mismatch} côté(s) IS/OOS vs best_params)",
            )
            checks["params_matched"] = {
                "ok": False, "blocking": True, "raison": "params_sha incohérent avec best_params",
                "verified": "fingerprint_mismatch", "formal_verification": True,
                "n_present": n_present, "n_expected": n_expected_sides, "n_mismatch": n_mismatch,
            }
        elif n_present == 0:
            checks["params_matched"] = {
                "ok": False, "blocking": False,
                "raison": "empreinte de paramètres absente dans les métriques IS/OOS — appariement par paramètres non vérifiable (résultats legacy)",
                "verified": "inferred_only", "formal_verification": False,
                "note": "appariement par fenêtre (window-id) vérifié ; appariement par paramètres sélectionnés inféré, non prouvé",
            }
        elif n_present == n_expected_sides:
            checks["params_matched"] = {
                "ok": True, "blocking": False, "raison": None,
                "verified": "fingerprint", "formal_verification": True,
                "n_present": n_present, "n_expected": n_expected_sides,
            }
        else:
            # partially fingerprinted: some IS/OOS sides carry no fingerprint —
            # an incomplete artifact, distinct from a fully-legacy run -> blocking
            _fail(
                "params_matched",
                f"empreinte de paramètres partielle ({n_present}/{n_expected_sides} côtés IS/OOS) — artefact incomplet",
            )
            checks["params_matched"] = {
                "ok": False, "blocking": True,
                "raison": f"empreinte de paramètres partielle ({n_present}/{n_expected_sides} côtés IS/OOS) — artefact incomplet",
                "verified": "partial", "formal_verification": False,
                "n_present": n_present, "n_expected": n_expected_sides,
                "n_missing": n_expected_sides - n_present,
            }

    # Upstream fingerprint reconciliation (§5.0), emitted by
    # ``stagewise_optimizer.build_stagewise_wfo_results``.  A detected anomaly
    # (internally inconsistent source fingerprint, or a sha256_json failure) is a
    # real defect and must BLOCK the run — it must never be downgraded to the
    # merely non-verifiable legacy case.
    recon = wfo_results.get("params_sha_reconciliation") if isinstance(wfo_results, Mapping) else None
    if isinstance(recon, Mapping) and recon.get("ok") is False:
        anomalies = recon.get("anomalies") or []
        codes = sorted({str(a.get("code")) for a in anomalies if isinstance(a, Mapping)})
        _fail(
            "params_sha_reconciliation",
            f"réconciliation des empreintes de paramètres en échec ({len(anomalies)} anomalie(s) : {', '.join(codes)})",
        )
        checks["params_sha_reconciliation"] = {
            "ok": False, "blocking": True,
            "raison": f"empreintes de paramètres non réconciliables ({len(anomalies)} anomalie(s))",
            "verified": "anomaly", "formal_verification": False,
            "codes": codes, "anomalies": list(anomalies),
        }

    ok = not causes
    return {
        "status": "ok" if ok else "non_evaluable",
        "ok": ok,
        "causes": causes,
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Q1 — Budget effectif
# ---------------------------------------------------------------------------

def compute_q1(
    wfo_results: Mapping[str, Any],
    all_trials: Optional[pd.DataFrame] = None,
    param_grid: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Q1 — effective budget: trials demanded / launched / completed / pruned."""
    windows = _get_windows(wfo_results)
    settings = _get_run_settings(wfo_results)
    merged = _merge_config(wfo_results, None)
    authoritative = _authoritative_params(wfo_results, param_grid, config)

    optimization_method = str(settings.get("optimization_method", merged.get("optimization_method", "bayesian"))).lower()
    if optimization_method == "grid":
        demanded_per_window = _fnum((wfo_results.get("timing") or {}).get("param_combinations"))
    else:
        # A recorded run budget wins; legacy results may use explicit config.
        # Never manufacture the optimizer's default as a historical run fact.
        raw_budget = settings.get("max_trials") if "max_trials" in settings else (
            config.get("max_trials") if isinstance(config, Mapping) else None
        )
        demanded_per_window = _fnum(raw_budget)
        if (isinstance(raw_budget, bool) or demanded_per_window is None
                or demanded_per_window < 1 or demanded_per_window != int(demanded_per_window)):
            demanded_per_window = None

    n_windows = len(windows)
    demanded = (demanded_per_window * n_windows) if (demanded_per_window is not None and n_windows) else None

    launched = 0
    completed = 0
    unique_per_window: dict = {}
    params_set_per_window: dict = {}

    for i, window in enumerate(windows):
        wid = _get_window_id(window, i + 1)
        trials = _window_trials(window)
        evaluations = _fnum((window or {}).get("evaluations"))
        launched += int(evaluations) if evaluations is not None else len(trials)

        df = _trials_dataframe(window)
        sc = _score_column(df)
        params_cols = _signature_columns(df, authoritative)
        if not df.empty and sc is not None:
            valid = pd.to_numeric(df[sc], errors="coerce")
            completed += int(valid.dropna().shape[0])
        if params_cols:
            sigs = [df[p].map(_coerce_param_value) for p in params_cols]
            uniq = int(pd.concat(sigs, axis=1).drop_duplicates().shape[0]) if sigs else 0
        else:
            uniq = 0
        unique_per_window[str(wid)] = uniq
        params_set_per_window[str(wid)] = params_cols

    pruned = (launched - completed) if launched else None

    cardinality = None
    if optimization_method == "grid":
        cardinality = _fnum((wfo_results.get("timing") or {}).get("param_combinations"))

    return sanitize_for_json({
        "id": "Q1",
        "label": "Budget effectif",
        "source": "window_results[].optimization_trials / evaluations + settings.max_trials",
        "demanded": _avail(demanded),
        "launched": _avail(_fnum(launched) if launched else None),
        "completed": _avail(_fnum(completed) if completed else None),
        "pruned_failed": _avail(_fnum(pruned) if pruned is not None else None),
        "valid_scores": _avail(_fnum(completed) if completed else None),
        "unique_configs_per_window": unique_per_window,
        "discrete_space_cardinality": _avail(cardinality, "indisponible (méthode non grid)"),
        "n_windows": n_windows,
    })


# ---------------------------------------------------------------------------
# Q2 — Diversité
# ---------------------------------------------------------------------------

def compute_q2(
    wfo_results: Mapping[str, Any],
    all_trials: Optional[pd.DataFrame] = None,
    param_grid: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Q2 — diversity: intra-window and inter-window duplicate rates (separate).

    Intra rate counts repeats *within* a window; inter rate counts configs that
    appear in *more than one* window (deduplicated per window first, so intra
    duplicates never contaminate the inter count).
    """
    windows = _get_windows(wfo_results)
    authoritative = _authoritative_params(wfo_results, param_grid, config)

    intra_rates: dict = {}
    intra_dupes: dict = {}
    window_distinct_sigs: list = []
    total_distinct = 0
    for i, window in enumerate(windows):
        wid = _get_window_id(window, i + 1)
        df = _trials_dataframe(window)
        params_cols = _signature_columns(df, authoritative)
        if df.empty or not params_cols:
            intra_rates[str(wid)] = None
            intra_dupes[str(wid)] = None
            continue
        sig = pd.concat([df[p].map(_coerce_param_value) for p in params_cols], axis=1)
        total = int(sig.shape[0])
        distinct = sig.drop_duplicates()
        unique = int(distinct.shape[0])
        dupes = total - unique
        intra_dupes[str(wid)] = dupes
        intra_rates[str(wid)] = (dupes / total) if total else None
        window_distinct_sigs.append(distinct)
        total_distinct += unique

    if window_distinct_sigs:
        pooled = pd.concat(window_distinct_sigs, ignore_index=True)
        unique_all = int(pooled.drop_duplicates().shape[0])
        # inter-window duplicates = distinct configs appearing in >1 window
        inter_dupes = total_distinct - unique_all
        inter_rate = (inter_dupes / total_distinct) if total_distinct else None
    else:
        inter_dupes = None
        inter_rate = None
        unique_all = None

    intra_vals = [v for v in intra_rates.values() if v is not None]
    return sanitize_for_json({
        "id": "Q2",
        "label": "Diversité",
        "source": "window_results[].optimization_trials — signatures de paramètres",
        "intra_window_duplicate_rate": _avail(_mean_of(intra_vals) if intra_vals else None),
        "intra_window_duplicates_by_window": intra_dupes,
        "intra_window_rate_by_window": intra_rates,
        "inter_window_duplicate_rate": _avail(inter_rate),
        "inter_window_duplicates": _avail(_fnum(inter_dupes) if inter_dupes is not None else None),
        "inter_window_unique_configs": _avail(_fnum(unique_all) if unique_all is not None else None),
    })


# ---------------------------------------------------------------------------
# Q3 — Forme de la distribution
# ---------------------------------------------------------------------------

def compute_q3(
    wfo_results: Mapping[str, Any],
    all_trials: Optional[pd.DataFrame] = None,
    param_grid: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Q3 — score distribution shape: P50/P90/P95/max, max−P95 gap, top size."""
    authoritative = _authoritative_params(wfo_results, param_grid, config)
    empty = sanitize_for_json({
        "id": "Q3",
        "label": "Forme de la distribution",
        "source": "combined_score poolé",
        "p50": _avail(None), "p90": _avail(None), "p95": _avail(None),
        "max": _avail(None), "max_minus_p95": _avail(None),
        "top_size": _avail(None), "distinct_configs_in_top": _avail(None),
    })

    raw_scores: list = []
    for window in _get_windows(wfo_results):
        df = _trials_dataframe(window)
        sc = _score_column(df)
        if df.empty or sc is None:
            continue
        raw_scores.extend(pd.to_numeric(df[sc], errors="coerce").tolist())

    # filter non-finite FIRST (F8) — handles NaN and Inf without a take on empty
    finite = [_fnum(s) for s in raw_scores]
    finite = [s for s in finite if s is not None]
    if not finite:
        return empty

    arr = np.asarray(finite, dtype=float)
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    p95 = float(np.percentile(arr, 95))
    mx = float(arr.max())
    max_minus_p95 = mx - p95

    top_size = int((arr >= p95).sum())
    distinct_in_top = _count_distinct_top(wfo_results, threshold=p95, authoritative=authoritative)

    return sanitize_for_json({
        "id": "Q3",
        "label": "Forme de la distribution",
        "source": "combined_score poolé (P50/P90/P95/max)",
        "p50": _avail(p50),
        "p90": _avail(p90),
        "p95": _avail(p95),
        "max": _avail(mx),
        "max_minus_p95": _avail(max_minus_p95),
        "top_size": _avail(_fnum(top_size)),
        "distinct_configs_in_top": _avail(_fnum(distinct_in_top) if distinct_in_top is not None else None),
    })


def _count_distinct_top(wfo_results: Mapping[str, Any], threshold: float, authoritative: Sequence[str]) -> Optional[int]:
    sigs: list = []
    for window in _get_windows(wfo_results):
        df = _trials_dataframe(window)
        sc = _score_column(df)
        params_cols = _signature_columns(df, authoritative)
        if df.empty or sc is None or not params_cols:
            continue
        valid = pd.to_numeric(df[sc], errors="coerce")
        top = df[valid >= threshold]
        if top.empty:
            continue
        sig = pd.concat([top[p].map(_coerce_param_value) for p in params_cols], axis=1)
        sigs.append(sig)
    if not sigs:
        return None
    return int(pd.concat(sigs, ignore_index=True).drop_duplicates().shape[0])


# ---------------------------------------------------------------------------
# Q4 — Stabilité des paramètres
# ---------------------------------------------------------------------------

def compute_q4(
    wfo_results: Mapping[str, Any],
    all_trials: Optional[pd.DataFrame] = None,
    param_grid: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Q4 — parameter stability across windows."""
    windows = _get_windows(wfo_results)
    ranges = _resolve_param_ranges(wfo_results, all_trials, param_grid, config)

    best_params_list: list = []
    for window in windows:
        bp = window.get("best_params") if isinstance(window, Mapping) else None
        if isinstance(bp, Mapping):
            best_params_list.append(bp)

    params_union: set = set()
    for bp in best_params_list:
        params_union.update(bp.keys())

    per_param: dict = {}
    for param in sorted(params_union):
        values: list = []
        for bp in best_params_list:
            v = bp.get(param)
            cv = _coerce_param_value(v)
            if isinstance(cv, (bool, str)):
                values.append(None)
            else:
                f = _fnum(cv)
                values.append(f if f is not None else None)

        n_selected = sum(1 for v in values if v is not None)
        selection_freq = (n_selected / len(values)) if values else None

        finite_vals = [v for v in values if v is not None]
        dispersion = None
        dispersion_type = None
        cv = None
        if len(finite_vals) >= 2:
            arr = np.asarray(finite_vals, dtype=float)
            rng = ranges.get(param)
            if rng is not None and not rng.get("categorical"):
                span = rng["max"] - rng["min"]
                if span > 0:
                    dispersion = float(arr.std(ddof=1) / span)
                    dispersion_type = "std_over_range"
            if dispersion is None:
                iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
                med = float(np.median(arr))
                dispersion = (iqr / abs(med)) if (abs(med) > 0 and iqr > 0) else None
                dispersion_type = "iqr_over_median" if dispersion is not None else None
            mean = float(arr.mean())
            std = float(arr.std(ddof=1))
            if mean > 0 and abs(mean) > 1e-9 and std >= 0:
                cv = std / mean

        stuck_bound = None
        rng = ranges.get(param)
        if rng is not None and not rng.get("categorical") and len(finite_vals) >= 2:
            lo, hi = rng["min"], rng["max"]
            distinct = len({round(v, 9) for v in finite_vals})
            if distinct >= 2:  # « fenêtres où il varie » — the param takes >1 value
                on_lo = sum(1 for v in finite_vals if abs(v - lo) < 1e-9)
                on_hi = sum(1 for v in finite_vals if abs(v - hi) < 1e-9)
                threshold = math.ceil(QUANT_VERDICT_BOUND_FRAC * len(finite_vals))
                if on_lo >= threshold:
                    stuck_bound = "min"
                elif on_hi >= threshold:
                    stuck_bound = "max"

        per_param[param] = {
            "selection_frequency": _avail(selection_freq),
            "n_windows_selected": n_selected,
            "n_windows_total": len(values),
            "dispersion_normalized": _avail(dispersion, None),
            "dispersion_type": dispersion_type,
            "cv": _avail(cv, "non applicable (moyenne non positive)"),
            "stuck_at_bound": stuck_bound,
        }

    return sanitize_for_json({
        "id": "Q4",
        "label": "Stabilité des paramètres",
        "source": "best_params par fenêtre (dispersion normalisée à la plage)",
        "parameters": per_param,
    })


# ---------------------------------------------------------------------------
# Q5 — Voisinage du gagnant
# ---------------------------------------------------------------------------

def compute_q5(
    wfo_results: Mapping[str, Any],
    all_trials: Optional[pd.DataFrame] = None,
    param_grid: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Q5 — winner neighbourhood (vocabulaire §5.3: V, V_perf)."""
    ranges = _resolve_param_ranges(wfo_results, all_trials, param_grid, config)
    windows = _get_windows(wfo_results)
    results: list = []

    for i, window in enumerate(windows):
        wid = _get_window_id(window, i + 1)
        df = _trials_dataframe(window)
        sc = _score_column(df)
        params_cols = _signature_columns(df, list(ranges.keys()))
        results.append(_q5_window(wid, df, sc, params_cols, ranges))

    isolated_degraded = sum(
        1 for e in results if e.get("isolated") is True and e.get("degradation_marquee") is True
    )

    return sanitize_for_json({
        "id": "Q5",
        "label": "Voisinage du gagnant",
        "source": "window_results[].optimization_trials + distance normalisée d(a,b) (§5.3)",
        "windows": results,
        "conjunction_isolated_degraded_windows": isolated_degraded,
    })


def _q5_window(wid: Any, df: pd.DataFrame, sc: Optional[str], params_cols: list, ranges: dict) -> dict:
    base = {
        "window": wid,
        "n_trials": int(len(df)) if not df.empty else 0,
        "winner_score": None,
        "n_V": None,
        "n_Vperf": None,
        "isolated": None,
        "degradation_marquee": None,
        "median_V_ratio": None,
        "z_winner": None,
        "z_median_V": None,
        "robust_neighborhood": None,
        "winner_on_boundary": None,
        "indeterminate": False,
        "raison": None,
    }
    if df.empty or sc is None or not params_cols:
        base["indeterminate"] = True
        base["raison"] = "aucun trial exploitable dans la fenêtre"
        return base

    work = df.copy()
    work["__score"] = pd.to_numeric(work[sc], errors="coerce")
    # filter FINITE scores only — `.notna()` keeps Inf, which must not become the
    # winner (F: review round 2)
    finite_mask = work["__score"].map(lambda x: _fnum(x) is not None)
    work = work[finite_mask].copy()
    if work.empty:
        base["indeterminate"] = True
        base["raison"] = "aucun score fini"
        return base

    work = work.sort_values("__score", ascending=False).reset_index(drop=True)
    sig = pd.concat([work[p].map(_coerce_param_value) for p in params_cols], axis=1)
    work["__sig"] = sig.astype(str).agg("|".join, axis=1)
    work = work.drop_duplicates(subset="__sig", keep="first").reset_index(drop=True)

    winner = work.iloc[0]
    winner_score = float(work.iloc[0]["__score"])
    base["winner_score"] = winner_score
    winner_params = {p: winner[p] for p in params_cols}

    all_scores = [float(work.iloc[i]["__score"]) for i in range(len(work))]

    # distance is undefined when there is no admissible dimension (p == 0),
    # regardless of the trial count (F: review round 3)
    if not ranges:
        base["indeterminate"] = True
        base["raison"] = "distance d non définie (p=0, aucun paramètre à plage positive)"
        base["n_V"] = 0
        return base

    distances = []
    for idx in range(1, len(work)):
        row = work.iloc[idx]
        d = _norm_distance(winner_params, {p: row[p] for p in params_cols}, ranges)
        if d is None:
            continue
        distances.append((idx, d, float(row["__score"])))

    if len(work) > 1 and not distances:
        base["indeterminate"] = True
        base["raison"] = "distance d non définie (paramètres à plage absents des essais)"
        base["n_V"] = 0
        return base

    V = [(idx, d, s) for idx, d, s in distances if d <= NEIGH_DIST_MAX]
    base["n_V"] = len(V)

    comparable = winner_score is not None and math.isfinite(winner_score)

    if comparable and winner_score > 0:
        # nominal case (§5.3 priority rule step 1)
        V_perf = [(idx, d, s) for idx, d, s in V if s >= NEIGH_PERF_RATIO * winner_score]
        if V:
            median_V = _median_of([s for _, _, s in V])
            median_ratio = (median_V / winner_score) if median_V is not None else None
            degradation = bool(median_ratio is not None and median_ratio < DEGRAD_MEDIAN_RATIO)
        else:
            # V = ∅ -> median convention 0 -> degradation marquee (F3, §5.3 line 3)
            median_ratio = 0.0
            degradation = True
        base["median_V_ratio"] = median_ratio
        base["robust_neighborhood"] = bool(
            len(V_perf) >= QUANT_VERDICT_NEIGH_MIN and median_ratio is not None
            and median_ratio >= NEIGH_PERF_RATIO
        )
    else:
        # alternative case (z-score) — §5.3 priority rule step 2
        z_winner = _robust_z(winner_score, all_scores)
        base["z_winner"] = z_winner
        if z_winner is None:
            base["indeterminate"] = True
            base["raison"] = "z non définie (échantillon < 5 ou dispersion nulle)"
            base["n_V"] = len(V)
            return base
        V_perf = []
        for idx, d, s in V:
            zs = _robust_z(s, all_scores)
            if zs is not None and zs >= ALT_VPERF_Z_MIN:
                V_perf.append((idx, d, s))
        median_V = _median_of([s for _, _, s in V])
        z_median_V = _robust_z(median_V, all_scores) if median_V is not None else None
        base["z_median_V"] = z_median_V
        degradation = bool(z_median_V is not None and z_median_V < ALT_DEGRAD_Z_MAX)
        # alternative "5 voisines performantes" = |V_perf| >= 5 AND z(median(V)) >= 0
        base["robust_neighborhood"] = bool(
            len(V_perf) >= QUANT_VERDICT_NEIGH_MIN and z_median_V is not None and z_median_V >= ALT_VPERF_Z_MIN
        )

    base["n_Vperf"] = len(V_perf)
    base["isolated"] = len(V_perf) < NEIGH_ISOLATED_LT
    base["degradation_marquee"] = degradation

    on_boundary = False
    for param, rng in ranges.items():
        if rng.get("categorical"):
            continue
        wv = _fnum(winner_params.get(param))
        if wv is None:
            continue
        if abs(wv - rng["min"]) < 1e-9 or abs(wv - rng["max"]) < 1e-9:
            on_boundary = True
            break
    base["winner_on_boundary"] = on_boundary
    return base


def _norm_distance(a: Mapping[str, Any], b: Mapping[str, Any], ranges: dict) -> Optional[float]:
    """Normalised distance ``d(a, b) = (1/p) Σ |a_i − b_i| / (max_i − min_i)``.

    Categorical params contribute 0/1; fixed dimensions (zero range) and params
    absent from either config are excluded from the sum.  ``p == 0`` -> None.
    """
    p = 0
    total = 0.0
    for param, rng in ranges.items():
        av = _coerce_param_value(a.get(param))
        bv = _coerce_param_value(b.get(param))
        if av is None or bv is None:
            continue
        if rng.get("categorical") or isinstance(av, (bool, str)) or isinstance(bv, (bool, str)):
            p += 1
            total += 0.0 if av == bv else 1.0
            continue
        fa = _fnum(av)
        fb = _fnum(bv)
        if fa is None or fb is None:
            continue
        span = rng["max"] - rng["min"]
        if span <= 0:
            continue
        p += 1
        total += abs(fa - fb) / span
    if p == 0:
        return None
    return total / p


def _robust_z(x: Optional[float], S: Sequence[float]) -> Optional[float]:
    """Robust z-score (§5.3) with IQR fallback; None when undefined."""
    if x is None:
        return None
    arr = np.asarray([s for s in S if s is not None and math.isfinite(float(s))], dtype=float)
    if arr.size < ZSCORE_MIN_SAMPLE:
        return None
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad > 0:
        return (x - med) / (ZSCORE_CONSISTENCY * mad)
    q75, q25 = float(np.percentile(arr, 75)), float(np.percentile(arr, 25))
    iqr = q75 - q25
    if iqr > 0:
        return (x - med) / (iqr / ZSCORE_IQR_DIVISOR)
    return None


# ---------------------------------------------------------------------------
# Q6 — Érosion IS → OOS
# ---------------------------------------------------------------------------

def compute_q6(wfo_results: Mapping[str, Any], per_bar_returns: Optional[Mapping[str, Any]] = None) -> dict:
    """Q6 — IS→OOS erosion, per window then aggregated."""
    is_perf = {r.get("window"): r for r in (wfo_results.get("in_sample_performance") or []) if isinstance(r, Mapping)}
    oos_perf = {r.get("window"): r for r in (wfo_results.get("out_of_sample_performance") or []) if isinstance(r, Mapping)}

    rows: list = []
    erosion_sharpe: list = []
    erosion_return: list = []
    degradation_count = 0
    n_evaluable = 0

    for wid in sorted(set(is_perf) | set(oos_perf)):
        is_row = is_perf.get(wid) or {}
        oos_row = oos_perf.get(wid) or {}
        r = _q6_window(wid, is_row, oos_row)
        rows.append(r)
        es = r.get("erosion_sharpe_pct", {}).get("value")
        er = r.get("erosion_return_pct", {}).get("value")
        if es is not None:
            erosion_sharpe.append(es)
            n_evaluable += 1
        if er is not None:
            erosion_return.append(er)
        if r.get("degradation") is True:
            degradation_count += 1

    # dispersion of the erosion values (§5.1 Q6)
    erosion_dispersion = None
    if len(erosion_sharpe) >= 2:
        arr = np.asarray(erosion_sharpe, dtype=float)
        erosion_dispersion = float(arr.std(ddof=1))

    return sanitize_for_json({
        "id": "Q6",
        "label": "Érosion IS → OOS",
        "source": "in_sample_performance vs out_of_sample_performance (par fenêtre appariée)",
        "windows": rows,
        "median_erosion_sharpe_pct": _avail(_median_of(erosion_sharpe)),
        "median_erosion_return_pct": _avail(_median_of(erosion_return)),
        "erosion_sharpe_dispersion": _avail(erosion_dispersion),
        "n_windows_evaluable": n_evaluable,
        "n_windows_undefined": len(rows) - n_evaluable,
        "windows_in_degradation": degradation_count,
        "n_windows": len(rows),
    })


def _q6_window(wid: Any, is_row: Mapping[str, Any], oos_row: Mapping[str, Any]) -> dict:
    row = {
        "window": wid,
        "sharpe_is": _fnum(is_row.get("sharpe")),
        "sharpe_oos": _fnum(oos_row.get("sharpe")),
        "return_is": _fnum(is_row.get("return")),
        "return_oos": _fnum(oos_row.get("return")),
        "pqs_is": _fnum(is_row.get("pqs")),
        "pqs_oos": _fnum(oos_row.get("pqs")),
        "dd_is": _fnum(is_row.get("max_drawdown")),
        "dd_oos": _fnum(oos_row.get("max_drawdown")),
        "erosion_sharpe_pct": _avail(None),
        "erosion_return_pct": _avail(None),
        "erosion_pqs_pct": _avail(None),
        "degradation": None,
    }

    s_is = row["sharpe_is"]
    s_oos = row["sharpe_oos"]
    if s_is is not None and s_is > 0 and s_oos is not None:
        er = (s_is - s_oos) / s_is * 100.0
        row["erosion_sharpe_pct"] = _avail(er)
        row["degradation"] = er > QUANT_VERDICT_EROSION_WATCH_PCT
    else:
        row["erosion_sharpe_pct"] = _avail(None, "dénominateur non positif ou OOS absent")

    r_is = row["return_is"]
    r_oos = row["return_oos"]
    if r_is is not None and r_is != 0 and r_oos is not None:
        row["erosion_return_pct"] = _avail((r_is - r_oos) / r_is * 100.0)
    else:
        row["erosion_return_pct"] = _avail(None, "dénominateur nul ou OOS absent")

    p_is = row["pqs_is"]
    p_oos = row["pqs_oos"]
    if p_is is not None and p_is != 0 and p_oos is not None:
        row["erosion_pqs_pct"] = _avail((p_is - p_oos) / p_is * 100.0)
    else:
        row["erosion_pqs_pct"] = _avail(None, "dénominateur nul ou PQS absent")

    dd_is = row["dd_is"]
    dd_oos = row["dd_oos"]
    if dd_is is not None and dd_oos is not None and dd_is != 0:
        row["dd_amplification_ratio"] = _avail(abs(dd_oos) / abs(dd_is))
    else:
        row["dd_amplification_ratio"] = _avail(None, "un des DD absent ou nul")

    return row


# ---------------------------------------------------------------------------
# Q7 — Sharpe non-annualisé homogène
# ---------------------------------------------------------------------------

def compute_q7(
    wfo_results: Mapping[str, Any],
    per_bar_returns: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Q7 — homogeneous non-annualised Sharpe, recomputed from per-bar returns."""
    is_perf = {r.get("window"): r for r in (wfo_results.get("in_sample_performance") or []) if isinstance(r, Mapping)}
    oos_perf = {r.get("window"): r for r in (wfo_results.get("out_of_sample_performance") or []) if isinstance(r, Mapping)}

    pbr = per_bar_returns if isinstance(per_bar_returns, Mapping) else {}
    is_series = pbr.get("is") if isinstance(pbr.get("is"), Mapping) else {}
    oos_series = pbr.get("oos") if isinstance(pbr.get("oos"), Mapping) else {}
    final_series = pbr.get("final")

    rows: list = []
    for wid in sorted(set(is_perf) | set(oos_perf)):
        sh_is = _sharpe_from_returns(is_series.get(wid))
        sh_oos = _sharpe_from_returns(oos_series.get(wid))
        rows.append({
            "window": wid,
            "sharpe_is_recomputed": _avail(sh_is, "série de rendements IS absente"),
            "sharpe_oos_recomputed": _avail(sh_oos, "série de rendements OOS absente"),
            "sharpe_is_reported": _fnum(is_perf.get(wid, {}).get("sharpe")),
            "sharpe_oos_reported": _fnum(oos_perf.get(wid, {}).get("sharpe")),
        })

    final_sharpe = _sharpe_from_returns(final_series)
    return sanitize_for_json({
        "id": "Q7",
        "label": "Sharpe non-annualisé homogène",
        "source": "mean/std(ddof=1) des rendements par barre (IS/OOS/Final)",
        "windows": rows,
        "final_sharpe_recomputed": _avail(final_sharpe, "série de rendements Final absente"),
        "final_sharpe_reported": _fnum((wfo_results.get("final_backtest_metrics") or {}).get("sharpe")),
    })


# ---------------------------------------------------------------------------
# Q8 — Sensibilité aux coûts
# ---------------------------------------------------------------------------

def compute_q8(
    wfo_results: Mapping[str, Any],
    final_trades: Optional[pd.DataFrame] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Q8 — cost sensitivity (§5.1 Q8, §5.3).  No ruin threshold from an edge."""
    merged = _merge_config(wfo_results, config)
    costs = _resolve_cost_convention(merged)
    convention = costs["convention"]
    costs_included = costs["costs_included"]
    cost_per_side_pct = costs["cost_per_side_pct"]
    fees_pct = costs["fees_pct"]
    slippage_bps = costs["slippage_bps"]

    oos_returns = [_fnum(r.get("return")) for r in (wfo_results.get("out_of_sample_performance") or []) if isinstance(r, Mapping)]
    oos_returns = [v for v in oos_returns if v is not None]
    # compounded (portfolio-equivalent) OOS return — NOT a plain sum of % (§5.1 Q8 / F6)
    base_oos_return = _compounded_return_pct(oos_returns)

    n_oos_trades = 0
    for r in (wfo_results.get("out_of_sample_performance") or []):
        if isinstance(r, Mapping):
            n_oos_trades += int(_fnum(r.get("n_trades")) or 0)

    result = {
        "id": "Q8",
        "label": "Sensibilité aux coûts",
        "source": "config.fees_pct / config.slippage_bps + out_of_sample_performance",
        "convention": convention,
        "costs_included": costs_included,
        "cost_per_side_pct": _avail(cost_per_side_pct),
        "fees_pct": _avail(fees_pct),
        "slippage_bps": _avail(slippage_bps),
        "base_oos_return_pct": _avail(base_oos_return, None),
        "base_oos_return_aggregation": "compounded",
        "additional_cost_scenario": None,
        "net_margin_per_side": None,
        "note": None,
    }

    if convention == "net_of_costs":
        # one extra round-trip (2 sides) per OOS trade — approximation, no re-run
        extra_cost_per_roundtrip_pct = cost_per_side_pct * 2.0
        approx_drag = extra_cost_per_roundtrip_pct * n_oos_trades
        scenario = base_oos_return - approx_drag if base_oos_return is not None else None
        result["cost_already_included_per_side_pct"] = _avail(cost_per_side_pct)
        result["additional_cost_scenario"] = {
            "label": "approximation explicite (sans réexécution)",
            "extra_cost_per_roundtrip_pct": extra_cost_per_roundtrip_pct,
            "approx_total_drag_pct": approx_drag,
            "oos_return_after_stress_pct": _avail(scenario, "rendement OOS de base indisponible"),
        }
        result["note"] = "coûts déjà inclus ; scénario additionnel approximé (pas de réexécution)"
    elif convention == "gross":
        result["net_margin_per_side"] = {
            "cost_per_side_pct": cost_per_side_pct,
            "note": "rendements bruts : aucune marge de coût déduite (coûts de base omis)",
        }
        result["note"] = "rendements bruts — coûts de base omis"
    else:
        result["note"] = (
            "convention des coûts inconnue (fees_pct/slippage_bps absents)"
            if convention == "unknown" else
            "composante de coût invalide (fees_pct/slippage_bps non numérique)"
        )

    return sanitize_for_json(result)


# ---------------------------------------------------------------------------
# P1 — Incertitude OOS
# ---------------------------------------------------------------------------

def compute_oos_uncertainty(
    wfo_results: Mapping[str, Any],
    final_trades: Optional[pd.DataFrame] = None,
    per_bar_returns: Optional[Mapping[str, Any]] = None,
    oos_trades: Optional[pd.DataFrame] = None,
    *,
    seed: int = 42,
    n_boot: int = 2000,
) -> dict:
    """P1 — OOS uncertainty: sample size, bootstrap CI, P&L concentration.

    ``final_trades`` is the Final-backtest trades (a distinct perimeter); the
    trade-concentration branch requires ``oos_trades`` (per-trade OOS P&L) of the
    same perimeter as the OOS windows.  Without ``oos_trades`` that branch is
    reported ``indisponible`` — a Final-backtest branch must never certify the
    absence of OOS concentration (§5.3, F4/F5).
    """
    oos_perf = wfo_results.get("out_of_sample_performance") or []
    n_windows = len(oos_perf)
    n_trades = 0
    oos_returns: list = []
    for r in oos_perf:
        if not isinstance(r, Mapping):
            continue
        n_trades += int(_fnum(r.get("n_trades")) or 0)
        ret = _fnum(r.get("return"))
        if ret is not None:
            oos_returns.append(ret)

    ci_low = ci_high = mean_ret = None
    if len(oos_returns) >= 2:
        rng = np.random.default_rng(seed)
        arr = np.asarray(oos_returns, dtype=float)
        means = []
        for _ in range(n_boot):
            sample = rng.choice(arr, size=len(arr), replace=True)
            means.append(float(sample.mean()))
        means = np.asarray(means)
        mean_ret = float(arr.mean())
        ci_low = float(np.percentile(means, 2.5))
        ci_high = float(np.percentile(means, 97.5))

    concentration = _concentration_status(oos_trades, oos_returns)

    return sanitize_for_json({
        "id": "OOS_UNCERTAINTY",
        "label": "Incertitude OOS",
        "source": "out_of_sample_performance + final_trades (bootstrap par fenêtre)",
        "n_windows": n_windows,
        "n_trades_oos": n_trades,
        "oos_return_mean_pct": _avail(mean_ret),
        "oos_return_ci95_low_pct": _avail(ci_low),
        "oos_return_ci95_high_pct": _avail(ci_high),
        "bootstrap": {"n_boot": n_boot, "seed": seed, "method": "resample_windows"},
        "concentration": concentration,
    })


def _concentration_status(oos_trades: Optional[pd.DataFrame], oos_returns: list) -> dict:
    """P&L concentration (§5.3) — two independent, unit-consistent checks.

    Trade branch (same $ unit): the top-k **OOS** trades cumulate > 50% of the
    total OOS-trade P&L.  Window branch (same % unit): the 2 best OOS windows
    cumulate > 60% of the total OOS P&L.  Each branch uses its **own**
    denominator of the same perimeter and unit (F5).  The trade branch is
    ``indisponible`` when ``oos_trades`` is not provided — the Final-backtest
    trades are a distinct perimeter and are never used as a proxy (F4/F5).
    ``indetermine`` only when no branch can be evaluated.
    """
    status = {
        "status": "indetermine",
        "raison": None,
        "top_k_trades_frac": None,
        "top2_windows_frac": None,
        "trade_branch_available": False,
    }

    trade_frac = None
    trade_evaluable = False
    if isinstance(oos_trades, pd.DataFrame) and not oos_trades.empty:
        col = "pnl" if "pnl" in oos_trades.columns else ("return" if "return" in oos_trades.columns else None)
        if col is not None:
            pnl = pd.to_numeric(oos_trades[col], errors="coerce").dropna()
            total_trade = float(pnl.sum()) if len(pnl) else 0.0
            if total_trade > 0:
                trade_evaluable = True
                n = int(len(pnl))
                k = max(1, math.ceil(0.10 * n))
                top_k_sum = float(pnl.nlargest(k).sum())
                trade_frac = top_k_sum / total_trade

    window_frac = None
    window_evaluable = False
    if len(oos_returns) >= 2:
        total_oos = float(np.sum(oos_returns))
        if total_oos > 0:
            window_evaluable = True
            top2 = float(np.sum(np.sort(oos_returns)[::-1][:2]))
            window_frac = top2 / total_oos

    status["top_k_trades_frac"] = trade_frac
    status["top2_windows_frac"] = window_frac
    status["trade_branch_available"] = trade_evaluable

    if not trade_evaluable and not window_evaluable:
        if oos_trades is None:
            status["raison"] = (
                "trades OOS par trade non fournis et P&L OOS total nul/négatif — critère non défini"
            )
        else:
            status["raison"] = "P&L total nul/négatif — critère non défini"
        return status

    concentrated = False
    if trade_evaluable and trade_frac is not None and trade_frac > QUANT_VERDICT_CONCENTRATION_TRADE_FRAC:
        concentrated = True
    if window_evaluable and window_frac is not None and window_frac > QUANT_VERDICT_CONCENTRATION_WINDOW_FRAC:
        concentrated = True

    status["status"] = "concentre" if concentrated else "non_concentre"
    return status


# ---------------------------------------------------------------------------
# Pré-verdict déterministe (§5.3)
# ---------------------------------------------------------------------------

def _partial_metric_signals(indicators: Mapping[str, Any]) -> list:
    """Detect partial-metric availability signals (§5.3 case b).

    A metric that is unavailable but whose run remains exploitable must cap the
    verdict at WATCH.  Covers Q6 erosion partiality and Q7 missing per-bar
    Sharpe series (IS/OOS/Final) — not only the erosion path.
    """
    signals: list = []
    q6 = indicators.get("Q6", {})
    n_undef = _fnum(q6.get("n_windows_undefined"))
    if n_undef is not None and n_undef > 0:
        signals.append(f"érosion Sharpe partielle ({int(n_undef)} fenêtre(s) non évaluable(s))")
    q7 = indicators.get("Q7", {})
    if (q7.get("final_sharpe_recomputed") or {}).get("available") is False:
        signals.append("Sharpe Final indisponible (rendements par barre absents)")
    for w in (q7.get("windows") or []):
        if not isinstance(w, dict):
            continue
        if (w.get("sharpe_is_recomputed") or {}).get("available") is False \
                or (w.get("sharpe_oos_recomputed") or {}).get("available") is False:
            signals.append("Sharpe par barre indisponible (certaines fenêtres IS/OOS)")
            break
    return signals


def compute_pre_verdict(
    indicators: Mapping[str, Any],
    integrity: Mapping[str, Any],
    wfo_results: Mapping[str, Any],
) -> dict:
    """Deterministic local pre-verdict GO/WATCH/NO_GO (§5.3 thresholds)."""
    criteria: dict = {}

    if not integrity.get("ok", False):
        criteria["C1_integrite"] = {"verdict": "NO_GO", "raison": "; ".join(integrity.get("causes", []))}
        return {
            "verdict": "NO_GO",
            "scope": VERDICT_SCOPE,
            "status": "non_evaluable",
            "criteria": criteria,
            "rationale": ["Contrôle d'intégrité en échec — non évaluable (pré-verdict local NO_GO)."],
        }
    criteria["C1_integrite"] = {"verdict": "GO", "raison": None}

    # C2 — échantillon OOS
    oos = indicators.get("OOS_UNCERTAINTY", {})
    n_windows = _fnum(oos.get("n_windows"))
    n_trades = _fnum(oos.get("n_trades_oos"))
    concentration = (oos.get("concentration") or {}).get("status")
    if n_windows is None or n_trades is None:
        c2 = {"verdict": "WATCH", "raison": "effectif OOS indéterminé"}
    elif n_windows >= QUANT_VERDICT_MIN_WINDOWS_GO and n_trades >= QUANT_VERDICT_MIN_TRADES_GO and concentration == "non_concentre":
        c2 = {"verdict": "GO", "raison": None}
    elif n_windows < QUANT_VERDICT_MIN_WINDOWS_WATCH or n_trades < QUANT_VERDICT_MIN_TRADES_WATCH:
        c2 = {"verdict": "NO_GO", "raison": f"effectif OOS insuffisant ({n_windows} fenêtres, {n_trades} trades)"}
    else:
        c2 = {"verdict": "WATCH", "raison": "effectif OOS intermédiaire ou concentration"}
    criteria["C2_echantillon_oos"] = c2

    # C3 — érosion Sharpe
    q6 = indicators.get("Q6", {})
    med_erosion = q6.get("median_erosion_sharpe_pct", {}).get("value")
    degradation_count = _fnum(q6.get("windows_in_degradation"))
    n_evaluable = _fnum(q6.get("n_windows_evaluable"))
    oos_sharpes = [_fnum(r.get("sharpe")) for r in (wfo_results.get("out_of_sample_performance") or []) if isinstance(r, Mapping)]
    oos_sharpes = [v for v in oos_sharpes if v is not None]
    med_oos_sharpe = _median_of(oos_sharpes)
    oos_returns = [_fnum(r.get("return")) for r in (wfo_results.get("out_of_sample_performance") or []) if isinstance(r, Mapping)]
    oos_returns = [v for v in oos_returns if v is not None]
    majority_positive = (sum(1 for v in oos_returns if v > 0) > QUANT_VERDICT_MAJORITY_FRAC * len(oos_returns)) if oos_returns else False

    # recurrence over *evaluable* windows only (F9)
    recurrent = (
        degradation_count is not None and n_evaluable is not None and n_evaluable > 0
        and (degradation_count / n_evaluable) >= QUANT_VERDICT_EROSION_RECURRENT_FRAC
    )
    if (med_oos_sharpe is not None and med_oos_sharpe <= 0) or recurrent:
        c3 = {"verdict": "NO_GO",
              "raison": "médiane Sharpe OOS ≤ 0" if (med_oos_sharpe is not None and med_oos_sharpe <= 0)
              else "érosion du Sharpe > 50 % récurrente"}
    elif med_erosion is not None and med_erosion < QUANT_VERDICT_EROSION_GO_PCT and majority_positive:
        c3 = {"verdict": "GO", "raison": None}
    elif med_erosion is not None and med_erosion <= QUANT_VERDICT_EROSION_WATCH_PCT:
        c3 = {"verdict": "WATCH", "raison": "érosion intermédiaire (30–50 %) ou fenêtres mitigées"}
    elif med_erosion is None:
        c3 = {"verdict": "WATCH", "raison": "érosion non calculable (dénominateurs non positifs)"}
    else:
        c3 = {"verdict": "WATCH", "raison": "érosion marquée mais non récurrente"}
    # partial-metric cap (§5.3 case b): some windows non-evaluable for the Sharpe
    # erosion -> the criterion can never be GO
    n_undefined = _fnum(q6.get("n_windows_undefined"))
    if n_undefined is not None and n_undefined > 0 and c3["verdict"] == "GO":
        c3 = {"verdict": "WATCH", "raison": f"érosion partiellement évaluable ({int(n_undefined)} fenêtre(s) non évaluable(s))"}
    criteria["C3_erosion_sharpe"] = c3

    # C4 — résultat net OOS + coûts
    q8 = indicators.get("Q8", {})
    convention = q8.get("convention")
    costs_included = bool(q8.get("costs_included"))
    base_oos = q8.get("base_oos_return_pct", {}).get("value")
    stress = None
    scenario = q8.get("additional_cost_scenario")
    if isinstance(scenario, dict):
        stress = scenario.get("oos_return_after_stress_pct", {}).get("value")
    if convention == "unknown" or (convention != "net_of_costs" and not costs_included):
        c4 = {"verdict": "NO_GO", "raison": "coûts de base omis ou provenance inconnue (rendements bruts)"}
    elif base_oos is not None and base_oos <= 0:
        c4 = {"verdict": "NO_GO", "raison": "résultat net OOS ≤ 0 en base"}
    elif base_oos is not None and base_oos > 0 and stress is not None and stress > 0:
        c4 = {"verdict": "GO", "raison": None}
    elif base_oos is not None and base_oos > 0:
        c4 = {"verdict": "WATCH", "raison": "positif en base, nul/négatif sous stress ou coût non testable"}
    else:
        c4 = {"verdict": "WATCH", "raison": "résultat net OOS indéterminé"}
    criteria["C4_resultat_net_oos"] = c4

    # C5 — robustesse de sélection (nominal + alternative, F4)
    q5 = indicators.get("Q5", {})
    conjunction = _fnum(q5.get("conjunction_isolated_degraded_windows"))
    any_stuck_bound = False
    for p, d in (indicators.get("Q4", {}).get("parameters", {})).items():
        if isinstance(d, dict) and d.get("stuck_at_bound") in ("min", "max"):
            any_stuck_bound = True
    n_robust_windows = sum(
        1 for w in (q5.get("windows", []) or [])
        if isinstance(w, dict) and w.get("robust_neighborhood") is True
    )

    if conjunction is not None and conjunction >= CONJ_WINDOWS_MIN:
        c5 = {"verdict": "NO_GO", "raison": f"gagnant isolé + dégradation marquée dans {int(conjunction)} fenêtres"}
    elif any_stuck_bound:
        c5 = {"verdict": "WATCH", "raison": "paramètre systématiquement en borne"}
    elif n_robust_windows >= 1:
        c5 = {"verdict": "GO", "raison": None}
    else:
        c5 = {"verdict": "WATCH", "raison": "plateau peu peuplé ou dispersion sensible au régime"}
    criteria["C5_robustesse_selection"] = c5

    verdicts = [c["verdict"] for c in criteria.values()]
    if "NO_GO" in verdicts:
        verdict = "NO_GO"
    elif all(v == "GO" for v in verdicts):
        verdict = "GO"
    else:
        verdict = "WATCH"

    # §5.3 case b — a partial (unavailable but run-exploitable) metric caps the
    # overall verdict at WATCH, never GO (P0)
    partial_signals = _partial_metric_signals(indicators)
    if partial_signals and verdict == "GO":
        verdict = "WATCH"

    rationale = [
        f"{key}: {c['verdict']} — {c.get('raison') or 'critère non satisfait'}"
        for key, c in criteria.items() if c["verdict"] != "GO"
    ]
    if partial_signals and verdict != "GO":
        rationale.append(f"Métrique partielle : {'; '.join(partial_signals)}")

    return {
        "verdict": verdict,
        "scope": VERDICT_SCOPE,
        "status": "ok",
        "criteria": criteria,
        "rationale": rationale or ["Tous les garde-fous applicables passent."],
    }


# ---------------------------------------------------------------------------
# Orchestrateur
# ---------------------------------------------------------------------------

def compute_quant_indicators(
    wfo_results: Mapping[str, Any],
    *,
    all_trials: Optional[pd.DataFrame] = None,
    final_trades: Optional[pd.DataFrame] = None,
    per_bar_returns: Optional[Mapping[str, Any]] = None,
    oos_trades: Optional[pd.DataFrame] = None,
    config: Optional[Mapping[str, Any]] = None,
    param_grid: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Compute the manifest, integrity check, Q1–Q8, OOS uncertainty and the
    deterministic pre-verdict for a finished WFO run."""
    manifest = build_run_manifest(
        wfo_results, config,
        all_trials=all_trials, final_trades=final_trades,
        per_bar_returns=per_bar_returns, oos_trades=oos_trades, param_grid=param_grid,
    )
    integrity = check_run_integrity(
        wfo_results,
        all_trials=all_trials, final_trades=final_trades,
        per_bar_returns=per_bar_returns, oos_trades=oos_trades, config=config,
        param_grid=param_grid,
    )

    indicators = {
        "Q1": compute_q1(wfo_results, all_trials, param_grid, config),
        "Q2": compute_q2(wfo_results, all_trials, param_grid, config),
        "Q3": compute_q3(wfo_results, all_trials, param_grid, config),
        "Q4": compute_q4(wfo_results, all_trials, param_grid, config),
        "Q5": compute_q5(wfo_results, all_trials, param_grid, config),
        "Q6": compute_q6(wfo_results, per_bar_returns),
        "Q7": compute_q7(wfo_results, per_bar_returns),
        "Q8": compute_q8(wfo_results, final_trades, config),
        "OOS_UNCERTAINTY": compute_oos_uncertainty(wfo_results, final_trades, per_bar_returns, oos_trades),
    }

    pre_verdict = compute_pre_verdict(indicators, integrity, wfo_results)

    return sanitize_for_json({
        "schema_version": SCHEMA_VERSION,
        "indicator_version": INDICATOR_VERSION,
        "generated_at": utc_now_iso(),
        "manifest": manifest,
        "integrity": integrity,
        "indicators": indicators,
        "pre_verdict": pre_verdict,
    })
