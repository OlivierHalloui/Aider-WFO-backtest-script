"""Runtime adapter for first executable Pine-imported strategy (V3 block 1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import importlib
import importlib.util
import os
import re
from typing import Any

import numpy as np
import pandas as pd
import vectorbtpro as vbt

from config import DEFAULT_PARAM_GRID
from indicators import CrossBBWLowSignal, SMAExit
from pine_v3.mtf import request_security_series


_TEST_ID_ALIASES = {
    "pine_strategy_test",
    "strategy_test",
    "bb_sma_only",
    "pine_atdmf_strategy_bb_sma_only_long",
}


def _normalize_strategy_id(value: str | None) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _load_strategy_test_sha1() -> str:
    """Best-effort source fingerprint for `docs/wfoe_v3/strategy_test.txt`."""
    try:
        repo_root = Path(__file__).resolve().parents[3]
        candidate = repo_root / "docs" / "wfoe_v3" / "strategy_test.txt"
        if not candidate.exists():
            return ""
        raw = candidate.read_text(encoding="utf-8", errors="ignore")
        return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()
    except Exception:
        return ""


_TEST_SOURCE_SHA1 = _load_strategy_test_sha1()


def _repo_root_dir() -> str:
    return str(Path(__file__).resolve().parents[3])


def _load_python_target(target: str):
    """
    Load a Python mapping target from:
    - dotted module path (`package.module`)
    - file path (`relative/or/abs/path.py`)
    """
    value = str(target or "").strip()
    if not value:
        raise ValueError("Empty mapping target.")

    if value.endswith(".py") or "/" in value or "\\" in value:
        path = value
        if not os.path.isabs(path):
            path = os.path.join(_repo_root_dir(), path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Mapped file not found: {path}")
        module_name = f"pine_ext_{hashlib.sha1(path.encode('utf-8')).hexdigest()[:10]}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot build import spec from file: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return importlib.import_module(value)


def _resolve_external_library_bindings(config: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    """
    Resolve external Pine import aliases to loaded Python modules.
    """
    cfg = config if isinstance(config, dict) else {}
    mapping = cfg.get("pine_import_mapping")
    mapping = mapping if isinstance(mapping, dict) else {}
    precheck = cfg.get("pine_precheck_report")
    precheck = precheck if isinstance(precheck, dict) else {}
    import_resolution = precheck.get("import_resolution")
    import_resolution = import_resolution if isinstance(import_resolution, list) else []

    aliases: list[str] = []
    if import_resolution:
        for row in import_resolution:
            if isinstance(row, dict):
                alias = str(row.get("alias") or "").strip()
                if alias:
                    aliases.append(alias)
    if not aliases:
        # Fallback: infer aliases from mapping keys.
        aliases = [str(k).strip() for k in mapping.keys() if str(k).strip()]

    bindings: dict[str, Any] = {}
    warnings: list[str] = []
    for alias in aliases:
        target = ""
        if alias in mapping:
            target = str(mapping.get(alias) or "").strip()
        elif alias.lower() in mapping:
            target = str(mapping.get(alias.lower()) or "").strip()
        if not target:
            continue
        try:
            bindings[alias] = _load_python_target(target)
        except Exception as e:
            warnings.append(f"Import alias '{alias}' mapping failed: {e}")
    return bindings, warnings


def _safe_series_like(value: Any, index: pd.Index):
    """
    Convert values returned by mapped external functions to a bool series aligned on index.
    """
    if isinstance(value, pd.Series):
        return value.reindex(index)
    if isinstance(value, pd.DataFrame):
        if value.shape[1] == 0:
            return pd.Series(False, index=index)
        return value.iloc[:, 0].reindex(index)
    try:
        arr = np.asarray(value).reshape(-1)
        if len(arr) == len(index):
            return pd.Series(arr, index=index)
    except Exception:
        pass
    return pd.Series(False, index=index)


def _extract_signal_from_external_result(result: Any, index: pd.Index):
    if isinstance(result, tuple) and len(result) >= 3:
        return _safe_series_like(result[2], index=index)
    if isinstance(result, list) and len(result) >= 3:
        return _safe_series_like(result[2], index=index)
    return _safe_series_like(result, index=index)


def supports_strategy_test_runtime(strategy_id: str | None, config: dict[str, Any] | None = None) -> bool:
    """
    Decide whether the current pine_imported config is supported by runtime block 1.

    Current support scope: strategy_test.txt (BB + SMA-only long test strategy).
    """
    sid = _normalize_strategy_id(strategy_id)
    if sid in _TEST_ID_ALIASES:
        return True

    cfg = config or {}
    source_name = str(cfg.get("pine_source_name") or "").strip().lower()
    source_file = str(cfg.get("pine_file_path") or "").strip().lower()
    if source_name and os.path.basename(source_name) in {"strategy_test.txt", "strategy_test.pine"}:
        return True
    if source_file and os.path.basename(source_file) in {"strategy_test.txt", "strategy_test.pine"}:
        return True

    source_sha1 = str(cfg.get("pine_source_sha1") or "").strip().lower()
    if _TEST_SOURCE_SHA1 and source_sha1 == _TEST_SOURCE_SHA1:
        return True
    return False


def _is_array_like(value: Any) -> bool:
    return hasattr(value, "__len__") and not isinstance(value, (str, bytes, dict))


def _broadcast(value: Any, length: int) -> np.ndarray:
    if _is_array_like(value):
        arr = np.asarray(value).reshape(-1)
        if len(arr) == 0:
            raise ValueError("Empty parameter array is not allowed.")
        if len(arr) == 1:
            return np.full(length, arr[0])
        if len(arr) != length:
            raise ValueError(f"Parameter length mismatch: expected {length}, got {len(arr)}.")
        return arr
    return np.full(length, value)


def _scalarize(value: Any) -> Any:
    if not _is_array_like(value):
        return value
    arr = np.asarray(value).reshape(-1)
    if len(arr) == 0:
        raise ValueError("Empty parameter array is not allowed.")
    out = arr[0]
    return out.item() if isinstance(out, np.generic) else out


def _normalize_columns(obj: Any) -> Any:
    if hasattr(obj, "columns"):
        out = obj.copy()
        out.columns = pd.RangeIndex(len(out.columns))
        return out
    return obj


def _align_series_to_columns(series: pd.Series, reference_obj: Any) -> Any:
    if not hasattr(reference_obj, "columns"):
        return series
    cols = reference_obj.columns
    aligned = pd.concat([series] * len(cols), axis=1)
    aligned.columns = pd.RangeIndex(len(cols))
    return aligned


def create_strategy_test_signals(
    df: pd.DataFrame,
    external_bindings: dict[str, Any] | None = None,
    **params: Any,
) -> dict[str, Any]:
    """Generate entry/exit signals for strategy_test Pine runtime."""
    timeperiod = params.get("timeperiod", 20)
    stdev = params.get("StDev", 2.0)
    fenetre_lowest = params.get("fenetre_lowest", 30)
    seuil_lowest = params.get("seuil_lowest", 3.5)
    user_exit_sma_length = params.get("user_exit_sma_length", 20)
    exit_sma_enabled = params.get("exit_sma_enabled", True)
    mtf_filter_enabled = params.get("mtf_filter_enabled", False)
    mtf_filter_timeframe = params.get("mtf_filter_timeframe", "")
    mtf_filter_sma_length = params.get("mtf_filter_sma_length", 20)
    mtf_filter_timing = params.get("mtf_filter_timing", "closing")
    mtf_filter_gaps = params.get("mtf_filter_gaps", "off")
    mtf_filter_mode = params.get("mtf_filter_mode", "above")

    all_params = [
        timeperiod,
        stdev,
        fenetre_lowest,
        seuil_lowest,
        user_exit_sma_length,
        exit_sma_enabled,
        mtf_filter_enabled,
        mtf_filter_sma_length,
    ]
    vector_len = 1
    lengths: list[int] = []
    for p in all_params:
        if _is_array_like(p):
            try:
                lengths.append(len(p))
            except Exception:
                pass
    non_scalar_lengths = sorted({l for l in lengths if l > 1})
    if len(non_scalar_lengths) > 1:
        raise ValueError(f"Inconsistent vectorized parameter lengths: {non_scalar_lengths}")
    if non_scalar_lengths:
        vector_len = non_scalar_lengths[0]

    if vector_len > 1:
        timeperiod = _broadcast(timeperiod, vector_len)
        stdev = _broadcast(stdev, vector_len)
        fenetre_lowest = _broadcast(fenetre_lowest, vector_len)
        seuil_lowest = _broadcast(seuil_lowest, vector_len)
        user_exit_sma_length = _broadcast(user_exit_sma_length, vector_len)
        exit_sma_enabled = _broadcast(exit_sma_enabled, vector_len)
        mtf_filter_enabled = _broadcast(mtf_filter_enabled, vector_len)
        mtf_filter_sma_length = _broadcast(mtf_filter_sma_length, vector_len)
    else:
        timeperiod = _scalarize(timeperiod)
        stdev = _scalarize(stdev)
        fenetre_lowest = _scalarize(fenetre_lowest)
        seuil_lowest = _scalarize(seuil_lowest)
        user_exit_sma_length = _scalarize(user_exit_sma_length)
        exit_sma_enabled = _scalarize(exit_sma_enabled)
        mtf_filter_enabled = _scalarize(mtf_filter_enabled)
        mtf_filter_sma_length = _scalarize(mtf_filter_sma_length)

    close = df["Close"]
    external_bindings = external_bindings if isinstance(external_bindings, dict) else {}
    bbt1_module = external_bindings.get("BBT1")

    bbands = vbt.talib("BBANDS").run(
        close,
        timeperiod=timeperiod,
        nbdevup=stdev,
        nbdevdn=stdev,
        matype=0,
        skipna=True,
    )
    upper = _normalize_columns(bbands.upperband)
    lower = _normalize_columns(bbands.lowerband)
    middle = _normalize_columns(bbands.middleband)
    close_aligned = _align_series_to_columns(close, upper)

    signal_t0: Any = None
    # If alias BBT1 is mapped to a Python module that exposes cross_bbw_low_signal,
    # use it as preferred source for T0 to reduce Pine/Python drift.
    if bbt1_module is not None and hasattr(bbt1_module, "cross_bbw_low_signal"):
        try:
            ext_out = bbt1_module.cross_bbw_low_signal(
                Prix=close,
                longueurBB=timeperiod,
                StDev=stdev,
                fenetre_lowest=fenetre_lowest,
                seuil_lowest=seuil_lowest,
            )
            ext_signal = _extract_signal_from_external_result(ext_out, index=close.index)
            signal_t0 = _align_series_to_columns(ext_signal.fillna(False).astype(bool), upper)
        except Exception:
            signal_t0 = None

    if signal_t0 is None:
        cross_bbw_low = CrossBBWLowSignal.run(
            upper_band=upper,
            lower_band=lower,
            middle_band=middle,
            fenetre_lowest=fenetre_lowest,
            seuil_lowest=seuil_lowest,
            per_column=True,
        )
        signal_t0 = _normalize_columns(cross_bbw_low.signal).astype(bool)

    # Pine reference:
    # signal_T1_bull = signal_T0_valide and ta.crossover(close, upper)
    cross_over_bb = (
        upper.lt(close, axis=0) & upper.shift(1).gt(close.shift(1), axis=0)
    ).fillna(False).astype(bool)
    entry_signal = (signal_t0 & cross_over_bb).fillna(False).astype(bool)

    # Optional first MTF filter to emulate `request.security`-style confirmation.
    # It intentionally remains conservative: align a higher timeframe SMA to base index
    # and require close relation above/below that SMA.
    mtf_timeframe = str(mtf_filter_timeframe or "").strip()
    if mtf_timeframe:
        if vector_len > 1:
            mtf_enabled_cols = np.asarray(mtf_filter_enabled, dtype=bool).reshape(-1)
        else:
            mtf_enabled_cols = np.array([bool(mtf_filter_enabled)], dtype=bool)

        if np.any(mtf_enabled_cols):
            mtf_sma_len = int(_safe_scalar(mtf_filter_sma_length, 20))
            mtf_sma = request_security_series(
                base_ohlcv=df[["Open", "High", "Low", "Close"]].copy(),
                timeframe=mtf_timeframe,
                expr_fn=lambda htf_df: vbt.talib("SMA")
                .run(htf_df["Close"], timeperiod=mtf_sma_len, skipna=True)
                .real.ffill(),
                timing=str(mtf_filter_timing or "closing"),
                gaps=str(mtf_filter_gaps or "off"),
                base_freq=None,
            )
            mtf_sma = mtf_sma.reindex(entry_signal.index)
            close_for_filter = close_aligned if hasattr(entry_signal, "columns") else close
            filter_mode = str(mtf_filter_mode or "above").strip().lower()
            if filter_mode == "below":
                mtf_condition = close_for_filter.lt(mtf_sma, axis=0)
            else:
                mtf_condition = close_for_filter.gt(mtf_sma, axis=0)

            if vector_len > 1 and hasattr(entry_signal, "columns"):
                mtf_mask = pd.DataFrame(
                    np.tile(mtf_enabled_cols, (len(entry_signal), 1)),
                    index=entry_signal.index,
                    columns=entry_signal.columns,
                )
                # If MTF is disabled for a column, leave signal unchanged for that column.
                effective_mtf_condition = mtf_condition | (~mtf_mask)
                entry_signal = (entry_signal & effective_mtf_condition).fillna(False).astype(bool)
            else:
                if bool(mtf_enabled_cols[0]):
                    entry_signal = (entry_signal & mtf_condition).fillna(False).astype(bool)

    sma_exit_ind = SMAExit.run(
        close=close_aligned,
        user_exit_sma_length=user_exit_sma_length,
        per_column=True,
    )
    exit_signal = _normalize_columns(sma_exit_ind.signal).fillna(False).astype(bool)

    if vector_len > 1 and _is_array_like(exit_sma_enabled):
        exit_mask = pd.DataFrame(
            np.tile(np.asarray(exit_sma_enabled, dtype=bool), (len(exit_signal), 1)),
            index=exit_signal.index,
            columns=exit_signal.columns,
        )
        exit_signal = exit_signal & exit_mask
    else:
        if not bool(exit_sma_enabled):
            exit_signal[:] = False

    return {
        "entry_signal": entry_signal,
        "exit_signal": exit_signal,
        "upper_band": upper,
        "lower_band": lower,
        "middle_band": middle,
    }


def _safe_scalar(value: Any, default: Any) -> Any:
    if _is_array_like(value):
        arr = np.asarray(value).reshape(-1)
        if len(arr) == 0:
            return default
        value = arr[0]
    if isinstance(value, np.generic):
        return value.item()
    return value if value is not None else default


def run_strategy_test_backtest(
    df: pd.DataFrame,
    params: dict[str, Any],
    timeframe: str = "5s",
    return_portfolio: bool = True,
    external_bindings: dict[str, Any] | None = None,
):
    """Backtest runtime for strategy_test Pine adapter."""
    signals = create_strategy_test_signals(df, external_bindings=external_bindings, **params)
    entries = signals["entry_signal"]
    exits = signals["exit_signal"]

    order_sizing_mode = str(_safe_scalar(params.get("order_sizing_mode", "percent_equity"), "percent_equity"))
    order_fixed_cash = float(_safe_scalar(params.get("order_fixed_cash", 10000.0), 10000.0))
    fees_pct = float(_safe_scalar(params.get("fees_pct", 0.0), 0.0))
    fees = fees_pct / 100.0

    if order_sizing_mode == "fixed_cash":
        size = order_fixed_cash
        size_type = "value"
    else:
        size = 1.0
        size_type = "percent"

    portfolio = vbt.Portfolio.from_signals(
        close=df["Close"],
        entries=entries,
        exits=exits,
        size=size,
        size_type=size_type,
        init_cash=10000,
        fees=fees,
        freq=timeframe,
    )

    if return_portfolio:
        return portfolio

    metric1_name = params.get("metric1_name", "sharpe_ratio")
    metric2_name = params.get("metric2_name", "total_return")
    weight_metric1 = params.get("weight_metric1", 1.0)
    weight_metric2 = params.get("weight_metric2", 0.0)

    def trade_stat(trades, attr_name, default=0.0):
        value = getattr(trades, attr_name, None)
        if value is not None:
            return value
        try:
            stats = trades.stats()
        except Exception:
            return default
        keys = [
            attr_name,
            attr_name.replace("_", " "),
            attr_name.replace("_", " ").title(),
            attr_name.replace("_", " ").capitalize(),
        ]
        for key in keys:
            try:
                value = stats.get(key) if hasattr(stats, "get") else stats[key]
            except Exception:
                value = None
            if value is not None:
                return value
        return default

    def get_metric(port, name):
        if name == "max_drawdown":
            return port.max_drawdown * 100 * -1
        if name == "sharpe_ratio":
            return port.sharpe_ratio
        if name == "total_return":
            return port.total_return * 100
        if name == "avg_gain_per_trade":
            return trade_stat(port.trades, "avg_winning_trade")
        if name == "avg_loss_per_trade":
            return trade_stat(port.trades, "avg_losing_trade") * -1
        if name == "win_rate":
            return port.trades.win_rate
        if name == "avg_pl_per_trade":
            total_ret = port.total_return * 100
            n_trades = port.trades.count()
            if hasattr(n_trades, "replace"):
                safe_trades = n_trades.replace(0, np.nan)
                avg_pl = total_ret / safe_trades
                return avg_pl.replace([np.inf, -np.inf], 0).fillna(0)
            try:
                if float(n_trades) == 0.0:
                    return 0.0
                avg_pl = total_ret / n_trades
                if np.isinf(avg_pl) or np.isnan(avg_pl):
                    return 0.0
                return avg_pl
            except Exception:
                return 0.0
        return 0.0

    m1 = get_metric(portfolio, metric1_name)
    m2 = get_metric(portfolio, metric2_name)
    return (weight_metric1 * m1 + weight_metric2 * m2) / (weight_metric1 + weight_metric2)


@dataclass(frozen=True)
class PineStrategyTestAdapter:
    """Executable adapter for `docs/wfoe_v3/strategy_test.txt`."""

    strategy_mode: str = "pine_imported"
    strategy_id: str = "pine_strategy_test"
    runtime_config: dict[str, Any] | None = None

    def _external_bindings(self) -> dict[str, Any]:
        bindings, _ = _resolve_external_library_bindings(self.runtime_config)
        return bindings

    def get_param_space(self) -> dict[str, Any]:
        # Supported optimization keys for this first runtime block.
        keys = [
            "timeperiod",
            "StDev",
            "fenetre_lowest",
            "seuil_lowest",
            "user_exit_sma_length",
            "mtf_filter_enabled",
            "mtf_filter_sma_length",
            "order_sizing_mode",
            "order_fixed_cash",
            "fees_pct",
        ]
        space: dict[str, Any] = {}
        for key in keys:
            if key in DEFAULT_PARAM_GRID:
                space[key] = list(DEFAULT_PARAM_GRID[key])
        if "mtf_filter_enabled" not in space:
            space["mtf_filter_enabled"] = [False, True]
        if "mtf_filter_sma_length" not in space:
            space["mtf_filter_sma_length"] = [10, 20, 30]
        return space

    def generate_signals(self, df, params: dict[str, Any]) -> Any:
        return create_strategy_test_signals(
            df,
            external_bindings=self._external_bindings(),
            **params,
        )

    def run_backtest(self, df, params: dict[str, Any], timeframe: str = "5s", return_portfolio: bool = True):
        return run_strategy_test_backtest(
            df,
            params,
            timeframe=timeframe,
            return_portfolio=return_portfolio,
            external_bindings=self._external_bindings(),
        )


def _contains_vectorized_params(params: dict[str, Any]) -> bool:
    for _, value in dict(params or {}).items():
        if _is_array_like(value):
            try:
                if len(value) > 1:
                    return True
            except Exception:
                continue
    return False


def _vectorized_param_length(params: dict[str, Any]) -> int:
    lengths: list[int] = []
    for _, value in dict(params or {}).items():
        if _is_array_like(value):
            try:
                n = int(len(value))
            except Exception:
                continue
            if n > 1:
                lengths.append(n)
    if not lengths:
        return 1
    uniq = sorted(set(lengths))
    if len(uniq) > 1:
        raise ValueError(f"Inconsistent vectorized parameter lengths: {uniq}")
    return int(uniq[0])


def _slice_vectorized_params(params: dict[str, Any], idx: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in dict(params or {}).items():
        if _is_array_like(value):
            arr = np.asarray(value).reshape(-1)
            if len(arr) == 0:
                out[key] = value
            elif len(arr) == 1:
                out[key] = arr[0].item() if isinstance(arr[0], np.generic) else arr[0]
            else:
                picked = arr[idx]
                out[key] = picked.item() if isinstance(picked, np.generic) else picked
        else:
            out[key] = value
    return out


def _parse_number(value: str) -> float | None:
    text = str(value or "").strip()
    try:
        return float(text)
    except Exception:
        return None


def _parse_literal(expr: str) -> Any:
    text = str(expr or "").strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    number = _parse_number(text)
    if number is not None:
        if re.fullmatch(r"-?\d+", text):
            return int(number)
        return float(number)
    return None


def _build_assignment_literal_map(spec: dict[str, Any]) -> dict[str, Any]:
    logic = spec.get("logic") if isinstance(spec.get("logic"), dict) else {}
    assignments = logic.get("assignments") if isinstance(logic.get("assignments"), list) else []
    out: dict[str, Any] = {}
    for row in assignments:
        if not isinstance(row, dict):
            continue
        targets = row.get("targets")
        if not isinstance(targets, list) or len(targets) != 1:
            continue
        target = str(targets[0] or "").strip()
        expr = str(row.get("expr") or "").strip()
        if not target or not expr:
            continue
        literal = _parse_literal(expr)
        if literal is not None:
            out[target] = literal
    return out


def _resolve_default_raw(default_raw: Any, literal_map: dict[str, Any]) -> Any:
    raw = str(default_raw or "").strip()
    if not raw:
        return None
    literal = _parse_literal(raw)
    if literal is not None:
        return literal
    visited: set[str] = set()
    token = raw
    while token in literal_map and token not in visited:
        visited.add(token)
        val = literal_map.get(token)
        if isinstance(val, str) and val in literal_map and val not in visited:
            token = val
            continue
        return val
    if raw.lower() in {"close", "open", "high", "low"}:
        return raw.lower()
    return raw


def _default_grid_from_value(name: str, value: Any) -> list[Any]:
    if name in DEFAULT_PARAM_GRID:
        return list(DEFAULT_PARAM_GRID[name])
    if isinstance(value, bool):
        return [False, True]
    if isinstance(value, int) and not isinstance(value, bool):
        if value <= 1:
            return [1, 2, 3]
        low = max(1, int(round(value * 0.7)))
        high = max(low + 1, int(round(value * 1.3)))
        return sorted({low, int(value), high})
    if isinstance(value, float):
        if abs(value) < 1e-9:
            return [0.0, 0.25, 0.5]
        low = round(value * 0.7, 6)
        high = round(value * 1.3, 6)
        return sorted({low, float(value), high})
    return [value]


def _build_param_space_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    inputs = spec.get("inputs") if isinstance(spec.get("inputs"), list) else []
    literal_map = _build_assignment_literal_map(spec)
    space: dict[str, Any] = {}
    for item in inputs:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        input_type = str(item.get("type") or "").strip().lower()
        default_value = _resolve_default_raw(item.get("default_raw"), literal_map)
        if name in DEFAULT_PARAM_GRID:
            space[name] = list(DEFAULT_PARAM_GRID[name])
            continue
        if input_type in {"timeframe", "source", "string"}:
            if default_value is not None:
                space[name] = [default_value]
            continue
        if input_type == "bool":
            space[name] = [False, True]
            continue
        if default_value is not None:
            space[name] = _default_grid_from_value(name, default_value)
    # Ensure execution params are always available.
    if "order_sizing_mode" not in space:
        space["order_sizing_mode"] = ["percent_equity", "fixed_cash"]
    if "order_fixed_cash" not in space:
        space["order_fixed_cash"] = [10000.0]
    if "fees_pct" not in space:
        space["fees_pct"] = [0.0]
    return space


def _series_like(value: Any, index: pd.Index, default: Any = np.nan):
    if isinstance(value, pd.Series):
        return value.reindex(index)
    if isinstance(value, pd.DataFrame):
        if value.shape[1] == 0:
            return pd.Series(default, index=index)
        return value.iloc[:, 0].reindex(index)
    if isinstance(value, np.ndarray):
        arr = value.reshape(-1)
        if len(arr) == len(index):
            return pd.Series(arr, index=index)
    if np.isscalar(value):
        return pd.Series(value, index=index)
    return pd.Series(default, index=index)


def _bool_series(value: Any, index: pd.Index):
    series = _series_like(value, index=index, default=False)
    try:
        return series.fillna(False).astype(bool)
    except Exception:
        return pd.Series(False, index=index)


def _coerce_length(length: Any, default: int = 20) -> int:
    if _is_array_like(length):
        arr = np.asarray(length).reshape(-1)
        if len(arr) == 0:
            return int(default)
        length = arr[0]
    try:
        return max(1, int(float(length)))
    except Exception:
        return int(default)


def shift_n(value: Any, bars: Any):
    n = _coerce_length(bars, default=0)
    if hasattr(value, "shift"):
        return value.shift(n)
    return value


def to_bool(value: Any):
    if isinstance(value, pd.DataFrame):
        return value.fillna(False).astype(bool)
    if isinstance(value, pd.Series):
        return value.fillna(False).astype(bool)
    return bool(value)


def round_series(value: Any, precision: Any = 0):
    p = _coerce_length(precision, default=0)
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value.round(p)
    try:
        return round(float(value), p)
    except Exception:
        return value


def round_to_mintick(value: Any, min_tick: Any = 1e-8):
    tick = float(_safe_scalar(min_tick, 1e-8) or 1e-8)
    tick = max(tick, 1e-12)
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return np.round(np.asarray(value) / tick) * tick
    try:
        return round(float(value) / tick) * tick
    except Exception:
        return value


def where(cond: Any, a: Any, b: Any):
    if isinstance(cond, pd.DataFrame):
        return pd.DataFrame(
            np.where(cond.fillna(False), np.asarray(a), np.asarray(b)),
            index=cond.index,
            columns=cond.columns,
        )
    if isinstance(cond, pd.Series):
        return pd.Series(np.where(cond.fillna(False), np.asarray(a), np.asarray(b)), index=cond.index)
    return a if bool(cond) else b


def ta_sma(source: Any, length: Any):
    n = _coerce_length(length, default=20)
    if isinstance(source, pd.DataFrame):
        return source.rolling(window=n, min_periods=1).mean()
    return _series_like(source, index=source.index if isinstance(source, pd.Series) else pd.Index([])).rolling(
        window=n, min_periods=1
    ).mean()


def ta_ema(source: Any, length: Any):
    n = _coerce_length(length, default=20)
    series = _series_like(source, index=source.index if isinstance(source, pd.Series) else pd.Index([]))
    return series.ewm(span=n, adjust=False, min_periods=1).mean()


def ta_rma(source: Any, length: Any):
    n = _coerce_length(length, default=20)
    alpha = 1.0 / max(1, n)
    series = _series_like(source, index=source.index if isinstance(source, pd.Series) else pd.Index([]))
    return series.ewm(alpha=alpha, adjust=False, min_periods=1).mean()


def ta_wma(source: Any, length: Any):
    n = _coerce_length(length, default=20)
    series = _series_like(source, index=source.index if isinstance(source, pd.Series) else pd.Index([]))
    weights = np.arange(1, n + 1, dtype=float)
    return series.rolling(window=n, min_periods=1).apply(
        lambda x: float(np.dot(x, weights[-len(x) :]) / np.sum(weights[-len(x) :])),
        raw=True,
    )


def ta_vwma(source: Any, length: Any):
    # Volume is optional in runtime context. Fallback to SMA when absent.
    return ta_sma(source, length)


def ta_stdev(source: Any, length: Any):
    n = _coerce_length(length, default=20)
    series = _series_like(source, index=source.index if isinstance(source, pd.Series) else pd.Index([]))
    return series.rolling(window=n, min_periods=1).std().fillna(0.0)


def ta_crossover(a: Any, b: Any):
    sa = _series_like(a, index=a.index if isinstance(a, pd.Series) else b.index if isinstance(b, pd.Series) else pd.Index([]))
    sb = _series_like(b, index=sa.index)
    return (sa.gt(sb) & sa.shift(1).le(sb.shift(1))).fillna(False)


def ta_crossunder(a: Any, b: Any):
    sa = _series_like(a, index=a.index if isinstance(a, pd.Series) else b.index if isinstance(b, pd.Series) else pd.Index([]))
    sb = _series_like(b, index=sa.index)
    return (sa.lt(sb) & sa.shift(1).ge(sb.shift(1))).fillna(False)


def ta_bb(source: Any, length: Any, stdev: Any):
    mid = ta_sma(source, length)
    dev = ta_stdev(source, length) * float(_safe_scalar(stdev, 2.0))
    upper = mid + dev
    lower = mid - dev
    return mid, upper, lower


def ta_linreg(source: Any, length: Any, offset: Any = 0):
    n = _coerce_length(length, default=20)
    series = _series_like(source, index=source.index if isinstance(source, pd.Series) else pd.Index([]))
    x = np.arange(n, dtype=float)

    def _linreg_roll(values: np.ndarray):
        if len(values) == 0:
            return np.nan
        xx = x[-len(values) :]
        try:
            coef = np.polyfit(xx, values, deg=1)
            # value at the latest bar.
            return float(np.polyval(coef, xx[-1]))
        except Exception:
            return float(values[-1])

    out = series.rolling(window=n, min_periods=1).apply(_linreg_roll, raw=True)
    off = _coerce_length(offset, default=0)
    if off:
        out = out.shift(off)
    return out


def ta_barssince(condition: Any):
    cond = _bool_series(condition, index=condition.index if isinstance(condition, pd.Series) else pd.Index([]))
    out = np.full(len(cond), np.nan, dtype=float)
    last_true = -1
    for i, flag in enumerate(cond.to_numpy()):
        if bool(flag):
            out[i] = 0.0
            last_true = i
        elif last_true >= 0:
            out[i] = float(i - last_true)
    return pd.Series(out, index=cond.index)


def _find_top_level(expr: str, char: str) -> int:
    depth = 0
    in_string = False
    quote = ""
    for i, ch in enumerate(expr):
        if in_string:
            if ch == quote and (i == 0 or expr[i - 1] != "\\"):
                in_string = False
            continue
        if ch in ("'", '"'):
            in_string = True
            quote = ch
            continue
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            continue
        if ch == char and depth == 0:
            return i
    return -1


def _extract_call_inner(expr: str, call_name: str) -> str | None:
    """Extract argument payload for `call_name(...)` with balanced scopes."""
    src = str(expr or "").strip()
    pattern = re.compile(rf"^{re.escape(call_name)}\s*\(", flags=re.IGNORECASE)
    match = pattern.match(src)
    if not match:
        return None
    i = match.end() - 1  # opening '('
    depth = 0
    in_string = False
    quote = ""
    for j in range(i, len(src)):
        ch = src[j]
        if in_string:
            if ch == quote and (j == 0 or src[j - 1] != "\\"):
                in_string = False
            continue
        if ch in ("'", '"'):
            in_string = True
            quote = ch
            continue
        if ch in "([{":
            depth += 1
            continue
        if ch in ")]}":
            depth -= 1
            if depth == 0:
                return src[i + 1 : j].strip()
    return None


def _split_top_level_csv(expr: str) -> list[str]:
    """Split CSV arguments at top-level only."""
    src = str(expr or "")
    out: list[str] = []
    depth = 0
    in_string = False
    quote = ""
    start = 0
    for i, ch in enumerate(src):
        if in_string:
            if ch == quote and (i == 0 or src[i - 1] != "\\"):
                in_string = False
            continue
        if ch in ("'", '"'):
            in_string = True
            quote = ch
            continue
        if ch in "([{":
            depth += 1
            continue
        if ch in ")]}":
            depth = max(0, depth - 1)
            continue
        if ch == "," and depth == 0:
            out.append(src[start:i].strip())
            start = i + 1
    tail = src[start:].strip()
    if tail:
        out.append(tail)
    return out


def _strip_enclosing_brackets(expr: str) -> str:
    text = str(expr or "").strip()
    if text.startswith("[") and text.endswith("]"):
        return text[1:-1].strip()
    return text


def _rewrite_ternary(expr: str) -> str:
    text = str(expr or "")
    while True:
        q = _find_top_level(text, "?")
        if q < 0:
            return text
        depth = 0
        in_string = False
        quote = ""
        colon = -1
        for i in range(q + 1, len(text)):
            ch = text[i]
            if in_string:
                if ch == quote and text[i - 1] != "\\":
                    in_string = False
                continue
            if ch in ("'", '"'):
                in_string = True
                quote = ch
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == ":" and depth == 0:
                colon = i
                break
        if colon < 0:
            return text
        cond = text[:q].strip()
        left = text[q + 1 : colon].strip()
        right = text[colon + 1 :].strip()
        text = f"where({cond}, {left}, {right})"


def _split_top_level_keyword(expr: str, keyword: str) -> list[str]:
    """Split expression by keyword (`and` / `or`) at top-level depth."""
    token = f" {keyword} "
    parts: list[str] = []
    depth = 0
    in_string = False
    quote = ""
    start = 0
    i = 0
    while i < len(expr):
        ch = expr[i]
        if in_string:
            if ch == quote and (i == 0 or expr[i - 1] != "\\"):
                in_string = False
            i += 1
            continue
        if ch in ("'", '"'):
            in_string = True
            quote = ch
            i += 1
            continue
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth == 0 and expr[i : i + len(token)].lower() == token:
            parts.append(expr[start:i])
            start = i + len(token)
            i = start
            continue
        i += 1
    parts.append(expr[start:])
    return [p.strip() for p in parts if str(p).strip()]


def _rewrite_boolean_ops(expr: str) -> str:
    text = str(expr or "").strip()
    if not text:
        return text
    or_parts = _split_top_level_keyword(text, "or")
    if len(or_parts) > 1:
        return " | ".join(f"({_rewrite_boolean_ops(part)})" for part in or_parts)
    and_parts = _split_top_level_keyword(text, "and")
    if len(and_parts) > 1:
        return " & ".join(f"({_rewrite_boolean_ops(part)})" for part in and_parts)
    lowered = text.lower()
    if lowered.startswith("not "):
        return f"(~({_rewrite_boolean_ops(text[4:])}))"
    return text


def _pine_expr_to_python(expr: str) -> str:
    text = str(expr or "").strip()
    if not text:
        return "False"
    text = _rewrite_ternary(text)
    text = _rewrite_boolean_ops(text)
    replacements = [
        (r"\btrue\b", "True"),
        (r"\bfalse\b", "False"),
        (r"\bna\b", "np.nan"),
        (r"\bbarmerge\.gaps_off\b", "'off'"),
        (r"\bbarmerge\.gaps_on\b", "'on'"),
        (r"\bbarmerge\.lookahead_off\b", "'lookahead_off'"),
        (r"\bbarmerge\.lookahead_on\b", "'lookahead_on'"),
        (r"\brequest\.security_lower_tf\s*\(", "request_security_lower_tf("),
        (r"\brequest\.security\s*\(", "request_security("),
        (r"\bstrategy\.position_size\b", "in_pos"),
        (r"\bsyminfo\.mintick\b", "min_tick"),
        (r"\bmath\.abs\s*\(", "np.abs("),
        (r"\bmath\.min\s*\(", "np.minimum("),
        (r"\bmath\.max\s*\(", "np.maximum("),
        (r"\bmath\.round_to_mintick\s*\(", "round_to_mintick("),
        (r"\bmath\.round\s*\(", "round_series("),
        (r"\bta\.crossover\s*\(", "ta_crossover("),
        (r"\bta\.crossunder\s*\(", "ta_crossunder("),
        (r"\bta\.barssince\s*\(", "ta_barssince("),
        (r"\bta\.sma\s*\(", "ta_sma("),
        (r"\bta\.ema\s*\(", "ta_ema("),
        (r"\bta\.rma\s*\(", "ta_rma("),
        (r"\bta\.wma\s*\(", "ta_wma("),
        (r"\bta\.vwma\s*\(", "ta_vwma("),
        (r"\bta\.stdev\s*\(", "ta_stdev("),
        (r"\bta\.bb\s*\(", "ta_bb("),
        (r"\bta\.linreg\s*\(", "ta_linreg("),
        (r"\bbool\s*\(", "to_bool("),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # Pine history reference: x[1] -> shift_n(x, 1)
    text = re.sub(
        r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_\.]*)\s*\[\s*(\d+)\s*\]",
        r"shift_n(\1, \2)",
        text,
    )
    return text


def _map_base_series_to_htf(value: Any, base_df: pd.DataFrame, htf_df: pd.DataFrame):
    """Best-effort mapping of base-index OHLCV aliases to HTF OHLCV series."""
    if not isinstance(value, pd.Series):
        return value
    if not value.index.equals(base_df.index):
        return value

    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in base_df.columns or col not in htf_df.columns:
            continue
        base_col = base_df[col]
        try:
            if value.equals(base_col):
                target_col = "Volume" if col == "Volume" else col
                return htf_df[target_col]
        except Exception:
            continue
    return value


def _build_htf_eval_env(base_env: dict[str, Any], base_df: pd.DataFrame, htf_df: pd.DataFrame) -> dict[str, Any]:
    """Build eval env for HTF expression in request.security."""
    env: dict[str, Any] = {}
    for key, value in dict(base_env or {}).items():
        if key in {"open", "high", "low", "close", "volume"}:
            continue
        if isinstance(value, pd.DataFrame):
            continue
        env[key] = _map_base_series_to_htf(value, base_df=base_df, htf_df=htf_df)

    env["open"] = htf_df["Open"]
    env["high"] = htf_df["High"]
    env["low"] = htf_df["Low"]
    env["close"] = htf_df["Close"]
    env["volume"] = htf_df["Volume"] if "Volume" in htf_df.columns else pd.Series(1.0, index=htf_df.index)
    return env


def _coerce_timeframe_value(value: Any) -> str:
    tf = str(_safe_scalar(value, "") or "").strip()
    return tf


def _extract_named_arg(args: list[str], name: str) -> str | None:
    token = f"{name}="
    for arg in args:
        candidate = str(arg or "").strip()
        lowered = candidate.lower().replace(" ", "")
        if lowered.startswith(token):
            return candidate.split("=", 1)[1].strip()
    return None


def _eval_request_security_assignment(
    expr_raw: str,
    env: dict[str, Any],
    base_df: pd.DataFrame,
):
    """
    Evaluate one assignment expression containing `request.security(...)`.
    """
    inner = _extract_call_inner(expr_raw, "request.security")
    if not inner:
        return None
    args = _split_top_level_csv(inner)
    if len(args) < 3:
        return None

    timeframe_raw = args[1]
    expr_part_raw = args[2]
    gaps_raw = _extract_named_arg(args[3:], "gaps")
    lookahead_raw = _extract_named_arg(args[3:], "lookahead")

    tf_expr = _pine_expr_to_python(timeframe_raw)
    timeframe_val = _safe_eval_expr(tf_expr, env, index=base_df.index)
    timeframe = _coerce_timeframe_value(timeframe_val)
    if not timeframe:
        return None

    gaps_mode = "off"
    if isinstance(gaps_raw, str) and gaps_raw.strip():
        gaps_eval = _safe_eval_expr(_pine_expr_to_python(gaps_raw), env, index=base_df.index)
        gaps_mode = str(_safe_scalar(gaps_eval, "off") or "off").strip().lower()
        if "gaps_on" in gaps_mode:
            gaps_mode = "on"
        elif "gaps_off" in gaps_mode:
            gaps_mode = "off"
    timing_mode = "closing"
    if isinstance(lookahead_raw, str) and lookahead_raw.strip():
        lookahead_eval = _safe_eval_expr(_pine_expr_to_python(lookahead_raw), env, index=base_df.index)
        lookahead_mode = str(_safe_scalar(lookahead_eval, "lookahead_off") or "lookahead_off").strip().lower()
        if "lookahead_on" in lookahead_mode:
            timing_mode = "opening"

    def _eval_one_expr(expr_one: str):
        expr_py = _pine_expr_to_python(expr_one)

        def _htf_expr(htf_df: pd.DataFrame):
            htf_env = _build_htf_eval_env(env, base_df=base_df, htf_df=htf_df)
            value = _safe_eval_expr(expr_py, htf_env, index=htf_df.index)
            if isinstance(value, (pd.Series, pd.DataFrame)):
                return value
            return _series_like(value, index=htf_df.index, default=np.nan)

        return request_security_series(
            base_ohlcv=base_df,
            timeframe=timeframe,
            expr_fn=_htf_expr,
            timing=timing_mode,
            gaps=gaps_mode,
            base_freq=None,
        )

    expr_content = _strip_enclosing_brackets(expr_part_raw)
    expr_parts = _split_top_level_csv(expr_content) if expr_part_raw.strip().startswith("[") else [expr_part_raw]
    values = [_eval_one_expr(part) for part in expr_parts]
    return {
        "values": values,
        "timeframe": timeframe,
        "gaps_mode": gaps_mode,
        "timing_mode": timing_mode,
        "expr_part_count": len(expr_parts),
    }


def _build_eval_env(df: pd.DataFrame, params: dict[str, Any], external_bindings: dict[str, Any], in_pos_value: bool):
    close = df["Close"]
    env: dict[str, Any] = {
        "__builtins__": {},
        "np": np,
        "pd": pd,
        "where": where,
        "shift_n": shift_n,
        "to_bool": to_bool,
        "round_series": round_series,
        "round_to_mintick": round_to_mintick,
        "ta_sma": ta_sma,
        "ta_ema": ta_ema,
        "ta_rma": ta_rma,
        "ta_wma": ta_wma,
        "ta_vwma": ta_vwma,
        "ta_stdev": ta_stdev,
        "ta_crossover": ta_crossover,
        "ta_crossunder": ta_crossunder,
        "ta_barssince": ta_barssince,
        "ta_bb": ta_bb,
        "ta_linreg": ta_linreg,
        "request_security": lambda *args, **kwargs: np.nan,
        "request_security_lower_tf": lambda *args, **kwargs: np.nan,
        "open": df["Open"],
        "high": df["High"],
        "low": df["Low"],
        "close": close,
        "volume": (df["Volume"] if "Volume" in df.columns else pd.Series(1.0, index=df.index)),
        "in_pos": pd.Series(bool(in_pos_value), index=df.index),
        "min_tick": 1e-8,
        "strategy": type("StrategyNamespace", (), {"long": "long", "short": "short"})(),
        "syminfo": type("SyminfoNamespace", (), {"tickerid": "SYMBOL", "mintick": 1e-8})(),
        "barmerge": type(
            "BarmergeNamespace",
            (),
            {
                "gaps_on": "on",
                "gaps_off": "off",
                "lookahead_on": "lookahead_on",
                "lookahead_off": "lookahead_off",
            },
        )(),
    }

    for key, value in dict(params or {}).items():
        env[str(key)] = value

    for alias, module in dict(external_bindings or {}).items():
        env[str(alias)] = module
    return env


def _safe_eval_expr(expr: str, env: dict[str, Any], index: pd.Index):
    try:
        return eval(expr, env, env)
    except Exception:
        return pd.Series(np.nan, index=index)


def _eval_assignments(
    assignments: list[dict[str, Any]],
    env: dict[str, Any],
    index: pd.Index,
    base_df: pd.DataFrame,
    warnings_out: list[str] | None = None,
):
    warn = warnings_out if isinstance(warnings_out, list) else []
    for row in assignments:
        if not isinstance(row, dict):
            continue
        targets = row.get("targets")
        expr_raw = row.get("expr")
        if not isinstance(targets, list) or not isinstance(expr_raw, str):
            continue
        expr_lower = expr_raw.lower()
        if "request.security_lower_tf" in expr_lower:
            warn.append("request.security_lower_tf détecté: non supporté en transpilation V3 beta.")
            value = pd.Series(np.nan, index=index)
            if len(targets) == 1:
                env[str(targets[0])] = value
            else:
                for target in targets:
                    env[str(target)] = value
            continue
        if "request.security" in expr_lower:
            rs_eval = _eval_request_security_assignment(
                expr_raw=expr_raw,
                env=env,
                base_df=base_df,
            )
            if isinstance(rs_eval, dict):
                rs_values = rs_eval.get("values") if isinstance(rs_eval.get("values"), list) else []
                if rs_values:
                    if len(targets) == 1:
                        env[str(targets[0])] = rs_values[0]
                    else:
                        for idx_target, target in enumerate(targets):
                            if idx_target < len(rs_values):
                                env[str(target)] = rs_values[idx_target]
                            else:
                                env[str(target)] = pd.Series(np.nan, index=index)
                    continue
            warn.append(f"request.security non évalué (ligne {row.get('line', '?')}).")
            value = pd.Series(np.nan, index=index)
            if len(targets) == 1:
                env[str(targets[0])] = value
            else:
                for target in targets:
                    env[str(target)] = value
            continue

        expr_py = _pine_expr_to_python(expr_raw)
        value = _safe_eval_expr(expr_py, env, index=index)
        if len(targets) == 1:
            env[str(targets[0])] = value
            continue
        if isinstance(value, (tuple, list)) and len(value) >= len(targets):
            for idx, target in enumerate(targets):
                env[str(target)] = value[idx]
            continue
        if isinstance(value, pd.DataFrame) and value.shape[1] >= len(targets):
            for idx, target in enumerate(targets):
                env[str(target)] = value.iloc[:, idx]
            continue
        # Fallback: assign same value to each target.
        for target in targets:
            env[str(target)] = value


def _build_transpiled_signals(
    df: pd.DataFrame,
    params: dict[str, Any],
    strategy_spec: dict[str, Any],
    external_bindings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = strategy_spec if isinstance(strategy_spec, dict) else {}
    logic = spec.get("logic") if isinstance(spec.get("logic"), dict) else {}
    assignments = logic.get("assignments") if isinstance(logic.get("assignments"), list) else []
    order_rules = logic.get("order_rules") if isinstance(logic.get("order_rules"), list) else []
    external_bindings = external_bindings if isinstance(external_bindings, dict) else {}

    entry_env = _build_eval_env(df, params, external_bindings, in_pos_value=False)
    exit_env = _build_eval_env(df, params, external_bindings, in_pos_value=True)
    warnings: list[str] = []
    _eval_assignments(
        assignments,
        entry_env,
        index=df.index,
        base_df=df,
        warnings_out=warnings,
    )
    _eval_assignments(
        assignments,
        exit_env,
        index=df.index,
        base_df=df,
        warnings_out=warnings,
    )

    entry_signal = pd.Series(False, index=df.index)
    exit_signal = pd.Series(False, index=df.index)

    for rule in order_rules:
        if not isinstance(rule, dict):
            continue
        action = str(rule.get("action") or "").strip().lower()
        cond_raw = str(rule.get("condition_expr") or "true").strip()
        cond_py = _pine_expr_to_python(cond_raw)
        if action == "entry":
            direction = str(rule.get("direction") or "").strip().lower()
            if direction and "short" in direction:
                warnings.append("strategy.entry short détecté: ignoré (long-only runtime).")
                continue
            cond_val = _safe_eval_expr(cond_py, entry_env, index=df.index)
            entry_signal = entry_signal | _bool_series(cond_val, index=df.index)
        elif action in {"exit", "close"}:
            cond_val = _safe_eval_expr(cond_py, exit_env, index=df.index)
            exit_signal = exit_signal | _bool_series(cond_val, index=df.index)
        elif action == "cancel":
            # No pending-order model in from_signals.
            continue

    if len(order_rules) == 0:
        warnings.append("Aucune règle strategy.entry/exit détectée dans le spec.")

    return {
        "entry_signal": entry_signal.fillna(False).astype(bool),
        "exit_signal": exit_signal.fillna(False).astype(bool),
        "transpile_warnings": warnings,
        "order_rule_count": len(order_rules),
        "assignment_count": len(assignments),
    }


def build_request_security_diagnostics(
    df: pd.DataFrame,
    params: dict[str, Any],
    strategy_spec: dict[str, Any],
    external_bindings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build deterministic diagnostics for `request.security` calls found in the spec.
    """
    spec = strategy_spec if isinstance(strategy_spec, dict) else {}
    logic = spec.get("logic") if isinstance(spec.get("logic"), dict) else {}
    assignments = logic.get("assignments") if isinstance(logic.get("assignments"), list) else []
    external_bindings = external_bindings if isinstance(external_bindings, dict) else {}

    env = _build_eval_env(df, params, external_bindings, in_pos_value=False)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for row in assignments:
        if not isinstance(row, dict):
            continue
        targets = row.get("targets")
        expr_raw = str(row.get("expr") or "")
        if not isinstance(targets, list) or "request.security" not in expr_raw.lower():
            # Keep env consistent for downstream assignments.
            _eval_assignments([row], env=env, index=df.index, base_df=df, warnings_out=warnings)
            continue

        eval_out = _eval_request_security_assignment(expr_raw=expr_raw, env=env, base_df=df)
        if not isinstance(eval_out, dict):
            warnings.append(f"request.security non analysé (ligne {row.get('line', '?')}).")
            continue
        values = eval_out.get("values") if isinstance(eval_out.get("values"), list) else []
        tf = str(eval_out.get("timeframe") or "")
        for idx_target, target in enumerate(targets):
            series = values[idx_target] if idx_target < len(values) else pd.Series(np.nan, index=df.index)
            if not isinstance(series, pd.Series):
                series = _series_like(series, index=df.index, default=np.nan)
            filled = series.ffill()
            change_count = int(filled.ne(filled.shift(1)).fillna(False).sum())
            non_na_count = int(series.notna().sum())
            first_valid = None
            try:
                first_valid_idx = series.first_valid_index()
                first_valid = str(pd.to_datetime(first_valid_idx, utc=True).isoformat()) if first_valid_idx is not None else None
            except Exception:
                first_valid = None
            rows.append(
                {
                    "line": int(row.get("line", 0) or 0),
                    "target": str(target),
                    "timeframe": tf,
                    "gaps_mode": str(eval_out.get("gaps_mode") or ""),
                    "timing_mode": str(eval_out.get("timing_mode") or ""),
                    "non_na_count": non_na_count,
                    "change_count": change_count,
                    "first_valid_utc": first_valid,
                    "base_bar_count": int(len(df)),
                }
            )

        # Store computed target values into env for subsequent expressions.
        if values:
            if len(targets) == 1:
                env[str(targets[0])] = values[0]
            else:
                for idx_target, target in enumerate(targets):
                    if idx_target < len(values):
                        env[str(target)] = values[idx_target]

    return {
        "schema_version": "pine_request_security_diagnostics.v1",
        "status": "available" if rows else "not_available",
        "request_security_count": len(rows),
        "rows": rows,
        "warnings": warnings,
    }


def _score_portfolio(portfolio, params: dict[str, Any]):
    metric1_name = params.get("metric1_name", "sharpe_ratio")
    metric2_name = params.get("metric2_name", "total_return")
    weight_metric1 = params.get("weight_metric1", 1.0)
    weight_metric2 = params.get("weight_metric2", 0.0)

    def trade_stat(trades, attr_name, default=0.0):
        value = getattr(trades, attr_name, None)
        if value is not None:
            return value
        try:
            stats = trades.stats()
        except Exception:
            return default
        keys = [
            attr_name,
            attr_name.replace("_", " "),
            attr_name.replace("_", " ").title(),
            attr_name.replace("_", " ").capitalize(),
        ]
        for key in keys:
            try:
                value = stats.get(key) if hasattr(stats, "get") else stats[key]
            except Exception:
                value = None
            if value is not None:
                return value
        return default

    def get_metric(port, name):
        if name == "max_drawdown":
            return port.max_drawdown * 100 * -1
        if name == "sharpe_ratio":
            return port.sharpe_ratio
        if name == "total_return":
            return port.total_return * 100
        if name == "avg_gain_per_trade":
            return trade_stat(port.trades, "avg_winning_trade")
        if name == "avg_loss_per_trade":
            return trade_stat(port.trades, "avg_losing_trade") * -1
        if name == "win_rate":
            return port.trades.win_rate
        if name == "avg_pl_per_trade":
            total_ret = port.total_return * 100
            n_trades = port.trades.count()
            if hasattr(n_trades, "replace"):
                safe_trades = n_trades.replace(0, np.nan)
                avg_pl = total_ret / safe_trades
                return avg_pl.replace([np.inf, -np.inf], 0).fillna(0)
            try:
                if float(n_trades) == 0.0:
                    return 0.0
                avg_pl = total_ret / n_trades
                if np.isinf(avg_pl) or np.isnan(avg_pl):
                    return 0.0
                return avg_pl
            except Exception:
                return 0.0
        return 0.0

    m1 = get_metric(portfolio, metric1_name)
    m2 = get_metric(portfolio, metric2_name)
    return (weight_metric1 * m1 + weight_metric2 * m2) / (weight_metric1 + weight_metric2)


def run_transpiled_pine_backtest(
    df: pd.DataFrame,
    params: dict[str, Any],
    strategy_spec: dict[str, Any],
    timeframe: str = "5s",
    return_portfolio: bool = True,
    external_bindings: dict[str, Any] | None = None,
):
    vec_len = _vectorized_param_length(params)
    if vec_len > 1:
        if return_portfolio:
            raise ValueError(
                "Transpiled Pine runtime does not return vectorized portfolio objects. "
                "Use return_portfolio=False in optimization mode."
            )
        scores = []
        for i in range(vec_len):
            scalar_params = _slice_vectorized_params(params, i)
            score_i = run_transpiled_pine_backtest(
                df=df,
                params=scalar_params,
                strategy_spec=strategy_spec,
                timeframe=timeframe,
                return_portfolio=False,
                external_bindings=external_bindings,
            )
            if np.isscalar(score_i):
                scores.append(float(score_i))
            elif hasattr(score_i, "values"):
                arr = np.asarray(score_i.values).reshape(-1)
                scores.append(float(arr[0]) if len(arr) else float("nan"))
            else:
                arr = np.asarray(score_i).reshape(-1)
                scores.append(float(arr[0]) if len(arr) else float("nan"))
        return pd.Series(scores)

    signals = _build_transpiled_signals(
        df=df,
        params=params,
        strategy_spec=strategy_spec,
        external_bindings=external_bindings,
    )
    entries = signals["entry_signal"]
    exits = signals["exit_signal"]

    order_sizing_mode = str(_safe_scalar(params.get("order_sizing_mode", "percent_equity"), "percent_equity"))
    order_fixed_cash = float(_safe_scalar(params.get("order_fixed_cash", 10000.0), 10000.0))
    fees_pct = float(_safe_scalar(params.get("fees_pct", 0.0), 0.0))
    fees = fees_pct / 100.0

    if order_sizing_mode == "fixed_cash":
        size = order_fixed_cash
        size_type = "value"
    else:
        size = 1.0
        size_type = "percent"

    portfolio = vbt.Portfolio.from_signals(
        close=df["Close"],
        entries=entries,
        exits=exits,
        size=size,
        size_type=size_type,
        init_cash=10000,
        fees=fees,
        freq=timeframe,
    )
    if return_portfolio:
        return portfolio
    return _score_portfolio(portfolio, params)


@dataclass(frozen=True)
class GeneratedPineRuntimeAdapter:
    """
    Generic runtime adapter for generated Pine strategies (non strategy_test).

    It executes deterministic logic artifacts (`logic.assignments`, `logic.order_rules`)
    extracted into `strategy_spec.v1`.
    """

    strategy_mode: str = "pine_imported"
    strategy_id: str = "pine_imported_generated"
    runtime_config: dict[str, Any] | None = None
    strategy_spec: dict[str, Any] | None = None

    def _spec(self) -> dict[str, Any]:
        if isinstance(self.strategy_spec, dict) and self.strategy_spec:
            return self.strategy_spec
        if isinstance(self.runtime_config, dict):
            spec = self.runtime_config.get("pine_strategy_spec")
            if isinstance(spec, dict) and spec:
                return spec
        return {}

    def _external_bindings(self) -> dict[str, Any]:
        bindings, _ = _resolve_external_library_bindings(self.runtime_config)
        return bindings

    def get_param_space(self) -> dict[str, Any]:
        spec = self._spec()
        space = _build_param_space_from_spec(spec)
        return space

    def generate_signals(self, df, params: dict[str, Any]) -> Any:
        spec = self._spec()
        return _build_transpiled_signals(
            df=df,
            params=params,
            strategy_spec=spec,
            external_bindings=self._external_bindings(),
        )

    def run_backtest(self, df, params: dict[str, Any], timeframe: str = "5s", return_portfolio: bool = True):
        spec = self._spec()
        return run_transpiled_pine_backtest(
            df=df,
            params=params,
            strategy_spec=spec,
            timeframe=timeframe,
            return_portfolio=return_portfolio,
            external_bindings=self._external_bindings(),
        )
