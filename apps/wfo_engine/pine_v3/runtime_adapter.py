"""Runtime adapter for first executable Pine-imported strategy (V3 block 1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import importlib
import importlib.util
import os
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
