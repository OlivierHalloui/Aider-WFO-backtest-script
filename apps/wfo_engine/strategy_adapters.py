"""Strategy adapter interfaces and native implementations for WFO engines."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from typing import Any, Protocol

from config import DEFAULT_PARAM_GRID, DEFAULT_STRATEGY_ID, DEFAULT_STRATEGY_MODE
from pine_v3.runtime_adapter import PineStrategyTestAdapter, supports_strategy_test_runtime
from strategy import create_signal_generators, run_backtest


class StrategyAdapter(Protocol):
    """Contract expected by optimization engines (`wfo` and `adaptive`)."""

    strategy_mode: str
    strategy_id: str

    def get_param_space(self) -> dict[str, Any]:
        """Return adapter-native parameter space metadata."""

    def generate_signals(self, df, params: dict[str, Any]) -> Any:
        """Generate entry/exit signals (optional, for diagnostics and future use)."""

    def run_backtest(self, df, params: dict[str, Any], timeframe: str = "5s", return_portfolio: bool = True):
        """Execute backtest and return portfolio or scalar score."""


@dataclass(frozen=True)
class ATDMFAdapter:
    """Native adapter that preserves current ATDMF execution behavior."""

    strategy_mode: str = DEFAULT_STRATEGY_MODE
    strategy_id: str = DEFAULT_STRATEGY_ID

    def get_param_space(self) -> dict[str, Any]:
        return {key: list(values) for key, values in DEFAULT_PARAM_GRID.items()}

    def generate_signals(self, df, params: dict[str, Any]) -> Any:
        return create_signal_generators(df, **params)

    def run_backtest(self, df, params: dict[str, Any], timeframe: str = "5s", return_portfolio: bool = True):
        return run_backtest(df, params, timeframe=timeframe, return_portfolio=return_portfolio)


_NATIVE_ADAPTER = ATDMFAdapter()


def _load_generated_adapter_from_file(module_path: str, config: dict[str, Any], strategy_id: str):
    """Best-effort loading of generated Pine adapter module from file path."""
    path = str(module_path or "").strip()
    if not path or not os.path.exists(path):
        return None
    module_name = f"wfoe_generated_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    create_adapter = getattr(module, "create_adapter", None)
    if callable(create_adapter):
        adapter = create_adapter(runtime_config=config)
        if adapter is not None:
            return adapter

    generated_cls = getattr(module, "GeneratedPineAdapter", None)
    if generated_cls is not None:
        try:
            return generated_cls(strategy_id=strategy_id, runtime_config=config)
        except TypeError:
            try:
                return generated_cls(runtime_config=config)
            except Exception:
                return None
    return None


def resolve_strategy_adapter(
    strategy_mode: str | None = None,
    strategy_id: str | None = None,
    config: dict[str, Any] | None = None,
) -> StrategyAdapter:
    """
    Resolve a strategy adapter from explicit args or config.

    Current scope:
    - supports `native_atdmf`
    - supports `pine_imported` for strategy_test runtime (V3 block 1)
    - raises `NotImplementedError` for other Pine strategies
    """
    mode = strategy_mode
    sid = strategy_id
    if isinstance(config, dict):
        mode = mode or config.get("strategy_mode")
        sid = sid or config.get("strategy_id")
    mode = str(mode or DEFAULT_STRATEGY_MODE).lower()
    sid = str(sid or DEFAULT_STRATEGY_ID)

    if mode == DEFAULT_STRATEGY_MODE:
        # Keep singleton for stable identity and tiny overhead.
        return _NATIVE_ADAPTER

    if mode == "pine_imported":
        cfg = config if isinstance(config, dict) else {}
        compat_mode = str(cfg.get("pine_compat_mode", "strict")).lower()
        if compat_mode == "strict":
            is_blocking = bool(cfg.get("pine_compatibility_blocking", False))
            if is_blocking:
                raise NotImplementedError(
                    "Pine strict mode blocks execution: unresolved/incompatible features remain."
                )
        generated_module_path = str(cfg.get("pine_generated_module_path") or "").strip()
        if generated_module_path:
            adapter = _load_generated_adapter_from_file(
                module_path=generated_module_path,
                config=cfg,
                strategy_id=sid,
            )
            if adapter is not None:
                return adapter
        if supports_strategy_test_runtime(strategy_id=sid, config=config):
            return PineStrategyTestAdapter(
                strategy_id=sid or "pine_strategy_test",
                runtime_config=cfg,
            )
        raise NotImplementedError(
            "Strategy mode 'pine_imported' is currently executable only for the "
            "test strategy runtime (strategy_test.txt / aliases: pine_strategy_test, "
            "pine_atdmf_strategy_bb_sma_only_long)."
        )

    raise NotImplementedError(
        f"Strategy mode '{mode}' (strategy_id='{sid}') is not supported yet."
    )
