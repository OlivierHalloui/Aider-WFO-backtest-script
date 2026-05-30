"""Run-layer orchestration services for optimization jobs."""

from __future__ import annotations

import logging
import time
from typing import Any

from adaptive_optimization import adaptive_continuous_optimization
from config import DEFAULT_STRATEGY_ID, DEFAULT_STRATEGY_MODE
from data_loading import load_data
from main import get_metrics_info, get_param_grid, get_wfo_settings
from services.error_log import write_error_log, collect_classic_wfo_entries
from strategy_adapters import resolve_strategy_adapter
from wfo import OptimizationInterrupted, walk_forward_optimization

logger = logging.getLogger(__name__)


def run_optimization_job(
    config: dict,
    control: Any = None,
    job_state: dict | None = None,
    df=None,
    run_ts: str | None = None,
    log_dir: str | None = None,
):
    """Run optimization without direct Streamlit calls (thread-safe job state updates).

    Parameters
    ----------
    config : dict
        Full WFO configuration dictionary.
    control : Any, optional
        Stop-signal object (must expose ``should_stop()`` and ``wait_if_paused()``).
    job_state : dict, optional
        Shared mutable dict updated in-place with progress, window, evaluations, etc.
    df : pandas.DataFrame, optional
        Pre-loaded OHLCV DataFrame.  When provided the data-loading step is skipped,
        which eliminates the full-file CSV read when multiple campaign configs share
        the same dataset.
    """
    try:
        strategy_mode = str(config.get("strategy_mode", DEFAULT_STRATEGY_MODE)).lower()
        strategy_id = str(config.get("strategy_id", DEFAULT_STRATEGY_ID))
        try:
            strategy_adapter = resolve_strategy_adapter(
                strategy_mode=strategy_mode,
                strategy_id=strategy_id,
                config=config,
            )
        except NotImplementedError as e:
            message = str(e)
            if job_state is not None:
                job_state["error"] = message
            return None, None, None

        if df is None:
            if job_state is not None:
                job_state["message"] = "Loading data..."

            _t0 = time.time()
            if config["from_file"]:
                df = load_data(
                    config["start_date"],
                    config["end_date"],
                    config["timeframe"],
                    from_file=True,
                    file_path=config["file_path"],
                )
            else:
                df = load_data(
                    config["start_date"],
                    config["end_date"],
                    config["timeframe"],
                    from_file=False,
                )
            _load_s = time.time() - _t0
            _lm, _ls = divmod(int(_load_s), 60)
            logger.info(
                "Data loaded in %dm %02ds: %d bars | %s → %s | tf=%s | source=%s",
                _lm, _ls,
                len(df) if df is not None else 0,
                df.index[0].strftime("%Y-%m-%d %H:%M") if df is not None and len(df) else "?",
                df.index[-1].strftime("%Y-%m-%d %H:%M") if df is not None and len(df) else "?",
                config.get("timeframe", "?"),
                "file" if config.get("from_file") else "api",
            )
        else:
            logger.info(
                "Data reused from cache: %d bars | %s → %s | tf=%s",
                len(df),
                df.index[0].strftime("%Y-%m-%d %H:%M") if len(df) else "?",
                df.index[-1].strftime("%Y-%m-%d %H:%M") if len(df) else "?",
                config.get("timeframe", "?"),
            )

        params_grid = get_param_grid(config)
        metrics_info = get_metrics_info(config)
        wfo_settings = get_wfo_settings(config)
        regime = str(getattr(wfo_settings, "optimization_regime", "classic")).lower()

        if job_state is not None:
            if regime == "adaptive_continuous":
                job_state["message"] = "Starting adaptive continuous optimization..."
            else:
                job_state["message"] = (
                    f"Starting {config['optimization_method'].upper()} on {config['n_windows']} windows..."
                )

        start_time = time.time()

        def status_callback(msg):
            if job_state is None:
                return
            if isinstance(msg, str):
                job_state["message"] = msg
            elif isinstance(msg, dict) and msg.get("type") == "stats":
                progress = msg.get("progress")
                if progress is not None:
                    try:
                        job_state["progress"] = min(max(float(progress), 0.0), 1.0)
                    except Exception:
                        pass
                else:
                    w = msg.get("window", 0)
                    job_state["progress"] = min(w / max(1, config["n_windows"]), 1.0)

                if msg.get("window") is not None:
                    job_state["window"] = msg.get("window")
                if msg.get("evaluations") is not None:
                    try:
                        job_state["evaluations"] = int(msg.get("evaluations"))
                    except Exception:
                        pass
                if msg.get("speed") is not None:
                    try:
                        job_state["speed"] = float(msg.get("speed"))
                    except Exception:
                        pass
                if msg.get("eta") is not None:
                    try:
                        job_state["eta_seconds"] = float(msg.get("eta"))
                    except Exception:
                        pass

                if msg.get("message"):
                    job_state["message"] = msg.get("message")
                else:
                    w = msg.get("window", 0)
                    job_state["message"] = f"Window {min(w, config['n_windows'])}/{config['n_windows']}"

        if regime == "adaptive_continuous":
            results = adaptive_continuous_optimization(
                df,
                param_grid=params_grid,
                metrics_info=metrics_info,
                timeframe=config["timeframe"],
                settings=wfo_settings,
                status_callback=status_callback,
                control=control,
                strategy_adapter=strategy_adapter,
            )
        else:
            results = walk_forward_optimization(
                df,
                param_grid=params_grid,
                metrics_info=metrics_info,
                timeframe=config["timeframe"],
                settings=wfo_settings,
                status_callback=status_callback,
                control=control,
                strategy_adapter=strategy_adapter,
            )

        elapsed = time.time() - start_time
        _m, _s = divmod(int(elapsed), 60)
        logger.info("Run complete in %dm %02ds", _m, _s)
        if job_state is not None:
            job_state["progress"] = 1.0
            job_state["message"] = "Optimization complete."

        # Write error log (even on success — captures 0-trade window warnings)
        _entries = collect_classic_wfo_entries(results)
        _log_path = write_error_log(
            run_type="classic",
            config=config,
            entries=_entries,
            output_dir=log_dir or "reports/error_logs",
            run_ts=run_ts,
            status="OK",
        )
        if job_state is not None:
            job_state["error_log_path"] = _log_path

        return results, df, elapsed

    except OptimizationInterrupted:
        if job_state is not None:
            job_state["message"] = "Stop requested. Optimization interrupted."
        logger.info("Run interrupted by stop request.")
        write_error_log(
            run_type="classic",
            config=config,
            entries=[],
            output_dir=log_dir or "reports/error_logs",
            run_ts=run_ts,
            status="INTERRUPTED",
        )
        return None, None, None
    except Exception as e:
        if job_state is not None:
            job_state["error"] = str(e)
        logger.error("Run failed: %s", e)
        _log_path = write_error_log(
            run_type="classic",
            config=config,
            entries=[{"level": "ERROR", "context": "run", "message": str(e)}],
            output_dir=log_dir or "reports/error_logs",
            run_ts=run_ts,
            status="FAILED",
            fatal_error=str(e),
        )
        if job_state is not None:
            job_state["error_log_path"] = _log_path
        return None, None, None
