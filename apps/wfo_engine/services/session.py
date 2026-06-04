"""Typed facade over st.session_state for WFO Engine core state.

Usage (read-only snapshot — does not write back automatically):
    state = WFOSessionState.read()
    if state.wfo_running:
        ...

To write, set st.session_state keys directly as before; this class is a
read-only view designed to eliminate scattered, untyped .get() calls.
It is intentionally importable without Streamlit installed (mock the
``_get`` callable in tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class WFOSessionState:
    """Immutable snapshot of WFO-related Streamlit session state keys."""

    # Run lifecycle
    wfo_running: bool
    wfo_results: dict | None
    wfo_job_state: dict | None
    wfo_control: Any | None
    wfo_thread: Any | None
    wfo_job_config: dict | None
    wfo_error_log_path: str | None

    # Data
    df: Any | None

    # Final backtest state
    final_portfolio: Any | None
    final_params: dict | None
    final_params_source: str | None
    final_timeframe: str | None

    @staticmethod
    def read(_get: Callable | None = None) -> "WFOSessionState":
        """Build a snapshot from st.session_state (or a provided getter for tests).

        Parameters
        ----------
        _get : callable, optional
            ``dict.get``-compatible callable.  Defaults to
            ``st.session_state.get``.  Pass a plain dict's ``.get`` in tests
            to avoid importing Streamlit.
        """
        if _get is None:
            import streamlit as st  # lazy import so module is testable without st
            _get = st.session_state.get

        return WFOSessionState(
            wfo_running=bool(_get("wfo_running", False)),
            wfo_results=_get("wfo_results"),
            wfo_job_state=_get("wfo_job_state"),
            wfo_control=_get("wfo_control"),
            wfo_thread=_get("wfo_thread"),
            wfo_job_config=_get("wfo_job_config"),
            wfo_error_log_path=_get("wfo_error_log_path"),
            df=_get("df"),
            final_portfolio=_get("final_portfolio"),
            final_params=_get("final_params"),
            final_params_source=_get("final_params_source"),
            final_timeframe=_get("final_timeframe"),
        )
