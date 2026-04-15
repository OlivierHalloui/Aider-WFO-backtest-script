"""
ui/stagewise_panel.py — Streamlit panel for launching a stagewise WFO campaign.

Runs run_stagewise_campaign() in a background thread (same pattern as
campaign_panel.py) and renders live progress via @st.fragment.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from stagewise_optimizer import (
    STAGE_PLAN,
    run_stagewise_campaign,
    _load_csv,
)
from ui.final_backtest_panel import load_stagewise_params_from_json

# ── constants ─────────────────────────────────────────────────────────────────
_DEFAULT_OUTPUT = "reports/stagewise/"
_POLL_SECONDS   = 2        # fragment refresh rate


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND WORKER
# ═══════════════════════════════════════════════════════════════════════════════
def _stagewise_worker(
    df: pd.DataFrame,
    final_df: pd.DataFrame | None,
    cfg: dict,
    run_state: dict,
    stop_event: threading.Event,
):
    """Runs in a daemon thread. Updates run_state in-place (no session_state writes)."""

    def _stage_cb(stage_num, stage_name, status, report):
        run_state["completed"].append({
            "stage":  stage_num,
            "name":   stage_name,
            "status": status,
            "is_sharpe":   report.get("is_avg_sharpe"),
            "oos_sharpe":  report.get("oos_avg_sharpe"),
            "is_return":   report.get("is_avg_return"),
            "oos_return":  report.get("oos_avg_return"),
            "elapsed":     report.get("elapsed_seconds"),
            "params":      report.get("consensus_best_params", {}),
        })
        run_state["current_stage"] = stage_num + 1

    # Build stage plan from selected stage indices
    selected = cfg.get("selected_stages", list(range(1, len(STAGE_PLAN) + 1)))
    plan = [s for i, s in enumerate(STAGE_PLAN, 1) if i in selected]

    run_state["n_stages"] = len(plan)
    run_state["current_stage"] = 1
    run_state["status"] = "running"

    try:
        final_report = run_stagewise_campaign(
            df=df,
            timeframe=cfg["timeframe"],
            direction=cfg["direction"],
            stage_plan=plan,
            n_windows=cfg["n_windows"],
            train_size=cfg["train_size"],
            output_dir=Path(cfg["output_dir"]),
            final_df=final_df,
            stage_callback=_stage_cb,
        )
        run_state["final_report"] = final_report
        run_state["status"] = "done"
    except Exception as exc:
        run_state["status"] = "error"
        run_state["error"] = str(exc)


# ═══════════════════════════════════════════════════════════════════════════════
# PROGRESS FRAGMENT
# ═══════════════════════════════════════════════════════════════════════════════
@st.fragment(run_every=_POLL_SECONDS)
def _render_stagewise_progress():
    run_state = st.session_state.get("_sw_run_state")
    if run_state is None:
        return

    status        = run_state.get("status", "idle")
    n_stages      = run_state.get("n_stages", len(STAGE_PLAN))
    current_stage = run_state.get("current_stage", 1)
    completed     = run_state.get("completed", [])
    selected      = st.session_state.get("_sw_cfg", {}).get(
        "selected_stages", list(range(1, n_stages + 1))
    )

    # ── global progress bar ──────────────────────────────────────────────────
    progress_pct = len(completed) / max(n_stages, 1)
    status_label = {
        "running": f"⏳ Stage {current_stage}/{n_stages} en cours…",
        "done":    "✅ Campagne terminée",
        "error":   f"❌ Erreur : {run_state.get('error', '')}",
        "idle":    "En attente…",
    }.get(status, status)

    st.progress(progress_pct, text=status_label)

    # ── per-stage cards ──────────────────────────────────────────────────────
    plan_names = [s["name"] for s in STAGE_PLAN]
    cols = st.columns(min(n_stages, 4))

    for i, stage_num in enumerate(selected):
        col = cols[i % len(cols)]
        plan_idx = stage_num - 1
        name = plan_names[plan_idx] if plan_idx < len(plan_names) else f"Stage {stage_num}"
        # Find completed report for this stage
        done = next((c for c in completed if c["stage"] == stage_num), None)

        with col:
            if done:
                st_icon = "✅" if done["status"] == "OK" else "❌"
                is_s  = done.get("is_sharpe")
                oos_s = done.get("oos_sharpe")
                is_r  = done.get("is_return")
                oos_r = done.get("oos_return")
                elapsed = done.get("elapsed", 0)
                st.markdown(
                    f"**{st_icon} Run {stage_num}**  \n"
                    f"{name.split('—')[-1].strip()}  \n"
                    f"IS Sharpe `{is_s:.2f}`  OOS `{oos_s:.2f}`  \n"
                    f"IS Ret `{(is_r or 0)*100:+.1f}%`  OOS `{(oos_r or 0)*100:+.1f}%`  \n"
                    f"⏱ {elapsed:.0f}s"
                )
            elif stage_num == current_stage and status == "running":
                st.markdown(f"**⏳ Run {stage_num}**  \n{name.split('—')[-1].strip()}  \n*en cours…*")
            else:
                st.markdown(f"**⬜ Run {stage_num}**  \n{name.split('—')[-1].strip()}")

    # ── auto-load params when done ───────────────────────────────────────────
    if status == "done":
        final_report = run_state.get("final_report", {})
        if final_report and not st.session_state.get("_sw_params_loaded"):
            load_stagewise_params_from_json(final_report)
            st.session_state["_sw_params_loaded"] = True

        # Show final backtest metrics if available
        fb = final_report.get("final_backtest", {})
        if fb.get("status") == "OK":
            m = fb.get("metrics", {})
            st.markdown("---")
            st.markdown("##### Backtest final")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sharpe",    f"{m.get('sharpe', 0):.3f}")
            c2.metric("Return",    f"{m.get('return', 0):+.2f}%")
            c3.metric("Max DD",    f"{m.get('max_drawdown', 0):.2f}%")
            c4.metric("Trades",    str(m.get("n_trades", 0)))

        st.success("Params chargés dans les inputs. Tu peux lancer le Final Backtest.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ═══════════════════════════════════════════════════════════════════════════════
def render_stagewise_panel(*, get_current_config, load_data):
    """Entry point — call from app.py."""

    st.markdown("### 🎯 Campagne WFO Stagewise")

    status = st.session_state.get("_sw_run_state", {}).get("status", "idle")
    is_running = (status == "running")

    # ── configuration ────────────────────────────────────────────────────────
    with st.expander("⚙️ Configuration de la campagne", expanded=(status == "idle")):
        config = get_current_config()

        col_a, col_b = st.columns(2)
        with col_a:
            sw_direction = st.selectbox(
                "Direction", ["long_only", "short_only", "both"],
                index=["long_only", "short_only", "both"].index(
                    config.get("strategy_direction", "long_only")
                ),
                key="_sw_direction",
                disabled=is_running,
            )
            sw_timeframe = st.text_input(
                "Timeframe", value=config.get("timeframe", "5s"),
                key="_sw_timeframe", disabled=is_running,
            )
            sw_n_windows = st.number_input(
                "Nb fenêtres WFO", min_value=2, max_value=20,
                value=6, key="_sw_n_windows", disabled=is_running,
            )
            sw_train_size = st.slider(
                "Train size (IS fraction)", 0.5, 0.9, 0.75, 0.05,
                key="_sw_train_size", disabled=is_running,
            )

        with col_b:
            sw_start = st.text_input(
                "Date début (YYYY-MM-DD)",
                value=str(config.get("start_date", "2025-04-01"))[:10],
                key="_sw_start", disabled=is_running,
            )
            sw_end = st.text_input(
                "Date fin (YYYY-MM-DD)",
                value=str(config.get("end_date", "2025-08-15"))[:10],
                key="_sw_end", disabled=is_running,
            )
            sw_output = st.text_input(
                "Répertoire de sortie", value=_DEFAULT_OUTPUT,
                key="_sw_output", disabled=is_running,
            )
            sw_data_file = st.text_input(
                "Fichier données (CSV)",
                value=config.get("data_file_path", ""),
                key="_sw_data_file", disabled=is_running,
                help="Chemin vers le CSV OHLCV 5s",
            )

        # Stage selection
        st.markdown("**Stages à exécuter**")
        stage_cols = st.columns(4)
        selected_stages = []
        for i, stage in enumerate(STAGE_PLAN):
            col = stage_cols[i % 4]
            label = f"Run {i+1}: {stage['name'].split('—')[-1].strip()}"
            checked = col.checkbox(label, value=True, key=f"_sw_stage_{i+1}",
                                   disabled=is_running)
            if checked:
                selected_stages.append(i + 1)

        # Final backtest option
        st.markdown("**Backtest final**")
        col_fb1, col_fb2, col_fb3 = st.columns([1, 2, 2])
        sw_run_fb = col_fb1.checkbox(
            "Activer", value=True, key="_sw_run_fb", disabled=is_running,
        )
        sw_fb_start = col_fb2.text_input(
            "Début backtest final", value=sw_start,
            key="_sw_fb_start", disabled=is_running or not sw_run_fb,
        )
        sw_fb_end = col_fb3.text_input(
            "Fin backtest final", value=sw_end,
            key="_sw_fb_end", disabled=is_running or not sw_run_fb,
        )

    # ── start / stop ─────────────────────────────────────────────────────────
    col_start, col_stop = st.columns(2)

    with col_start:
        start_disabled = is_running or not selected_stages or not sw_data_file
        if st.button(
            "▶ Démarrer la campagne stagewise",
            disabled=start_disabled,
            type="primary",
            use_container_width=True,
            key="_sw_btn_start",
        ):
            # Load data
            try:
                df = _load_csv(sw_data_file, sw_start, sw_end)
            except Exception as exc:
                st.error(f"Erreur chargement données : {exc}")
                st.stop()

            final_df = None
            if sw_run_fb:
                try:
                    final_df = _load_csv(sw_data_file, sw_fb_start, sw_fb_end)
                except Exception as exc:
                    st.warning(f"Backtest final désactivé (erreur chargement) : {exc}")

            # Build campaign config
            cfg = {
                "timeframe":       sw_timeframe,
                "direction":       sw_direction,
                "n_windows":       int(sw_n_windows),
                "train_size":      float(sw_train_size),
                "output_dir":      sw_output,
                "selected_stages": selected_stages,
            }
            st.session_state["_sw_cfg"] = cfg

            # Shared mutable state (thread writes in-place)
            run_state: dict = {
                "status":        "running",
                "n_stages":      len(selected_stages),
                "current_stage": 1,
                "completed":     [],
                "final_report":  None,
                "error":         None,
            }
            st.session_state["_sw_run_state"] = run_state
            st.session_state["_sw_params_loaded"] = False
            st.session_state["_sw_start_time"] = time.time()

            stop_event = threading.Event()
            st.session_state["_sw_stop_event"] = stop_event

            t = threading.Thread(
                target=_stagewise_worker,
                args=(df, final_df, cfg, run_state, stop_event),
                daemon=True,
            )
            add_script_run_ctx(t, get_script_run_ctx())
            st.session_state["_sw_thread"] = t
            t.start()
            st.rerun()

    with col_stop:
        if st.button(
            "⏹ Arrêter",
            disabled=not is_running,
            use_container_width=True,
            key="_sw_btn_stop",
        ):
            ev = st.session_state.get("_sw_stop_event")
            if ev:
                ev.set()
            rs = st.session_state.get("_sw_run_state")
            if rs:
                rs["status"] = "error"
                rs["error"] = "Arrêté par l'utilisateur"
            st.rerun()

    # ── live progress ─────────────────────────────────────────────────────────
    if status in ("running", "done", "error"):
        st.markdown("---")
        elapsed_total = time.time() - st.session_state.get("_sw_start_time", time.time())
        st.caption(f"Temps écoulé : {elapsed_total/60:.1f} min")
        _render_stagewise_progress()
