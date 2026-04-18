"""
ui/stagewise_panel.py — Streamlit panel for launching a stagewise WFO campaign.

Runs run_stagewise_campaign() in a background thread (same pattern as
campaign_panel.py) and renders live progress via @st.fragment.
"""

from __future__ import annotations

import io
import json
import threading
import time
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from stagewise_optimizer import (
    STAGE_PLAN,
    run_stagewise_campaign,
    build_stagewise_wfo_results,
    _load_csv,
)
from ui.final_backtest_panel import load_stagewise_params_from_json

# ── constants ─────────────────────────────────────────────────────────────────
_DEFAULT_OUTPUT = "reports/stagewise"
_POLL_SECONDS   = 2        # fragment refresh rate


@st.cache_data(show_spinner=False)
def _get_csv_date_range(path: str, mtime: float) -> tuple[str, str]:
    """Return (first_date, last_date) as YYYY-MM-DD strings. Cached per file+mtime."""
    try:
        df = pd.read_csv(path, index_col="Open time", parse_dates=True,
                         usecols=["Open time"])
        df.index = pd.to_datetime(df.index, utc=True)
        return str(df.index.min())[:10], str(df.index.max())[:10]
    except Exception:
        return "", ""


# ── ZIP auto-save helper ───────────────────────────────────────────────────────
def _auto_save_stagewise_zip(
    synthetic_wfo: dict,
    final_report: dict,
    cfg: dict,
    ts: str,
) -> str | None:
    """
    Build a wfo_results-compatible ZIP and save it to reports/.

    The ZIP contains results.json (loadable by the standard import flow) and
    stagewise_final_report.json (full per-stage data for audit).

    Returns the saved path, or None on error.
    """
    try:
        payload = {
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source":      "stagewise",
            "config": {
                "timeframe":         cfg.get("timeframe"),
                "strategy_direction": cfg.get("direction"),
                "order_sizing_mode": cfg.get("order_sizing_mode"),
                "order_fixed_cash":  cfg.get("order_fixed_cash"),
                "fees_pct":          cfg.get("fees_pct"),
                "n_windows":         cfg.get("n_windows"),
                "train_size":        cfg.get("train_size"),
                "data_file_path":    cfg.get("data_file"),
            },
            "wfo_results": synthetic_wfo,
            "has_final_portfolio": False,
        }
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("results.json",
                        json.dumps(payload, indent=2, default=str))
            zf.writestr("stagewise_final_report.json",
                        json.dumps(final_report, indent=2, default=str))

        zip_path = Path("reports") / f"wfo_results_stagewise_{ts}.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        zip_path.write_bytes(buf.getvalue())

        # Expose bytes so the sidebar download button picks them up immediately
        st.session_state["results_zip_bytes"] = buf.getvalue()
        st.session_state["results_zip_path"]  = str(zip_path)
        return str(zip_path)
    except Exception as exc:
        st.warning(f"Auto-save ZIP échoué : {exc}")
        return None


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

    # Sizing + pqs_n_ref are not in PARAM_DEFAULTS/FULL_PARAM_REGISTRY — pass as
    # initial_fixed_params so they propagate into every backtest call via
    # _FixedParamAdapter (merged ahead of any stage-specific param search).
    sizing_fixed = {
        "order_sizing_mode": cfg.get("order_sizing_mode", "percent_equity"),
        "order_fixed_cash":  float(cfg.get("order_fixed_cash", 10000.0)),
        "fees_pct":          float(cfg.get("fees_pct", 0.0)),
        "pqs_n_ref":         int(cfg.get("pqs_n_ref", 50)),
    }

    try:
        final_report = run_stagewise_campaign(
            df=df,
            timeframe=cfg["timeframe"],
            direction=cfg["direction"],
            stage_plan=plan,
            n_windows=cfg["n_windows"],
            train_size=cfg["train_size"],
            anchored=cfg.get("anchored", False),
            output_dir=Path(cfg["output_dir"]),
            final_df=final_df,
            stage_callback=_stage_cb,
            data_file_path=cfg.get("data_file", ""),
            initial_fixed_params=sizing_fixed,
            optimization_metric=cfg.get("optimization_metric", "sharpe_ratio"),
            secondary_metric=cfg.get("secondary_metric", "total_return"),
            metric_weights=cfg.get("metric_weights", (1.0, 0.0)),
            parallel_backend=cfg.get("parallel_backend", "dask"),
            neighbor_count=cfg.get("neighbor_count", 5),
            pqs_n_ref=cfg.get("pqs_n_ref", 50),
            stage_consensus_method=cfg.get("stage_consensus_method", "median_mode"),
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
                # Safe formatting — value may be None (failed stage) or nan (no trades)
                _fs = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else "n/a"
                _fr = lambda v: f"{v:+.1f}%" if isinstance(v, (int, float)) else "n/a"
                st.markdown(
                    f"**{st_icon} Run {stage_num}**  \n"
                    f"{name.split('—')[-1].strip()}  \n"
                    f"IS Sharpe `{_fs(is_s)}`  OOS `{_fs(oos_s)}`  \n"
                    f"IS Ret `{_fr(is_r)}`  OOS `{_fr(oos_r)}`  \n"
                    f"⏱ {elapsed:.0f}s"
                )
            elif stage_num == current_stage and status == "running":
                st.markdown(f"**⏳ Run {stage_num}**  \n{name.split('—')[-1].strip()}  \n*en cours…*")
            else:
                st.markdown(f"**⬜ Run {stage_num}**  \n{name.split('—')[-1].strip()}")

    # ── convert to standard wfo_results when done ────────────────────────────
    if status == "done":
        final_report = run_state.get("final_report", {})
        if final_report and not st.session_state.get("_sw_params_loaded"):
            # Build standard wfo_results so regular visualization + ZIP export work
            _cfg = st.session_state.get("_sw_cfg", {})
            _run_ts = _cfg.get("run_ts", time.strftime("%Y%m%d_%H%M%S"))
            synthetic_wfo = build_stagewise_wfo_results(final_report)
            if synthetic_wfo:
                st.session_state["wfo_results"] = synthetic_wfo
                st.session_state["wfo_run_metadata"] = {
                    "run_id": f"stagewise_{_run_ts}",
                    "status": "completed",
                    "strategy_mode": "native_atdmf",
                    "source": "stagewise",
                    "n_stages": final_report.get("n_stages"),
                    "campaign_start": final_report.get("campaign_start"),
                    "campaign_end": final_report.get("campaign_end"),
                }
                # Auto-save a wfo_results-compatible ZIP for replay and historization
                _zip_path = _auto_save_stagewise_zip(synthetic_wfo, final_report, _cfg, _run_ts)
                if _zip_path:
                    st.session_state["_sw_zip_path"] = _zip_path

            # Also populate sidebar inputs with the accumulated best params
            load_stagewise_params_from_json(final_report)
            st.session_state["_sw_params_loaded"] = True

            # Full app rerun: WFO visualization + sidebar export + Final Backtest activate
            try:
                st.rerun(scope="app")
            except TypeError:
                st.rerun()

        # Show optional headless final backtest metrics (if campaign included one)
        fb = final_report.get("final_backtest", {}) if final_report else {}
        if fb.get("status") == "OK":
            m = fb.get("metrics", {})
            st.markdown("---")
            st.markdown("##### Backtest final (headless)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sharpe",  f"{m.get('sharpe', 0):.3f}")
            c2.metric("Return",  f"{m.get('return', 0):+.2f}%")
            c3.metric("Max DD",  f"{m.get('max_drawdown', 0):.2f}%")
            c4.metric("Trades",  str(m.get("n_trades", 0)))

        if st.session_state.get("wfo_results", {}).get("_source") == "stagewise":
            _zip_path = st.session_state.get("_sw_zip_path", "")
            _cfg_disp = st.session_state.get("_sw_cfg", {})
            st.info(
                f"✅ Résultats WFO chargés — visualisation IS/OOS disponible ci-dessous.  \n"
                f"📁 Fichiers stage : `{_cfg_disp.get('output_dir', '')}` "
                f"| 💾 ZIP : `{_zip_path}`"
            )
        else:
            st.success("Params chargés dans les inputs. Tu peux lancer le Final Backtest.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ═══════════════════════════════════════════════════════════════════════════════
def render_stagewise_panel(*, get_current_config, load_data):
    """Entry point — call from app.py."""

    st.markdown("### 🎯 Campagne WFO Stagewise")

    status = st.session_state.get("_sw_run_state", {}).get("status", "idle")
    is_running = (status == "running")

    # ── read all settings from sidebar config ─────────────────────────────────
    config = get_current_config()

    sw_direction   = config.get("strategy_direction", "long_only")
    sw_timeframe   = config.get("timeframe", "5s")
    sw_n_windows   = int(config.get("n_windows", 6))
    sw_train_size  = float(config.get("train_size", 0.75))
    sw_anchored    = bool(config.get("anchored", False))
    sw_start       = str(config.get("start_date", ""))[:10]
    sw_end         = str(config.get("end_date", ""))[:10]

    opt_metric     = config.get("metric1_name", "sharpe_ratio")
    sec_metric     = config.get("metric2_name", "total_return")
    w1             = float(config.get("weight_metric1", 1.0))
    w2             = float(config.get("weight_metric2", 0.0))
    metric_weights = (w1, w2)

    parallel_backend = config.get("parallel_backend", "dask")
    neighbor_count   = int(config.get("neighbor_count", 5))
    pqs_n_ref        = int(config.get("pqs_n_ref", 50))

    order_sizing_mode = config.get("order_sizing_mode", "percent_equity")
    order_fixed_cash  = float(config.get("order_fixed_cash", 10000.0))
    fees_pct          = float(config.get("fees_pct", 0.0))

    # Data file: prefer sidebar CSV path; show input only when not configured
    from_file    = config.get("from_file", False)
    _sidebar_csv = config.get("file_path", "")
    sw_data_file = _sidebar_csv if (from_file and _sidebar_csv) else ""

    # ── configuration ────────────────────────────────────────────────────────
    with st.expander("⚙️ Configuration de la campagne", expanded=(status == "idle")):

        sizing_label = order_sizing_mode
        if order_sizing_mode == "fixed_cash":
            sizing_label += f" ({order_fixed_cash:,.0f}$)"
        st.info(
            f"**Paramètres hérités du panneau latéral**  \n"
            f"📅 Période WFO : `{sw_start}` → `{sw_end}` | Timeframe : `{sw_timeframe}` "
            f"| Direction : `{sw_direction}`  \n"
            f"🪟 Fenêtres : `{sw_n_windows}` | Train size : `{sw_train_size:.0%}` "
            f"| Ancré : `{sw_anchored}`  \n"
            f"📊 Métrique : `{opt_metric}` ({w1:.2f}) + `{sec_metric}` ({w2:.2f}) "
            f"| PQS n_ref : `{pqs_n_ref}`  \n"
            f"⚙️ Backend : `{parallel_backend}` | Voisins : `{neighbor_count}` "
            f"| Fees : `{fees_pct:.4f}%`  \n"
            f"💾 Sizing : `{sizing_label}`"
        )

        # Data file — only show input when not pre-configured from sidebar
        if from_file and _sidebar_csv:
            st.caption(f"📁 Données : `{_sidebar_csv}`")
        else:
            sw_data_file = st.text_input(
                "Fichier données (CSV)",
                value=sw_data_file,
                key="_sw_data_file",
                disabled=is_running,
                help="Chemin vers le CSV OHLCV. Activez 'Charger depuis fichier CSV' dans la sidebar pour pré-remplir automatiquement.",
            )

        sw_output = st.text_input(
            "Répertoire de sortie", value=_DEFAULT_OUTPUT,
            key="_sw_output", disabled=is_running,
        )

        _sc_methods = ["median_mode", "best_window", "weighted_oos_median"]
        _sc_labels  = {
            "median_mode":        "Consensus médiane/mode (défaut)",
            "best_window":        "Best-window du run (IS+OOS)",
            "weighted_oos_median": "Médiane pondérée OOS",
        }
        sw_stage_consensus = st.selectbox(
            "Consensus cross-fenêtres stagewise (Niveau 3)",
            options=_sc_methods,
            format_func=lambda x: _sc_labels[x],
            index=0,
            key="_sw_stage_consensus",
            disabled=is_running,
            help="Méthode d'agrégation des best_params de chaque fenêtre WFO pour fixer les paramètres du run suivant.",
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

        # When data file changes, refresh final backtest date defaults from the CSV
        if sw_data_file and sw_data_file != st.session_state.get("_sw_last_data_file"):
            if Path(sw_data_file).exists():
                try:
                    import os as _os
                    _mtime = _os.path.getmtime(sw_data_file)
                    _ds, _de = _get_csv_date_range(sw_data_file, _mtime)
                    if _ds and _de:
                        st.session_state["_sw_fb_start"] = _ds
                        st.session_state["_sw_fb_end"]   = _de
                except Exception:
                    pass
            st.session_state["_sw_last_data_file"] = sw_data_file

        st.session_state.setdefault("_sw_fb_start", sw_start)
        st.session_state.setdefault("_sw_fb_end",   sw_end)

        # Final backtest option
        st.markdown("**Backtest final**")
        col_fb1, col_fb2, col_fb3 = st.columns([1, 2, 2])
        sw_run_fb = col_fb1.checkbox(
            "Activer", value=True, key="_sw_run_fb", disabled=is_running,
        )
        sw_fb_start = col_fb2.text_input(
            "Début backtest final",
            key="_sw_fb_start", disabled=is_running or not sw_run_fb,
        )
        sw_fb_end = col_fb3.text_input(
            "Fin backtest final",
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
            width="stretch",
            key="_sw_btn_start",
        ):
            # Load data
            try:
                df = _load_csv(sw_data_file, sw_start, sw_end)
            except Exception as exc:
                st.error(f"Erreur chargement données : {exc}")
                st.stop()

            # Store df so visualization is available after campaign completes
            st.session_state["df"] = df
            st.session_state["opt_start_date"] = sw_start
            st.session_state["opt_end_date"] = sw_end

            final_df = None
            if sw_run_fb:
                try:
                    final_df = _load_csv(sw_data_file, sw_fb_start, sw_fb_end)
                except Exception as exc:
                    st.warning(f"Backtest final désactivé (erreur chargement) : {exc}")

            # Timestamped output dir — each run gets its own subdirectory
            _run_ts = time.strftime("%Y%m%d_%H%M%S")
            _run_output = str(Path(sw_output) / _run_ts)

            # Build campaign config — all settings from sidebar UI
            cfg = {
                "timeframe":           sw_timeframe,
                "direction":           sw_direction,
                "n_windows":           sw_n_windows,
                "train_size":          sw_train_size,
                "anchored":            sw_anchored,
                "output_dir":          _run_output,
                "run_ts":              _run_ts,
                "selected_stages":     selected_stages,
                "data_file":           sw_data_file,
                "order_sizing_mode":   order_sizing_mode,
                "order_fixed_cash":    order_fixed_cash,
                "fees_pct":            fees_pct,
                "optimization_metric":    opt_metric,
                "secondary_metric":      sec_metric,
                "metric_weights":        metric_weights,
                "parallel_backend":      parallel_backend,
                "neighbor_count":        neighbor_count,
                "pqs_n_ref":             pqs_n_ref,
                "stage_consensus_method": st.session_state.get("_sw_stage_consensus", "median_mode"),
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
            width="stretch",
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
