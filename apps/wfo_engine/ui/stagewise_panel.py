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

from domain.serialization import sanitize_for_json as _sanitize_for_json

from stagewise_optimizer import (
    STAGE_PLAN,
    FULL_PARAM_REGISTRY,
    run_stagewise_campaign,
    build_stagewise_wfo_results,
    propose_stage_settings,
    _load_csv,
)
from config import DEFAULT_PARAM_GRID
from services.error_log import write_error_log, collect_stagewise_entries
from ui.final_backtest_panel import load_stagewise_params_from_json

# ── constants ─────────────────────────────────────────────────────────────────
_DEFAULT_OUTPUT = "reports/stagewise"
_POLL_SECONDS   = 2        # fragment refresh rate


def _param_label(key: str, param_ranges: dict) -> str:
    """Format a param key as 'key [min→max, ×step]' or 'key [True/False]'."""
    spec = FULL_PARAM_REGISTRY.get(key)
    if spec is None:
        return key
    if isinstance(spec, list):
        return f"{key}  [True/False]"
    min_v, max_v, step = param_ranges.get(key, spec)
    def _f(v: float) -> str:
        return str(int(v)) if float(v) == int(float(v)) else f"{v:g}"
    return f"{key}  [{_f(min_v)}→{_f(max_v)}, ×{_f(step)}]"


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
                        json.dumps(_sanitize_for_json(payload), indent=2, allow_nan=False, default=str))
            zf.writestr("stagewise_final_report.json",
                        json.dumps(_sanitize_for_json(final_report), indent=2, allow_nan=False, default=str))

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

    plan = cfg.get("stage_plan", STAGE_PLAN)
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
            param_ranges=cfg.get("param_ranges"),
        )
        run_state["final_report"] = final_report

        # Write error log — captures FAILED stages + non-positive OOS Sharpe warnings
        _entries = collect_stagewise_entries(final_report)
        _failed  = [s for s in final_report.get("stages", []) if s.get("status") == "FAILED"]
        _log_path = write_error_log(
            run_type="stagewise",
            config=cfg,
            entries=_entries,
            output_dir=cfg["output_dir"],
            run_ts=cfg.get("run_ts"),
            status="FAILED" if _failed else "OK",
        )
        run_state["error_log_path"] = _log_path
        run_state["status"] = "done"
    except Exception as exc:
        _log_path = write_error_log(
            run_type="stagewise",
            config=cfg,
            entries=[{"level": "ERROR", "context": "campaign", "message": str(exc)}],
            output_dir=cfg.get("output_dir", "reports/error_logs"),
            run_ts=cfg.get("run_ts"),
            status="FAILED",
            fatal_error=str(exc),
        )
        run_state["error_log_path"] = _log_path
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
    _plan = st.session_state.get("_sw_cfg", {}).get("stage_plan", STAGE_PLAN)

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
    cols = st.columns(min(n_stages, 4))

    for i, stage in enumerate(_plan):
        stage_num = i + 1
        col = cols[i % len(cols)]
        name = stage.get("name", f"Run {stage_num}")
        done = next((c for c in completed if c["stage"] == stage_num), None)

        with col:
            if done:
                st_icon = "✅" if done["status"] == "OK" else "❌"
                is_s  = done.get("is_sharpe")
                oos_s = done.get("oos_sharpe")
                is_r  = done.get("is_return")
                oos_r = done.get("oos_return")
                elapsed = done.get("elapsed") or 0
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
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Sharpe",      f"{m.get('sharpe', 0):.3f}")
            c2.metric("Return",      f"{m.get('return', 0):+.2f}%")
            c3.metric("Mean P&L %",  f"{m.get('avg_pl_per_trade', 0):+.4f}%")
            c4.metric("Max DD",      f"{m.get('max_drawdown', 0):.2f}%")
            c5.metric("Trades",      str(m.get("n_trades", 0)))

        if st.session_state.get("wfo_results", {}).get("_source") == "stagewise":
            _zip_path = st.session_state.get("_sw_zip_path", "")
            _cfg_disp = st.session_state.get("_sw_cfg", {})
            _log_path = run_state.get("error_log_path", "")
            st.info(
                f"✅ Résultats WFO chargés — visualisation IS/OOS disponible ci-dessous.  \n"
                f"📁 Fichiers stage : `{_cfg_disp.get('output_dir', '')}` "
                f"| 💾 ZIP : `{_zip_path}`"
            )
            if _log_path:
                _lp = Path(_log_path)
                if _lp.exists():
                    st.download_button(
                        "📋 Télécharger journal erreurs",
                        data=_lp.read_text(encoding="utf-8"),
                        file_name=_lp.name,
                        mime="application/json",
                        key="dl_sw_error_log_info",
                    )
        else:
            _log_path = run_state.get("error_log_path", "")
            st.success("Params chargés dans les inputs. Tu peux lancer le Final Backtest.")
            if _log_path:
                _lp = Path(_log_path)
                if _lp.exists():
                    st.download_button(
                        "📋 Télécharger journal erreurs",
                        data=_lp.read_text(encoding="utf-8"),
                        file_name=_lp.name,
                        mime="application/json",
                        key="dl_sw_error_log_success",
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ═══════════════════════════════════════════════════════════════════════════════
def render_stagewise_panel(*, get_current_config, load_data):
    """Entry point — call from app.py."""

    st.markdown("### 🎯 Campagne WFO Stagewise")

    # R9 — cleanup orphaned thread state after crash or completion so the next
    # run does not inherit a stale _sw_run_state.  This mirrors the cleanup in
    # app.py for the classic WFO thread.
    _sw_thread = st.session_state.get("_sw_thread")
    if _sw_thread is not None and not _sw_thread.is_alive():
        _dead_status = st.session_state.get("_sw_run_state", {}).get("status", "idle")
        if _dead_status not in ("running",):
            for _k in ["_sw_thread", "_sw_stop_event"]:
                st.session_state.pop(_k, None)

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

        # ── Dynamic run builder ───────────────────────────────────────────────
        st.markdown("**Runs à exécuter**")

        # Build effective ranges from sidebar (Strategy Configuration panel)
        _param_ranges: dict = {}
        for _p, (_d_min, _d_max, _d_step) in DEFAULT_PARAM_GRID.items():
            _param_ranges[_p] = (
                float(st.session_state.get(f"min_{_p}", _d_min)),
                float(st.session_state.get(f"max_{_p}", _d_max)),
                float(st.session_state.get(f"step_{_p}", _d_step)),
            )

        _ALL_PARAM_KEYS = list(FULL_PARAM_REGISTRY.keys())

        # Initialize session_state run list from STAGE_PLAN defaults once
        if "_sw_runs" not in st.session_state or not st.session_state["_sw_runs"]:
            st.session_state["_sw_runs"] = [
                {"id": i + 1, "name": s["name"]}
                for i, s in enumerate(STAGE_PLAN)
            ]
            st.session_state["_sw_run_id_counter"] = len(STAGE_PLAN) + 1
            for i, s in enumerate(STAGE_PLAN):
                st.session_state[f"_sw_run_{i+1}_params"] = list(s["optimize"])
                st.session_state[f"_sw_run_{i+1}_settings_src"] = "algo"

        runs = st.session_state["_sw_runs"]

        # Track which run to delete (do it after rendering to avoid index shift)
        _delete_id = None

        for run in runs:
            rid = run["id"]
            with st.container(border=True):
                hcol1, hcol2, hcol3 = st.columns([3, 6, 1])
                run["name"] = hcol1.text_input(
                    "Nom du run",
                    value=run.get("name", f"Run {rid}"),
                    key=f"_sw_run_{rid}_name",
                    disabled=is_running,
                    label_visibility="collapsed",
                )
                _cur_params = st.session_state.get(
                    f"_sw_run_{rid}_params",
                    [k for k in _ALL_PARAM_KEYS[:4]],
                )
                selected_params = hcol2.multiselect(
                    "Paramètres à optimiser",
                    options=_ALL_PARAM_KEYS,
                    default=_cur_params,
                    format_func=lambda k: _param_label(k, _param_ranges),
                    key=f"_sw_run_{rid}_params",
                    disabled=is_running,
                    label_visibility="collapsed",
                )
                with hcol3:
                    if st.button("🗑", key=f"_sw_run_{rid}_del", disabled=is_running,
                                 help="Supprimer ce run"):
                        _delete_id = rid

                # Proposed settings from algorithm — uses sidebar ranges for n_combos
                _proposal = propose_stage_settings(selected_params, _param_ranges)
                _src_key = f"_sw_run_{rid}_settings_src"
                st.session_state.setdefault(_src_key, "algo")

                pcol1, pcol2 = st.columns([3, 1])
                _method_label = {
                    "grid":     "Grid exhaustif",
                    "bayesian": "Bayésien",
                }.get(_proposal["method"], _proposal["method"])
                pcol1.caption(
                    f"💡 Algorithme → **{_method_label}** | "
                    f"Trials : **{_proposal['max_trials']}** | "
                    f"Patience : **{_proposal['patience']}** | "
                    f"Stabilité : **{_proposal['stability']}** | "
                    f"Combinaisons : **{_proposal['n_combos']:,}**"
                )
                _src = pcol2.radio(
                    "Settings source",
                    options=["algo", "sidebar"],
                    format_func=lambda x: "Algorithme" if x == "algo" else "Panneau latéral",
                    index=0 if st.session_state[_src_key] == "algo" else 1,
                    key=_src_key,
                    disabled=is_running,
                    horizontal=True,
                    label_visibility="collapsed",
                )

        # Delete run outside the loop
        if _delete_id is not None:
            st.session_state["_sw_runs"] = [
                r for r in runs if r["id"] != _delete_id
            ]
            st.rerun()

        if not is_running:
            if st.button("➕ Ajouter un run", key="_sw_btn_add_run"):
                new_id = st.session_state["_sw_run_id_counter"]
                st.session_state["_sw_runs"].append(
                    {"id": new_id, "name": f"Run {new_id}"}
                )
                st.session_state[f"_sw_run_{new_id}_params"] = []
                st.session_state[f"_sw_run_{new_id}_settings_src"] = "algo"
                st.session_state["_sw_run_id_counter"] = new_id + 1
                st.rerun()

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
        _runs_now = st.session_state.get("_sw_runs", [])
        _has_valid_runs = any(
            bool(st.session_state.get(f"_sw_run_{r['id']}_params"))
            for r in _runs_now
        )
        start_disabled = is_running or not _has_valid_runs or not sw_data_file
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

            # Build stage_plan from dynamic run builder widget values
            _sidebar_method  = config.get("optimization_method", "bayesian")
            _sidebar_trials  = int(config.get("max_trials", 100))
            _sidebar_patience = config.get("patience_level", "Low")
            _stage_plan = []
            for _r in st.session_state.get("_sw_runs", []):
                _rid = _r["id"]
                _params = st.session_state.get(f"_sw_run_{_rid}_params", [])
                if not _params:
                    continue
                _src   = st.session_state.get(f"_sw_run_{_rid}_settings_src", "algo")
                _prop  = propose_stage_settings(_params)
                if _src == "algo":
                    _method   = _prop["method"]
                    _trials   = _prop["max_trials"]
                    _patience = _prop["patience"]
                    _stability = _prop["stability"]
                else:
                    _method   = _sidebar_method
                    _trials   = _sidebar_trials
                    _patience = _sidebar_patience
                    _stability = neighbor_count
                _stage_plan.append({
                    "name":            st.session_state.get(f"_sw_run_{_rid}_name", _r["name"]),
                    "optimize":        _params,
                    "fixed_overrides": {},
                    "method":          _method,
                    "max_trials":      _trials,
                    "patience":        _patience,
                    "stability":       _stability,
                })

            # Build campaign config — all settings from sidebar UI
            cfg = {
                "timeframe":           sw_timeframe,
                "direction":           sw_direction,
                "n_windows":           sw_n_windows,
                "train_size":          sw_train_size,
                "anchored":            sw_anchored,
                "output_dir":          _run_output,
                "run_ts":              _run_ts,
                "stage_plan":          _stage_plan,
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
                "param_ranges":          _param_ranges,
            }
            st.session_state["_sw_cfg"] = cfg

            # Shared mutable state (thread writes in-place)
            run_state: dict = {
                "status":        "running",
                "n_stages":      len(_stage_plan),
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
