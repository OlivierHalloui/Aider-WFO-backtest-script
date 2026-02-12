import argparse
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from analyze_exit_signals import run_exit_signal_analysis


st.set_page_config(
    page_title="Exit Signal Analyzer",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Exit Signal Analyzer")
st.markdown(
    "Analyse des signaux de sortie TradingView et generation de ponderations. "
    "Chargez un CSV, ajustez les options, puis lancez l'analyse."
)


# -----------------------------
# Sidebar - File selection
# -----------------------------
with st.sidebar:
    st.header("Fichier CSV")
    uploaded = st.file_uploader("Uploader un CSV TradingView", type=["csv"])
    csv_path = st.text_input(
        "Chemin du fichier CSV (si pas d'upload)",
        value="",
        help="Chemin local vers le CSV a analyser.",
    )

    outdir = st.text_input(
        "Dossier de sortie (optionnel)",
        value="",
        help="Laisser vide pour generer un dossier horodate dans reports/.",
    )

    pdf_path = st.text_input(
        "Chemin du rapport PDF (optionnel)",
        value="",
        help="Laisser vide pour utiliser <outdir>/exit_signal_report.pdf.",
    )


# Handle uploaded file persistence
input_path = None
if uploaded is not None:
    file_id = getattr(uploaded, "file_id", uploaded.name + str(uploaded.size))
    if st.session_state.get("uploaded_file_id") != file_id:
        uploads_dir = os.path.join("reports", "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        safe_name = uploaded.name.replace("/", "_").replace("\\", "_")
        save_path = os.path.join(uploads_dir, f"{file_id}_{safe_name}")
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.session_state["uploaded_file_id"] = file_id
        st.session_state["uploaded_file_path"] = save_path
    input_path = st.session_state.get("uploaded_file_path")
else:
    input_path = csv_path.strip() or None


# -----------------------------
# Sidebar - Options
# -----------------------------
with st.sidebar:
    st.header("Filtres")
    type_filter = st.text_input(
        "Type contient",
        value="Exit",
        help="Filtre sur la colonne Type (ex: Exit).",
    )
    exclude_signals = st.text_area(
        "Exclure les signaux (liste separee par virgules)",
        value="Open",
        help="Ex: Open, SomeSignal",
    )
    min_exits = st.number_input(
        "Min exits pour graphiques",
        min_value=1,
        value=1,
        help="Exclut les signaux trop rares des graphiques.",
    )

    st.header("Ponderation et fiabilite")
    min_exits_weight = st.number_input(
        "Min exits pour poids",
        min_value=1,
        value=20,
        help="Sous ce seuil, le poids est force a 0.",
    )
    confidence_k = st.number_input(
        "Facteur confiance k",
        min_value=1.0,
        value=50.0,
        step=1.0,
        help="Controle la penalisation des petits echantillons.",
    )
    negative_pnl_threshold = st.number_input(
        "Seuil P&L negatif",
        value=0.0,
        help="Si total P&L <= seuil, poids = 0.",
    )
    negative_expectancy_threshold = st.number_input(
        "Seuil expectancy negatif",
        value=0.0,
        help="Si expectancy <= seuil, poids = 0.",
    )
    min_positive_weight_pct = st.number_input(
        "Poids minimum pour signaux non-negatifs (%)",
        min_value=0.0,
        value=1.0,
        step=0.1,
        help="Garantit un plancher pour les signaux eligibles.",
    )
    base_exit_pct = st.number_input(
        "Base exit % a repartir",
        min_value=0.0,
        value=10.0,
        step=0.1,
        help="Ex: 10 signifie que la somme des poids vaut 10% de la position.",
    )

    st.header("Poids des criteres")
    presets = {
        "Agressif": {
            "w_expectancy": 0.35,
            "w_winrate": 0.10,
            "w_profit_factor": 0.15,
            "w_sharpe": 0.10,
            "w_drawdown": 0.05,
            "w_total_pnl": 0.20,
            "w_stability": 0.05,
        },
        "Equilibre": {
            "w_expectancy": 0.30,
            "w_winrate": 0.15,
            "w_profit_factor": 0.10,
            "w_sharpe": 0.15,
            "w_drawdown": 0.10,
            "w_total_pnl": 0.10,
            "w_stability": 0.10,
        },
        "Conservateur": {
            "w_expectancy": 0.20,
            "w_winrate": 0.20,
            "w_profit_factor": 0.10,
            "w_sharpe": 0.20,
            "w_drawdown": 0.15,
            "w_total_pnl": 0.05,
            "w_stability": 0.10,
        },
    }
    preset_choice = st.selectbox(
        "Preset",
        options=["Aucun", "Agressif", "Equilibre", "Conservateur"],
        help="Applique un ensemble de poids predefini.",
    )
    if st.button("Appliquer le preset"):
        if preset_choice != "Aucun":
            for key, val in presets[preset_choice].items():
                st.session_state[key] = val

    for key, default in presets["Equilibre"].items():
        st.session_state.setdefault(key, default)
    w_expectancy = st.slider(
        "Poids expectancy",
        0.0,
        1.0,
        value=st.session_state["w_expectancy"],
        step=0.05,
        key="w_expectancy",
        help="Influence de l'expectancy dans le score global.",
    )
    w_winrate = st.slider(
        "Poids win rate",
        0.0,
        1.0,
        value=st.session_state["w_winrate"],
        step=0.05,
        key="w_winrate",
        help="Importance du win rate.",
    )
    w_profit_factor = st.slider(
        "Poids profit factor",
        0.0,
        1.0,
        value=st.session_state["w_profit_factor"],
        step=0.05,
        key="w_profit_factor",
        help="Importance du profit factor (poids reduit).",
    )
    w_sharpe = st.slider(
        "Poids sharpe-like",
        0.0,
        1.0,
        value=st.session_state["w_sharpe"],
        step=0.05,
        key="w_sharpe",
        help="Ratio moyen/volatilite des P&L %.",
    )
    w_drawdown = st.slider(
        "Poids drawdown",
        0.0,
        1.0,
        value=st.session_state["w_drawdown"],
        step=0.05,
        key="w_drawdown",
        help="Penalise les signaux avec gros drawdown.",
    )
    w_total_pnl = st.slider(
        "Poids P&L total",
        0.0,
        1.0,
        value=st.session_state["w_total_pnl"],
        step=0.05,
        key="w_total_pnl",
        help="Favorise les signaux contribuant au P&L total.",
    )
    w_stability = st.slider(
        "Poids stabilite",
        0.0,
        1.0,
        value=st.session_state["w_stability"],
        step=0.05,
        key="w_stability",
        help="Penalise l'instabilite (downside deviation).",
    )

    st.caption(
        f"Somme des poids: {w_expectancy + w_winrate + w_profit_factor + w_sharpe + w_drawdown + w_total_pnl + w_stability:.2f}"
    )


# -----------------------------
# Run analysis
# -----------------------------
run = st.button("Lancer l'analyse")

if run:
    if not input_path or not os.path.exists(input_path):
        st.error("Fichier CSV introuvable. Verifiez le chemin ou uploadez un fichier.")
    else:
        exclude_list = [s.strip() for s in exclude_signals.split(",") if s.strip()]
        args = argparse.Namespace(
            input=input_path,
            outdir=outdir or None,
            type_filter=type_filter,
            exclude_signal=exclude_list,
            min_exits=int(min_exits),
            pdf_report=pdf_path or None,
            min_exits_weight=int(min_exits_weight),
            confidence_k=float(confidence_k),
            negative_pnl_threshold=float(negative_pnl_threshold),
            negative_expectancy_threshold=float(negative_expectancy_threshold),
            min_positive_weight_pct=float(min_positive_weight_pct),
            base_exit_pct=float(base_exit_pct),
            w_expectancy=float(w_expectancy),
            w_winrate=float(w_winrate),
            w_profit_factor=float(w_profit_factor),
            w_sharpe=float(w_sharpe),
            w_drawdown=float(w_drawdown),
            w_total_pnl=float(w_total_pnl),
            w_stability=float(w_stability),
        )

        with st.spinner("Analyse en cours..."):
            try:
                result = run_exit_signal_analysis(args, verbose=False)
            except SystemExit as exc:
                st.error(str(exc))
                st.stop()
            except Exception as exc:
                st.error(f"Erreur pendant l'analyse: {exc}")
                st.stop()

        st.success("Analyse terminee")
        st.write(f"Dossier de sortie: `{result['outdir']}`")
        if result.get("pdf_path"):
            st.write(f"PDF: `{result['pdf_path']}`")

        # Metrics from journal
        def compute_metrics(pnl_usdt: pd.Series, pnl_pct: pd.Series) -> dict:
            pnl_usdt = pnl_usdt.dropna()
            pnl_pct = pnl_pct.dropna()
            total_pnl = float(pnl_usdt.sum())
            avg_pnl = float(pnl_usdt.mean()) if len(pnl_usdt) else 0.0
            wins = pnl_usdt[pnl_usdt > 0]
            losses = pnl_usdt[pnl_usdt < 0]
            win_rate = float((pnl_usdt > 0).mean()) if len(pnl_usdt) else 0.0
            profit_factor = float("inf")
            if losses.sum() < 0:
                profit_factor = float(wins.sum() / abs(losses.sum()))
            sharpe_like = 0.0
            if len(pnl_pct) and pnl_pct.std(ddof=0) > 0:
                sharpe_like = float(pnl_pct.mean() / pnl_pct.std(ddof=0))
            cum = pnl_usdt.cumsum()
            drawdown = cum - cum.cummax()
            max_dd = float(drawdown.min()) if len(drawdown) else 0.0
            return {
                "trades": int(len(pnl_usdt)),
                "total_pnl": total_pnl,
                "avg_pnl": avg_pnl,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "max_drawdown": max_dd,
                "sharpe_like": sharpe_like,
            }

        # Summary tables
        summary = result["summary"].copy()
        st.subheader("Resume des signaux")
        st.dataframe(summary)

        st.subheader("Poids proposes")
        weights_view = summary.sort_values("weight_pct", ascending=False)[
            [
                "Signal",
                "exits",
                "total_pnl_usdt",
                "expectancy_usdt",
                "weight_pct",
                "recommended_exit_pct",
            ]
        ]
        st.dataframe(weights_view)

        # Simulation with weighted exits
        st.subheader("Simulation des performances (re-ponderee)")
        if base_exit_pct <= 0:
            st.warning("Base exit % doit etre > 0 pour simuler.")
        else:
            exit_df = result["exit_df"].copy()
            exit_df = exit_df.sort_values("Date and time")
            weight_map = summary.set_index("Signal")["recommended_exit_pct"].to_dict()
            exit_df["recommended_exit_pct"] = exit_df["Signal"].map(weight_map).fillna(0.0)
            factor = exit_df["recommended_exit_pct"] / float(base_exit_pct)
            exit_df["sim_pnl_usdt"] = exit_df["Net P&L USDT"] * factor
            exit_df["sim_pnl_pct"] = exit_df["Net P&L %"] * factor
            exit_df["cum_pnl_usdt"] = exit_df["Net P&L USDT"].cumsum()
            exit_df["sim_cum_pnl_usdt"] = exit_df["sim_pnl_usdt"].cumsum()

            st.caption(
                "Simulation approximate: P&L de chaque sortie repondere selon le % recommande. "
                "Ne re-joue pas la strategie si un signal est coupe a 0%."
            )

            st.line_chart(
                exit_df.set_index("Date and time")[["cum_pnl_usdt", "sim_cum_pnl_usdt"]],
                height=350,
            )

            # Metrics comparison
            st.subheader("Metriques journal (original vs re-pondere)")
            orig_metrics = compute_metrics(exit_df["Net P&L USDT"], exit_df["Net P&L %"])
            sim_metrics = compute_metrics(exit_df["sim_pnl_usdt"], exit_df["sim_pnl_pct"])

            col_o, col_s = st.columns(2)
            with col_o:
                st.markdown("**Original**")
                st.metric("P/L total (USDT)", f"{orig_metrics['total_pnl']:.2f}")
                st.metric("P/L moyen par trade", f"{orig_metrics['avg_pnl']:.4f}")
                st.metric("Win rate", f"{orig_metrics['win_rate']*100:.2f}%")
                pf = "inf" if orig_metrics["profit_factor"] == float("inf") else f"{orig_metrics['profit_factor']:.2f}"
                st.metric("Profit factor", pf)
                st.metric("Max drawdown (USDT)", f"{orig_metrics['max_drawdown']:.2f}")
                st.metric("Sharpe-like", f"{orig_metrics['sharpe_like']:.3f}")

            with col_s:
                st.markdown("**Re-pondere**")
                st.metric("P/L total (USDT)", f"{sim_metrics['total_pnl']:.2f}")
                st.metric("P/L moyen par trade", f"{sim_metrics['avg_pnl']:.4f}")
                st.metric("Win rate", f"{sim_metrics['win_rate']*100:.2f}%")
                pf = "inf" if sim_metrics["profit_factor"] == float("inf") else f"{sim_metrics['profit_factor']:.2f}"
                st.metric("Profit factor", pf)
                st.metric("Max drawdown (USDT)", f"{sim_metrics['max_drawdown']:.2f}")
                st.metric("Sharpe-like", f"{sim_metrics['sharpe_like']:.3f}")

            with st.expander("Voir les donnees simulees"):
                st.dataframe(exit_df)

        # Downloads
        st.subheader("Telechargements")
        st.download_button(
            "Telecharger summary CSV",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name=f"exit_signal_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        st.download_button(
            "Telecharger weights CSV",
            data=summary.sort_values("weight_pct", ascending=False).to_csv(index=False).encode("utf-8"),
            file_name=f"exit_signal_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        if result.get("pdf_path") and os.path.exists(result["pdf_path"]):
            with open(result["pdf_path"], "rb") as f:
                st.download_button(
                    "Telecharger rapport PDF",
                    data=f.read(),
                    file_name=os.path.basename(result["pdf_path"]),
                    mime="application/pdf",
                )

        # Charts
        st.subheader("Graphiques")
        if result.get("charts_ok"):
            for chart_path in result.get("chart_paths", []):
                if os.path.exists(chart_path):
                    st.image(chart_path, caption=os.path.basename(chart_path))
        else:
            st.warning(
                "Les graphiques ne sont pas disponibles (dependances plotting manquantes)."
            )
