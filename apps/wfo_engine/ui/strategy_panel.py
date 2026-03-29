"""Strategy parameter configuration panel for the WFOE central UI.

Renders the strategy configuration in the main Streamlit panel using 4 tabs:
  - Signaux Entrée  (T0, T1, T2 cascade)
  - Sorties         (exit mechanisms + toggles)
  - Filtres Croisements (Phase 7 — placeholder)
  - MTF             (Phase 6 — placeholder)

All widget values are persisted in st.session_state via their key= parameters.
get_current_config() in app.py reads them back from st.session_state.
"""

import streamlit as st
from config import DEFAULT_PARAM_GRID

# ---------------------------------------------------------------------------
# Parameter help text (single source of truth — imported by expert_panel.py)
# ---------------------------------------------------------------------------

PARAMETER_HELP = {
    # T0 — Bollinger Bands
    'timeperiod':       "Période des bandes de Bollinger (longueurBB).",
    'StDev':            "Nombre d'écarts-types pour les bandes de Bollinger (stdev_UTM).",
    'coeff_medianeBBW': "Coefficient du signal de compression BBW (bollingerHorizontalV3).",
    # T0 — BBW / CrossBBWLow
    'fenetre_lowest':   "Fenêtre de recherche du plus bas BBW (fn_cross_bbw_low_signal).",
    'seuil_lowest':     "Seuil multiplicateur du plus bas BBW (BBW ≤ bbw_lowest × seuil).",
    # T0 — Ecart BB%
    'coef_mediane':     "Coefficient médiane pour le signal d'écart borné BB% (Ecart_Bollinger_borne).",
    'longueur_mediane': "Longueur de fenêtre de la médiane pour le signal d'écart BB%.",
    'Nb_bars_above':    "Barres consécutives au-dessus de la médiane BB% requises (Nb_bars_above_BB).",
    # Exit — SMA
    'user_exit_sma_length': "Longueur de la SMA pour la sortie croisement baissier (SMAExit).",
    # Exit — SAR
    'sar_start':        "Valeur initiale du Parabolic SAR (SAR_start).",
    'sar_increment':    "Incrément du Parabolic SAR (SAR_increment).",
    'sar_maximum':      "Valeur maximale du facteur d'accélération SAR (SAR_maximum).",
    # Exit — MACD
    'macd_fast_length':   "Période rapide du MACD (SMA ou EMA selon macd_ma_type). Pine V6: 9.",
    'macd_slow_length':   "Période lente du MACD. Pine V6: 19.",
    'macd_signal_length': "Période de la ligne signal MACD. Pine V6: 6.",
}


# ---------------------------------------------------------------------------
# param_input helper — renders checkbox + min/max/step for one optimizable param
# ---------------------------------------------------------------------------

def param_input(
    key: str,
    label: str,
    default_min: float,
    default_max: float,
    default_step: float,
):
    """Render one optimizable parameter row (checkbox + min/max/step inputs).

    Returns (enabled, min_val, max_val, step_val).
    Widget session_state keys: check_{key}, min_{key}, max_{key}, step_{key}.
    """
    display_label = str(label).replace("_", " ")
    c_toggle, c_label = st.columns([0.14, 0.86])
    with c_toggle:
        enabled = st.checkbox(
            "Activer",
            value=True,
            key=f"check_{key}",
            label_visibility="collapsed",
            help=PARAMETER_HELP.get(key, "Active/désactive ce paramètre dans l'optimisation."),
        )
    with c_label:
        st.markdown(f"**{display_label}**")
        st.caption(PARAMETER_HELP.get(key, ""))

    c2, c3, c4 = st.columns([1, 1, 1])
    with c2:
        min_val = st.number_input(
            "Min",
            value=float(default_min),
            key=f"min_{key}",
            disabled=not enabled,
            help=f"Borne minimale testée pour `{key}`.",
        )
    with c3:
        max_val = st.number_input(
            "Max",
            value=float(default_max),
            key=f"max_{key}",
            disabled=not enabled,
            help=f"Borne maximale testée pour `{key}`.",
        )
    with c4:
        step_val = st.number_input(
            "Step",
            value=float(default_step),
            key=f"step_{key}",
            disabled=not enabled,
            help=f"Pas d'incrément entre Min et Max pour `{key}`.",
        )
    st.markdown(
        "<div style='height:0.15rem;border-bottom:1px solid "
        "rgba(120,145,170,0.20);margin:0.25rem 0 0.45rem 0;'></div>",
        unsafe_allow_html=True,
    )
    return enabled, min_val, max_val, step_val


# ---------------------------------------------------------------------------
# Main panel renderer
# ---------------------------------------------------------------------------

def render_strategy_panel() -> None:
    """Render the strategy configuration panel in the main Streamlit area.

    Four tabs:
      - Signaux Entrée  : T0 / T1 / T2 cascade (phases 1–4)
      - Sorties         : exit mechanisms (phases 1, 5)
      - Filtres Croisements : UTM/UTP crossings (phase 7 — placeholder)
      - MTF             : multi-timeframe filters (phase 6 — placeholder)

    All values are written to st.session_state via widget key= parameters.
    """
    with st.expander("⚙️ Configuration Stratégie", expanded=True):
        tab_entry, tab_exits, tab_filters, tab_mtf = st.tabs([
            "🎯 Signaux Entrée",
            "🚪 Sorties",
            "🔀 Filtres Croisements",
            "⏱ MTF",
        ])

        # -----------------------------------------------------------------------
        # TAB 1 — Signaux Entrée
        # -----------------------------------------------------------------------
        with tab_entry:
            st.caption(
                "Paramètres de la cascade T0 → T1 → T2 — alignement Pine V6 phases 1–4. "
                "Cochez ✅ pour inclure dans l'optimisation par grille."
            )

            # --- T0: Bollinger Bands ---
            st.markdown("##### 🔷 T0 — Bandes de Bollinger")
            col_bb1, col_bb2 = st.columns(2)
            with col_bb1:
                for p in ('timeperiod', 'StDev'):
                    d = DEFAULT_PARAM_GRID[p]
                    param_input(p, p, *d)
            with col_bb2:
                for p in ('coeff_medianeBBW',):
                    d = DEFAULT_PARAM_GRID[p]
                    param_input(p, p, *d)

            st.divider()

            # --- T0: BBW compression ---
            st.markdown("##### 🔷 T0 — Compression BBW")
            col_bbw1, col_bbw2 = st.columns(2)
            with col_bbw1:
                for p in ('fenetre_lowest', 'seuil_lowest'):
                    d = DEFAULT_PARAM_GRID[p]
                    param_input(p, p, *d)
            with col_bbw2:
                for p in ('nb_bars_under_bbw_mini', 'nb_bars_entre_bb'):
                    d = DEFAULT_PARAM_GRID[p]
                    param_input(p, p, *d)

            st.divider()

            # --- T0: Ecart BB% ---
            st.markdown("##### 🔷 T0 — Écart Bollinger borné (BB%)")
            col_ecart1, col_ecart2 = st.columns(2)
            with col_ecart1:
                for p in ('coef_mediane', 'longueur_mediane'):
                    d = DEFAULT_PARAM_GRID[p]
                    param_input(p, p, *d)
            with col_ecart2:
                for p in ('Nb_bars_above',):
                    d = DEFAULT_PARAM_GRID[p]
                    param_input(p, p, *d)

            st.divider()

            # --- T1: RoC filter ---
            st.markdown("##### 🔶 T1 — Filtre momentum RoC")
            use_roc = st.checkbox(
                "Activer filtre RoC",
                value=bool(st.session_state.get("use_roc_filter", True)),
                key="use_roc_filter",
                help=(
                    "Active le filtre de momentum RoC sur le signal T1. "
                    "Désactiver supprime la condition RoC du signal d'entrée, "
                    "ce qui augmente le nombre de trades mais réduit la sélectivité."
                ),
            )
            if use_roc:
                st.caption("Paramètres fixes — valeurs Pine V6 par défaut.")
                c_roc1, c_roc2 = st.columns(2)
                with c_roc1:
                    st.number_input(
                        "Dépassement SMA RoC",
                        value=float(st.session_state.get("depassement_sma_roc", 0.01)),
                        min_value=0.0, step=0.005, format="%.3f",
                        key="depassement_sma_roc",
                        help=(
                            "RoC = (high−low)/high×100. Signal valide si "
                            "RoC ≥ Depass × SMA20(RoC). Pine V6: 0.01."
                        ),
                    )
                with c_roc2:
                    st.number_input(
                        "RoC Max",
                        value=float(st.session_state.get("roc_max_t1", 100.0)),
                        min_value=1.0, step=1.0,
                        key="roc_max_t1",
                        help="RoC ≤ RoC_Max × SMA20(RoC) — filtre mouvements extrêmes. Pine V6: 100.",
                    )

            st.divider()

            # --- T2 toggle ---
            st.markdown("##### 🟢 T2 — Signal final (breakout high après T1)")
            use_t2 = st.checkbox(
                "Activer signal T2",
                value=st.session_state.get("use_t2_signal", False),
                key="use_t2_signal",
                help=(
                    "T2 = T1[−1] AND high[t] > high[t−1].\n\n"
                    "Ordre stop-buy placé au High de la barre T1 + mintick (0.01).\n\n"
                    "Désactivé : entrée directe sur crossover de la bande supérieure (T1)."
                ),
            )
            if use_t2:
                st.success(
                    "T2 activé : ordre stop-buy au High T1 + 0.01. "
                    "⚠️ Réduit le nombre de trades."
                )
                st.markdown("###### Filtre Divergence BB")
                st.checkbox(
                    "Activer filtre Divergence BB",
                    value=st.session_state.get("use_divergence_bb", True),
                    key="use_divergence_bb",
                    help=(
                        "Divergence BB = upper band ↑ ET lower band ↓ simultanément sur la barre T1.\n\n"
                        "Filtre supplémentaire : T2 n'est valide que si les bandes sont en expansion "
                        "sur la barre setup (T1)."
                    ),
                )
                _optimize_div = st.checkbox(
                    "Optimiser Divergence BB (True / False)",
                    value=st.session_state.get("check_use_divergence_bb", False),
                    key="check_use_divergence_bb",
                    help="Si coché, l'optimiseur teste T2 avec et sans filtre Divergence BB.",
                )
                if _optimize_div:
                    st.session_state["use_divergence_bb_values"] = [True, False]
                    st.info("Divergence BB sera optimisée : [True, False].")
                else:
                    st.session_state["use_divergence_bb_values"] = None
            else:
                st.info("T2 désactivé : entrée directe sur crossover bande supérieure (T1).")
                st.session_state["use_divergence_bb_values"] = None

        # -----------------------------------------------------------------------
        # TAB 2 — Sorties
        # -----------------------------------------------------------------------
        with tab_exits:
            st.caption(
                "Mécanismes de sortie et paramètres associés. "
                "Les paramètres optimisables (✅) sont inclus dans la grille de recherche."
            )

            # --- SMA Exit (always on) ---
            st.markdown("##### SMA Exit *(toujours actif)*")
            for p in ('user_exit_sma_length',):
                d = DEFAULT_PARAM_GRID[p]
                param_input(p, p, *d)

            st.divider()

            # --- SAR Exit ---
            col_sar_hd, col_sar_toggle = st.columns([0.85, 0.15])
            with col_sar_hd:
                st.markdown("##### SAR Exit")
            with col_sar_toggle:
                sar_on = st.checkbox(
                    "On",
                    value=st.session_state.get("exit_sar_enabled", True),
                    key="exit_sar_enabled",
                    help="Sortie Parabolic SAR : crossunder(low, SAR_persistent) quand SAR > SMA. QTY=50% Pine.",
                )
            if sar_on:
                c_s1, c_s2, c_s3 = st.columns(3)
                with c_s1:
                    param_input('sar_start', 'sar_start', *DEFAULT_PARAM_GRID['sar_start'])
                with c_s2:
                    param_input('sar_increment', 'sar_increment', *DEFAULT_PARAM_GRID['sar_increment'])
                with c_s3:
                    param_input('sar_maximum', 'sar_maximum', *DEFAULT_PARAM_GRID['sar_maximum'])

            st.divider()

            # --- Cross SAR/SMA Exit ---
            st.markdown("##### Cross SAR/SMA Exit")
            st.checkbox(
                "Activer cross SAR/SMA",
                value=st.session_state.get("exit_cross_sar_sma_enabled", True),
                key="exit_cross_sar_sma_enabled",
                help=(
                    "Sortie quand SMA croise sous SAR (SAR passe au-dessus de la SMA). "
                    "QTY=100% Pine. Défaut Pine V6: activé."
                ),
            )

            st.divider()

            # --- MACD Exit ---
            col_macd_hd, col_macd_toggle = st.columns([0.85, 0.15])
            with col_macd_hd:
                st.markdown("##### MACD Exit")
            with col_macd_toggle:
                macd_on = st.checkbox(
                    "On",
                    value=st.session_state.get("exit_macd_enabled", True),
                    key="exit_macd_enabled",
                    help="Sortie sur croisement baissier MACD. QTY=25% Pine.",
                )
            if macd_on:
                c_mt, c_ma, c_mb = st.columns(3)
                with c_mt:
                    st.selectbox(
                        "Type MA MACD",
                        options=["sma", "ema"],
                        index=0 if st.session_state.get("macd_ma_type", "sma") == "sma" else 1,
                        key="macd_ma_type",
                        help="SMA = fidèle Pine V6 (fast_ma=ta.sma). EMA = comportement historique Python.",
                    )
                with c_ma:
                    st.checkbox(
                        "Type A (signal ↓)",
                        value=st.session_state.get("exit_macd_type_a", True),
                        key="exit_macd_type_a",
                        help="Croisement baissier MACD avec ligne signal en baisse (fn_MACD_croisement_type_A_bear).",
                    )
                with c_mb:
                    st.checkbox(
                        "Type B (simple)",
                        value=st.session_state.get("exit_macd_type_b", True),
                        key="exit_macd_type_b",
                        help="Croisement baissier MACD simple sans condition sur le signal (fn_MACD_croisement_type_B_bear).",
                    )
                c_f, c_s, c_sig = st.columns(3)
                with c_f:
                    param_input('macd_fast_length', 'macd_fast_length', *DEFAULT_PARAM_GRID['macd_fast_length'])
                with c_s:
                    param_input('macd_slow_length', 'macd_slow_length', *DEFAULT_PARAM_GRID['macd_slow_length'])
                with c_sig:
                    param_input('macd_signal_length', 'macd_signal_length', *DEFAULT_PARAM_GRID['macd_signal_length'])

            st.divider()

            # --- Phase 5 additional exits ---
            st.markdown("##### Sorties additionnelles (Pine V6 — Phase 5)")

            col_ex1, col_ex2 = st.columns(2)
            with col_ex1:
                st.checkbox(
                    "Retour BB (pivot bas bande inf.)",
                    value=st.session_state.get("exit_retour_bb_enabled", False),
                    key="exit_retour_bb_enabled",
                    help=(
                        "Sortie sur pivot bas détecté sur la bande inférieure BB "
                        "(fn_BB_retournement_lower_BB). QTY=50% Pine. Défaut: désactivé."
                    ),
                )
                if st.session_state.get("exit_retour_bb_enabled", False):
                    c_pv1, c_pv2 = st.columns(2)
                    with c_pv1:
                        st.number_input(
                            "Bars left pivot",
                            value=int(st.session_state.get("nb_bars_left_pivot", 2)),
                            min_value=1, max_value=20, step=1,
                            key="nb_bars_left_pivot",
                            help="Barres à gauche du pivot bas (Bars_left_pivot=2).",
                        )
                    with c_pv2:
                        st.number_input(
                            "Bars right pivot",
                            value=int(st.session_state.get("nb_bars_right_pivot", 2)),
                            min_value=1, max_value=10, step=1,
                            key="nb_bars_right_pivot",
                            help="Barres à droite du pivot bas (Bars_right_pivot=2).",
                        )

                st.checkbox(
                    "Regline Exit (crossunder régression linéaire)",
                    value=st.session_state.get("exit_regline_enabled", False),
                    key="exit_regline_enabled",
                    help=(
                        "Sortie quand le cours croise sous la droite de régression linéaire "
                        "(ta.crossunder(close, ta.linreg)). QTY=100% Pine. Défaut: désactivé."
                    ),
                )
                if st.session_state.get("exit_regline_enabled", False):
                    c_rl1, c_rl2 = st.columns(2)
                    with c_rl1:
                        st.number_input(
                            "Périodes régression",
                            value=int(st.session_state.get("nombre_periodes_reglin", 15)),
                            min_value=5, max_value=100, step=1,
                            key="nombre_periodes_reglin",
                            help="Longueur de la fenêtre de régression linéaire (Periode_Reglin=15).",
                        )
                    with c_rl2:
                        st.number_input(
                            "Bars back (offset)",
                            value=int(st.session_state.get("i_bars_back", 1)),
                            min_value=0, max_value=10, step=1,
                            key="i_bars_back",
                            help="Décalage temporel de la régression (iBarsBack=1).",
                        )

            with col_ex2:
                st.checkbox(
                    "Baisse Volatilité (%BB crossunder)",
                    value=st.session_state.get("exit_volat_down_enabled", False),
                    key="exit_volat_down_enabled",
                    help=(
                        "Sortie quand BBR = (close−lower)/(upper−lower) croise sous seuil_overbought "
                        "(BB_baisse_volat_signal_exit). QTY=25% Pine. Défaut: désactivé."
                    ),
                )
                if st.session_state.get("exit_volat_down_enabled", False):
                    st.number_input(
                        "Seuil overbought BB",
                        value=float(st.session_state.get("seuil_overbought_bb", 0.85)),
                        min_value=0.5, max_value=1.5, step=0.01, format="%.2f",
                        key="seuil_overbought_bb",
                        help="Seuil BBR pour le signal de baisse de volatilité (seuil_overbough_BB=0.85).",
                    )

        # -----------------------------------------------------------------------
        # TAB 3 — Filtres Croisements (Phase 7 — placeholder)
        # -----------------------------------------------------------------------
        with tab_filters:
            st.info(
                "**Phase 7 — À venir** : filtres de croisements UTM/UTP "
                "(Stochastique, SMA 7/23 anticipées, MACD en entrée).\n\n"
                "Ces filtres conditionnent le signal T1 en bloquant les entrées lorsque des "
                "croisements baissiers récents sont détectés sur la timeframe principale (UTM) "
                "ou supérieure (UTP).",
                icon="🔜",
            )
            with st.expander("Aperçu des filtres prévus (Phase 7)"):
                st.markdown("""
**UTM — Croisements timeframe principale :**
- `activ_cross_UTM_type_A_bear` *(défaut: True)* — crossunder K/D stochastique avec D décroissant
- `activ_cross_UTM_type_B_bear` *(défaut: False)* — crossunder K/D simple
- `activ_cross_UTM_type_A_bull` *(défaut: False)* — crossover K/D avec D croissant
- `histo_croisements_UTM_bear` *(défaut: 0)* — fenêtre historique en barres

**UTP — Filtres timeframe supérieure (1 min) :**
- Croisements STO / SMA 7/23 / MACD bear — bloquent T1 si actifs dans `histo_croisements_UTP` (défaut: 3)

**Stochastique :**
- periodK=14, smoothK=3, periodD=3, seuil_overbought=80

**SMA 7/23 anticipées** *(formule simplifiée)* :
```
sma7_anticip = SMA(close, 7) + (close − close[7]) / 7
```
Croisements type A (avec confirmation tendance) et type B (simple).
                """)

        # -----------------------------------------------------------------------
        # TAB 4 — MTF (Phase 6 — placeholder)
        # -----------------------------------------------------------------------
        with tab_mtf:
            st.info(
                "**Phase 6 — À venir** : filtre de tendance multi-timeframe (UTP SMA bullish).\n\n"
                "La SMA calculée sur la timeframe UTP (30S par défaut) conditionne le signal T1 : "
                "entrée bloquée si la tendance de la SMA courte ou longue est baissière.",
                icon="🔜",
            )
            with st.expander("Aperçu des paramètres MTF prévus (Phase 6)"):
                st.markdown("""
**UTP SMA Bullish :**
- `use_mtf_utp_sma` — activation (défaut: True)
- `tf_utp` — timeframe UTP (défaut: 30S)
- `lg_sma_utp1` — SMA courte UTP (défaut: 7)
- `lg_sma_utp2` — SMA longue UTP (défaut: 18)
- `sma_utp_duo` — OR(sma1↑, sma2↑) si True, sinon seulement sma1↑

**Filtres UTP croisements bear (1 min) :**
- STO / SMA 7/23 / MACD bear sur TF 1 min — via `request_security_series` de `pine_v3/mtf.py`

**Sorties UTC (données 1S requises) :**
- `exit_bb_utc_enabled` — sortie crossunder bande inférieure BB calculée sur 1S
- `exit_sma_utc_enabled` — sortie crossunder SMA calculée sur 1S
- `histo_bb_utc` / `histo_sma_utc` — fenêtre temporelle post-entrée (barres)
                """)
