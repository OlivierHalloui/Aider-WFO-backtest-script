# Changelog

Toutes les évolutions notables de l'application WFO sont documentées ici.

## 2026-06-13

Phase 5 du plan d'audit (`IMPLEMENTATION_PLAN.md`) :

- **Détection des runs interrompus (v1)** : `run_optimization_job` écrit un marqueur `completed.json` à chaque sortie (succès/stop/erreur). Au démarrage d'une session, `scan_orphaned_runs()` repère les répertoires `reports/runs/` avec checkpoint sans marqueur — notification UI avec résumé (fenêtres complétées, progression, timestamp) et bouton d'acquittement. La reprise d'un run interrompu n'est pas implémentée (chantier séparé).
- **Slippage configurable** : `slippage_bps` (sidebar, sous les frais) appliqué à chaque exécution via VectorBT. Défaut 0 = comportement historique sans friction. Ordre de grandeur crypto spot liquide : 1–5 bps.
- **Cache KDTree SNV** : en grid search, les indices de voisinage de la sélection SNV sont cachés entre fenêtres (ordre canonique + remap) — sélection ~4× plus rapide sur grosses grilles. Jamais actif en Bayesian/Optuna/régimes guidés (combos différents par fenêtre).
- **Dédup neural search : abandonné après profiling** — 0 % de doublons sur grille représentative (17 params, pool 3000) ; le coût des doublons sur petites grilles est négligeable (forward MLP, backtests dédupliqués par le cache).

## 2026-06-12

Corrections issues de l'audit (`AUDIT_REPORT.md`, plan `IMPLEMENTATION_PLAN.md`) :

- **Durabilité checkpoint réparée** : les erreurs d'écriture de `checkpoint.json` et des résultats de fenêtres ne sont plus silencieuses (`logger.error` + `exc_info`) ; les échecs remontent dans `job_state["warnings"]`.
- **Exports JSON valides (RFC 7159)** : `sanitize_for_json()` convertit `NaN`/`Inf` en `null` ; les deux paths d'export ZIP (`export_panel`, `stagewise_panel`) utilisent le sanitizer + `allow_nan=False`. Note : `NaN` → `null` est lossy au réimport.
- **Harvest run-resolution idempotent** : flag `harvested` dans `job_state` — un résultat de run ne peut plus être moissonné deux fois.
- **`MemoryError` re-levée** dans l'évaluation des combos (`wfo.py`) au lieu d'être convertie en NaN ; `safe_float()` restreint aux exceptions de conversion.
- **Parallélisme fenêtres opt-in** : `WFOSettings.max_parallel_windows` (défaut 1 = série, 0 = auto si ≥ 4 cœurs). Actif uniquement en régime classique — `nn_guided` et `prev_best_grid` restent séquentiels par design. Benchmark local 2 CPUs : speedup 1.43× (sous le seuil 1.5×) — défaut série conservé.
- **Holdout final optionnel** : `WFOSettings.holdout_fraction` réserve une tranche finale de données jamais vue par l'optimisation ; bornes exposées dans `wfo_results['holdout']`.
- **SVI** : docstring/tooltip requalifiés (IS₂ = re-ranking sur IS récent, pas un holdout OOS) ; `itertuples` remplace `iterrows`.
- **Thompson sampling vectorisé** (`adaptive_optimization.py`) : un seul appel `rng.normal` pour tous les candidats, équivalence statistique préservée (masquage des params absents).
- **`data_loading.py` robustifié** : valeurs CSV non numériques converties en NaN avec warning (au lieu de `ValueError`) ; fetch Binance avec message d'erreur actionnable et garde sur données vides.
- **Cleanup thread stagewise** : l'état `_sw_thread`/`_sw_stop_event` orphelin est purgé après fin ou crash du thread.
- **Facade `WFOSessionState` câblée** (`app.py`) : les lectures des clés run-resolution (`wfo_running`, `wfo_thread`, `wfo_job_state`, `wfo_prev_state`, `wfo_control`, `wfo_job_config`) et des sections visualisation/backtest stagewise passent par des snapshots typés `WFOSessionState.read()` — un snapshot par zone, pris après les écritures mid-run (import ZIP, panel stagewise). Les écritures restent directes (facade read-only par design).
- **Tests** : 80 nouveaux tests — services (`run_service`, `serialization`, `data_loading`), smoke engine sans VBT, intégration VBT réelle (backtest, métriques, SVI, adaptive, stagewise).

## 2026-04-18

- **WFO Stagewise — héritage complet de la configuration UI** : le panneau stagewise supprime les champs redondants (direction, timeframe, fenêtres, dates, fichier données). Tous ces paramètres sont lus depuis le panneau latéral (`get_current_config()`). Un bandeau d'information affiche les valeurs héritées. Le fichier de données se pré-remplit automatiquement si "Charger depuis fichier CSV" est actif dans la sidebar.
- **Sélection des meilleurs paramètres — 3 niveaux configurables** :
  - **Niveau 1 — intra-fenêtre** (`WFO Engine Settings`) : `SNV` Stabilité Voisinage KDTree (défaut), `SVI` Validation Interne IS₂ (évalue le top-K sur les 30% finaux de IS), `Score brut max`.
  - **Niveau 2 — cross-fenêtres** (`WFO Engine Settings`) : `Best IS+OOS` moyenne (défaut), `Best OOS seul`, `Robust Set` vote pondéré top-N, `Médiane pondérée OOS` agrégation de toutes les fenêtres.
  - **Niveau 3 — cross-stages stagewise** (panneau campagne) : `Consensus médiane/mode` (défaut), `Best-window du run` IS+OOS, `Médiane pondérée OOS`.
- `wfo.py` : ajout de `get_svi_best_params()` — évalue les top-K candidats IS sur une sous-période IS₂ de validation, fallback vers `raw_max` si IS₂ vide ou tous les candidats échouent.
- `final_backtest_panel.py` : `_select_final_params_from_results()` refactorisé en dispatcher sur `cross_window_method` ; ajout de `_select_best_params_oos_only()`, `_select_weighted_oos_params()`, `_combined_score_from_row()`.
- `stagewise_optimizer.py` : ajout de `_best_window_consensus()`, `_weighted_oos_median_consensus()`, `_weighted_median_1d()`, `_apply_stage_consensus()` ; `run_stagewise_campaign()` accepte `stage_consensus_method`.
- `config.py` : `WFOSettings` enrichi de `selection_method`, `svi_top_k`, `svi_is2_fraction`, `cross_window_method`. Rétrocompatibilité : les valeurs par défaut reproduisent exactement le comportement antérieur.

## 2026-04-05

- **Flags `optimize_*` par toggle booléen** : chaque signal d'entrée/sortie activable (SAR, MACD, MACD type A/B, RoC, T2, divergence_BB, cross SAR/SMA, retour BB, linreg, volat_down) dispose désormais d'une case à cocher "Optimiser" dans le panneau stratégie. Quand désactivée, le toggle est fixé à sa valeur courante dans la grille au lieu d'être varié `[True, False]`, ce qui réduit l'espace de recherche.
- `main.py` : `get_param_grid()` respecte les flags `optimize_*` ; `calculate_combinations()` dans `app.py` est synchronisé pour afficher le bon comptage.
- **Sélecteur de fenêtre pour le Final Backtest** : un `selectbox` permet de choisir la fenêtre WFO dont les paramètres seront utilisés (ou "Meilleure fenêtre (auto)"). `run_final_backtest_logic()` accepte un `window_id` optionnel.
- **Charts final backtest scindés en deux figures** : graphique prix de l'actif séparé du graphique returns % (portfolio vs Buy & Hold), avec annotations de performance finale sur chaque courbe.
- `ui/data_utils.py` : `_coerce_arrow_value` décode les `bytes` en UTF-8 ; `arrow_safe_df` force-convertit les colonnes `Value` des `Series` en `str` homogène pour éviter les erreurs d'inférence PyArrow sur types mixtes.
- Table des paramètres WFO (onglet résultats) : remplacée par un `st.dataframe` avec numérotation des fenêtres (colonne `Window`), plus lisible que la figure Plotly précédente.

## 2026-04-02

- **Correction grille de paramètres figés** : `np.arange` remplacé par une expansion par comptage dans `main.py` pour éviter le drift IEEE 754 quand `min == max`.
- **Cohérence Buy & Hold** : `init_cash` hardcodé remplacé par `order_fixed_cash` dans `strategy.py`.
- `app.py` : import `arrow_safe_df` centralisé depuis `ui/data_utils` (source unique) ; protection de `window_info_df`, `page_df` et `trades_df`.
- **Badges IS/OOS** sur le chart IS/OOS pour identifier la meilleure fenêtre IS et la meilleure fenêtre OOS.
- **UX load-params** : le bouton sidebar "Load Best Params" remplacé par un `selectbox` de fenêtre + bouton dédié.
- `ui/final_backtest_panel.py` : `load_best_params_into_inputs` accepte un `window_id` et expose `_get_params_for_window` ; protection de `param_records` et `pf.trades.stats()` contre les erreurs silencieuses.
- `ui/strategy_panel.py` : branches `else check=False` pour SAR/MACD désactivés afin d'éviter la restitution involontaire des clés par Streamlit lors du widget cleanup.

## 2026-03-30

- **Barre d'onglets fixe** : injection CSS pour maintenir la barre de tabs en haut de page lors du scroll (`feat: sticky tab bar`).
- **Suppression FutureWarning `fillna` booléen** : helper `_bool_fill` dans `strategy.py` pour éviter les avertissements pandas sur les colonnes booléennes.
- **Chart IS/OOS** : utilise désormais la métrique sélectionnée (`metric1`) au lieu du Sharpe ratio codé en dur.
- **Score PQS — facteur de confiance** : multiplication par `√(n_trades / n_ref)` pour pénaliser les fenêtres avec peu de trades ; `n_ref` configurable dans l'UI (défaut : 50).
- **Perf Phase B1 — bisection récursive** : remplacement du fallback séquentiel O(N) par une bisection récursive `_eval_chunk_bisect` dans `wfo.py`. Pour K combos défaillants dans un chunk de N, coût réduit à O(K·log N) au lieu de O(N) appels séquentiels.
- **Perf Phase A** :
  - Cache backtest borné (`_BoundedCache`, maxsize=5000, éviction FIFO) pour prévenir la croissance mémoire multi-Go sur les longs runs WFO.
  - `strategy.py` T2 : patch numpy ciblé sur les barres T2 uniquement (supprime ~800 Ko d'allocation par trial).
  - `_resolve_metric_value()` extrait en fonction de module dans `final_backtest_panel.py` (source unique, dédoublonnage).
  - Logs fallback chunking enrichis (paramètres du premier combo défaillant, plage, comptage NaN).

## 2026-03-29

- **Signal T2 — prix stop-buy** : prix d'entrée fixé à `High[T1] + 0.01 mintick` via `price=` dans `from_signals`.
- **`use_divergence_bb`** : filtre évaluable sur la barre T1 (setup), activable/désactivable, collapsé à valeur fixe quand T2 est désactivé.
- **`use_roc_filter`** : filtre RoC T1 désormais activable/désactivable par l'utilisateur.
- **Métrique PQS (Profit Quality Score)** : `(Return% / |MaxDD%|) × AvgP&L%` ajoutée à `metrics.py`, `portfolio_metrics`, `wfo.py`, `strategy.py`, `final_backtest_panel.py` et `app.py`. Affichée dans la barre de synthèse WFO, les lignes IS/OOS et les cartes du final backtest.
- **Sidebar Final Backtest** : nouvel expander consolidant Robust Tests (Level 1), plage de dates et bouton Run (déplacé hors de la section optimisation WFO).
- `main.py` : paramètres SAR/MACD collapsés à leurs valeurs fixes quand les sorties correspondantes sont désactivées (suppression de dimensions mortes dans la grille GP).

## 2026-03-15

- **Panneau stratégie central** (`ui/strategy_panel.py`) : expander 4 onglets (Signaux Entrée / Sorties / Filtres Croisements / MTF) déplacé de la sidebar vers l'interface centrale. `PARAMETER_HELP` est désormais la source unique dans `strategy_panel.py`, importée par `expert_panel.py`.
- `app.py` : suppression du dict `PARAMETER_HELP` et des expanders sidebar ; `get_current_config()` lit les clés widget depuis `st.session_state`.
- Nouveaux flags de config : `use_t2_signal`, `exit_cross_sar_sma_enabled`, `exit_retour_bb_enabled`, `exit_regline_enabled`, `exit_volat_down_enabled`, `macd_ma_type` + 9 paramètres stratégie fixes.
- `config.py` : `macd_ma_type: str = 'sma'` ajouté à `WFOSettings`.
- **Correction cast int MACD SMA Numba** : `fast_length`, `slow_length`, `signal_length` castés en `int` dans `macd_exit_signal_nb` (même classe de bug que `9af4f2c`, déclenché uniquement avec `use_sma=True`).

## 2026-02-26

- `.streamlit/config.toml` : `maxMessageSize=2048` ajouté (limite message Streamlit portée à 2 Go).
- **Répertoire d'upload persistant** (`~/.atdmf/uploads`) : les CSV uploadés ne sont plus stockés dans `/tmp` et survivent aux redémarrages du serveur ; configurable via variable d'environnement `WFOE_UPLOAD_DIR`.
- `app.py` : détection de chemin de fichier manquant au chargement d'une config (gestion des anciens chemins `/tmp` et fichiers déplacés).
- **Table paramètres Best Window** adaptée au thème sombre (en-tête bleu marine, lignes alternées sombres, texte clair, bordure accent bleue).
- **Correction cast float64 → int dans les kernels Numba** : paramètres VBT float64 castés en int pour les longueurs BB, SAR, SMA, MACD.
- **Correction signe `bb_ecart`** : signe inversé corrigé dans le calcul de l'écart Bollinger Bands.

## 2026-02-21

- **Correctif critique : cache indicateurs par fenêtre** (`c1e9bf3`) : le cache BB/SAR/SMA/MACD/linreg était clé uniquement sur les valeurs de paramètres, pas sur le DataFrame. Les tableaux IS étaient réutilisés silencieusement pour l'OOS (index/longueur différents) → désalignement → signaux NaN → 0 trades OOS dans toutes les fenêtres. Correction : `(len(df), df.index[0])` inclus dans chaque clé de cache.
- Extension du cache indicateurs à SAR, SMA exit, MACD exit et linreg exit.
- **Optimisations algorithmiques** :
  - `linreg_exit_nb` : O(n×length) → O(n) incrémental (accumulateurs somme_x/somme_x² glissants).
  - `rolling_median` : O(n×w×log w) → O(n×w) via `np.partition` (~7× plus rapide pour window=150).
  - `trade_stat` stats mis en cache une fois par bloc IS/OOS.
  - `iterrows` → `to_dict('records')` dans `adaptive_optimization` et `neural_search`.
  - `argpartition` O(N) pour le top-K dans `build_active_grid` (était O(N log N)).
  - `lru_cache` sur `_value_token` (~150K appels par run adaptatif).
- Ajout du **panneau comparaison de campagnes** (`campaign_panel.py`) avec mise en surbrillance et affichage côte-à-côte.

## 2026-02-18

- **Correction** : suppression de `per_column=True` invalide sur `CrossSARSMAExit` dans `strategy.py`.

## 2026-02-17

- **Alignement stratégie native avec Pine V6 (phases 1–5)** :
  - Phase 1 : MACD configurable SMA/EMA (`macd_ma_type`), défauts 9/19/6.
  - Phase 2 : Signal T0 complet (`nb_bars_under_bbw` + `bbandcross_barssince`).
  - Phase 3 : Cascade T1 avec filtre `DepassementRoC`.
  - Phase 4 : Signal T2 avec `divergence_BB` + dépassement du High.
  - Phase 5 : Sorties Cross SAR/SMA, pivot_low (retour BB), linreg, volat_down.
  - Mise à jour des défauts `DEFAULT_PARAM_GRID` pour correspondre aux valeurs Pine V6.

## 2026-02-15

- **Refactoring majeur de l'application** (`6ff48d0`) :
  - Phase 0 : extraction de `metrics.py` (`trade_stat`, `calc_avg_pl`, `safe_float`, `portfolio_metrics`) et `neural_search.py` (`NeuralSearchGuide`) ; `WFOSettings` converti en `@dataclass` avec `from_config()` ; recherche de voisinage O(n²) → `scipy.spatial.KDTree`.
  - Phase 1 : 93 tests unitaires ajoutés (`test_metrics`, `test_neural_search`, `test_wfo_unit`) ; migration `print()` → `logging` dans `wfo.py`, `adaptive_optimization.py`, `main.py`.
  - Phase 2 : `app.py` réduit de 10 907 à ~5 770 lignes (-47%) par extraction de `ui/pine_panel.py` (34 fonctions), `ui/expert_panel.py` (30 fonctions), `ui/export_panel.py` (export ZIP/PDF) et `ui/final_backtest_panel.py` (logique final backtest) ; pattern dependency injection.
  - Aucune modification des calculs d'indicateurs, de la logique stratégie ou des résultats de backtest.
  - Limite d'upload Streamlit portée à 2 Go.

## 2026-02-12

- Runtime transpilation Pine V3 enrichi pour les ordres `short`:
  - `strategy.order(...)` est désormais extrait dans `strategy_spec.v1` (action `order`) avec récupération robuste de `id`/`direction` (nommés ou positionnels),
  - support d'exécution `short` côté runtime transpilé via `short_entry_signal` / `short_exit_signal`,
  - `run_transpiled_pine_backtest` branche ces signaux sur `vectorbtpro.Portfolio.from_signals` (`short_entries`/`short_exits`),
  - contrat `capabilities` étendu avec `uses_strategy_order`,
  - tests ajoutés:
    - `apps/wfo_engine/tests/test_pine_v3_spec.py::test_spec_extracts_strategy_order_short_direction`,
    - `apps/wfo_engine/tests/test_pine_v3_runtime_mtf_transpile.py::test_generated_runtime_supports_short_order_signals`.
- Hardening supplémentaire sur la sémantique des ordres Pine:
  - `strategy_spec.v1` enrichi avec `logic.order_rules[*].args_positional` et `args_named`,
  - nouveau rapport runtime `pine_order_semantics.v1` dans `apps/wfo_engine/pine_v3/runtime_adapter.py`,
  - en mode strict, blocage explicite si `limit/stop/trail_*` sont détectés (`pine_enforce_order_semantics`),
  - support contrôlé du sizing d'entrée `qty_percent` / `qty` dans le runtime transpilé (avec `entry_size_mode_hint` et séries d'override),
  - blocage strict si mix `qty_percent` + `qty` sur un même run (modes mixtes non déterministes),
  - UI compatibilité enrichie: détection `uses_order_price_controls` (bloquant strict) et `uses_order_qty_controls` (partiel),
  - nouveau verrou UI `Enforcer sémantique des ordres (strict)`,
  - intégration end-to-end du rapport `pine_order_semantics_report`:
    - gate d'exécution Pine V3 (blocker `order_semantics_not_passed` en mode strict),
    - export/replay ZIP (`pine_order_semantics_report.json` + `results.json`),
    - manifeste d'artefacts V3 et contexte Expert (`pine_v3_artifacts`),
  - tests ajoutés/étendus: `apps/wfo_engine/tests/test_pine_v3_order_semantics.py`.
- V3 bêta renforcée sur les librairies Pine externes (lot suivant):
  - nouveau contrat runtime `pine_external_call_contract.v1` dans `apps/wfo_engine/pine_v3/runtime_adapter.py`,
  - vérification stricte des appels `Alias.fonction(...)` détectés dans `strategy_spec.v1` (alias résolu + fonction callable),
  - blocage explicite en mode strict si mapping/fonction externe invalide (`NotImplementedError` actionnable),
  - `GeneratedPineRuntimeAdapter` enrichi: le diagnostic `external_call_contract` est attaché aux signaux générés,
  - UI enrichie avec le verrou `pine_enforce_external_call_contract` (persisté dans config).
  - tests ajoutés: `apps/wfo_engine/tests/test_pine_v3_external_contract.py`.
- P2.3 implémenté: campagne de parité CI multi-scénarios avec rapport diff publiable:
  - nouveau module `apps/wfo_engine/pine_v3/parity_ci.py` (campagne déterministe + scénarios runtime MTF si `vectorbtpro` disponible),
  - nouveau runner CLI `apps/wfo_engine/tests/run_pine_parity_ci.py` (sortie JSON + code retour bloquant),
  - nouveau rapport standard `pine_parity_ci_report.v1` avec synthèse `passed/failed/blockers`,
  - tests ajoutés: `apps/wfo_engine/tests/test_pine_v3_parity_ci.py`,
  - documentation de clôture ajoutée: `docs/wfoe_v3/v3_beta_completion.md` (checklist et commandes de validation).
- P1.2 implémenté: support MTF `request.security` dans la transpilation runtime V3:
  - extraction `strategy_spec.v1` enrichie avec `logic.request_security_calls` (y compris affectations multi-lignes),
  - runtime transpilé: évaluation dédiée de `request.security(...)` (timeframe, gaps, lookahead) via les helpers MTF `request_security_series`,
  - prise en charge des affectations multi-cibles (`[a, b] = request.security(..., [expr1, expr2])`),
  - diagnostics dédiés `pine_request_security_diagnostics.v1` pour tracer les séries MTF générées.
- Intégration parser Pine optionnelle (recommandation d'architecture):
  - `strategy_spec.v1` supporte désormais un backend parser configurable (`auto`, `regex`, `pynescript`),
  - en mode `auto`, WFOE tente `pynescript` puis fallback regex déterministe si indisponible/échec,
  - traçabilité ajoutée dans `strategy_spec.transcription` (backend demandé/utilisé, fallback, statut `pynescript`),
  - UI enrichie avec le sélecteur `Pine Spec Parser Backend` et affichage des métriques parser dans le panneau spec.
- P2.1 amorcé: assistant LLM de migration Pine -> `strategy_spec.v1`:
  - nouveau module `apps/wfo_engine/pine_v3/llm_migration.py` (prompts déterministes, appel gateway, fallback sécurisé),
  - règle de sûreté: sortie LLM revalidée par `validate_strategy_spec_v1`; en cas d'échec, fallback vers spec déterministe baseline,
  - UI enrichie avec section "Assistant LLM migration Pine -> spec (P2.1)" et option d'application du draft validé,
  - traçabilité ajoutée (`pine_llm_generation_trace.v1`) avec hash prompts/sortie, provider/model et statut d'acceptation,
  - export/reload ZIP enrichi avec `pine_llm_migration_report.json`.
- P2.2 amorcé: catalogue local de stratégies Pine:
  - nouveau module `apps/wfo_engine/pine_v3/catalog.py` (registre local versionné par `strategy_id + source_sha1`),
  - snapshot source stocké dans `reports/pine_catalog/sources/` pour rechargement robuste,
  - UI Pine enrichie avec section "Catalogue stratégies Pine (P2.2)" (recherche + chargement d'une entrée),
  - auto-upsert du catalogue après génération d'un `strategy_spec.v1` valide,
  - traçabilité run enrichie avec le dernier `catalog_entry` et persistance dans `results.json`.
- Preuve de parité dédiée MTF ajoutée:
  - nouveau rapport déterministe `pine_mtf_parity_proof.v1` (`apps/wfo_engine/pine_v3/mtf_parity.py`),
  - UI enrichie avec section "Preuve de parité MTF request.security (P1.2)" + métriques/checks/blockers,
  - export/replay enrichis: `pine_request_security_diagnostics.json` et `pine_mtf_parity_proof_report.json`,
  - contexte/config enrichis avec statuts MTF (`pine_request_security_diagnostics_*`, `pine_mtf_parity_proof_*`).
- P1.4 (lot 4) implémenté: gate d'exécution Pine V3 (hardening bêta):
  - nouveau module déterministe `pine_execution_gate.v1` (bloqueur/no-go avant `Start WFO` en mode `pine_imported`),
  - règles de gate: `beta_ready=true` obligatoire, plus `parity_pass=true` si une référence Pine est fournie et que le verrou de parité est activé,
  - UI enrichie: option `pine_enforce_parity_gate`, statut gate, blockers détaillés, et verrouillage du bouton `Start WFO` si gate en échec,
  - export/replay enrichi avec `pine_execution_gate_report.json` et persistance dans `results.json`/session,
  - tests unitaires ajoutés: `apps/wfo_engine/tests/test_pine_v3_execution_gate.py`.
- P1.4 (lot 3) implémenté: parité avancée événements/trades:
  - extension du rapport `pine_parity_report.v1` avec `detail_checks` (entry/exit events + trades détaillés),
  - matching temporel configurable (tolérance secondes) et ratios minimum de correspondance,
  - UI enrichie avec seuils détaillés, compteurs ref/current et affichage de `detail_pass`,
  - support du format de référence v1 avec événements/trades (`reference_events`, `reference_trades`) dans le calcul de parité.
- P1.4 (lot 2) implémenté: référence Pine structurée + contrôles qualité:
  - format canonique de référence `pine_parity_reference.v1` (payload JSON normalisé),
  - validation déterministe (`pine_parity_reference_validation.v1`) avec erreurs/avertissements (métriques requises, cohérence counts, valeurs non finies),
  - UI parité adaptée: import/apply JSON avec normalisation/migration legacy vers v1 et affichage du statut de validation,
  - export/replay enrichi: `pine_parity_reference.v1.json` + `pine_parity_reference_validation.json` (fallback legacy conservé).
- P1.4 (lot 1) amorcé: validation de parité Pine/Python en UI + persistance artefacts:
  - nouvel écran "Validation de parité Pine/Python (P1.4)" avec import JSON de référence, seuils configurables et rapport `parity_pass`,
  - stockage session/payload de `pine_parity_report` et `pine_parity_reference_metrics`,
  - export/replay ZIP enrichi avec `pine_parity_report.json` et `pine_parity_reference_metrics.json`,
  - enrichissement du contexte Expert (`pine_v3_artifacts`) avec statut de parité.
- V3 beta readiness renforcé:
  - nouveau rapport déterministe `pine_beta_readiness.v1` calculé depuis précheck + compat + spec + codegen + module généré,
  - affichage UI (score, blockers runtime, statut go/no-go),
  - export/replay via `pine_beta_readiness_report.json` (ZIP + `results.json` + payload session).
- Compatibilité Pine stricte élargie:
  - détection additionnelle: `strategy.order`, entrées short, `pyramiding`, boucles (`for/while`) et `switch`,
  - conversion en blockers S0 en mode strict avec recommandations actionnables dédiées.
- Runtime Pine généré (non `strategy_test`) amélioré:
  - support des chunks de paramètres vectorisés côté `GeneratedPineRuntimeAdapter`
    (évaluation interne par combinaison, sans fallback moteur),
  - test de non-régression ajouté sur exécution vectorisée d'une stratégie générée.
- V3 Pine codegen/runtime: passage d'un adapter généré délégué (`strategy_test`) à un adapter généré autonome:
  - `apps/wfo_engine/pine_v3/spec.py` enrichi avec extraction de logique (`logic.assignments`, `logic.order_rules`) depuis le script Pine.
  - `apps/wfo_engine/pine_v3/runtime_adapter.py` enrichi avec `GeneratedPineRuntimeAdapter`:
    - transpilation déterministe d'expressions Pine (subset exécutable),
    - exécution des règles `strategy.entry/exit/close` pour produire des signaux,
    - backtest générique via `vectorbtpro.Portfolio.from_signals`.
  - `apps/wfo_engine/pine_v3/codegen.py` mis à jour pour générer un module Python autonome basé sur `GENERATED_STRATEGY_SPEC` (plus de délégation systématique vers `PineStrategyTestAdapter`).
  - Test runtime ajouté pour une stratégie non `strategy_test`:
    - `apps/wfo_engine/tests/test_pine_strategy_test_adapter.py::test_generated_pine_adapter_transpiles_non_strategy_test`.

- V3 block 1 implémenté: premier mode `pine_imported` exécutable pour la stratégie de test `docs/wfoe_v3/strategy_test.txt`.
- Ajout d'un runtime dédié `apps/wfo_engine/pine_v3/runtime_adapter.py`:
  - génération des signaux d'entrée/sortie BB + SMA (version test),
  - backtest scalaire et vectorisé compatible avec le moteur WFO existant.
- Résolution d'adapter étendue (`apps/wfo_engine/strategy_adapters.py`):
  - `resolve_strategy_adapter` retourne désormais un adapter Pine pour `strategy_test` (ID alias/source reconnue),
  - messages d'erreur explicites pour les autres stratégies Pine non encore supportées.
- Déblocage du lancement WFO en UI pour le cas supporté:
  - vérification de compatibilité stricte conservée,
  - validation runtime via `resolve_strategy_adapter` avant lancement.
- Final backtest UI rendu agnostique du mode stratégie:
  - remplacement des appels directs `run_backtest` par `strategy_adapter.run_backtest` (cas classique et "Rejouer une fenêtre").
- `get_param_grid` adapté en mode `pine_imported`:
  - SAR/MACD forcés à `False` pour éviter des combinaisons inutiles sur la stratégie test.
- Tests ajoutés: `apps/wfo_engine/tests/test_pine_strategy_test_adapter.py` (résolution adapter + exécution scalaire/vectorisée).
- Import Pine enrichi: la stratégie peut désormais être accompagnée de fichiers de librairie (`.txt/.pine`) via l'UI.
  - Persistance locale des librairies dans `reports/pine_imports/`.
  - Précheck Pine enrichi avec le contexte librairies (compte/noms fournis).
  - Export ZIP des librairies (`pine_libraries_manifest.json` + `pine_libraries/*`) et rechargement renforcé côté replay.
  - Traçabilité ajoutée dans `config`/`results.json` (`pine_library_files`, `pine_library_names`, `pine_library_paths`).
- P1.3 (phase assistée) amorcé: mapping explicite des imports Pine vers des modules Python locaux.
  - UI ajoutée: section "Mapping imports Pine -> modules Python" avec validation immédiate des cibles (`.py` local ou module importable).
  - Précheck enrichi avec `import_resolution` (résolu/non résolu par import), compteurs et mapping saisi utilisateur.
  - Compatibilité `strict` ajustée: un import externe devient non-bloquant si 100% des imports sont résolus (fichier librairie + mapping Python valide).
  - Export/replay enrichi: `pine_import_mapping` dans `results.json` et `import_mapping.json` dans le ZIP.
  - `strategy_spec.v1` enrichi avec `imports_detail` et `import_resolution` (si disponibles).
- Branchement runtime du mapping imports (phase assistée):
  - `resolve_strategy_adapter(...)` transmet désormais la config Pine complète au runtime adapter.
  - En mode `pine_imported`, le strict est appliqué côté resolver si `pine_compatibility_blocking=true` (même hors UI).
  - Le runtime `strategy_test` charge les modules Python mappés (alias Pine) et tente d'utiliser `BBT1.cross_bbw_low_signal` quand disponible, avec fallback sécurisé sur l'implémentation locale.
  - Tests étendus sur l'adapter (`strict blocking` + propagation `runtime_config`).
- Analyse des fonctions importées renforcée (précheck):
  - extraction des appels `Alias.fonction(...)` dans la stratégie Pine,
  - extraction des fonctions déclarées dans les fichiers de librairie Pine fournis,
  - comparaison automatique appelées/trouvées/manquantes par import (`import_resolution` enrichi),
  - intégration des compteurs de couverture dans le score de compatibilité strict.
- P1.1 amorcé: génération déterministe de module Python depuis `strategy_spec.v1`:
  - nouveau module `apps/wfo_engine/pine_v3/codegen.py` (`generate_strategy_module_from_spec`),
  - génération d'un `generated_strategy.py` persistant dans `reports/pine_imports/generated/`,
  - affichage du module généré dans l'UI (section "Generated Strategy Module (P1.1)"),
  - export/replay ZIP enrichi avec `generated_strategy.py`,
  - `resolve_strategy_adapter` charge dynamiquement le module généré si disponible.
- Ajout d'une brique MTF V3 dédiée `apps/wfo_engine/pine_v3/mtf.py` pour préparer le support `request.security`:
  - helpers `resample_ohlcv`, `realign_series_to_base`, `request_security_series`, `request_security_signal`.
  - alignement anti-lookahead via `realign_opening`/`realign_closing` (VectorBT Pro).
  - tests unitaires ajoutés: `apps/wfo_engine/tests/test_pine_v3_mtf.py`.

## 2026-02-11

- Lancement du lot 0 de cadrage pour WFOE V3 (support Pine Script v6) avec nouveaux livrables dans `docs/wfoe_v3/`:
- `README.md`: perimetre, references et objectifs du lot 0.
- `compatibilite_pine_v6.md`: matrice de compatibilite Pine v6 (niveaux S0/S1/S2/S3) et scope V3 initial.
- `architecture_cible_v3.md`: architecture cible, interface `StrategyAdapter`, pipeline d'artefacts et integration au moteur existant.
- `backlog_v3_p0_p1_p2.md`: backlog priorise avec criteres d'acceptation (P0/P1/P2) et estimation de charge.
- `risques_et_garde_fous.md`: registre de risques, mitigations et Definition of Done lot 0.
- Implementation P0.1 (mode strategie) dans WFOE:
- Ajout des champs `strategy_mode` et `strategy_id` dans la config et la persistance JSON.
- Ajout des controles UI `Strategy Mode`/`Strategy ID` dans `WFO Engine Settings`.
- Blocage explicite du lancement en mode `pine_imported` (non executable a ce stade) avec message utilisateur.
- Propagation de `strategy_mode`/`strategy_id` dans la tracabilite run (UI + exports).
- Validation defensive cote services (`run_service.py` et `main.py`) pour refuser les modes non supportes.
- Implementation P0.2 (import + pre-analyse Pine):
- Ajout d'un importeur `Import Pine Strategy (.txt/.pine)` dans la section `WFO Engine Settings` quand `Strategy Mode = pine_imported`.
- Sauvegarde locale du fichier importe dans `reports/pine_imports/` pour tracabilite et reprise de session.
- Ajout d'une pre-analyse syntaxique minimale:
  - verification `//@version`
  - verification presence `strategy(...)`
  - detection de features critiques (`request.security`, `request.security_lower_tf`, ordres `strategy.*`, `import`)
  - statut `valid/invalid` avec erreurs et avertissements actionnables.
- Exposition des metadonnees Pine dans la configuration sauvegardee (`pine_file_path`, `pine_precheck_status`, `pine_source_sha1`, etc.).
- Implementation P0.3 (rapport de compatibilite Pine):
- Ajout d'un mode de compatibilite Pine (`strict`, `assist`, `manual`) dans l'UI.
- Generation d'un rapport de compatibilite `pine_compatibility.v1` (niveaux S0/S1/S2/S3) base sur la pre-analyse.
- Calcul d'un score de compatibilite, liste des items bloquants et recommandations actionnables.
- Blocage explicite en mode `strict` si incompatibilites S0 detectees.
- Export du rapport dans les artefacts (`results.json`, `pine_compatibility_report.json`, `pine_precheck_report.json`) et rechargement depuis ZIP.
- Implementation P0.4 (`strategy_spec.v1` + validation schema):
- Ajout d'un module dedie `apps/wfo_engine/pine_v3/spec.py` pour:
  - generer un `strategy_spec.v1` a partir d'un script Pine (metadata source, imports, inputs, capabilities),
  - valider strictement le contrat JSON via `strategy_spec_validation.v1`.
- Integration UI: generation/validation automatique du spec apres pre-analyse Pine valide, avec panneau de lecture du spec et des erreurs/warnings de validation.
- Persistance du spec dans la configuration courante (`strategy_spec_schema_version`, `strategy_spec_valid`, `strategy_spec_sha256`).
- Export/reimport des artefacts spec dans les ZIP:
  - `strategy_spec.v1.json`
  - `strategy_spec_validation.json`
  - + miroir dans `results.json` pour exploitation ulterieure.
- Implementation P0.5 (`StrategyAdapter` + branchement moteur):
- Ajout du module `apps/wfo_engine/strategy_adapters.py` avec:
  - interface `StrategyAdapter`,
  - implementation `ATDMFAdapter` (mode natif),
  - resolver central `resolve_strategy_adapter(...)`.
- Refactor du moteur WFO et adaptatif pour supprimer les appels directs `run_backtest`:
  - `wfo.py` et `adaptive_optimization.py` utilisent désormais `strategy_adapter.run_backtest(...)`.
- Propagation de l'adapter depuis l'orchestration:
  - `services/run_service.py` résout l'adapter selon `strategy_mode/strategy_id` et l'injecte dans les moteurs.
  - `main.py` utilise aussi l'adapter (y compris pour le final backtest CLI).
- Résultat: le moteur d'optimisation est découplé de la logique ATDMF en dur, prêt pour les adapters Pine futurs.
- Correction du chemin vectorisé grid pour éviter les fallback intempestifs:
  - Normalisation robuste des paramètres array/scalaires dans `strategy.py` (chunks len=1/len>1).
  - Alignement des colonnes de signaux vectorisés (notamment `SMAExit` en `per_column=True`).
  - Gestion sûre des paramètres d'exécution (`order_sizing_mode`, `order_fixed_cash`, `fees_pct`) en mode vectorisé.
- Implémentation P0.6 (export artefacts V3 + rechargement renforcé):
  - Export explicite dans le ZIP des artefacts standards V3:
    - `strategy_source.pine.txt`
    - `compatibility_report.json`
    - `strategy_spec.v1.json`
    - `generation_trace.json`
  - Ajout d'un manifeste d'artefacts (`pine_artifacts_manifest.json`) et propagation dans `replay_manifest.json`.
  - Rechargement ZIP renforcé:
    - lecture des noms canoniques + legacy,
    - persistance locale de la source Pine rechargée dans `reports/pine_imports/`,
    - auto-validation du `strategy_spec` si le fichier de validation est absent,
    - injection des artefacts Pine dans le contexte Expert pour exploitation analytique.

## 2026-02-08

- Ajout d'une traçabilité d'exécution dans l'UI et les exports:
- Génération d'un `run_id` par lancement d'optimisation.
- Capture des empreintes SHA256 de la configuration et des résultats.
- Capture des métadonnées Git (branche, commit, remote, état dirty).
- Export d'un fichier `audit_trace.json` dans les ZIP de résultats.
- Affichage d'un panneau `Traçabilité du run` dans l'interface Streamlit.

- Renforcement de la lisibilité du moteur adaptatif:
- Ajout de docstrings et commentaires structurants dans `adaptive_optimization.py`.

## 2026-02-09

- Extension du package d'export pour exploitation statistique et rejeu:
- Sauvegarde de tous les trials par fenêtre (`optimization_trials`) côté moteur WFO/adaptatif.
- Export `all_trials.csv` + `trials/window_<id>.csv`.
- Export `window_info.csv` (dates IS/OOS, compteurs d'évaluations).
- Export `replay_manifest.json` (config + traçabilité + contexte de rejeu).
- Option UI `Package complet (rejeu + stats)` qui force `df.csv` en mode `full`.
- Chargement automatique des fichiers `all_trials.csv` et `window_info.csv` depuis un ZIP.
- Nouveau choix utilisateur du snapshot marché dans l'export:
- `manifest_only` (sans snapshot prix), `parquet_zstd` (compact), `csv_downsampled`, `csv_full`.
- Ajout d'un descripteur de source de données dans `replay_manifest.json` (path, taille, période, lignes chargées).
- Ajout de profils adaptatifs sélectionnables dans l'UI:
- `Smoke Test`, `Rapide`, `Équilibré`, `Robuste`, `Réactif`, `Conservateur`, `Exploratoire`, `Custom`.
- Application explicite du preset via bouton (pas d'auto-réécriture des champs).
- Panneau explicatif par profil (objectif, avantages, limites, spécificités, impact durée).
- Estimation de charge et durée (bougies, cycles, trials, ETA) selon période/timeframe configurées.
- Améliorations ergonomiques UI:
- Migration `use_container_width` vers `width` pour anticiper la dépréciation Streamlit.
- Clarification du mode adaptatif: `Optimization Method` affichée mais désactivée/explicitement non applicable.
- Carte de charge contextuelle en sidebar (`cycles/trials/ETA`) en mode adaptatif.
- Table `All Trials` filtrable et paginée (fenêtres, score min, taille de page).
- Application des profils adaptatifs passée en action explicite (bouton), sans auto-réécriture silencieuse.
- Ajout d'animations d'attente pendant l'optimisation:
- Carte de statut animée (progression, elapsed, ETA, cycle/fenêtre).
- Badge `RUNNING`, barre de progression shimmer et bouton `Stop WFO` avec pulse visuel.
- Nouveau volet `Adaptive Insights` dans les résultats:
- Graphiques de convergence des scores par cycle (best, médiane, IQR).
- Graphique de gap IS/OOS (return et sharpe) pour visualiser la généralisation.
- Graphique de compression de grille (combinaisons actives vs baseline + trials testés).
- Graphique d'évolution des poids relatifs par paramètre (top 8).
- Vue top valeurs historiques par paramètre (`top_values_by_parameter`).
- Ajout d'un guide d'interprétation directement sous chaque graphique adaptatif.
- Ajout du module MVP `Expert IA`:
- Nouveau package `expert/` (gateway LLM OpenAI-compatible, prompt builder, schema, analyzer, storage, service).
- Nouvel onglet UI `🤖 Expert IA` pour générer une interprétation IA des résultats d'optimisation.
- Contrat JSON de sortie `expert.v1` avec validation minimale et mode dégradé en cas de réponse partielle.
- Sauvegarde des interprétations dans `reports/expert/` + journal `expert_audit.jsonl`.
- Mise en forme de l'analyse Expert pour lecture humaine dans l'UI:
- Remplacement de l'affichage JSON brut par un rapport structuré (évaluation globale, constats, généralisation, diagnostics adaptatifs, actions, alertes).
- Export utilisateur du rapport en Markdown (`.md`) au lieu d'un JSON brut.
- Édition des prompts Expert dans l'UI avant exécution:
- Affichage du prompt système et du prompt utilisateur dans un panneau éditable.
- Bouton `Charger prompts auto` pour régénérer le prompt par défaut à partir du contexte courant.
- Lancement de l'analyse avec les prompts réellement modifiés par l'utilisateur.
- Prompt codé mis à jour: réponse exigée en français avec tolérance aux anglicismes métier.
## 2026-02-10

- Isolation complète de WFO Engine dans un dossier dédié:
- Déplacement des scripts coeur (`app.py`, `main.py`, `wfo.py`, `strategy.py`, `indicators.py`, `data_loading.py`, `config.py`, `adaptive_optimization.py`, `visualization.py`, `wfo_save.py`) vers `apps/wfo_engine/`.
- Déplacement des packages associés (`expert/`, `domain/`, `services/`, `ui/`) vers `apps/wfo_engine/`.
- Déplacement des tests WFO vers `apps/wfo_engine/tests/`.
- Ajout de `apps/wfo_engine/README.md` et `apps/wfo_engine/requirements.txt`.
- Mise à jour des chemins de CI et des instructions de validation (`CONTRIBUTING.md`, PR template, README).
- Ajout d'un launcher unique `scripts/run_wfoe.sh` pour lancer WFO Engine sans retenir les chemins.
- Nettoyage racine: suppression des fichiers non liés à l'exécution (`BACKEND_INFO.txt`, `BAYESIAN_CONFIG.txt`, `BAYESIAN_LIMITS.txt`, `PARALLEL_BACKEND_STATUS.txt`, `GEMINI.md`), suppression du doublon `requirements.txt` racine et purge des caches (`.ipynb_checkpoints/`, `__pycache__/`, `.pytest_cache/`).
- Harmonisation du nommage des exports ZIP de résultats: `Save Results to Disk` utilise désormais la même règle que `Save Config` (mêmes composantes période/timeframe/méthode/fenêtres/trials/régime), avec préfixe `results_wfo_...zip`.
- UX Templates Expert améliorée: boutons `Sauvegarder/Charger/Supprimer` désormais désactivés quand l'action n'est pas possible, messages d'aide explicites, et ajout d'un module `Importer/Exporter templates JSON` (modes `Fusionner` ou `Remplacer`).
- Correctif prise en compte des templates Expert:
- Résolution des alias de clés de prompts lors du chargement/import (`system_prompt/system/prompt_system`, `user_prompt/user/prompt/...`).
- Verrouillage automatique des prompts custom après chargement de template pour éviter l'écrasement silencieux.
- Affichage du `Template actif` et du `Template utilisé pour cette analyse` pour vérification explicite.
- Injection automatique du contexte WFO dans le prompt utilisateur template si absent (ou via placeholder `{{AUTO_WFO_CONTEXT}}`) pour éviter les diagnostics "insufficient_data" sur runs valides.
- Persistance templates Expert fiabilisée après réorganisation du repo:
- Chemin canonique des templates unifié vers `reports/expert/prompt_templates.json` (racine repo).
- Chargement rétrocompatible avec l'ancien chemin et auto-migration/fusion des templates existants.
- Affichage des rapports Expert sauvegardés dans l'UI:
- Ajout d'un panneau "Historique des rapports Expert sauvegardés" permettant de sélectionner/charger un fichier `reports/expert/expert_*.json` dans `expert_last_response`.
- Ajout d'un bouton de purge de l'affichage courant (`Vider rapport affiché`).
- Durcissement du chemin de persistance `ExpertStorage` pour écrire de manière canonique dans `reports/expert/` indépendamment du dossier de lancement.
- Correctif UX Templates Expert: suppression de l'exception Streamlit `cannot be modified after the widget ... is instantiated` via un mécanisme de verrouillage différé (`expert_lock_prompts_pending`) appliqué avant instanciation du checkbox.
- Export PDF ajouté:
- Nouveau bouton sidebar `📄 Générer rapport PDF` dans `Export Results`.
- Génération d'un rapport PDF multi-pages incluant synthèse WFO, comparaison IS/OOS, stabilité paramétrique et résumé/courbe du final backtest (quand disponible).
- Nouveau bouton `⬇️ Download Report (PDF)` et sauvegarde disque avec nommage harmonisé `report_wfo_<...>.pdf`.

- Réorganisation multi-apps:
- Déplacement de `dbad_app.py` vers `apps/dbad/app.py`.
- Déplacement de `analyze_exit_signals.py` vers `apps/exit_signals/analyze_exit_signals.py`.
- Déplacement de `exit_signal_app.py` vers `apps/exit_signals/exit_signal_app.py`.
- Ajout de `README.md` et `requirements.txt` dédiés dans `apps/dbad/` et `apps/exit_signals/`.

- Durcissement rapide de la gouvernance GitHub:
- Ajout d'un workflow CI minimal (`.github/workflows/ci.yml`) avec `pytest -q` et vérification de syntaxe (`py_compile`).
- Ajout d'un template de Pull Request (`.github/pull_request_template.md`).
- Ajout des politiques `CONTRIBUTING.md` et `SECURITY.md`.
- Ajout d'un fichier `LICENSE` propriétaire (all rights reserved).
- Renforcement de `.gitignore` pour ignorer caches Python/Jupyter, secrets Streamlit et artefacts locaux volumineux (`Data/`, `reports/`, `fichiers_configuration_WFO/`).

- Upgrade Expert IA Phase 1:
- Prompt Expert orienté stratégie: injection du contexte entrée/sortie, modules de sortie actifs et rôle des paramètres.
- Ajout d'alertes déterministes (sans LLM) affichées avant l'analyse: overfitting IS/OOS, instabilité paramètres, insuffisance de trials, runs peu concluants.
- Ajout d'une gestion de templates de prompts dans l'UI Expert (sauvegarder/charger/supprimer).
- Rapport Expert enrichi avec section d'alertes déterministes et section d'alignement stratégie.
- Couverture explicite du final backtest dans l'Expert:
- Transmission des métriques finales (return, sharpe, drawdown, win rate, trades) et comparaison buy&hold quand disponibles.
- Affichage du résumé final backtest transmis dans l'UI Expert.
- Prompt mis à jour pour exiger une section d'analyse `final_backtest_assessment`.
- Sélection guidée des modèles LLM dans l'UI Expert:
- Ajout d'un catalogue de modèles par provider (`openai`, `grok`, `gemini`) avec descriptions métier.
- Sélecteur `Modèle LLM` basé sur liste + option `Autre (saisie libre)`.
- Placeholder/hint de `Base URL` adapté automatiquement au provider choisi.
- Correctifs de compatibilité OpenAI GPT-5 pour Expert IA:
- Gestion des paramètres API selon modèle (`max_completion_tokens` pour chat GPT-5, température personnalisée ignorée quand non supportée).
- Ajout du format JSON explicite dans les requêtes OpenAI (`response_format` / `text.format`) pour limiter les sorties non JSON.
- Détection explicite des réponses tronquées/incomplètes (`finish_reason=length`, `incomplete_details.reason=max_output_tokens`) avec message d'action.
- Prompt Expert rendu plus concis pour réduire le risque de troncature; `Max tokens` UI relevé à 3000 par défaut.
- Data Configuration UX:
- Ajout d'un sélecteur de fichier CSV (`Browse CSV file from disk`) pour choisir un fichier depuis le disque via l'interface.
- Le fichier uploadé est persisté en copie locale temporaire et renseigne automatiquement `File Path`.
- Synchronisation automatique des dates Start/End lors d'un nouvel upload.
- Correctifs robustesse métriques (audit Python):
- Correction du calcul `avg_pl_per_trade` pour les portefeuilles vectorisés (évite le test ambigu sur Series).
- Alignement du même correctif dans le moteur adaptatif.
- Documentation interne renforcée:
- Ajout de docstrings sur les fonctions principales (`app.py`, `data_loading.py`, `wfo.py`, `expert/*`, `config.py`) pour améliorer lisibilité et maintenance.
- Correctif chemin vectorisé `grid` (suppression fallback lié aux colonnes dupliquées):
- Cause racine: alignement many-to-many Pandas sur colonnes MultiIndex non uniques dans `create_signal_generators`, provoquant une explosion de dimensions (`32 -> 131072`) lors des masques `exit`.
- Correction: normalisation systématique des colonnes d'indicateurs en `RangeIndex` uniques avant opérations booléennes et alignements.
- Correction de régression scalaire associée: gestion explicite du cas `Series` pour `sar_exit_signal` afin d'éviter `AttributeError: 'Series' object has no attribute 'columns'`.
