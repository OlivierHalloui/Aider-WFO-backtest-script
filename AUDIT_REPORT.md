# Rapport d'Audit — WFO Engine

**Date** : 2026-06-11 — **révision v2** après contre-vérification du code (sévérités R1/P6 revues à la baisse, erreurs factuelles R5/R8 corrigées, dépendances inter-fenêtres P1 documentées)  
**Périmètre** : ~14 000 LOC, 9 modules core, 21 fichiers de tests  
**Modèle d'analyse** : exploration statique du code source + contre-vérification ciblée des findings  

---

## Résumé Exécutif

L'application WFO Engine est **fonctionnellement correcte** dans ses mécaniques fondamentales (fenêtrage WFO, indicateurs techniques, logique T0/T1/T2). Trois domaines nécessitent une intervention :

| Domaine | Sévérité maximale | Nombre de findings |
|---------|-------------------|--------------------|
| Performance de calcul | 🔴 CRITIQUE | 12 |
| Robustesse applicative | 🔴 HAUTE | 14 |
| Pertinence métier WFO | 🔴 HAUTE | 5 |
| Couverture tests | 🟡 MOYENNE | 7 angles morts |

---

## 1. Performance de Calcul

### 1.1 Boucle fenêtres 100% série — P1 🔴 CRITIQUE

**Fichier** : `apps/wfo_engine/wfo.py:729–957`

La boucle principale de walk-forward itère séquentiellement sur toutes les fenêtres. `ThreadPoolExecutor` est importé (ligne 9) et `parallel_backend` est un setting présent dans `WFOSettings`, mais **aucun des deux n'est utilisé**.

**Indépendance conditionnelle (révision v2)** : les fenêtres ne sont indépendantes qu'en **régime d'optimisation standard**. En régime `nn_guided`, `nn_guide.update()` (`wfo.py:940–944`) entraîne le guide sur les résultats de chaque fenêtre pour construire la grille de la suivante ; en régime `prev_best_grid` (`wfo.py:783–793`), la grille de la fenêtre i+1 est dérivée des best params de la fenêtre i. **La parallélisation n'est applicable telle quelle qu'en régime standard** — les régimes guidés sont séquentiels par design.

**Impact estimé (non mesuré)** : borne haute théorique ~N× pour N fenêtres. Gain réel probablement inférieur : l'évaluation intra-fenêtre est déjà vectorisée par chunks (`_eval_chunk_vectorized`) et VBT/Numba peuvent déjà saturer les cœurs. **À benchmarker avant d'investir dans l'implémentation.**

```python
# Situation actuelle
for i in range(settings.n_windows):   # SÉRIE — aucune raison fonctionnelle
    optimization_results, eval_count = optimize_parameters(...)
    in_sample_portfolio = strategy_adapter.run_backtest(...)
    out_sample_portfolio = strategy_adapter.run_backtest(...)
```

**Note mode ancré** : en mode ancré, les fenêtres ont des IS croissants mais restent indépendantes après découpage — la parallélisation reste possible.

---

### 1.2 Thompson sampling non-vectorisé — P2 🔴 CRITIQUE

**Fichier** : `apps/wfo_engine/adaptive_optimization.py:155–172`

`score_candidate_thompson()` itère sur tous les paramètres pour chaque candidat en appelant `rng.normal()` P fois au lieu d'une fois.

```python
def score_candidate_thompson(self, candidate):
    total = 0.0
    for name in self.param_names:          # O(P) boucle par candidat
        ...
        sigma = np.sqrt(max(v, 1e-6) / (c + 1.0))
        total += float(self.rng.normal(m, sigma))   # appel unitaire
    return total / used
```

**Impact** : 3000 candidats × 20 params = 60 000 appels `rng.normal()` au lieu de 1 appel vectorisé.

---

### 1.3 SVI : `.iterrows()` + k backtests séquentiels — P3 🔴 CRITIQUE

**Fichier** : `apps/wfo_engine/wfo.py:518–533`

`get_svi_best_params()` itère les top_k candidats avec `.iterrows()` (anti-pattern pandas) et lance k backtests IS₂ en série.

```python
for _, row in candidates.iterrows():           # .iterrows() = slow
    params = row.drop(...).to_dict()           # dict rebuild par ligne
    score = strategy_adapter.run_backtest(...) # backtest SÉRIE
```

**Impact** : pour top_k=20, chaque fenêtre paie 20 backtests séquentiels supplémentaires.

---

### 1.4 KDTree reconstruit à chaque fenêtre — P4 🟡 HAUTE

**Fichier** : `apps/wfo_engine/wfo.py:486–488`

Pour la sélection SNV, un `KDTree` est reconstruit intégralement à chaque fenêtre, même si la grille de paramètres n'a pas changé. La normalisation du space param est recalculée en même temps.

```python
tree = KDTree(norm_vals)                          # O(N log N) à chaque fenêtre
_, indices = tree.query(norm_vals, k=k)
smoothed = np.nanmean(scores[indices], axis=1)
```

**Impact** : 1000 combos × 10 fenêtres = 10 000 requêtes KDTree qui pourraient être 1000.

**Limite d'applicabilité (révision v2)** : le cache n'a de valeur qu'en **grid search** avec grille identique entre fenêtres. En Bayesian/Optuna et en régimes guidés (`nn_guided`, `prev_best_grid`), les combos évalués diffèrent par fenêtre — cache inopérant.

---

### 1.5 Fenêtres intra-stage non parallélisées — P5 🟡 HAUTE

**Fichier** : `apps/wfo_engine/stagewise_optimizer.py:825–947`

Même problème que P1, reproduit dans `run_stagewise_campaign()`. Chaque stage appelle `walk_forward_optimization()` en série. Les 11 stages sont séquentiels par design (dépendance entre stages), mais les fenêtres **à l'intérieur** de chaque stage pourraient être parallèles.

---

### 1.6 Sampling NN sans déduplification — P6 🟢 BASSE (rétrogradé v2)

**Fichier** : `apps/wfo_engine/neural_search.py:189–196`

`build_guided_grid()` génère des candidats aléatoires sans garantie d'unicité.

```python
for _ in range(sample_count):
    candidate = {}
    for key, values in base_param_grid.items():
        idx = int(self.rng.integers(0, len(values)))
        candidate[key] = values[idx]
    sampled_rows.append(candidate)    # doublons possibles
```

**Impact (révisé à la baisse v2)** : les évaluations passent par le backtest cache keyé `(data_signature, params)` (`wfo.py:185–193`) — un doublon est un **cache hit quasi gratuit**, pas un backtest gaspillé. Pour 3000 samples d'une grille 1000 combos, ~68% de doublons, mais leur coût réel ≈ lookups dict superflus. Fix utile uniquement si profiling montre un coût mesurable.

---

### 1.7 Autres bottlenecks (priorité MOYENNE)

| # | Fichier | Lignes | Problème |
|---|---------|--------|----------|
| P7 | `stagewise_optimizer.py` | 367–392 | `mode()` O(N log N) par param, DataFrame reconstruit à chaque consensus |
| P8 | `neural_search.py` | 106–124 | Full-batch SGD sans mini-batch ni early-stop (60 epochs × full matrix) |
| P9 | `adaptive_optimization.py` | 155–159 | Buffer `records` sans `deque` — slice `[-max:]` crée copie complète à chaque overflow |
| P10 | `wfo.py` | 619–650 | `_BoundedCache` FIFO au lieu de LRU — éviction des anciens params haute qualité |
| P11 | `wfo.py` | 928–931 | `to_dict('records')` sur DataFrame complet à chaque fenêtre dans la boucle |
| P12 | `neural_search.py` | 62–67 | `_encode_rows()` : double boucle O(M×P) avec dict lookup par cellule |

---

## 2. Robustesse Applicative

### 2.1 Race condition run-resolution — R1 🟡 MOYENNE (rétrogradé v2)

**Fichier** : `apps/wfo_engine/app.py:3550–3593`

Le thread daemon WFO écrit `job_state['results']` (ligne 3746) pendant que le main Streamlit thread lit et supprime la même dict (lignes 3569, 3590).

**Analyse de sévérité (révision v2)** : le scénario « clobber post-harvest » initialement décrit est en pratique impossible — `status` est écrit **une seule fois** par le worker avant la mort du thread, et `completed` vs `stopped`/`error` sont mutuellement exclusifs : la branche restore ne peut pas s'exécuter après un harvest réussi avec un status stable. Le double-harvest écrirait les **mêmes valeurs** (idempotent). Streamlit sérialise par ailleurs largement les reruns d'une même session.

**Risque résiduel** : fenêtre étroite entre lecture du status et pop des clés si deux script-runs se chevauchent (interruption Streamlit mid-block). Un flag `harvested` reste une assurance bon marché, mais ce n'est pas un risque actif bloquant.

```python
# Main thread (app.py:3554–3592)
if not wfo_thread.is_alive():          # condition stable après mort du thread
    status = wfo_job_state.get('status')   # écrit une fois, avant mort du thread
    if status == 'completed':
        st.session_state['wfo_results'] = wfo_job_state['results']  # harvest
    else:
        _restore_state_snapshot(...)   # exclusif avec 'completed' — pas de clobber
```

---

### 2.2 Durabilité checkpoint cassée — R3 🔴 HAUTE

**Fichier** : `apps/wfo_engine/services/run_service.py:30–55`

`_checkpoint_job_state()` et `_write_window_result()` catchent toutes les exceptions en `logger.debug` — un echec d'écriture disque est **indistinguable d'un succès** côté UI.

```python
def _checkpoint_job_state(job_state, run_dir):
    try:
        run_dir.mkdir(parents=True, exist_ok=True)   # dans le try-block !
        ...
        os.replace(tmp, run_dir / "checkpoint.json")
    except Exception:
        logger.debug("Checkpoint write failed (non-fatal)", exc_info=True)  # SILENCIEUX
```

Si `run_dir` n'existe pas ou si les permissions manquent, **Chantier 3 (durabilité) est non-opérationnel sans aucun signal**.

---

### 2.3 Traceback perdu dans error log — R4 🔴 HAUTE

**Fichier** : `apps/wfo_engine/services/run_service.py:260–263`

```python
except Exception as e:
    logger.error("Run failed: %s", e)    # Pas de exc_info=True
```

Seul `str(e)` est conservé. Pour des exceptions imbriquées (ex. `ImportError` depuis `vectorbtpro`), la trace complète est perdue. Le diagnostic post-mortem devient impossible.

---

### 2.4 Bare `except Exception` dans hot paths — R5 🟡 HAUTE

**Fichiers** : `wfo.py:81–93`, `metrics.py:12–27`

```python
# wfo.py:81–93
except Exception as e:
    logger.debug("Single-combo eval failed: %s", e2)
    return [float('nan')]    # masque MemoryError (OOM)
```

**Correction factuelle (v2)** : `KeyboardInterrupt` et `SystemExit` héritent de `BaseException`, pas `Exception` — `except Exception` les laisse **déjà** se propager. Le seul masquage réel est `MemoryError`.

**Nuance design** : le catch large + NaN est défendable pour un optimiseur évaluant des milliers de combos arbitraires — un combo dégénéré (`KeyError`, `IndexError`, exceptions pandas/VBT) ne doit pas tuer le run entier. Le fix correct est de **re-lever `MemoryError` explicitement**, pas de restreindre à une liste blanche d'exceptions (qui transformerait chaque type d'erreur imprévu en crash de run complet). Les NaN silencieux restent un point d'attention : un taux de NaN élevé devrait être loggé en warning agrégé.

---

### 2.5 `session.py` orphelin — R6 🟡 MOYENNE

**Fichier** : `apps/wfo_engine/services/session.py` (71 LOC)

Module créé pour Chantier 1.1 (facade typée `WFOSessionState`) mais **jamais importé ni utilisé** dans `app.py`. Les 614 accès `st.session_state.get()` non typés et non validés persistent dans `app.py`. Le Chantier 2 (fragment progress bar) suppose ce module wired.

---

### 2.6 NaN/Infinity = JSON non-spec — R7 🟡 MOYENNE

**Fichiers** : `domain/serialization.py:70–82`, `ui/export_panel.py:275`

Les `float('nan')` et `float('inf')` Python produisent `NaN` et `Infinity` dans le JSON — **invalides selon RFC 7159**. Les parsers `jq`, les APIs REST, et les navigateurs rejettent ce JSON.

**Correction factuelle (v2)** : `default=` n'est appelé que pour les types **non-sérialisables** — `float('nan')` est sérialisable nativement et produit `NaN` dans **les deux** paths. Le `default=str` du path stagewise (ligne 98) ne corrige donc **pas** NaN/Inf : les deux exports émettent du JSON invalide. La divergence `default=str` ne concerne que les autres types (datetime, numpy non gérés). Fix : sanitizer NaN/Inf→`null` + `json.dumps(..., allow_nan=False)` comme garde-fou dans les deux paths.

---

### 2.7 Thread stagewise sans cleanup — R9 🟡 MOYENNE

**Fichier** : `apps/wfo_engine/ui/stagewise_panel.py:690–698`

Après crash du thread stagewise, `_sw_thread`, `_sw_run_state`, `_sw_stop_event` persistent en session. Le run suivant hérite de l'état corrompu. Aucun équivalent du cleanup WFO présent dans `app.py:3590–3591`.

---

### 2.8 Autres findings robustesse (priorité MOYENNE)

| # | Fichier | Lignes | Problème |
|---|---------|--------|----------|
| R8 | `stagewise_panel.py` vs `export_panel.py` | 98 vs 275 | Deux paths divergents pour types non-JSON ; NaN/Inf invalides dans **les deux** (`default=str` ne corrige pas NaN) |
| R10 | `data_loading.py` | 166–176 | `dtype=float64` sur cellule `"N/A"` → `ValueError` au lieu de NaN + fallback |
| R11 | `data_loading.py` | 193–201 | Binance fetch sans try/except ; `['BTCUSDT']` KeyError si fetch vide ; geo-block documenté mais non géré |
| R12 | `stagewise_optimizer.py` | 881–897 | Stage failure → `continue` sans option `halt_on_error` ; campagne continue sur params brisés |
| R13 | `neural_search.py` | 131–134 | `x_std ≈ 0` → division silencieuse → NaN en prédiction → guidance désactivée sans log |
| R14 | `metrics.py` | 12–27 | `safe_float()` : bare `except Exception` devrait capturer `(ValueError, TypeError)` uniquement |

---

### 2.9 Chantier 3.3 non implémenté

`run_service.py` écrit des checkpoints (3.1) et des résultats par fenêtre (3.2), mais **la logique de détection et de reprise d'un run interrompu (3.3) n'existe pas**. Les checkpoints sont écrits dans le vide : sans mécanisme de relecture au démarrage, la durabilité n'apporte aucune valeur opérationnelle.

---

## 3. Pertinence Fonctionnelle Métier

### 3.1 SVI IS₂ circulaire — M1 🔴 HAUTE

**Fichier** : `apps/wfo_engine/wfo.py:497–538`

La sélection SVI est présentée comme une validation sur holdout IS₂, mais IS₂ est une tranche chronologique de l'IS sur lequel l'optimisation a déjà tourné. La procédure est :

1. Optimiser tous les combos sur IS complet → scores IS
2. Sélectionner top_k par score IS
3. Réévaluer top_k sur IS₂ ⊂ IS → **données déjà vues**
4. Retourner le meilleur IS₂

**Conséquence (nuancée v2)** : IS₂ n'est **pas un holdout** — la terminologie est trompeuse et doit être corrigée. Le re-ranking sur la fin de l'IS a néanmoins une valeur réelle : il pénalise les combos performants uniquement en début de période et favorise la stabilité récente. Il ne fournit simplement **aucune garantie hors-échantillon**. À documenter comme « re-ranking sur IS récent » ; un vrai holdout hors-IS reste une amélioration possible (non bloquante).

---

### 3.2 Slippage absent — M2 🟡 HAUTE

**Fichier** : `apps/wfo_engine/strategy.py:951–957`

Seul un fee fixe en % est modélisé. L'entrée T2 stop-buy utilise `MINTICK = 0.01` comme approximation du prix d'exécution, mais ce n'est pas un modèle de slippage. Pour des backtests crypto sur données OHLCV, l'absence de modèle de slippage surestime systématiquement la performance (bid-ask spread, market impact, partial fills non modélisés).

---

### 3.3 Robust Set : mélange de régimes — M3 🟡 MOYENNE

**Fichier** : `apps/wfo_engine/ui/final_backtest_panel.py:310–468`

La méthode `robust_set` agrège les paramètres de toutes les fenêtres avec un vote pondéré par score OOS. Deux fenêtres de régimes opposés (bull market + crash) contribuent avec le même poids si leurs scores OOS sont identiques. Le paramètre résultant peut être médiocre pour les deux régimes. Aucune détection de régime ou pondération par similarité de contexte n'est implémentée.

---

### 3.4 WFO ancré : fenêtres early et données "futures" — M4 🟡 MOYENNE

**Fichier** : `apps/wfo_engine/wfo.py:744–754`

En mode ancré (`anchored=True`), la fenêtre 1 utilise toutes les données disponibles comme IS. Les indicateurs techniques calculés sur la fenêtre 1 "voient" donc des données chronologiquement plus récentes que leur point d'optimisation. C'est un choix de design acceptable pour simuler un "compte croissant", mais **non documenté** dans l'UI — l'utilisateur peut confondre avec un bias lookahead.

---

### 3.5 Points fonctionnellement corrects

Les éléments suivants ont été vérifiés et sont conformes aux bonnes pratiques :

| Composant | Fichier | Verdict |
|-----------|---------|---------|
| Ségrégation IS/OOS fenêtres | `wfo.py:738–758` | ✅ Correct — pas de leakage |
| SNV (KDTree voisinage) | `wfo.py:465–495` | ✅ Méthodologie sound, edge cases gérés |
| Level 2 best_oos / best_is_oos | `final_backtest_panel.py:71–190` | ✅ Sélection cross-window correcte |
| Indicateurs techniques | `indicators.py` | ✅ Numba-optimisés, NaN-safe |
| Logique T0/T1/T2 (entrée/sortie) | `strategy.py:754–903` | ✅ Sémantiques stop-buy conformes Pine V6 |
| Frais de transaction | `strategy.py:956–957` | ✅ % → décimal correct, appliqué via VBT |
| `WFOSettings.from_config()` | `config.py:102–110` | ✅ Forward/backward compatible |
| Thread-safety du cache backtest | `wfo.py:625–652` | ✅ `threading.Lock` correct |
| Calcul PQS | `metrics.py:147–182` | ✅ Formule correcte, division par zéro gérée |

---

## 4. Couverture Tests

| Couche | Fichiers de tests | État CI |
|--------|-------------------|---------|
| Pine V3 (spec, codegen, parity) | 13 fichiers | ✅ Tourne en CI |
| Core WFO engine (wfo, adaptive, stagewise) | 3 fichiers, VBT-gated | ⚠️ Skip total en CI |
| Services layer (run_service, error_log, export_utils, serialization, session) | **0 test** | ❌ Angle mort total |
| Data loading (edge cases CSV, Binance) | 0 test | ❌ Non testé |
| Session state lifecycle + race conditions | 0 test | ❌ Non testé |
| Durabilité / checkpoint roundtrip | 0 test | ❌ Fonctionnalité non validée |
| Export/import ZIP roundtrip | 0 test | ❌ Non testé |

La suite CI ne valide que Pine V3 + syntax check. **Aucune régression engine n'est détectée en CI.**

---

## 5. Fichiers par Niveau de Risque

| Fichier | LOC | Risque | Raison |
|---------|-----|--------|--------|
| `apps/wfo_engine/app.py` | 6219 | 🟡 HAUTE | Monolithe 6219 LOC, 614 session accesses non typés, race théorique (R1 rétrogradée) |
| `apps/wfo_engine/services/run_service.py` | 275 | 🔴 CRITIQUE | Durabilité cassée, traceback perdu |
| `apps/wfo_engine/wfo.py` | 1057 | 🔴 CRITIQUE | Perf série, SVI circulaire, bare except |
| `apps/wfo_engine/adaptive_optimization.py` | 599 | 🟡 HAUTE | Thompson sampling, records buffer |
| `apps/wfo_engine/stagewise_optimizer.py` | 1096 | 🟡 HAUTE | Perf série intra-stage, stage failure handling |
| `apps/wfo_engine/ui/stagewise_panel.py` | 721 | 🟡 HAUTE | Thread sans cleanup session |
| `apps/wfo_engine/neural_search.py` | 313 | 🟡 MOYENNE | x_std zero-div (R13) ; doublons sampling mitigés par backtest cache |
| `apps/wfo_engine/domain/serialization.py` | 99 | 🟡 MOYENNE | NaN/Inf JSON non-spec |
| `apps/wfo_engine/data_loading.py` | 225 | 🟡 MOYENNE | CSV coercion, Binance sans fallback |
| `apps/wfo_engine/services/session.py` | 71 | 🟡 MOYENNE | Orphelin (Chantier 1.1 incomplet) |
