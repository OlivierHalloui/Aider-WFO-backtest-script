# Plan d'Implémentation — WFO Engine Audit

**Basé sur** : `AUDIT_REPORT.md` (2026-06-11, révision v2)  
**Branche cible** : issues créées depuis `main`  
**Convention commits** : `fix(scope)`, `perf(scope)`, `feat(scope)`, `refactor(scope)`

**Révision v2** : tests engine avancés AVANT les refactors perf (la CI ne détecte aucune régression engine — contradiction du plan v1 avec le propre finding de l'audit). Benchmark go/no-go ajouté avant IMPL-P1. Snippet IMPL-P1 corrigé (`parallel_backend` est un string, pas un entier). IMPL-R5 inversé (re-raise `MemoryError` au lieu de liste blanche dangereuse). IMPL-P6 rétrogradé (mitigé par backtest cache).

---

## Vue d'ensemble

```
Phase 1 (S1–S2) : Risques actifs bloquants
  ├── Débloquer durabilité (exceptions swallowed)        ← priorité 1
  ├── Uniformiser sérialisation JSON (sanitizer + allow_nan=False)
  ├── Exception handling hot paths (re-raise MemoryError)
  └── Flag harvest run-resolution (assurance — R1 rétrogradée MOYENNE)

Phase 2 (S3) : Filet de sécurité AVANT refactors perf
  ├── Tests services layer (run_service, serialization, data_loading)
  ├── Smoke tests engine (stub adapter, sans VBT si possible)
  └── Benchmark parallélisation fenêtres → go/no-go IMPL-P1

Phase 3 (S4–S5) : Performance critique
  ├── Paralléliser fenêtres WFO (régime standard UNIQUEMENT, si benchmark concluant)
  ├── Vectoriser Thompson sampling
  ├── Fix SVI iterrows + batching IS₂
  └── Documenter/requalifier SVI méthodologie

Phase 4 (S6) : Refactoring & robustesse
  ├── Wirer session.py (scope réduit : clés run-resolution d'abord)
  ├── Cleanup thread stagewise
  └── Robustifier data_loading.py

Phase 5 (S7+) : Durabilité & optimisations longues
  ├── Chantier 3.3 (v1 : détection orphaned runs ; v2 : reprise)
  ├── Cache KDTree SNV (grid search uniquement)
  ├── Dédup neural search sampling (priorité basse — si profiling justifie)
  └── Modèle de slippage
```

---

## Phase 1 — Risques Actifs (Semaines 1–2)

### IMPL-R3 : Débloquer durabilité checkpoint

**Fichier** : `apps/wfo_engine/services/run_service.py:22–55`  
**Référence audit** : R3, R4  
**Commit** : `fix(run_service): propager erreurs checkpoint, ajouter exc_info`

**Problème** : Toutes les exceptions dans `_checkpoint_job_state()` et `_write_window_result()` sont swallowées en `logger.debug`. Chantier 3 est silencieusement non-opérationnel si le répertoire n'est pas accessible. **Finding vérifié exact — priorité 1 de la phase.**

**Implémentation** :

```python
def _checkpoint_job_state(job_state: dict, run_dir: pathlib.Path) -> None:
    # Précondition hors try-block principal
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("Checkpoint dir inaccessible %s: %s", run_dir, exc, exc_info=True)
        if job_state is not None:
            job_state.setdefault("warnings", []).append(
                f"Checkpoint write failed: {exc}"
            )
        return

    try:
        snap = {k: job_state.get(k) for k in _CHECKPOINT_KEYS}
        tmp = run_dir / "checkpoint.json.tmp"
        tmp.write_text(json.dumps(snap, default=str), encoding="utf-8")
        os.replace(tmp, run_dir / "checkpoint.json")
    except Exception as exc:
        logger.error("Checkpoint write failed: %s", exc, exc_info=True)   # ← exc_info
        if job_state is not None:
            job_state.setdefault("warnings", []).append(
                f"Checkpoint write failed: {exc}"
            )

def _write_window_result(window: int, payload: dict, run_dir: pathlib.Path) -> None:
    try:
        win_dir = run_dir / "windows"
        win_dir.mkdir(parents=True, exist_ok=True)
        ...
        os.replace(tmp, win_dir / f"window_{window:04d}.json")
    except Exception as exc:
        logger.error("Window %d result write failed: %s", window, exc, exc_info=True)
```

**Pour R4** (`run_service.py:263`) :
```python
except Exception as e:
    if job_state is not None:
        job_state["error"] = str(e)
    logger.error("Run failed: %s", e, exc_info=True)   # ← ajouter exc_info=True
```

Les `warnings` accumulés dans `job_state` doivent remonter dans l'UI (notice après run).

---

### IMPL-R7 : Uniformiser sérialisation JSON

**Fichiers** : `domain/serialization.py`, `ui/export_panel.py:275`, `ui/stagewise_panel.py:98–100`  
**Référence audit** : R7, R8 (corrigés v2)  
**Commit** : `fix(serialization): NaN/Inf JSON-safe, allow_nan=False, paths export unifiés`

**Problème (corrigé v2)** : `float('nan')`/`float('inf')` produisent `NaN`/`Infinity` dans le JSON (invalide RFC 7159) dans **les deux paths d'export**. `default=str` côté stagewise ne corrige PAS NaN/Inf — `default=` n'est appelé que pour les types non-sérialisables, et `float('nan')` est sérialisable nativement.

**Implémentation dans `domain/serialization.py`** :

```python
def sanitize_for_json(value):
    """Convertit valeurs Python/numpy en types JSON-valides (RFC 7159)."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None    # None = null JSON, valide RFC 7159
        return value
    if isinstance(value, np.floating):
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(v) for v in value]
    return value
```

**Dans les deux paths d'export** — sanitizer + garde-fou :

```python
from domain.serialization import sanitize_for_json

# allow_nan=False : tout NaN résiduel non couvert par le sanitizer lève
# ValueError immédiatement au lieu de produire du JSON invalide silencieux
zf.writestr(
    "results.json",
    json.dumps(sanitize_for_json(payload), indent=2, allow_nan=False, default=str),
)
```

Appliquer à `export_panel.py:275` ET `stagewise_panel.py:98–100` (même pattern, même garde-fou).

**Note migration** : NaN→null est lossy (impossible de distinguer NaN d'une valeur absente au réimport). Acceptable pour l'export ; vérifier que l'import ZIP tolère `null` sur les champs métriques.

---

### IMPL-R5 : Exception handling hot paths

**Fichiers** : `wfo.py:81–93`, `metrics.py:12–27`  
**Référence audit** : R5, R14 (corrigés v2)  
**Commit** : `fix(wfo,metrics): re-raise MemoryError, safe_float ciblé`

**Correction d'analyse (v2)** : `KeyboardInterrupt`/`SystemExit` héritent de `BaseException` — `except Exception` les laisse **déjà** passer. Seul `MemoryError` est réellement masqué.

**Approche v1 abandonnée** : restreindre à `(ValueError, TypeError, RuntimeError, ArithmeticError)` serait **dangereux** — `KeyError`, `IndexError`, `AttributeError` et les exceptions pandas/VBT custom sont fréquentes sur combos dégénérés ; un seul combo cassé tuerait le run entier au lieu de produire un NaN. Le catch large + NaN est le design correct pour un optimiseur évaluant des milliers de combos arbitraires.

**wfo.py:81–93** — re-lever uniquement le non-récupérable :

```python
# Avant
except Exception as e2:
    logger.debug("Single-combo eval failed: %s — params: %s", e2, combo_params)
    return [float('nan')]

# Après
except MemoryError:
    raise                      # OOM : ne jamais masquer
except Exception as e2:
    logger.debug("Single-combo eval failed: %s — params: %s", e2, combo_params)
    return [float('nan')]
```

**Amélioration complémentaire** : compter les NaN par fenêtre et logger un `warning` agrégé si le taux dépasse un seuil (ex. 10%) — les NaN silencieux massifs signalent un problème de config, pas des combos dégénérés isolés.

**metrics.py:12–27** — `safe_float()` convertit des scalaires : le narrowing est approprié ici (les types d'échec sont connus) :

```python
except (ValueError, TypeError, OverflowError):
    return float(default)
```

---

### IMPL-R1 : Flag harvest run-resolution (assurance)

**Fichier** : `apps/wfo_engine/app.py:3550–3593`  
**Référence audit** : R1, R2 (rétrogradés MOYENNE en v2)  
**Commit** : `fix(app): flag harvest idempotent sur run-resolution`

**Sévérité revue (v2)** : le scénario « clobber post-harvest » est en pratique impossible — `status` est écrit une fois avant la mort du thread, et `completed` vs `stopped`/`error` sont mutuellement exclusifs. Le double-harvest serait idempotent. Risque résiduel : chevauchement de script-runs entre lecture du status et pop des clés. Le flag reste une assurance bon marché (0.5j) — implémenter en fin de phase, pas en premier.

**Implémentation** :

```python
# À la création du job_state (app.py:~3705)
job_state = {
    "status": "running",
    "results": None,
    "harvested": False,    # ← nouveau flag
    ...
}

# Seam run-resolution (app.py:3554–3592)
if (wfo_thread is not None
        and not wfo_thread.is_alive()
        and wfo_job_state is not None
        and not wfo_job_state.get('harvested', False)):   # ← guard

    wfo_job_state['harvested'] = True    # ← set AVANT lecture résultats
    status = wfo_job_state.get('status')
    # ... harvest / restore / cleanup inchangés
```

**Tests** : simuler double passage avec `status='completed'` → vérifier harvest unique.

---

## Phase 2 — Filet de Sécurité (Semaine 3)

**Justification (v2)** : l'audit constate « aucune régression engine n'est détectée en CI » (tests core VBT-gated, skippés). Refactorer `wfo.py` (Phase 3) sans filet contredit ce finding. Tests d'abord.

### IMPL-TESTS : Tests services layer + smoke engine

**Nouveaux fichiers** : `apps/wfo_engine/tests/test_run_service.py`, `test_serialization.py`, `test_data_loading.py`, `test_wfo_smoke.py`  
**Commit** : `test(services,wfo): couverture run_service, serialization, data_loading, smoke engine`

**test_run_service.py** — cas prioritaires :
```python
def test_checkpoint_write_fail_logs_error(tmp_path, caplog):
    """Exception IO → logger.error avec exc_info, warning dans job_state."""
    run_dir = tmp_path / "nonexistent" / "run"
    run_dir.parent.chmod(0o000)   # rendre inaccessible
    job_state = {}
    with caplog.at_level(logging.ERROR):
        _checkpoint_job_state(job_state, run_dir)
    assert "warnings" in job_state
    assert any(r.levelname == "ERROR" for r in caplog.records)

def test_checkpoint_roundtrip(tmp_path):
    """checkpoint.json relisible et conforme aux _CHECKPOINT_KEYS."""
    ...
```

**test_serialization.py** :
```python
def test_nan_serializes_to_null():
    assert sanitize_for_json(float('nan')) is None

def test_inf_serializes_to_null():
    assert sanitize_for_json(float('inf')) is None

def test_nested_nan_in_dict():
    d = {"a": float('nan'), "b": {"c": float('inf')}}
    result = sanitize_for_json(d)
    assert result == {"a": None, "b": {"c": None}}
    json.dumps(result, allow_nan=False)   # ne doit pas lever
```

**test_data_loading.py** :
```python
def test_csv_with_na_string(tmp_path):
    """Cellule 'N/A' → NaN, pas ValueError."""
    csv = "Open time,Open,High,Low,Close,Volume\n2024-01-01,100,110,90,N/A,1000\n"
    f = tmp_path / "test.csv"
    f.write_text(csv)
    df = load_csv_data(str(f))
    assert df['Close'].isna().any()
```

**test_wfo_smoke.py** — filet pour Phase 3 :
- Stub `StrategyAdapter` (Protocol — pas besoin de VBT si `wfo.py` est importable sans, à vérifier ; sinon VBT-gated local obligatoire avant tout commit Phase 3)
- `walk_forward_optimization()` sur données synthétiques, 3 fenêtres, grille minuscule → vérifier structure `wfo_results`, nombre de fenêtres, déterminisme avec seed fixe
- Référence figée : sérialiser les `best_params` du smoke run AVANT la parallélisation, comparer APRÈS

---

### IMPL-BENCH : Benchmark parallélisation fenêtres → go/no-go

**Commit** : aucun (mesure jetable, résultats consignés dans l'issue IMPL-P1)

L'impact « 10× » de l'audit v1 n'a jamais été mesuré. L'évaluation intra-fenêtre est déjà vectorisée par chunks (`_eval_chunk_vectorized`) et VBT/Numba peuvent déjà saturer les cœurs. Avant d'investir 5j :

1. Mesurer durée série actuelle sur données réelles (`n_windows=4`, grille représentative), par fenêtre
2. Mesurer utilisation CPU pendant l'évaluation intra-fenêtre (`psutil`/`htop`) — si déjà > ~70% multi-cœurs, le gain fenêtre-parallèle sera faible
3. Prototype : `ThreadPoolExecutor`, 2 workers, 2 fenêtres, régime standard → speedup réel

**Critère go/no-go** : speedup prototype ≥ 1.5× → implémenter IMPL-P1. Sinon dépriorisér et documenter.

---

## Phase 3 — Performance Critique (Semaines 4–5)

### IMPL-P1 : Paralléliser fenêtres WFO (si IMPL-BENCH concluant)

**Fichier** : `apps/wfo_engine/wfo.py:729–957`  
**Référence audit** : P1 (révisé v2)  
**Commit** : `perf(wfo): paralléliser fenêtres en régime standard`

**Contrainte régimes guidés (v2 — bloquante)** : en `nn_guided`, `nn_guide.update()` (`wfo.py:940–944`) entraîne le guide sur chaque fenêtre pour construire la grille de la suivante ; en `prev_best_grid` (`wfo.py:783–793`), la grille i+1 dépend des best params i. **Ces régimes sont séquentiels par design.** Paralléliser uniquement si `optimization_regime == 'standard'` ; sinon fallback boucle série existante. Documenter dans l'UI.

**Contraintes VBT** :
- VectorBT Pro utilise des objets non-sérialisables (portfolios) → résultats ne peuvent pas traverser `ProcessPoolExecutor` directement
- Approche recommandée : paralléliser la phase d'**optimisation** (retourne dicts/DataFrames sérialisables), garder les backtests finaux IS/OOS dans le process principal
- Première approche : `ThreadPoolExecutor` (évite le pickling ; Numba relâche partiellement le GIL) — mesurer avant de passer à `ProcessPoolExecutor`

**Intégration existante à préserver** :
- **Checkpoints incrémentaux** (chantier 3.2) : écrire les résultats au fil des completions (`as_completed`), pas en fin de batch — sinon la durabilité régresse
- **`stop_event`** : vérifier entre les soumissions ; annuler les futures pending sur stop
- **Progress callback** : compter les fenêtres terminées, pas l'index courant
- **Backtest cache partagé** : `_BoundedCache` a déjà un lock — vérifié OK pour accès concurrent

**Architecture** :

```python
def _optimize_window(window_args: dict) -> dict:
    """Worker isolé par fenêtre — retourne dict sérialisable uniquement."""
    ...
    return {
        'window_index': i,
        'best_params': best_params,
        'optimization_results': optimization_results.to_dict('records'),
        'eval_count': eval_count,
    }

# CORRECTION v2 : settings.parallel_backend est un STRING ('dask', 'ray',
# 'pathos', 'threadpool' — cf. config.py:54). Ne PAS le passer à max_workers
# (TypeError immédiat). Ajouter un champ entier dédié dans WFOSettings :
#   max_parallel_windows: int = 0   # 0 = auto (os.cpu_count())
max_workers = settings.max_parallel_windows or os.cpu_count()

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(_optimize_window, args): args['i']
               for args in window_args_list}
    for fut in concurrent.futures.as_completed(futures):
        result = fut.result()
        _write_window_result(result['window_index'], ..., run_dir)   # checkpoint au fil de l'eau
        if stop_event.is_set():
            for f in futures:
                f.cancel()
            break

# Phase backtest IS/OOS post-parallélisation (série, objets VBT, ordre garanti)
for opt_result in sorted(window_opt_results, key=lambda x: x['window_index']):
    ...
```

**Vérification** : `test_wfo_smoke.py` (référence figée Phase 2) doit produire des `best_params` identiques en série et en parallèle, seed fixe.

---

### IMPL-P2 : Vectoriser Thompson sampling

**Fichier** : `apps/wfo_engine/adaptive_optimization.py:155–172`  
**Référence audit** : P2  
**Commit** : `perf(adaptive): vectoriser Thompson sampling`

```python
def score_candidates_thompson_batch(self, candidates: list[dict]) -> np.ndarray:
    """Score vectorisé de N candidats en un seul appel numpy."""
    n = len(candidates)
    p = len(self.param_names)

    means = np.zeros((n, p))
    sigmas = np.zeros((n, p))
    for j, name in enumerate(self.param_names):
        m, v, c = self._get_stats(name)
        means[:, j] = m
        sigmas[:, j] = np.sqrt(max(v, 1e-6) / (c + 1.0))

    samples = self.rng.normal(means, sigmas)    # 1 appel pour toute la matrice
    return np.mean(samples, axis=1)             # shape (n,)
```

Remplacer la boucle appelante par un appel batch, puis `argpartition` sur les scores. **Vérifier l'équivalence statistique** : la version scalaire moyenne sur `used` (params effectivement présents) — reproduire le masquage si certains candidats n'ont pas tous les params.

---

### IMPL-P3 : Fix SVI — iterrows + IS₂ batch

**Fichier** : `apps/wfo_engine/wfo.py:518–533`  
**Référence audit** : P3  
**Commit** : `perf(wfo): SVI itertuples, backtests IS₂ parallèles`

```python
# Remplacer iterrows par itertuples
for row in candidates.itertuples(index=False):
    params = {k: getattr(row, k) for k in param_cols}
    ...
```

Les k backtests IS₂ sont indépendants : si `run_backtest` ne supporte pas le batch natif, regrouper avec `ThreadPoolExecutor(max_workers=min(top_k, os.cpu_count()))`. Gain dominant = parallélisation des backtests, pas le remplacement d'`iterrows` (micro-optimisation à k≤20).

---

### IMPL-M1 : Documenter SVI IS₂ (pas de redesign)

**Fichier** : `apps/wfo_engine/wfo.py:497–538`, docstring + tooltip UI  
**Référence audit** : M1 (nuancé v2)  
**Commit** : `docs(wfo): clarifier SVI IS₂ comme re-ranking sur IS récent`

Docstring de `get_svi_best_params()` :

```python
"""
Sélection SVI (re-ranking sur IS récent).

NOTE MÉTHODOLOGIQUE : IS₂ est une tranche chronologique de la période IS
sur laquelle l'optimisation a déjà tourné — ce n'est PAS un holdout
hors-échantillon. Cette fonction re-classe les top_k candidats sur les
données IS récentes : elle pénalise les combos performants uniquement en
début de période et favorise la stabilité récente, mais ne fournit aucune
garantie de robustesse OOS.

Pour une sélection par stabilité structurelle, utiliser
selection_method='snv'. Un vrai holdout hors-IS n'est pas implémenté.
"""
```

Retirer le mot « holdout » de la docstring actuelle (ligne ~499) et du tooltip UI. Ajouter note dans l'UI (panel settings).

---

## Phase 4 — Refactoring & Robustesse (Semaine 6)

### IMPL-R6 : Wirer session.py (scope réduit)

**Fichiers** : `services/session.py`, `app.py`  
**Référence audit** : R6  
**Commit** : `refactor(app): wirer WFOSessionState sur les clés run-resolution`

**Scope révisé (v2)** : migrer 614 accès vers < 200 en 3j dans un fichier de 6219 lignes sans test UI = risque de régression élevé. Scope Phase 4 limité :

1. Clés run-resolution (`wfo_thread`, `wfo_job_state`, `wfo_running`, `wfo_results`, `wfo_prev_state`) — la zone la plus sensible
2. Clés résultats/config actives
3. Objectif réaliste : < 450 accès restants ; migration complète = chantier séparé hors plan

**Vérification** : `grep -c "st.session_state" apps/wfo_engine/app.py` avant/après ; smoke test manuel run complet (lancement, stop, erreur, harvest).

---

### IMPL-R9 : Cleanup thread stagewise

**Fichier** : `apps/wfo_engine/ui/stagewise_panel.py`  
**Référence audit** : R9  
**Commit** : `fix(stagewise_panel): cleanup session state après fin/crash thread`

Miroir de la logique `app.py:3590–3591` :

```python
# Dans le fragment de polling stagewise
if sw_thread is not None and not sw_thread.is_alive():
    run_state = st.session_state.get("_sw_run_state", {})
    # ... harvest résultats

    # Cleanup systématique (success ET error)
    for key in ["_sw_thread", "_sw_run_state", "_sw_stop_event"]:
        st.session_state.pop(key, None)
    st.rerun()
```

---

### IMPL-R10/R11 : Robustifier data_loading.py

**Fichier** : `apps/wfo_engine/data_loading.py`  
**Référence audit** : R10, R11  
**Commit** : `fix(data_loading): coercion CSV gracieuse, Binance fallback`

**CSV** :
```python
df = pd.read_csv(file_path, usecols=usecols)       # pas de dtype strict
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')   # "N/A" → NaN

nan_count = df[['Open', 'High', 'Low', 'Close']].isna().sum().sum()
if nan_count > 0:
    logger.warning("CSV: %d valeurs non-numériques converties en NaN", nan_count)
```

**Binance** :
```python
try:
    data_obj = vbt.BinanceData.fetch(
        ["BTCUSDT"], start=start_date, end=end_date, timeframe=base_timeframe,
        client_config=dict(base_endpoint='1'),
    )
    df_1s = data_obj.data.get('BTCUSDT')
    if df_1s is None or df_1s.empty:
        raise ValueError("Binance fetch returned empty data for BTCUSDT")
except Exception as exc:
    logger.error("Binance fetch failed: %s", exc, exc_info=True)
    raise RuntimeError(
        f"Impossible de charger les données Binance. "
        f"Vérifier la connectivité et le geo-block (api1.binance.com). "
        f"Erreur: {exc}"
    ) from exc
```

---

## Phase 5 — Durabilité & Optimisations Longues (Semaines 7+)

### IMPL-3.3 : Chantier 3.3 — Orphaned runs (v1 : détection)

**Fichier** : `apps/wfo_engine/services/run_service.py`, `app.py`  
**Commit** : `feat(run_service): détection de runs interrompus`

**Logique v1 (scope de ce plan)** :
1. Au démarrage Streamlit, scanner `reports/runs/` pour répertoires avec `checkpoint.json` sans `completed.json`
2. Si trouvés : notification UI « Run interrompu détecté » + résumé (fenêtres complétées, config, timestamp)

**Reprise (v2, chantier séparé)** — contraintes identifiées :
- `walk_forward_optimization()` doit accepter `resume_from_window: int = 0`
- **Dépendances séquentielles (ajout v2)** : la reprise doit restaurer l'état des régimes guidés (`nn_guide` entraîné, `prev_window_best_params`), sinon les fenêtres reprises divergent d'un run ininterrompu. Soit checkpointer cet état (sérialisation du guide), soit restreindre la reprise au régime standard — décision à prendre avant implémentation.

---

### IMPL-P4 : Cache KDTree SNV (grid search uniquement)

**Fichier** : `apps/wfo_engine/wfo.py:480–495`  
**Référence audit** : P4 (révisé v2)  
**Commit** : `perf(wfo): cache KDTree SNV si grille inchangée`

**Limite d'applicabilité (v2)** : utile uniquement en **grid search** avec grille identique entre fenêtres. En Bayesian/Optuna et régimes guidés, les combos diffèrent par fenêtre — guard explicite : ne pas cacher dans ces modes.

```python
# Cache borné (éviter croissance module-level non maîtrisée)
_snv_cache: dict = {}   # key=(grid_hash, k) → (tree, norm_vals, indices)
_SNV_CACHE_MAX = 4

def get_stable_best_params(results, param_grid, neighbor_count, ...):
    if optimization_method != 'grid':
        # combos différents par fenêtre → cache inopérant
        tree, norm_vals = _build_tree(results, param_grid)
    else:
        grid_hash = hash(frozenset((k, tuple(v)) for k, v in param_grid.items()))
        cache_key = (grid_hash, neighbor_count)
        if cache_key not in _snv_cache:
            if len(_snv_cache) >= _SNV_CACHE_MAX:
                _snv_cache.pop(next(iter(_snv_cache)))
            _snv_cache[cache_key] = _build_tree(results, param_grid)
        tree, norm_vals = _snv_cache[cache_key]
    ...
```

Note : si la grille est identique, les **indices de voisinage** (`tree.query`) sont aussi cachables — seul le `nanmean` des scores doit être recalculé par fenêtre.

---

### IMPL-P6 : Dédup neural search sampling (priorité basse)

**Fichier** : `apps/wfo_engine/neural_search.py:189–196`  
**Référence audit** : P6 (rétrogradé BASSE en v2)  
**Commit** : `perf(neural_search): hash-dedup sampling`

**Révision v2** : les doublons frappent le backtest cache keyé `(data_signature, params)` (`wfo.py:185–193`) — coût réel ≈ un lookup dict, pas un backtest. **Implémenter uniquement si profiling montre un coût mesurable.**

**Si implémenté** — hash-dedup, **pas** d'énumération cartésienne (`itertools.product` explose sur grilles réelles à ~20 params ; l'approche v1 du plan était inapplicable) :

```python
seen = set()
sampled_rows = []
attempts = 0
max_attempts = sample_count * 10   # borne anti-boucle infinie sur petites grilles
while len(sampled_rows) < sample_count and attempts < max_attempts:
    candidate = {k: values[int(self.rng.integers(0, len(values)))]
                 for k, values in base_param_grid.items()}
    key = tuple(sorted(candidate.items()))
    if key not in seen:
        seen.add(key)
        sampled_rows.append(candidate)
    attempts += 1
```

---

### IMPL-M2 : Modèle de slippage configurable

**Fichier** : `apps/wfo_engine/strategy.py`, `config.py`  
**Référence audit** : M2  
**Commit** : `feat(strategy): modèle de slippage configurable`

**Approche minimale** — slippage fixe en points de base :

```python
# Dans WFOSettings
slippage_bps: float = 0.0   # basis points par trade (défaut = 0 = comportement actuel)

# Dans strategy.py — appliquer via VBT slippage param
pf = vbt.Portfolio.from_signals(
    ...,
    slippage=settings.slippage_bps / 10000,   # VBT accepte fraction
)
```

Exposer dans l'UI sidebar comme paramètre optionnel avec tooltip expliquant l'impact. Défaut 0 = aucun changement de comportement pour les runs existants.

---

## Vérification Continue

```bash
# Syntax check après chaque phase
python -m py_compile apps/wfo_engine/app.py apps/wfo_engine/wfo.py \
  apps/wfo_engine/adaptive_optimization.py apps/wfo_engine/stagewise_optimizer.py \
  apps/wfo_engine/services/run_service.py

# Pine V3 tests (ne doit pas régresser)
pytest -q apps/wfo_engine/tests/test_pine_v3_*.py \
  apps/wfo_engine/tests/test_pine_strategy_test_adapter.py

# Nouveaux tests Phase 2 — à exécuter avant TOUT commit Phase 3+
pytest apps/wfo_engine/tests/test_run_service.py \
  apps/wfo_engine/tests/test_serialization.py \
  apps/wfo_engine/tests/test_data_loading.py \
  apps/wfo_engine/tests/test_wfo_smoke.py -v
```

---

## Matrice de suivi

| ID | Phase | Référence audit | Fichier principal | Effort estimé | Statut |
|----|-------|-----------------|-------------------|---------------|--------|
| IMPL-R3 | 1 | R3, R4 | `run_service.py:22–55` | 0.5j | ✅ DONE |
| IMPL-R7 | 1 | R7, R8 | `serialization.py`, 2 panels export | 1j | ✅ DONE |
| IMPL-R5 | 1 | R5, R14 | `wfo.py:81–93`, `metrics.py:12` | 0.5j | ✅ DONE |
| IMPL-R1 | 1 | R1, R2 | `app.py:3550–3593` | 0.5j | ✅ DONE |
| IMPL-TESTS | 2 | — | `tests/test_*.py` | 3j | ✅ DONE (33 tests) |
| IMPL-BENCH | 2 | P1 | mesure jetable | 0.5j | ✅ DONE — NO-GO (1.43× sur 2 CPUs < seuil 1.5×) |
| IMPL-P1 | 3 | P1 | `wfo.py:729–957` | 5j (go/no-go IMPL-BENCH) | ⏸ SKIP — 2 CPUs, speedup 1.43× insuffisant ; réévaluer sur machine ≥4 CPUs |
| IMPL-P2 | 3 | P2 | `adaptive_optimization.py:155–172` | 1j | ✅ DONE |
| IMPL-P3 | 3 | P3 | `wfo.py:518–533` | 1j | ✅ DONE |
| IMPL-M1 | 3 | M1 | `wfo.py:497–538` | 0.5j | ✅ DONE |
| IMPL-R6 | 4 | R6 | `session.py`, `app.py` | 3j (scope réduit) | ✅ DONE (scope réduit : run-resolution + résultats/viz ; 614 → 587 accès ; reste = chantier séparé) |
| IMPL-R9 | 4 | R9 | `stagewise_panel.py` | 0.5j | ✅ DONE |
| IMPL-R10/R11 | 4 | R10, R11 | `data_loading.py` | 1j | ✅ DONE |
| IMPL-3.3 | 5 | — | `run_service.py`, `app.py` | 2j (v1 détection seule) | ✅ DONE (v1 : scan + notification + acquittement ; reprise = chantier séparé) |
| IMPL-P4 | 5 | P4 | `wfo.py:480–495` | 1j | ✅ DONE (cache canonique lexsort, grid uniquement ; bench 4.0× sur 12k combos × 6 fenêtres) |
| IMPL-P6 | 5 | P6 | `neural_search.py:189–196` | 0.5j (si profiling justifie) | ⏸ SKIP — profiling 2026-06-13 : 0% doublons sur grille représentative (17 params, 1.3e8 combos, pool 3000) ; doublons des petites grilles ne coûtent qu'un forward MLP, backtests dédupliqués par le cache |
| IMPL-M2 | 5 | M2 | `strategy.py`, `config.py` | 2j | ✅ DONE (slippage_bps via param_grid → vbt from_signals, défaut 0) |
