# Plan de refactoring — WFO Engine

> **Destinataire : LLM exécutant.** Ce document décrit 3 chantiers de correction/amélioration sur l'application Streamlit `apps/wfo_engine/app.py`. Lis intégralement la section « Correction préalable » avant toute action : deux des trois critiques initiales ne tiennent pas après lecture du code, et le plan a été réécrit en conséquence.
>
> **Pré-requis garanti : VectorBT Pro est installé dans l'environnement d'exécution** (venv `/root/vbt_env`). Tu peux donc lancer la suite de tests COMPLÈTE, y compris les tests qui exercent `wfo.py`, `strategy.py`, `indicators.py`, `adaptive_optimization.py`, `stagewise_optimizer.py` — pas seulement les tests Pine légers. Vérifier au démarrage : `/root/vbt_env/bin/python -c "import vectorbtpro; print(vectorbtpro.__version__)"`.

---

## Correction préalable (à lire avant tout)

Une critique antérieure affirmait 3 défauts. **Après lecture du code, seuls 1 sur 3 tient tel quel.** Ne pas implémenter de solution basée sur les formulations originales.

| Critique initiale | Verdict après lecture code | Preuve |
|---|---|---|
| **1. God file `app.py`** | ✅ **Valide** | 6218 lignes, 614 accès `st.session_state`, 34 `def`/`class` top-level |
| **2. Streamlit re-render bloquant à chaque interaction** | ⚠️ **Partiellement faux** — le vrai coût est ailleurs | WFO tourne dans un thread (voir ci-dessous), pas dans le flux UI |
| **3. Pas d'async/worker, UI bloquée, refresh tue le run** | ❌ **Largement faux** | Worker thread + stop control + progression live **existent déjà** |

### Ce qui existe déjà dans le code (ne pas réimplémenter)

- **Worker thread** : `app.py:3758` — `threading.Thread(target=_wfo_worker, daemon=True)`. Le WFO ne bloque pas l'UI.
- **Stop propre** : classe `WFOControl` (`app.py:3061`), `request_stop()` appelé à `app.py:3766`, vérifié via `should_stop()` dans `wfo.py`/`adaptive_optimization.py`.
- **Progression live** : `job_state` dict mutable partagé (`app.py:3685`), alimenté par `status_callback` (`services/run_service.py:113`).
- **Auto-refresh sans interaction** : `app.py:6215-6219` — boucle `time.sleep(0.8) + st.rerun()` tant que `wfo_running`.
- **Couche logique découplée** : `services/run_service.py:run_optimization_job()` est déjà sans appel Streamlit (prend `control` + `job_state`).
- **Snapshot/restore d'état** : `_capture_state_snapshot()` / `_restore_state_snapshot()` autour du run.

**Implication :** le chantier 3 ne consiste PAS à « ajouter un worker ». Il consiste à rendre durable un run déjà asynchrone (voir Chantier 3 reformulé).

---

## Chantier 1 — Découper le god file `app.py`

### Diagnostic
`app.py` = 6218 lignes mélangeant config sidebar, logique métier, rendu UI, et état implicite (`st.session_state` accédé 614 fois, jamais typé). Non testable unitairement.

### Objectif
`app.py` < ~400 lignes : routeur + orchestration. Toute logique extraite vers `services/` (logique) et `ui/` (rendu). Les dossiers `services/` et `ui/` existent déjà — **enrichir, ne pas recréer.**

### Stratégie : extraction progressive, PAS réécriture
Réécrire un fichier de 6200 lignes piloté par `st.session_state` = casse garantie. Procéder par petites extractions vérifiables, une par commit.

### Étapes

**1.1 — Créer `services/session.py` : façade d'état typée**

Tuer l'état implicite. Wrapper dataclass au-dessus de `st.session_state` :

```python
# services/session.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import streamlit as st


@dataclass
class WFOSessionState:
    """Vue typée de st.session_state. Source unique des clés de session WFO."""
    wfo_running: bool
    wfo_results: dict | None
    wfo_job_state: dict | None
    wfo_control: Any | None
    df: Any | None

    @staticmethod
    def read() -> "WFOSessionState":
        g = st.session_state.get
        return WFOSessionState(
            wfo_running=bool(g("wfo_running", False)),
            wfo_results=g("wfo_results"),
            wfo_job_state=g("wfo_job_state"),
            wfo_control=g("wfo_control"),
            df=g("df"),
        )
```

Migration : remplacer les accès dispersés `st.session_state.get('wfo_results')` par `WFOSessionState.read().wfo_results`, **par lots**, en validant `py_compile` à chaque lot. Ne pas tout migrer d'un coup.

**1.2 — Extraire le lancement de run vers `services/run_launcher.py`**

Le bloc `app.py:3682-3761` (construction `job_state`, `WFOControl`, `_wfo_worker`, démarrage thread) → fonction `launch_wfo_run(config, callbacks) -> RunHandle`. `app.py` ne garde que l'appel + le rendu du bouton.

**1.3 — Extraire la résolution de run terminé vers `services/run_launcher.py`**

Le bloc `app.py:3550-3593` (détection thread mort, transfert résultats vers session, métadonnées) → `resolve_finished_run(handle) -> RunOutcome`.

**1.4 — Découper le rendu principal**

Le corps après ligne ~5000 (rendu résultats, onglets) → panels dans `ui/`. Suivre le motif des panels existants (`ui/final_backtest_panel.py`, `ui/stagewise_panel.py`).

### Critère de succès
- `python -m py_compile apps/wfo_engine/app.py` passe à chaque étape.
- **Suite de tests COMPLÈTE verte** (VBT Pro installé) — pas seulement les tests Pine légers. Lancer l'intégration WFO end-to-end :
  ```bash
  PYTHONPATH=apps/wfo_engine /root/vbt_env/bin/python -m pytest apps/wfo_engine/tests/ -q
  ```
  Inclut `test_wfo_integration.py`, `test_wfo_unit.py`, `test_repro.py`, `test_strategy_short.py`, `test_metrics.py` — qui exercent réellement le moteur WFO + VBT.
- `test_repro.py` (reproductibilité) doit rester vert après chaque extraction — c'est le filet anti-régression numérique : si une extraction altère un résultat de backtest, ce test casse.
- `session.py` testable sans Streamlit (mock `st.session_state` = dict).

### Garde-fous
- **Un commit par extraction.** Jamais d'extraction multiple non vérifiée.
- Ne pas changer le comportement — refactoring pur. Aucune modif de `CHANGELOG.md` côté fonctionnel.
- Conserver les noms de clés `st.session_state` exacts (sérialisation/import ZIP en dépend : voir `ui/export_panel.py:1535+`).

---

## Chantier 2 — Coût du re-render (reformulé)

### Diagnostic réel (≠ critique initiale)
Le problème n'est PAS « blocage à chaque interaction ». C'est que pendant un run, `app.py:6215-6219` ré-exécute **l'intégralité du script 6200 lignes toutes les 0,8 s** :

```python
# app.py:6215-6219 (état actuel)
if st.session_state.get('wfo_running'):
    live_thread = st.session_state.get('wfo_thread')
    if live_thread is not None and live_thread.is_alive():
        time.sleep(0.8)
    st.rerun()   # ← réexécute TOUT le script
```

Sur un run long (campagne stagewise de plusieurs heures), c'est ~4500 réexécutions complètes du script par heure, chacune relisant `st.session_state` des centaines de fois et reconstruisant toute l'UI.

### Solution : `@st.fragment` (Streamlit 1.50 installé — disponible)

Isoler la carte de progression dans un fragment à auto-refresh, pour que SEUL le fragment se rafraîchisse, pas les 6200 lignes.

```python
# Remplacer la boucle sleep+rerun globale par un fragment ciblé.
# Le fragment se ré-exécute seul toutes les 0.8s via run_every.

@st.fragment(run_every="0.8s")
def render_live_progress():
    state = WFOSessionState.read()        # cf. Chantier 1.1
    if not state.wfo_running:
        return
    job = state.wfo_job_state or {}
    st.sidebar.progress(float(job.get("progress", 0.0)))
    st.sidebar.caption(job.get("message", "Running..."))
    _render_running_status_card(job, current_conf)
    # Quand le thread est mort, déclencher UN rerun global pour
    # résoudre le run (transfert résultats → session) :
    thread = state.wfo_control  # ou handle dédié
    if job.get("status") in ("completed", "stopped", "error"):
        st.rerun()  # rerun global ponctuel, pas en boucle
```

Et **supprimer** la boucle `time.sleep(0.8) + st.rerun()` de `app.py:6215-6219`.

### Précautions
- `@st.fragment(run_every=...)` requiert Streamlit ≥ 1.33 → OK (1.50 installé). Vérifier : `python -c "import streamlit; print(streamlit.__version__)"`.
- Le fragment ne doit écrire dans `st.session_state` que des clés non lues ailleurs pendant le même cycle (limitation fragment). La résolution finale du run (transfert `job_state['results']` → `wfo_results`) doit rester dans un rerun global, déclenché une seule fois à la complétion.
- Tester : lancer un petit run, vérifier que la progression avance ET que le reste de la page ne « clignote » plus.

### Gain attendu
Réexécutions complètes du script : de ~1/0.8s à ~0 pendant le run. UI fluide, CPU UI quasi nul (le coût réel reste le thread WFO, inchangé).

---

## Chantier 3 — Durabilité du run (le vrai manque)

### Diagnostic réel (≠ critique initiale)
Le worker existe. Le vrai trou est la **durabilité** :

- Les résultats n'atterrissent dans `job_state['results']` qu'à la **complétion** (`app.py:3746` — unique écriture). Pendant le run : RAM uniquement.
- Persistance disque = **export ZIP manuel seulement** (`ui/export_panel.py`).
- Le handle du thread vit dans `st.session_state['wfo_thread']`. Session navigateur perdue → thread orphelin, résultats inatteignables.
- Redémarrage du process Streamlit (deploy, OOM, crash) pendant une campagne de plusieurs heures → **run perdu, zéro reprise.**

C'est le risque le plus coûteux en pratique sur un VPS mono-process.

> Note : passer du thread à un subprocess ne résout PAS la durabilité en soi (un subprocess enfant meurt aussi avec le parent, sauf détaché). La solution est le **checkpoint disque incrémental**, orthogonal au choix thread/process. On garde donc le thread existant.

### Solution : checkpoint disque incrémental

**3.1 — Écrire `job_state` sur disque à chaque mise à jour de progression**

Dans `services/run_service.py:status_callback` (ligne 113), après mise à jour de `job_state`, sérialiser un snapshot léger (sans le DataFrame, sans les gros résultats) :

```python
# services/run_service.py — dans status_callback, après maj job_state
import json, pathlib
def _checkpoint(job_state, run_dir):
    snap = {k: job_state.get(k) for k in
            ("status", "progress", "message", "window",
             "evaluations", "run_id", "started_at_utc")}
    p = pathlib.Path(run_dir) / "checkpoint.json"
    p.write_text(json.dumps(snap, default=str))
```

Chemin : `reports/runs/<run_id>/checkpoint.json`. `run_id` existe déjà (`app.py:3683`).

**3.2 — Persister les résultats par fenêtre, au fil de l'eau**

Aujourd'hui les résultats par fenêtre n'arrivent qu'en bloc final. Pour vraie reprise : à la fin de chaque fenêtre WFO (dans `wfo.py`, là où `status_callback` reçoit un message `type=="stats"` avec `window`), écrire le résultat partiel de la fenêtre sur disque :

```
reports/runs/<run_id>/windows/window_<N>.json
```

Ainsi un crash en fenêtre 7/10 conserve les fenêtres 1-6.

**3.3 — Détection et reprise au démarrage**

Au lancement de `app.py`, scanner `reports/runs/` pour des runs `status == "running"` sans thread vivant (= orphelins après crash). Proposer dans l'UI : « Run `<id>` interrompu (fenêtre N/M). [Reprendre] [Archiver] ». Reprise = relancer `walk_forward_optimization` à partir de la première fenêtre manquante.

> Si la reprise complète (3.3) est trop lourde en V1, livrer au minimum 3.1 + 3.2 : même sans reprise auto, les résultats partiels sur disque évitent la perte totale et permettent une reconstruction manuelle.

### Précautions
- **Ne pas sérialiser le DataFrame OHLCV** dans les checkpoints (plusieurs Mo à Go). Seulement métadonnées + résultats par fenêtre.
- Réutiliser `_sanitize_for_json` (déjà dans `app.py`) et `domain/serialization.py` pour la sérialisation sûre (datetime UTC, types numpy).
- `reports/` est gitignored (voir `CLAUDE.md` § Key Conventions) — OK, ne rien committer.
- Écriture atomique : écrire dans `checkpoint.json.tmp` puis `os.replace()` pour éviter un fichier corrompu si crash pendant l'écriture.

### Critère de succès
- Tuer le process Streamlit (`kill`) pendant un run multi-fenêtres **réel** (VBT Pro installé → vrai backtest, vraies fenêtres) → relancer → les fenêtres déjà calculées sont sur disque et détectées.
- Test de reprise reproductible : un run repris depuis checkpoint doit produire des résultats identiques à un run complet sans interruption (s'appuyer sur le déterminisme validé par `test_repro.py`).

---

## Ordre d'exécution recommandé

| Ordre | Chantier | Effort | Impact | Pourquoi ce rang |
|---|---|---|---|---|
| 1 | **3.1 + 3.2** checkpoint disque | ~1-2 j | ★★★★★ | Seul vrai risque de perte de données ; indépendant du reste |
| 2 | **2** fragment progression | ~heures | ★★★★ | Gain UX/CPU immédiat, peu risqué |
| 3 | **1.1** `session.py` | ~1 j | ★★★ | Prérequis propre pour 1.2-1.4 et pour le fragment du chantier 2 |
| 4 | **1.2-1.4** extraction `app.py` | ~3-4 j | ★★★ | Maintenabilité long terme, faire par petits commits |
| 5 | **3.3** reprise auto | ~2 j | ★★ | Confort ; 3.1+3.2 couvrent déjà la non-perte |

> 1.1 (`session.py`) est référencé par le chantier 2 (le fragment lit l'état via la façade). Si on fait 2 avant 1.1, lire `st.session_state` directement dans le fragment puis migrer ensuite — acceptable.

---

## Contraintes transverses (rappel `CLAUDE.md`)

- **PYTHONPATH** doit inclure `apps/wfo_engine` : `PYTHONPATH=apps/wfo_engine /root/vbt_env/bin/python -m pytest ...`
- Vérif syntaxe avant push : `python -m py_compile apps/wfo_engine/app.py apps/wfo_engine/main.py apps/wfo_engine/wfo.py ...` (liste complète dans `CLAUDE.md`).
- **VectorBT Pro EST installé** dans l'environnement d'exécution (venv `/root/vbt_env`) → lancer la suite **complète** localement, moteur WFO inclus. ⚠️ Distinction à garder en tête : la **CI GitHub** (`.github/workflows/ci.yml`) reste, elle, sans VBT Pro (Pine + py_compile seulement). Donc tout test que tu ajoutes et qui importe `vectorbtpro` ne tournera PAS en CI — soit le garder local, soit le marquer `@pytest.mark.skipif` sur absence de `vectorbtpro` pour ne pas casser la CI.
- UI en français, code/commentaires/docstrings en anglais.
- Commits conventionnels : `refactor(wfoe):`, `feat(wfoe):`, `perf(wfoe):`.
- Mettre à jour `CHANGELOG.md` (en français) **uniquement** quand le comportement observable change (chantiers 2 et 3 oui ; chantier 1 = refactoring pur, non).
