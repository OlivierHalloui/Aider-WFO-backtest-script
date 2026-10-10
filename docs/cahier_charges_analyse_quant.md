# Cahier des charges v2 — Section « Analyse quant » dans l'UI WFO Engine

| | |
|---|---|
| **Projet** | ATDMF Strategy Walk-Forward Optimizer (WFOE) |
| **Objet** | Ajout d'une section « Analyse quant » dans l'UI Streamlit |
| **Réf.** | `apps/wfo_engine/app.py`, `apps/wfo_engine/ui/`, pair A2A `wfo-quant` (127.0.0.1:9924) |
| **Statut** | ✅ **v2.10 — VALIDÉE par `wfo-quant` le 2026-10-08** (« Définitions calculables entièrement définies : oui ») — prêt pour T1 |
| **Dépendances** | run WFO terminé (`wfo_results`), pair A2A `wfo-quant` opérationnel |

> **Historique** — v1 rédigée par l'orchestrateur ; revuée par `wfo-quant` →
> verdict *« pas encore exploitable tel quel »* avec 4 points bloquants et un
> remaniement de schéma. **Cette v2 intègre l'ensemble de ses remarques.**

---

## 1. Objet et contexte

Après un run WFO, l'UI expose les résultats (métriques IS/OOS, Final Backtest,
grille de trials) mais **n'offre aucune lecture critique** : sur-ajustement,
stabilité des paramètres, sensibilité aux coûts de transaction ne sont évalués
nulle part. Ces analyses sont aujourd'hui faites à la main, en demandant au
pair A2A `wfo-quant` d'inspecter un ZIP de résultats.

Cette fonctionnalité intègre cette étape dans l'UI :

1. **un bloc de calculs** produits localement (indicateurs quantitatifs
   déterministes, recalculés à chaque affichage) ;
2. **un bouton** qui déclenche une analyse d'interprétation auprès du pair
   `wfo-quant` ;
3. **une zone d'affichage** de l'interprétation renvoyée par le pair.

**Principe de répartition des responsabilités** (imposé par `wfo-quant`, §5.2) :
> l'application **calcule et scelle les faits** ainsi qu'un **pré-verdict
> déterministe** ; le pair ne renvoie qu'un **verdict interprété et ses
> références** (`evidence_refs`) ; l'UI **signale tout désaccord sans modifier
> les faits**.

Deux niveaux de lecture donc : le **quantitatif** (toujours affiché, calculé par
l'application, reproductible) et le **qualitatif** (à la demande, produit par
l'expert A2A, non déterministe).

---

## 2. Besoins utilisateurs

| ID | En tant que… | je veux… | afin de… |
|---|---|---|---|
| U1 | analyste | voir les indicateurs de robustesse dès la fin d'un run | juger vite si le run est exploitable |
| U2 | analyste | lancer une analyse d'interprétation en un clic | ne pas sortir de l'UI pour consulter `wfo-quant` |
| U3 | analyste | lire l'interprétation à côté des chiffres | confronter le qualitatif au quantitatif |
| U4 | analyste | relancer l'analyse après réglage | itérer sans recharger la page |
| U5 | analyste | exporter l'analyse avec les résultats | tracer l'avis d'expert dans le ZIP de run |
| U6 | analyste | savoir quand une métrique n'est **pas calculable** | ne jamais confondre « absent » et « zéro » |

---

## 3. Périmètre

**Inclus**
- Nouveau panneau `ui/quant_analysis_panel.py` (onglet dédié).
- Calcul local des indicateurs et pré-verdict (§5.1), manifeste et contrôle
  d'intégrité (§5.0).
- Appel A2A au pair `wfo-quant` + affichage structuré de la réponse (§5.2).
- Politique de données manquantes + seuils de verdict (§5.3).
- Export du JSON d'analyse dans le ZIP (`services/export_utils.py`).

**Exclus**
- Modification des métriques existantes ou du moteur WFO.
  > **Dérogation T1 (validée 2026-10-08, arbitrage A)** : pour satisfaire le
  > contrôle bloquant §5.0 « appariées par fenêtre et par paramètres
  > sélectionnés », `wfo.py` et `adaptive_optimization.py` joignent désormais un
  > champ **`params_sha`** (sha256 du `best_params`) aux lignes
  > `in_sample_performance` / `out_of_sample_performance` et à `window_results`.
  > C'est un ajout **purement annotatif** : aucune métrique ni logique
  > d'optimisation n'est modifiée.
- Éditeur de prompt libre (le prompt est généré par l'application).
- Tout autre pair A2A (`wfo-reviewer`, `wfo-data`, …) — `wfo-quant` uniquement.
- **Ordonnancement de trading à partir du verdict** : un `GO` vise un usage
  **exploratoire / validation OOS**, jamais une autorisation de mise en
  production (voir `verdict_scope`, §5.2).

---

## 4. Architecture et flux

```
┌───────────────────────── UI Streamlit ─────────────────────────┐
│  onglet « Analyse quant » — ui/quant_analysis_panel.py         │
│   ├─ 5.0 manifeste + contrôle d'intégrité (déterministe)       │
│   ├─ 5.1 indicateurs Q1–Q8 + pré-verdict (calcul local)        │
│   │     └─ services/quant_indicators.py  ◄── wfo_results,      │
│   │                                          all_trials,       │
│   │                                          final_trades,     │
│   │                                          returns par barre │
│   ├─ [ Lancer l'analyse wfo-quant ]  ☑ auto en fin de run      │
│   │     └─ services/quant_expert.py                            │
│   │           └─ POST JSON-RPC SendMessage ──► 127.0.0.1:9924  │
│   └─ 5.2 verdict + findings (JSON canonique rendu en tableaux) │
└────────────────────────────────────────────────────────────────┘
```

**Flux nominal**
1. Un run est terminé → `wfo_results` + artefacts en `session_state`.
2. Le panneau calcule le **manifeste**, exécute le **contrôle d'intégrité**,
   puis affiche les indicateurs Q1–Q8 et le **pré-verdict déterministe**.
3. L'utilisateur clique **« Lancer l'analyse wfo-quant »** (ou le hook de fin de
   run le déclenche, §5.2).
4. `services/quant_expert.py` construit un prompt **compact et borné** (§5.2) et
   envoie un `SendMessage` JSON-RPC à `127.0.0.1:9924`.
5. La réponse est validée contre le **schéma JSON** (§5.2) puis stockée en
   `session_state['quant_analysis']`.
6. L'UI affiche verdict + findings et **signale tout désaccord** entre le
   pré-verdict local et l'avis du pair (sans jamais modifier les faits).
7. À l'export ZIP : `quant_analysis.json` (**artefact canonique**) et son rendu
   dérivé `quant_analysis.md`.

> **Artéfact canonique** : le **JSON validé** est la source de vérité de
> l'analyse. Le Markdown n'est qu'un rendu dérivé pour lecture/édition.

---

## 5. Spécification fonctionnelle

### 5.0 Manifeste des données et contrôle d'intégrité — *prérequis (P0)*

**Manifeste** (`services/quant_indicators.py` → `build_run_manifest()`) : transmis
au pair et affiché en tête de panneau. Sans périmètre et provenance vérifiés,
aucune comparaison IS/OOS/Final ni aucun Sharpe n'est fiable.

`run_id`, `input_digest` (hash des artefacts), `engine_version`,
`metrics_version`, `seed`, `optimization_method`, `selection_method`,
`mode` (`anchored`/`rolling`), `n_windows` + dates des fenêtres, `embargo`/
`purge`, `timeframe`, période du Final, **source et granularité des rendements
(par barre vs par trade)**, conventions d'unités (`return_pct`, `win_rate_pct`,
`sharpe_per_bar`), **coûts inclus ou non** (§5.1 Q8).

**Contrôle d'intégrité pré-appel** (`check_run_integrity()`) — bloquant :
run terminé, colonnes requises présentes, valeurs finies, fenêtres IS/OOS
**appariées par fenêtre et par paramètres sélectionnés**, trades/rendements
disponibles, identifiants uniques, cohérence `n_trials` / lignes / essais
valides, frais déjà appliqués. En cas d'échec → statut **« non évaluable +
cause »** affiché, et **aucun appel A2A n'est lancé**.

### 5.1 Bloc « Calculs » — toujours affiché, calculé localement

Module `services/quant_indicators.py`, fonctions pures testables.

| # | Indicateur | Définition | Affichage |
|---|---|---|---|
| Q1 | **Budget effectif** | essais *demandés / lancés / terminés / échoués-prunés* ; scores valides ; **configurations uniques par fenêtre** ; cardinalité de l'espace discret si connue | barre + table |
| Q2 | **Diversité** | taux de doublons **intra-fenêtre** et **inter-fenêtres** (séparés — une répétition entre fenêtres WFO est légitime) | 2 KPI |
| Q3 | **Forme de la distribution** | quantiles **P50 / P90 / P95 / max** du `combined_score` ; écart `max − P95` ; taille du haut de distribution ; nb de **jeux distincts** dans le top | histogramme + KPI |
| Q4 | **Stabilité des paramètres** | **dispersion normalisée à la plage** du paramètre (ou IQR) + fréquence de sélection par fenêtre + paramètres **bloqués sur une borne**. `CV` seulement pour une variable positive dont la moyenne est éloignée de zéro | table triée |
| Q5 | **Voisinage du gagnant** | **voisins distincts évalués dans la même fenêtre** ; distance normalisée sur plages (règle explicite pour catégories et bornes) ; signalement **gagnant isolé / en bordure** | sparkline + alerte |
| Q6 | **Érosion IS → OOS** | **par fenêtre** puis agrégé : érosion Sharpe/return/PQS **seulement si dénominateurs et signes le permettent** ; amplitudes `\|DD\|` comparées seulement si les deux DD sont définis ; résumé médiane + dispersion + nb de fenêtres en dégradation | table par fenêtre + code couleur |
| Q7 | **Sharpe non-annualisé homogène** | recalcul via `metrics._sharpe_from_returns` **à partir des séries de rendements par barre** IS/OOS/Final (fournies explicitement). Si la série est absente → **« indisponible »**, jamais de substitution par un fallback de provenance différente | table homogène |
| Q8 | **Sensibilité aux coûts** | ① établir la **convention du portefeuille** (rendements de `trades.returns` nets ou bruts de frais/slippage) ; ② si nets → afficher le **coût déjà inclus** et un **scénario de coût additionnel** (réexécution si possible, sinon **approximation explicitement étiquetée**) ; ③ si bruts → marge nette par côté. **Pas de « seuil de ruine » déduit d'un edge moyen** | KPI alerte + étiquette |

**Incertitude OOS (P1)** — par fenêtre : nb de fenêtres, trades OOS, durée,
exposition ; **intervalle de confiance** du rendement/edge par **bootstrap en
blocs ou par fenêtre** si l'effectif le permet ; **concentration du P&L**
(meilleurs trades / meilleures fenêtres).

**Pré-verdict déterministe** : l'application produit son propre verdict
GO/WATCH/NO_GO **avant** l'appel A2A, selon les seuils §5.3. Il est affiché à
côté de celui du pair ; tout désacord est signalé.

**Règles**
- Aucun appel réseau dans §5.0/§5.1 : 100 % recalculé et reproductible.
- Chaque valeur est **sourcée** (champ + formule) pour être vérifiable.
- Cache et export liés au **hash du run + version des indicateurs**, invalidés
  au changement de réglage ou de run.

### 5.2 Bloc « Interprétation » — à la demande

**Déclenchement** — bouton **et** analyse automatique optionnelle (arbitrage §11.2 = B)
- `Lancer l'analyse wfo-quant` : inactif sans run, si le contrôle d'intégrité a
  échoué, ou si le pair est injoignable (sondage
  `/.well-known/agent-card.json`).
- `st.checkbox("Analyse automatique en fin de run")` (défaut **coché**) : hook
  `services/quant_expert.auto_analyze_run()`. Décachable pour itérer vite.
  **Point de déclenchement** (report explicitement accepté, §10) : les preuves
  exigées par §5.0 incluent `final_trades`, qui n'existent qu'**après le
  backtest final** (sélection L2 cross-fenêtres + `final_backtest_panel`).
  Le hook est donc câblé en fin de run WFO (`services/run_service.py`) mais ne
  devient **exploitable** qu'une fois le backtest final produit ; le
  déclenchement effectif automatique est intégré en **T5**, après le backtest
  final. En attendant, un appel sans `final_trades` renvoie un
  `non_evaluable` explicite et **aucun appel A2A** (§5.3 cas a) — jamais une
  analyse sur preuves incomplètes.
- Pendant l'appel : `st.status` avec temps écoulé (réponse attendue 2–6 min).
- Labels secondaires : « Relancer l'analyse », « Effacer ».

**Prompt envoyé** (généré par `services/quant_expert.py`, jamais éditable) —
**compact et borné**, avec **troncature déclarée** :
- le **manifeste** (§5.0) ;
- les **agrégats vérifiés** Q1–Q8 + le pré-verdict local ;
- les **tableaux par fenêtre** (IS/OOS appariés) et quelques essais/voisins
  pertinents (seuil de taille + mention explicite de ce qui est tronqué) ;
- le rappel des sémantiques (Sharpe non-annualisé,
  `calc_avg_pl = mean(trades.returns)*100`, unités) ;
- consigne : **distinguer faits, inférences et hypothèses**, et répondre **uniquement
  avec le JSON §5.2-schema**, en français, une seule réponse, sans question en
  retour. Le pair **ne recopie pas** les nombres déjà fournis — il les interprète.

**Schéma de sortie — `quant_analysis.v1`** (artefact canonique)

```json
{
  "schema_version": "quant_analysis.v1",
  "run_id": "…", "input_digest": "…",
  "indicator_version": "…", "generated_at": "…",

  "verdict": "GO | WATCH | NO_GO",
  "verdict_scope": "exploratoire | validation_oos | deploiement",
  "confidence": "faible | moyenne | elevee",
  "verdict_justification": "1 phrase, chiffres à l'appui",

  "blocking_findings": [{ "id": "…", "constat": "…", "evidence_refs": ["Q6", "W3"] }],
  "limitations": ["…"],

  "findings": [
    { "id": "F1", "niveau": "majeur | mineur | info",
      "constat": "…", "evidence_refs": ["Q3", "Q5"],
      "interpretation": "…", "condition": "… | null" }
  ],

  "recommandations": [
    { "action": "…", "priorite": 1, "motif": "…", "critere_de_validation": "…" }
  ],

  "accord_avec_preverdict": true,
  "preverdict_local": "GO | WATCH | NO_GO"
}
```

> **`diagnostics` n'est PAS dans la sortie du pair.** C'est un bloc d'**entrée** :
> il est calculé et **scellé par l'application** (§5.1) et transmis dans le
> prompt. Le pair ne renvoie que du **qualitatif interprété**.

**Source de vérité et contrôle d'égalité** (fixés)
- **Source de vérité des chiffres : l'application** (§5.0/§5.1). Tout nombre
  présent dans le prompt est définitif.
- Le pair **ne renvoie aucune métrique chiffrée à recalculer**.

**Nomenclature des identifiants** (`evidence_refs`) — fermée, non extensible :

| Forme | Désigne | Exemple |
|---|---|---|
| `Q1`…`Q8` | indicateur §5.1 | `Q6` |
| `W<n>` | fenêtre n, 1-indexé selon `window_info.csv` | `W3` |
| `M.<champ>` | champ du manifeste §5.0 | `M.timeframe` |
| `P.<param>` | paramètre de la grille | `P.timeperiod` |
| `T<n>` | ligne n de `all_trials.csv` | `T17` |

Toute `evidence_refs` hors nomenclature → `findings` marqué
`reference_invalide` dans l'UI.

**Règle de citation des chiffres** — dans `constat` / `interpretation`, tout
nombre est exprimé **sous forme de référence littérale `{{ID.champ}}`** (ex.
`{{Q6.erosion_sharpe_pct}}`, `{{W3.sharpe_per_bar}}`), **jamais en valeur
numérique nue**. L'application substitue la valeur scellée à l'affichage et
contrôle que la référence existe ; un littéral numérique détecté → le passage
est marqué `chiffre_non_reference` et signalé. Rapprochement ainsi
**formellement vérifiable** (parsing de `{{…}}`), sans comparaison flottante.

- Si le pair renvoie malgré tout un champ `diagnostics` → **ignoré et signalé**
  (hors schéma).

**Règles de schéma**
- Toute métrique indisponible est `null` **avec** `raison` explicite — **jamais
  `NaN`/`Infinity`, jamais `0` par défaut** (validateur JSON bloquant).
- Enums et unités figés (`return_pct`, `win_rate_pct`, `sharpe_per_bar`, …).
- `evidence_refs` pointe vers Q1–Q8 / fenêtre / champ source ; l'application les
  **vérifie** et signale les références invalides.

**Rendu UI** : bandeau de verdict codé (🟢 `GO` / 🟠 `WATCH` / 🔴 `NO_GO`) avec
`verdict_scope` + `confidence` · pré-verdict local à côté · `blocking_findings`
en rouge · tableaux `findings` (avec `evidence_refs`) · **diagnostics locaux**
(§5.1, bloqué en lecture) · recommandations numérotées (`action`, `motif`,
`critere_de_validation`) · horodatage + `context_id` A2A en pied de bloc.

### 5.3 Politique des données manquantes et seuils de verdict

**Données manquantes** — un champ absent, non fini, ou un échantillon OOS trop
faible **ne se transforme jamais en 0 puis en badge vert**. Deux cas distincts
(à ne pas confondre) :

| Cas | Déclenchement | Verdict affiché | Appel A2A |
|---|---|---|---|
| **a. Intégrité KO** (§5.0) | contrôle bloquant en échec | **`NO_GO`** — c'est le **pré-verdict local** (déterministe), accompagné du statut `non_evaluable` + de la cause | **aucun** (pas de données fiables à interpréter) |
| **b. Métrique partielle** | métrique indisponible mais run exploitable | verdict **plafonné à `WATCH`** ; champ `null` + `raison` | oui, avec `limitations` |

→ Donc pas de contradiction : en cas a, le verdict affiché est le pré-verdict
local `NO_GO` (jamais celui du pair, absent par construction) et le bandeau
précise `non_evaluable — <cause>` plutôt que de laisser croire à une analyse.

**Critères calculables** pour le pré-verdict déterministe (définitions exactes —
chaque critère doit être évaluable par une formule, sans appréciation) :

**Vocabulaire du voisinage** (strict — deux notions à ne jamais confondre) :
- **V — voisin** : jeu de paramètres **distinct**, évalué dans la **même fenêtre**
  que le gagnant, et à une distance **`d ≤ 0,10`** (proximité seule).
- **V_perf — voisin performant** : élément de **V** dont le score est en plus
  **≥ 80 %** du score du gagnant (proximité **et** performance).

**Règle de priorité des mesures** (à appliquer dans cet ordre) :
1. `score_gagnant > 0` **et** comparable → **cas nominal** : seuils relatifs au
   gagnant (`≥ 80 %`, `< 50 %`).
2. Sinon (`score_gagnant ≤ 0` **ou** non comparable) → **cas alternatif** :
   seuils `z` (mesure alternative ci-dessous, bas de section).
3. Si le cas alternatif est **indéterminé** (`z` non définie) → statut
   `indeterminate`, plafond `WATCH`.

→ Donc « score ≤ 0 » ne signifie **pas** « non applicable » : cela déclenche le
cas alternatif. « Non applicable / `indeterminate` » n'arrive qu'à l'étape 3.

| Critère flou | Définition calculable |
|---|---|
| « concentration excessive » du P&L | `k = max(1, ceil(0,10 × n_trades))` ; concentration si le cumul des `k` meilleurs trades **> 50 %** du P&L total **ou** le cumul des **2** meilleures fenêtres **> 60 %** du P&L OOS. **Si `P&L_total ≤ 0`** → critère **non défini**, statut `indetermine` (jamais « conforme ») |
| « érosion > 50 % **récurrente** » | **≥ 50 % des fenêtres OOS** ont une érosion du Sharpe **> 50 %** |
| « majorité des fenêtres OOS positives » | **> 50 %** des fenêtres OOS ont `return_pct > 0` |
| « 5 configurations voisines performantes » | **`\|V_perf\| ≥ 5`** **et** la médiane des scores de **`V`** est **≥ 80 %** du score du gagnant. Si `score_gagnant ≤ 0` ou non comparable → **cas alternatif** (`z ≥ 0`) ; `indeterminate` seulement si `z` non définie |
| distance normalisée `d(a,b)` | `d = (1/p) × Σ_i \|a_i − b_i\| / (max_i − min_i)` sur les `p` paramètres à **plage strictement positive**. Paramètres **catégoriels** : `0` si identique, `1` sinon. **Dimensions figées (plage nulle) : exclues** de la somme. Si `p = 0` → `d` non définie, statut `indetermine` |
| « paramètre systématiquement en borne » | sélectionné **sur la même borne** (min ou max) dans **≥ 80 %** des fenêtres où il varie (arrondi : `ceil`) |
| **« gagnant isolé »** | **`\|V_perf\| < 2`** — donc 0 ou 1 seul voisin performant. Si `score_gagnant ≤ 0` ou non comparable → **cas alternatif** (`V_perf` = `z ≥ 0`) ; `indeterminate` seulement si `z` non définie |
| **« dégradation marquée des voisins »** | la **médiane des scores de `V`** est **< 50 %** du score du gagnant. Si `score_gagnant ≤ 0` ou non comparable → **cas alternatif** (`z(médiane(V)) < −0,5`) ; `indeterminate` seulement si `z` non définie |
| **« sur ≥ 2 fenêtres »** | la conjonction (*gagnant isolé* **et** *dégradation marquée*) est vérifiée dans **≥ 2** fenêtres distinctes (comptage entier) |
| « effectif OOS suffisant » | `total_windows ≥ 5` **et** `n_trades_OOS ≥ 100` (sinon dégradé, voir tableau) |

**Table de vérité du critère de voisinage** (tous les cas ; `V_perf` et `V`
au sens du vocabulaire ci-dessus — aucune appréciation possible au-delà) :

| score du gagnant | `\|V_perf\|` | gagnant isolé | dégradation marquée | conjonction (≥ 2 fenêtres) |
|---|---|---|---|---|
| **> 0** | ≥ 2 | faux | vraie **ssi** médiane(`V`) < 50 % × score | selon les colonnes |
| **> 0** | 1 | vrai | vraie **ssi** médiane(`V`) < 50 % × score | selon les colonnes |
| **> 0** | 0 | vrai | vraie **ssi** médiane(`V`) < 50 % × score. **Uniquement si `V = ∅`** : médiane conventionnée à **0**, donc vraie | selon les colonnes |
| **≤ 0** ou non comparable | (cas alternatif) | `z ≥ 0` pour `V_perf` | `z(médiane(V)) < −0,5` | selon les colonnes |
| — | — | `z` **non définie** | `z` **non définie** | `indeterminate` → plafond **`WATCH`** |

> La **dégradation** ne dépend **que** de `médiane(`V`)`, jamais de `|V_perf|` :
> `|V_perf| = 0` ne prouve rien sur la dégradation (un `V` non vide peut avoir
> une médiane ≥ 50 % du gagnant — cas de voisins proches mais modestes). Seul
> `V = ∅` déclenche la convention « médiane = 0 ». Et la convention ne vaut que
> pour un score du gagnant **strictement positif** (ligne 3) ; dans le cas `≤ 0`,
> aucune comparaison en pourcentage du maximum n'est effectuée : on passe au
> **cas alternatif** (`z`, règle de priorité ci-dessus), et `indeterminate`
> uniquement si `z` n'est pas définie.

**Seuils de verdict** (défauts configurables `config.py: QUANT_VERDICT_*` ;
garde-fous de revue, **pas** des seuils statistiquement universels ; supposent
des fenêtres OOS comparables, des rendements **nets** et un effectif suffisant) :

| Critère | 🟢 GO | 🟠 WATCH | 🔴 NO_GO |
|---|---|---|---|
| **Intégrité** | toutes les données obligatoires cohérentes | métrique non essentielle indisponible et explicitée | appariement IS/OOS impossible, provenance des rendements/coûts inconnue, incohérence matérielle |
| **Échantillon OOS** | ≥ **5 fenêtres** et ≥ **100 trades** OOS, sans concentration excessive | **3–4 fenêtres** ou **30–99 trades** | **< 3 fenêtres** ou **< 30 trades** |
| **Érosion Sharpe** (médiane, par fenêtre) | **< 30 %** et majorité des fenêtres OOS positives | **30–50 %**, ou fenêtres mitigées | **> 50 %** **récurrente** (≥ 50 % des fenêtres), ou médiane Sharpe OOS ≤ 0 |
| **Résultat net OOS + coûts** | positif **après** scénario de coût additionnel | positif en base, nul/négatif sous stress, ou coût non testable | **≤ 0 en base**, ou coûts de base omis |
| **Robustesse de sélection** | ≥ **5 configurations voisines distinctes** (définition calculable ci-dessous), médiane des voisins ≥ **80 %** du gagnant, aucun paramètre systématiquement en borne | plateau peu peuplé ou dispersion sensible au régime | **gagnant isolé** *et* **dégradation marquée des voisins** dans **≥ 2 fenêtres** (définitions ci-dessous) |

> **Mesure alternative pour scores ≤ 0 ou non comparables** (remplace les seuils
> relatifs au maximum `≥ 0,95 × max` et `≥ 80 % du gagnant`). Soit `S`
> l'ensemble des scores **valides et finis** de la fenêtre :
>
> `z(x) = (x − médiane(S)) / (1,4826 × MAD(S))`, avec
> `MAD(S) = médiane_{s∈S} |s − médiane(S)|` (estimateur robuste de l'écart-type).
> **Repli** si `MAD(S) = 0` : `z(x) = (x − médiane(S)) / (IQR(S) / 1,349)` ;
> si `IQR(S) = 0` aussi (ou `|S| < 5`) → `z` **non définie** → statut
> `indeterminate`, plafond `WATCH`.
>
> - **`V_perf`** (cas alternatif) = élément de `V` avec **`z ≥ 0`** (score ≥
>   médiane de la fenêtre).
> - **« dégradation marquée »** (cas alternatif) = **`z(médiane(V)) < −0,5`**.
>
> Ces deux seuils (`z ≥ 0`, `z < −0,5`) remplacent respectivement `≥ 80 %` et
> `< 50 %` du cas nominal ; les comptages (`|V_perf| ≥ 5`, `< 2`, `≥ 2`
> fenêtres) sont inchangés.

**Règle d'agrégation** : défaillance d'intégrité **ou** OOS net négatif →
`NO_GO` obligatoire ; données valides mais preuves insuffisantes / critères
mitigés → `WATCH` ; `GO` exige que **tous les garde-fous applicables passent**.
Le budget consommé et le taux de doublons sont des **indices diagnostiques**,
jamais des critères de verdict seuls.

---

## 6. Spécification technique

### 6.1 Appel A2A — `services/quant_expert.py`
- Client JSON-RPC minimal (`urllib.request`, **aucune nouvelle dépendance**) vers
  `http://127.0.0.1:9924` — méthode `SendMessage`.
- **Timeout applicatif 900 s** (réponse attendue 2–6 min, jusqu'à ~9 min).
- **En cas de timeout** : déclarer explicitement le délai dépassé et **conserver
  le `context_id` pour reprise** — ne **pas** aller lire
  `~/.hermes/a2a_conversations/*.jsonl` (détail de stockage privé d'Hermes,
  susceptible de changer et de mélanger les contextes). Suivi par le **protocole
  A2A** uniquement.
- **Cache** : clé = **hash du run + `indicator_version`** (invalidé au changement
  de réglage ou de run) → pas de double appel sur rerun. « Relancer » force.
- Gestion d'erreurs explicite : pair injoignable / `input-required` / timeout /
  réponse vide ou hors schéma → message franc dans l'UI + marche à suivre.

### 6.2 Prérequis pair (à documenter dans `README` / `docs/`)
- Profil `wfo-quant` avec `platforms.a2a.enabled: true`, `extra.port: 9924`.
- **`approvals.mode: off`** obligatoire : sinon le gate d'approbation bloque les
  outils du pair en session headless (« Silence is not consent ») et l'appel
  renvoie `input-required`.
- Pair **local uniquement** (127.0.0.1, sans token) — condition du bypass.

### 6.3 Intégration UI
- **Onglet dédié « Analyse quant »** (arbitrage §11.3 = A) dans `app.py`, au même
  niveau que les onglets résultats, placé **après** `final_backtest_panel`.
- `ui/quant_analysis_panel.py` : UI en français, code/commentaires/docstrings en
  anglais, `PARAMETER_HELP` de `ui/strategy_panel.py` comme source d'aide unique.
- Thème sombre existant ; pas de nouvelle dépendance front.

### 6.4 Export
- `services/export_utils.py` : `quant_analysis.json` (**canonique**) +
  `quant_analysis.md` (rendu dérivé) + `quant_indicators.json` + `run_manifest.json`,
  réimportés si présents (replay).

---

## 7. Non-régression et tests

| Couche | Tests |
|---|---|
| `quant_indicators.py` | unitaires sur dataframes synthétiques : Q1–Q8, manifeste, contrôle d'intégrité ; cas limites (0 trade, 1 trial, tous doublons, OOS vide, score max ≤ 0, fenêtre non appariée) |
| `quant_expert.py` | mock du pair : JSON conforme / hors schéma / timeout / `input-required` / réponse vide ; validation de schéma (rejet `NaN`, `0` par défaut) |
| `check_run_integrity` | cas bloquants → aucun appel A2A émis |
| UI | smoke test : panneau affiché avec `wfo_results` factice ; bouton inactif sans run ou intégrité KO |
| Export | `quant_analysis.json` + `.md` présents dans le ZIP après analyse |

Commandes (depuis `apps/wfo_engine`, `$VBPY` = Python `vectorbtpro-run`) :
```bash
PYTHONPATH=. $VBPY -m pytest tests/test_quant_indicators.py tests/test_quant_expert.py -q
PYTHONPATH=. $VBPY -m pytest tests/ -q -p no:randomly
python -m py_compile services/quant_indicators.py services/quant_expert.py ui/quant_analysis_panel.py app.py
```

**Critère de non-régression** : la suite existante reste à `339 passed, 1 skipped`
au minimum (les tests ajoutés n'en modifient aucun).

---

## 8. Critères d'acceptation

1. Manifeste + contrôle d'intégrité affichés ; en cas d'échec → « non évaluable
   + cause » et **aucun appel A2A**.
2. Q1–Q8 affichés sans aucun appel réseau, reproductibles (2 affichages
   identiques) ; toute métrique non calculable est `« indisponible »` — jamais `0`.
3. Bouton inactif sans run ou intégrité KO, actif sinon.
4. Un clic produit un **JSON conforme à `quant_analysis.v1`** affiché en
   tableaux + verdict, en < 15 min, `context_id` visible.
5. Pré-verdict local affiché à côté du verdict du pair ; tout **désaccord**
   signalé sans modification des faits.
6. Pair injoignable / timeout / réponse hors schéma → erreur explicite **sans
   planter**, `context_id` conservé pour reprise.
7. `quant_analysis.json` (canonique) + `.md` figurent dans le ZIP exporté.
8. Suite de tests verte (voir §7).

---

## 9. Risques et points d'attention

| Risque | Impact | Parade |
|---|---|---|
| Pair lent (2–9 min) | UX d'attente | `st.status` + cache + `context_id` conservé pour reprise |
| Gate d'approbation non désactivé | analyse impossible (`input-required`) | prérequis §6.2 + détection au démarrage du panneau |
| Réponse LLM non déterministe | interprétations variables | faits scellés localement + pré-verdict + `evidence_refs` vérifiées |
| Double soustraction des coûts | edge erroné | Q8.1 : convention du portefeuille établie avant tout calcul |
| Fallback de provenance différente (Q7) | Sharpe non comparable | séries par barre exigées, sinon « indisponible » |
| Donnée absente lue comme 0 | badge vert trompeur | §5.3 : `null` + `raison`, plafond `WATCH` |
| `wfo_results` incomplet | plantage | contrôle d'intégrité bloquant + fonctions pures |
| Le pair « invente » des nombres | avis erroné | prompt = faits déjà calculés + `evidence_refs` vérifiées par l'app |
| Lecture de stockage privé Hermes | fragilité, mélange de contextes | §6.1 : protocole A2A uniquement, jamais le JSONL |

---

## 10. Découpage de réalisation

| Tâche | Contenu | Dépend de |
|---|---|---|
| T1 | `services/quant_indicators.py` : manifeste, `check_run_integrity`, Q1–Q8, pré-verdict + tests | — |
| T2 | `services/quant_expert.py` : client A2A, schéma `quant_analysis.v1` + validateur, cache + tests | — |
| T3 | `ui/quant_analysis_panel.py` : §5.0 + §5.1 (calculs, pré-verdict) | T1 |
| T4 | §5.2 : bouton + rendu structuré + hook d'auto-analyse (`auto_analyze_run`) | T2, T3 |
| T5 | Intégration `app.py` (onglet) + **déclenchement de l'auto-analyse après le backtest final** + export `export_utils.py` | T3, T4 |
| T6 | Doc (`README`, `CHANGELOG`) + prérequis pair | T5 |

Ordre **T1 → T2 → T3 → T4 → T5 → T6**, avec revue `wfo-reviewer` à la fin de
chaque lot puis push sur accord.

**Report explicitement accepté (revue T4)** — le déclenchement *effectif* de
l'analyse automatique est déplacé de T4 vers T5 : §5.0 exige `final_trades`
comme preuve, et ceux-ci ne sont produits qu'**après le backtest final**
(sélection L2 + `final_backtest_panel`), pas en fin de run WFO. T4 livre le hook
complet et transmet toutes les preuves disponibles ; sans `final_trades` il
renvoie un `non_evaluable` explicite **sans appel A2A** (§5.3 cas a). Aucun
backtest final implicite n'est lancé dans `run_optimization_job` (décision de
revue).

---

## 11. Arbitrages et état de validation

### 11.1 Arbitrages — tranchés le 2026-10-08

| # | Question | Choix |
|---|---|---|
| 1 | Format de l'interprétation | **B — tableau structuré + verdict codé** (JSON `quant_analysis.v1` rendu en tableaux + badge GO/WATCH/NO_GO) |
| 2 | Déclenchement | **B — bouton + analyse automatique optionnelle** (checkbox défaut coché, hook de fin de run) |
| 3 | Emplacement | **A — onglet dédié « Analyse quant »** (après `final_backtest_panel`) |

### 11.2 Revue `wfo-quant` (2026-10-08) — **prise en compte en v2**

| # | Remarque de `wfo-quant` | Traitement en v2 |
|---|---|---|
| B1 | Q8 peut soustraire les coûts deux fois | §5.1 Q8 : **convention du portefeuille** exigée avant tout calcul ; pas de « seuil de ruine » |
| B2 | Q7 non réalisable avec les entrées annoncées | §5.1 Q7 : **séries de rendements par barre** exigées, sinon « indisponible », sans fallback |
| B3 | IS/OOS et « top ≥95 % » mal définis | §5.1 Q6 : **appariement par fenêtre/paramètres** ; Final ≠ 3ᵉ fenêtre ; §5.3 : alternative normalisée si score ≤ 0 |
| B4 | Aucune politique pour données manquantes | §5.3 : `null` + `raison`, contrôle bloquant, plafond `WATCH` |
| I1 | §4 Markdown vs §5.2 JSON incohérents | §4 + §6.4 : **JSON canonique**, Markdown dérivé |
| I2 | Schéma trop pauvre / champs dupliqués | §5.2 : `schema_version`, `run_id`, `input_digest`, `confidence`, `verdict_scope`, `findings[]` + `evidence_refs`, `couts`, statuts à la place des booléens |
| I3 | Lire le JSONL = stockage privé | §6.1 : **protocole A2A uniquement**, `context_id` conservé pour reprise |
| I4 | Cache/export liés au run | §5.1 règles + §6.1 : clé = hash du run + `indicator_version` |

### 11.3 Revue v2 par `wfo-quant` (2026-10-08) — **NON VALIDE, 3 points traités en v2.1**

| # | Point bloquant restant | Traitement en v2.1 |
|---|---|---|
| B1 | `diagnostics` redemandait au pair des chiffres déjà scellés | `diagnostics` retiré de la sortie = bloc **entrée** scellé par l'app ; **source de vérité et contrôle d'égalité** définis (`evidence_refs` vérifiées, alerte de désaccord) |
| B2 | « non évaluable » vs `NO_GO` contradictoires | §5.3 : 2 cas distincts (intégrité KO → pré-verdict local `NO_GO` + `non_evaluable`, **sans** appel A2A ; métrique partielle → plafond `WATCH`) |
| B3 | critères non calculables | §5.3 : **définitions calculables** (concentration, érosion récurrente, voisins, borne, effectif) |

### 11.4 Revue v2.1 par `wfo-quant` (2026-10-08) — **NON VALIDÉE, 2 points traités en v2.2**

| # | Point bloquant restant | Traitement en v2.2 |
|---|---|---|
| B1 | `W3` sans identifiant de fenêtre défini ; égalité des chiffres en texte libre non vérifiable | §5.2 : **nomenclature fermée** d'identifiants (`Q<n>`, `W<n>`, `M.<champ>`, `P.<param>`, `T<n>`) + **citation des chiffres uniquement sous `{{ID.champ}}`** (substitution + parsing, pas de comparaison flottante) |
| B3 | concentration non définie si P&L ≤ 0 ; arrondi des 10 %, distance des voisins, scores non positifs à spécifier | §5.3 : `k = max(1, ceil(0,10×n))`, cas `P&L_total ≤ 0` → `indetermine`, **formule exacte de `d(a,b)`** (catégorielles, dimensions figées exclues), cas score ≤ 0 → `indeterminate` + plafond `WATCH` |

### 11.5 Revue v2.2 par `wfo-quant` (2026-10-08) — **1 point traité en v2.3**

| # | Point bloquant restant | Traitement en v2.3 |
|---|---|---|
| B3 | `NO_GO` « gagnant isolé et dégradation marquée » non calculable | §5.3 : **« gagnant isolé » = < 2 voisins admissibles** ; **« dégradation marquée » = médiane des voisins < 50 %** du gagnant ; **« ≥ 2 fenêtres » = conjonction vérifiée dans ≥ 2 fenêtres** |

→ Schéma + nomenclature + règle `{{ID.champ}}` : **validés** par `wfo-quant`.

### 11.12 Revue v2.9 par `wfo-quant` (2026-10-08) — **1 point traité en v2.10**

| # | Point bloquant restant | Traitement en v2.10 |
|---|---|---|
| B3 | une note affirmait encore `indeterminate` dès `score ≤ 0`, contradictoire avec la règle de priorité | §5.3 : note harmonisée — `≤ 0` → **cas alternatif `z`** ; `indeterminate` **uniquement si `z` non définie** |

### 11.13 État final — ✅ **VALIDÉE le 2026-10-08**

> Verdict de `wfo-quant` : *« VALIDÉE — la note du §5.3 respecte désormais la
> règle de priorité : `score ≤ 0` → cas alternatif `z` ; `indeterminate`
> seulement si `z` n'est pas définie. Définitions calculables entièrement
> définies : oui. »*

- [x] Schéma `quant_analysis.v1` + nomenclature + règle `{{ID.champ}}` — **validés**
- [x] Seuils + définitions calculables (§5.3) — **validés**
- [x] Validation finale — **obtenue** → **lancement de T1**
