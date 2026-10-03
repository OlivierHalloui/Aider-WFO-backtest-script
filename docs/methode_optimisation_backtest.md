# Méthodologie d'optimisation & de backtest — WFO Engine

> Document destiné à un lecteur **non spécialiste** (ou à un LLM non spécialisé).
> Aucun prérequis en mathématiques ou en finance n'est supposé. Chaque terme
> technique est défini dans le [glossaire](#7-glossaire).

---

## 1. Les bases : qu'est-ce qu'un backtest ?

Un **backtest** = rejouer une stratégie de trading sur des données historiques
pour voir ce qu'elle aurait gagné ou perdu.

Concrètement, dans le WFO Engine :
- on charge un historique de prix (par exemple BTC/USDT en bougies de 5 secondes) ;
- la stratégie **ATDMF** génère des signaux d'achat/vente à partir d'indicateurs
  (Bollinger Bands, SAR, MACD, moyennes mobiles) ;
- on simule les ordres et on mesure le résultat (score de performance).

Une seule passe de backtest sur une fenêtre de 17 259 barres prend du temps de
calcul. C'est ce qu'on appelle une **évaluation** (= un backtest complet).

> **À retenir** : une « évaluation » = un backtest. Le calcul est lent, donc le
> nombre de backtests qu'on peut lancer est limité. C'est le nerf de la guerre.

---

## 2. Le problème : trouver les bons paramètres

La stratégie a beaucoup de **paramètres réglables** (longueurs de fenêtres,
seuils, etc.). Leur valeur change les signaux, donc le gain. Exemples de
paramètres : `timeperiod` (21 valeurs possibles), `StDev` (23 valeurs),
`user_exit_sma_length` (25 valeurs)…

On cherche la **combinaison de paramètres** qui maximise le score.

### La difficulté : le nombre de combinaisons explose

Si 11 paramètres ont chacun une vingtaine de valeurs, le nombre de combinaisons
est le produit de leurs cardinalités. Dans un cas réel du projet :

```
21 × 23 × 7 × 8 × 19 × 31 × 16 × 8 × 5 × 5 × 25
= 1 274 501 760 000 combinaisons   (≈ 1,27 × 10¹²)
```

Tester les 1 274 milliards de combinaisons est **impossible** : même à 1 seconde
par backtest, cela prendrait ~40 000 ans.

> **Le paradoxe** : l'espace de recherche est gigantesque, mais le budget
> d'évaluations est minuscule (quelques dizaines à quelques centaines de
> backtests). Tout l'art consiste à choisir **judicieusement** les quelques
> combinaisons à tester.

---

## 3. Le budget d'évaluation

Le **budget d'évaluation** = le nombre maximal de backtests qu'on s'autorise.

- Dans un config, c'est le champ `max_trials` (ex. `max_trials = 50`).
- Plus le backtest est long, plus le budget est serré.
- La qualité de l'optimisation dépend du rapport **budget / nombre de paramètres**.

**Règle de pouce** : prévoir **au moins ~10 évaluations par paramètre optimisé**.
Pour 11 paramètres → au moins **110-150 évaluations**. Un budget de 50 pour
11 paramètres est trop faible : le modèle n'a pas assez de points pour apprendre.

---

## 4. Les méthodes d'optimisation (du simple au sophistiqué)

Toutes cherchent à maximiser le score, mais explorent différemment.

### 4.1 Grid Search (énumération)
- **Principe** : tester **toutes** les combinaisons d'une grille.
- **Quand** : grille réduite (quelques milliers de combinaisons max).
- **Avantage** : exhaustif, reproductible, aucune hypothèse.
- **Limite** : impossible au-delà de ~10 000 combinaisons (trop lent).

### 4.2 Bayesian Optimization (BO) — méthode « bayesian »
- **Principe** : construire un **modèle prédictif** (un *surrogate*, souvent un
  Processus Gaussien = GP) qui estime « où se cachent les bons scores », puis
  tester le point le plus prometteur. On alterne **exploration** (zones inconnues)
  et **exploitation** (meilleures zones connues).
- **Analogie** : chercher le sommet d'une montagne dans le brouillard — on palpe
  le terrain, on modélise la forme, et on monte là où ça semble haut.
- **Quand** : budget limité (50-500), dimension faible à modérée (≤ ~15).
- **Limite** : le GP devient inefficace au-delà de ~20 dimensions.

### 4.3 Optuna (TPE) — méthode « optuna »
- **Principe** : framework d'optimisation utilisant le **TPE** (Tree-structured
  Parzen Estimator), un modèle bayésien plus simple qu'un GP. Bon pour les
  espaces mixtes (nombres + booléens).
- **Quand** : bon défaut général, surtout avec des drapeaux on/off.
- **À noter** : Optuna est un **cadre**, pas une méthode unique (il embarque
  TPE, CMA-ES, GP…).

### 4.4 TuRBO (Trust-Region BO) — méthode « turbo »
- **Principe** : BO **locale**. On découpe l'espace en petites **régions de
  confiance** (boîtes autour des bons points trouvés) et on optimise dedans. Si
  ça progresse, la boîte **grandit** ; si ça stagne, elle **rétrécit**. Quand une
  boîte s'effondre, on **redémarre** ailleurs.
- **Analogie** : chercher un trésor en quadrillant de petits carrés autour des
  indices prometteurs, plutôt que tout fouiller.
- **Quand** : **5 à 20 paramètres numériques**, budget de quelques centaines
  d'évaluations. **Excellent choix pour le WFO Engine.**
- **Référence** : Eriksson et al., *Scalable Global Optimization via Local
  Bayesian Optimization*, NeurIPS 2019.

### 4.5 BADS (Bayesian Adaptive Direct Search) — méthode « bads »
- **Principe** : combine une **recherche directe sur un maillage** (on teste les
  voisins du point courant, comme un pas d'escalier) et un **GP** qui classe les
  voisins pour n'évaluer d'abord que les plus prometteux. Le maillage s'affine
  quand on stagne.
- **Analogie** : avancer sur un terrain en escaliers, en consultant une carte
  (le GP) pour savoir quel pas tenter en premier.
- **Quand** : objectif **irrégulier / en marches d'escalier** (comme un score de
  backtest qui saute avec les trades), 5-20 dimensions. **Très bon pour le WFO.**
- **Référence** : Acerbi & Ma, *Practical Bayesian Optimization for Model
  Fitting with BADS*, NeurIPS 2017.

### Tableau récapitulatif

| Méthode | Budget typique | Dimensions | Idéal quand… |
|---|---|---|---|
| Grid | ~10³-10⁴ | faible | grille petite, exhaustivité requise |
| Bayesian (GP) | 50-500 | ≤ 15 | budget serré, surface lisse |
| Optuna (TPE) | 50-500 | ≤ 30 | espaces mixtes (booléens + nombres) |
| **TuRBO** | 100-400 | **5-20** | budget limité, dimension modérée |
| **BADS** | 100-400 | **5-20** | score en escalier, optimum local net |

---

## 5. Comment régler un run (mode d'emploi)

1. **Choisir la méthode** :
   - grille réduite → `grid` ;
   - ≤ 15 paramètres numériques, budget limité → **`turbo`** ou **`bads`** ;
   - beaucoup de drapeaux on/off → `optuna`.
2. **Régler `max_trials`** (le budget) : ≥ 10 × le nombre de paramètres optimisés.
3. **Fixer `random_state`** (graine aléatoire) pour être reproductible.
4. **Lancer le WFO** (`n_windows` fenêtres train/test chronologiques).
5. **Analyser** les résultats avec les garde-fous (section 6) — **jamais** le
   seul meilleur score in-sample.

Dans le WFO Engine, la méthode se règle via `optimization_method` :
`grid`, `bayesian`, `optuna`, `turbo`, ou `bads`.

---

## 6. Les garde-fous : ne pas se tromper soi-même

Optimiser beaucoup de paramètres **à un risque majeur** : le
**sur-apprentissage** (*overfitting*) = trouver une combinaison qui brille sur les
données passées mais échoue en vrai. C'est LE piège de l'optimisation financière.

Les garde-fous à appliquer **toujours** :

### 6.1 WFO chronologique + holdout final intact
- On optimise sur une période (**in-sample / IS**), on teste sur la suivante
  (**out-of-sample / OOS**), jamais l'inverse.
- Conserver une **période de test finale** (holdout) qui ne sert **jamais** à
  choisir la méthode, les plages ni le candidat — juste à la toute fin.

### 6.2 Plateau paramétrique (pas un pic isolé)
- Préférer un candidat dont les **voisins** sont aussi performants (un
  **plateau**) à un maximum isolé (un **pic**, souvent du bruit).
- Vérifier la **stabilité des paramètres** d'une fenêtre à l'autre.

### 6.3 PBO / CSCV — Probabilité de Sur-apprentissage
- Mesure la probabilité que le « gagnant » in-sample soit perdant OOS.
- Conserver les performances de **tous** les candidats essayés (pas seulement
  le vainqueur).
- Réf. : Bailey, Borwein, López de Prado, Zhu, *The Probability of Backtest
  Overfitting*, J. Computational Finance, 2017.

### 6.4 CPCV — Validation Croisée Purge & Combinatoire
- Produit **plusieurs chemins** de validation OOS pour voir la **distribution**
  des résultats (pas une seule trajectoire favorable).
- Avec *purge* et *embargo* pour éviter la fuite d'information temporelle.
- Réf. : López de Prado, *Advances in Financial Machine Learning*, 2018.

### 6.5 Deflated Sharpe Ratio (DSR)
- Corrige le ratio de Sharpe sélectionné pour le **biais de sélection** (nombre
  d'essais multiples) et la **non-normalité** des rendements.
- Journaliser **tous** les essais (seeds, changements de plages).
- Réf. : Bailey & López de Prado, *The Deflated Sharpe Ratio*, J. Portfolio
  Management, 2014.

### Critère de décision final
Retenir la méthode qui améliore la **distribution des performances OOS
chronologiques, nettes de coûts** (frais, slippage) — **pas** le meilleur score
in-sample.

---

## 7. Glossaire

| Terme | Définition simple |
|---|---|
| **Backtest** | Simulation d'une stratégie sur l'historique des prix. |
| **Évaluation** | Un backtest complet d'une combinaison de paramètres. |
| **Budget d'évaluation** | Nombre maximal de backtests autorisés. |
| **Paramètre optimisable** | Réglage avec plusieurs valeurs possibles à tester. |
| **In-sample (IS)** | Période d'entraînement (on optimise dessus). |
| **Out-of-sample (OOS)** | Période de test (on valide dessus, jamais optimisée). |
| **Walk-Forward (WFO)** | Enchaînement de fenêtres IS→OOS glissantes dans le temps. |
| **Overfitting / sur-apprentissage** | Avoir mémorisé le bruit passé au lieu de la vraie dynamique. |
| **Surrogate / GP** | Modèle prédictif qui remplace le backtest coûteux (Processus Gaussien). |
| **Trust region** | Petite zone locale autour d'un bon point, où on affine la recherche. |
| **Acquisition (EI)** | Règle pour choisir le prochain point à tester (Expected Improvement). |
| **Grid Search** | Tester exhaustivement toutes les combinaisons d'une grille. |
| **TuRBO** | BO locale par régions de confiance (Eriksson 2019). |
| **BADS** | Recherche directe sur maillage guidée par un GP (Acerbi & Ma 2017). |
| **PBO / CSCV** | Probabilité de sur-apprentissage / validation croisée symétrique combinatoire. |
| **CPCV** | Validation croisée purgée et combinatoire (chemins OOS multiples). |
| **DSR** | Sharpe corrigé du biais de sélection et de la non-normalité. |
| **Plateau** | Zone où plusieurs voisins sont bons (plus robuste qu'un pic isolé). |

---

## 8. En résumé (pour un non-spécialiste)

1. Un **backtest** = rejouer la stratégie sur l'historique → c'est lent.
2. Trop de **paramètres** × trop de valeurs = combinaisons impossibles à tout tester.
3. Le **budget** (nombre de backtests) est la vraie limite.
4. Les méthodes **TuRBO** et **BADS** sont les mieux adaptées quand on a
   ~5-20 paramètres et un budget limité : elles explorent intelligemment au lieu
   de tout balayer.
5. **Toujours** valider avec les garde-fous (WFO chronologique, plateau, PBO,
   CPCV, DSR) — le meilleur score passé est presque toujours un leurre.
