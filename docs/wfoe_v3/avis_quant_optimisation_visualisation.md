# Avis experts — optimisation des paramètres et visualisation des backtests

Date : 2026-09-26. Source : les 5 pairs A2A (wfo-quant, wfo-reviewer, wfo-implementer, wfo-tester, wfo-data), sollicités via a2a\_call/a2a\_orchestrate. Synthèse et croisement avec l'état du projet : orchestrateur wfo-engine.

## Objet

Évaluation de la pertinence des techniques d'optimisation/sélection de paramètres et des outils de visualisation du moteur WFO (app Streamlit, VectorBT Pro, stratégie ATDMF crypto, walk-forward ancré ou non).

## Guide de lecture (lecteur non spécialiste du quant)

L'idée centrale du moteur : pour éviter de se leurrer avec une stratégie qui « marche sur le passé mais pas en réel », on découpe l'historique des prix en fenêtres successives. Sur chaque fenêtre, on choisit les paramètres en ne regardant QUE la partie passée (IS, l'« entraînement »), puis on mesure la performance sur la partie future (OOS, le « test ») qu'on n'avait pas utilisée pour choisir. En enchaînant ces fenêtres, on simule ce qu'on aurait réellement gagné/perdu en production.

Tout l'enjeu des avis ci-dessous est de garantir que cette simulation est honnête : qu'aucune information future ne « fuit » dans une décision passée, que les résultats sont reproductibles et nets des frais, et que les graphiques montrent la vraie performance plutôt qu'une sélection avantageuse. Les termes spécialisés (IS, OOS, fuite, lookahead, Sharpe, drawdown…) sont définis dans le Lexique en fin de document.

## Contexte du projet (état actuel)

Optimisation : grid, bayesian (skopt), optuna. Régimes : classic, prev\_best\_grid, nn\_guided (NeuralSearchGuide), adaptive\_continuous (rolling + narrowing progressif).

Sélection à 3 niveaux :

- L1 intra-fenêtre : snv (voisinage KDTree), svi (top-K ré-évalué sur IS₂), raw\_max.

- L2 cross-fenêtres : best\_is\_oos, best\_oos, robust\_set, weighted\_oos.

- L3 consensus stagewise : median\_mode, best\_window, weighted\_oos\_median.

Anti-overfitting : métrique PQS (√(n\_trades/n\_ref)), neighbor smoothing (SNV).

Visualisation : visualize\_wfo\_results (courbes IS/OOS), create\_parameter\_performance\_map (cartes de paramètres), visualize\_robustness\_metrics, rapport PDF.

## Partie 1 — Optimisation et sélection (wfo-quant) : « pertinent, mais insuffisant »

### Points critiques (à faire)

1. Supprimer la sélection sur l'OOS évalué. `best\_oos` et `weighted\_oos` (niveau L2) créent une fuite si la performance OOS sert à la fois à choisir les paramètres et à annoncer la performance finale. Cible : L1 sur IS/IS₂, L2–L3 sur des fenêtres historiques déjà closes, puis évaluation sur une période future jamais consultée. Le holdout final reste scellé jusqu'au gel de la méthode.

2. Auditer la chronologie. Purger les observations dont les labels ou positions débordent entre IS, IS₂ et OOS ; ajouter un embargo adapté à l'horizon des trades. Agréger les rendements OOS en ordre temporel, sans double comptage des périodes chevauchantes. Le CPCV (Combinatorial Purged Cross-Validation) est utile pour estimer la dispersion, en complément, non en remplacement du walk-forward chronologique et du holdout.

3. Mesurer la stratégie négociable. Optimiser et rapporter des résultats nets de commissions, spread, slippage, financement et contraintes de liquidité propres à la crypto ; re-tester sous coûts dégradés. Journaliser graines, données, découpages, espace de recherche et nombre total d'essais (y compris variantes de stratégie, sélecteurs L1–L3 et réglages manuels). Fixer le budget d'essais avant l'expérience.

4. Ne pas confondre stabilité et preuve statistique. SNV favorise un plateau local, or un plateau peut être dû à des régimes corrélés. PQS ∝ √n\_trades pénalise un petit échantillon sans corriger le biais de sélection multiple. Rapporter la distribution des performances OOS par fenêtre, le nombre de trades et une mesure corrigée des essais : le Deflated Sharpe Ratio (Bailey & López de Prado, The Deflated Sharpe Ratio, 2014), avec ses hypothèses et un nombre d'essais effectif documenté.

### Points optionnels

- Estimer la probabilité de surapprentissage par CSCV/PBO.

- Comparer grid, Optuna, NN et adaptation continue à budget d'évaluations identique.

- Vérifier qu'aucune information postérieure à chaque décision n'entre dans le rétrécissement de l'espace (régime adaptatif).

## Partie 2 — Visualisation (wfo-quant) : « utile pour explorer, pas suffisante pour valider »

### Points critiques (à faire)

1. Vue temporelle complète : courbe de capital OOS nette recomposée, benchmark crypto pertinent et stratégie passive aux mêmes coûts, drawdown en dessous, frontières IS/OOS, changements de paramètres et périodes sans position visibles. Afficher rendement, volatilité, drawdown maximal, exposition et turnover (pas seulement une courbe IS/OOS).

2. Rendre visibles les échecs cachés par l'agrégat : tableaux et graphiques par fenêtre (rendement net, drawdown, trades, frais, exposition), distribution des P&L de trades, durée de détention et contribution des plus gros gains/pertes. Distinguer explicitement trade, fenêtre et observation temporelle : ils ne sont pas indépendants.

3. Carte de robustesse exploitable : performance OOS nette sur des coupes de paramètres voisins, avec effectifs et incertitude ; signaler les cellules peu échantillonnées et les pics isolés. Les cartes actuelles ne prouvent rien si elles montrent surtout l'IS ou si l'échelle de couleurs masque les pertes.

### Points optionnels

- Profil de risque et stress : rendements mensuels, queues de distribution, drawdowns par régime, sensibilité aux coûts et à un décalage d'exécution ; comparaison aux benchmarks et aux variantes simples de la stratégie.

- Le PDF doit exposer ces diagnostics et les limites, plutôt que reproduire seulement des graphiques favorables.

## Partie 3 — Avis croisés des experts par domaine

### 3.1 wfo-reviewer (qualité de code, bugs, fuites, conventions)

1. Fuite de sélection L1/L3 par imbrication : le consensus L3 peut être calibré sur les fenêtres OOS successives (seuil, poids des niveaux). Tout méta-paramètre choisi après observation des scores OOS est une fuite implicite → verrouiller a priori ou optimiser en walk-forward supérieur.

2. Fuite de prétraitement : normalisation SNV ajustée uniquement sur IS ; warm-up des indicateurs sans débordement IS/OOS (fit sur série complète = fuite).

3. Biais de sélection du rapport : le PDF montre le meilleur run sans correction de multiplicité ; tracer le nombre d'essais total et interdire la re-sélection depuis l'UI.

4. Reproductibilité : Optuna non sérialisé, n\_jobs\>1 non déterministe → seed global + sampler + export study (.sqlite/JSON) dans la traçabilité.

5. Divergence de définitions métriques hors metrics.py (annualisation 365 vs 252, Sharpe sans rf) → centraliser + golden test.

6. Silence des erreurs : pas d'except nu, fenêtres échouées marquées (sinon biais de survie).

7. Complétude des fenêtres WFO : tracer le taux de fenêtres rejetées, l'afficher dans le PDF.

8. Dette : wfo.py monolithique (~53 Ko) ; clés de cache Streamlit incluant toute la config de split/seed.

9. Anti-overfitting PQS/SNV : appliqués dans le pipeline de sélection, pas en post-processing UI (verrouiller par test d'intégration).

10. Transpilation Pine : code généré idempotent et versionné (hash Pine source ↔ Python généré) dans la traçabilité.

11. Tests à prioriser : (a) test canari anti-lookahead, (b) golden tests métriques, (c) test reproductibilité bit-identique.

12. Dépendances figées (numpy 1.23.5, numba 0.56.4) : éviter les API numpy 2.x ; figer VBT/optuna dans la traçabilité.

Top 3 : 2 (fuite de prétraitement), 4 (reproductibilité), 11a (test canari).

### 3.2 wfo-implementer (faisabilité, performance, priorisation)

1. Priorité 1 — purge/embargo aux frontières IS→IS₂→OOS, définie depuis l'intervalle réel des trades ; geler la sélection avant l'OOS ; tester frontières + fenêtres ancrées/non ancrées.

2. Priorité 2 — equity OOS nette + benchmark : segments OOS assemblés chronologiquement, sans remise à zéro artificielle ; frais/slippage/financement inclus ; benchmark aligné. C'est le contrôle visuel le plus utile.

3. Priorité 3 — Deflated Sharpe Ratio : calcul peu coûteux, mais le piège est le nombre d'essais (compter aussi Bayesian/Optuna + variantes de sélection).

4. Priorité 4 — diagnostics : distribution des trades (durée, rendement net, concentration) avant heatmap ; heatmap sur les points réellement évalués, pas une grille interpolée.

5. Approche performance : calculer purge/rendements nets une fois côté moteur (tableaux typés, numba seulement après profilage), transmettre des résultats compacts à Streamlit/Plotly ; cache par config+données+seed. Faisabilité élevée.

### 3.3 wfo-tester (tests, reproductibilité, parité)

1. Fuite L2/L3 — le risque n°1 à tester : test canari de causalité temporelle (perturber une fenêtre future, vérifier que les sélections passées restent bit-identiques) ; assertion de timestamp (`max(ts\_source) \< début\_fenêtre`) instrumentée ; test synthétique à divergence contrôlée ; test L3 « chaque étage ne voit que la matrice tronquée à son horizon ».

2. Métriques nettes : golden test (PnL calculé à la main, comparaison 1e-9) ; property-based `net ≤ gross` ; sensibilité au turnover ; snapshot de régression (syrupy).

3. Visualisation : tester le pipeline de données, pas les pixels — la série passée au renderer doit être la concaténation stricte des segments OOS (sans IS intercalé) ; cross-check recomposer les métriques depuis la série affichée ; vérifier l'alignement marqueur/params sélectionnés.

4. Reproductibilité : test run-twice bit-identique ; manifest (hash data + git SHA + config) ; ordre du grid trié.

5. Parité Pine : tolérances séparées indicateurs vs résultats de stratégie ; catalogue des divergences connues en xfail documentés.

Priorité unique : le test canari de causalité (point 1).

### 3.4 wfo-data (données, lookahead, provenance)

1. Lookahead MTF (`request.security`/`barmerge.lookahead\_off`) : risque n°1 ; exiger une sémantique lookahead\_off stricte ; vérifier que l'implémentation historique = temps réel.

2. Convention d'horodatage : ancrage unique (open-time vs close-time, label left/right) ; un décalage d'une barre = lookahead/lag silencieux.

3. Alignement des séries : jointure sur index strict (pas de merge\_asof/ffill qui fait fuiter la barre suivante) ; ffill arrêté au dernier point confirmé.

4. Barres dupliquées/manquantes : dédoublonner, détecter les trous, gérer les NaN OHLCV ; gap-fill non tradé.

5. Warm-up des indicateurs : exclure les N premières barres NaN ; scaling appris sur IS uniquement.

6. Prix d'exécution : entrée au open de la barre suivante, pas à la close de la barre de signal.

7. Isolation stricte IS/OOS : split sur l'horodatage, jamais sur l'index relatif.

8. Provenance : versionner la source + hasher les données ; golden file (seed + hash).

Priorités : (1) lookahead\_off, (3) alignement des séries, (6) prix d'exécution. Le point (2) (convention d'horodatage) mérite un test de non-régression dédié.

## Partie 4 — Synthèse transversale et convergences

Thème 1 — Fuite / lookahead (convergence maximale, les 5 experts) :

- L2 best\_oos/weighted\_oos = fuite structurelle (quant, tester, reviewer).

- Test canari de causalité temporelle (tester) : le seul qui détecte une fuite structurelle indépendamment de l'implémentation.

- Fuite de prétraitement (reviewer) : SNV/warm-up ajustés sur IS uniquement.

- Lookahead MTF strict + prix d'exécution à la barre suivante (data).

- Fuite méta-paramètres L3 (reviewer).

- Purge/embargo définie depuis l'horizon réel des trades (quant, implementer).

Thème 2 — Reproductibilité : seed global + sampler + n\_jobs=1 + export study (reviewer, tester) ; manifest hash data + git SHA + config (tester, data) ; test run-twice et ordre de grid trié (tester).

Thème 3 — Métriques nettes / négociables : net de commission/spread/slippage/funding/liquidité (quant, data, implementer) ; golden test + property-based + sensibilité turnover (tester).

Thème 4 — Cohérence des métriques : source unique (metrics.py), annualisation 365 vs 252, Sharpe sans rf ; cross-check depuis la série affichée (reviewer, tester).

Thème 5 — Visualisation : equity OOS nette + benchmark + drawdown (quant, implementer) ; tester le pipeline de données plutôt que les pixels (tester) ; distribution des trades avant heatmap, points réellement évalués (implementer).

Thème 6 — Échecs silencieux / complétude : pas d'except nu, fenêtres échouées tracées (reviewer) ; barres dupliquées/manquantes, gap-fill non tradé (data).

Thème 7 — Dette / conventions : wfo.py monolithique, clés de cache, hash Pine source ↔ Python généré (reviewer).

## Croisement avec l'état du projet

Déjà présent : grid/bayesian/optuna ; 3 niveaux de sélection ; PQS ; neighbor smoothing ; visualize\_wfo\_results ; parameter maps ; robustness ; PDF.

Manquant : Deflated Sharpe ; purge/embargo ; holdout final scellé ; métriques nettes complètes ; equity OOS nette + benchmark ; distribution des trades ; heatmap de robustesse ; test canari anti-lookahead ; reproductibilité garantie (seed/study export) ; prix d'exécution à la barre suivante ; convention d'horodatage unique ; golden files de données.

Points de convergence avec les audits précédents :

- La fuite best\_oos/weighted\_oos est un angle mort réel du projet.

- L'IS₂ (SVI) n'est pas un holdout (terminologie trompeuse).

- Le bug de fréquence MTF (resample '15m' = mois) a été corrigé et durci séparément (commits 1f8e035 et 3c5e066) — mais le durcissement \_to\_pandas\_freq est lui-même un point d'entrée possible de fuite (cf. avis data, lookahead\_off).

## Plan d'action priorisé (proposition, non engagé)

Priorité 1 (critique, fuite/reproductibilité) :

1. Test canari de causalité temporelle (fuite L2/L3) — wfo-tester (test) + wfo-implementer (instrumentation).

2. Corriger la fuite OOS en L2 (best\_oos/weighted\_oos) — wfo-quant (méthodo) + wfo-implementer (code).

3. Lookahead\_off strict + prix d'exécution à la barre suivante + alignement des séries — wfo-data.

4. Purge/embargo aux frontières IS/IS₂/OOS — wfo-quant + wfo-data.

5. Reproductibilité (seed global + export study + manifest) — wfo-implementer + wfo-tester.

6. Métriques nettes complètes + golden tests — wfo-implementer + wfo-tester.

Priorité 2 (critique, visualisation) : 7. Equity OOS nette recomposée + benchmark + drawdown — wfo-implementer. 8. Tableaux par fenêtre + distribution des trades — wfo-implementer. 9. Heatmap de robustesse (points réellement évalués) — wfo-implementer. 10. Deflated Sharpe Ratio + distribution OOS par fenêtre — wfo-quant.

Priorité 3 (optionnel) : CSCV/PBO ; comparaison à budget égal ; profil de risque/stress ; enrichissement du PDF ; découpage de wfo.py ; hash Pine source.

## Lexique et acronymes

### Cadre walk-forward

- WFO (Walk-Forward Optimization) : méthode d'évaluation qui découpe l'historique en fenêtres successives ; sur chacune, on optimise sur le passé puis on teste sur le futur, en avançant d'un pas.

- IS (In-Sample, « en échantillon ») : la partie passée des données, utilisée pour choisir/optimiser les paramètres. Équivalent de l'« entraînement » en machine learning.

- OOS (Out-Of-Sample, « hors échantillon ») : la partie future, non utilisée pour choisir, sur laquelle on mesure la performance. Équivalent du « test ».

- IS₂ : sous-découpage de l'IS utilisé pour ré-évaluer les meilleurs candidats. Ce n'est PAS un vrai test (pas de futur), d'où la remarque « l'IS₂ n'est pas un holdout ».

- Holdout : jeu de données final mis de côté, jamais consulté pendant le développement, utilisé une seule fois à la fin pour la mesure finale.

- Fenêtre ancrée / non ancrée : fenêtre qui garde toujours le même point de départ (ancrée) ou qui glisse entièrement (non ancrée).

### Surapprentissage, fuites et leurs remèdes

- Surapprentissage (overfitting) : les paramètres « apprennent » le bruit du passé et ne généralisent pas au futur.

- Fuite (leakage) : information future utilisée par erreur dans une décision passée → performance gonflée à tort.

- Lookahead : cas particulier de fuite où une valeur future est connue trop tôt (ex. utiliser la clôture d'une barre avant qu'elle ne soit clôturée).

- Purge : retirer les observations dont la période de détention chevauche la frontière IS/OOS (sinon elles contaminent le test).

- Embargo : délai de sécurité supplémentaire après la coupure IS/OOS, adapté à l'horizon des trades.

- CPCV (Combinatorial Purged Cross-Validation) : recombinaison de découpages purgés pour estimer la dispersion (l'incertitude) des performances.

- CSCV (Combinatorial Symmetric Cross-Validation) : méthode pour estimer la probabilité de surapprentissage.

- PBO (Probability of Backtest Overfitting) : probabilité que la stratégie choisie soit en réalité inférieure à la médiane des stratégies testées.

- Deflated Sharpe Ratio : Sharpe « déflaté » du nombre d'essais tentés — car tester beaucoup de configurations finit toujours par en trouver une bonne par hasard.

- Biais de sélection multiple (multiple testing) : plus on teste de configurations, plus le « meilleur » résultat est optimiste par pur hasard.

- Test canari : test volontairement « empoisonné » (on injecte un bug connu et on vérifie que les tests le détectent) — prouve que la suite de tests protège réellement.

### Métriques de performance

- Sharpe ratio : rendement excédentaire (au-dessus du taux sans risque) divisé par la volatilité ; mesure le rendement « par unité de risque ».

- Taux sans risque (rf, risk-free) : rendement d'un placement sûr de référence, soustrait pour isoler la performance de la stratégie.

- Annualisation 365 vs 252 : conversion d'une performance journalière en performance annuelle (252 = jours de bourse, 365 = année calendaire ; pour la crypto, souvent 365). Une divergence ici fausse tous les chiffres.

- Drawdown : baisse du capital depuis son dernier sommet — la « perte maximale subie en cours de route », plus parlante qu'un rendement total.

- Equity curve (courbe de capital) : la valeur du portefeuille au fil du temps, le graphique de référence d'un backtest.

- Turnover : taux de rotation du portefeuille (combien on achète/vend), qui se traduit directement en frais.

- Exposition : part du capital réellement engagée sur le marché.

- Benchmark : référence de comparaison (ex. acheter-et-conserver le BTC) pour savoir si la stratégie « bat le marché ».

- P&L (Profit & Loss) : gain ou perte d'un trade.

- Métriques nettes vs brutes : nettes = après déduction des coûts ; brutes = avant. Seules les nettes sont « négociables » en réel.

- Slippage : écart entre le prix voulu et le prix réellement obtenu à l'exécution.

- Spread : écart entre prix d'achat et prix de vente (bid/ask).

- Funding : frais périodiques des contrats perpétuels en crypto.

### Sélection de paramètres (niveaux L1/L2/L3)

- L1 : sélection des paramètres au sein d'une fenêtre.

- L2 : sélection entre les fenêtres (pour le backtest final).

- L3 : consensus entre les étapes d'une campagne par étapes.

- Stagewise : campagne en N étapes successives ; chaque étape fige les paramètres de l'étape précédente et n'optimise que les siens.

- SNV (neighbor smoothing, lissage par voisinage) : lisser la performance d'un paramètre par la moyenne de ses voisins dans l'espace des paramètres, pour privilégier les zones stables (un plateau) plutôt qu'un pic isolé.

- SVI : ré-évaluer les meilleurs candidats sur IS₂ et retenir le top-K.

- KDTree : structure de données pour trouver rapidement les voisins proches dans un espace multi-dimensionnel.

- PQS : métrique interne ∝ √(nombre de trades / référence), qui pénalise les stratégies à trop peu de trades (statistiquement peu fiables).

### Optimisation

- Grid search : tester exhaustivement toutes les combinaisons d'une grille.

- Bayesian (skopt) : optimisation guidée par un modèle probabiliste qui essaye en priorité les zones prometteuses (moins d'essais que le grid).

- Optuna : bibliothèque d'optimisation de paramètres (échantillonneur + arrêt précoce).

- Seed (graine) : graine aléatoire qui rend un tirage reproductible.

- n\_jobs : nombre de processeurs utilisés en parallèle (source fréquente de non-reproductibilité).

- Budget d'essais : nombre total d'évaluations autorisé, fixé avant de commencer.

### Données et temps

- MTF (Multi-TimeFrame) : utiliser plusieurs horizons de temps (ex. 15 min et 1 jour) dans une même stratégie.

- Resampling : agréger des barres fines en barres plus grosses.

- OHLCV : Open, High, Low, Close, Volume — les champs d'une barre de prix.

- Open-time / close-time : horodater une barre par son ouverture ou par sa clôture (une divergence d'une barre = fuite/lag silencieux).

- label left/right : rattacher la valeur agrégée au bord gauche ou droit de l'intervalle.

- barmerge.lookahead\_off : option Pine/TradingView qui interdit d'utiliser la clôture d'une barre haute avant qu'elle ne soit confirmée sur la barre basse.

- ffill / merge\_asof : méthodes de jointure qui complètent les valeurs manquantes ; ffill (reporter la dernière valeur) peut faire fuiter la barre suivante.

- Warm-up : période initiale où les indicateurs ne sont pas encore calculables (valeurs NaN), à exclure des métriques.

- Gap-fill : combler les trous (barres manquantes) dans une série.

### Tests et reproductibilité

- Golden test / golden file : comparer le résultat à une référence figée et chiffrée.

- Property-based test : vérifier une propriété toujours vraie (ex. métrique nette ≤ métrique brute).

- Snapshot de régression (syrupy) : comparer automatiquement la sortie à un instantané enregistré, pour détecter toute dérive de définition.

- xfail : test marqué « échec attendu et documenté » (pour les divergences connues).

- Manifest : fiche d'identité d'un run (hash des données, version du code, configuration), pour pouvoir le rejouer.

- Idempotent : rejouer la même entrée produit exactement la même sortie.

## Références

- Bailey & López de Prado, The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality (2014).

- López de Prado, Advances in Financial Machine Learning (2018) — purge, embargo, CPCV, CSCV/PBO.

