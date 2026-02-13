# Matrice de compatibilite Pine v6 - WFOE V3

## Objectif

Eviter une promesse "100% Pine" non realiste en V1.  
Le scope V3 initial est un **subset Pine v6 certifie**, testable et tracable.

## Niveaux de support

- `S0` Non supporte (bloquant)
- `S1` Support partiel (avec limites explicites)
- `S2` Supporte en V3 initiale
- `S3` Supporte et verifie par tests de parite Pine/Python

## Matrice fonctionnelle

| Domaine | Feature Pine v6 | Niveau cible V3 | Notes |
|---|---|---:|---|
| Structure | `strategy(...)` | S3 | Mapping vers metadata de run WFOE |
| Inputs | `input.int/float/bool/string/source/timeframe` | S2 | `source/timeframe` en S1 si usage complexe |
| Series | index `[1]`, `na`, `nz`, `barssince` | S2 | Important pour signaux temporises |
| TA de base | `ta.sma/ema/rma/wma/vwma`, `ta.bb`, `ta.macd`, `ta.sar` | S3 | Couvre les besoins ATDMF-like |
| TA avancee | fonctions exotiques non mappees | S1 | Fallback via "unsupported report" |
| MTF | `request.security` | S1 | Support restreint: resolution discrete, lookahead off |
| MTF bas TF | `request.security_lower_tf` | S0/S1 | Priorite basse (complexe/perf) |
| Ordres | `strategy.entry/exit/close/cancel` | S2 | Mappage sur moteur portefeuille actuel |
| Qty | `qty_percent`, qty fixes | S2 | Cohesion avec `order_sizing_mode` existant |
| Conditions | bool complexes + ternaires | S2 | Parsing AST requis |
| Fonctions user | fonctions locales Pine | S2 | Limite recursion/non standard |
| Librairies externes | `import user/lib/version` | S0 initial | Ex: `BBT1` devra etre re-implantee localement |
| Plot/UI | `plot`, `plotshape`, labels | S0 | Hors scope backtest |
| Alerts | `alert_message` | S1 | Conserve metadonnees sans effet runtime |
| Session/time | filtres horaires/jours | S2 | Via pandas index/timezone |

## Analyse appliquee a la strategie exemple ATDMF

Le script fourni utilise:

1. `request.security` multi-timeframe (UTP/UTC)  
2. `strategy.entry/exit/close/cancel`  
3. beaucoup de fonctions de librairie externe `BBT1.*`  
4. logique de signaux et sorties conditionnelles riches

Conclusion:

- Faisable en V3, **a condition** de traiter `BBT1` comme un module Python local cible (pas un import Pine dynamique).
- Le vrai risque n'est pas le Pine de base mais la parite de la librairie externe.

## Regles de compatibilite en UI (a implementer lot 1)

1. Afficher un score de compatibilite global (`0-100`).
2. Lister les elements bloques (`S0`) avant toute execution.
3. Proposer 3 modes:
   - `strict`: bloque si feature non supportee
   - `assist`: tente generation partielle + avertissements
   - `manual`: utilisateur complete les parties manquantes
4. Si `import ... as X` est detecte:
   - exiger un mapping explicite `X -> module Python` en mode `strict`
   - afficher le statut de resolution import par import (resolu/non resolu)
   - verifier aussi la couverture fonctionnelle: appels `X.foo(...)` detectes dans la strategie vs fonctions presentes dans la librairie importee
   - au runtime V3, consommer les modules mappes (phase assistee) avec fallback explicite si contrat fonctionnel incomplet
   - en mode strict, appliquer un contrat runtime `Alias.fonction(...)` resolu + callable avant execution backtest

## Decision Lot 0

Pour la V3 initiale, objectif officiel:

- Support **S2/S3** des strategies sans import externe Pine
- Support **S1** de `request.security`
- Blocage explicite pour `request.security_lower_tf` avance et `import` externe non mappe
