# Backlog V3 - Priorisation P0 / P1 / P2

## Etat d'avancement (2026-02-13)

- P0.1 -> P0.6: implémentés.
- P1.1 -> P1.4: implémentés (avec gate d'exécution + parité détaillée + preuve MTF).
- P2.1: implémenté (assistant LLM avec revalidation déterministe).
- P2.2: implémenté (catalogue local + recherche/rechargement UI).
- P2.3: implémenté (campagne CI de parité + artefact JSON publié).

## P0 - Fondations obligatoires (demarrage)

### P0.1 - Mode strategie dans la config

- Ajouter `strategy_mode` dans config (`native_atdmf`, `pine_imported`).
- Ajouter identifiant de strategie (`strategy_id`, hash source).

Critere d'acceptation:

1. Un run WFO peut etre execute sans regression en mode `native_atdmf`.
2. Les metadonnees strategie apparaissent dans la trace et l'export ZIP.

### P0.2 - Upload et pre-analyse Pine

- UI pour charger un fichier Pine/TXT.
- Verification basique: presence `//@version=`, `strategy(`, syntaxe minimale.

Critere d'acceptation:

1. L'utilisateur voit un rapport immediate "valide / invalide".
2. Les erreurs sont lisibles et actionnables.

### P0.3 - Rapport de compatibilite

- Scanner des features Pine detectees.
- Score + liste `supporte / partiel / non supporte`.

Critere d'acceptation:

1. Si feature bloquante (`S0`), mode strict bloque l'execution.
2. Le rapport est exporte dans les artefacts du run.

### P0.4 - Contrat `strategy_spec.v1`

- Schema JSON stable pour signaux, indicateurs, exits, parametres.
- Serialisation + validation stricte.

Critere d'acceptation:

1. Un spec invalide est refuse avant runtime.
2. Le spec valide est versionne avec le run.

### P0.5 - Interface `StrategyAdapter`

- Introduire une interface commune.
- Brancher `ATDMFAdapter` sur l'existant.

Critere d'acceptation:

1. Le moteur WFO n'appelle plus directement la logique ATDMF en dur.
2. Les tests existants ATDMF passent sans regression.

### P0.6 - Export artefacts V3

- Ajouter au ZIP: source Pine, rapport compatibilite, spec, trace generation.

Critere d'acceptation:

1. Upload d'un ZIP historique permet de recharger ces artefacts.
2. Module Expert IA peut exploiter ces artefacts pour ses analyses.

## P1 - Capacite Pine exploitable

### P1.1 - Codegen Python depuis `strategy_spec.v1`

- Generer un module Python standardise.
- Runner securise (sans eval arbitraire du script source).

Critere d'acceptation:

1. Le module genere respecte l'interface `StrategyAdapter`.
2. En cas d'echec codegen, message guide + fallback propre.

### P1.2 - Support MTF partiel (`request.security`)

- Supporter resolutions frequentes (`1s`, `5s`, `30s`, `1m`, etc.).
- Politique explicite de lookahead.

Critere d'acceptation:

1. Les signaux MTF sont reproductibles sur meme dataset.
2. Les limites MTF sont documentees dans le rapport compatibilite.

### P1.3 - Bibliotheques Pine externes (phase assistee)

- Detecter `import ... as X`.
- Permettre mappage manuel vers module Python local.

Critere d'acceptation:

1. Sans mapping, run bloque en mode strict.
2. Avec mapping valide, run possible et traceable.

### P1.4 - Ecran de validation parite

- Afficher comparaison Pine ref vs Python (si reference disponible).

Critere d'acceptation:

1. L'utilisateur voit les ecarts sur signaux/trades/perf.
2. Un statut `parity_pass` est enregistre dans le run.

## P2 - Industrialisation

### P2.1 - Assistant LLM de migration Pine -> spec

- Agent assiste la conversion, jamais source de verite unique.

Critere d'acceptation:

1. Toute sortie LLM est revalidee par parser/validator deterministe.
2. Prompt + modele + hash sortie sont journalises.

### P2.2 - Catalogue de strategies

- Enregistrer strategies importees, versions, compatibilite, perf.

Critere d'acceptation:

1. Recherche et re-jeu d'une strategie importee depuis UI.
2. Historique de versions exploitable.

### P2.3 - Test suite de parite CI

- Campagnes auto multi-strategies/multi-timeframes.

Critere d'acceptation:

1. Pipeline CI bloque si derive parite > seuil.
2. Rapport diff automatiquement publie.

## Estimation charge (ordre de grandeur)

- P0: 2 a 3 semaines
- P1: 4 a 7 semaines
- P2: 3 a 5 semaines

Total V3 progressive: 9 a 15 semaines selon profondeur Pine/MTF voulue.
