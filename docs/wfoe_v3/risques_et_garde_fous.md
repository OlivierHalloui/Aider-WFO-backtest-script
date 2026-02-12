# V3 - Registre de risques et garde-fous

## Risques majeurs

### R1 - Faux sentiment de "compatibilite Pine complete"

Impact:

- attentes utilisateur irrealisables
- dette technique rapide

Mitigation:

1. Scope officiel "subset Pine v6 certifie"
2. Score de compatibilite visible avant run
3. Blocage strict sur features S0

### R2 - Ecart Pine vs Python (parite insuffisante)

Impact:

- conclusions WFO fausses
- invalidation de la strategie en production

Mitigation:

1. Tests de parite obligatoires sur jeux de reference
2. Seuils d'ecarts explicites et versionnes
3. Rapport d'ecarts archive dans le run

### R3 - Dependance excessive au LLM

Impact:

- generation instable
- erreurs silencieuses

Mitigation:

1. LLM assiste mais non souverain
2. Validation deterministe post-generation
3. Traces completes (prompt, modele, version, hash)

### R4 - Cout/performance sur gros datasets

Impact:

- latence elevee, UX degradee
- risques RAM/CPU

Mitigation:

1. Execution par chunks lorsque possible
2. Limites de securite configurables
3. Instrumentation temps/memoire dans telemetry run

### R5 - Regressions sur le moteur actuel

Impact:

- destabilisation de `native_atdmf`

Mitigation:

1. Adapter pattern et tests de non-regression
2. Feature flag `strategy_mode`
3. Livraisons incrementales avec rollback simple

## Garde-fous de gouvernance technique

1. Aucun merge V3 sans tests unitaires + integration cibles.
2. Aucune execution Pine en mode "best effort" sans avertissement explicite.
3. Chaque run doit etre rejouable via artefacts exportes.
4. Les changements UI doivent conserver un mode "simple" pour usage courant.

## Definition of Done - Lot 0

Lot 0 est termine si:

1. matrice compatibilite validee
2. architecture cible validee
3. backlog priorise avec criteres d'acceptation
4. risques/mitigations documentes
5. decision go/no-go sur demarrage P0 prise
