# WFOE V3 - Lot 0 (Cadrage)

Ce dossier contient les livrables de cadrage pour la V3 de WFO Engine (WFOE), avec objectif: backtester des strategies Pine Script v6 dans le moteur WFO existant.

## Livrables

- `compatibilite_pine_v6.md`: matrice de compatibilite Pine v6 et niveau de support cible.
- `architecture_cible_v3.md`: architecture cible, interfaces, flux d'execution.
- `backlog_v3_p0_p1_p2.md`: backlog priorise par lots (P0/P1/P2) avec criteres d'acceptation.
- `risques_et_garde_fous.md`: risques techniques et garde-fous de qualite.

## Portee Lot 0

Lot 0 est un lot de cadrage:

1. Definir clairement ce qui sera supporte en V3 (subset Pine v6).
2. Definir une architecture qui n'implique pas de patch dynamique de `apps/wfo_engine/app.py`.
3. Produire un backlog actionnable pour lancer le lot 1.

## Campagne de parité CI (P2.3)

Commande locale:

```bash
PYTHONPATH=apps/wfo_engine python apps/wfo_engine/tests/run_pine_parity_ci.py --output reports/ci/pine_parity_ci_report.json
```

- Le fichier `reports/ci/pine_parity_ci_report.json` contient le détail des scénarios et des dérives.
- Le code retour est bloquant (`exit 1`) si la parité échoue ou si des blockers sont détectés.
- En CI GitHub, le rapport est publié en artefact (`pine-parity-ci-report`).

## References

- Base code actuelle: `apps/wfo_engine/`
- Exemple strategy Pine fournie: `pine_script_trading_view_strategy/ATDMF_strategy long BTCUSDC_05S-MEXC V6_02.txt`
- Reference doc Pine mentionnee: `https://github.com/codenamedevan/pinescriptv6`
