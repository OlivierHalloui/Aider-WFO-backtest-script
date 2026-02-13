# WFOE V3 Beta - Etat de completion

Date de consolidation: 2026-02-13

## Statut des lots

- P0.1 -> P0.6: `done`
- P1.1 -> P1.4: `done`
- P2.1 -> P2.2: `done`
- P2.3: `done` (campagne de parité CI via script dédié)

## Validation minimale locale

```bash
conda run -n vectorbtpro-run pytest -q apps/wfo_engine/tests/test_pine_v3_*.py apps/wfo_engine/tests/test_pine_strategy_test_adapter.py
```

```bash
PYTHONPATH=apps/wfo_engine conda run -n vectorbtpro-run python apps/wfo_engine/tests/run_pine_parity_ci.py --output reports/ci/pine_parity_ci_report.json
```

Critère attendu:

- `status=passed` dans `reports/ci/pine_parity_ci_report.json`

## Artefacts V3 à vérifier dans les exports

- `pine_compatibility_report.json`
- `pine_strategy_spec.v1.json`
- `pine_codegen_report.json`
- `pine_generation_trace.json`
- `pine_parity_report.json`
- `pine_parity_reference.v1.json`
- `pine_parity_reference_validation.json`
- `pine_mtf_parity_proof_report.json`
- `pine_execution_gate_report.json`
- `pine_order_semantics_report.json`
- `pine_llm_migration_report.json` (si assistant utilisé)

## Limitations beta connues

- `request.security_lower_tf` reste hors périmètre stable (bloquant strict/gate beta).
- Les imports Pine externes nécessitent mapping explicite vers modules Python locaux.
- Les ordres pending Pine (`limit/stop/trail_*`) restent bloquants en mode strict (`pine_order_semantics`).
- La campagne CI inclut des scénarios runtime MTF uniquement si `vectorbtpro` est disponible.
