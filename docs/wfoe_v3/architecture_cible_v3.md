# Architecture cible WFOE V3 (Pine -> WFO)

## Principes d'architecture

1. **Pas de patch dynamique** de `apps/wfo_engine/app.py`.
2. Separation claire: ingestion Pine, compilation spec, runtime strategie.
3. Chaque etape produit un artefact sauvegardable pour replay.
4. Toute execution est versionnee (spec, codegen, config WFO, dataset descriptor).

## Vue d'ensemble (pipeline)

1. Upload script Pine (`.txt` / `.pine`)
2. Analyse syntaxique/statique
3. Evaluation compatibilite (S0/S1/S2/S3)
4. Generation `strategy_spec.v1.json`
5. Generation module Python strategie (adapter)
6. Execution WFO via moteur existant
7. Export resultats + artefacts (spec, rapport compatibilite, code genere)

## Modules cibles

Nouveau package propose:

- `apps/wfo_engine/pine_v3/ingestion.py`
- `apps/wfo_engine/pine_v3/parser.py`
- `apps/wfo_engine/pine_v3/compat.py`
- `apps/wfo_engine/pine_v3/spec_models.py`
- `apps/wfo_engine/pine_v3/spec_builder.py`
- `apps/wfo_engine/pine_v3/codegen.py`
- `apps/wfo_engine/pine_v3/runtime_adapter.py`
- `apps/wfo_engine/pine_v3/artifacts.py`

## Contrat runtime (interface strategie)

Interface cible a introduire:

```python
class StrategyAdapter(Protocol):
    def get_param_space(self) -> dict: ...
    def generate_signals(self, df, params: dict) -> dict: ...
    def run_backtest(self, df, params: dict, timeframe: str, return_portfolio: bool = True): ...
```

Impls:

- `ATDMFAdapter` (legacy, base actuelle)
- `PineAdapter` (nouveau, base spec/codegen)

## Integrations avec modules existants

- `apps/wfo_engine/main.py`
  - selectionne un adapter selon `strategy_mode` (`native_atdmf` / `pine_imported`)
- `apps/wfo_engine/services/run_service.py`
  - charge l'adapter choisi avant appel WFO/adaptatif
- `apps/wfo_engine/wfo.py`
  - reste moteur agnostique, ne manipule que `params` + score
- `apps/wfo_engine/app.py`
  - nouvelle section UI "Strategie Pine"
  - affiche rapport compatibilite, spec, codegen status

## Artefacts standards V3

Par run:

1. `strategy_source.pine.txt`
2. `compatibility_report.json`
3. `strategy_spec.v1.json`
4. `generated_strategy.py`
5. `generation_trace.json` (prompts/versions/hash si LLM utilise)

## Decisions techniques lot 0

1. Pas d'execution directe d'AST Pine en prod V1.
2. On passe par `strategy_spec.v1` comme contrat stable.
3. La generation LLM est assistee, mais validation deterministic obligatoire avant run.

## Validation de parite

Definition minimale:

1. meme dataset OHLCV
2. meme timeframe
3. memes parametres
4. comparaison:
   - timestamps signaux entree/sortie
   - nombre de trades
   - pnl cumule
   - max drawdown

Seuils d'acceptation (V3 initiale):

- ecart trades <= 2%
- ecart pnl cumule <= 3%
- ecart max drawdown <= 3 points de base relatifs
