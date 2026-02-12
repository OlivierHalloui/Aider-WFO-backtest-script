# Contributing

## Scope
This repository is used for ATDMF WFO development and maintenance.

## Development workflow
1. Create a feature branch from `main`.
2. Make focused commits with clear messages (`feat(...)`, `fix(...)`, `refactor(...)`).
3. Run local checks before pushing:
   - `pytest -q`
   - `python -m py_compile apps/wfo_engine/app.py apps/wfo_engine/main.py apps/wfo_engine/wfo.py apps/wfo_engine/adaptive_optimization.py apps/wfo_engine/strategy.py apps/wfo_engine/indicators.py apps/wfo_engine/data_loading.py apps/wfo_engine/config.py`
4. Open a Pull Request to `main` with:
   - What changed
   - Why it changed
   - Validation performed

## Pull request checklist
- No secrets or API keys in code/config.
- No large local artifacts committed (`Data/`, `reports/`).
- Changelog updated when behavior changes.
- Backward compatibility considered for config and result import/export.

## Code style
- Prefer small, testable functions and explicit naming.
- Add short docstrings/comments for non-obvious logic.
- Avoid broad `except Exception` unless justified.
