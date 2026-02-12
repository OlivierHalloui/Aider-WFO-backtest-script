## Summary
- What changed?
- Why?

## Validation
- [ ] `pytest -q`
- [ ] `python -m py_compile apps/wfo_engine/app.py apps/wfo_engine/main.py apps/wfo_engine/wfo.py apps/wfo_engine/adaptive_optimization.py apps/wfo_engine/strategy.py apps/wfo_engine/indicators.py apps/wfo_engine/data_loading.py apps/wfo_engine/config.py`

## Risks / Backward compatibility
- Config compatibility impact:
- Results import/export impact:

## Checklist
- [ ] No secrets committed
- [ ] No local artifacts (`Data/`, `reports/`) committed
- [ ] `CHANGELOG.md` updated if behavior changed
