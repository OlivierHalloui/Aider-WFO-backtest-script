# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ATDMF Strategy Walk-Forward Optimizer (WFOE) — a Streamlit application for walk-forward optimization and backtesting of the ATDMF cryptocurrency trading strategy using VectorBT Pro. Supports Grid, Bayesian, and Optuna optimization with configurable multi-level parameter selection.

A Pine Script V3 subsystem enables importing TradingView Pine V6 strategies, transpiling them to Python, and validating parity between Pine and Python execution.

## Build & Run

**Python 3.10+** required. Key dependencies: `streamlit`, `vectorbtpro`, `pandas`, `numpy`, `numba`, `optuna`, `scikit-optimize`, `plotly`.

```bash
pip install -r requirements.txt

# Run the main app
streamlit run apps/wfo_engine/app.py

# Alternative launcher (handles streamlit args)
bash scripts/run_wfoe.sh
```

**PYTHONPATH must include `apps/wfo_engine`** for imports to resolve. The CI sets `PYTHONPATH: apps/wfo_engine` explicitly. For local dev:
```bash
PYTHONPATH=apps/wfo_engine pytest ...
PYTHONPATH=apps/wfo_engine python -c "from config import ..."
```

## Testing

```bash
# Run all Pine V3 tests (lightweight — only needs pytest, numpy, pandas)
pytest -q apps/wfo_engine/tests/test_pine_v3_*.py apps/wfo_engine/tests/test_pine_strategy_test_adapter.py

# Run a single test file
pytest apps/wfo_engine/tests/test_pine_v3_spec.py

# Run a single test by name
pytest apps/wfo_engine/tests/test_pine_v3_spec.py -k test_spec_extracts_strategy_order_short_direction

# Run parity CI campaign (generates JSON report)
python apps/wfo_engine/tests/run_pine_parity_ci.py --output reports/ci/pine_parity_ci_report.json

# Syntax check on core modules (run before pushing)
python -m py_compile apps/wfo_engine/app.py apps/wfo_engine/main.py apps/wfo_engine/wfo.py apps/wfo_engine/adaptive_optimization.py apps/wfo_engine/strategy.py apps/wfo_engine/indicators.py apps/wfo_engine/data_loading.py apps/wfo_engine/config.py apps/wfo_engine/metrics.py apps/wfo_engine/neural_search.py apps/wfo_engine/stagewise_optimizer.py apps/wfo_engine/ui/stagewise_panel.py apps/wfo_engine/ui/final_backtest_panel.py
```

CI (`.github/workflows/ci.yml`) runs on push to `main` and PRs: Pine V3 tests, parity CI campaign, and syntax check. VectorBT Pro is **not** installed in CI — only `pytest`, `numpy`, `pandas`.

## Architecture

### Data Flow

1. **UI** (`app.py`) collects config → **run_service** orchestrates
2. **`resolve_strategy_adapter()`** picks the right adapter based on `strategy_mode`/`strategy_id`
3. **WFO engine** (`wfo.py`) or **adaptive engine** (`adaptive_optimization.py`) calls `adapter.run_backtest()` for each parameter combination across train/test windows
4. Results flow back to UI for visualization and export (ZIP with JSON/CSV artifacts)

### Strategy Adapter Pattern (`strategy_adapters.py`)

The optimization engines are decoupled from strategy logic through a `Protocol`:
- **`ATDMFAdapter`** — native mode, delegates to `strategy.py` (Bollinger Bands, SAR, MACD, SMA indicators via `indicators.py`)
- **`PineStrategyTestAdapter`** — Pine V3 test runtime (`pine_v3/runtime_adapter.py`)
- **Generated Pine adapters** — dynamically loaded from `pine_v3/codegen.py` output

All adapters implement: `get_param_space()`, `generate_signals()`, `run_backtest()`.

### Core Modules (`apps/wfo_engine/`)

- **`app.py`**: Streamlit UI entry point — sidebar config, session state, panel orchestration. Imports from `ui/` panels, `services/`, `expert/`, `pine_v3/`, and `domain/`
- **`wfo.py`**: Walk-forward optimization engine — orchestrates train/test windows, optimization methods, Level 1 parameter selection (SNV/SVI/raw_max), backtest caching. Key functions: `walk_forward_optimization`, `get_stable_best_params` (SNV), `get_svi_best_params` (SVI)
- **`wfo_save.py`**: Persistence of WFO run results (JSON/CSV serialization)
- **`stagewise_optimizer.py`**: Stagewise WFO campaign — runs N sequential stages, each fixing the previous stage's consensus best params. Key functions: `run_stagewise_campaign`, `build_stagewise_wfo_results`, `_apply_stage_consensus` (Level 3 dispatch). Produces a standard `wfo_results` dict compatible with all visualization and export flows
- **`strategy.py`**: ATDMF strategy logic — signal generation factory (`create_signal_generators`), backtest runner (`run_backtest`) building VectorBT portfolios
- **`indicators.py`**: Technical indicators (Bollinger Bands, SAR, MACD, SMA) — Numba-optimized, VectorBT integrated
- **`adaptive_optimization.py`**: Rolling optimization without fixed WFO windows, progressive grid narrowing
- **`data_loading.py`**: CSV/Binance OHLCV loading with date filtering
- **`config.py`**: Default parameter grid, strategy mode constants, `WFOSettings` dataclass. Key fields: `selection_method`, `svi_top_k`, `svi_is2_fraction`, `cross_window_method`, `optimization_metric`, `neighbor_count`, `pqs_n_ref`
- **`metrics.py`**: Shared metric helpers (`trade_stat`, `calc_avg_pl`, `safe_float`, `calc_pqs`) used by `wfo.py`, `adaptive_optimization.py`, and `main.py`
- **`neural_search.py`**: `NeuralSearchGuide` for guided optimization — extracted from `wfo.py`

### Pine V3 Subsystem (`apps/wfo_engine/pine_v3/`)

Pipeline: **Pine text → spec.v1 JSON → Python runtime → VectorBT signals**

- **`spec.py`**: Parses Pine Script → `strategy_spec.v1` JSON (regex-based with optional `pynescript` backend)
- **`runtime_adapter.py`**: Executes spec as Python — generates VectorBT-compatible entry/exit signals, handles MTF resampling. Contains `PineStrategyTestAdapter` and `GeneratedPineRuntimeAdapter`
- **`codegen.py`**: Generates standalone Python module from spec
- **`parity.py`** / **`parity_ci.py`**: Validates Pine/Python equivalence (trade counts ±2%, PnL ±3% tolerance). CI campaign with JSON reporting
- **`execution_gate.py`**: Pre-execution beta readiness and parity gate (blocks `Start WFO` if gate fails)
- **`mtf.py`** / **`mtf_parity.py`**: Multi-timeframe resampling (`request.security`) and MTF parity proof
- **`catalog.py`**: Local strategy versioning by `(id, source_sha1)`
- **`llm_migration.py`**: LLM-assisted Pine→spec conversion with deterministic fallback

### Supporting Modules

- **`expert/`**: LLM gateway for WFO analysis (OpenAI-compatible API) — `llm_gateway.py`, `prompt_builder.py`, `analyzer.py`; storage in `reports/expert/`
- **`services/`**: Run orchestration (`run_service.py`), ZIP export (`export_utils.py`), audit trail (`traceability.py`), timeframe/date helpers (`runtime_utils.py`)
- **`ui/`**: Streamlit UI panels:
  - `strategy_panel.py` — 4-tab strategy parameter editor (entry signals, exits, filters, MTF); `PARAMETER_HELP` is single source of truth imported by `expert_panel.py`
  - `stagewise_panel.py` — stagewise WFO campaign launcher; inherits all config from sidebar; Level 3 consensus method selector; auto-saves timestamped ZIP
  - `final_backtest_panel.py` — final backtest display + Level 2 cross-window selection dispatch (`_select_final_params_from_results`); implements `best_is_oos`, `best_oos`, `robust_set`, `weighted_oos`
  - `pine_panel.py` — Pine V3 import/parity
  - `expert_panel.py` — LLM analysis
  - `export_panel.py` — ZIP export/import
  - `expert_report.py` — report rendering
  - `data_utils.py` — downsampling, trade metrics
- **`domain/`**: Cross-cutting utilities — `serialization.py` (JSON sanitization, UTC datetime, hashing, type conversion)
- **`visualization.py`**: WFO results charts, parameter maps, PDF reports

### Other Apps

- **`apps/dbad/`**: Binance dataset downloader
- **`apps/exit_signals/`**: Trade exit analysis tool

## Commit Style

Conventional commits: `feat(scope)`, `fix(scope)`, `refactor(scope)`, `docs(scope)`, `ci(scope)`.

## Parameter Selection Architecture

Three independent selection levels, each configurable in the UI:

| Level | Scope | Methods | Config key | Where |
|---|---|---|---|---|
| **L1** intra-fenêtre | Parmi les combos IS d'une fenêtre | `snv` (KDTree voisinage), `svi` (top-K sur IS₂), `raw_max` | `selection_method` | WFO Engine Settings sidebar |
| **L2** cross-fenêtres | Pour le Final Backtest | `best_is_oos`, `best_oos`, `robust_set`, `weighted_oos` | `cross_window_method` | WFO Engine Settings sidebar |
| **L3** cross-stages | Consensus entre fenêtres d'un run stagewise | `median_mode`, `best_window`, `weighted_oos_median` | `stage_consensus_method` | Panneau campagne stagewise |

All defaults reproduce the pre-feature behavior (SNV / best_is_oos / median_mode).

## Stagewise WFO

`stagewise_optimizer.py` runs N sequential optimization stages. Each stage:
1. Fixes all params from previous stages via `_FixedParamAdapter`
2. Optimizes only its declared subset of params via `walk_forward_optimization`
3. Applies Level 3 consensus (`_apply_stage_consensus`) → updates `accumulated_best`

`build_stagewise_wfo_results(final_report)` reconstructs a standard `wfo_results` dict from the last stage's data, enabling full IS/OOS visualization, ZIP export, Final Backtest, and Comparative Backtest flows without code duplication.

The stagewise panel inherits all sidebar config (timeframe, direction, windows, dates, metrics, sizing, execution). It only exposes: output directory, stage selection checkboxes, final backtest dates, and Level 3 consensus method.

## Key Conventions

- Strategy adapters follow the `StrategyAdapter` Protocol — never call `run_backtest` directly from optimization engines
- Pine V3 spec (`strategy_spec.v1`) is the single source of truth for imported strategies — always validate spec before runtime
- `Data/`, `reports/`, `fichiers_configuration_WFO/` are gitignored — never commit large artifacts
- Update `CHANGELOG.md` when behavior changes (changelog is in French)
- The UI language is French; code/comments/docstrings are in English
- `WFOSettings.from_config(config_dict)` is the canonical way to build settings from `get_current_config()` output — it silently ignores unknown keys
