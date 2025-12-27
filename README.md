# ATDMF Strategy Walk-Forward Optimizer

Streamlit app to run walk-forward optimization (WFO) and backtests for the ATDMF strategy using VectorBT Pro.

## Features
- Walk-forward optimization (anchored or unanchored)
- Grid, Bayesian, or Optuna optimization
- Parameter stability selection with neighbor smoothing
- In-sample vs out-of-sample performance visualization
- Final backtest on the best window or a user-selected window
- Config import/export (JSON)

## Requirements
- Python 3.10+
- VectorBT Pro
- Streamlit

Install dependencies in your environment (example):

```bash
pip install -r requirements.txt
```

If you use a Conda env, activate it before running the app.

## Quick Start

```bash
streamlit run app.py
```

The app opens in your browser. Configure data, parameters, and optimization settings in the sidebar, then click **Start WFO**.

## Configuration

The sidebar controls:
- Date range and timeframe
- Data source (local file or Binance API)
- Parameter ranges and selection
- WFO settings (windows, train size, backend, trials)
- Performance metrics and weights

You can save the current configuration as JSON and reload it later.

## Notes
- Bayesian and Optuna methods operate on discrete parameter grids.
- The final backtest can use the best overall window or a selected window.
- Some metrics rely on trade stats; ensure your VectorBT Pro version exposes these stats.

## Project Structure
- `app.py`: Streamlit UI
- `wfo.py`: WFO engine and optimization logic
- `strategy.py`: ATDMF strategy logic and backtest runner
- `data_loading.py`: Data loading utilities
- `visualization.py`: Additional charts and reporting
- `docs/`: VectorBT Pro documentation
- `pine_script_trading_view_strategy/`: TradingView scripts

## License
Proprietary / internal use unless stated otherwise.
