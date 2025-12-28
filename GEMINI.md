# ATDMF Backtest V1

## Project Overview
ATDMF Strategy Optimizer is a professional backtesting framework designed for evaluating and optimizing trading strategies using **Walk-Forward Optimization (WFO)**. It leverages **VectorBT Pro** for high-performance backtesting and **Streamlit** for an interactive dashboard.

## Architecture & Technologies
-   **Frontend**: [Streamlit](https://streamlit.io/) (`app.py`) - Reactive dashboard for configuration, monitoring, and visualization.
-   **Backtesting Engine**: [VectorBT Pro](https://vectorbt.dev/) - High-performance, vector-based backtesting.
-   **Optimization**: Custom Walk-Forward Optimization (WFO) engine (`wfo.py`, `wfo_save.py`).
-   **Strategy Logic**: Defined in `strategy.py`.
-   **Indicators**: Custom technical indicators in `indicators.py`.
-   **Data Management**: `data_loading.py` handles local and API-based data retrieval.
-   **Visualization**: Plotly-based visualizations in `visualization.py`.

## Directory Structure
-   `app.py`: Main Streamlit application entry point.
-   `main.py`: Core logic for managing parameters and metrics.
-   `strategy.py`: Implementation of the ATDMF trading strategy.
-   `indicators.py`: Technical analysis indicators.
-   `wfo.py`: Implementation of the Walk-Forward Optimization process.
-   `config.py`: Default settings and configuration constants.
-   `docs/`: Documentation and research materials.
-   `data/`: (Optional) Local CSV/H5 data storage.

## Building and Running
1.  **Environment Setup**:
    Ensure you have a Python environment (3.8+) with the necessary dependencies installed.
    *Required: streamlit, pandas, numpy, plotly, vectorbt[pro] (Note: vbt pro requires a license).*

2.  **Run the Dashboard**:
    ```bash
    streamlit run app.py
    ```

3.  **Configuration**:
    The application allows loading and saving configurations via JSON files (e.g., `config.json`).

## Development Conventions
-   **State Management**: Uses `st.session_state` extensively for maintaining UI state across interactions.
-   **Modularity**: Strategy logic is decoupled from the UI and optimization engine.
-   **Performance**: Utilizes `numba` and `parallel` backends where applicable for speed.

## Current Strategy Status vs. Pine Script (Gap Analysis)
*As of Dec 27, 2025*

The Python implementation (`strategy.py`, `indicators.py`) is currently a subset of the reference TradingView Pine Script (`ATDMF_strategy long BTCUSDC_05S-MEXC V6_01.pine`).

### Implemented Features
- **Signal T0 Core**:
  - `BollingerHorizontal` (Flatness check)
  - `EcartBollingerBorne` (Bandwidth expansion check)
  - `CrossBBWLowSignal` (Low volatility threshold)
- **Basic Exits**:
  - `SMAExit` (Simple Moving Average cross exit)

### Missing Features (To Be Implemented)
1.  **Indicators**:
    -   **RoC (Rate of Change)**: `depassement_RoC_long` validation.
    -   **Linear Regression**: `pente_regline_bull` (Slope trend filter).
    -   **Stochastic**: `Sto_overbought`, Cross Types A & B (Bull/Bear).
    -   **MACD**: Cross Types A & B (Bull/Bear).
    -   **SMA 7/23**: Anticipated crossover logic.
    -   **Parabolic SAR**: Trend direction and trailing stop values.
    -   **Pivot Points**: For `BB_retournement` logic.

2.  **Entry Logic (T1 & T2)**:
    -   **Multi-Timeframe Trend**: `UT_principale_SMA_Bullish` (e.g., 1m trend filter for 5s strategy).
    -   **T1 Confirmation**: Combining T0 with RoC, RegLine, and Oscillator filters.
    -   **T2 Signal**: Re-entry/Confirmation logic based on High breakout and Divergence.
    -   **Duration Logic**: `NB_bars_under_BBW` loop (counting persistent low volatility) and `BBandcrossBarssince`.

3.  **Exit Strategies**:
    -   **BB UTC**: Close on Lower BB cross (Timeframe Courte).
    -   **SMA UTC**: Close on SMA cross (Timeframe Courte).
    -   **Oscillator Exits**: Stochastic and MACD Bearish crosses.
    -   **TAC**: "Time And Change" aggressive exit.
    -   **Return to BB**: Exit on return to lower band.
    -   **Volat Down**: Exit on bandwidth contraction (%B).

---
*Last updated: December 27, 2025*
