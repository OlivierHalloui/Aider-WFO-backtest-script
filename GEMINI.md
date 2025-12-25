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

---
*Last updated: December 23, 2025*