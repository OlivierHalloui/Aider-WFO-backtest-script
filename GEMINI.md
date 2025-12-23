# ATDMF Strategy Optimizer

## Project Overview
This project is a sophisticated trading strategy optimization platform built with Python. It utilizes **VectorBT Pro** for high-performance backtesting and focuses on **Walk-Forward Optimization (WFO)** to robustly tune the "ATDMF" trading strategy (likely based on Bollinger Bands and custom signals).

The system features a **Streamlit**-based Graphical User Interface (GUI) for easy configuration, execution, and result visualization, alongside a Command Line Interface (CLI) for automated workflows.

## Architecture & Key Components

### 1. User Interface
- **`app.py`**: The main entry point for the Streamlit web application. It provides:
  - Configuration sidebar (Data, Strategy, WFO settings).
  - Execution controls for the WFO engine.
  - Interactive visualization of results (Equity curves, Parameter stability, Heatmaps).
  - Robustness analysis (Radar charts).

### 2. Core Logic & Entry Points
- **`main.py`**: Acts as the central orchestrator. It can:
  - Launch the Streamlit GUI (`subprocess.run(["streamlit", "run", "app.py"])`).
  - Run in CLI mode using flags like `--no-gui` or `--config <path>`.
  - Coordinate data loading, optimization, and reporting.

### 3. Optimization Engines
- **`wfo_v2.py` / `wfo.py`**: Implements the Walk-Forward Optimization logic. It supports:
  - **Methods:** Grid Search, Bayesian Optimization, Optuna.
  - **Validation:** Anchored vs. Sliding windows.
  - **Performance:** Parallel processing (Thread, Dask, Ray) and Numba acceleration.

### 4. Trading Strategy
- **`strategy_v2.py` / `strategy.py`**: Defines the "ATDMF" trading logic.
- **`indicators.py`**: Contains custom technical indicators (likely Numba-optimized) used by the strategy.

### 5. Data & Configuration
- **`data_loading.py`**: Handles data ingestion from local CSV files or the Binance API.
- **`config.json`**: Stores persistent settings for the optimization run (Dates, Parameter ranges, File paths).
- **`config.py`**: Defines default constants and configuration classes (e.g., `WFOSettings`).

## Usage

### Prerequisite
Ensure **VectorBT Pro** and other dependencies are installed. (Note: VectorBT Pro is a proprietary library).

### Running the GUI (Recommended)
To start the interactive dashboard:
```bash
streamlit run app.py
```
*Alternatively, running `python main.py` will also launch the Streamlit app.*

### Running via CLI
To run a headless optimization using default settings or a specific config file:
```bash
# Run with defaults (Grid Search, 1 Window)
python main.py --no-gui

# Run with a specific configuration file
python main.py --config config.json
```

## Configuration (`config.json`)
The `config.json` file controls the experiment. Key fields include:
- **Data:** `start_date`, `end_date`, `timeframe`, `file_path`.
- **WFO:** `n_windows` (splits), `train_size` (ratio), `optimization_method` (`grid`, `bayesian`, `optuna`).
- **Parameters:** Ranges for strategy parameters (e.g., `timeperiod_min`, `timeperiod_max`, `step`).

## Directory Structure
- `docs/`: Documentation for VectorBT Pro.
- `WFO_Results/`: Output directory for CSV results, best parameters, and logs.
- `WFO_Reports/`: Generated PDF reports.
- `.ipynb_checkpoints/`: Jupyter notebook checkpoints (implies analysis is also done in notebooks).

## Development Notes
- **Numba:** Extensive use of `@njit` (implied by file names in `__pycache__` and performance focus) for speed.
- **Refactoring:** Presence of `_v2` files indicates an active or recent refactoring phase. `wfo_v2.py` seems to be the "SOTA" (State-of-the-Art) version referenced in the GUI.
