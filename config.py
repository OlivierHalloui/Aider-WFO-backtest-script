# Import necessary libraries for configuration
import os

DEFAULT_START_DATE = "2025-01-19"
DEFAULT_END_DATE = "2025-01-31"
DEFAULT_TIMEFRAME = '5s'
DEFAULT_DATA_FILE = (
    "/home/olivier/Downloads/ATDMF_strategy_V5_long/ATDMF_strategy_long_"
    "BTCFDUSD05S/Data/Binance_BTCUSDT_OHLCV_B_2025-01-19_2025-01-31_1s/"
    "Binance_BTCUSDT_OHLCV_B_2025-01-19_2025-01-31_5S.csv"
)

DEFAULT_PARAM_GRID = {
    'timeperiod': (10, 30, 5),
    'StDev': (1, 2.5, 0.5),
    'coeff_medianeBBW': (0.8, 1.6, 0.4),
    'coef_mediane': (1, 1.5, 0.5),
    'fenetre_lowest': (30, 60, 10),
    'seuil_lowest': (1.0, 3.5, 0.5),
    'longueur_mediane': (50, 150, 50),
    'Nb_bars_above': (2, 6, 2),
    'user_exit_sma_length': (10, 30, 10),
    'sar_start': (0.02, 0.05, 0.01),
    'sar_increment': (0.02, 0.05, 0.01),
    'sar_maximum': (0.1, 0.3, 0.05)
}

# ======================================================================
# CONFIGURATION SETTINGS
# ======================================================================

# Walk-Forward Optimization settings
class WFOSettings:
    def __init__(self):
        self.n_windows = 1               # Number of windows to divide data into
        self.train_size = 0.5            # Proportion of window for training
        self.anchored = False            # Whether to use anchored (fixed start date) WFO
        self.optimization_metric = "sharpe_ratio"  # Main metric to optimize
        self.secondary_metric = "total_return"     # Secondary metric to optimize
        self.metric_weights = (1.0, 0.0) # Weights for primary and secondary metrics
        self.parallel_backend = "dask"   # Parallelization backend ('dask', 'ray', 'pathos', 'threadpool')
        self.use_numba = True            # Whether to use Numba for accelerated computations
        self.chunk_size = 200            # Chunk size for parallelization
        self.optimization_method = "bayesian"  # Optimization method ('bayesian', 'optuna', 'grid')
        self.patience_level = "Medium"   # Patience for Bayesian/Optuna early stopping
        self.max_trials = 200            # Maximum number of trials for Bayesian/Optuna optimization
        self.neighbor_count = 5          # Neighbor count for stability selection
        self.exit_sar_enabled = True     # Enable Parabolic SAR exit
