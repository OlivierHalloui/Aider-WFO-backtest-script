# Import necessary libraries for configuration
import os
from datetime import date

TODAY = date.today().isoformat()
DEFAULT_START_DATE = TODAY
DEFAULT_END_DATE = TODAY
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
    'sar_maximum': (0.1, 0.3, 0.05),
    'macd_fast_length': (8, 16, 2),
    'macd_slow_length': (20, 40, 2),
    'macd_signal_length': (5, 15, 2)
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
        self.optimization_regime = "classic"   # 'classic', 'prev_best_grid', 'nn_guided', 'adaptive_continuous'
        self.patience_level = "Medium"   # Patience for Bayesian/Optuna early stopping
        self.max_trials = 200            # Maximum number of trials for Bayesian/Optuna optimization
        self.neighbor_count = 5          # Neighbor count for stability selection
        self.nn_min_samples = 500        # Min cumulative trials before enabling NN guidance
        self.nn_candidate_pool_size = 3000  # Random candidates scored by the NN per window
        self.nn_top_k = 250              # Top candidate count used to build next guided grid
        self.nn_exploration_ratio = 0.15 # Fraction of baseline values kept for exploration
        self.nn_hidden_size = 32         # Hidden neurons for the lightweight MLP
        self.nn_epochs = 60              # Training epochs after each window
        self.nn_learning_rate = 0.01     # Training learning rate
        self.nn_l2 = 1e-4                # L2 regularization
        self.adaptive_train_bars = 5000  # Lookback bars used for each adaptive training cycle
        self.adaptive_cycle_bars = 1000  # Bars advanced/evaluated per cycle
        self.adaptive_trials_per_cycle = 150  # Number of parameter trials per cycle
        self.adaptive_candidate_pool_size = 3000  # Random candidates ranked before selecting trials
        self.adaptive_keep_ratio = 0.40  # Fraction of best values kept per parameter in guided grid
        self.adaptive_exploration_ratio = 0.20  # Fraction of random exploratory trials/values
        self.adaptive_min_values_per_param = 2  # Minimum number of retained values per parameter
        self.adaptive_decay = 0.98       # Exponential memory decay for historical value stats
        self.adaptive_ucb_beta = 0.75    # Uncertainty bonus for value ranking (UCB-like)
        self.adaptive_warmup_trials = 300  # Minimum historical trials before shrinking the grid
        self.adaptive_max_cycles = 0     # 0 = no cap, otherwise max number of adaptive cycles
        self.adaptive_oos_weight = 2.0   # Additional weight of OOS best score in value stats
        self.exit_sar_enabled = True     # Enable Parabolic SAR exit
        self.exit_macd_enabled = True    # Enable MACD exit
        self.exit_macd_type_a = True     # MACD exit type A (crossunder + signal falling)
        self.exit_macd_type_b = True     # MACD exit type B (crossunder)
