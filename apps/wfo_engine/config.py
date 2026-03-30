# Import necessary libraries for configuration
import os
from dataclasses import dataclass, field, fields
from datetime import date
from typing import Tuple

TODAY = date.today().isoformat()
DEFAULT_START_DATE = TODAY
DEFAULT_END_DATE = TODAY
DEFAULT_TIMEFRAME = '5s'
DEFAULT_STRATEGY_MODE = "native_atdmf"
DEFAULT_STRATEGY_ID = "atdmf_native_v2"
DEFAULT_DATA_FILE = os.environ.get("WFOE_DEFAULT_DATA_FILE", "")
WFOE_UPLOAD_DIR = os.environ.get(
    "WFOE_UPLOAD_DIR",
    os.path.join(os.path.expanduser("~"), ".atdmf", "uploads"),
)

DEFAULT_PARAM_GRID = {
    'timeperiod': (8, 20, 2),
    'StDev': (0.8, 2.0, 0.3),
    'coeff_medianeBBW': (0.9, 1.5, 0.3),
    'coef_mediane': (0.7, 1.1, 0.1),
    'fenetre_lowest': (40, 120, 20),
    'seuil_lowest': (1.0, 3.5, 0.5),
    'longueur_mediane': (50, 150, 50),
    'Nb_bars_above': (1, 6, 1),
    'nb_bars_under_bbw_mini': (1, 8, 1),
    'nb_bars_entre_bb': (1, 10, 1),
    'user_exit_sma_length': (8, 20, 2),
    'sar_start': (0.02, 0.05, 0.01),
    'sar_increment': (0.02, 0.05, 0.01),
    'sar_maximum': (0.1, 0.3, 0.05),
    'macd_fast_length': (6, 14, 2),
    'macd_slow_length': (14, 26, 2),
    'macd_signal_length': (4, 10, 2)
}

# ======================================================================
# CONFIGURATION SETTINGS
# ======================================================================

# Walk-Forward Optimization settings
@dataclass
class WFOSettings:
    """Runtime settings container for classic, NN-guided and adaptive WFO engines."""

    n_windows: int = 1                          # Number of windows to divide data into
    train_size: float = 0.5                     # Proportion of window for training
    anchored: bool = False                      # Whether to use anchored (fixed start date) WFO
    optimization_metric: str = "sharpe_ratio"   # Main metric to optimize
    secondary_metric: str = "total_return"      # Secondary metric to optimize
    metric_weights: Tuple[float, float] = (1.0, 0.0)  # Weights for primary and secondary metrics
    parallel_backend: str = "dask"              # Parallelization backend ('dask', 'ray', 'pathos', 'threadpool')
    use_numba: bool = True                      # Whether to use Numba for accelerated computations
    chunk_size: int = 200                       # Chunk size for parallelization
    optimization_method: str = "bayesian"       # Optimization method ('bayesian', 'optuna', 'grid')
    optimization_regime: str = "classic"        # 'classic', 'prev_best_grid', 'nn_guided', 'adaptive_continuous'
    patience_level: str = "Medium"              # Patience for Bayesian/Optuna early stopping
    max_trials: int = 200                       # Maximum number of trials for Bayesian/Optuna optimization
    neighbor_count: int = 5                     # Neighbor count for stability selection
    nn_min_samples: int = 500                   # Min cumulative trials before enabling NN guidance
    nn_candidate_pool_size: int = 3000          # Random candidates scored by the NN per window
    nn_top_k: int = 250                         # Top candidate count used to build next guided grid
    nn_exploration_ratio: float = 0.15          # Fraction of baseline values kept for exploration
    nn_hidden_size: int = 32                    # Hidden neurons for the lightweight MLP
    nn_epochs: int = 60                         # Training epochs after each window
    nn_learning_rate: float = 0.01              # Training learning rate
    nn_l2: float = 1e-4                         # L2 regularization
    adaptive_train_bars: int = 5000             # Lookback bars used for each adaptive training cycle
    adaptive_cycle_bars: int = 5000             # Bars advanced/evaluated per cycle
    adaptive_trials_per_cycle: int = 150        # Number of parameter trials per cycle
    adaptive_candidate_pool_size: int = 3000    # Random candidates ranked before selecting trials
    adaptive_keep_ratio: float = 0.40           # Fraction of best values kept per parameter in guided grid
    adaptive_exploration_ratio: float = 0.20    # Fraction of random exploratory trials/values
    adaptive_min_values_per_param: int = 2      # Minimum number of retained values per parameter
    adaptive_decay: float = 0.98                # Exponential memory decay for historical value stats
    adaptive_ucb_beta: float = 0.75             # Uncertainty bonus for value ranking (UCB-like)
    adaptive_warmup_trials: int = 300           # Minimum historical trials before shrinking the grid
    adaptive_max_cycles: int = 0                # 0 = no cap, otherwise max number of adaptive cycles
    adaptive_oos_weight: float = 2.0            # Additional weight of OOS best score in value stats
    pqs_n_ref: int = 50                          # PQS confidence factor reference trade count (√(n_trades/n_ref))
    macd_ma_type: str = 'sma'                   # MA type for MACD calculation ('sma' = Pine V6, 'ema' = legacy)
    use_roc_filter: bool = True                 # Enable T1 RoC momentum filter
    use_t2_signal: bool = False                  # Enable T2 cascade (high breakout on next bar after T1)
    use_divergence_bb: bool = True               # T2: require BB divergence on T1 bar (optimizable when T2 active)
    exit_sar_enabled: bool = True               # Enable Parabolic SAR exit
    exit_macd_enabled: bool = True              # Enable MACD exit
    exit_macd_type_a: bool = True               # MACD exit type A (crossunder + signal falling)
    exit_macd_type_b: bool = True               # MACD exit type B (crossunder)
    exit_cross_sar_sma_enabled: bool = True     # Cross SAR/SMA exit (SAR crosses above SMA)
    exit_retour_bb_enabled: bool = False        # Retour BB exit (pivot low on lower band)
    exit_regline_enabled: bool = False          # Linear regression exit
    exit_volat_down_enabled: bool = False       # Volatility down / %BB exit

    @classmethod
    def from_config(cls, config: dict) -> "WFOSettings":
        """Create a WFOSettings instance from a config dict.

        Only keys that match a known field name are used; missing keys
        fall back to the field default.
        """
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in config.items() if k in known}
        return cls(**kwargs)
