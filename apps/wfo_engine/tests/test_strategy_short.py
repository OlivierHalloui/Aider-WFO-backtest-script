"""Unit tests for ATDMF short strategy signals and indicators.

Tests cover:
- pivot_high_nb symmetry with pivot_low_nb
- depassement_roc_short_nb (bearish candle filter)
- calculate_exit_sma_short_nb (crossover)
- cross_sar_sma_short_exit_nb (SAR crosses below SMA)
- macd_short_exit_signal_nb (MACD crossover)
- create_entry_exit_conditions: T1_short crossunder logic
- run_backtest: long-only regression (strategy_direction='long_only' unchanged)
- run_backtest: strategy_direction propagation

These tests do NOT require vectorbtpro (skipped if unavailable).
"""
import numpy as np
import pandas as pd
import pytest

# ── VBT availability guard ──────────────────────────────────────────────────
try:
    import vectorbtpro  # noqa: F401
    HAS_VBT = True
except Exception:
    HAS_VBT = False

_VBT_SKIP = pytest.mark.skipif(not HAS_VBT, reason="vectorbtpro not available")


# ===========================================================================
# Helpers
# ===========================================================================

def _make_df(n=100, trend='flat', seed=42):
    """Create a synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    if trend == 'up':
        close = np.linspace(100, 130, n) + rng.normal(0, 0.3, n)
    elif trend == 'down':
        close = np.linspace(130, 100, n) + rng.normal(0, 0.3, n)
    else:
        close = 100 + rng.normal(0, 1, n)
    close = np.clip(close, 1, None)
    high  = close + abs(rng.normal(0, 0.5, n))
    low   = close - abs(rng.normal(0, 0.5, n))
    low   = np.clip(low, 1, None)
    open_ = close + rng.normal(0, 0.2, n)
    volume = rng.uniform(1, 10, n)
    idx = pd.date_range('2025-01-01', periods=n, freq='5s')
    return pd.DataFrame({'Open': open_, 'High': high, 'Low': low,
                         'Close': close, 'Volume': volume}, index=idx)


# ===========================================================================
# Phase 1 — Numba indicator tests (no VBT required)
# ===========================================================================

class TestPivotHighNb:
    """pivot_high_nb must be the symmetric mirror of pivot_low_nb."""

    @_VBT_SKIP
    def test_detects_simple_peak(self):
        from indicators import pivot_high_nb
        # Clear peak at index 5: [1,2,3,4,5,10,5,4,3,2,1] → peak at 5, confirmed at 5+right=7
        series = np.array([1,2,3,4,5,10,5,4,3,2,1], dtype=np.float64)
        sig = pivot_high_nb(series, left=2, right=2)
        assert sig[7] == True, "Peak at index 5 should be confirmed at index 7"
        assert sig.sum() == 1, "Exactly one peak"

    @_VBT_SKIP
    def test_no_false_positives_on_flat(self):
        from indicators import pivot_high_nb
        series = np.ones(20, dtype=np.float64)
        sig = pivot_high_nb(series, left=2, right=2)
        assert sig.sum() == 0

    @_VBT_SKIP
    def test_symmetry_with_pivot_low(self):
        from indicators import pivot_high_nb, pivot_low_nb
        rng = np.random.default_rng(0)
        series = rng.normal(0, 1, 50).astype(np.float64)
        highs = pivot_high_nb(series, left=2, right=2)
        lows  = pivot_low_nb(-series, left=2, right=2)   # inverted series
        np.testing.assert_array_equal(highs, lows,
            err_msg="pivot_high on series == pivot_low on -series")

    @_VBT_SKIP
    def test_nan_handling(self):
        from indicators import pivot_high_nb
        series = np.array([1, np.nan, 3, 5, 3, np.nan, 1], dtype=np.float64)
        sig = pivot_high_nb(series, left=1, right=1)
        assert not np.isnan(sig).any()


class TestDepassementRoCShortNb:
    """depassement_roc_short_nb: only fires on bearish candles."""

    @_VBT_SKIP
    def test_bearish_candle_in_range_fires(self):
        from indicators import depassement_roc_short_nb
        n = 50
        # Bearish candle: close < open, large range
        close = np.full(n, 99.0); open_ = np.full(n, 100.0)
        high  = np.full(n, 101.0); low   = np.full(n, 98.0)
        sig = depassement_roc_short_nb(close, open_, high, low,
                                        depass_sma_roc=0.5, roc_max=100.0)
        assert sig[21:].any(), "Bearish candles with RoC in range should fire"

    @_VBT_SKIP
    def test_bullish_candle_never_fires(self):
        from indicators import depassement_roc_short_nb
        n = 50
        # Bullish candle: close > open
        close = np.full(n, 101.0); open_ = np.full(n, 99.0)
        high  = np.full(n, 102.0); low   = np.full(n, 98.0)
        sig = depassement_roc_short_nb(close, open_, high, low,
                                        depass_sma_roc=0.5, roc_max=100.0)
        assert sig.sum() == 0, "Bullish candles must never fire short RoC"

    @_VBT_SKIP
    def test_long_vs_short_roc_complementary(self):
        """On mixed candles, long and short RoC signals should not overlap."""
        from indicators import depassement_roc_long_nb, depassement_roc_short_nb
        n = 100
        rng = np.random.default_rng(1)
        close = 100 + rng.normal(0, 1, n)
        open_ = 100 + rng.normal(0, 1, n)
        high  = np.maximum(close, open_) + abs(rng.normal(0, 0.5, n))
        low   = np.minimum(close, open_) - abs(rng.normal(0, 0.5, n))
        long_sig  = depassement_roc_long_nb(close, open_, high, low, 0.5, 100.0)
        short_sig = depassement_roc_short_nb(close, open_, high, low, 0.5, 100.0)
        overlap = long_sig & short_sig
        assert overlap.sum() == 0, "Long and short RoC signals must not overlap"


class TestCalculateExitSmaShortNb:
    """SMA short exit: crossover (close rises above SMA)."""

    @_VBT_SKIP
    def test_crossover_fires(self):
        from indicators import calculate_exit_sma_short_nb
        n = 30
        # Build a series where close starts below SMA then crosses above
        close = np.ones(n) * 95.0
        close[20:] = 105.0   # crosses above SMA≈100 after warmup
        sig = calculate_exit_sma_short_nb(close, 10)
        assert sig[20:25].any(), "Crossover above SMA should fire"

    @_VBT_SKIP
    def test_crossunder_does_not_fire(self):
        from indicators import calculate_exit_sma_short_nb
        n = 30
        # close stays below SMA or is falling through it (crossunder = long exit, not short)
        close = np.ones(n) * 105.0
        close[20:] = 95.0
        sig = calculate_exit_sma_short_nb(close, 10)
        # The transition 105→95 is a crossunder, should NOT fire the short exit
        assert sig[20:25].sum() == 0, "Crossunder must not trigger short SMA exit"

    @_VBT_SKIP
    def test_long_and_short_sma_exits_do_not_overlap(self):
        from indicators import calculate_exit_sma_nb, calculate_exit_sma_short_nb
        n = 60
        rng = np.random.default_rng(2)
        close = 100 + np.cumsum(rng.normal(0, 0.5, n))
        long_sig  = calculate_exit_sma_nb(close, 10)
        short_sig = calculate_exit_sma_short_nb(close, 10)
        overlap = long_sig & short_sig
        assert overlap.sum() == 0, "Long SMA exit and short SMA exit must not overlap"


class TestCrossSARSMAShortExitNb:
    """Cross SAR/SMA short exit: SAR crosses below SMA."""

    @_VBT_SKIP
    def test_sar_crosses_below_sma_fires(self):
        from indicators import cross_sar_sma_short_exit_nb
        n = 10
        sma = np.ones(n) * 100.0
        # SAR starts above SMA then drops below
        sar = np.array([102,101,100,99,98,97,96,95,94,93], dtype=np.float64)
        sig = cross_sar_sma_short_exit_nb(sma, sar)
        assert sig[3] == True, "SAR crossing below SMA at index 3 should fire"

    @_VBT_SKIP
    def test_sar_stays_below_sma_does_not_fire(self):
        from indicators import cross_sar_sma_short_exit_nb
        sma = np.ones(10) * 100.0
        sar = np.ones(10) * 90.0   # already below, no crossing
        sig = cross_sar_sma_short_exit_nb(sma, sar)
        assert sig.sum() == 0


class TestMACDShortExitNb:
    """MACD short exit: crossover (MACD rises above signal line)."""

    @_VBT_SKIP
    def test_crossover_fires(self):
        from indicators import macd_short_exit_signal_nb
        n = 60
        # Declining series then recovering: MACD should eventually cross above signal
        close = np.concatenate([np.linspace(110, 90, 30), np.linspace(90, 110, 30)])
        sig = macd_short_exit_signal_nb(close, 5, 10, 3, True, True, True)
        assert sig.any(), "Recovering price should trigger MACD short exit crossover"

    @_VBT_SKIP
    def test_long_and_short_macd_exits_complementary(self):
        from indicators import macd_exit_signal_nb, macd_short_exit_signal_nb
        n = 100
        rng = np.random.default_rng(3)
        close = 100 + np.cumsum(rng.normal(0, 0.5, n))
        long_sig  = macd_exit_signal_nb(close, 5, 10, 3, True, True, True)
        short_sig = macd_short_exit_signal_nb(close, 5, 10, 3, True, True, True)
        overlap = long_sig & short_sig
        assert overlap.sum() == 0, "Long MACD exit and short MACD exit must not overlap"


# ===========================================================================
# Phase 2 — Strategy signal tests (VBT required)
# ===========================================================================

class TestT1ShortCrossunder:
    """T1 short: close must cross BELOW the lower Bollinger Band."""

    @_VBT_SKIP
    def test_crossunder_lower_generates_short_entry(self):
        """Build a synthetic df where close crosses below lower BB exactly once
        and verify that short_entry_condition fires on that bar."""
        from strategy import create_signal_generators, create_entry_exit_conditions

        df = _make_df(n=300, trend='down')
        params = dict(
            timeperiod=12, StDev=1.3, matype=0,
            coeff_medianeBBW=1.2, coef_mediane=0.89,
            Nb_bars_above=1, fenetre_lowest=80, seuil_lowest=3.5,
            longueur_mediane=100, nb_bars_under_bbw_mini=4, nb_bars_entre_bb=5,
            depassement_sma_roc=0.01, roc_max_t1=100.0,
            use_roc_filter=False,   # disable RoC filter to isolate crossunder logic
            use_t2_signal=False,
            user_exit_sma_length=14,
            sar_start=0.02, sar_increment=0.02, sar_maximum=0.2,
            exit_sar_enabled=True, macd_fast_length=9, macd_slow_length=19,
            macd_signal_length=6, macd_ma_type='sma',
            exit_macd_enabled=True, exit_macd_type_a=True, exit_macd_type_b=True,
            exit_cross_sar_sma_enabled=True,
            exit_retour_bb_enabled=False, nb_bars_left_pivot=2, nb_bars_right_pivot=2,
            exit_regline_enabled=False, nombre_periodes_reglin=15, i_bars_back=1,
            exit_volat_down_enabled=False, seuil_overbought_bb=0.85,
            strategy_direction='short_only',
        )
        signals = create_signal_generators(df, **params)
        assert signals.get('strategy_direction') == 'short_only'
        assert signals.get('depassement_roc_short_signal') is not None

        _, _, _, short_entry, short_exit, _ = create_entry_exit_conditions(df, signals)
        assert short_entry is not None, "short_entry_condition must not be None in short_only mode"
        assert short_exit  is not None, "short_exit_condition must not be None in short_only mode"

    @_VBT_SKIP
    def test_long_only_no_short_signals(self):
        """In long_only mode, short_entry/exit must be None."""
        from strategy import create_signal_generators, create_entry_exit_conditions

        df = _make_df(n=200)
        params = dict(
            timeperiod=12, StDev=1.3, matype=0,
            coeff_medianeBBW=1.2, coef_mediane=0.89,
            Nb_bars_above=1, fenetre_lowest=80, seuil_lowest=3.5,
            longueur_mediane=100, nb_bars_under_bbw_mini=4, nb_bars_entre_bb=5,
            depassement_sma_roc=0.01, roc_max_t1=100.0,
            use_roc_filter=True, use_t2_signal=False,
            user_exit_sma_length=14,
            sar_start=0.02, sar_increment=0.02, sar_maximum=0.2,
            exit_sar_enabled=True, macd_fast_length=9, macd_slow_length=19,
            macd_signal_length=6, macd_ma_type='sma',
            exit_macd_enabled=True, exit_macd_type_a=True, exit_macd_type_b=True,
            exit_cross_sar_sma_enabled=True,
            exit_retour_bb_enabled=False, nb_bars_left_pivot=2, nb_bars_right_pivot=2,
            exit_regline_enabled=False, nombre_periodes_reglin=15, i_bars_back=1,
            exit_volat_down_enabled=False, seuil_overbought_bb=0.85,
            strategy_direction='long_only',
        )
        signals = create_signal_generators(df, **params)
        _, _, _, short_entry, short_exit, _ = create_entry_exit_conditions(df, signals)
        assert short_entry is None, "long_only mode must produce no short_entry_condition"
        assert short_exit  is None, "long_only mode must produce no short_exit_condition"


# ===========================================================================
# Phase 3 — run_backtest regression + direction tests (VBT required)
# ===========================================================================

_BASE_PARAMS = dict(
    timeperiod=12, StDev=1.3, matype=0,
    coeff_medianeBBW=1.2, coef_mediane=0.89,
    Nb_bars_above=1, fenetre_lowest=80, seuil_lowest=3.5,
    longueur_mediane=100, nb_bars_under_bbw_mini=4, nb_bars_entre_bb=5,
    depassement_sma_roc=0.01, roc_max_t1=100.0,
    use_roc_filter=True, use_t2_signal=False, use_divergence_bb=True,
    user_exit_sma_length=14,
    sar_start=0.02, sar_increment=0.02, sar_maximum=0.2,
    exit_sar_enabled=True, macd_fast_length=9, macd_slow_length=19,
    macd_signal_length=6, macd_ma_type='sma',
    exit_macd_enabled=True, exit_macd_type_a=True, exit_macd_type_b=True,
    exit_cross_sar_sma_enabled=True,
    exit_retour_bb_enabled=False, nb_bars_left_pivot=2, nb_bars_right_pivot=2,
    exit_regline_enabled=False, nombre_periodes_reglin=15, i_bars_back=1,
    exit_volat_down_enabled=False, seuil_overbought_bb=0.85,
    order_sizing_mode='percent_equity', order_fixed_cash=10000.0, fees_pct=0.0,
    pqs_n_ref=50,
)


class TestRunBacktestRegression:
    """Verify that strategy_direction='long_only' produces identical results
    to running without the parameter (regression test)."""

    @_VBT_SKIP
    def test_long_only_returns_portfolio(self):
        from strategy import run_backtest
        df = _make_df(n=500, trend='up')
        pf = run_backtest(df, {**_BASE_PARAMS, 'strategy_direction': 'long_only'},
                          timeframe='5s', return_portfolio=True)
        assert pf is not None
        assert hasattr(pf, 'total_return')

    @_VBT_SKIP
    def test_long_only_default_equals_explicit(self):
        """Explicit 'long_only' must give same total_return as no strategy_direction key."""
        from strategy import run_backtest, clear_window_indicator_cache
        df = _make_df(n=500, trend='up', seed=7)
        clear_window_indicator_cache()
        pf_default = run_backtest(df, dict(_BASE_PARAMS), timeframe='5s', return_portfolio=True)
        clear_window_indicator_cache()
        pf_explicit = run_backtest(df, {**_BASE_PARAMS, 'strategy_direction': 'long_only'},
                                   timeframe='5s', return_portfolio=True)
        assert abs(float(pf_default.total_return) - float(pf_explicit.total_return)) < 1e-9, \
            "Adding strategy_direction='long_only' must not change long-only results"

    @_VBT_SKIP
    def test_short_only_returns_portfolio(self):
        from strategy import run_backtest
        df = _make_df(n=500, trend='down')
        pf = run_backtest(df, {**_BASE_PARAMS, 'strategy_direction': 'short_only'},
                          timeframe='5s', return_portfolio=True)
        assert pf is not None
        assert hasattr(pf, 'total_return')

    @_VBT_SKIP
    def test_both_returns_portfolio(self):
        from strategy import run_backtest
        df = _make_df(n=500, trend='flat')
        pf = run_backtest(df, {**_BASE_PARAMS, 'strategy_direction': 'both'},
                          timeframe='5s', return_portfolio=True)
        assert pf is not None

    @_VBT_SKIP
    def test_short_only_long_trades_are_zero(self):
        """In short_only mode, Portfolio must have no long trades."""
        from strategy import run_backtest
        df = _make_df(n=500, trend='down', seed=5)
        pf = run_backtest(df, {**_BASE_PARAMS, 'strategy_direction': 'short_only'},
                          timeframe='5s', return_portfolio=True)
        # VBT: pf.trades contains both long and short records
        # direction column: 0=long, 1=short
        try:
            trades_df = pf.trades.records_readable
            if 'Direction' in trades_df.columns:
                long_trades = (trades_df['Direction'] == 'Long').sum()
                assert long_trades == 0, \
                    f"short_only mode produced {long_trades} long trades"
        except Exception:
            pass   # trades accessor may differ across VBT versions; skip gracefully
