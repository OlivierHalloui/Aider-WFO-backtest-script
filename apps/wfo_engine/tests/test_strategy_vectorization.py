"""Parity harness for strategy.py vectorization (grid mode).

Regression tests for the vectorization fixes:
- B1: ``strategy_direction='both'`` vectorized must keep short signals.
- B2: grid-style params (``strategy_direction`` as ``np.array``, wfo.py:48)
  must NOT silently zero out trades.
- M1: mixed scalar/vectorized flags honor scalar ``use_t2_signal`` semantics.

Each test asserts the synthetic data actually produces non-zero signals
(``assert signals > 0`` per column) so parity is never validated on empty data.

Requires vectorbtpro (skipped if unavailable).
"""
import numpy as np
import pandas as pd
import pytest

try:
    import vectorbtpro  # noqa: F401
    HAS_VBT = True
except Exception:
    HAS_VBT = False

_VBT_SKIP = pytest.mark.skipif(not HAS_VBT, reason="vectorbtpro not available")


def _make_df(n=400, seed=42):
    """Sinusoid + noise so close crosses BB bands in both directions."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    close = 100 + 15 * np.sin(2 * np.pi * t / 60) + rng.normal(0, 0.5, n)
    close = np.clip(close, 1, None)
    high = close + abs(rng.normal(0, 0.6, n))
    low = close - abs(rng.normal(0, 0.6, n))
    low = np.clip(low, 1, None)
    open_ = close + rng.normal(0, 0.2, n)
    volume = rng.uniform(1, 10, n)
    idx = pd.date_range('2025-01-01', periods=n, freq='5s')
    return pd.DataFrame({'Open': open_, 'High': high, 'Low': low,
                         'Close': close, 'Volume': volume}, index=idx)


_BASE = dict(
    timeperiod=20, StDev=2.0, matype=0,
    coeff_medianeBBW=1.2, coef_mediane=0.89,
    Nb_bars_above=1, fenetre_lowest=80, seuil_lowest=3.5,
    longueur_mediane=100, nb_bars_under_bbw_mini=4, nb_bars_entre_bb=5,
    depassement_sma_roc=0.01, roc_max_t1=100.0,
    use_roc_filter=False, use_t2_signal=True, use_divergence_bb=True,
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


def _vec(params):
    """Mirror wfo.py:48 — every param becomes np.array(len k)."""
    return {key: np.array([val, val]) for key, val in params.items()}


def _arr(obj):
    return None if obj is None else np.asarray(obj, dtype=bool)


def _col(vec_obj, i):
    a = _arr(vec_obj)
    if a.ndim == 2:
        return a[:, i]
    return a


def _signals(df, params):
    from strategy import create_signal_generators, create_entry_exit_conditions, \
        clear_window_indicator_cache
    clear_window_indicator_cache()
    sig = create_signal_generators(df, **params)
    return create_entry_exit_conditions(df, sig)


class TestGridVectorizationParity:
    """B1/B2/M1 non-regression tests for strategy.py vectorization."""

    @_VBT_SKIP
    def test_both_direction_keeps_short_signals(self):
        """B1: vectorized 'both' must not drop shorts (strategy_direction np.array)."""
        from strategy import create_signal_generators, clear_window_indicator_cache

        df = _make_df()
        params_both = dict(_BASE, strategy_direction='both')
        vparams = _vec(params_both)
        vparams['strategy_direction'] = np.array(['both', 'both'])  # wfo.py style

        sig_v = create_signal_generators(df, **vparams)
        e_v, _, _, se_v, _, _ = _signals(df, vparams)
        e_s, _, _, se_s, _, _ = _signals(df, params_both)

        n_long_seq = int(_arr(e_s).sum())
        n_short_seq = int(_arr(se_s).sum()) if se_s is not None else 0
        n_short_vec = int(_arr(se_v).sum()) if se_v is not None else 0

        # Synthetic data MUST produce real signals (both directions).
        assert n_long_seq > 0, "synthetic data produced no long entries"
        assert n_short_seq > 0, "synthetic data produced no short entries"

        assert se_v is not None, "short_entry_condition is None in vectorized both"
        assert n_short_vec >= n_short_seq, "vectorized shorts should equal or exceed sequential (2 combos)"
        # parity: vectorized col 0 == sequential
        assert np.array_equal(_col(se_v, 0), _arr(se_s)), "short_entry parity vec[0] vs seq"
        assert np.array_equal(_col(e_v, 0), _arr(e_s)), "long entry parity vec[0] vs seq"

    @_VBT_SKIP
    def test_grid_style_direction_does_not_zero_trades(self):
        """B2: grid-style strategy_direction np.array must not zero out trades."""
        from strategy import run_backtest, clear_window_indicator_cache

        df = _make_df()
        vparams = _vec(dict(_BASE, strategy_direction='long_only'))
        vparams['strategy_direction'] = np.array(['long_only', 'long_only'])

        clear_window_indicator_cache()
        scores = run_backtest(df, vparams, timeframe='5s', return_portfolio=False)
        arr = np.asarray(scores.values if hasattr(scores, 'values') else scores, dtype=float)
        assert len(arr) == 2, f"expected 2 scores, got {len(arr)}"

        clear_window_indicator_cache()
        pf = run_backtest(df, vparams, timeframe='5s', return_portfolio=True)
        n_trades = len(pf.trades.records_readable)
        assert n_trades > 0, "grid-style run_backtest produced 0 trades"

    @_VBT_SKIP
    def test_mixed_flags_parity(self):
        """M1: mixed scalar/vectorized flags honor scalar use_t2 semantics."""
        df = _make_df()
        combo1 = dict(_BASE, strategy_direction='both', use_t2_signal=True, use_divergence_bb=True)
        combo2 = dict(_BASE, strategy_direction='both', use_t2_signal=False, use_divergence_bb=False)
        vparams = {key: np.array([combo1[key], combo2[key]]) for key in _BASE}
        vparams['strategy_direction'] = np.array(['both', 'both'])

        me_v, mx_v, _, mse_v, msx_v, _ = _signals(df, vparams)

        for i, combo in enumerate((combo1, combo2)):
            e_s, x_s, _, se_s, sx_s, _ = _signals(df, combo)
            for name, vec_obj, seq_obj in (
                ('entry', me_v, e_s), ('exit', mx_v, x_s),
                ('short_entry', mse_v, se_s), ('short_exit', msx_v, sx_s),
            ):
                vc = _col(vec_obj, i)
                sc = _arr(seq_obj)
                assert (vc is None) == (sc is None), f"combo{i} {name}: None mismatch"
                if vc is not None:
                    assert np.array_equal(vc, sc), f"combo{i} {name}: parity mismatch"

    @_VBT_SKIP
    def test_distinct_combos_parity(self):
        """Parity with distinct numeric params per column (timeperiod, StDev, sma)."""
        df = _make_df()
        combo1 = dict(_BASE, strategy_direction='both',
                      timeperiod=20, StDev=2.0, user_exit_sma_length=14,
                      use_roc_filter=True, use_t2_signal=True, use_divergence_bb=True)
        combo2 = dict(_BASE, strategy_direction='both',
                      timeperiod=14, StDev=2.5, user_exit_sma_length=20,
                      use_roc_filter=False, use_t2_signal=False, use_divergence_bb=False)
        vparams = {key: np.array([combo1[key], combo2[key]]) for key in _BASE}
        vparams['strategy_direction'] = np.array(['both', 'both'])

        me_v, mx_v, _, mse_v, msx_v, _ = _signals(df, vparams)

        for i, combo in enumerate((combo1, combo2)):
            e_s, x_s, _, se_s, sx_s, _ = _signals(df, combo)
            for name, vec_obj, seq_obj in (
                ('entry', me_v, e_s), ('exit', mx_v, x_s),
                ('short_entry', mse_v, se_s), ('short_exit', msx_v, sx_s),
            ):
                vc = _col(vec_obj, i)
                sc = _arr(seq_obj)
                assert (vc is None) == (sc is None), f"combo{i} {name}: None mismatch"
                if vc is not None:
                    assert np.array_equal(vc, sc), f"combo{i} {name}: parity mismatch"

    @_VBT_SKIP
    def test_heterogeneous_direction_raises(self):
        """R2: non-uniform strategy_direction across combos must raise ValueError."""
        from strategy import create_signal_generators
        df = _make_df()
        vparams = _vec(dict(_BASE, strategy_direction='both'))
        vparams['strategy_direction'] = np.array(['long_only', 'short_only'])
        with pytest.raises(ValueError):
            create_signal_generators(df, **vparams)

    @_VBT_SKIP
    def test_macd_cache_no_poison_toggle(self):
        """R1: toggling exit_macd_enabled must not poison the indicator cache."""
        from strategy import create_signal_generators
        df = _make_df()
        p_on = dict(_BASE, strategy_direction='long_only', exit_macd_enabled=True)
        p_off = dict(_BASE, strategy_direction='long_only', exit_macd_enabled=False)

        sig_on = create_signal_generators(df, **p_on)
        n_on = int(np.asarray(sig_on['macd_exit_signal'], dtype=bool).sum())
        sig_off = create_signal_generators(df, **p_off)
        n_off = int(np.asarray(sig_off['macd_exit_signal'], dtype=bool).sum())
        sig_on2 = create_signal_generators(df, **p_on)
        n_on2 = int(np.asarray(sig_on2['macd_exit_signal'], dtype=bool).sum())

        assert n_on > 0, "MACD exit should fire when enabled"
        assert n_off == 0, "MACD exit should be all-False when disabled"
        assert n_on2 == n_on, f"MACD cache poisoned after on->off->on: {n_on} -> {n_on2}"

    @_VBT_SKIP
    def test_sar_cache_no_poison_toggle(self):
        """R1 (symmetric): toggling exit_sar_enabled must not poison the cache."""
        from strategy import create_signal_generators
        df = _make_df()
        p_on = dict(_BASE, strategy_direction='long_only', exit_sar_enabled=True)
        p_off = dict(_BASE, strategy_direction='long_only', exit_sar_enabled=False)

        sig_on = create_signal_generators(df, **p_on)
        n_on = int(np.asarray(sig_on['sar_exit_signal'], dtype=bool).sum())
        sig_off = create_signal_generators(df, **p_off)
        n_off = int(np.asarray(sig_off['sar_exit_signal'], dtype=bool).sum())
        sig_on2 = create_signal_generators(df, **p_on)
        n_on2 = int(np.asarray(sig_on2['sar_exit_signal'], dtype=bool).sum())

        assert n_on > 0, "SAR exit should fire when enabled"
        assert n_off == 0, "SAR exit should be all-False when disabled"
        assert n_on2 == n_on, f"SAR cache poisoned after on->off->on: {n_on} -> {n_on2}"

    @_VBT_SKIP
    def test_volat_seuil_per_column(self):
        """R3: seuil_overbought_bb must vary per column (regression for flat[0])."""
        from strategy import create_signal_generators
        df = _make_df()
        combo1 = dict(_BASE, strategy_direction='long_only',
                      exit_volat_down_enabled=True, seuil_overbought_bb=0.8)
        combo2 = dict(_BASE, strategy_direction='long_only',
                      exit_volat_down_enabled=True, seuil_overbought_bb=0.6)
        vparams = {key: np.array([combo1[key], combo2[key]]) for key in _BASE}
        vparams['strategy_direction'] = np.array(['long_only', 'long_only'])

        sig_v = create_signal_generators(df, **vparams)
        volat_v = _arr(sig_v['volat_down_exit_signal'])
        assert volat_v.ndim == 2, "volat_down_exit_signal should be 2-D"

        sig_s0 = create_signal_generators(df, **combo1)
        v0 = _arr(sig_s0['volat_down_exit_signal'])
        sig_s1 = create_signal_generators(df, **combo2)
        v1 = _arr(sig_s1['volat_down_exit_signal'])

        assert v0.sum() > 0 and v1.sum() > 0, "volat_down should fire for both thresholds"
        assert not np.array_equal(v0, v1), "distinct thresholds must produce distinct signals"
        assert np.array_equal(volat_v[:, 0], v0), "col0 parity mismatch"
        assert np.array_equal(volat_v[:, 1], v1), "col1 parity mismatch"
