"""Unit tests for the metrics.py module."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, PropertyMock

# Guard VBT-dependent imports
try:
    import vectorbtpro as vbt
    HAS_VBT = True
except Exception:
    HAS_VBT = False

from metrics import safe_float, to_scalar_score, python_scalar, trade_stat, calc_avg_pl, portfolio_metrics


# ---------------------------------------------------------------------------
# safe_float
# ---------------------------------------------------------------------------

class TestSafeFloat:
    def test_regular_float(self):
        assert safe_float(3.14) == 3.14

    def test_numpy_scalar(self):
        assert safe_float(np.float64(2.5)) == 2.5

    def test_numpy_int_scalar(self):
        assert safe_float(np.int64(7)) == 7.0

    def test_bool_true(self):
        assert safe_float(True) == 1.0

    def test_bool_false(self):
        assert safe_float(False) == 0.0

    def test_numpy_bool_true(self):
        assert safe_float(np.bool_(True)) == 1.0

    def test_numpy_bool_false(self):
        assert safe_float(np.bool_(False)) == 0.0

    def test_nan_returns_default(self):
        assert safe_float(float('nan')) == 0.0

    def test_inf_returns_default(self):
        assert safe_float(float('inf')) == 0.0

    def test_neg_inf_returns_default(self):
        assert safe_float(float('-inf')) == 0.0

    def test_none_returns_default(self):
        assert safe_float(None) == 0.0

    def test_string_returns_default(self):
        assert safe_float("hello") == 0.0

    def test_custom_default(self):
        assert safe_float(None, default=-1.0) == -1.0

    def test_nan_with_custom_default(self):
        assert safe_float(float('nan'), default=99.0) == 99.0

    def test_int_value(self):
        assert safe_float(42) == 42.0


# ---------------------------------------------------------------------------
# to_scalar_score
# ---------------------------------------------------------------------------

class TestToScalarScore:
    def test_scalar_float(self):
        assert to_scalar_score(5.0) == 5.0

    def test_scalar_int(self):
        assert to_scalar_score(3) == 3.0

    def test_none_returns_nan(self):
        assert np.isnan(to_scalar_score(None))

    def test_nan_returns_nan(self):
        assert np.isnan(to_scalar_score(float('nan')))

    def test_inf_returns_nan(self):
        assert np.isnan(to_scalar_score(float('inf')))

    def test_array_returns_mean(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert to_scalar_score(arr) == pytest.approx(2.0)

    def test_empty_array_returns_nan(self):
        assert np.isnan(to_scalar_score(np.array([])))

    def test_series_returns_finite_mean(self):
        s = pd.Series([1.0, 2.0, np.nan, 4.0])
        assert to_scalar_score(s) == pytest.approx(7.0 / 3.0)

    def test_all_nan_array_returns_nan(self):
        assert np.isnan(to_scalar_score(np.array([np.nan, np.nan])))

    def test_array_with_inf_filters(self):
        arr = np.array([1.0, np.inf, 3.0])
        # Inf is filtered, finite mean of [1.0, 3.0] = 2.0
        assert to_scalar_score(arr) == pytest.approx(2.0)

    def test_all_inf_returns_nan(self):
        assert np.isnan(to_scalar_score(np.array([np.inf, -np.inf])))


# ---------------------------------------------------------------------------
# python_scalar
# ---------------------------------------------------------------------------

class TestPythonScalar:
    def test_np_int64(self):
        result = python_scalar(np.int64(10))
        assert result == 10
        assert type(result) is int

    def test_np_float64(self):
        result = python_scalar(np.float64(3.14))
        assert result == pytest.approx(3.14)
        assert type(result) is float

    def test_np_bool_true(self):
        result = python_scalar(np.bool_(True))
        assert result is True
        assert type(result) is bool

    def test_np_bool_false(self):
        result = python_scalar(np.bool_(False))
        assert result is False
        assert type(result) is bool

    def test_regular_int_passthrough(self):
        result = python_scalar(42)
        assert result == 42
        assert type(result) is int

    def test_regular_float_passthrough(self):
        result = python_scalar(2.5)
        assert result == 2.5
        assert type(result) is float

    def test_string_passthrough(self):
        result = python_scalar("abc")
        assert result == "abc"


# ---------------------------------------------------------------------------
# trade_stat
# ---------------------------------------------------------------------------

class TestTradeStat:
    def test_direct_attribute(self):
        trades = MagicMock()
        trades.avg_winning_trade = 5.5
        assert trade_stat(trades, 'avg_winning_trade') == 5.5

    def test_fallback_to_stats_dict(self):
        trades = MagicMock(spec=[])  # no auto-attributes
        stats_dict = {'avg winning trade': 3.0}
        trades.stats = MagicMock(return_value=stats_dict)
        # Attribute doesn't exist, should fall back to stats()
        result = trade_stat(trades, 'avg_winning_trade')
        assert result == 3.0

    def test_fallback_title_case(self):
        trades = MagicMock(spec=[])
        stats_dict = {'Avg Winning Trade': 7.0}
        trades.stats = MagicMock(return_value=stats_dict)
        result = trade_stat(trades, 'avg_winning_trade')
        assert result == 7.0

    def test_default_when_missing(self):
        trades = MagicMock(spec=[])
        trades.stats = MagicMock(return_value={})
        result = trade_stat(trades, 'nonexistent_attr', default=-1.0)
        assert result == -1.0

    def test_default_when_stats_raises(self):
        trades = MagicMock(spec=[])
        trades.stats = MagicMock(side_effect=RuntimeError("no stats"))
        result = trade_stat(trades, 'some_attr', default=0.0)
        assert result == 0.0


# ---------------------------------------------------------------------------
# calc_avg_pl
# ---------------------------------------------------------------------------

class TestCalcAvgPl:
    def test_scalar_path(self):
        portfolio = MagicMock()
        portfolio.total_return = 0.10  # 10%
        trades_mock = MagicMock()
        trades_mock.count.return_value = 5
        portfolio.trades = trades_mock
        result = calc_avg_pl(portfolio)
        # 0.10 * 100 / 5 = 2.0
        assert result == pytest.approx(2.0)

    def test_zero_trades(self):
        portfolio = MagicMock()
        portfolio.total_return = 0.05
        trades_mock = MagicMock()
        trades_mock.count.return_value = 0
        portfolio.trades = trades_mock
        assert calc_avg_pl(portfolio) == 0.0

    def test_none_trades_count(self):
        portfolio = MagicMock()
        portfolio.total_return = 0.05
        trades_mock = MagicMock()
        trades_mock.count.return_value = None
        portfolio.trades = trades_mock
        assert calc_avg_pl(portfolio) == 0.0

    def test_vectorized_series_path(self):
        portfolio = MagicMock()
        portfolio.total_return = 0.10
        trades_mock = MagicMock()
        n_trades = pd.Series([5, 0, 10])
        trades_mock.count.return_value = n_trades
        portfolio.trades = trades_mock
        result = calc_avg_pl(portfolio)
        # total_ret = 10.0; safe_trades = [5, NaN, 10]; avg_pl = [2.0, NaN, 1.0] -> fillna(0) -> [2.0, 0.0, 1.0]
        expected = pd.Series([2.0, 0.0, 1.0])
        pd.testing.assert_series_equal(result, expected)

    def test_exception_returns_zero(self):
        portfolio = MagicMock()
        portfolio.total_return = PropertyMock(side_effect=RuntimeError("broken"))
        # Accessing total_return raises
        type(portfolio).total_return = PropertyMock(side_effect=RuntimeError("broken"))
        assert calc_avg_pl(portfolio) == 0.0


# ---------------------------------------------------------------------------
# portfolio_metrics
# ---------------------------------------------------------------------------

class TestPortfolioMetrics:
    def test_basic_output(self):
        portfolio = MagicMock()
        portfolio.total_return = 0.15
        portfolio.sharpe_ratio = 1.2
        portfolio.max_drawdown = 0.05
        portfolio.calmar_ratio = 2.0
        portfolio.sortino_ratio = 1.5

        trades_mock = MagicMock()
        trades_mock.__len__ = MagicMock(return_value=10)
        trades_mock.win_rate = 0.6
        trades_mock.count.return_value = 10
        portfolio.trades = trades_mock

        result = portfolio_metrics(portfolio, window_id="w1")
        assert result['window'] == "w1"
        assert result['return'] == pytest.approx(15.0)
        assert result['sharpe'] == pytest.approx(1.2)
        assert result['max_drawdown'] == pytest.approx(5.0)
        assert result['n_trades'] == 10

    def test_trades_len_fails_gracefully(self):
        """When len(portfolio.trades) raises, n_trades defaults to 0.
        Note: portfolio.trades is still accessible for other fields; only
        __len__ raises."""
        portfolio = MagicMock()
        portfolio.total_return = 0.0
        portfolio.sharpe_ratio = 0.0
        portfolio.max_drawdown = 0.0
        portfolio.calmar_ratio = 0.0
        portfolio.sortino_ratio = 0.0

        trades_mock = MagicMock()
        trades_mock.__len__ = MagicMock(side_effect=RuntimeError("no len"))
        trades_mock.win_rate = 0.0
        trades_mock.count.return_value = 0
        portfolio.trades = trades_mock

        result = portfolio_metrics(portfolio, window_id=1)
        assert result['n_trades'] == 0
        assert result['window'] == 1
