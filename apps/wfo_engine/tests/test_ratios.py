"""Unit tests for the non-annualised ratio helpers in ``metrics``.

The WFO tests exercise only the fallback path (mocked portfolios without
``returns``); these cover the actual mean/std computation.
"""
import numpy as np
import pytest

from metrics import (
    _calmar_from_returns,
    _sharpe_from_returns,
    _sortino_from_returns,
)


class FakePf:
    def __init__(self, returns, sharpe_ratio=999.0, sortino_ratio=999.0, calmar_ratio=999.0):
        self.returns = returns
        self.sharpe_ratio = sharpe_ratio
        self.sortino_ratio = sortino_ratio
        self.calmar_ratio = calmar_ratio


class TestSharpeFromReturns:
    def test_raw_mean_over_std(self):
        rets = np.array([0.01, -0.02, 0.03, 0.0, 0.02])
        expected = rets.mean() / rets.std(ddof=1)
        got = _sharpe_from_returns(FakePf(rets), 999.0)
        assert got == pytest.approx(expected)
        assert got < 5  # non-annualised: must NOT be the ~2500x inflated figure

    def test_nan_returns_are_filtered(self):
        raw = np.array([0.01, -0.02, 0.03])
        rets = np.concatenate([[np.nan], raw])
        expected = raw.mean() / raw.std(ddof=1)
        assert _sharpe_from_returns(FakePf(rets), 999.0) == pytest.approx(expected)

    def test_returns_exposed_as_method(self):
        raw = np.array([0.01, -0.02, 0.03, 0.02])

        class MethodPf:
            sharpe_ratio = 999.0

            def returns(self):
                return raw

        assert _sharpe_from_returns(MethodPf(), 999.0) == pytest.approx(raw.mean() / raw.std(ddof=1))

    def test_fallback_without_returns(self):
        class Mock:
            sharpe_ratio = 1.2

        assert _sharpe_from_returns(Mock(), 1.2) == 1.2

    def test_fallback_when_zero_std(self):
        rets = np.array([0.5, 0.5, 0.5, 0.5])
        # zero variance -> NaN (never leak the annualised fallback into a raw column)
        assert np.isnan(_sharpe_from_returns(FakePf(rets), 999.0))

    def test_multicolumn_returns_not_pooled(self):
        col0 = np.array([0.01, -0.02, 0.03, 0.02])
        col1 = np.array([0.5, 0.5, -0.5, 0.5])  # distinct asset, must not mix
        rets = np.column_stack([col0, col1])
        assert _sharpe_from_returns(FakePf(rets), 999.0) == pytest.approx(col0.mean() / col0.std(ddof=1))


class TestSortinoCalmar:
    def test_sortino_uses_downside_deviation(self):
        rets = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        downside = np.minimum(rets, 0.0)
        expected = rets.mean() / np.sqrt(np.mean(downside ** 2))
        assert _sortino_from_returns(FakePf(rets), 999.0) == pytest.approx(expected)

    def test_calmar_return_over_maxdrawdown(self):
        rets = np.array([0.01, -0.05, 0.03, 0.02])
        cum = np.cumprod(1 + rets)
        peak = np.maximum.accumulate(cum)
        max_dd = np.max((peak - cum) / peak)
        expected = (cum[-1] - 1.0) / max_dd
        assert _calmar_from_returns(FakePf(rets), 999.0) == pytest.approx(expected)
