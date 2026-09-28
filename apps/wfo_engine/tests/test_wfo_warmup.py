"""Causal warm-up and scoring boundaries (5-second bars, no resampling)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('vectorbtpro')

from config import WFOSettings
from wfo import walk_forward_optimization


class Trades:
    win_rate = 0.0

    def __len__(self):
        return 0

    def count(self):
        return 0

    def stats(self):
        return {}


class Portfolio:
    def __init__(self, score):
        self.total_return = score / 100
        self.sharpe_ratio = score
        self.max_drawdown = 0.05
        self.calmar_ratio = score
        self.sortino_ratio = score
        self.trades = Trades()


class RollingAdapter:
    def __init__(self):
        self.calls = []

    def run_backtest(self, df, params, timeframe='5s', return_portfolio=True, trade_start=0):
        assert timeframe == '5s'
        indicator = df['Close'].rolling(20, min_periods=20).mean()
        scored = indicator.iloc[trade_start:]
        self.calls.append((df.index[-1], df.index[trade_start], scored.index, scored.iloc[0]))
        score = float(scored.fillna(0).sum())
        if return_portfolio:
            return Portfolio(score)
        if isinstance(params['length'], np.ndarray):
            return np.repeat(score, len(params['length']))
        return score


def run(df, start, adapter, **settings):
    return walk_forward_optimization(
        df, param_grid={'length': [20]},
        metrics_info={'metric1_name': 'total_return', 'metric2_name': 'sharpe_ratio',
                      'weight_metric1': 1.0, 'weight_metric2': 0.0},
        timeframe='5s',
        settings=WFOSettings(n_windows=2, train_size=0.5, selection_method='raw_max',
                             optimization_method='grid', **settings),
        strategy_adapter=adapter, selected_start=start,
    )


def test_warmup_and_causal_scoring():
    idx = pd.date_range('2024-01-01', periods=120, freq='5s')
    full = pd.DataFrame({'Close': np.arange(120, dtype=float) + 100}, index=idx)
    start = idx[30]
    adapter = RollingAdapter()
    result = run(full, start, adapter, holdout_fraction=0.1)
    selected = full.iloc[30:]
    assert np.isnan(selected['Close'].rolling(20, min_periods=20).mean().iloc[0])
    first = adapter.calls[0]
    assert first[1] == start
    assert np.isfinite(first[3])
    assert first[3] == pytest.approx(full['Close'].iloc[11:31].mean())
    assert all(call[0] <= call[2][-1] for call in adapter.calls)
    assert all(call[1] >= start for call in adapter.calls)
    assert result['holdout']['start'] == idx[111]
    assert result['window_results'][0]['window_info']['start_date'] == start
    assert result['window_results'][-1]['window_info']['end_date'] == idx[110]
    # Portfolio starts at the selected bar: prefix prices cannot contribute to P&L.
    assert result['in_sample_performance'][0]['return'] == pytest.approx(
        full['Close'].rolling(20).mean().iloc[30:50].sum())


def test_future_perturbation_does_not_change_earlier_signals():
    idx = pd.date_range('2024-01-01', periods=120, freq='5s')
    data = pd.DataFrame({'Close': np.arange(120, dtype=float) + 100}, index=idx)
    original, changed = RollingAdapter(), RollingAdapter()
    a = run(data, idx[30], original)
    modified = data.copy()
    modified.loc[idx[95]:, 'Close'] += 100000
    b = run(modified, idx[30], changed)
    for before, after in zip(original.calls, changed.calls):
        if before[0] < idx[95]:
            assert before[3] == after[3]
    assert a['window_results'][0]['optimization_trials'] == b['window_results'][0]['optimization_trials']
    assert a['in_sample_performance'][0] == b['in_sample_performance'][0]
    assert a['out_of_sample_performance'][0] == b['out_of_sample_performance'][0]
