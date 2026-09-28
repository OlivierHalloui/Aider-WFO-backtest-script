"""Executable warm-up contracts without the commercial VectorBT dependency."""

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd

from config import compute_warmup_bars


ENGINE = Path(__file__).resolve().parents[1]


def _load(name, path, patcher):
    spec = importlib.util.spec_from_file_location(name, ENGINE / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    patcher.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_resolve_generated_pine_warmup_new_and_persisted(tmp_path, monkeypatch):
    calls = []

    class Runtime:
        def __init__(self, strategy_id, runtime_config, strategy_spec):
            self.strategy_id = strategy_id
            self.runtime_config = runtime_config
            self.strategy_spec = strategy_spec

        def run_backtest(self, df, params, timeframe='5s', return_portfolio=True, trade_start=0):
            calls.append((trade_start, df.index[trade_start], self.runtime_config, self.strategy_spec))
            return df['Close'].rolling(20).mean().iloc[trade_start]

    runtime = types.ModuleType('pine_v3.runtime_adapter')
    runtime.__dict__['GeneratedPineRuntimeAdapter'] = Runtime
    runtime.__dict__['PineStrategyTestAdapter'] = type('PineStrategyTestAdapter', (), {})
    runtime.__dict__['supports_strategy_test_runtime'] = lambda **kwargs: False
    strategy = types.ModuleType('strategy')
    strategy.__dict__['create_signal_generators'] = lambda *args, **kwargs: None
    strategy.__dict__['run_backtest'] = lambda *args, **kwargs: None
    package = types.ModuleType('pine_v3')
    package.__path__ = [str(ENGINE / 'pine_v3')]
    with monkeypatch.context() as patcher:
        patcher.setitem(sys.modules, 'pine_v3', package)
        patcher.setitem(sys.modules, 'pine_v3.runtime_adapter', runtime)
        patcher.setitem(sys.modules, 'strategy', strategy)
        adapters = _load('warmup_test_strategy_adapters', 'strategy_adapters.py', patcher)
        codegen = _load('warmup_test_codegen', 'pine_v3/codegen.py', patcher)
        spec = {'strategy': {'id': 'pine_warmup_test'}}
        source = codegen.build_generated_strategy_source(spec)
        assert 'return_portfolio: bool = True, trade_start: int = 0' in source
        assert 'return_portfolio=return_portfolio, trade_start=trade_start' in source
        idx = pd.date_range('2024-01-01', periods=60, freq='5s')
        df = pd.DataFrame({'Close': np.arange(60, dtype=float)}, index=idx)
        for legacy in (False, True):
            module_path = tmp_path / ('legacy.py' if legacy else 'new.py')
            module_path.write_text(
                source.replace(', trade_start: int = 0):', '):').replace(
                    ', trade_start=trade_start)', ')',
                ) if legacy else source,
                encoding='utf-8',
            )
            config = {'pine_generated_module_path': str(module_path),
                      'pine_import_mapping': {'BBT1': 'local'}}
            adapter = adapters.resolve_strategy_adapter(
                strategy_mode='pine_imported', strategy_id='pine_warmup_test', config=config,
            )
            score = adapter.run_backtest(df, {}, trade_start=30)
            assert score == df['Close'].iloc[11:31].mean()
            assert calls[-1][0:2] == (30, idx[30])
            assert calls[-1][2]['pine_import_mapping'] == {'BBT1': 'local'}
            assert calls[-1][3] == spec


def test_stagewise_selected_period_excludes_causal_prefix(tmp_path, monkeypatch):
    vbt = types.ModuleType('vectorbtpro')
    fake_adapters = types.ModuleType('strategy_adapters')
    fake_adapters.__dict__['resolve_strategy_adapter'] = lambda **kwargs: object()
    fake_wfo = types.ModuleType('wfo')
    calls = []

    def walk_forward_optimization(df, **kwargs):
        selected_start = pd.Timestamp(kwargs['selected_start']).tz_localize('UTC')
        selected = df.loc[selected_start:]
        calls.append((len(df), len(selected), df.index[0], selected.index[0]))
        return {'best_params': [{'timeperiod': 30}], 'window_results': [],
                'in_sample_performance': [], 'out_of_sample_performance': []}

    fake_wfo.__dict__['walk_forward_optimization'] = walk_forward_optimization
    with monkeypatch.context() as patcher:
        patcher.setitem(sys.modules, 'vectorbtpro', vbt)
        patcher.setitem(sys.modules, 'strategy_adapters', fake_adapters)
        patcher.setitem(sys.modules, 'wfo', fake_wfo)
        data_loading = _load('warmup_test_data_loading', 'data_loading.py', patcher)
        patcher.setitem(sys.modules, 'data_loading', data_loading)
        stagewise = _load('warmup_test_stagewise', 'stagewise_optimizer.py', patcher)
        idx = pd.date_range('2024-01-01', periods=18007, freq='5s')
        path = tmp_path / 'bars.csv'
        pd.DataFrame({'Open time': idx, 'Open': np.arange(len(idx)),
                      'High': np.arange(len(idx)), 'Low': np.arange(len(idx)),
                      'Close': np.arange(len(idx))}).to_csv(path, index=False)
        warmup = compute_warmup_bars({'timeperiod': (10, 30, 1)})
        df = stagewise._load_csv(str(path), '2024-01-02', '2024-01-02', warmup_bars=warmup)
        assert warmup == 287
        plan = [{'name': 'one', 'optimize': ['timeperiod'], 'method': 'grid',
                 'max_trials': 1, 'patience': 'Low', 'fixed_overrides': {}}]
        report = stagewise.run_stagewise_campaign(
            df, stage_plan=plan, output_dir=tmp_path / 'run', selected_start=idx[17280],
        )
        assert report['stages'][0]['status'] == 'OK'
        assert calls == [(1014, 727, pd.Timestamp(idx[16993], tz='UTC'),
                          pd.Timestamp(idx[17280], tz='UTC'))]
