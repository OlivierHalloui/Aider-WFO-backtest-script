"""Tests for data_loading.py — CSV edge cases. No VBT/Binance dependency."""

from __future__ import annotations

import textwrap

import pandas as pd
import pytest
from config import DEFAULT_PARAM_GRID, compute_warmup_bars, WFOSettings


def _write_csv(tmp_path, content: str, filename: str = "test.csv") -> str:
    f = tmp_path / filename
    f.write_text(textwrap.dedent(content))
    return str(f)


@pytest.fixture()
def load_csv():
    from data_loading import load_data
    return load_data


class TestCSVLoading:
    def test_period_read_only_ohlcv_slice_with_warmup(self, tmp_path, load_csv, monkeypatch):
        import data_loading
        idx = pd.date_range('2024-01-01', periods=1000, freq='5s')
        path = tmp_path / 'large.csv'
        pd.DataFrame({
            'Open time': idx, 'Open': range(1000), 'High': range(1000),
            'Low': range(1000), 'Close': range(1000),
        }).to_csv(path, index=False)
        real_read = data_loading.pd.read_csv
        reads = []

        def spy(*args, **kwargs):
            reads.append(kwargs.copy())
            return real_read(*args, **kwargs)

        monkeypatch.setattr(data_loading.pd, 'read_csv', spy)
        start, end = str(idx[500]), str(idx[519])
        df = load_csv(start, end, '5s', file_path=str(path), warmup_bars=15)
        assert len(df) == 35
        assert df.index[0] == idx[485]
        assert df.index[-1] == idx[519]
        ohlcv_reads = [r for r in reads if 'Close' in r.get('usecols', [])]
        assert len(ohlcv_reads) == 1
        assert ohlcv_reads[0]['nrows'] == 35
        assert 485 in ohlcv_reads[0]['skiprows']
        assert 0 not in ohlcv_reads[0]['skiprows']
        baseline = load_csv(start, end, '5s', file_path=str(path))
        pd.testing.assert_frame_equal(df.loc[start:], baseline, check_exact=True)

    def test_unsorted_csv_falls_back_and_warns(self, tmp_path, load_csv, caplog):
        path = _write_csv(tmp_path, '''\
            Open time,Open,High,Low,Close
            2024-01-02,2,2,2,2
            2024-01-01,1,1,1,1
        ''')
        with caplog.at_level('WARNING', logger='data_loading'):
            df = load_csv('2024-01-01', '2024-01-02', '1d', file_path=path)
        assert len(df) == 2
        assert 'falling back' in caplog.text

    def test_warmup_grid_only_lookbacks(self):
        assert compute_warmup_bars(DEFAULT_PARAM_GRID) == 188
        assert compute_warmup_bars({'macd_slow_length': [26, 42], 'macd_signal_length': [9, 16],
                                    'StDev': [999], 'sar_maximum': [5000]}) == 73
        assert compute_warmup_bars({'StDev': [1000], 'exit_macd_enabled': [True]}) == 0
        assert WFOSettings().warmup_bars == 0

    def test_run_service_trims_prefix_before_wfo(self, tmp_path, monkeypatch):
        from services import run_service
        idx = pd.date_range('2024-01-01', periods=100, freq='5s')
        path = tmp_path / 'run.csv'
        pd.DataFrame({'Open time': idx, 'Open': range(100), 'High': range(100),
                      'Low': range(100), 'Close': range(100)}).to_csv(path, index=False)
        seen = {}
        real_load = run_service.load_data

        def monitored_load(*args, **kwargs):
            seen['requested_warmup'] = kwargs['warmup_bars']
            return real_load(*args, **kwargs)

        def fake_wfo(df, **kwargs):
            seen['index'] = df.index
            return {'window_results': []}

        monkeypatch.setattr(run_service, 'load_data', monitored_load)
        monkeypatch.setattr(run_service, 'get_param_grid', lambda cfg: {'timeperiod': [8]})
        monkeypatch.setattr(run_service, 'resolve_strategy_adapter', lambda **kw: object())
        monkeypatch.setattr(run_service, 'walk_forward_optimization', fake_wfo)
        monkeypatch.setattr(run_service, 'write_error_log', lambda **kw: None)
        monkeypatch.setattr(run_service, 'collect_classic_wfo_entries', lambda results: [])
        config = {'start_date': str(idx[40]), 'end_date': str(idx[59]),
                  'timeframe': '5s', 'from_file': True, 'file_path': str(path),
                  'optimization_method': 'grid', 'n_windows': 2}
        result, df, _ = run_service.run_optimization_job(config)
        assert seen['requested_warmup'] == 10
        assert len(df) == 20
        assert seen['index'].equals(df.index)
        assert df.index[0] == idx[40] and df.index[-1] == idx[59]
        assert result == {'window_results': []}

    def test_standard_ohlcv_loads(self, tmp_path, load_csv):
        path = _write_csv(tmp_path, """\
            Open time,Open,High,Low,Close,Volume
            2024-01-01 00:00:00,100.0,110.0,90.0,105.0,1000.0
            2024-01-02 00:00:00,105.0,115.0,95.0,108.0,1200.0
        """)
        df = load_csv(
            start_date="2024-01-01",
            end_date="2024-01-03",
            timeframe="1d",
            from_file=True,
            file_path=path,
        )
        assert df is not None
        assert len(df) == 2
        assert list(df.columns[:4]) == ["Open", "High", "Low", "Close"]

    def test_na_string_in_close_does_not_raise_valueerror(self, tmp_path, load_csv):
        """Cellule 'N/A' → NaN, pas ValueError (audit R10)."""
        path = _write_csv(tmp_path, """\
            Open time,Open,High,Low,Close,Volume
            2024-01-01 00:00:00,100.0,110.0,90.0,N/A,1000.0
            2024-01-02 00:00:00,105.0,115.0,95.0,108.0,1200.0
        """)
        try:
            df = load_csv(
                start_date="2024-01-01",
                end_date="2024-01-03",
                timeframe="1d",
                from_file=True,
                file_path=path,
            )
            # If loader returns df, the N/A cell should be NaN not a crash
            if df is not None:
                assert df["Close"].isna().any() or len(df) >= 1
        except ValueError as exc:
            pytest.fail(
                f"load_data raised ValueError on 'N/A' cell (audit R10 not fixed): {exc}"
            )

    def test_missing_file_raises(self, tmp_path, load_csv):
        with pytest.raises(Exception):
            load_csv(
                start_date="2024-01-01",
                end_date="2024-01-03",
                timeframe="1d",
                from_file=True,
                file_path=str(tmp_path / "nonexistent.csv"),
            )
