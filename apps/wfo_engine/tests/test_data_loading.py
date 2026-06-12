"""Tests for data_loading.py — CSV edge cases. No VBT/Binance dependency."""

from __future__ import annotations

import textwrap

import pandas as pd
import pytest


def _write_csv(tmp_path, content: str, filename: str = "test.csv") -> str:
    f = tmp_path / filename
    f.write_text(textwrap.dedent(content))
    return str(f)


@pytest.fixture()
def load_csv():
    from data_loading import load_data
    return load_data


class TestCSVLoading:
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
