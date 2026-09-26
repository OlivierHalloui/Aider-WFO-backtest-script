import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure WFO Engine modules are importable from the dedicated app folder.
TEST_DIR = os.path.dirname(__file__)
WFO_ENGINE_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
sys.path.append(WFO_ENGINE_DIR)

HAS_VBT = importlib.util.find_spec("vectorbtpro") is not None
pytestmark = pytest.mark.skipif(not HAS_VBT, reason="vectorbtpro is required for MTF tests")

if HAS_VBT:
    from data_loading import _to_pandas_freq
    from pine_v3.mtf import request_security_series, request_security_signal, resample_ohlcv


def _mock_ohlcv(n=12, freq="1h"):
    index = pd.date_range("2025-01-01", periods=n, freq=freq, tz="UTC")
    close = np.arange(100, 100 + n, dtype=float)
    return pd.DataFrame(
        {
            "Open": close - 0.2,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": np.ones(n, dtype=float),
        },
        index=index,
    )


def test_resample_pine_minutes_is_not_monthly():
    base = _mock_ohlcv(n=24, freq="1min")
    out = resample_ohlcv(base, "15m")

    assert out.index.equals(base.index[[0, 15]])
    assert out["Open"].tolist() == pytest.approx([99.8, 114.8])
    assert out["Close"].tolist() == pytest.approx([114.0, 123.0])
    assert out["Volume"].tolist() == pytest.approx([15.0, 9.0])


@pytest.mark.parametrize(
    ("timeframe", "expected"),
    [
        ("1m", "60s"),
        ("24h", "1D"),
        ("1440m", "1D"),
        ("48h", "2D"),
        ("1500m", "90000s"),
        ("1w", "1W"),
        ("1W", "1W"),
    ],
)
def test_pine_frequency_normalization(timeframe, expected):
    assert _to_pandas_freq(timeframe) == expected


def test_pine_month_is_month_end_not_minute():
    monthly_freq = _to_pandas_freq("1M")
    assert pd.tseries.frequencies.to_offset(monthly_freq) == pd.offsets.MonthEnd(1)
    assert _to_pandas_freq("1m") == "60s"

    base = _mock_ohlcv(n=40, freq="1D")
    out = resample_ohlcv(base, "1M")
    assert out.index.equals(pd.DatetimeIndex(["2025-01-31", "2025-02-28"], tz="UTC"))
    assert out["Close"].tolist() == pytest.approx([130.0, 139.0])


@pytest.mark.parametrize("timeframe", ["24h", "1440m"])
def test_day_expressed_in_short_units_is_not_monthly(timeframe):
    base = _mock_ohlcv(n=48, freq="1h")
    out = resample_ohlcv(base, timeframe)
    assert out.index.equals(base.index[[0, 24]])
    assert out["Close"].tolist() == pytest.approx([123.0, 147.0])


def test_pine_week_resamples_by_week():
    base = _mock_ohlcv(n=10, freq="1D")
    out = resample_ohlcv(base, "1w")
    assert out.index.equals(pd.DatetimeIndex(["2025-01-05", "2025-01-12"], tz="UTC"))


@pytest.mark.parametrize("timeframe", ["24h", "1440m", "1w", "1M"])
def test_request_security_passes_safe_calendar_frequency_to_vbt(monkeypatch, timeframe):
    import pine_v3.mtf as mtf

    captured = {}
    original_resampler = mtf.vbt.Resampler

    def capture_resampler(**kwargs):
        captured.update(kwargs)
        return original_resampler(**kwargs)

    monkeypatch.setattr(mtf.vbt, "Resampler", capture_resampler)
    base = _mock_ohlcv(n=40, freq="1D")
    request_security_series(
        base,
        timeframe=timeframe,
        expr_fn=lambda htf: htf["Close"],
        base_freq="1d",
    )
    assert captured["source_freq"] == _to_pandas_freq(timeframe)
    assert captured["target_freq"] == "1D"


def test_request_security_normalizes_both_resampler_frequencies(monkeypatch):
    import pine_v3.mtf as mtf

    captured = {}
    original_resampler = mtf.vbt.Resampler

    def capture_resampler(**kwargs):
        captured.update(kwargs)
        return original_resampler(**kwargs)

    monkeypatch.setattr(mtf.vbt, "Resampler", capture_resampler)
    base = _mock_ohlcv(n=30, freq="1min")
    out = request_security_series(
        base,
        timeframe="15m",
        expr_fn=lambda htf: htf["Close"],
        base_freq="1m",
    )

    assert captured["source_freq"] == "900s"
    assert captured["target_freq"] == "60s"
    assert out.iloc[:14].isna().all()
    assert float(out.iloc[14]) == 114.0
    assert float(out.iloc[28]) == 114.0
    assert float(out.iloc[29]) == 129.0


def test_request_security_closing_alignment_no_lookahead():
    base = _mock_ohlcv(n=12, freq="1h")
    # 4h close values are at 03:00, 07:00, 11:00 and then propagated.
    out = request_security_series(
        base,
        timeframe="4h",
        expr_fn=lambda htf: htf["Close"],
        timing="closing",
        gaps="off",
        base_freq="1h",
    )
    assert pd.isna(out.iloc[0])
    assert pd.isna(out.iloc[1])
    assert pd.isna(out.iloc[2])
    # First 4h close becomes known at 03:00.
    assert float(out.iloc[3]) == 103.0
    # Next segment uses previous known 4h close until new one at 07:00.
    assert float(out.iloc[4]) == 103.0
    assert float(out.iloc[6]) == 103.0
    assert float(out.iloc[7]) == 107.0


def test_request_security_opening_alignment():
    base = _mock_ohlcv(n=12, freq="1h")
    out = request_security_series(
        base,
        timeframe="4h",
        expr_fn=lambda htf: htf["Open"],
        timing="opening",
        gaps="off",
        base_freq="1h",
    )
    # First 4h open is known at 00:00 and propagated until 03:00.
    assert float(out.iloc[0]) == pytest.approx(99.8)
    assert float(out.iloc[3]) == pytest.approx(99.8)
    # Next 4h open starts at 04:00.
    assert float(out.iloc[4]) == pytest.approx(103.8)


def test_request_security_signal_sparse_with_gaps_on():
    base = _mock_ohlcv(n=15, freq="1min")
    # On 5m bars, this creates [True, False, True] then realigns to 1m.
    out = request_security_signal(
        base,
        timeframe="5min",
        signal_fn=lambda htf: pd.Series([True, False, True], index=htf.index[:3]),
        timing="closing",
        gaps="on",
        base_freq="1min",
    )
    assert out.dtype == bool
    # With gaps on, events should stay sparse (not propagated each minute).
    assert int(out.sum()) == 2
