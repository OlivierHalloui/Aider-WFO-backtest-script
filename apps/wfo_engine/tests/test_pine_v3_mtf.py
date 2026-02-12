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
    from pine_v3.mtf import request_security_series, request_security_signal


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
