import importlib.util
import os
import sys

import pytest


# Ensure WFO Engine modules are importable from the dedicated app folder.
TEST_DIR = os.path.dirname(__file__)
WFO_ENGINE_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
sys.path.append(WFO_ENGINE_DIR)

HAS_VBT = importlib.util.find_spec("vectorbtpro") is not None
pytestmark = pytest.mark.skipif(not HAS_VBT, reason="vectorbtpro is required for these tests")

if HAS_VBT:
    from main import get_param_grid


def test_get_param_grid_includes_default_for_unselected_params():
    """Ensure non-selected strategy params are injected as fixed defaults."""
    config = {
        "selected_params": ["timeperiod"],
        "timeperiod_min": 10,
        "timeperiod_max": 10,
        "timeperiod_step": 1,
    }
    grid = get_param_grid(config)

    assert "timeperiod" in grid
    assert grid["timeperiod"] == [10]
    assert "Nb_bars_above" in grid
    assert grid["Nb_bars_above"] == [5]


def test_get_param_grid_adds_fixed_execution_settings():
    """Execution settings must always be present as fixed singleton values."""
    config = {"selected_params": ["timeperiod"]}
    grid = get_param_grid(config)

    assert grid["order_sizing_mode"] == ["percent_equity"]
    assert grid["order_fixed_cash"] == [10000.0]
    assert grid["fees_pct"] == [0.0]
