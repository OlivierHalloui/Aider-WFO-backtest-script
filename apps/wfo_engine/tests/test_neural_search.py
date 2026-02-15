"""Unit tests for the neural_search.py module."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from neural_search import (
    _safe_float,
    _match_prev_value_to_candidates,
    _build_prev_best_grid,
    NeuralSearchGuide,
)


# ---------------------------------------------------------------------------
# _safe_float
# ---------------------------------------------------------------------------

class TestSafeFloat:
    def test_regular_float(self):
        assert _safe_float(3.14) == 3.14

    def test_numpy_float(self):
        assert _safe_float(np.float64(2.0)) == 2.0

    def test_bool_true(self):
        assert _safe_float(True) == 1.0

    def test_bool_false(self):
        assert _safe_float(False) == 0.0

    def test_nan_returns_default(self):
        assert _safe_float(float('nan')) == 0.0

    def test_inf_returns_default(self):
        assert _safe_float(float('inf')) == 0.0

    def test_none_returns_default(self):
        assert _safe_float(None) == 0.0

    def test_string_returns_default(self):
        assert _safe_float("abc") == 0.0

    def test_custom_default(self):
        assert _safe_float(None, default=-5.0) == -5.0


# ---------------------------------------------------------------------------
# _match_prev_value_to_candidates
# ---------------------------------------------------------------------------

class TestMatchPrevValueToCandidates:
    def test_exact_match(self):
        assert _match_prev_value_to_candidates(10, [5, 10, 15]) == 10

    def test_no_match_returns_none(self):
        assert _match_prev_value_to_candidates(99, [1, 2, 3]) is None

    def test_none_candidates_returns_none(self):
        assert _match_prev_value_to_candidates(5, None) is None

    def test_bool_match(self):
        assert _match_prev_value_to_candidates(True, [False, True]) is True

    def test_numpy_bool_match(self):
        result = _match_prev_value_to_candidates(np.bool_(True), [False, True])
        assert result is True

    def test_numeric_close_match(self):
        # Float precision: 1.0000000000000002 should match 1.0
        result = _match_prev_value_to_candidates(1.0 + 1e-15, [0.5, 1.0, 1.5])
        assert result == 1.0

    def test_numpy_int_to_float_match(self):
        result = _match_prev_value_to_candidates(np.int64(2), [1.0, 2.0, 3.0])
        assert result == 2.0

    def test_empty_candidates(self):
        assert _match_prev_value_to_candidates(5, []) is None

    def test_string_no_numeric_fallback(self):
        # Strings don't participate in numeric fallback
        assert _match_prev_value_to_candidates("abc", [1, 2, 3]) is None


# ---------------------------------------------------------------------------
# _build_prev_best_grid
# ---------------------------------------------------------------------------

class TestBuildPrevBestGrid:
    def test_all_params_matched(self):
        base_grid = {'a': [1, 2, 3], 'b': [10, 20, 30]}
        prev_best = {'a': 2, 'b': 20}
        guided, info = _build_prev_best_grid(base_grid, prev_best)
        assert guided['a'] == [2]
        assert guided['b'] == [20]
        assert info['fixed_params'] == 2
        assert info['guided_combinations'] == 1

    def test_partial_match(self):
        base_grid = {'a': [1, 2, 3], 'b': [10, 20, 30]}
        prev_best = {'a': 2, 'b': 99}  # b not in candidates
        guided, info = _build_prev_best_grid(base_grid, prev_best)
        assert guided['a'] == [2]
        assert guided['b'] == [10, 20, 30]  # full list preserved
        assert info['fixed_params'] == 1

    def test_no_prev_params(self):
        base_grid = {'a': [1, 2, 3]}
        prev_best = {}
        guided, info = _build_prev_best_grid(base_grid, prev_best)
        assert guided['a'] == [1, 2, 3]
        assert info['fixed_params'] == 0

    def test_none_prev_params(self):
        base_grid = {'a': [1, 2]}
        guided, info = _build_prev_best_grid(base_grid, None)
        assert guided['a'] == [1, 2]
        assert info['fixed_params'] == 0

    def test_reduction_ratio(self):
        base_grid = {'a': [1, 2, 3], 'b': [10, 20]}
        prev_best = {'a': 1, 'b': 10}
        _, info = _build_prev_best_grid(base_grid, prev_best)
        # baseline = 6, guided = 1
        assert info['baseline_combinations'] == 6
        assert info['guided_combinations'] == 1
        assert info['reduction_ratio'] == pytest.approx(5.0 / 6.0)

    def test_empty_grid(self):
        guided, info = _build_prev_best_grid({}, {'a': 1})
        assert guided == {}
        assert info['baseline_combinations'] == 0
        assert info['guided_combinations'] == 0


# ---------------------------------------------------------------------------
# NeuralSearchGuide — instantiation and basic methods
# ---------------------------------------------------------------------------

class TestNeuralSearchGuide:
    @staticmethod
    def _make_settings(**overrides):
        """Create a mock settings object with default neural search attributes."""
        defaults = {
            'nn_max_records': 200000,
            'nn_min_samples': 5,  # low for testing
            'nn_candidate_pool_size': 50,
            'nn_top_k': 10,
            'nn_exploration_ratio': 0.15,
            'nn_hidden_size': 8,
            'nn_epochs': 5,
            'nn_learning_rate': 0.01,
            'nn_l2': 1e-4,
            'random_state': 42,
        }
        defaults.update(overrides)
        settings = MagicMock()
        for k, v in defaults.items():
            setattr(settings, k, v)
        return settings

    def test_instantiation(self):
        grid = {'a': [1, 2, 3], 'b': [0.1, 0.2]}
        settings = self._make_settings()
        guide = NeuralSearchGuide(grid, settings)
        assert guide.param_keys == ['a', 'b']
        assert guide.trained is False
        assert guide.W1 is None

    def test_encode_value_numeric(self):
        grid = {'x': [1, 2]}
        guide = NeuralSearchGuide(grid, self._make_settings())
        assert guide._encode_value('x', 5) == 5.0

    def test_encode_value_bool(self):
        grid = {'flag': [True, False]}
        guide = NeuralSearchGuide(grid, self._make_settings())
        assert guide._encode_value('flag', True) == 1.0
        assert guide._encode_value('flag', False) == 0.0

    def test_encode_value_string_categorical(self):
        grid = {'mode': ['fast', 'slow']}
        guide = NeuralSearchGuide(grid, self._make_settings())
        v1 = guide._encode_value('mode', 'fast')
        v2 = guide._encode_value('mode', 'slow')
        assert v1 != v2  # Different categories get different codes

    def test_get_parameter_weights_untrained(self):
        grid = {'a': [1, 2]}
        guide = NeuralSearchGuide(grid, self._make_settings())
        assert guide.get_parameter_weights() == {}

    def test_predict_scores_untrained_returns_none(self):
        grid = {'a': [1, 2]}
        guide = NeuralSearchGuide(grid, self._make_settings())
        assert guide.predict_scores([{'a': 1}]) is None

    def test_build_guided_grid_untrained(self):
        grid = {'a': [1, 2, 3]}
        guide = NeuralSearchGuide(grid, self._make_settings())
        result_grid, info = guide.build_guided_grid(grid)
        assert result_grid == grid
        assert info['enabled'] is False
        assert info['reason'] == 'insufficient_samples'

    def test_update_and_train(self):
        grid = {'a': [1, 2, 3, 4, 5], 'b': [0.1, 0.2, 0.3]}
        settings = self._make_settings(nn_min_samples=5, nn_epochs=3)
        guide = NeuralSearchGuide(grid, settings)

        # Create synthetic optimization results with enough rows
        rng = np.random.default_rng(0)
        n = 10
        df = pd.DataFrame({
            'a': rng.choice([1, 2, 3, 4, 5], size=n),
            'b': rng.choice([0.1, 0.2, 0.3], size=n),
            'combined_score': rng.uniform(0, 1, size=n),
        })
        guide.update(df, grid)
        assert guide.trained is True
        assert guide.W1 is not None

    def test_predict_after_training(self):
        grid = {'a': [1, 2, 3], 'b': [0.1, 0.2]}
        settings = self._make_settings(nn_min_samples=5, nn_epochs=3)
        guide = NeuralSearchGuide(grid, settings)

        rng = np.random.default_rng(1)
        n = 10
        df = pd.DataFrame({
            'a': rng.choice([1, 2, 3], size=n),
            'b': rng.choice([0.1, 0.2], size=n),
            'combined_score': rng.uniform(0, 1, size=n),
        })
        guide.update(df, grid)
        scores = guide.predict_scores([{'a': 2, 'b': 0.1}])
        assert scores is not None
        assert len(scores) == 1
        assert np.isfinite(scores[0])

    def test_get_parameter_weights_after_training(self):
        grid = {'a': [1, 2, 3], 'b': [0.1, 0.2]}
        settings = self._make_settings(nn_min_samples=5, nn_epochs=3)
        guide = NeuralSearchGuide(grid, settings)

        rng = np.random.default_rng(2)
        n = 10
        df = pd.DataFrame({
            'a': rng.choice([1, 2, 3], size=n),
            'b': rng.choice([0.1, 0.2], size=n),
            'combined_score': rng.uniform(0, 1, size=n),
        })
        guide.update(df, grid)
        weights = guide.get_parameter_weights()
        assert set(weights.keys()) == {'a', 'b'}
        assert all(0.0 <= w <= 1.0 for w in weights.values())
        assert pytest.approx(sum(weights.values()), abs=1e-6) == 1.0

    def test_build_guided_grid_after_training(self):
        grid = {'a': [1, 2, 3, 4, 5], 'b': [0.1, 0.2, 0.3, 0.4, 0.5]}
        settings = self._make_settings(nn_min_samples=5, nn_epochs=3, nn_top_k=5)
        guide = NeuralSearchGuide(grid, settings)

        rng = np.random.default_rng(3)
        n = 15
        df = pd.DataFrame({
            'a': rng.choice(grid['a'], size=n),
            'b': rng.choice(grid['b'], size=n),
            'combined_score': rng.uniform(0, 1, size=n),
        })
        guide.update(df, grid)
        guided_grid, info = guide.build_guided_grid(grid)
        assert info['enabled'] is True
        assert info['trained'] is True
        # Guided grid should have equal or fewer values per key
        for key in grid:
            assert len(guided_grid[key]) <= len(grid[key])
            assert len(guided_grid[key]) >= 1

    def test_update_with_empty_dataframe_noop(self):
        grid = {'a': [1, 2]}
        guide = NeuralSearchGuide(grid, self._make_settings())
        guide.update(pd.DataFrame(), grid)
        assert guide.trained is False

    def test_update_with_none_noop(self):
        grid = {'a': [1, 2]}
        guide = NeuralSearchGuide(grid, self._make_settings())
        guide.update(None, grid)
        assert guide.trained is False

    def test_update_no_combined_score_column(self):
        grid = {'a': [1, 2]}
        guide = NeuralSearchGuide(grid, self._make_settings())
        df = pd.DataFrame({'a': [1, 2], 'other': [0.5, 0.6]})
        guide.update(df, grid)
        assert guide.trained is False

    def test_records_truncation(self):
        grid = {'a': [1, 2]}
        settings = self._make_settings(nn_max_records=5, nn_min_samples=2, nn_epochs=1)
        guide = NeuralSearchGuide(grid, settings)

        rng = np.random.default_rng(4)
        df = pd.DataFrame({
            'a': rng.choice([1, 2], size=10),
            'combined_score': rng.uniform(0, 1, size=10),
        })
        guide.update(df, grid)
        assert len(guide.records) <= 5
        assert len(guide.targets) <= 5
