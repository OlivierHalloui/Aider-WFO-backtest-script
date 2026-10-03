import numpy as np
import pytest

from advanced_optimizers import GridEncoder, bads_optimize, turbo_optimize

# Objective with a known optimum: a=3, b=0.7, c=True -> score 1.0
GRID = {
    'a': [0, 1, 2, 3, 4, 5],          # int, optimum 3
    'b': [0.1, 0.3, 0.5, 0.7, 0.9],   # float, optimum 0.7
    'c': [True, False],               # bool, optimum True
}
FIXED = {'z': 1.0}
METRICS = {'metric1_name': 'x'}
OPT = {'a': 3, 'b': 0.7, 'c': True}


def evaluate(params):
    assert set(FIXED) <= set(params), "fixed params missing from evaluated dict"
    assert set(METRICS) <= set(params), "metrics_info missing from evaluated dict"
    return -((params['a'] - 3) ** 2) - ((params['b'] - 0.7) ** 2) + (1.0 if params['c'] else 0.0)


class TestGridEncoder:
    def test_encode_decode_roundtrip(self):
        enc = GridEncoder.from_grid(GRID)
        assert enc.decode(enc.encode(OPT)) == OPT

    def test_dim_and_single_value_dims(self):
        enc = GridEncoder.from_grid({'x': [5], 'y': [1, 2]})
        assert enc.dim == 2
        assert enc.decode(np.array([0.9, 0.0])) == {'x': 5, 'y': 1}

    def test_decode_snaps_to_nearest_candidate(self):
        enc = GridEncoder.from_grid(GRID)
        assert enc.decode(np.array([0.6, 0.8, 0.0]))['b'] == 0.7


class TestTurbo:
    def test_finds_known_optimum(self):
        rows = turbo_optimize(evaluate, GRID, FIXED, METRICS,
                              n_evals=40, n_init=10, rng=np.random.default_rng(0))
        best = max(rows, key=lambda r: r['combined_score'])
        assert (best['a'], best['b'], best['c']) == (3, 0.7, True)
        assert best['combined_score'] == pytest.approx(1.0)

    def test_respects_evaluation_budget(self):
        rows = turbo_optimize(evaluate, GRID, FIXED, METRICS,
                              n_evals=30, n_init=10, rng=np.random.default_rng(0))
        assert len(rows) <= 30

    def test_rows_carry_fixed_metrics_and_score(self):
        rows = turbo_optimize(evaluate, GRID, FIXED, METRICS,
                              n_evals=15, n_init=5, rng=np.random.default_rng(0))
        r = rows[0]
        assert r['z'] == 1.0 and r['metric1_name'] == 'x' and 'combined_score' in r


class TestBads:
    def test_finds_known_optimum(self):
        rows = bads_optimize(evaluate, GRID, FIXED, METRICS,
                             n_evals=40, n_init=8, rng=np.random.default_rng(0))
        best = max(rows, key=lambda r: r['combined_score'])
        assert (best['a'], best['b'], best['c']) == (3, 0.7, True)
        assert best['combined_score'] == pytest.approx(1.0)

    def test_respects_evaluation_budget(self):
        rows = bads_optimize(evaluate, GRID, FIXED, METRICS,
                             n_evals=30, n_init=8, rng=np.random.default_rng(0))
        assert len(rows) <= 30


class TestRobustness:
    def test_nan_score_does_not_poison_the_gp(self):
        # One config returns NaN (a no-trade backtest). The GP must stay active
        # and still find the optimum instead of degrading to random search.
        def evaluate_with_nan(params):
            if params['a'] == 1:
                return float('nan')
            return evaluate(params)

        rows = turbo_optimize(evaluate_with_nan, GRID, FIXED, METRICS,
                              n_evals=40, n_init=10, rng=np.random.default_rng(0))
        best = max(rows, key=lambda r: r['combined_score'])
        assert (best['a'], best['b'], best['c']) == (3, 0.7, True)

    def test_reproducible_with_same_seed(self):
        r1 = turbo_optimize(evaluate, GRID, FIXED, METRICS,
                            n_evals=30, n_init=10, rng=np.random.default_rng(0))
        r2 = turbo_optimize(evaluate, GRID, FIXED, METRICS,
                            n_evals=30, n_init=10, rng=np.random.default_rng(0))
        assert r1 == r2, "TuRBO is not reproducible at a fixed seed"

        b1 = bads_optimize(evaluate, GRID, FIXED, METRICS,
                           n_evals=30, n_init=8, rng=np.random.default_rng(0))
        b2 = bads_optimize(evaluate, GRID, FIXED, METRICS,
                           n_evals=30, n_init=8, rng=np.random.default_rng(0))
        assert b1 == b2, "BADS is not reproducible at a fixed seed"

    def test_evaluated_configs_are_unique(self):
        rows = turbo_optimize(evaluate, GRID, FIXED, METRICS,
                              n_evals=40, n_init=10, rng=np.random.default_rng(0))
        keys = {(r['a'], r['b'], r['c']) for r in rows}
        assert len(keys) == len(rows), "duplicate configs wasted the evaluation budget"

        rows = bads_optimize(evaluate, GRID, FIXED, METRICS,
                             n_evals=40, n_init=8, rng=np.random.default_rng(0))
        keys = {(r['a'], r['b'], r['c']) for r in rows}
        assert len(keys) == len(rows), "duplicate configs wasted the evaluation budget"
