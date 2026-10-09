"""Unit tests for services/quant_indicators.py (T1).

Synthetic-data tests covering the manifest, the blocking integrity check, the
indicators Q1–Q8, the OOS uncertainty block, the deterministic pre-verdict and
the §7 edge cases (0 trades, 1 trial, all duplicates, empty OOS, max score ≤ 0,
unmatched window).  No VectorBT / Streamlit dependency — only numpy + pandas.
"""

import math

import numpy as np
import pandas as pd
import pytest

from services import quant_indicators as q
from services.quant_indicators import (
    INDICATOR_VERSION,
    SCHEMA_VERSION,
    build_run_manifest,
    check_run_integrity,
    compute_oos_uncertainty,
    compute_pre_verdict,
    compute_q1,
    compute_q2,
    compute_q3,
    compute_q4,
    compute_q5,
    compute_q6,
    compute_q7,
    compute_q8,
    compute_quant_indicators,
    _avail,
    _norm_distance,
    _robust_z,
    _sharpe_from_returns,
)
from domain.serialization import sha256_json


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

def _trial(tp, sd, score):
    return {"timeperiod": tp, "StDev": sd, "combined_score": score}


def _window(window_id, trials, best_params, evaluations=None):
    return {
        "window_info": {
            "window": window_id,
            "start_date": f"2026-01-0{window_id} 00:00:00",
            "end_date": f"2026-02-0{window_id} 00:00:00",
            "in_sample_start": f"2026-01-0{window_id} 00:00:00",
            "in_sample_end": f"2026-01-0{window_id} 12:00:00",
            "out_sample_start": f"2026-01-0{window_id} 12:00:00",
            "out_sample_end": f"2026-02-0{window_id} 00:00:00",
        },
        "optimization_trials": trials,
        "optimization_trials_count": len(trials),
        "optimization_results": trials[:5],
        "evaluations": evaluations if evaluations is not None else len(trials),
        "best_params": best_params,
    }


def _perf(window_id, ret, sharpe, pqs=1.0, dd=-5.0, n_trades=30):
    return {
        "window": window_id,
        "return": ret,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "win_rate": 0.55,
        "pqs": pqs,
        "n_trades": n_trades,
    }


def make_results(
    n_windows=5,
    trials_per_window=6,
    oos_returns=None,
    oos_sharpes=None,
    best_params=None,
    settings=None,
    run_id="run-123",
):
    """Build a synthetic, well-formed wfo_results dict."""
    windows = []
    in_sample_performance = []
    out_of_sample_performance = []
    best_params_list = []

    default_trials = [
        _trial(10, 2.0, 10.0),
        _trial(10, 2.02, 9.5),
        _trial(10, 2.04, 9.0),
        _trial(10, 2.06, 8.5),
        _trial(10, 2.08, 8.0),
        _trial(11, 2.0, 7.0),
    ]

    for i in range(n_windows):
        wid = i + 1
        trials = default_trials[:trials_per_window]
        bp = best_params[i] if best_params else {"timeperiod": 10, "StDev": 2.0}
        windows.append(_window(wid, trials, bp, evaluations=len(trials)))
        in_sample_performance.append(_perf(wid, 8.0, 0.6))
        ret = oos_returns[i] if oos_returns else 5.0
        sh = oos_sharpes[i] if oos_sharpes else 0.4
        out_of_sample_performance.append(_perf(wid, ret, sh))
        best_params_list.append(bp)

    base_settings = {
        "n_windows": n_windows,
        "train_size": 0.5,
        "anchored": False,
        "optimization_method": "bayesian",
        "optimization_regime": "classic",
        "selection_method": "snv",
        "max_trials": 200,
        "neighbor_count": 5,
        "timeframe": "5s",
    }
    if settings:
        base_settings.update(settings)

    return {
        "run_id": run_id,
        "window_results": windows,
        "in_sample_performance": in_sample_performance,
        "out_of_sample_performance": out_of_sample_performance,
        "best_params": best_params_list,
        "settings": base_settings,
    }


def make_final_trades(n=20, pnl=(2.0, -1.0, 3.0)):
    """Synthetic final_trades DataFrame (pf.trades.records-like)."""
    rng = np.random.default_rng(7)
    pnls = rng.choice(pnl, size=n)
    records = []
    for i, p in enumerate(pnls):
        records.append({
            "return": p / 100.0,
            "pnl": p,
            "entry_price": 100.0,
            "size": 1.0,
            "entry_value": 100.0,
            "Direction": "Long",
        })
    return pd.DataFrame(records)


PARAM_GRID = {"timeperiod": (5, 15), "StDev": (1.0, 3.0)}


def _add_params_sha(results):
    """Inject the engine-side params fingerprint (sha256 of best_params) into
    the IS/OOS metric rows, mimicking what wfo.py now emits."""
    is_map = {r.get("window"): r for r in results.get("in_sample_performance", [])}
    oos_map = {r.get("window"): r for r in results.get("out_of_sample_performance", [])}
    for w in results.get("window_results", []):
        bp = w.get("best_params")
        wid = (w.get("window_info") or {}).get("window")
        sha = sha256_json(bp)
        w["params_sha"] = sha
        if wid in is_map:
            is_map[wid]["params_sha"] = sha
        if wid in oos_map:
            oos_map[wid]["params_sha"] = sha
    return results


# ---------------------------------------------------------------------------
# _avail / helpers
# ---------------------------------------------------------------------------

class TestAvailPolicy:
    def test_missing_is_none_with_raison(self):
        a = _avail(None, "indisponible")
        assert a["value"] is None
        assert a["available"] is False
        assert a["raison"] == "indisponible"

    def test_value_is_available(self):
        a = _avail(3.5)
        assert a["value"] == 3.5
        assert a["available"] is True
        assert a["raison"] is None


class TestSharpeFromReturns:
    def test_basic(self):
        rets = np.array([0.01, -0.02, 0.03, 0.01, 0.005])
        expected = float(rets.mean() / rets.std(ddof=1))
        assert _sharpe_from_returns(rets) == pytest.approx(expected)

    def test_none_returns_none(self):
        assert _sharpe_from_returns(None) is None

    def test_zero_variance_returns_none(self):
        assert _sharpe_from_returns(np.array([0.01, 0.01, 0.01, 0.01])) is None

    def test_insufficient_returns_none(self):
        assert _sharpe_from_returns(np.array([0.01])) is None

    def test_nan_filtered(self):
        rets = np.array([0.01, np.nan, 0.03, 0.01, 0.005])
        clean = rets[np.isfinite(rets)]
        assert _sharpe_from_returns(rets) == pytest.approx(clean.mean() / clean.std(ddof=1))


class TestNormDistance:
    def test_zero_for_identical(self):
        ranges = {"timeperiod": {"min": 5.0, "max": 15.0, "categorical": False}}
        assert _norm_distance({"timeperiod": 10}, {"timeperiod": 10}, ranges) == 0.0

    def test_normalized(self):
        ranges = {
            "timeperiod": {"min": 5.0, "max": 15.0, "categorical": False},
            "StDev": {"min": 1.0, "max": 3.0, "categorical": False},
        }
        a = {"timeperiod": 10, "StDev": 2.0}
        b = {"timeperiod": 12, "StDev": 2.0}
        # (1/2) * (|10-12|/10 + 0) = 0.1
        assert _norm_distance(a, b, ranges) == pytest.approx(0.1)

    def test_categorical_contributes_01(self):
        ranges = {"flag": {"min": 0.0, "max": 1.0, "categorical": True}}
        assert _norm_distance({"flag": True}, {"flag": True}, ranges) == 0.0
        assert _norm_distance({"flag": True}, {"flag": False}, ranges) == 1.0

    def test_fixed_dim_excluded(self):
        # no positive-range dims -> p == 0 -> None
        ranges = {"fixed": {"min": 1.0, "max": 1.0, "categorical": False}}
        assert _norm_distance({"fixed": 1.0}, {"fixed": 1.0}, ranges) is None

    def test_empty_ranges_none(self):
        assert _norm_distance({"a": 1}, {"a": 2}, {}) is None


class TestRobustZ:
    def test_defined_with_dispersion(self):
        S = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        z = _robust_z(6.0, S)
        assert z is not None and z > 0

    def test_undefined_for_small_sample(self):
        assert _robust_z(1.0, [1.0, 2.0, 3.0]) is None

    def test_undefined_for_zero_dispersion(self):
        assert _robust_z(5.0, [5.0, 5.0, 5.0, 5.0, 5.0]) is None


# ---------------------------------------------------------------------------
# 5.0 Manifest
# ---------------------------------------------------------------------------

class TestManifest:
    def test_structure(self):
        results = make_results()
        m = build_run_manifest(results, {"fees_pct": 0.1, "slippage_bps": 5.0})
        assert m["schema_version"] == SCHEMA_VERSION
        assert m["indicator_version"] == INDICATOR_VERSION
        assert m["run_id"] == "run-123"
        assert m["mode"] == "rolling"
        assert m["n_windows"] == 5
        assert m["n_windows_computed"] == 5
        assert m["optimization_method"] == "bayesian"
        assert m["selection_method"] == "snv"
        assert len(m["window_dates"]) == 5
        assert m["units"]["return"] == "return_pct"
        assert m["units"]["sharpe"] == "sharpe_per_bar"
        assert m["units"]["win_rate"] == "win_rate_pct"
        assert m["input_digest"] is not None

    def test_anchored_mode(self):
        results = make_results(settings={"anchored": True})
        m = build_run_manifest(results)
        assert m["mode"] == "anchored"

    def test_costs_included_net(self):
        results = make_results()
        m = build_run_manifest(results, {"fees_pct": 0.1, "slippage_bps": 0.0})
        assert m["costs"]["included"] is True
        assert m["costs"]["convention"] == "net_of_costs"

    def test_costs_gross(self):
        results = make_results()
        m = build_run_manifest(results, {"fees_pct": 0.0, "slippage_bps": 0.0})
        assert m["costs"]["included"] is False
        assert m["costs"]["convention"] == "gross"


# ---------------------------------------------------------------------------
# 5.0 Integrity
# ---------------------------------------------------------------------------

class TestIntegrity:
    CFG = {"fees_pct": 0.1, "slippage_bps": 0.0}

    def test_ok(self):
        results = make_results()
        ft = make_final_trades()
        integ = check_run_integrity(results, all_trials=None, final_trades=ft, config=self.CFG)
        assert integ["ok"] is True
        assert integ["status"] == "ok"
        assert integ["causes"] == []

    def test_costs_provenance_unknown_blocks(self):
        # no fees_pct / slippage_bps anywhere -> unknown convention -> non-evaluable
        results = make_results()
        integ = check_run_integrity(results, final_trades=make_final_trades(), config=None)
        assert integ["ok"] is False
        assert any("coûts" in c or "cout" in c for c in integ["causes"])

    def test_is_oos_both_empty_blocks(self):
        results = make_results(n_windows=1)
        results["in_sample_performance"] = []
        results["out_of_sample_performance"] = []
        integ = check_run_integrity(results, final_trades=make_final_trades(), config=self.CFG)
        assert integ["ok"] is False

    def test_missing_final_trades(self):
        results = make_results()
        integ = check_run_integrity(results, final_trades=None, config=self.CFG)
        assert integ["ok"] is False
        assert integ["status"] == "non_evaluable"
        assert any("final_trades" in c for c in integ["causes"])

    def test_empty_final_trades(self):
        results = make_results()
        integ = check_run_integrity(results, final_trades=pd.DataFrame(), config=self.CFG)
        assert integ["ok"] is False

    def test_no_windows(self):
        results = make_results()
        results["window_results"] = []
        integ = check_run_integrity(results, final_trades=make_final_trades(), config=self.CFG)
        assert integ["ok"] is False
        assert any("fenêtre" in c for c in integ["causes"])

    def test_is_oos_unpaired(self):
        results = make_results()
        results["out_of_sample_performance"] = results["out_of_sample_performance"][:-1]
        integ = check_run_integrity(results, final_trades=make_final_trades(), config=self.CFG)
        assert integ["ok"] is False
        assert any("apparié" in c for c in integ["causes"])

    def test_non_finite_value(self):
        results = make_results()
        results["out_of_sample_performance"][0]["return"] = float("inf")
        integ = check_run_integrity(results, final_trades=make_final_trades(), config=self.CFG)
        assert integ["ok"] is False
        assert any("non finie" in c for c in integ["causes"])

    def test_missing_best_params(self):
        results = make_results()
        results["window_results"][0]["best_params"] = None
        integ = check_run_integrity(results, final_trades=make_final_trades(), config=self.CFG)
        assert integ["ok"] is False
        assert any("best_params" in c for c in integ["causes"])

    def test_no_trials(self):
        results = make_results()
        for w in results["window_results"]:
            w["optimization_trials"] = []
            w["optimization_results"] = []
        integ = check_run_integrity(results, final_trades=make_final_trades(), config=self.CFG)
        assert integ["ok"] is False
        assert any("trial" in c for c in integ["causes"])


# ---------------------------------------------------------------------------
# Q1 — Budget
# ---------------------------------------------------------------------------

class TestQ1:
    def test_budget_counts(self):
        results = make_results(n_windows=3, trials_per_window=5)
        out = compute_q1(results)
        assert out["n_windows"] == 3
        assert out["completed"]["value"] == 15  # 5 valid scores x 3 windows
        assert out["launched"]["value"] == 15
        assert out["pruned_failed"]["value"] == 0
        assert out["valid_scores"]["value"] == 15

    def test_unique_configs_per_window(self):
        results = make_results(n_windows=1, trials_per_window=6)
        out = compute_q1(results)
        # 6 trials, all distinct configs
        assert out["unique_configs_per_window"]["1"] == 6


# ---------------------------------------------------------------------------
# Q2 — Diversité
# ---------------------------------------------------------------------------

class TestQ2:
    def test_no_duplicates(self):
        results = make_results(n_windows=1, trials_per_window=6)
        out = compute_q2(results)
        assert out["intra_window_duplicate_rate"]["value"] == 0.0
        assert out["inter_window_duplicate_rate"]["value"] == 0.0

    def test_inter_window_duplication_is_legitimate(self):
        # identical configs across windows -> intra 0, inter > 0 (expected, §5.1 Q2)
        results = make_results(n_windows=2, trials_per_window=6)
        out = compute_q2(results)
        assert out["intra_window_duplicate_rate"]["value"] == 0.0
        # 12 trials, 6 unique configs -> 6 inter-window dupes -> rate 0.5
        assert out["inter_window_duplicate_rate"]["value"] == pytest.approx(0.5)

    def test_all_duplicates_intra(self):
        # every window has the same single config repeated
        trials = [_trial(10, 2.0, 9.0), _trial(10, 2.0, 8.0), _trial(10, 2.0, 7.0)]
        results = make_results(n_windows=2, trials_per_window=3)
        for w in results["window_results"]:
            w["optimization_trials"] = trials
            w["optimization_trials_count"] = 3
            w["evaluations"] = 3
        out = compute_q2(results)
        # 3 trials -> 1 unique -> 2 dupes -> rate 2/3
        assert out["intra_window_duplicate_rate"]["value"] == pytest.approx(2 / 3)
        # inter-window: 2 windows, each 1 distinct config (the same one) ->
        # 1 config appearing in >1 window -> inter rate 1/2 (deduped per window)
        assert out["inter_window_duplicate_rate"]["value"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Q3 — Distribution
# ---------------------------------------------------------------------------

class TestQ3:
    def test_percentiles(self):
        results = make_results(n_windows=1, trials_per_window=6)
        out = compute_q3(results)
        scores = [10.0, 9.5, 9.0, 8.5, 8.0, 7.0]
        assert out["max"]["value"] == pytest.approx(10.0)
        assert out["p50"]["value"] == pytest.approx(float(np.percentile(scores, 50)))
        assert out["p95"]["value"] == pytest.approx(float(np.percentile(scores, 95)))
        assert out["max_minus_p95"]["value"] == pytest.approx(10.0 - float(np.percentile(scores, 95)))

    def test_empty(self):
        results = make_results(n_windows=1, trials_per_window=0)
        for w in results["window_results"]:
            w["optimization_trials"] = []
        out = compute_q3(results)
        assert out["p50"]["available"] is False


# ---------------------------------------------------------------------------
# Q4 — Stabilité
# ---------------------------------------------------------------------------

class TestQ4:
    def test_selection_frequency(self):
        results = make_results(n_windows=3, best_params=[
            {"timeperiod": 10, "StDev": 2.0},
            {"timeperiod": 11, "StDev": 2.0},
            {"timeperiod": 10, "StDev": 2.0},
        ])
        out = compute_q4(results, param_grid=PARAM_GRID)
        p = out["parameters"]["timeperiod"]
        assert p["selection_frequency"]["value"] == pytest.approx(1.0)
        assert p["n_windows_selected"] == 3

    def test_stuck_at_bound(self):
        # timeperiod stuck at min (5) in >= ceil(0.8*5)=4 windows, varies (also 6)
        best = [
            {"timeperiod": 5, "StDev": 2.0},
            {"timeperiod": 5, "StDev": 2.0},
            {"timeperiod": 5, "StDev": 2.0},
            {"timeperiod": 5, "StDev": 2.0},
            {"timeperiod": 6, "StDev": 2.0},
        ]
        results = make_results(n_windows=5, best_params=best)
        out = compute_q4(results, param_grid=PARAM_GRID)
        assert out["parameters"]["timeperiod"]["stuck_at_bound"] == "min"

    def test_dispersion_normalized(self):
        results = make_results(n_windows=3, best_params=[
            {"timeperiod": 5, "StDev": 2.0},
            {"timeperiod": 10, "StDev": 2.0},
            {"timeperiod": 15, "StDev": 2.0},
        ])
        out = compute_q4(results, param_grid=PARAM_GRID)
        p = out["parameters"]["timeperiod"]
        assert p["dispersion_type"] == "std_over_range"
        # std of [5,10,15] ddof=1 = 5; range = 10 -> 0.5
        assert p["dispersion_normalized"]["value"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Q5 — Voisinage
# ---------------------------------------------------------------------------

class TestQ5:
    def test_nominal_isolated_and_degradation(self):
        # winner score 10, neighbours within d<=0.10 have scores 4 (median 4 < 5)
        trials = [
            _trial(10, 2.0, 10.0),   # winner
            _trial(10, 2.1, 4.0),    # d = 0.1/2 / ... -> small; V, score 4
            _trial(10, 2.2, 4.0),    # d = 0.1; V
            _trial(11, 2.0, 4.0),    # d = (1/10)/2 = 0.05; V
            _trial(14, 3.0, 3.0),    # far
        ]
        results = make_results(n_windows=1, trials_per_window=5)
        results["window_results"][0]["optimization_trials"] = trials
        results["window_results"][0]["optimization_trials_count"] = 5
        out = compute_q5(results, param_grid=PARAM_GRID)
        w = out["windows"][0]
        assert w["winner_score"] == 10.0
        assert w["n_V"] >= 3
        assert w["n_Vperf"] == 0  # all neighbour scores < 8
        assert w["isolated"] is True
        assert w["degradation_marquee"] is True  # median 4 < 5

    def test_nominal_well_connected(self):
        # winner with >=5 performant neighbours (score >= 8)
        trials = [
            _trial(10, 2.0, 10.0),
            _trial(10, 2.02, 9.5),
            _trial(10, 2.04, 9.0),
            _trial(10, 2.06, 8.5),
            _trial(10, 2.08, 8.0),
            _trial(10, 2.10, 8.0),
        ]
        results = make_results(n_windows=1, trials_per_window=6)
        results["window_results"][0]["optimization_trials"] = trials
        results["window_results"][0]["optimization_trials_count"] = 6
        out = compute_q5(results, param_grid=PARAM_GRID)
        w = out["windows"][0]
        assert w["isolated"] is False
        assert w["degradation_marquee"] is False

    def test_alternative_case_score_le_zero(self):
        # winner score <= 0 -> alternative z-score case (needs >=5 distinct scores)
        trials = [
            _trial(10, 2.0, 0.0),
            _trial(10, 2.1, -1.0),
            _trial(10, 2.2, -2.0),
            _trial(11, 2.0, -3.0),
            _trial(11, 2.1, -4.0),
            _trial(12, 2.0, -5.0),
        ]
        results = make_results(n_windows=1, trials_per_window=6)
        results["window_results"][0]["optimization_trials"] = trials
        results["window_results"][0]["optimization_trials_count"] = 6
        out = compute_q5(results, param_grid=PARAM_GRID)
        w = out["windows"][0]
        # winner score is 0 (not > 0) -> alternative case, not nominal
        assert w["winner_score"] == 0.0
        # alternative case does not crash and yields a verdict flag
        assert w["isolated"] in (True, False)

    def test_conjunction_count(self):
        # two windows each isolated + degraded
        trials = [
            _trial(10, 2.0, 10.0),
            _trial(10, 2.1, 4.0),
            _trial(10, 2.2, 4.0),
            _trial(11, 2.0, 4.0),
        ]
        results = make_results(n_windows=2, trials_per_window=4)
        for w in results["window_results"]:
            w["optimization_trials"] = trials
            w["optimization_trials_count"] = 4
        out = compute_q5(results, param_grid=PARAM_GRID)
        assert out["conjunction_isolated_degraded_windows"] == 2


# ---------------------------------------------------------------------------
# Q6 — Érosion
# ---------------------------------------------------------------------------

class TestQ6:
    def test_sharpe_erosion(self):
        results = make_results(n_windows=1, oos_sharpes=[0.2])
        results["in_sample_performance"][0]["sharpe"] = 1.0
        out = compute_q6(results)
        w = out["windows"][0]
        assert w["erosion_sharpe_pct"]["value"] == pytest.approx((1.0 - 0.2) / 1.0 * 100)
        assert w["degradation"] is True  # 80% > 50%

    def test_erosion_undefined_when_denominator_nonpositive(self):
        results = make_results(n_windows=1, oos_sharpes=[0.2])
        results["in_sample_performance"][0]["sharpe"] = -0.1
        out = compute_q6(results)
        w = out["windows"][0]
        assert w["erosion_sharpe_pct"]["available"] is False


# ---------------------------------------------------------------------------
# Q7 — Sharpe homogène
# ---------------------------------------------------------------------------

class TestQ7:
    def test_recompute(self):
        results = make_results(n_windows=1)
        rets = np.array([0.01, -0.02, 0.03, 0.01, 0.005])
        pbr = {"is": {1: rets}, "oos": {1: rets}, "final": rets}
        out = compute_q7(results, pbr)
        expected = rets.mean() / rets.std(ddof=1)
        assert out["windows"][0]["sharpe_is_recomputed"]["value"] == pytest.approx(expected)
        assert out["final_sharpe_recomputed"]["value"] == pytest.approx(expected)

    def test_unavailable_without_series(self):
        results = make_results(n_windows=1)
        out = compute_q7(results, None)
        assert out["windows"][0]["sharpe_is_recomputed"]["available"] is False
        assert out["final_sharpe_recomputed"]["available"] is False


# ---------------------------------------------------------------------------
# Q8 — Coûts
# ---------------------------------------------------------------------------

class TestQ8:
    def test_net_convention(self):
        results = make_results(n_windows=5)
        out = compute_q8(results, make_final_trades(), config={"fees_pct": 0.1, "slippage_bps": 0.0})
        assert out["convention"] == "net_of_costs"
        assert out["costs_included"] is True
        assert out["cost_already_included_per_side_pct"]["value"] == pytest.approx(0.1)
        assert out["additional_cost_scenario"] is not None

    def test_gross_convention(self):
        results = make_results(n_windows=5)
        out = compute_q8(results, make_final_trades(), config={"fees_pct": 0.0, "slippage_bps": 0.0})
        assert out["convention"] == "gross"
        assert out["costs_included"] is False
        assert out["net_margin_per_side"] is not None


# ---------------------------------------------------------------------------
# P1 — Incertitude OOS
# ---------------------------------------------------------------------------

class TestOosUncertainty:
    def test_trade_concentration_detected(self):
        # one huge OOS trade dominating the P&L -> trade branch concentre
        oos_trades = make_final_trades(n=20)
        oos_trades["pnl"] = [100.0] + [-1.0] * 19
        oos_returns = [1.0, 1.0, 1.0, 1.0, 1.0]  # windows NOT concentrated
        results = make_results(n_windows=5, oos_returns=oos_returns)
        out = compute_oos_uncertainty(results, None, None, oos_trades)
        c = out["concentration"]
        assert c["trade_branch_available"] is True
        # k = ceil(0.1*20)=2 ; top2 = 100 + (-1) = 99 ; total = 100 - 19 = 81
        assert c["top_k_trades_frac"] == pytest.approx(99 / 81)
        assert c["status"] == "concentre"

    def test_window_concentration_detected(self):
        oos_returns = [50.0, 40.0, 1.0, 1.0, 1.0]
        results = make_results(n_windows=5, oos_returns=oos_returns)
        out = compute_oos_uncertainty(results, None, None, None)
        c = out["concentration"]
        assert c["top2_windows_frac"] == pytest.approx(90 / 93)
        assert c["status"] == "concentre"

    def test_no_concentration(self):
        oos_returns = [2.0, 2.0, 2.0, 2.0, 2.0]
        results = make_results(n_windows=5, oos_returns=oos_returns)
        out = compute_oos_uncertainty(results, None, None, None)
        assert out["concentration"]["status"] == "non_concentre"

    def test_indeterminate_when_pnl_nonpositive(self):
        results = make_results(n_windows=5, oos_returns=[-1.0, -1.0, -1.0, -1.0, -1.0])
        out = compute_oos_uncertainty(results, None, None, None)
        assert out["concentration"]["status"] == "indetermine"

    def test_trade_branch_unavailable_without_oos_trades(self):
        # Final trades must never certify the absence of OOS concentration (F4)
        results = make_results(n_windows=5, oos_returns=[2.0] * 5)
        out = compute_oos_uncertainty(results, make_final_trades(n=20), None, None)
        c = out["concentration"]
        assert c["trade_branch_available"] is False
        assert c["top_k_trades_frac"] is None

    def test_bootstrap_ci(self):
        results = make_results(n_windows=5, oos_returns=[1.0, 2.0, 3.0, 4.0, 5.0])
        out = compute_oos_uncertainty(results, make_final_trades(), seed=1)
        assert out["oos_return_ci95_low_pct"]["value"] is not None
        assert out["oos_return_ci95_high_pct"]["value"] is not None
        assert out["oos_return_mean_pct"]["value"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Pré-verdict
# ---------------------------------------------------------------------------

class TestPreVerdict:
    def _ctx(self, **kw):
        results = make_results(**kw.get("results_kw", {}))
        ft = make_final_trades()
        ind = compute_quant_indicators(
            results, final_trades=ft, config={"fees_pct": 0.1, "slippage_bps": 0.0},
            param_grid=PARAM_GRID,
        )
        return ind, results

    def test_go_requires_all_criteria(self):
        # a "clean" run: many windows/trades, low erosion, positive OOS, robust neighbours
        results = make_results(n_windows=6, oos_returns=[5.0] * 6, oos_sharpes=[0.4] * 6)
        for w in results["window_results"]:
            w["optimization_trials"] = [
                _trial(10, 2.0, 10.0), _trial(10, 2.02, 9.5), _trial(10, 2.04, 9.0),
                _trial(10, 2.06, 8.5), _trial(10, 2.08, 8.0), _trial(10, 2.10, 8.0),
            ]
            w["optimization_trials_count"] = 6
        ft = make_final_trades(n=30, pnl=(2.0, 1.0, 3.0, 4.0))
        ind = compute_quant_indicators(
            results, final_trades=ft, config={"fees_pct": 0.1, "slippage_bps": 0.0},
            param_grid=PARAM_GRID,
        )
        assert ind["pre_verdict"]["verdict"] in ("GO", "WATCH")

    def test_integrity_failure_forces_no_go(self):
        results = make_results(n_windows=5)
        # break integrity: no final trades
        ind = compute_quant_indicators(results, final_trades=None)
        assert ind["integrity"]["ok"] is False
        assert ind["pre_verdict"]["verdict"] == "NO_GO"
        assert ind["pre_verdict"]["status"] == "non_evaluable"

    def test_negative_oos_net_forces_no_go(self):
        results = make_results(n_windows=5, oos_returns=[-2.0] * 5, oos_sharpes=[-0.2] * 5)
        ft = make_final_trades(n=20, pnl=(-1.0, -2.0))
        ind = compute_quant_indicators(
            results, final_trades=ft, config={"fees_pct": 0.1, "slippage_bps": 0.0},
            param_grid=PARAM_GRID,
        )
        assert ind["pre_verdict"]["verdict"] == "NO_GO"

    def test_gross_costs_forces_no_go(self):
        results = make_results(n_windows=5, oos_returns=[5.0] * 5)
        ind = compute_quant_indicators(
            results, final_trades=make_final_trades(),
            config={"fees_pct": 0.0, "slippage_bps": 0.0}, param_grid=PARAM_GRID,
        )
        assert ind["pre_verdict"]["criteria"]["C4_resultat_net_oos"]["verdict"] == "NO_GO"
        assert ind["pre_verdict"]["verdict"] == "NO_GO"


# ---------------------------------------------------------------------------
# Orchestrateur
# ---------------------------------------------------------------------------

class TestOrchestrator:
    def test_full_pipeline(self):
        results = make_results(n_windows=5)
        ft = make_final_trades()
        out = compute_quant_indicators(
            results, final_trades=ft,
            config={"fees_pct": 0.1, "slippage_bps": 0.0},
            param_grid=PARAM_GRID,
            per_bar_returns={"is": {1: [0.01, 0.02, 0.03]}, "oos": {1: [0.01, 0.02, 0.03]}, "final": [0.01, 0.02, 0.03]},
        )
        assert out["schema_version"] == SCHEMA_VERSION
        assert out["manifest"]["input_digest"] is not None
        assert set(out["indicators"].keys()) == {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "OOS_UNCERTAINTY"}
        assert out["pre_verdict"]["verdict"] in ("GO", "WATCH", "NO_GO")

    def test_reproducible(self):
        results = make_results(n_windows=5)
        ft = make_final_trades()
        cfg = {"fees_pct": 0.1, "slippage_bps": 0.0}
        a = compute_quant_indicators(results, final_trades=ft, config=cfg, param_grid=PARAM_GRID)
        b = compute_quant_indicators(results, final_trades=ft, config=cfg, param_grid=PARAM_GRID)
        # generated_at differs, but everything else is identical
        a_minus_ts = {k: v for k, v in a.items() if k != "generated_at"}
        b_minus_ts = {k: v for k, v in b.items() if k != "generated_at"}
        assert a_minus_ts == b_minus_ts


# ---------------------------------------------------------------------------
# §7 edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_one_trial(self):
        results = make_results(n_windows=1, trials_per_window=1)
        out = compute_quant_indicators(results, final_trades=make_final_trades())
        assert out["indicators"]["Q3"]["max"]["value"] is not None

    def test_max_score_le_zero(self):
        trials = [_trial(10, 2.0, 0.0), _trial(10, 2.1, -1.0), _trial(11, 2.0, -2.0),
                  _trial(11, 2.1, -3.0), _trial(12, 2.0, -4.0)]
        results = make_results(n_windows=1, trials_per_window=5)
        results["window_results"][0]["optimization_trials"] = trials
        results["window_results"][0]["optimization_trials_count"] = 5
        out = compute_q5(results, param_grid=PARAM_GRID)
        assert out["windows"][0]["winner_score"] <= 0

    def test_empty_oos(self):
        results = make_results(n_windows=1)
        results["out_of_sample_performance"] = []
        # integrity fails because IS/OOS unpaired
        integ = check_run_integrity(results, final_trades=make_final_trades())
        assert integ["ok"] is False

    def test_unmatched_window(self):
        results = make_results(n_windows=3)
        results["out_of_sample_performance"][1]["window"] = 99
        integ = check_run_integrity(results, final_trades=make_final_trades(), config={"fees_pct": 0.1, "slippage_bps": 0.0})
        assert integ["ok"] is False


# ---------------------------------------------------------------------------
# Review findings (wfo-reviewer, NO_GO -> fixes)
# ---------------------------------------------------------------------------

class TestReviewFindings:
    def test_digest_invalidates_on_score_change(self):
        results = make_results(n_windows=1)
        ft = make_final_trades()
        cfg = {"fees_pct": 0.1}
        m1 = build_run_manifest(results, cfg, final_trades=ft)
        results["window_results"][0]["optimization_trials"][0]["combined_score"] = 9999.0
        m2 = build_run_manifest(results, cfg, final_trades=ft)
        assert m1["input_digest"] != m2["input_digest"]

    def test_digest_invalidates_on_config_change(self):
        results = make_results(n_windows=1)
        ft = make_final_trades()
        m1 = build_run_manifest(results, {"fees_pct": 0.1}, final_trades=ft)
        m2 = build_run_manifest(results, {"fees_pct": 0.9}, final_trades=ft)
        assert m1["input_digest"] != m2["input_digest"]

    def test_digest_invalidates_on_trade_change(self):
        results = make_results(n_windows=1)
        ft = make_final_trades(n=10)
        m1 = build_run_manifest(results, {"fees_pct": 0.1}, final_trades=ft)
        ft2 = ft.copy()
        ft2.loc[0, "pnl"] = 999999.0
        m2 = build_run_manifest(results, {"fees_pct": 0.1}, final_trades=ft2)
        assert m1["input_digest"] != m2["input_digest"]

    def test_winner_no_neighbours_degraded(self):
        # F3: V=∅ -> median convention 0 -> degradation marquee (nominal case)
        trials = [_trial(10, 2.0, 10.0), _trial(15, 3.0, 1.0)]
        results = make_results(n_windows=1, trials_per_window=2)
        results["window_results"][0]["optimization_trials"] = trials
        results["window_results"][0]["optimization_trials_count"] = 2
        out = compute_q5(results, param_grid=PARAM_GRID)
        w = out["windows"][0]
        assert w["n_V"] == 0
        assert w["isolated"] is True
        assert w["degradation_marquee"] is True

    def test_alternative_case_robust_neighborhood_can_be_go(self):
        # F4: alternative case (winner <= 0) must be able to flag robust neighbourhood
        trials = [
            _trial(10, 2.0, 0.0),     # winner
            _trial(10, 2.01, -0.01),
            _trial(10, 2.02, -0.02),
            _trial(10, 2.03, -0.03),
            _trial(10, 2.04, -0.04),
            _trial(10, 2.05, -0.05),
            _trial(10, 2.5, -10.0),
            _trial(10, 2.6, -11.0),
            _trial(10, 2.7, -12.0),
            _trial(10, 2.8, -13.0),
            _trial(10, 2.9, -14.0),
        ]
        results = make_results(n_windows=1, trials_per_window=11)
        results["window_results"][0]["optimization_trials"] = trials
        results["window_results"][0]["optimization_trials_count"] = 11
        out = compute_q5(results, param_grid=PARAM_GRID)
        w = out["windows"][0]
        assert w["winner_score"] == 0.0
        assert w["n_Vperf"] >= 5
        assert w["robust_neighborhood"] is True

    def test_concentration_separate_denominators(self):
        # F5: window branch uses only OOS %, trade branch uses only OOS-trade P&L
        oos_returns = [50.0, 40.0, 1.0, 1.0, 1.0]
        oos_trades = make_final_trades(n=20, pnl=(1.0, 1.0, 1.0))
        results = make_results(n_windows=5, oos_returns=oos_returns)
        out = compute_oos_uncertainty(results, None, None, oos_trades)
        c = out["concentration"]
        # top2 windows = 90 / 93 -> > 60% -> concentre
        assert c["top2_windows_frac"] == pytest.approx(90 / 93)
        # trade branch (equal pnls) is NOT concentrated
        assert c["top_k_trades_frac"] == pytest.approx(2 / 20)
        assert c["status"] == "concentre"

    def test_cost_unknown_vs_zero(self):
        # F6: absent fees -> unknown; explicit zeros -> gross
        results = make_results(n_windows=1)
        out_unknown = compute_q8(results, make_final_trades(), config={})
        assert out_unknown["convention"] == "unknown"
        assert out_unknown["costs_included"] is None
        out_gross = compute_q8(results, make_final_trades(), config={"fees_pct": 0.0, "slippage_bps": 0.0})
        assert out_gross["convention"] == "gross"
        assert out_gross["costs_included"] is False

    def test_no_score_range_inference(self):
        # F7: range resolution must never treat combined_score as a parameter
        results = make_results(n_windows=1)
        ranges = q._resolve_param_ranges(results, None, param_grid=None, config=None)
        assert "combined_score" not in ranges
        # with an explicit grid, only the grid params are returned
        ranges2 = q._resolve_param_ranges(results, None, param_grid=PARAM_GRID, config=None)
        assert set(ranges2.keys()) == {"timeperiod", "StDev"}

    def test_q3_inf_nan_scores_no_crash(self):
        # F8: NaN/Inf scores must be filtered without a take on an empty array
        trials = [
            _trial(10, 2.0, float("inf")),
            _trial(10, 2.1, float("nan")),
            _trial(11, 2.0, 5.0),
        ]
        results = make_results(n_windows=1, trials_per_window=3)
        results["window_results"][0]["optimization_trials"] = trials
        out = compute_q3(results)
        assert out["max"]["value"] == 5.0
        assert out["p50"]["value"] == 5.0

    def test_digest_invalidates_on_returns_order(self):
        # F1 round 2: permutation of a returns series must change the digest
        results = make_results(n_windows=1)
        pbr1 = {"final": [0.0, 1.0, 2.0]}
        pbr2 = {"final": [2.0, 1.0, 0.0]}
        m1 = build_run_manifest(results, {"fees_pct": 0.1}, per_bar_returns=pbr1)
        m2 = build_run_manifest(results, {"fees_pct": 0.1}, per_bar_returns=pbr2)
        assert m1["input_digest"] != m2["input_digest"]

    def test_cost_invalid_component_blocks(self):
        # F2 round 2: a declared but non-numeric cost component must block integrity
        results = make_results(n_windows=1)
        integ = check_run_integrity(
            results, final_trades=make_final_trades(),
            config={"fees_pct": 0.1, "slippage_bps": "bad"},
        )
        assert integ["ok"] is False
        assert any("invalide" in c for c in integ["causes"])
        # Q8 must report the invalid convention, not silently treat it as 0
        out = compute_q8(results, make_final_trades(), config={"fees_pct": 0.1, "slippage_bps": "bad"})
        assert out["convention"] == "invalid"
        assert out["costs_included"] is None

    def test_partial_metric_caps_watch(self):
        # F3 round 2: missing IS Sharpe in one window -> C3 must not be GO.
        # Low erosion on the evaluable windows would otherwise qualify for GO.
        results = make_results(n_windows=5, oos_returns=[5.0] * 5, oos_sharpes=[0.45] * 5)
        for r in results["in_sample_performance"]:
            r["sharpe"] = 0.5  # erosion = (0.5-0.45)/0.5 = 10% < 30%
        results["in_sample_performance"][0]["sharpe"] = None
        ft = make_final_trades(n=30)
        ind = compute_quant_indicators(
            results, final_trades=ft, config={"fees_pct": 0.1, "slippage_bps": 0.0},
            param_grid=PARAM_GRID,
        )
        assert ind["indicators"]["Q6"]["n_windows_undefined"] >= 1
        assert ind["pre_verdict"]["criteria"]["C3_erosion_sharpe"]["verdict"] == "WATCH"

    def test_q5_inf_winner_ignored(self):
        # F5 round 2: an Inf score must never become the winner
        trials = [
            _trial(10, 2.0, float("inf")),
            _trial(10, 2.1, 8.0),
            _trial(11, 2.0, 7.0),
        ]
        results = make_results(n_windows=1, trials_per_window=3)
        results["window_results"][0]["optimization_trials"] = trials
        out = compute_q5(results, param_grid=PARAM_GRID)
        w = out["windows"][0]
        assert w["winner_score"] == 8.0  # finite winner, not Inf

    def test_params_matched_reported(self):
        # F6 round 4: param-level pairing is inferred-only, honestly NOT ok
        results = make_results(n_windows=3)
        integ = check_run_integrity(
            results, final_trades=make_final_trades(), config={"fees_pct": 0.1, "slippage_bps": 0.0}
        )
        pm = integ["checks"]["params_matched"]
        assert pm["ok"] is False
        assert pm["blocking"] is False
        assert pm["verified"] == "inferred_only"
        assert pm["formal_verification"] is False
        # non-blocking: overall integrity is still ok
        assert integ["ok"] is True

    def test_digest_invalidates_on_param_grid_change(self):
        # F1 round 4: param_grid ranges must be part of the digest
        results = make_results(n_windows=1)
        m1 = build_run_manifest(results, {"fees_pct": 0.1}, param_grid={"timeperiod": (5, 15)})
        m2 = build_run_manifest(results, {"fees_pct": 0.1}, param_grid={"timeperiod": (5, 50)})
        assert m1["input_digest"] != m2["input_digest"]

    def test_params_matched_inconsistent_blocks(self):
        # F6 round 3: inconsistent best_params key sets across windows must block
        results = make_results(n_windows=3)
        results["window_results"][1]["best_params"] = {"different": 1}
        integ = check_run_integrity(
            results, final_trades=make_final_trades(), config={"fees_pct": 0.1, "slippage_bps": 0.0}
        )
        assert integ["ok"] is False
        assert any("best_params" in c for c in integ["causes"])

    def test_params_matched_fingerprint_verified(self):
        # F6 round 5: matching engine-side fingerprint -> formally verified
        results = _add_params_sha(make_results(n_windows=3))
        integ = check_run_integrity(
            results, final_trades=make_final_trades(), config={"fees_pct": 0.1, "slippage_bps": 0.0}
        )
        pm = integ["checks"]["params_matched"]
        assert pm["ok"] is True
        assert pm["verified"] == "fingerprint"
        assert pm["formal_verification"] is True

    def test_params_matched_fingerprint_mismatch_blocks(self):
        # F6 round 5: tampered params_sha -> blocking
        results = _add_params_sha(make_results(n_windows=3))
        results["in_sample_performance"][0]["params_sha"] = "deadbeef"
        integ = check_run_integrity(
            results, final_trades=make_final_trades(), config={"fees_pct": 0.1, "slippage_bps": 0.0}
        )
        assert integ["ok"] is False
        assert integ["checks"]["params_matched"]["verified"] == "fingerprint_mismatch"

    def test_params_value_change_detected(self):
        # F6 round 5: changing a best_params value (without re-signing) is caught
        results = _add_params_sha(make_results(n_windows=3))
        results["window_results"][0]["best_params"]["timeperiod"] = 999
        integ = check_run_integrity(
            results, final_trades=make_final_trades(), config={"fees_pct": 0.1, "slippage_bps": 0.0}
        )
        assert integ["ok"] is False
        assert integ["checks"]["params_matched"]["verified"] == "fingerprint_mismatch"

    def test_params_matched_missing_oos_side_is_partial(self):
        # F6 round 7: one missing OOS side -> partial -> blocking (incomplete artifact)
        results = _add_params_sha(make_results(n_windows=3))
        del results["out_of_sample_performance"][0]["params_sha"]
        integ = check_run_integrity(
            results, final_trades=make_final_trades(), config={"fees_pct": 0.1, "slippage_bps": 0.0}
        )
        pm = integ["checks"]["params_matched"]
        assert pm["verified"] == "partial"
        assert pm["formal_verification"] is False
        assert pm["ok"] is False
        assert integ["ok"] is False  # blocking
        assert any("partielle" in c for c in integ["causes"])

    def test_params_matched_single_window_fingerprint_is_partial(self):
        # F6 round 8: fingerprint on only one window (all 3 sides) -> partial
        results = _add_params_sha(make_results(n_windows=3))
        for i in (1, 2):
            results["in_sample_performance"][i].pop("params_sha", None)
            results["out_of_sample_performance"][i].pop("params_sha", None)
            results["window_results"][i].pop("params_sha", None)
        integ = check_run_integrity(
            results, final_trades=make_final_trades(), config={"fees_pct": 0.1, "slippage_bps": 0.0}
        )
        pm = integ["checks"]["params_matched"]
        assert pm["verified"] == "partial"
        assert pm["formal_verification"] is False
        assert pm["n_present"] == 3  # IS + OOS + window-level of window 1 only
        assert pm["n_expected"] == 9
        assert integ["ok"] is False  # blocking

    def test_params_matched_partial_forces_no_go(self):
        # F6 round 7: partial fingerprint -> non_evaluable -> pre-verdict NO_GO
        results = _add_params_sha(make_results(n_windows=3))
        del results["out_of_sample_performance"][0]["params_sha"]
        ind = compute_quant_indicators(
            results, final_trades=make_final_trades(), config={"fees_pct": 0.1, "slippage_bps": 0.0}
        )
        assert ind["integrity"]["ok"] is False
        assert ind["pre_verdict"]["verdict"] == "NO_GO"
        assert ind["pre_verdict"]["status"] == "non_evaluable"

    def test_digest_invalidates_on_oos_trades_change(self):
        # F1 round 3: oos_trades content must be part of the digest
        results = make_results(n_windows=1)
        oos1 = make_final_trades(n=20, pnl=(1.0, 1.0, 1.0))
        oos2 = oos1.copy()
        oos2.loc[0, "pnl"] = 999999.0
        m1 = build_run_manifest(results, {"fees_pct": 0.1}, oos_trades=oos1)
        m2 = build_run_manifest(results, {"fees_pct": 0.1}, oos_trades=oos2)
        assert m1["input_digest"] != m2["input_digest"]

    def test_cardinality_incoherence_blocks(self):
        # P0 round 9: declared count / evaluations incoherent -> blocking
        results = make_results(n_windows=1)
        w = results["window_results"][0]
        w["optimization_trials_count"] = 999
        w["evaluations"] = 1
        integ = check_run_integrity(
            results, final_trades=make_final_trades(), config={"fees_pct": 0.1, "slippage_bps": 0.0}
        )
        assert integ["ok"] is False
        assert any("cardinalité" in c for c in integ["causes"])

    def test_missing_oos_return_blocks(self):
        # P0 round 9: missing OOS return (required field) -> blocking
        results = make_results(n_windows=1)
        results["out_of_sample_performance"][0]["return"] = None
        integ = check_run_integrity(
            results, final_trades=make_final_trades(), config={"fees_pct": 0.1, "slippage_bps": 0.0}
        )
        assert integ["ok"] is False
        assert any("obligatoire" in c for c in integ["causes"])

    def test_n_trades_nan_blocks(self):
        # P0 round 10: n_trades=NaN -> blocking (presence is not enough)
        results = make_results(n_windows=1)
        results["out_of_sample_performance"][0]["n_trades"] = float("nan")
        integ = check_run_integrity(
            results, final_trades=make_final_trades(), config={"fees_pct": 0.1, "slippage_bps": 0.0}
        )
        assert integ["ok"] is False

    def test_float_trials_count_blocks(self):
        # P0 round 10: optimization_trials_count=6.7 -> non-integer -> blocking
        results = make_results(n_windows=1)
        results["window_results"][0]["optimization_trials_count"] = 6.7
        integ = check_run_integrity(
            results, final_trades=make_final_trades(), config={"fees_pct": 0.1, "slippage_bps": 0.0}
        )
        assert integ["ok"] is False

    def test_n_windows_mismatch_blocks(self):
        # P0 round 10: settings.n_windows=99 for 5 computed windows -> blocking
        results = make_results(n_windows=5)
        results["settings"]["n_windows"] = 99
        integ = check_run_integrity(
            results, final_trades=make_final_trades(), config={"fees_pct": 0.1, "slippage_bps": 0.0}
        )
        assert integ["ok"] is False
        assert any("n_windows" in c for c in integ["causes"])

    def test_n_windows_none_nan_float_block(self):
        # P0 round 11: n_windows None / NaN / non-integer -> blocking, no truncation
        for bad in (None, float("nan"), 5.7):
            results = make_results(n_windows=5)
            results["settings"]["n_windows"] = bad
            integ = check_run_integrity(
                results, final_trades=make_final_trades(), config={"fees_pct": 0.1, "slippage_bps": 0.0}
            )
            assert integ["ok"] is False, f"n_windows={bad!r} doit bloquer"

    def test_evaluations_overflow_blocks(self):
        # P0 round 10: evaluations=999 for 6 trial rows -> blocking
        results = make_results(n_windows=1)
        results["window_results"][0]["evaluations"] = 999
        integ = check_run_integrity(
            results, final_trades=make_final_trades(), config={"fees_pct": 0.1, "slippage_bps": 0.0}
        )
        assert integ["ok"] is False
        assert any("cardinalité" in c for c in integ["causes"])

    def test_partial_metric_signals_detects_q7(self):
        # P0 round 9: partial-metric helper flags a missing Final Sharpe
        indicators = {"Q6": {}, "Q7": {"final_sharpe_recomputed": {"available": False}, "windows": []}}
        signals = q._partial_metric_signals(indicators)
        assert any("Final" in s for s in signals)

    def test_metric_column_not_param(self):
        # P1 round 9: a metric column (return) must never be treated as a param
        trials = [
            {"timeperiod": 10, "StDev": 2.0, "combined_score": 10.0, "return": 5.0},
            {"timeperiod": 10, "StDev": 2.0, "combined_score": 8.0, "return": 7.0},
        ]
        results = make_results(n_windows=1, trials_per_window=2)
        results["window_results"][0]["optimization_trials"] = trials
        results["window_results"][0]["optimization_trials_count"] = 2
        out = compute_q5(results, param_grid=PARAM_GRID)
        w = out["windows"][0]
        # same authoritative params -> dedupe to 1 config -> no distinct neighbour
        assert w["n_V"] == 0

    def test_window_fingerprint_corruption_blocks(self):
        # P1 round 9: corrupting window_results.params_sha -> mismatch -> blocking
        results = _add_params_sha(make_results(n_windows=3))
        results["window_results"][0]["params_sha"] = "corrupt"
        integ = check_run_integrity(
            results, final_trades=make_final_trades(), config={"fees_pct": 0.1, "slippage_bps": 0.0}
        )
        assert integ["ok"] is False
        assert integ["checks"]["params_matched"]["verified"] == "fingerprint_mismatch"

    def test_series_index_digest(self):
        # P2 round 9: a returns Series with a different index must change the digest
        results = make_results(n_windows=1)
        s1 = pd.Series([0.01, 0.02, 0.03], index=[1, 2, 3])
        s2 = pd.Series([0.01, 0.02, 0.03], index=[10, 20, 30])
        m1 = build_run_manifest(results, {"fees_pct": 0.1}, per_bar_returns={"final": s1})
        m2 = build_run_manifest(results, {"fees_pct": 0.1}, per_bar_returns={"final": s2})
        assert m1["input_digest"] != m2["input_digest"]

    def test_cost_none_is_unknown(self):
        # F3 round 3: explicit None must be unknown, not a proven zero
        results = make_results(n_windows=1)
        integ = check_run_integrity(
            results, final_trades=make_final_trades(), config={"fees_pct": None}
        )
        assert integ["ok"] is False
        out = compute_q8(results, make_final_trades(), config={"fees_pct": None})
        assert out["convention"] == "unknown"
        assert out["costs_included"] is None

    def test_q5_single_trial_no_range_indeterminate(self):
        # F4 round 3: p=0 must be indeterminate even with a single trial
        trials = [_trial(10, 2.0, 10.0)]
        results = make_results(n_windows=1, trials_per_window=1)
        results["window_results"][0]["optimization_trials"] = trials
        out = compute_q5(results, param_grid=None, config=None)
        w = out["windows"][0]
        assert w["indeterminate"] is True
        assert w["isolated"] is None
        assert w["degradation_marquee"] is None
