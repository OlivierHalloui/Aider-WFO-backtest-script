import os
import sys
import importlib.util

import numpy as np
import pandas as pd
import pytest

TEST_DIR = os.path.dirname(__file__)
WFO_ENGINE_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
sys.path.insert(0, WFO_ENGINE_DIR)

HAS_VBT = importlib.util.find_spec("vectorbtpro") is not None
pytestmark = pytest.mark.skipif(not HAS_VBT, reason="vectorbtpro required")

from stagewise_optimizer import (
    propose_stage_settings,
    _consensus_params,
    PARAM_DEFAULTS,
    FULL_PARAM_REGISTRY,
    build_stagewise_wfo_results,
)
from domain.serialization import sha256_json
from services.quant_indicators import check_run_integrity

if HAS_VBT:
    from stagewise_optimizer import run_stagewise_campaign


def generate_mock_data(n=200):
    rng = np.random.default_rng(42)
    index = pd.date_range("2025-01-01", periods=n, freq="5s")
    close = rng.uniform(100, 200, size=n)
    return pd.DataFrame(
        {"Close": close, "Open": close, "High": close + 1.0, "Low": close - 1.0},
        index=index,
    )


# ── propose_stage_settings (pure, no VBT) ─────────────────────────────────────

class TestProposeStageSettings:
    def test_returns_required_keys(self):
        result = propose_stage_settings(["timeperiod", "StDev"])
        assert {"method", "max_trials", "patience", "stability", "n_combos"}.issubset(result)

    def test_small_combo_selects_grid(self):
        result = propose_stage_settings(["use_t2_signal"])
        assert result["method"] == "grid"

    def test_large_combo_selects_bayesian(self):
        # timeperiod (21 values) × StDev (23 values) × fenetre_lowest (19) > 100k → bayesian
        result = propose_stage_settings(["timeperiod", "StDev", "fenetre_lowest"])
        assert result["method"] == "bayesian"
        assert result["max_trials"] > 0

    def test_param_ranges_override(self):
        # With narrow custom ranges, combo count drops → grid may be selected
        result = propose_stage_settings(
            ["timeperiod", "StDev"],
            param_ranges={"timeperiod": (10, 12, 2), "StDev": (1.0, 1.5, 0.5)},
        )
        assert result["n_combos"] <= 6
        assert result["method"] == "grid"

    def test_empty_params_returns_defaults(self):
        result = propose_stage_settings([])
        assert result["method"] == "grid"
        assert result["n_combos"] == 0


# ── _consensus_params (pure, no VBT) ──────────────────────────────────────────

class TestConsensusParams:
    def _make_wfo_results(self, rows):
        return {"best_params": rows}

    def test_numeric_median(self):
        rows = [{"timeperiod": 10}, {"timeperiod": 12}, {"timeperiod": 14}]
        result = _consensus_params(self._make_wfo_results(rows))
        assert result["timeperiod"] == 12

    def test_float_median(self):
        rows = [{"StDev": 1.0}, {"StDev": 2.0}, {"StDev": 3.0}]
        result = _consensus_params(self._make_wfo_results(rows))
        assert abs(result["StDev"] - 2.0) < 1e-9

    def test_boolean_majority_vote_true(self):
        rows = [
            {"use_t2_signal": True},
            {"use_t2_signal": True},
            {"use_t2_signal": False},
        ]
        result = _consensus_params(self._make_wfo_results(rows))
        assert result["use_t2_signal"] is True

    def test_boolean_majority_vote_false(self):
        rows = [
            {"exit_sar_enabled": False},
            {"exit_sar_enabled": False},
            {"exit_sar_enabled": True},
        ]
        result = _consensus_params(self._make_wfo_results(rows))
        assert result["exit_sar_enabled"] is False

    def test_empty_best_params_returns_empty(self):
        result = _consensus_params({"best_params": []})
        assert result == {}

    def test_mixed_params(self):
        rows = [
            {"timeperiod": 10, "use_t2_signal": True,  "StDev": 1.0},
            {"timeperiod": 14, "use_t2_signal": False, "StDev": 2.0},
            {"timeperiod": 12, "use_t2_signal": True,  "StDev": 1.5},
        ]
        result = _consensus_params(self._make_wfo_results(rows))
        assert result["timeperiod"] == 12
        assert result["use_t2_signal"] is True
        assert abs(result["StDev"] - 1.5) < 1e-9


# ── run_stagewise_campaign smoke test ─────────────────────────────────────────

MINI_STAGE_PLAN = [
    {
        "name": "Mini Stage 1 — timeperiod only",
        "optimize": ["timeperiod"],
        "fixed_overrides": {
            "use_t2_signal": True,
            "use_roc_filter": False,
            "use_divergence_bb": False,
            "exit_sar_enabled": False,
            "exit_macd_enabled": False,
            "exit_cross_sar_sma_enabled": False,
            "exit_retour_bb_enabled": False,
            "exit_regline_enabled": False,
            "exit_volat_down_enabled": False,
        },
        "method": "grid",
        "max_trials": 2,
        "patience": "Low",
        "stability": 3,
    },
]


def test_run_stagewise_campaign_smoke(tmp_path):
    """Stagewise campaign with 1 mini stage must return final_report with accumulated_best_params."""
    df = generate_mock_data(200)
    report = run_stagewise_campaign(
        df=df,
        timeframe="5s",
        direction="long_only",
        stage_plan=MINI_STAGE_PLAN,
        n_windows=1,
        train_size=0.6,
        output_dir=tmp_path / "stagewise_out",
        optimization_metric="sharpe_ratio",
        secondary_metric="total_return",
        metric_weights=(1.0, 0.0),
        parallel_backend="sequential",
        neighbor_count=3,
        param_ranges={"timeperiod": (10, 12, 2)},
    )
    assert isinstance(report, dict)
    assert "stages" in report
    assert len(report["stages"]) == 1
    assert "final_best_params" in report
    final_params = report["final_best_params"]
    assert isinstance(final_params, dict)
    assert "timeperiod" in final_params


def test_stagewise_first_window_receives_csv_prefix(tmp_path, monkeypatch):
    import stagewise_optimizer as stagewise
    from config import compute_warmup_bars

    idx = pd.date_range('2024-01-01', periods=18007, freq='5s')
    path = tmp_path / 'bars.csv'
    pd.DataFrame({'Open time': idx, 'Open': np.arange(len(idx)),
                  'High': np.arange(len(idx)), 'Low': np.arange(len(idx)),
                  'Close': np.arange(len(idx))}).to_csv(path, index=False)
    warmup = compute_warmup_bars({'timeperiod': (10, 30, 1)})
    df = stagewise._load_csv(str(path), '2024-01-02', '2024-01-02', warmup_bars=warmup)
    selected_start = df.index[warmup]
    assert warmup == 287
    assert len(df) == 1014
    assert len(df.loc[selected_start:]) == 727

    calls = []

    def fake_wfo(df, **kwargs):
        calls.append((df.index, kwargs['selected_start']))
        return {'best_params': [{'timeperiod': 30}], 'window_results': [],
                'in_sample_performance': [], 'out_of_sample_performance': []}

    monkeypatch.setattr(stagewise, 'walk_forward_optimization', fake_wfo)
    report = stagewise.run_stagewise_campaign(
        df, stage_plan=MINI_STAGE_PLAN, n_windows=1, output_dir=tmp_path / 'run',
        selected_start=selected_start,
    )
    assert report['stages'][0]['status'] == 'OK'
    assert calls[0][0][0] == pd.Timestamp(idx[16993], tz='UTC')
    assert calls[0][1] == pd.Timestamp(idx[17280], tz='UTC')


# ── §5.0 « appariées par paramètres sélectionnés » sur un run stagewise ──────

class TestStagewiseParamsFingerprint:
    """The fingerprint must cover the COMPLETE params actually used.

    ``build_stagewise_wfo_results`` merges the stage-fixed params into each
    window's ``best_params``.  The ``params_sha`` left by
    ``walk_forward_optimization`` only covers the stage-searched subset, so it
    must be recomputed on the merged set and propagated to the window row and to
    both IS/OOS metric rows — otherwise §5.0 rejects the run.
    """

    FIXED = {"StDev": 1.5, "fenetre_lowest": 10}

    def _final_report(self):
        partial_1, partial_2 = {"timeperiod": 12}, {"timeperiod": 14}
        lswr = {
            "window_results": [
                {"window_info": {"window": 1}, "best_params": partial_1,
                 "params_sha": sha256_json(partial_1)},
                {"window_info": {"window": 2}, "best_params": partial_2,
                 "params_sha": sha256_json(partial_2)},
            ],
            "in_sample_performance": [
                {"window": 1, "return": 1.0, "params_sha": sha256_json(partial_1)},
                {"window": 2, "return": 2.0, "params_sha": sha256_json(partial_2)},
            ],
            "out_of_sample_performance": [
                {"window": 1, "return": 0.5, "params_sha": sha256_json(partial_1)},
                {"window": 2, "return": 0.6, "params_sha": sha256_json(partial_2)},
            ],
            "settings": {}, "timing": {},
        }
        return {
            "last_stage_wfo_results": lswr,
            "last_stage_fixed_params": dict(self.FIXED),
            "n_stages": 2, "stages": [],
        }

    def test_fixed_params_are_fingerprinted(self):
        results = build_stagewise_wfo_results(self._final_report())
        for wr in results["window_results"]:
            assert set(self.FIXED).issubset(wr["best_params"]), "paramètres fixes absents"
            assert wr["params_sha"] == sha256_json(wr["best_params"])
            # l'empreinte de l'étape seule ne doit pas survivre
            stage_only = sha256_json({"timeperiod": wr["best_params"]["timeperiod"]})
            assert wr["params_sha"] != stage_only

    def test_fingerprint_propagated_to_is_and_oos_rows(self):
        results = build_stagewise_wfo_results(self._final_report())
        expected = {i + 1: wr["params_sha"] for i, wr in enumerate(results["window_results"])}
        assert len(expected) == 2
        for key in ("in_sample_performance", "out_of_sample_performance"):
            rows = results[key]
            assert len(rows) == 2
            for row in rows:
                assert row["params_sha"] == expected[row["window"]], f"{key} fenêtre {row['window']}"

    def test_check_run_integrity_params_matched_ok(self):
        """End-to-end: §5.0 accepte un run stagewise (point bloquant de revue)."""
        results = build_stagewise_wfo_results(self._final_report())
        report = check_run_integrity(results)
        check = report["checks"]["params_matched"]
        assert check["ok"] is True, check
        assert check.get("verified") == "fingerprint", check
        assert check.get("formal_verification") is True, check
        # 3 emplacements (fenêtre + IS + OOS) × 2 fenêtres = 6 empreintes présentes
        assert check.get("n_present") == check.get("n_expected") == 6, check
        assert not [c for c in report["causes"] if "best_params" in str(c)], report["causes"]

    def test_corrupt_source_fingerprint_is_not_laundered(self):
        """Une empreinte source discordante ne doit JAMAIS être blanchie.

        Point bloquant de revue : `_rekey_metrics` écrasait la valeur source par
        celle attendue, ce qui transformait une erreur d'appariement en run
        « vérifié ».  La discordance doit survivre et être signalée par §5.0.
        """
        report = self._final_report()
        lswr = report["last_stage_wfo_results"]
        # l'IS de la fenêtre 1 porte une empreinte qui ne correspond à rien
        lswr["in_sample_performance"][0]["params_sha"] = "corrupt"

        results = build_stagewise_wfo_results(report)
        assert results["in_sample_performance"][0]["params_sha"] == "corrupt", (
            "l'empreinte source discordante a été écrasée"
        )
        # les autres emplacements restent cohérents (conversion normale)
        assert results["window_results"][0]["params_sha"] == sha256_json(results["window_results"][0]["best_params"])

        recon = results["params_sha_reconciliation"]
        assert recon["ok"] is False
        assert recon["anomalies"][0]["code"] == "empreinte_source_incoherente"

        report_int = check_run_integrity(results)
        check = report_int["checks"]["params_matched"]
        assert check["ok"] is False, check
        assert check.get("verified") == "fingerprint_mismatch", check
        assert check.get("blocking") is True, check
        assert check.get("n_mismatch") == 1, check
        # le signal explicite est lui-même bloquant
        assert report_int["checks"]["params_sha_reconciliation"]["blocking"] is True

    def test_source_equal_to_complete_hash_still_blocks(self):
        """Cas de coïncidence : la source vaut le hash COMPLET, pas celui d'étape.

        La règle « source discordante → bloquant » doit être garantie par
        construction, pas par absence de coïncidence de valeurs.
        """
        report = self._final_report()
        lswr = report["last_stage_wfo_results"]
        # l'IS fenêtre 1 porte l'empreinte des paramètres COMPLETS :
        # elle contredit son propre best_params (partiel) mais correspondrait
        # au best_params reconstruit -> elle ne doit pourtant pas être certifiée.
        complete_sha = sha256_json({**self.FIXED, **{"timeperiod": 12}})
        lswr["in_sample_performance"][0]["params_sha"] = complete_sha

        results = build_stagewise_wfo_results(report)
        recon = results["params_sha_reconciliation"]
        assert recon["ok"] is False, "la discordance n'a pas été détectée"
        assert any(a["code"] == "empreinte_source_incoherente" for a in recon["anomalies"])

        report_int = check_run_integrity(results)
        assert report_int["ok"] is False, "le run a été certifié malgré la discordance"
        assert report_int["checks"]["params_sha_reconciliation"]["blocking"] is True
        assert any("réconciliation des empreintes" in str(c) for c in report_int["causes"]), report_int["causes"]

    def test_hash_failure_blocks_and_preserves_sources(self, monkeypatch):
        """Échec de `sha256_json` → bloquant, sans écraser les sources par None.

        Point bloquant de revue : un échec de hash ne doit pas devenir un faux
        cas legacy (non bloquant).
        """
        import stagewise_optimizer as stagewise

        monkeypatch.setattr(stagewise, "sha256_json", lambda _v: None)

        report = self._final_report()
        original = report["last_stage_wfo_results"]["in_sample_performance"][0]["params_sha"]
        results = build_stagewise_wfo_results(report)

        # les sources ne sont PAS remplacées par None
        assert results["in_sample_performance"][0]["params_sha"] == original
        assert results["window_results"][0]["params_sha"] == original

        recon = results["params_sha_reconciliation"]
        assert recon["ok"] is False
        assert all(a["code"] == "hash_echec" for a in recon["anomalies"]), recon["anomalies"]

        report_int = check_run_integrity(results)
        assert report_int["ok"] is False
        check = report_int["checks"]["params_sha_reconciliation"]
        assert check["blocking"] is True
        assert check.get("formal_verification") is False
        assert "hash_echec" in check["codes"]

    def test_absent_source_fingerprint_is_not_certified(self):
        """Source sans empreinte → elle reste absente, jamais certifiée.

        Point bloquant de revue : l'absence de preuve n'est pas une preuve.  Un
        run legacy doit rester signalé non vérifié (§5.0 ``inferred_only`` /
        ``partial``), jamais ``formal_verification=True`` par reconstruction.
        """
        report = self._final_report()
        report["last_stage_wfo_results"]["out_of_sample_performance"][1].pop("params_sha")

        results = build_stagewise_wfo_results(report)
        assert "params_sha" not in results["out_of_sample_performance"][1], (
            "une empreinte a été fabriquée sur une source qui n'en portait pas"
        )

        check = check_run_integrity(results)["checks"]["params_matched"]
        assert check.get("formal_verification") is False, check
        assert check.get("verified") == "partial", check        # 5/6 côtés couverts
        assert check["ok"] is False and check["blocking"] is True, check

    def test_fully_legacy_run_stays_inferred_only(self):
        """Run entièrement legacy (aucune empreinte) → ``inferred_only``, non bloquant."""
        report = self._final_report()
        lswr = report["last_stage_wfo_results"]
        for side in ("in_sample_performance", "out_of_sample_performance"):
            for row in lswr[side]:
                row.pop("params_sha", None)
        for wr in lswr["window_results"]:
            wr.pop("params_sha", None)

        results = build_stagewise_wfo_results(report)
        for wr in results["window_results"]:
            assert "params_sha" not in wr

        check = check_run_integrity(results)["checks"]["params_matched"]
        assert check.get("verified") == "inferred_only", check
        assert check.get("formal_verification") is False, check
        assert check["blocking"] is False, check   # legacy = non vérifiable, pas fautif
