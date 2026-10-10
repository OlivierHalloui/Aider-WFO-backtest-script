"""Unit tests for T5 — §5.2 artefact export + post-final-backtest trigger.

Covers ``services/export_utils.py`` (canonical ``quant_analysis.json`` + derived
Markdown) and ``ui/quant_analysis_panel.py`` helpers (``final_trades_frame``,
``all_trials_frame``, ``run_auto_quant_after_final_backtest``).
"""

from __future__ import annotations

import json

import pandas as pd

from services.export_utils import quant_analysis_json, quant_analysis_markdown
from services.quant_expert import build_sealed_context
from ui.quant_analysis_panel import (
    all_trials_frame,
    final_trades_frame,
    run_auto_quant_after_final_backtest,
)


def _analysis():
    return {
        "schema_version": "quant_analysis.v1",
        "run_id": "run-1",
        "input_digest": "abc",
        "indicator_version": "1.0.0",
        "generated_at": "2026-10-09T00:00:00+00:00",
        "verdict": "WATCH",
        "verdict_scope": "exploratoire",
        "confidence": "moyenne",
        "verdict_justification": "Érosion de {{Q6.median_erosion_sharpe_pct}} sur {{Q1.budget}} évaluations.",
        "blocking_findings": [{"id": "B1", "constat": "Résultat négatif.", "evidence_refs": ["Q8"]}],
        "findings": [
            {"id": "F1", "niveau": "majeur", "constat": "Érosion marquée ({{Q6.median_erosion_sharpe_pct}}).",
             "evidence_refs": ["Q6", "W1"], "interpretation": "Sur-ajustement probable.",
             "condition": "si > 40 %"},
        ],
        "recommandations": [
            {"action": "Réduire l'espace", "priorite": 1, "motif": "Budget {{Q1.budget}}.",
             "critere_de_validation": "Q1 ≥ 200"},
        ],
        "limitations": ["Séries par barre absentes."],
        "accord_avec_preverdict": False,
        "preverdict_local": "WATCH",
    }


def _diagnostics():
    return {
        "manifest": {"run_id": "run-1", "input_digest": "abc", "timeframe": "5s"},
        "indicators": {
            "Q1": {"budget": {"value": 286, "available": True, "raison": None}},
            "Q6": {"median_erosion_sharpe_pct": {"value": 12.5, "available": True, "raison": None}},
        },
        "pre_verdict": {"verdict": "WATCH", "scope": "exploratoire"},
    }


# ---------------------------------------------------------------------------
# quant_analysis_json — artefact canonique
# ---------------------------------------------------------------------------

def test_quant_analysis_json_round_trip():
    out = quant_analysis_json(_analysis())
    parsed = json.loads(out)
    assert parsed["schema_version"] == "quant_analysis.v1"
    assert parsed["verdict"] == "WATCH"


def test_quant_analysis_json_keys_sorted_and_stable():
    """L'artefact doit être byte-stable pour une analyse donnée."""
    a = quant_analysis_json(_analysis())
    b = quant_analysis_json(_analysis())
    assert a == b
    keys = list(json.loads(a).keys())
    assert keys == sorted(keys)


def test_quant_analysis_json_rejects_nan():
    """§5.2 : NaN/Infinity rejetés, jamais émis."""
    bad = {"verdict": "GO", "score": float("nan")}
    try:
        quant_analysis_json(bad)
    except ValueError:
        pass                                   # allow_nan=False lève -> conforme
    else:
        # sanitize_for_json peut avoir neutralisé le NaN en amont : alors il ne
        # doit pas subsister dans le document émis.
        assert "NaN" not in quant_analysis_json(bad)


def test_quant_analysis_json_empty_is_valid():
    assert json.loads(quant_analysis_json(None)) == {}
    assert json.loads(quant_analysis_json("pas un mapping")) == {}


# ---------------------------------------------------------------------------
# quant_analysis_markdown — export dérivé
# ---------------------------------------------------------------------------

def test_markdown_has_expected_sections():
    md = quant_analysis_markdown(_analysis(), sealed=build_sealed_context(_diagnostics()))
    for section in ("# Analyse quant", "## Justification", "## Constats bloquants",
                    "## Constats", "## Recommandations", "## Limites déclarées"):
        assert section in md, section
    assert "WATCH" in md
    assert "run-1" in md and "abc" in md


def test_markdown_resolves_citations_and_keeps_them_traceable():
    sealed = build_sealed_context(_diagnostics())
    md = quant_analysis_markdown(_analysis(), sealed=sealed)
    assert "12.5" in md                          # valeur de l'app (source de vérité)
    assert "«Q6.median_erosion_sharpe_pct»" in md  # traçabilité conservée
    assert "286" in md


def test_markdown_without_sealed_leaves_citations_verbatim():
    md = quant_analysis_markdown(_analysis(), sealed=None)
    assert "{{Q6.median_erosion_sharpe_pct}}" in md   # jamais inventé


def test_markdown_empty_analysis():
    assert "Aucune analyse disponible" in quant_analysis_markdown(None)
    assert "Aucune analyse disponible" in quant_analysis_markdown({})


# ---------------------------------------------------------------------------
# Evidence frames
# ---------------------------------------------------------------------------

class _Trades:
    def __init__(self, frame):
        self.records_readable = frame


class _Portfolio:
    def __init__(self, frame):
        self.trades = _Trades(frame)


def test_final_trades_frame_normalizes_vectorbt_columns():
    """VectorBT capitalise (`Return`, `PnL`) ; le pipeline lit en minuscules."""
    frame = pd.DataFrame({"Return": [0.02, -0.01], "PnL": [2.0, -1.0],
                          "Size": [1.0, 1.0], "Direction": ["Long", "Short"]})
    out = final_trades_frame(_Portfolio(frame))
    assert out is not None
    for col in ("return", "pnl", "size", "direction"):
        assert col in out.columns, col
    assert list(out["return"]) == [0.02, -0.01]
    assert list(out["pnl"]) == [2.0, -1.0]


def test_final_trades_frame_none_on_missing_or_empty():
    assert final_trades_frame(None) is None
    assert final_trades_frame(_Portfolio(pd.DataFrame())) is None

    class _NoTrades:
        trades = None

    assert final_trades_frame(_NoTrades()) is None


def test_all_trials_frame_collects_across_windows():
    wfo = {
        "window_results": [
            {"optimization_trials": [{"score": 1.0}, {"score": 2.0}]},
            {"optimization_trials": [{"score": 3.0}]},
            {"optimization_trials": []},
        ]
    }
    out = all_trials_frame(wfo)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 3
    assert list(out["score"]) == [1.0, 2.0, 3.0]
    assert all_trials_frame({"window_results": []}) is None
    assert all_trials_frame(None) is None


# ---------------------------------------------------------------------------
# Trigger après le backtest final (§5.2 / §10)
# ---------------------------------------------------------------------------

def test_auto_after_final_backtest_disabled_is_noop():
    pf = _Portfolio(pd.DataFrame({"Return": [0.01], "PnL": [1.0]}))
    assert run_auto_quant_after_final_backtest(pf, config={"quant_auto_analysis": False}) == {}
    # config absent -> défaut coché : on tente (mais le runner injecté intercepte)
    assert run_auto_quant_after_final_backtest(pf, config=None,
                                              runner=lambda d, **k: {"status": "ok"})["status"] == "ok"


def test_auto_after_final_backtest_forwards_final_trades(monkeypatch):
    """`final_trades` doit atteindre `compute_quant_indicators` (preuve §5.0)."""
    pf = _Portfolio(pd.DataFrame({"Return": [0.02, -0.01], "PnL": [2.0, -1.0]}))
    seen = {}

    def fake_compute(source, **kwargs):
        seen.update(kwargs)
        return {"manifest": {"run_id": "r", "input_digest": "d"},
                "integrity": {"ok": True, "causes": [], "checks": {}},
                "indicators": {}, "pre_verdict": {"verdict": "GO", "scope": "exploratoire", "criteria": {}, "rationale": []}}

    monkeypatch.setattr("services.quant_indicators.compute_quant_indicators", fake_compute)
    out = run_auto_quant_after_final_backtest(
        pf,
        wfo_results={"window_results": []},
        config={"quant_auto_analysis": True, "fees_pct": 0.1},
        runner=lambda d, **k: {"status": "ok"},
    )
    assert out["status"] == "ok"
    trades = seen["final_trades"]
    assert isinstance(trades, pd.DataFrame)
    assert len(trades) == 2
    assert list(trades["return"]) == [0.02, -0.01]
    assert seen["config"] == {"quant_auto_analysis": True, "fees_pct": 0.1}


def test_auto_after_final_backtest_never_raises(monkeypatch):
    """Le déclenchement ne doit jamais casser le backtest final (§5.2)."""
    monkeypatch.setattr(
        "services.quant_expert.auto_analyze_run",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    pf = _Portfolio(pd.DataFrame({"Return": [0.01], "PnL": [1.0]}))
    out = run_auto_quant_after_final_backtest(pf, config={"quant_auto_analysis": True})
    assert out["status"] == "error"
    assert "boom" in out["raison"]
    assert out["marche_a_suivre"]


def test_auto_after_final_backtest_integrity_ko_issues_no_network_call(monkeypatch):
    """Réserve T5 : intégrité KO -> aucun appel A2A (§5.3 cas a).

    `probe_agent_card` est remplacé par une sentinelle qui lève : s'il est
    atteint, un aller réseau a été tenté et le test échoue.
    """
    pf = _Portfolio(pd.DataFrame({"Return": [0.01], "PnL": [1.0]}))

    def _must_not_probe(*_a, **_k):
        raise AssertionError("un sondage réseau a été émis alors que l'intégrité est KO")

    monkeypatch.setattr("services.quant_expert.probe_agent_card", _must_not_probe)

    # wfo_results vides -> check_run_integrity échoue -> call_quant_expert
    # retourne non_evaluable SANS toucher au réseau.
    out = run_auto_quant_after_final_backtest(
        pf,
        wfo_results={"window_results": []},
        config={"quant_auto_analysis": True},
    )
    assert out["status"] == "non_evaluable"
    assert out["analysis"] is None
    assert out["marche_a_suivre"]


# ---------------------------------------------------------------------------
# Correctifs de revue T5 (5 points bloquants)
# ---------------------------------------------------------------------------

def test_config_checkbox_key_is_honored_by_trigger():
    """Point 1 : `quant_auto_analysis` lu depuis la config, pas un défaut figé."""
    pf = _Portfolio(pd.DataFrame({"Return": [0.01], "PnL": [1.0]}))

    def _sentinel(*_a, **_k):
        raise AssertionError("l'auto-analyse a tourné alors que la case est décochée")

    assert run_auto_quant_after_final_backtest(
        pf, config={"quant_auto_analysis": False}, runner=_sentinel,
    ) == {}
    # la clé absente -> défaut coché : le chemin est tenté
    assert run_auto_quant_after_final_backtest(
        pf, config={}, runner=lambda d, **k: {"status": "ok"},
    )["status"] == "ok"


def test_persist_analysis_result_survives_rerun_and_can_be_cleared():
    """Point 2 : l'analyse manuelle est persistée (rerun + export ZIP)."""
    from ui.quant_analysis_panel import persist_analysis_result

    state = {}
    ok = {"status": "ok", "analysis": {"run_id": "run-1", "input_digest": "abc"}}
    persist_analysis_result(state, ok)
    assert state["quant_analysis"]["analysis"]["run_id"] == "run-1"

    # un échec ne doit PAS écraser une analyse déjà obtenue
    persist_analysis_result(state, {"status": "timeout", "analysis": None})
    assert state["quant_analysis"]["analysis"]["run_id"] == "run-1"

    # « Effacer » purge
    persist_analysis_result(state, {"status": "cleared", "analysis": None})
    assert "quant_analysis" not in state


def test_evidence_cache_key_uses_content_not_cardinality():
    """Point 3 : cardinalité constante mais rendements différents -> clés distinctes."""
    from ui.quant_analysis_panel import evidence_cache_key

    a = pd.DataFrame({"return": [0.02, -0.01], "pnl": [2.0, -1.0]})
    b = pd.DataFrame({"return": [0.05, -0.04], "pnl": [5.0, -4.0]})   # même taille
    assert len(a) == len(b)
    assert evidence_cache_key("run", a) != evidence_cache_key("run", b)
    # même contenu -> même clé (pas de re-calcul inutile)
    assert evidence_cache_key("run", a) == evidence_cache_key("run", a.copy())


def test_results_signature_covers_metrics_not_just_identity():
    """Un changement de métrique OOS doit invalider les diagnostics en cache."""
    from ui.quant_analysis_panel import _results_signature

    base = {
        "run_id": "r",
        "window_results": [{"best_params": {"p": 1}}],
        "in_sample_performance": [{"window": 1, "return": 5.0}],
        "out_of_sample_performance": [{"window": 1, "return": 5.0}],
        "best_params": [{"p": 1}],
    }
    changed = {
        **base,
        "out_of_sample_performance": [{"window": 1, "return": 9.0}],   # 5.0 -> 9.0
    }
    assert _results_signature(base) != _results_signature(changed)
    # identité seule (run_id, nb fenêtres, best_params) inchangée -> la sonde
    # précédente passait ; c'est bien la métrique qui doit invalider.
    assert (base["run_id"], len(base["window_results"]), base["best_params"]) == (
        changed["run_id"], len(changed["window_results"]), changed["best_params"],
    )
    # même contenu -> même signature
    assert _results_signature(base) == _results_signature({**base})


def test_evidence_cache_key_covers_every_input():
    """Point 2 de la 2ᵉ revue : config et grille font partie de l'invalidation."""
    from ui.quant_analysis_panel import evidence_cache_key

    trades = pd.DataFrame({"return": [0.02], "pnl": [2.0]})
    base = evidence_cache_key("run", trades, config={"fees_pct": 0.1}, param_grid={"p": [1]})
    assert evidence_cache_key("run", trades, config={"fees_pct": 0.5}, param_grid={"p": [1]}) != base
    assert evidence_cache_key("run", trades, config={"fees_pct": 0.1}, param_grid={"p": [2]}) != base
    assert evidence_cache_key("run", trades, config={"fees_pct": 0.1},
                              param_grid={"p": [1]}, all_trials=pd.DataFrame({"s": [1.0]})) != base
    assert evidence_cache_key("run", trades, config={"fees_pct": 0.1}, param_grid={"p": [1]}) == base


def test_ensure_quant_diagnostics_single_path_digest(monkeypatch):
    """Trigger et onglet donnent le MÊME digest, via un état **proxy** (pas dict).

    C'est la condition pour que l'auto-analyse soit acceptée à l'affichage et
    à l'export (`analysis_matches_run`).  `st.session_state` est un
    `SessionStateProxy` (MutableMapping), jamais un `dict`.
    """
    from collections.abc import MutableMapping

    from ui.quant_analysis_panel import ensure_quant_diagnostics

    class ProxyState(MutableMapping):
        """Stand-in for Streamlit's SessionStateProxy."""

        def __init__(self):
            self._d = {}

        def __getitem__(self, k):
            return self._d[k]

        def __setitem__(self, k, v):
            self._d[k] = v

        def __delitem__(self, k):
            del self._d[k]

        def __iter__(self):
            return iter(self._d)

        def __len__(self):
            return len(self._d)

    assert not isinstance(ProxyState(), dict)      # bien un proxy, pas un dict

    seen = {}

    def fake_compute(source, **kwargs):
        seen.setdefault("calls", []).append(kwargs)
        import hashlib
        sig = hashlib.sha256(repr(sorted(map(str, kwargs.items()))).encode()).hexdigest()[:8]
        return {"manifest": {"run_id": "r", "input_digest": sig},
                "integrity": {"ok": True, "causes": [], "checks": {}},
                "indicators": {}, "pre_verdict": {"verdict": "GO", "scope": "s", "criteria": {}, "rationale": []}}

    monkeypatch.setattr("services.quant_indicators.compute_quant_indicators", fake_compute)

    pf = _Portfolio(pd.DataFrame({"Return": [0.02], "PnL": [2.0]}))
    wfo = {"window_results": [{"optimization_trials": [{"score": 1.0}]}]}
    cfg = {"fees_pct": 0.1, "quant_auto_analysis": True}

    state_a, state_b = ProxyState(), ProxyState()
    diag_a, trials_a, _grid_a = ensure_quant_diagnostics(
        state_a, wfo_results=wfo, config=cfg, final_portfolio=pf,
    )
    diag_b, trials_b, _grid_b = ensure_quant_diagnostics(
        state_b, wfo_results=wfo, config=cfg, final_portfolio=pf,
    )
    assert diag_a is not None and diag_b is not None
    assert diag_a["manifest"]["input_digest"] == diag_b["manifest"]["input_digest"]
    assert trials_a is not None and len(trials_a) == 1

    # le cache partagé fonctionne AUSSI avec le proxy : pas de recalcul
    assert "quant_diagnostics" in state_a and "quant_diag_key" in state_a
    n = len(seen["calls"])
    ensure_quant_diagnostics(state_a, wfo_results=wfo, config=cfg, final_portfolio=pf)
    assert len(seen["calls"]) == n, "le cache a été contourné avec un état proxy"

    # persistance dans le proxy également
    from ui.quant_analysis_panel import persist_analysis_result
    persist_analysis_result(state_a, {"status": "ok", "analysis": {"run_id": "r"}})
    assert state_a["quant_analysis"]["analysis"]["run_id"] == "r"


def test_quant_artifacts_refuse_foreign_analysis():
    """Point 4 : une analyse étrangère au run n'entre jamais dans le ZIP."""
    from services.export_utils import quant_artifacts_for_run

    diag = {"manifest": {"run_id": "run-1", "input_digest": "abc"}}
    mine = dict(_analysis())
    assert quant_artifacts_for_run(mine, diag) is not None

    foreign = dict(_analysis())
    foreign["run_id"] = "AUTRE-RUN"
    assert quant_artifacts_for_run(foreign, diag) is None
    assert quant_artifacts_for_run({}, diag) is None
    assert quant_artifacts_for_run(None, diag) is None

    # analyse sans identifiants -> refusée également
    assert quant_artifacts_for_run({"verdict": "GO"}, diag) is None


def test_render_panel_integrity_ko_never_probes_network(monkeypatch):
    """Point 5 : l'UI ne sonde pas le pair quand l'intégrité est KO (§5.3 cas a)."""
    from tests.test_quant_analysis_panel import StubContainer as _Stub
    from ui.quant_analysis_panel import render_quant_expert_panel

    def _must_not_probe(*_a, **_k):
        raise AssertionError("sondage réseau émis alors que l'intégrité est KO")

    monkeypatch.setattr("services.quant_expert.probe_agent_card", _must_not_probe)
    stub = _Stub()
    result = render_quant_expert_panel(
        diagnostics={
            "manifest": {"run_id": "r", "input_digest": "d"},
            "integrity": {"ok": False, "causes": ["run_present: vide"], "checks": {}},
            "pre_verdict": {"verdict": "NO_GO", "scope": "exploratoire"},
        },
        container=stub, show_toggle=False,
    )
    assert result == {}
    assert any("NO_GO" in t for t in stub.texts())
