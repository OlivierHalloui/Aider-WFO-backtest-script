"""Unit tests for T5 — §5.2 artefact export + post-final-backtest trigger.

Covers ``services/export_utils.py`` (canonical ``quant_analysis.json`` + derived
Markdown) and ``ui/quant_analysis_panel.py`` helpers (``final_trades_frame``,
``all_trials_frame``, ``run_auto_quant_after_final_backtest``).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

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
    """Une analyse étrangère au run n'entre jamais dans le ZIP."""
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


# ---------------------------------------------------------------------------
# Replay (§6.4) et isolation inter-runs
# ---------------------------------------------------------------------------


def test_restore_rejects_digest_from_other_run_with_same_run_id():
    """A manifest carrying B's replay digest must not attach to A."""
    from services.export_utils import restore_quant_artifacts
    from services.quant_indicators import build_run_manifest

    run_a = {"run_id": "A", "window_results": [], "score": 1.0}
    run_b = {**run_a, "score": 9.0}
    manifest_b = build_run_manifest(run_b, config={"fees_pct": 0.1})
    restored = restore_quant_artifacts(
        {"run_manifest.json": manifest_b}, run_a, config={"fees_pct": 0.1},
    )
    assert restored == {"analysis": None, "diagnostics": None}


def test_restored_q1_survives_next_render_and_invalidates_on_run_change():
    """Archived facts survive rerenders but never contaminate a changed run."""
    from services.export_utils import apply_quant_restore, restore_quant_artifacts
    from services.quant_indicators import compute_quant_indicators
    from ui.quant_analysis_panel import ensure_quant_diagnostics

    run = {"run_id": "A", "window_results": []}
    config = {"fees_pct": 0.1}
    diagnostics = compute_quant_indicators(run, config=config)
    archived_indicators = {"Q1": {"budget": {"value": 987654, "available": True}}}
    restored = restore_quant_artifacts({
        "run_manifest.json": diagnostics["manifest"],
        "quant_indicators.json": {"indicators": archived_indicators},
    }, run, config=config)
    state = {}
    apply_quant_restore(state, restored)
    for _ in range(2):
        actual, _, _ = ensure_quant_diagnostics(state, wfo_results=run, config=config)
        assert actual["indicators"] == archived_indicators
        assert actual["source"] == "archived"
        assert actual["integrity"]["ok"] is False
    # Loading a ZIP does not replace the UI config: archive facts still survive.
    actual, _, _ = ensure_quant_diagnostics(state, wfo_results=run, config={"fees_pct": 0.9})
    assert actual["indicators"] == archived_indicators
    changed, _, _ = ensure_quant_diagnostics(
        state, wfo_results={**run, "run_id": "B"}, config=config,
    )
    assert changed["manifest"]["run_id"] == "B"
    assert changed.get("source") != "archived"
    assert changed["indicators"] != archived_indicators


def test_zip_export_replay_digest_matches_serialized_results(monkeypatch):
    """The payload builder may enrich results before they are serialized."""
    import zipfile
    from types import SimpleNamespace

    from services.export_utils import restore_quant_artifacts
    from ui import export_panel

    run = {"run_id": "A", "window_results": []}
    config = {"fees_pct": 0.1}
    state = {"wfo_results": run}
    def unexpected_error(message):
        raise AssertionError(message)

    monkeypatch.setattr(export_panel, "st", SimpleNamespace(
        session_state=state, error=unexpected_error, warning=unexpected_error,
    ))

    def build_payload():
        enriched = {**run, "robust_set_summary": {"selected": [1]}}
        state["wfo_results"] = enriched
        return {"wfo_results": enriched, "config": config}

    empty = lambda *args, **kwargs: {}
    export_args = dict(
        build_results_payload=build_payload,
        build_expert_context_pack_for_export=empty,
        compute_pine_order_semantics_report=empty,
        build_pine_beta_readiness_report=empty,
        build_pine_execution_gate_report=empty,
        resolve_pine_source_for_artifacts=empty,
        resolve_pine_libraries_for_artifacts=lambda: [],
        build_pine_generation_trace=empty,
        build_pine_artifacts_manifest=empty,
        build_window_info_dataframe=lambda results: pd.DataFrame(),
        build_trials_dataframe_from_results=lambda results: pd.DataFrame(),
    )
    archive = export_panel._export_results_zip(**export_args)
    with zipfile.ZipFile(archive) as zf:
        payload = json.loads(zf.read("results.json"))
        artifacts = {name: json.loads(zf.read(name)) for name in (
            "run_manifest.json", "quant_indicators.json",
        )}
    restored = restore_quant_artifacts(
        artifacts, payload["wfo_results"], config=payload["config"],
    )
    assert restored["diagnostics"] is not None
    from ui.quant_analysis_panel import ensure_quant_diagnostics
    archive.seek(0)
    export_panel._load_results_zip(
        archive, validate_strategy_spec_v1=empty,
        apply_parity_reference_payload=empty,
        persist_pine_source_text=empty,
        persist_generated_strategy_text=empty,
        persist_pine_library_text=empty,
    )
    assert state["quant_diagnostics"]["source"] == "archived"
    retained, _, _ = ensure_quant_diagnostics(
        state, wfo_results=payload["wfo_results"], config={"fees_pct": 0.9},
    )
    assert retained is not None and retained["source"] == "archived"
    assert retained["indicators"] == artifacts["quant_indicators.json"]["indicators"]
    export_args["build_results_payload"] = lambda: {
        "wfo_results": payload["wfo_results"], "config": {"fees_pct": 0.9},
    }
    second_archive = export_panel._export_results_zip(**export_args)
    with zipfile.ZipFile(second_archive) as zf:
        second_payload = json.loads(zf.read("results.json"))
        second_artifacts = {name: json.loads(zf.read(name)) for name in (
            "run_manifest.json", "quant_indicators.json",
        )}
    assert second_payload["config"] == config
    assert restore_quant_artifacts(
        second_artifacts, second_payload["wfo_results"], config=second_payload["config"],
    )["diagnostics"] is not None


@pytest.mark.parametrize("mutation", ["config", "metric", "missing_digest", "bad_digest"])
def test_replay_rejects_changed_or_unverifiable_inputs(mutation):
    """Replay binding includes config and content, and fails closed."""
    from services.export_utils import restore_quant_artifacts
    from services.quant_indicators import build_run_manifest

    run = {"run_id": "A", "score": 1.0}
    config = {"fees_pct": 0.1}
    manifest = build_run_manifest(run, config=config)
    if mutation == "config":
        config = {"fees_pct": 0.9}
    elif mutation == "metric":
        run = {**run, "score": 2.0}
    elif mutation == "missing_digest":
        manifest.pop("input_digest_replay")
    else:
        manifest["input_digest_replay"] = "arbitrary"
    assert restore_quant_artifacts(
        {"run_manifest.json": manifest}, run, config=config,
    ) == {"analysis": None, "diagnostics": None}


def test_archive_invalidates_on_content_change_even_with_same_run_id():
    """A reused run_id must not preserve facts from older run content."""
    from services.export_utils import apply_quant_restore, restore_quant_artifacts
    from services.quant_indicators import build_run_manifest
    from ui.quant_analysis_panel import ensure_quant_diagnostics

    run = {"run_id": "A", "score": 1.0}
    state = {}
    apply_quant_restore(state, restore_quant_artifacts(
        {"run_manifest.json": build_run_manifest(run)}, run,
    ))
    actual, _, _ = ensure_quant_diagnostics(state, wfo_results={**run, "score": 2.0})
    assert actual is not None
    assert actual.get("source") != "archived"
    assert "quant_diag_key" in state


def test_app_payload_builder_preserves_archive_before_robust_summary_refresh():
    """Reviewer probe: archived top-N=2 must survive widget top-N=3."""
    import ast
    import datetime
    from pathlib import Path
    from types import SimpleNamespace

    from domain.serialization import sanitize_for_json
    from services.export_utils import apply_quant_restore, restore_quant_artifacts
    from services.quant_indicators import build_run_manifest
    from ui.final_backtest_panel import _build_robust_set_summary
    from ui.quant_analysis_panel import ensure_quant_diagnostics

    # Execute the real builder without launching app.py's Streamlit top-level UI.
    source = ast.parse((Path(__file__).parents[1] / "app.py").read_text())
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                and n.name == "_build_results_payload")
    config = {"robust_top_n_per_window": 2}
    run = {"run_id": "A", "window_results": []}
    run["robust_set_summary"] = _build_robust_set_summary(run, config)
    archived = {"Q1": {"budget": {"value": 987654, "available": True}}}
    state = {"wfo_results": run}
    apply_quant_restore(state, restore_quant_artifacts({
        "run_manifest.json": build_run_manifest(run, config=config),
        "quant_indicators.json": {"indicators": archived},
    }, run, config=config))
    namespace = {
        "get_current_config": lambda: {"robust_top_n_per_window": 3},
        "WFOSessionState": SimpleNamespace(read=lambda: SimpleNamespace(
            wfo_results=state["wfo_results"], final_portfolio=None)),
        "st": SimpleNamespace(session_state=state),
        "_build_robust_set_summary": _build_robust_set_summary,
        "_build_traceability_payload": lambda **kwargs: kwargs,
        "_sanitize_for_json": sanitize_for_json,
        "datetime": datetime,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "app.py", "exec"), namespace)
    payload = namespace["_build_results_payload"]()
    actual, _, _ = ensure_quant_diagnostics(
        state, wfo_results=payload["wfo_results"], config=payload["config"],
    )
    assert actual is not None and actual["indicators"] == archived
    assert payload["wfo_results"] == run
    assert payload["config"] == config
    assert payload["traceability"]["config_snapshot"] == config


def test_restore_quant_artifacts_round_trip():
    """§6.4 : les artefacts présents sont réimportés (ZIP -> import -> état)."""
    from services.export_utils import (
        quant_analysis_json,
        restore_quant_artifacts,
    )

    diag = {
        "manifest": {"run_id": "run-1", "input_digest": "abc", "timeframe": "5s"},
        "indicators": {"Q1": {"budget": {"value": 286, "available": True, "raison": None}}},
        "integrity": {"ok": True, "causes": [], "checks": {}},
        "pre_verdict": {"verdict": "WATCH", "scope": "exploratoire"},
    }
    analysis = dict(_analysis())                     # run_id=run-1, input_digest=abc
    wfo = {"run_id": "run-1", "window_results": []}
    from services.quant_indicators import build_run_manifest
    diag["manifest"] = build_run_manifest(wfo)
    analysis["input_digest"] = diag["manifest"]["input_digest"]

    # chemin aller : ce que l'export écrit (bloc scellé complet)
    artifacts = {
        "quant_analysis.json": json.loads(quant_analysis_json(analysis)),
        "quant_indicators.json": {
            "indicators": diag["indicators"],
            "integrity": diag["integrity"],
            "pre_verdict": diag["pre_verdict"],
        },
        "run_manifest.json": diag["manifest"],
    }
    restored = restore_quant_artifacts(artifacts, wfo)
    assert restored["analysis"] is not None
    assert restored["analysis"]["verdict"] == "WATCH"
    assert restored["diagnostics"] is not None
    assert restored["diagnostics"]["manifest"]["run_id"] == "run-1"
    assert restored["diagnostics"]["indicators"]["Q1"]["budget"]["value"] == 286
    # `integrity` est RECALCULÉ sur le run importé (jamais cru sur pièce) et
    # `pre_verdict` restauré, sinon `expert_gate` refuserait (« cause inconnue »).
    assert "checks" in restored["diagnostics"]["integrity"]
    assert restored["diagnostics"]["integrity"]["ok"] is False   # run minimal -> KO, honnête
    # §5.3 cas a : intégrité KO => pré-verdict local NO_GO, jamais un WATCH restauré
    assert restored["diagnostics"]["pre_verdict"]["verdict"] == "NO_GO"

    # le gate suit le contrôle recalculé : ici KO -> refus, sans appel réseau
    from ui.quant_analysis_panel import expert_gate

    enabled, reason = expert_gate(restored["diagnostics"], peer_reachable=True)
    assert enabled is False
    assert "intégrité" in reason


def test_restore_rejects_tampered_digest_certifying_itself():
    """Reproduction du revueur : bon `run_id`, `input_digest` arbitraire des deux
    côtés, et un artefact qui prétend `integrity.ok=True`.
    L'intégrité ne doit JAMAIS être crue sur pièce : elle est recalculée, donc le
    run est refusé et l'analyse n'est pas restaurée.
    """
    from services.export_utils import restore_quant_artifacts
    from ui.quant_analysis_panel import expert_gate

    tampered = {
        "quant_analysis.json": {"run_id": "run-1", "input_digest": "ARBITRAIRE",
                                "verdict": "GO", "preverdict_local": "GO"},
        "quant_indicators.json": {
            "indicators": {},
            "integrity": {"ok": True, "causes": [], "checks": {}},   # FAUSSEMENT ok
            "pre_verdict": {"verdict": "GO", "scope": "deploiement"},
        },
        "run_manifest.json": {"run_id": "run-1", "input_digest": "ARBITRAIRE"},
    }
    # A matching replay digest does not certify missing raw evidence.
    from services.quant_indicators import build_run_manifest
    tampered["run_manifest.json"]["input_digest_replay"] = build_run_manifest(
        {"run_id": "run-1", "window_results": []},
    )["input_digest_replay"]
    # run importé vide -> le contrôle recalculé est KO, malgré le ok=True du fichier
    restored = restore_quant_artifacts(tampered, {"run_id": "run-1", "window_results": []})
    assert restored["diagnostics"] is not None
    assert restored["diagnostics"]["integrity"]["ok"] is False, "l'artefat s'est auto-certifié"
    enabled, reason = expert_gate(restored["diagnostics"], peer_reachable=True)
    assert enabled is False, "expert_gate a accepté un artefact auto-certifié"

    # un manifeste SANS input_digest n'est pas rattachable -> refusé
    no_digest = {"run_manifest.json": {"run_id": "run-1"}}
    assert restore_quant_artifacts(no_digest, {"run_id": "run-1"})["diagnostics"] is None


def test_apply_quant_restore_two_successive_imports():
    """Deux imports successifs : l'état du run A ne survit jamais au run B."""
    from services.export_utils import apply_quant_restore

    state = {}
    apply_quant_restore(state, {
        "diagnostics": {"manifest": {"run_id": "A"}, "indicators": {"Q1": {}},
                        "integrity": {"ok": True}, "pre_verdict": {"verdict": "GO"}},
        "analysis": {"run_id": "A", "input_digest": "a", "verdict": "GO"},
    })
    assert state["quant_diagnostics"]["manifest"]["run_id"] == "A"
    assert state["quant_analysis"]["analysis"]["run_id"] == "A"

    # import du run B SANS artefacts quant -> tout l'état de A est purgé
    apply_quant_restore(state, {"analysis": None, "diagnostics": None})
    assert "quant_diagnostics" not in state
    assert "quant_analysis" not in state
    assert "quant_diag_key" not in state

    # puis import du run B avec ses artefacts -> état de B
    apply_quant_restore(state, {
        "diagnostics": {"manifest": {"run_id": "B"}, "indicators": {}, "integrity": {}, "pre_verdict": {}},
        "analysis": {"run_id": "B", "input_digest": "b", "verdict": "NO_GO"},
    })
    assert state["quant_diagnostics"]["manifest"]["run_id"] == "B"
    assert state["quant_analysis"]["analysis"]["run_id"] == "B"


def test_apply_quant_restore_accepts_session_proxy():
    """`st.session_state` est un SessionStateProxy, pas un dict."""
    from collections.abc import MutableMapping

    from services.export_utils import apply_quant_restore

    class ProxyState(MutableMapping):
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

    state = ProxyState()
    assert not isinstance(state, dict)
    state["quant_analysis"] = {"analysis": {"run_id": "OLD"}}
    apply_quant_restore(state, {"analysis": None, "diagnostics": None})
    assert "quant_analysis" not in state, "l'état précédent n'a pas été purgé via le proxy"
    apply_quant_restore(None, {"analysis": {"x": 1}})   # ne doit jamais lever


def test_restore_quant_artifacts_rejects_foreign_manifest():
    """Un manifeste d'un autre run n'est jamais restauré (§6.4)."""
    from services.export_utils import restore_quant_artifacts

    artifacts = {
        "quant_indicators.json": {"Q1": {}},
        "run_manifest.json": {"run_id": "AUTRE-RUN", "input_digest": "zzz"},
    }
    restored = restore_quant_artifacts(artifacts, {"run_id": "run-1", "window_results": []})
    assert restored["diagnostics"] is None
    assert restored["analysis"] is None


def test_restore_quant_artifacts_analysis_needs_matching_diagnostics():
    """Une analyse sans faits scellés rattachables n'est pas restaurée."""
    from services.export_utils import restore_quant_artifacts

    analysis = dict(_analysis())
    # pas de manifeste -> pas de rattachement vérifiable -> refus
    assert restore_quant_artifacts({"quant_analysis.json": analysis},
                                   {"run_id": "run-1"})["analysis"] is None
    # manifeste cohérent -> acceptée
    from services.quant_indicators import build_run_manifest
    manifest = build_run_manifest({"run_id": "run-1"})
    analysis["input_digest"] = manifest["input_digest"]
    ok = restore_quant_artifacts(
        {"quant_analysis.json": analysis,
         "run_manifest.json": manifest},
        {"run_id": "run-1"},
    )
    assert ok["analysis"] is not None


def test_restore_quant_artifacts_survives_corrupt_json():
    """Un artefact corrompu ne doit jamais faire échouer l'import."""
    from services.export_utils import restore_quant_artifacts

    out = restore_quant_artifacts({"quant_analysis.json": "pas un mapping"},
                                  {"run_id": "run-1"})
    assert out == {"analysis": None, "diagnostics": None}
    assert restore_quant_artifacts(None, None) == {"analysis": None, "diagnostics": None}


def test_export_uses_current_run_diagnostics_not_stale_session(monkeypatch):
    """Isolation : exporter le run B avec les diagnostics du run A en session.

    `ensure_quant_diagnostics` recalcule pour le run exporté — les faits d'un
    autre run ne doivent jamais fuiter dans le ZIP (§5.0).
    """
    from ui.quant_analysis_panel import ensure_quant_diagnostics

    def fake_compute(source, **kwargs):
        rid = source.get("run_id")
        return {"manifest": {"run_id": rid, "input_digest": f"digest-{rid}"},
                "integrity": {"ok": True, "causes": [], "checks": {}},
                "indicators": {}, "pre_verdict": {"verdict": "GO", "scope": "s", "criteria": {}, "rationale": []}}

    monkeypatch.setattr("services.quant_indicators.compute_quant_indicators", fake_compute)

    state = {}
    # run A laisse ses diagnostics en session
    ensure_quant_diagnostics(state, wfo_results={"run_id": "A", "window_results": []})
    assert state["quant_diagnostics"]["manifest"]["run_id"] == "A"

    # on exporte/runne B : le chemin commun doit servir les faits de B
    diag_b, _t, _g = ensure_quant_diagnostics(state, wfo_results={"run_id": "B", "window_results": []})
    assert diag_b is not None
    assert diag_b["manifest"]["run_id"] == "B"
    assert diag_b["manifest"]["input_digest"] == "digest-B"
    assert state["quant_diagnostics"]["manifest"]["run_id"] == "B"


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
