"""Unit tests for ui/quant_analysis_panel.py (T3).

Covers §5.0 (manifest + blocking integrity), §5.1 (sourced Q1–Q8 / OOS tables,
per-window before aggregates, « indisponible » policy — never 0) and §5.3
(pre-verdict criteria + rationale) of the cahier des charges.

Two levels of input are exercised on purpose:

- **hand-written** diagnostics (edge cases, absent blocks, empty inputs) ;
- **real T1 output** from ``compute_q6`` / ``compute_q7`` — this is what catches
  schema drift between T1 and the panel (e.g. reading a field T1 never emits).

The rendering helpers are exercised through a stub container so **no Streamlit
runtime is required**.
"""

from __future__ import annotations

import pandas as pd
import pytest

from services.quant_indicators import compute_q6, compute_q7
from ui.quant_analysis_panel import (
    INDICATOR_TITLES,
    MISSING,
    aggregate_table,
    all_indicator_flat,
    all_indicator_tables,
    criteria_table,
    fmt_metric,
    indicator_table,
    integrity_causes,
    integrity_ok,
    integrity_table,
    manifest_table,
    pre_verdict_rationale,
    q6_window_table,
    q7_window_table,
    render_quant_analysis_panel,
    unavailable_rows,
    verdict_badge,
)


# ---------------------------------------------------------------------------
# Stub Streamlit container (records calls; usable as a context manager)
# ---------------------------------------------------------------------------

class StubContainer:
    def __init__(self):
        self.calls: list[tuple] = []
        self.click = False   # simule l'état du bouton Streamlit (False = pas de clic)

    def __getattr__(self, name):
        if name == "button":
            def _button(*args, **kwargs):
                self.calls.append((name, args, kwargs))
                return self.click
            return _button

        def _recorder(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self
        return _recorder

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def texts(self) -> list[str]:
        return [str(a[0]) for (_n, a, _k) in self.calls if a]

    def kinds(self) -> list[str]:
        return [n for (n, _a, _k) in self.calls]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _avail(value, available=True, raison=None):
    return {"value": value, "available": available, "raison": raison}


def _diagnostics(integrity_ok_flag=True):
    return {
        "schema_version": "quant_indicators.v1",
        "indicator_version": "1.0.0",
        "generated_at": "2026-10-09T00:00:00+00:00",
        "manifest": {"run_id": "run-1", "input_digest": "abc", "timeframe": "5s", "n_windows": 2},
        "integrity": {
            "ok": integrity_ok_flag,
            "causes": [] if integrity_ok_flag else ["run_present: wfo_results absent ou vide"],
            "checks": {"run_present": {"ok": integrity_ok_flag, "raison": None}},
        },
        "indicators": {
            "Q1": {"budget": _avail(286), "plafond": _avail(500)},
            "Q6": {
                "median_erosion_sharpe_pct": _avail(12.5),
                "sharpe_final_recomputed": _avail(None, available=False, raison="rendements par barre absents"),
                "windows": [
                    {"window": 1, "erosion_sharpe_pct": _avail(10.0), "defined": True, "raison": None},
                    {"window": 2, "erosion_sharpe_pct": _avail(None, available=False, raison="dénominateur ≤ 0"),
                     "defined": False, "raison": "dénominateur ≤ 0"},
                ],
            },
            "OOS_UNCERTAINTY": {"n_windows": _avail(2), "n_trades_oos": _avail(120)},
        },
        "pre_verdict": {
            "verdict": "WATCH",
            "scope": "exploratoire",
            "status": "ok",
            "criteria": {
                "C1_integrite": {"verdict": "GO", "raison": None},
                "C3_erosion_sharpe": {"verdict": "WATCH", "raison": "érosion intermédiaire"},
            },
            "rationale": ["C3_erosion_sharpe: WATCH — érosion intermédiaire"],
        },
    }


# Real T1 output — the schema the panel MUST match (no invented fields).
def _real_wfo_results():
    return {
        "in_sample_performance": [
            {"window": 1, "sharpe": 0.030, "return": 10.0, "pqs": 2.0, "max_drawdown": -2.0},
            {"window": 2, "sharpe": -0.010, "return": 5.0, "pqs": 0.0, "max_drawdown": -1.0},
        ],
        "out_of_sample_performance": [
            {"window": 1, "sharpe": 0.012, "return": 4.0, "pqs": 1.0, "max_drawdown": -4.0},
            {"window": 2, "sharpe": 0.005, "return": 1.0, "pqs": 0.5, "max_drawdown": -3.0},
        ],
    }


# ---------------------------------------------------------------------------
# §5.3 — « indisponible », jamais 0
# ---------------------------------------------------------------------------

def test_fmt_metric_unavailable_never_zero():
    assert fmt_metric(_avail(None, available=False, raison="x")) == MISSING
    assert fmt_metric(_avail(0.0, available=False, raison="x")) == MISSING   # jamais 0
    assert MISSING == "indisponible"


def test_fmt_metric_available_and_edge_cases():
    assert fmt_metric(_avail(12.5)) == "12.5"
    assert fmt_metric(_avail(None)) == "—"          # disponible mais vide
    assert fmt_metric(_avail(True)) == "oui"
    assert fmt_metric(_avail(False)) == "non"
    assert fmt_metric(3) == "3"


def test_verdict_badge():
    assert verdict_badge("GO") == "🟢 GO"
    assert verdict_badge("WATCH") == "🟠 WATCH"
    assert verdict_badge("NO_GO") == "🔴 NO_GO"
    assert verdict_badge(None) == "None"   # jamais deviné, rendu verbatim


# ---------------------------------------------------------------------------
# §5.0 — manifest + integrity
# ---------------------------------------------------------------------------

def test_manifest_table_lists_fields():
    df = manifest_table(_diagnostics()["manifest"])
    assert list(df.columns) == ["champ", "valeur"]
    assert set(df["champ"]) == {"run_id", "input_digest", "timeframe", "n_windows"}
    assert df.loc[df["champ"] == "timeframe", "valeur"].iloc[0] == "5s"


def test_integrity_table_and_flags():
    integrity = _diagnostics()["integrity"]
    df = integrity_table(integrity)
    assert list(df.columns) == ["contrôle", "statut", "détail"]
    assert df.iloc[0]["contrôle"] == "run_present"
    assert df.iloc[0]["statut"] == "OK"
    assert integrity_ok(integrity) is True
    assert integrity_causes(integrity) == []


def test_integrity_ko_flags_and_causes():
    integrity = _diagnostics(integrity_ok_flag=False)["integrity"]
    assert integrity_ok(integrity) is False
    assert integrity_table(integrity).iloc[0]["statut"] == "ÉCHEC"
    assert integrity_causes(integrity) == ["run_present: wfo_results absent ou vide"]


def test_integrity_empty_inputs_no_crash():
    assert integrity_table(None).empty
    assert integrity_ok(None) is False
    assert integrity_causes(None) == []
    assert manifest_table(None).empty


# ---------------------------------------------------------------------------
# §5.1 — sourced indicator tables
# ---------------------------------------------------------------------------

def test_indicator_table_sourced_references():
    indicators = _diagnostics()["indicators"]
    df = indicator_table(indicators, "Q1")
    assert list(df.columns) == ["référence", "valeur", "disponible", "raison"]
    # chaque valeur est sourcée : le préfixe du bloc trace le champ exact
    assert set(df["référence"]) == {"Q1.budget", "Q1.plafond"}
    assert df.loc[df["référence"] == "Q1.budget", "valeur"].iloc[0] == "286"


def test_indicator_table_unavailable_kept_with_reason():
    indicators = _diagnostics()["indicators"]
    df = indicator_table(indicators, "Q6")
    row = df[df["référence"] == "Q6.sharpe_final_recomputed"].iloc[0]
    assert row["valeur"] == MISSING
    assert bool(row["disponible"]) is False
    assert row["raison"] == "rendements par barre absents"


def test_absent_block_yields_empty_table():
    """Un bloc absent est « non calculé » (table vide), jamais une ligne « — »."""
    df = indicator_table(_diagnostics()["indicators"], "Q4")
    assert df.empty
    assert list(df.columns) == ["référence", "valeur", "disponible", "raison"]


def test_all_indicator_tables_skips_absent_blocks():
    tables = all_indicator_tables(_diagnostics()["indicators"])
    assert set(tables) == {"Q1", "Q6", "OOS_UNCERTAINTY"}   # Q2–Q5, Q7, Q8 absents
    assert all(INDICATOR_TITLES[b] for b in tables)


def test_aggregate_table_excludes_window_rows():
    agg = aggregate_table(_diagnostics()["indicators"], "Q6")
    assert not agg["référence"].str.startswith("Q6.windows[").any()
    assert "Q6.median_erosion_sharpe_pct" in set(agg["référence"])


def test_per_window_entries_sourced_with_index():
    """Les entrées de fenêtre portent une référence indexée (Q6.windows[0].…)."""
    df = indicator_table(_diagnostics()["indicators"], "Q6")
    refs = set(df["référence"])
    assert "Q6.windows[0].erosion_sharpe_pct" in refs
    assert "Q6.windows[1].erosion_sharpe_pct" in refs


def test_unavailable_rows_only_non_available():
    df = unavailable_rows(_diagnostics()["indicators"])
    assert not df.empty
    assert (~df["disponible"].astype(bool)).all()
    assert "Q6.sharpe_final_recomputed" in set(df["référence"])
    assert "Q6.windows[1].erosion_sharpe_pct" in set(df["référence"])


def test_all_indicator_flat_includes_windows():
    flat = all_indicator_flat(_diagnostics()["indicators"])
    assert flat["référence"].str.startswith("Q6.windows[").any()


# ---------------------------------------------------------------------------
# §5.1 — per-window wide tables
# ---------------------------------------------------------------------------

def test_q6_window_table_complete_and_availability_from_avail():
    """Le champ « érosion calculée » vient de `_avail.available`, jamais d'un
    champ `defined` que T1 n'émet pas (point bloquant 2 de la revue)."""
    df = q6_window_table(_diagnostics()["indicators"])
    assert "érosion calculée" in df.columns
    assert [bool(v) for v in df["érosion calculée"]] == [True, False]  # dérivé de `available`
    assert df.iloc[0]["érosion Sharpe %"] == "10"
    assert df.iloc[1]["érosion Sharpe %"] == MISSING          # jamais 0
    assert df.iloc[0]["référence"] == "Q6.windows[0]"


def test_q6_window_table_carries_every_is_oos_field():
    """Point bloquant 1 : pas seulement l'érosion Sharpe — toutes les colonnes."""
    df = q6_window_table(_diagnostics()["indicators"])
    for col in ("Sharpe IS", "Sharpe OOS", "Return IS", "Return OOS", "PQS IS",
                "PQS OOS", "DD IS", "DD OOS", "érosion Sharpe %", "érosion Return %",
                "érosion PQS %", "amplif. DD", "dégradation"):
        assert col in df.columns, col


def test_q6_window_table_real_compute_q6_output():
    """Schéma réel de T1 : l'érosion calculée reflète `_avail`, pas `defined`."""
    q6 = compute_q6(_real_wfo_results())
    df = q6_window_table({"Q6": q6})
    assert len(df) == 2
    # fenêtre 1 : sharpe_is 0.030 > 0 -> érosion calculée = (0.030-0.012)/0.030 = 60 %
    assert bool(df.iloc[0]["érosion calculée"]) is True
    assert df.iloc[0]["érosion Sharpe %"] == "60"
    # fenêtre 2 : sharpe_is <= 0 -> non calculable, jamais 0 ni False silencieux
    assert bool(df.iloc[1]["érosion calculée"]) is False
    assert df.iloc[1]["érosion Sharpe %"] == MISSING
    # les sources IS/OOS sont bien restituées
    assert df.iloc[0]["Sharpe IS"] == "0.03"
    assert df.iloc[0]["Sharpe OOS"] == "0.012"
    assert df.iloc[0]["amplif. DD"] == "2"          # |-4| / |-2|


def test_q7_window_table_real_compute_q7_output():
    q7 = compute_q7(_real_wfo_results())   # sans séries par barre
    df = q7_window_table({"Q7": q7})
    assert len(df) == 2
    assert (df["Sharpe IS recalculé"] == MISSING).all()
    assert (df["Sharpe OOS recalculé"] == MISSING).all()
    assert df.iloc[0]["Sharpe IS rapporté"] == "0.03"


def test_q7_window_entries_audited_as_unavailable():
    """Point bloquant 3 : les Sharpe par fenêtre indisponibles doivent être audités."""
    q7 = compute_q7(_real_wfo_results())   # sans séries par barre
    df = unavailable_rows({"Q7": q7})
    refs = set(df["référence"])
    assert "Q7.windows[0].sharpe_is_recomputed" in refs
    assert "Q7.windows[0].sharpe_oos_recomputed" in refs
    assert "Q7.windows[1].sharpe_is_recomputed" in refs
    assert "Q7.final_sharpe_recomputed" in refs
    assert (df["valeur"] == MISSING).all()


# ---------------------------------------------------------------------------
# §5.3 — criteria table + rationale
# ---------------------------------------------------------------------------

def test_criteria_table_labels_and_verdicts():
    df = criteria_table(_diagnostics()["pre_verdict"])
    assert list(df.columns) == ["critère", "verdict", "raison"]
    labels = set(df["critère"])
    assert labels == {"C1 — Intégrité", "C3 — Érosion du Sharpe"}   # libellés, pas les clés brutes
    assert df.loc[df["critère"] == "C1 — Intégrité", "verdict"].iloc[0] == "🟢 GO"
    assert df.loc[df["critère"] == "C3 — Érosion du Sharpe", "verdict"].iloc[0] == "🟠 WATCH"

def test_c4_label_does_not_call_gross_returns_net():
    pre = {"criteria": {"C4_resultat_net_oos": {"verdict": "GO", "raison": None}}}
    label = criteria_table(pre)["critère"].iloc[0]
    assert "brut ou net selon convention" in label


def test_pre_verdict_rationale():
    assert pre_verdict_rationale(_diagnostics()["pre_verdict"]) == [
        "C3_erosion_sharpe: WATCH — érosion intermédiaire"
    ]
    assert pre_verdict_rationale(None) == []


# ---------------------------------------------------------------------------
# Rendering (stub container — no Streamlit)
# ---------------------------------------------------------------------------

def test_render_no_diagnostics_shows_info():
    stub = StubContainer()
    render_quant_analysis_panel(diagnostics=None, container=stub)
    assert "info" in stub.kinds()
    assert any("Aucune analyse disponible" in t for t in stub.texts())


def test_render_with_diagnostics_shows_sections():
    stub = StubContainer()
    render_quant_analysis_panel(diagnostics=_diagnostics(), container=stub)
    text = "\n".join(stub.texts())
    assert "Pré-verdict local" in text
    assert "§5.0 — Contrôle d'intégrité" in text
    assert "§5.1 — Indicateurs de robustesse" in text
    assert "§5.3 — Critères du pré-verdict" in text
    assert "Intégrité OK" in text
    assert "Q1 — Budget effectif d'évaluation" in text
    # les métriques indisponibles sont signalées, jamais affichées à 0
    assert "indisponible" in text
    assert "dénominateur ≤ 0" in text


def test_render_q6_per_window_before_aggregates():
    """§5.1 « par fenêtre puis agrégé » : l'ordre d'affichage est vérifié."""
    stub = StubContainer()
    render_quant_analysis_panel(diagnostics=_diagnostics(), container=stub)
    marks = [t for t in stub.texts() if "Q6 —" in t]
    assert len(marks) == 2
    assert "par fenêtre" in marks[0]
    assert "agrégats" in marks[1]


def test_render_integrity_ko_blocks_expert():
    stub = StubContainer()
    render_quant_analysis_panel(diagnostics=_diagnostics(integrity_ok_flag=False), container=stub)
    text = "\n".join(stub.texts())
    assert "Non évaluable" in text
    assert "Aucune analyse d'expert ne sera lancée" in text
    assert "run_present" in text            # la cause est exposée
    assert "error" in stub.kinds()


def test_render_never_displays_zero_for_missing_metric():
    """§5.3 : une métrique absente ne doit jamais s'afficher « 0 »."""
    diag = _diagnostics()
    diag["indicators"]["Q7"] = {"final_sharpe_recomputed": _avail(None, available=False, raison="absent")}
    tables = all_indicator_flat(diag["indicators"])
    missing = tables[~tables["disponible"].astype(bool)]
    assert (missing["valeur"] == MISSING).all()
    assert not (missing["valeur"] == "0").any()


# ===========================================================================
# T4 — §5.2 « Interprétation » : gate, bouton, rendu structuré
# ===========================================================================

from services.quant_expert import auto_analyze_run, build_sealed_context, substitute_citations  # noqa: E402
from ui.quant_analysis_panel import (  # noqa: E402
    EXPERT_STATUS_KINDS,
    analysis_matches_run,
    blocking_findings_table,
    expand_citations,
    expert_gate,
    expert_status_message,
    finding_badge,
    findings_table,
    limitations_list,
    recommendations_table,
    render_expert_analysis,
    render_quant_expert_panel,
    run_expert_analysis,
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
        "blocking_findings": [
            {"id": "B1", "constat": "Sharpe final non recalculable ({{Q6.sharpe_final_recomputed}}).",
             "evidence_refs": ["Q6"]},
        ],
        "limitations": ["Plafond d'évaluation : {{Q1.plafond}}."],
        "findings": [
            {"id": "F1", "niveau": "majeur", "constat": "Érosion IS → OOS marquée.",
             "evidence_refs": ["Q6", "W1"], "interpretation": "Sur-ajustement probable.",
             "condition": "si plus de 40 %"},
            {"id": "F2", "niveau": "info", "constat": "Budget correct.", "evidence_refs": ["Q1"],
             "interpretation": "Rien à signaler.", "condition": None},
        ],
        "recommandations": [
            {"action": "Réduire l'espace de recherche", "priorite": 1,
             "motif": "Budget {{Q1.budget}} trop large.", "critere_de_validation": "Q1 ≥ 200 essais"},
        ],
        "accord_avec_preverdict": False,
        "preverdict_local": "WATCH",
    }


# --- badges + tables -------------------------------------------------------

def test_finding_badge():
    assert finding_badge("majeur") == "🔴 majeur"
    assert finding_badge("mineur") == "🟠 mineur"
    assert finding_badge("info") == "🔵 info"
    assert finding_badge(None) == "None"          # jamais deviné


def test_findings_table_shape_and_rows():
    df = findings_table(_analysis())
    assert list(df.columns) == [
        "id", "niveau", "constat", "références", "interprétation", "condition", "marqueurs",
    ]
    assert len(df) == 2
    assert df.iloc[0]["niveau"] == "🔴 majeur"
    assert df.iloc[0]["références"] == "Q6, W1"
    assert df.iloc[1]["condition"] is None          # condition facultative
    assert df.iloc[1]["niveau"] == "🔵 info"
    assert (df["marqueurs"] == "").all()            # fixture propre : aucun marqueur


def test_blocking_and_recommendations_and_limitations():
    b = blocking_findings_table(_analysis())
    assert list(b.columns) == ["id", "constat", "références"]
    assert b.iloc[0]["id"] == "B1"

    r = recommendations_table(_analysis())
    assert list(r.columns) == ["action", "priorité", "motif", "critère de validation"]
    assert r.iloc[0]["priorité"] == 1

    assert len(limitations_list(_analysis())) == 1
    assert limitations_list(None) == []


# --- §5.2 gate (règles d'activation du bouton) -----------------------------

def test_gate_no_diagnostics():
    enabled, reason = expert_gate(None, peer_reachable=True)
    assert enabled is False
    assert "Aucun run chargé" in reason


def test_gate_integrity_failure_blocks():
    enabled, reason = expert_gate(_diagnostics(integrity_ok_flag=False), peer_reachable=True)
    assert enabled is False
    assert "intégrité" in reason
    assert "run_present" in reason        # la cause est exposée


def test_gate_peer_unreachable_blocks():
    enabled, reason = expert_gate(_diagnostics(), peer_reachable=False)
    assert enabled is False
    assert "agent-card" in reason


def test_gate_all_green():
    enabled, reason = expert_gate(_diagnostics(), peer_reachable=True)
    assert enabled is True
    assert reason == ""


# --- §6.1 statuts ----------------------------------------------------------

def test_status_message_severity_mapping():
    assert EXPERT_STATUS_KINDS["ok"] == "success"
    assert EXPERT_STATUS_KINDS["timeout"] == "warning"
    assert EXPERT_STATUS_KINDS["non_evaluable"] == "error"


def test_status_message_carries_reason_and_next_step():
    severity, msg = expert_status_message({
        "status": "timeout", "raison": "délai dépassé (900 s)",
        "marche_a_suivre": "Relancer pour reprendre.", "cached": True,
    })
    assert severity == "warning"
    assert "délai dépassé" in msg
    assert "Marche à suivre" in msg and "Relancer pour reprendre." in msg
    assert "cache" in msg


def test_status_message_empty_result_is_neutral():
    severity, msg = expert_status_message({})
    assert severity == "error"
    assert msg == "Aucun détail."


# --- §5.2 substitution {{ID.champ}} ---------------------------------------

def test_expand_citations_resolves_value_and_keeps_reference():
    sealed = build_sealed_context(_diagnostics())
    rendered, unresolved = expand_citations(
        "Érosion de {{Q6.median_erosion_sharpe_pct}}.", sealed
    )
    assert "12.5" in rendered                     # valeur de l'app, source de vérité
    assert "«Q6.median_erosion_sharpe_pct»" in rendered   # traçabilité conservée
    assert unresolved == []


def test_expand_citations_unresolved_left_verbatim_and_reported():
    sealed = build_sealed_context(_diagnostics())
    rendered, unresolved = expand_citations("Voir {{Q9.inexistant}}.", sealed)
    assert "{{Q9.inexistant}}" in rendered        # jamais supprimé, jamais inventé
    assert unresolved == ["Q9.inexistant"]


def test_expand_citations_uses_indisponible_policy():
    """Une valeur `_avail` indisponible reste « indisponible », jamais 0."""
    sealed = build_sealed_context(_diagnostics())
    rendered, _ = expand_citations("{{Q6.sharpe_final_recomputed}}", sealed)
    assert "indisponible" in rendered
    assert "0" not in rendered.replace("«Q6.sharpe_final_recomputed»", "")


def test_substitute_citations_is_shared_with_t2():
    """Le panneau ne duplique pas la logique de résolution §5.2."""
    sealed = build_sealed_context(_diagnostics())
    out, bad = substitute_citations("{{Q1.budget}}", sealed)
    assert "286" in out and bad == []


# --- rendu structuré -------------------------------------------------------

def test_render_expert_analysis_sections_and_accord():
    stub = StubContainer()
    unresolved = render_expert_analysis(
        _analysis(), sealed=build_sealed_context(_diagnostics()), container=stub
    )
    text = "\n".join(stub.texts())
    assert "interprétation wfo-quant" in text
    assert "🟠 WATCH" in text
    assert "exploratoire" in text and "moyenne" in text
    assert "Désaccord" in text                      # accord_avec_preverdict = False
    assert "constat(s) bloquant(s)" in text
    assert "Constats" in text and "Recommandations" in text and "Limites déclarées" in text
    assert "12.5" in text                            # valeur résolue
    assert "indisponible" in text                    # politique « indisponible » appliquée aux citations
    assert unresolved == []


def test_render_expert_analysis_reports_unresolved_citations():
    analysis = _analysis()
    analysis["verdict_justification"] = "Voir {{Q9.inconnu}}."
    stub = StubContainer()
    unresolved = render_expert_analysis(
        analysis, sealed=build_sealed_context(_diagnostics()), container=stub
    )
    assert unresolved == ["Q9.inconnu"]
    assert any("citation(s) non résolue(s)" in t for t in stub.texts())


def test_render_expert_analysis_accord_true_shows_success():
    analysis = _analysis()
    analysis["accord_avec_preverdict"] = True
    stub = StubContainer()
    render_expert_analysis(analysis, sealed=build_sealed_context(_diagnostics()), container=stub)
    assert any("Accord avec le pré-verdict" in t for t in stub.texts())
    assert "success" in stub.kinds()


# --- bouton + flux ---------------------------------------------------------

def test_panel_disabled_when_gate_fails_disables_button():
    stub = StubContainer()
    calls = []
    result = render_quant_expert_panel(
        diagnostics=_diagnostics(integrity_ok_flag=False),
        container=stub, peer_reachable=True, show_toggle=False,
        runner=lambda *a, **k: calls.append(k) or {"status": "ok"},
    )
    assert result == {}                              # rien n'a été lancé
    assert calls == []                               # aucun appel réseau
    assert any(k.get("disabled") for (_n, _a, k) in stub.calls if "disabled" in k), "bouton non désactivé"
    assert any("intégrité" in t for t in stub.texts())


def test_panel_runs_with_injected_runner():
    stub = StubContainer()
    seen = {}

    def fake_runner(diagnostics, **kwargs):
        seen.update(kwargs)
        return {"status": "ok", "analysis": _analysis(), "raison": None, "marche_a_suivre": None}

    result = render_quant_expert_panel(
        diagnostics=_diagnostics(), container=stub, peer_reachable=True,
        show_toggle=False, run_now=True, runner=fake_runner,
    )
    assert result["status"] == "ok"
    assert seen["force"] is False                    # pas de clic -> pas de bypass du cache
    assert "interprétation wfo-quant" in "\n".join(stub.texts())


def test_panel_manual_click_forces_cache_bypass():
    stub = StubContainer()
    seen = {}

    def fake_runner(diagnostics, **kwargs):
        seen.update(kwargs)
        return {"status": "ok", "analysis": _analysis()}

    # le clic sur le bouton force le bypass du cache
    stub = StubContainer()
    stub.click = True
    render_quant_expert_panel(
        diagnostics=_diagnostics(), container=stub, peer_reachable=True,
        show_toggle=False, runner=fake_runner,
    )
    assert seen["force"] is True                     # clic manuel = force=True (§5.2)


def test_panel_non_ok_status_shows_marche_a_suivre_and_no_analysis():
    stub = StubContainer()

    def fake_runner(diagnostics, **kwargs):
        return {"status": "unreachable", "analysis": None,
                "raison": "pair injoignable", "marche_a_suivre": "Démarrer wfo-quant."}

    result = render_quant_expert_panel(
        diagnostics=_diagnostics(), container=stub, peer_reachable=True,
        show_toggle=False, run_now=True, runner=fake_runner,
    )
    assert result["status"] == "unreachable"
    text = "\n".join(stub.texts())
    assert "pair injoignable" in text
    assert "Démarrer wfo-quant." in text
    assert "interprétation wfo-quant" not in text     # pas de rendu sans analyse
    assert "error" in stub.kinds()


def test_run_expert_analysis_delegates_every_kwarg():
    seen = {}

    def fake_runner(diagnostics, **kwargs):
        seen["diag"] = diagnostics
        seen.update(kwargs)
        return {"status": "ok"}

    out = run_expert_analysis(
        diagnostics=_diagnostics(), wfo_results={"a": 1}, all_trials=None,
        param_grid={"p": [1]}, cache={"k": 1}, context_id="ctx-1", force=True,
        runner=fake_runner,
    )
    assert out == {"status": "ok"}
    assert seen["diag"] is not None
    assert seen["wfo_results"] == {"a": 1}
    assert seen["param_grid"] == {"p": [1]}
    assert seen["cache"] == {"k": 1}
    assert seen["context_id"] == "ctx-1"
    assert seen["force"] is True


# --- correctifs de revue (NO_GO T4, 3 points bloquants) --------------------

def test_integrity_ko_never_shows_peer_analysis():
    """§5.3 cas a : en intégrité KO, seul le pré-verdict local NO_GO s'affiche."""
    stub = StubContainer()
    calls = []
    result = render_quant_expert_panel(
        diagnostics=_diagnostics(integrity_ok_flag=False),
        analysis=_analysis(),                     # une analyse existe pourtant
        container=stub, peer_reachable=True, show_toggle=False,
        runner=lambda *a, **k: calls.append(k) or {"status": "ok"},
    )
    assert result == {}
    assert calls == []                            # aucun appel A2A (§5.3 cas a)
    text = "\n".join(stub.texts())
    assert "🔴 NO_GO" in text and "pré-verdict local" in text
    assert "non_evaluable" in text
    assert "interprétation wfo-quant" not in text  # l'avis du pair n'est PAS rendu
    assert any(k.get("disabled") for (_n, _a, k) in stub.calls if "disabled" in k), "bouton non désactivé"


def test_analysis_from_other_run_is_not_displayed():
    """§5.2 : une analyse conservée ne s'affiche que si elle vient du run courant."""
    analysis = _analysis()
    analysis["run_id"] = "autre-run"
    stub = StubContainer()
    render_quant_expert_panel(
        diagnostics=_diagnostics(), analysis=analysis,
        container=stub, peer_reachable=True, show_toggle=False,
    )
    text = "\n".join(stub.texts())
    assert "interprétation wfo-quant" not in text
    assert "autre run" in text


def test_analysis_matches_run_semantics():
    assert analysis_matches_run(_analysis(), _diagnostics()) is True
    other = _analysis()
    other["run_id"] = "X"
    assert analysis_matches_run(other, _diagnostics()) is False
    other = _analysis()
    other["input_digest"] = "DIFFERENT"
    assert analysis_matches_run(other, _diagnostics()) is False
    assert analysis_matches_run(None, _diagnostics()) is False
    # Point bloquant de revue : un artefact SANS identifiants n'est JAMAIS accepté
    assert analysis_matches_run({"verdict": "GO"}, _diagnostics()) is False
    assert analysis_matches_run({"run_id": "run-1"}, _diagnostics()) is False
    # identifiants manquants côté run -> pas d'affichage non plus
    bare = {"manifest": {}}
    assert analysis_matches_run(_analysis(), bare) is False


def test_findings_marked_reference_invalide():
    """§5.2 : `evidence_refs` hors nomenclature → `reference_invalide`."""
    analysis = _analysis()
    analysis["findings"][0]["evidence_refs"] = ["Q6", "ZZZ"]
    df = findings_table(analysis)
    assert "reference_invalide" in df.iloc[0]["marqueurs"]
    assert df.iloc[1]["marqueurs"] == ""


def test_findings_marked_chiffre_non_reference():
    """§5.2 : un nombre nu au lieu de `{{ID.champ}}` → `chiffre_non_reference`."""
    analysis = _analysis()
    analysis["findings"][0]["constat"] = "Érosion de 42 %."       # nombre nu
    df = findings_table(analysis)
    assert "chiffre_non_reference" in df.iloc[0]["marqueurs"]


def test_render_shows_context_id_footer_and_warnings():
    """§5.2 « Rendu UI » : horodatage + context_id A2A en pied de bloc."""
    stub = StubContainer()
    render_expert_analysis(
        _analysis(), sealed=build_sealed_context(_diagnostics()),
        container=stub, context_id="ctx-abc123", warnings=["référence douteuse"],
    )
    text = "\n".join(stub.texts())
    assert "ctx-abc123" in text
    assert "Horodatage de rendu" in text
    assert "référence douteuse" in text          # avertissements du validateur exposés
    assert "pré-verdict local : 🟠 WATCH" in text # pré-verdict local à côté du verdict


def test_render_numbers_recommendations():
    """§5.2 : recommandations numérotées."""
    stub = StubContainer()
    render_expert_analysis(_analysis(), sealed=build_sealed_context(_diagnostics()), container=stub)
    frames = [a[0] for (n, a, _k) in stub.calls if n == "dataframe" and a]
    reco = [f for f in frames if "critère de validation" in list(getattr(f, "columns", []))]
    assert reco and "n°" in list(reco[0].columns)


def test_relancer_and_effacer_labels_rendered_and_wired():
    """§5.2 ligne 184 : « Relancer l'analyse » et « Effacer » sont rendus ET câblés."""
    stub = StubContainer()
    cleared = []
    render_quant_expert_panel(
        diagnostics=_diagnostics(), analysis=_analysis(),
        container=stub, peer_reachable=True, show_toggle=False,
        on_clear=lambda: cleared.append(True),
    )
    labels = [a[0] for (n, a, _k) in stub.calls if n == "button" and a]
    assert "Effacer" in labels and "Relancer l'analyse" in labels

    # clic sur « Effacer » -> callback exécuté, analyse purgée
    stub2 = StubContainer()
    stub2.click = True
    result = render_quant_expert_panel(
        diagnostics=_diagnostics(), analysis=_analysis(),
        container=stub2, peer_reachable=True, show_toggle=False,
        on_clear=lambda: cleared.append(True),
    )
    assert result["status"] == "cleared"
    assert cleared == [True]


def test_status_block_with_elapsed_timer():
    """§5.2 ligne 183 : st.status avec temps écoulé pendant l'appel."""
    stub = StubContainer()

    def fake_runner(diagnostics, **kwargs):
        return {"status": "ok", "analysis": _analysis()}

    render_quant_expert_panel(
        diagnostics=_diagnostics(), container=stub, peer_reachable=True,
        show_toggle=False, run_now=True, runner=fake_runner,
    )
    kinds = [n for (n, _a, _k) in stub.calls]
    assert "status" in kinds                       # bloc st.status présent
    text = "\n".join(stub.texts())
    assert "Terminé en" in text                    # temps écoulé affiché


def test_auto_analyze_run_disabled_is_noop():
    assert auto_analyze_run(_diagnostics(), enabled=False) == {}


def test_auto_analyze_run_delegates_with_cache_on():
    """Le hook de fin de run applique le cache (force=False) — §6.1."""
    seen = {}

    def fake_runner(diagnostics, **kwargs):
        seen.update(kwargs)
        return {"status": "ok", "analysis": _analysis()}

    out = auto_analyze_run(_diagnostics(), cache={"k": 1}, runner=fake_runner)
    assert out["status"] == "ok"
    assert seen["force"] is False
    assert seen["cache"] == {"k": 1}


def test_auto_analyze_run_forwards_evidence_to_diagnostics(monkeypatch):
    """Point bloquant de revue : config / final_trades / param_grid doivent
    atteindre `compute_quant_indicators`, sinon l'intégrité §5.0 est KO."""
    seen = {}

    def fake_compute(source, **kwargs):
        seen.update(kwargs)
        return _diagnostics()

    monkeypatch.setattr("services.quant_indicators.compute_quant_indicators", fake_compute)
    trades = pd.DataFrame({"pnl": [1.0, -0.5]})
    auto_analyze_run(
        {"window_results": []},
        final_trades=trades,
        config={"fees_pct": 0.1, "slippage_bps": 2.0},
        param_grid={"timeperiod": [10, 12]},
        runner=lambda d, **k: {"status": "ok"},
    )
    assert seen["config"] == {"fees_pct": 0.1, "slippage_bps": 2.0}   # provenance des coûts
    assert seen["param_grid"] == {"timeperiod": [10, 12]}             # références P.<param>
    assert isinstance(seen["final_trades"], pd.DataFrame)             # preuve des trades
    assert len(seen["final_trades"]) == 2


def test_reference_exists_presence_vs_value():
    """Non-régression : `P.<param>` teste la PRÉSENCE dans la grille, pas la valeur.

    Différence assumée avec `resolve_reference` (qui renvoie la valeur) :
    une entrée de grille présente avec une valeur `None` existe toujours.
    """
    from services.quant_expert import reference_exists, resolve_reference

    sealed = {"param_grid": {"foo": None}}
    assert reference_exists("P.foo", sealed) is True      # présence
    assert resolve_reference("P.foo", sealed) is None     # valeur
    assert reference_exists("P.bar", sealed) is False
    assert reference_exists("ZZZ", sealed) is False
