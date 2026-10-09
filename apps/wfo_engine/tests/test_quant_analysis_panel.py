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

    def __getattr__(self, name):
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
