"""« Analyse quant » panel — local calculations and pre-verdict display (T3).

Renders §5.0 (run manifest + blocking integrity check) and §5.1 (the Q1–Q8
indicators, the OOS-uncertainty block and the deterministic local pre-verdict)
of the cahier des charges ``docs/cahier_charges_analyse_quant.md``.

Design rules enforced here:

- the panel is a **read-only view** of :func:`services.quant_indicators
  .compute_quant_indicators` output — it recomputes nothing and never invents a
  number ;
- every displayed value is **sourced** (``référence`` column traces the exact
  field, including per-window entries such as ``Q6.windows[0].erosion_sharpe_pct``)
  so the reader can verify it against the sealed facts (§5.1) ;
- a per-window table is shown **before** the aggregates of the same block (§5.1:
  « par fenêtre puis agrégé ») so a per-window collapse is never hidden by a mean ;
- a non-computable metric is shown as **« indisponible »** with its reason —
  never as ``0`` (§5.3) ;
- availability is read from the ``_avail`` leaf (``available``), never
  recalculated here ;
- a failed integrity check shows the « non évaluable + cause » status and must
  hide/short-circuit any expert call (§5.0 / §5.3 case a) — the button itself
  belongs to T4.

The display helpers are pure (pandas only) and unit-testable without Streamlit;
only :func:`render_quant_analysis_panel` touches the ``streamlit`` module, and
it imports it lazily.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Labels (UI in French)
# ---------------------------------------------------------------------------

INDICATOR_TITLES = {
    "Q1": "Q1 — Budget effectif d'évaluation",
    "Q2": "Q2 — Diversité des évaluations",
    "Q3": "Q3 — Forme de la distribution des scores",
    "Q4": "Q4 — Stabilité des paramètres",
    "Q5": "Q5 — Voisinage du gagnant",
    "Q6": "Q6 — Érosion IS → OOS",
    "Q7": "Q7 — Sharpe non-annualisé homogène",
    "Q8": "Q8 — Sensibilité aux coûts",
    "OOS_UNCERTAINTY": "Incertitude OOS",
}

VERDICT_BADGES = {
    "GO": "🟢 GO",
    "WATCH": "🟠 WATCH",
    "NO_GO": "🔴 NO_GO",
}

CRITERION_LABELS = {
    "C1_integrite": "C1 — Intégrité",
    "C2_echantillon_oos": "C2 — Échantillon OOS",
    "C3_erosion_sharpe": "C3 — Érosion du Sharpe",
    "C4_resultat_net_oos": "C4 — Résultat net OOS + coûts",
    "C5_robustesse_selection": "C5 — Robustesse de sélection",
}

MISSING = "indisponible"

# Blocks whose per-window rows are rendered as a wide table BEFORE the aggregates.
WINDOW_BLOCKS = ("Q6", "Q7")


# ---------------------------------------------------------------------------
# Formatting (pure)
# ---------------------------------------------------------------------------

def verdict_badge(verdict: Any) -> str:
    """🟢/🟠/🔴 badge; an unknown verdict is shown verbatim (never guessed)."""
    return VERDICT_BADGES.get(verdict, str(verdict))


def fmt_metric(entry: Any) -> str:
    """Human-readable value of an ``_avail``-shaped entry.

    A non-available metric renders as ``« indisponible »`` — **never ``0``** (§5.3).
    """
    if _is_avail_leaf(entry):
        if not entry.get("available"):
            return MISSING
        value = entry.get("value")
        return "—" if value is None else _fmt_scalar(value)
    return _fmt_scalar(entry)


def _cell(value: Any) -> str:
    """Render either an ``_avail`` leaf or a plain scalar."""
    return fmt_metric(value) if _is_avail_leaf(value) else _fmt_scalar(value)


def _fmt_scalar(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "oui" if value else "non"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _is_avail_leaf(obj: Any) -> bool:
    return isinstance(obj, Mapping) and "value" in obj and "available" in obj


def _mapping(value: Any) -> Mapping[str, Any]:
    """``value`` if it is a mapping, else an empty one (never ``None``)."""
    return value if isinstance(value, Mapping) else {}


def _is_available(entry: Any) -> Optional[bool]:
    """Availability of an ``_avail`` leaf; ``None`` when not ``_avail``-shaped."""
    return bool(entry.get("available")) if _is_avail_leaf(entry) else None


# ---------------------------------------------------------------------------
# Flattening (pure, sourced)
# ---------------------------------------------------------------------------

def _flatten(prefix: str, obj: Any, rows: list) -> None:
    """Flatten nested indicator structures into sourced ``(ref, value)`` rows.

    A **list of mappings** (per-window entries) is expanded with an indexed
    reference (``Q7.windows[1].sharpe_is_recomputed``) so that per-window values
    stay visible and auditable — they are never collapsed to a bare count (§5.1).
    """
    if _is_avail_leaf(obj):
        rows.append({
            "référence": prefix,
            "valeur": fmt_metric(obj),
            "disponible": _is_available(obj),
            "raison": obj.get("raison"),
        })
        return
    if isinstance(obj, Mapping):
        if not obj:
            rows.append({"référence": prefix, "valeur": "—", "disponible": True, "raison": None})
            return
        for key, value in obj.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), value, rows)
        return
    if isinstance(obj, (list, tuple)):
        if obj and all(isinstance(item, Mapping) for item in obj):
            for i, item in enumerate(obj):
                _flatten(f"{prefix}[{i}]", item, rows)
            return
        rows.append({
            "référence": prefix,
            "valeur": f"[{len(obj)} élément(s)]",
            "disponible": True,
            "raison": None,
        })
        return
    rows.append({"référence": prefix, "valeur": _fmt_scalar(obj), "disponible": True, "raison": None})


# ---------------------------------------------------------------------------
# §5.0 — manifest + integrity
# ---------------------------------------------------------------------------

def manifest_table(manifest: Any) -> pd.DataFrame:
    """Flat ``champ / valeur`` view of the run manifest (§5.0)."""
    manifest = _mapping(manifest)
    rows = [{"champ": str(k), "valeur": _fmt_scalar(v)} for k, v in manifest.items()]
    return pd.DataFrame(rows, columns=["champ", "valeur"])


def integrity_table(integrity: Any) -> pd.DataFrame:
    """One row per integrity check (§5.0 blocking check)."""
    integrity = _mapping(integrity)
    checks = _mapping(integrity.get("checks"))
    rows = []
    for name, detail in checks.items():
        if isinstance(detail, Mapping):
            rows.append({
                "contrôle": str(name),
                "statut": "OK" if detail.get("ok") else "ÉCHEC",
                "détail": detail.get("raison"),
            })
        else:
            rows.append({"contrôle": str(name), "statut": _fmt_scalar(detail), "détail": None})
    return pd.DataFrame(rows, columns=["contrôle", "statut", "détail"])


def integrity_ok(integrity: Any) -> bool:
    return bool(_mapping(integrity).get("ok"))


def integrity_causes(integrity: Any) -> list:
    integrity = _mapping(integrity)
    return [str(c) for c in (integrity.get("causes") or [])]


# ---------------------------------------------------------------------------
# §5.1 — indicators
# ---------------------------------------------------------------------------

def indicator_table(indicators: Any, bloc: str) -> pd.DataFrame:
    """Full flattened, sourced view of one block (aggregates **and** per-window).

    An absent/empty block yields an **empty** frame (nothing to source), so the
    caller can distinguish « not computed » from a computed block.
    """
    indicators = _mapping(indicators)
    rows: list = []
    body = indicators.get(bloc)
    if isinstance(body, Mapping) and body:
        _flatten(bloc, body, rows)
    return pd.DataFrame(rows, columns=["référence", "valeur", "disponible", "raison"])


def aggregate_table(indicators: Any, bloc: str) -> pd.DataFrame:
    """Block view **without** the per-window rows (those get their own table)."""
    table = indicator_table(indicators, bloc)
    if table.empty:
        return table
    prefix = f"{bloc}.windows["
    return table[~table["référence"].str.startswith(prefix, na=False)].reset_index(drop=True)


def all_indicator_tables(indicators: Any) -> dict:
    """``{bloc: aggregate-DataFrame}`` for every block actually present (§5.1)."""
    indicators = _mapping(indicators)
    out = {}
    for bloc in INDICATOR_TITLES:
        table = aggregate_table(indicators, bloc)
        if not table.empty:
            out[bloc] = table
    return out


def all_indicator_flat(indicators: Any) -> pd.DataFrame:
    """Full flat view of every present block — per-window entries included.

    This is the source of the « indisponible » audit so a missing per-window
    metric is never invisible (§5.1).
    """
    frames = [t for t in (indicator_table(_mapping(indicators), b) for b in INDICATOR_TITLES) if not t.empty]
    if not frames:
        return pd.DataFrame(columns=["référence", "valeur", "disponible", "raison"])
    return pd.concat(frames, ignore_index=True)


def unavailable_rows(indicators: Any) -> pd.DataFrame:
    """Every non-computable metric across all blocks (« indisponible » audit)."""
    merged = all_indicator_flat(indicators)
    if merged.empty:
        return merged
    return merged[~merged["disponible"].fillna(False).astype(bool)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# §5.1 — per-window wide tables (shown BEFORE the aggregates)
# ---------------------------------------------------------------------------

def _window_entries(indicators: Any, bloc: str) -> list:
    body = _mapping(_mapping(indicators).get(bloc))
    return [w for w in (body.get("windows") or []) if isinstance(w, Mapping)]


def q6_window_table(indicators: Any) -> pd.DataFrame:
    """Complete Q6 erosion **per window** (§5.1: par fenêtre puis agrégé).

    Carries every per-window field produced by ``compute_q6``: the IS/OOS
    sources, the three erosions, the drawdown amplification and the degradation
    flag.  Availability comes from the ``_avail`` leaf — never recomputed.
    """
    rows = []
    for i, w in enumerate(_window_entries(indicators, "Q6")):
        rows.append({
            "référence": f"Q6.windows[{i}]",
            "fenêtre": w.get("window"),
            "Sharpe IS": _cell(w.get("sharpe_is")),
            "Sharpe OOS": _cell(w.get("sharpe_oos")),
            "Return IS": _cell(w.get("return_is")),
            "Return OOS": _cell(w.get("return_oos")),
            "PQS IS": _cell(w.get("pqs_is")),
            "PQS OOS": _cell(w.get("pqs_oos")),
            "DD IS": _cell(w.get("dd_is")),
            "DD OOS": _cell(w.get("dd_oos")),
            "érosion Sharpe %": fmt_metric(w.get("erosion_sharpe_pct")),
            "érosion Return %": fmt_metric(w.get("erosion_return_pct")),
            "érosion PQS %": fmt_metric(w.get("erosion_pqs_pct")),
            "amplif. DD": fmt_metric(w.get("dd_amplification_ratio")),
            "érosion calculée": _is_available(w.get("erosion_sharpe_pct")),
            "dégradation": _cell(w.get("degradation")),
        })
    columns = [
        "référence", "fenêtre", "Sharpe IS", "Sharpe OOS", "Return IS", "Return OOS",
        "PQS IS", "PQS OOS", "DD IS", "DD OOS",
        "érosion Sharpe %", "érosion Return %", "érosion PQS %", "amplif. DD",
        "érosion calculée", "dégradation",
    ]
    return pd.DataFrame(rows, columns=columns)


def q7_window_table(indicators: Any) -> pd.DataFrame:
    """Q7 per window: recomputed (from per-bar returns) vs reported Sharpe."""
    rows = []
    for i, w in enumerate(_window_entries(indicators, "Q7")):
        rows.append({
            "référence": f"Q7.windows[{i}]",
            "fenêtre": w.get("window"),
            "Sharpe IS recalculé": fmt_metric(w.get("sharpe_is_recomputed")),
            "Sharpe OOS recalculé": fmt_metric(w.get("sharpe_oos_recomputed")),
            "Sharpe IS rapporté": _cell(w.get("sharpe_is_reported")),
            "Sharpe OOS rapporté": _cell(w.get("sharpe_oos_reported")),
        })
    columns = [
        "référence", "fenêtre", "Sharpe IS recalculé", "Sharpe OOS recalculé",
        "Sharpe IS rapporté", "Sharpe OOS rapporté",
    ]
    return pd.DataFrame(rows, columns=columns)


_WINDOW_TABLES = {"Q6": q6_window_table, "Q7": q7_window_table}


# ---------------------------------------------------------------------------
# §5.3 — pre-verdict
# ---------------------------------------------------------------------------

def criteria_table(pre_verdict: Any) -> pd.DataFrame:
    """One row per aggregation criterion (C1…C5) with its verdict and reason."""
    pre_verdict = _mapping(pre_verdict)
    criteria = _mapping(pre_verdict.get("criteria"))
    rows = []
    for key, detail in criteria.items():
        detail = _mapping(detail)
        rows.append({
            "critère": CRITERION_LABELS.get(key, key),
            "verdict": verdict_badge(detail.get("verdict")),
            "raison": detail.get("raison"),
        })
    return pd.DataFrame(rows, columns=["critère", "verdict", "raison"])


def pre_verdict_rationale(pre_verdict: Any) -> list:
    pre_verdict = _mapping(pre_verdict)
    return [str(r) for r in (pre_verdict.get("rationale") or [])]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_quant_analysis_panel(
    *,
    diagnostics: Optional[Mapping[str, Any]] = None,
    show_manifest: bool = True,
    container=None,
) -> None:
    """Render the « Analyse quant » calculations block (§5.0 + §5.1 + §5.3).

    ``diagnostics`` is the dict returned by
    :func:`services.quant_indicators.compute_quant_indicators`.  When it is
    ``None`` the panel explains what is missing instead of rendering empty
    tables — the button (T4) stays disabled until a run is loaded.
    """
    import streamlit as st  # local import: helpers above stay Streamlit-free

    target = container if container is not None else st

    if not isinstance(diagnostics, Mapping):
        target.info(
            "Aucune analyse disponible : lancez un run WFO (ou chargez des résultats) "
            "pour afficher le manifeste, les indicateurs Q1–Q8 et le pré-verdict."
        )
        return

    manifest = _mapping(diagnostics.get("manifest"))
    integrity = _mapping(diagnostics.get("integrity"))
    indicators = _mapping(diagnostics.get("indicators"))
    pre_verdict = _mapping(diagnostics.get("pre_verdict"))

    # --- §5.3 pre-verdict banner -----------------------------------------
    verdict = pre_verdict.get("verdict")
    target.markdown(f"### Pré-verdict local — {verdict_badge(verdict)}")
    target.caption(
        f"Périmètre : `{pre_verdict.get('scope')}` · statut : `{pre_verdict.get('status')}` · "
        f"calculé localement par `services/quant_indicators.py` (déterministe)."
    )

    # --- §5.0 integrity (blocking) ---------------------------------------
    ok = integrity_ok(integrity)
    target.markdown("#### §5.0 — Contrôle d'intégrité")
    if ok:
        target.success("Intégrité OK — le run est exploitable.")
    else:
        causes = integrity_causes(integrity)
        target.error(
            "**Non évaluable** — contrôle d'intégrité en échec : "
            + ("; ".join(causes) if causes else "cause inconnue")
            + ". Aucune analyse d'expert ne sera lancée."
        )
    table = integrity_table(integrity)
    if not table.empty:
        target.dataframe(_arrow_safe(table), use_container_width=True)
    else:
        target.caption("Aucun contrôle détaillé disponible.")

    if show_manifest:
        with target.expander("Manifeste des données (§5.0)", expanded=False):
            target.dataframe(_arrow_safe(manifest_table(manifest)), use_container_width=True)

    # --- §5.1 indicators --------------------------------------------------
    target.markdown("#### §5.1 — Indicateurs de robustesse (calcul local)")
    tables = all_indicator_tables(indicators)
    if not tables and not any(_window_entries(indicators, b) for b in WINDOW_BLOCKS):
        target.warning("Aucun indicateur calculé (run incomplet).")
    else:
        for bloc in INDICATOR_TITLES:
            # per-window FIRST (§5.1 « par fenêtre puis agrégé »), then aggregates
            window_factory = _WINDOW_TABLES.get(bloc)
            if window_factory is not None:
                wtable = window_factory(indicators)
                if not wtable.empty:
                    target.markdown(f"**{INDICATOR_TITLES.get(bloc, bloc)} — par fenêtre**")
                    target.dataframe(_arrow_safe(wtable), use_container_width=True)
            agg = tables.get(bloc)
            if agg is not None and not agg.empty:
                target.markdown(f"**{INDICATOR_TITLES.get(bloc, bloc)} — agrégats**")
                target.dataframe(_arrow_safe(agg), use_container_width=True)

        missing = unavailable_rows(indicators)
        if not missing.empty:
            target.warning(
                f"**{len(missing)} métrique(s) indisponible(s)** — affichée(s) "
                f"« {MISSING} » (jamais 0), voir la colonne `raison` :"
            )
            target.dataframe(_arrow_safe(missing), use_container_width=True)

    # --- §5.3 criteria table + rationale ---------------------------------
    target.markdown("#### §5.3 — Critères du pré-verdict")
    crit = criteria_table(pre_verdict)
    if not crit.empty:
        target.dataframe(_arrow_safe(crit), use_container_width=True)
    for line in pre_verdict_rationale(pre_verdict):
        target.markdown(f"- {line}")


def _arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Local fallback for the Arrow-safe conversion (T3 keeps its own copy).

    ``ui.data_utils.arrow_safe_df`` is reused when importable so the panel stays
    consistent with the rest of the application.
    """
    try:
        from ui.data_utils import arrow_safe_df

        return arrow_safe_df(df)
    except Exception:  # noqa: BLE001 — display must never raise
        return df
