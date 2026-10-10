"""Helpers for export/replay metadata generation."""

from __future__ import annotations

import datetime
import json
import os
from typing import Any, Mapping

from domain.serialization import json_safe, sanitize_for_json, utc_now_iso


def build_data_source_descriptor(config_snapshot: dict | None, df) -> dict[str, Any]:
    descriptor = {
        "from_file": None,
        "file_path": None,
        "file_exists": None,
        "file_size_bytes": None,
        "file_mtime_utc": None,
        "timeframe": None,
        "requested_start_date": None,
        "requested_end_date": None,
        "loaded_rows": None,
        "loaded_start": None,
        "loaded_end": None,
    }
    if isinstance(config_snapshot, dict):
        file_path = config_snapshot.get("file_path")
        from_file = bool(config_snapshot.get("from_file", False))
        descriptor["from_file"] = from_file
        descriptor["file_path"] = file_path
        descriptor["timeframe"] = config_snapshot.get("timeframe")
        descriptor["requested_start_date"] = config_snapshot.get("start_date")
        descriptor["requested_end_date"] = config_snapshot.get("end_date")
        if from_file and isinstance(file_path, str):
            exists = os.path.exists(file_path)
            descriptor["file_exists"] = exists
            if exists:
                try:
                    descriptor["file_size_bytes"] = int(os.path.getsize(file_path))
                except Exception:
                    descriptor["file_size_bytes"] = None
                try:
                    descriptor["file_mtime_utc"] = datetime.datetime.fromtimestamp(
                        os.path.getmtime(file_path),
                        tz=datetime.timezone.utc,
                    ).isoformat()
                except Exception:
                    descriptor["file_mtime_utc"] = None
    if df is not None and hasattr(df, "empty") and not df.empty:
        try:
            descriptor["loaded_rows"] = int(len(df))
            descriptor["loaded_start"] = json_safe(df.index[0])
            descriptor["loaded_end"] = json_safe(df.index[-1])
        except Exception:
            pass
    return sanitize_for_json(descriptor)


def build_replay_manifest(
    payload: dict,
    snapshot_mode_requested: str,
    snapshot_mode_actual: str,
    has_df_snapshot: bool,
    data_source: dict,
    run_meta: dict | None = None,
    final_params: dict | None = None,
    final_start_date: str | None = None,
    final_end_date: str | None = None,
    final_file_path: str | None = None,
    v3_artifacts: dict | None = None,
) -> dict[str, Any]:
    run_meta = run_meta or {}
    v3_artifacts = v3_artifacts or {}
    manifest = {
        "created_at_utc": utc_now_iso(),
        "package_type": "full_replay_and_stats" if has_df_snapshot else "stats_and_manifest",
        "data_snapshot_mode_requested": snapshot_mode_requested,
        "data_snapshot_mode_actual": snapshot_mode_actual,
        "contains_df_snapshot": bool(has_df_snapshot),
        "replay_readiness": "strict" if bool(has_df_snapshot) else "reference_only",
        "run_id": run_meta.get("run_id"),
        "status": run_meta.get("status"),
        "traceability": payload.get("traceability"),
        "config": payload.get("config"),
        "data_source": data_source,
        "final_backtest": {
            "has_final_portfolio": bool(payload.get("has_final_portfolio")),
            "final_params": final_params,
            "final_start_date": final_start_date,
            "final_end_date": final_end_date,
            "final_file_path": final_file_path,
        },
        "v3_artifacts": v3_artifacts,
    }
    return sanitize_for_json(manifest)


# ---------------------------------------------------------------------------
# §5.2 artefact export (T5) — quant_analysis.v1 canonical JSON + derived Markdown
# ---------------------------------------------------------------------------

def quant_analysis_json(analysis: Mapping[str, Any] | None) -> str:
    """Canonical ``quant_analysis.v1`` JSON artefact (§5.2).

    The JSON is the **canonical** export; the Markdown below is derived from it.
    ``allow_nan=False`` rejects ``NaN``/``Infinity`` instead of emitting them
    (§5.2 schema rules), and keys are sorted so the artefact is byte-stable for
    a given analysis.
    """
    payload = analysis if isinstance(analysis, Mapping) else {}
    return json.dumps(
        sanitize_for_json(payload),
        indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True, default=str,
    )


def _manifest_matches_run(
    manifest: Mapping[str, Any] | None,
    wfo_results: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None = None,
) -> bool:
    """Does a restored manifest belong to the imported run? (§6.4 replay)

    The original evidence digest must be present, and the replay digest must
    match the imported run and config. Legacy archives lacking that fingerprint
    cannot be verified and are rejected.
    """
    manifest = manifest if isinstance(manifest, Mapping) else {}
    wfo = wfo_results if isinstance(wfo_results, Mapping) else {}
    run_id = wfo.get("run_id")
    if run_id is not None and manifest.get("run_id") != run_id:
        return False
    from services.quant_indicators import compute_replay_input_digest

    try:
        expected = compute_replay_input_digest(wfo_results, config)
    except Exception:  # noqa: BLE001 — fail closed on unreadable replay inputs
        return False
    return bool(
        manifest.get("input_digest") and expected
        and manifest.get("input_digest_replay") == expected
    )


def restore_quant_artifacts(
    artifacts: Mapping[str, Any] | None,
    wfo_results=None,
    *,
    config: Mapping[str, Any] | None = None,
) -> dict:
    """§6.4 — replay: accept only the quant artefacts that belong to this run.

    ``artifacts`` maps file names to decoded JSON payloads
    (``quant_analysis.json``, ``quant_indicators.json``, ``run_manifest.json``).
    Returns ``{"analysis": … | None, "diagnostics": … | None}``; a foreign or
    unverifiable artefact is **dropped**, never restored.

    The stored ``integrity`` is treated as a **claim**, not as a fact: it is
    recomputed from the imported run with ``check_run_integrity``.  Restoring a
    stored ``ok: True`` would let a tampered artefact certify itself.
    """
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    block = artifacts.get("quant_indicators.json")
    block = block if isinstance(block, Mapping) else {}
    manifest = artifacts.get("run_manifest.json")
    analysis = artifacts.get("quant_analysis.json")

    diagnostics = None
    if isinstance(manifest, Mapping) or block:
        if _manifest_matches_run(manifest, wfo_results, config):
            diagnostics = {
                "manifest": manifest if isinstance(manifest, Mapping) else {},
                "indicators": block.get("indicators") or {},
                "integrity": _recompute_integrity(wfo_results, config),
                "pre_verdict": block.get("pre_verdict") or {},
                "replay_config": dict(config) if isinstance(config, Mapping) else {},
            }
            # §5.3 cas a : un contrôle recalculé en échec impose le pré-verdict
            # local `NO_GO`.  Conserver un pré-verdict restauré à `WATCH` à côté
            # d'une intégrité KO serait contradictoire.
            if not diagnostics["integrity"].get("ok"):
                scope = (block.get("pre_verdict") or {}).get("scope", "exploratoire")
                diagnostics["pre_verdict"] = {
                    "verdict": "NO_GO", "scope": scope, "status": "non_evaluable",
                    "criteria": {},
                    "rationale": ["replay : contrôle d'intégrité recalculé en échec"],
                }

    if isinstance(analysis, Mapping) and analysis:
        try:
            from ui.quant_analysis_panel import analysis_matches_run

            if not analysis_matches_run(analysis, diagnostics):
                analysis = None
        except Exception:  # noqa: BLE001 — never raise on a corrupt artefact
            analysis = None
    else:
        analysis = None

    return {"analysis": analysis, "diagnostics": diagnostics}


def _recompute_integrity(wfo_results, config) -> dict:
    """Integrity of the imported run, recomputed — never trusted from the file.

    The ZIP carries the facts, **not** the raw evidence (``final_trades`` and the
    per-bar returns are not exported), so a replay cannot re-establish a full
    §5.0 verdict.  The artefacts are therefore **archives, not certificates**:
    the control is recomputed from what is available and honestly reports what
    is missing instead of certifying the file's own claim.
    """
    try:
        from services.quant_indicators import check_run_integrity

        return check_run_integrity(
            wfo_results if isinstance(wfo_results, Mapping) else {},
            config=config if isinstance(config, Mapping) else None,
        )
    except Exception as exc:  # noqa: BLE001 — an unreadable run is not "ok"
        return {"ok": False, "status": "non_evaluable",
                "causes": [f"contrôle d'intégrité non recalculable ({exc})"], "checks": {}}


def apply_quant_restore(state, restored) -> None:
    """Replace the session's quant state with what the import restored (§6.4).

    The previous state is **always** cleared first: importing a ZIP without quant
    artefacts (or with rejected ones) must never leave the previous run's facts
    around.  Accepts any mutable mapping (``st.session_state`` is a
    ``SessionStateProxy``, not a ``dict``).
    """
    if state is None or not (hasattr(state, "pop") and hasattr(state, "__setitem__")):
        return
    for key in ("quant_analysis", "quant_diagnostics", "quant_diag_key"):
        try:
            state.pop(key, None)
        except Exception:  # noqa: BLE001 — never break the import on a state quirk
            pass
    restored = restored if isinstance(restored, Mapping) else {}
    if restored.get("diagnostics") is not None:
        state["quant_diagnostics"] = {**restored["diagnostics"], "source": "archived"}
    if restored.get("analysis") is not None:
        state["quant_analysis"] = {"status": "ok", "analysis": restored["analysis"]}


def quant_artifacts_for_run(
    analysis: Mapping[str, Any] | None,
    diagnostics: Mapping[str, Any] | None,
    *,
    sealed: Mapping[str, Any] | None = None,
) -> tuple[str, str] | None:
    """``(json, md)`` artefacts to write in the ZIP, or ``None`` to write none.

    An analysis that does not belong to the current run is **never** exported —
    the same guard as the UI (``analysis_matches_run``).
    """
    if not isinstance(analysis, Mapping) or not analysis:
        return None
    try:
        from ui.quant_analysis_panel import analysis_matches_run

        if not analysis_matches_run(analysis, diagnostics):
            return None
    except Exception:  # noqa: BLE001 — export must never raise
        return None
    return quant_analysis_json(analysis), quant_analysis_markdown(analysis, sealed=sealed)


def _md_cite(text: Any, sealed: Mapping[str, Any] | None) -> str:
    """Resolve ``{{ID.champ}}`` for the derived export (traceable numbers)."""
    if not isinstance(text, str):
        return "" if text is None else str(text)
    try:
        from services.quant_expert import substitute_citations

        rendered, _ = substitute_citations(text, sealed or {})
        return rendered
    except Exception:  # noqa: BLE001 — export must never raise on a citation
        return text


def quant_analysis_markdown(
    analysis: Mapping[str, Any] | None,
    *,
    sealed: Mapping[str, Any] | None = None,
) -> str:
    """Derived, human-readable export of ``quant_analysis.v1`` (§5.2).

    Numbers are written as their **resolved** value with the reference kept
    (``valeur «ID.champ»``), so the export stays traceable to the sealed facts —
    the same substitution policy as the UI.
    """
    if not isinstance(analysis, Mapping) or not analysis:
        return "# Analyse quant\n\nAucune analyse disponible.\n"

    def cite(value):
        return _md_cite(value, sealed)

    def refs(item):
        return ", ".join(str(r) for r in (item.get("evidence_refs") or [])) or "—"

    lines = [
        "# Analyse quant — interprétation wfo-quant",
        "",
        f"- **Verdict** : {analysis.get('verdict')}",
        f"- **Portée** : `{analysis.get('verdict_scope')}`",
        f"- **Confiance** : `{analysis.get('confidence')}`",
        f"- **Pré-verdict local** : {analysis.get('preverdict_local')}",
        f"- **Accord avec le pré-verdict** : {analysis.get('accord_avec_preverdict')}",
        f"- **Run** : `{analysis.get('run_id')}` · digest `{analysis.get('input_digest')}`",
        f"- **Généré le** : `{analysis.get('generated_at')}` · indicateurs `{analysis.get('indicator_version')}`",
        "",
        "## Justification",
        "",
        cite(analysis.get("verdict_justification")),
        "",
    ]

    blocking = analysis.get("blocking_findings") or []
    if blocking:
        lines += ["## Constats bloquants", ""]
        for item in blocking:
            if not isinstance(item, Mapping):
                continue
            lines.append(f"- **{item.get('id')}** — {cite(item.get('constat'))} _(réf. {refs(item)})_")
        lines.append("")

    findings = analysis.get("findings") or []
    if findings:
        lines += ["## Constats", ""]
        for item in findings:
            if not isinstance(item, Mapping):
                continue
            condition = item.get("condition")
            lines.append(
                f"- **{item.get('id')}** [{item.get('niveau')}] — {cite(item.get('constat'))} "
                f"_(réf. {refs(item)})_"
            )
            lines.append(f"  - Interprétation : {cite(item.get('interpretation'))}")
            if condition:
                lines.append(f"  - Condition : {cite(condition)}")
        lines.append("")

    recos = analysis.get("recommandations") or []
    if recos:
        lines += ["## Recommandations", ""]
        for i, item in enumerate(recos, start=1):
            if not isinstance(item, Mapping):
                continue
            lines.append(f"{i}. **{cite(item.get('action'))}** (priorité {item.get('priorite')})")
            lines.append(f"   - Motif : {cite(item.get('motif'))}")
            lines.append(f"   - Critère de validation : {cite(item.get('critere_de_validation'))}")
        lines.append("")

    limits = analysis.get("limitations") or []
    if limits:
        lines += ["## Limites déclarées", ""]
        lines += [f"- {cite(x)}" for x in limits]
        lines.append("")

    lines += [
        "---",
        "",
        "_Artefact dérivé de `quant_analysis.json` (canonique). "
        "Les citations `{{ID.champ}}` sont résolues contre les faits scellés de l'application._",
        "",
    ]
    return "\n".join(lines)
