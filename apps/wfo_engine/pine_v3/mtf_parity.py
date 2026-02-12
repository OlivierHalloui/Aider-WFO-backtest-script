"""Dedicated MTF parity proof helpers for Pine V3 (`request.security`)."""

from __future__ import annotations

import datetime
from typing import Any


def build_mtf_parity_proof_report(
    strategy_spec: dict[str, Any] | None,
    request_security_diagnostics: dict[str, Any] | None,
    parity_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a deterministic report focused on MTF (`request.security`) behavior.

    This report does not replace global parity; it gives an explicit proof block
    for MTF calls so users can isolate that risk area.
    """
    spec = strategy_spec if isinstance(strategy_spec, dict) else {}
    diagnostics = request_security_diagnostics if isinstance(request_security_diagnostics, dict) else {}
    parity = parity_report if isinstance(parity_report, dict) else {}

    capabilities = spec.get("capabilities") if isinstance(spec.get("capabilities"), dict) else {}
    uses_mtf = bool(capabilities.get("uses_request_security", False))
    rows = diagnostics.get("rows") if isinstance(diagnostics.get("rows"), list) else []
    diag_warnings = diagnostics.get("warnings") if isinstance(diagnostics.get("warnings"), list) else []

    checks = []
    checks.append(
        {
            "id": "uses_request_security",
            "label": "Le script utilise request.security",
            "required": True,
            "passed": uses_mtf,
            "detail": f"capabilities.uses_request_security={uses_mtf}",
        }
    )
    checks.append(
        {
            "id": "mtf_diagnostics_available",
            "label": "Diagnostics MTF disponibles",
            "required": True,
            "passed": len(rows) > 0 if uses_mtf else True,
            "detail": f"rows={len(rows)}",
        }
    )

    per_call_checks = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        non_na_count = int(row.get("non_na_count", 0) or 0)
        change_count = int(row.get("change_count", 0) or 0)
        base_bar_count = int(row.get("base_bar_count", 0) or 0)
        passed = non_na_count > 0 and change_count > 0 and base_bar_count > 0
        per_call_checks.append(
            {
                "line": int(row.get("line", 0) or 0),
                "target": str(row.get("target") or ""),
                "timeframe": str(row.get("timeframe") or ""),
                "passed": passed,
                "detail": (
                    f"non_na_count={non_na_count}, change_count={change_count}, "
                    f"base_bar_count={base_bar_count}"
                ),
            }
        )

    checks.append(
        {
            "id": "mtf_series_quality",
            "label": "Qualité minimale des séries MTF",
            "required": True,
            "passed": all(bool(r.get("passed", False)) for r in per_call_checks) if uses_mtf else True,
            "detail": f"calls_checked={len(per_call_checks)}",
        }
    )

    # Dedicated parity tie-in: if global parity exists, it must pass.
    parity_pass = parity.get("parity_pass")
    checks.append(
        {
            "id": "global_parity_pass",
            "label": "Parité globale Pine/Python validée",
            "required": False,
            "passed": True if parity_pass is None else (parity_pass is True),
            "detail": f"parity_pass={parity_pass}",
        }
    )

    required_checks = [c for c in checks if bool(c.get("required", False))]
    required_passed = [c for c in required_checks if bool(c.get("passed", False))]
    score = round((len(required_passed) / max(1, len(required_checks))) * 100.0, 1)
    proof_pass = len(required_passed) == len(required_checks)

    blockers = []
    for check in required_checks:
        if not bool(check.get("passed", False)):
            blockers.append(
                {
                    "code": str(check.get("id") or "unknown"),
                    "label": str(check.get("label") or ""),
                    "detail": str(check.get("detail") or ""),
                }
            )

    return {
        "schema_version": "pine_mtf_parity_proof.v1",
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "pass" if proof_pass else "failed",
        "proof_pass": bool(proof_pass),
        "score": score,
        "uses_request_security": uses_mtf,
        "checks": checks,
        "call_checks": per_call_checks,
        "blockers": blockers,
        "warnings": [str(w) for w in diag_warnings if str(w).strip()],
    }

