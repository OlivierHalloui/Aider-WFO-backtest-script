"""Execution gate helpers for Pine V3 beta runs."""

from __future__ import annotations

import datetime


def build_execution_gate_report(
    strategy_mode: str | None,
    beta_readiness_report: dict | None = None,
    parity_reference_payload: dict | None = None,
    parity_reference_validation: dict | None = None,
    parity_report: dict | None = None,
    enforce_parity_when_reference: bool = True,
) -> dict:
    """Return a deterministic run gate for Pine V3 execution.

    Rules:
    - Native mode is always pass-through.
    - `pine_imported` requires `beta_ready=true`.
    - If a parity reference is provided and enforcement is enabled, then:
      `parity_reference_validation.valid=true` and `parity_report.parity_pass=true`
      are required.
    """
    mode = str(strategy_mode or "").strip().lower()
    beta = beta_readiness_report if isinstance(beta_readiness_report, dict) else {}
    parity_ref_payload = parity_reference_payload if isinstance(parity_reference_payload, dict) else {}
    parity_ref_validation = (
        parity_reference_validation if isinstance(parity_reference_validation, dict) else {}
    )
    parity = parity_report if isinstance(parity_report, dict) else {}

    checks: list[dict] = []
    blockers: list[dict] = []
    warnings: list[str] = []

    if mode != "pine_imported":
        return {
            "schema_version": "pine_execution_gate.v1",
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "strategy_mode": mode or "native_atdmf",
            "status": "not_applicable",
            "can_run": True,
            "checks": [],
            "blockers": [],
            "warnings": [],
        }

    beta_ready = bool(beta.get("beta_ready", False))
    beta_available = bool(beta)
    checks.append(
        {
            "id": "beta_ready",
            "label": "V3 Beta readiness",
            "required": True,
            "passed": beta_ready,
            "detail": (
                f"available={beta_available}, score={beta.get('readiness_score')}, status={beta.get('status')}"
            ),
        }
    )
    if not beta_available:
        blockers.append(
            {
                "code": "beta_readiness_missing",
                "label": "Rapport beta readiness absent",
                "detail": "Lance la pré-analyse Pine pour générer le gate V3.",
            }
        )
    elif not beta_ready:
        blockers.append(
            {
                "code": "beta_readiness_failed",
                "label": "V3 beta readiness non validée",
                "detail": (
                    f"readiness_score={beta.get('readiness_score')}, "
                    f"runtime_blockers={len(beta.get('runtime_blockers', []) or [])}"
                ),
            }
        )

    has_parity_reference = bool(parity_ref_payload)
    if has_parity_reference and enforce_parity_when_reference:
        ref_valid = bool(parity_ref_validation.get("valid", False))
        checks.append(
            {
                "id": "parity_reference_valid",
                "label": "Référence Pine valide",
                "required": True,
                "passed": ref_valid,
                "detail": f"errors={len(parity_ref_validation.get('errors', []) or [])}",
            }
        )
        if not ref_valid:
            blockers.append(
                {
                    "code": "parity_reference_invalid",
                    "label": "Référence de parité invalide",
                    "detail": "Corrige le JSON de référence avant exécution.",
                }
            )

        parity_pass = parity.get("parity_pass")
        parity_available = bool(parity)
        checks.append(
            {
                "id": "parity_pass",
                "label": "Parité Pine/Python validée",
                "required": True,
                "passed": parity_pass is True,
                "detail": f"available={parity_available}, status={parity.get('status')}, parity_pass={parity_pass}",
            }
        )
        if not parity_available:
            blockers.append(
                {
                    "code": "parity_report_missing",
                    "label": "Rapport de parité absent",
                    "detail": "Calcule `pine_parity_report` avant de lancer le run.",
                }
            )
        elif parity_pass is not True:
            blockers.append(
                {
                    "code": "parity_not_passed",
                    "label": "Parité non validée",
                    "detail": f"status={parity.get('status')}, parity_pass={parity_pass}",
                }
            )
    elif has_parity_reference and not enforce_parity_when_reference:
        warnings.append(
            "Référence Pine présente mais gate de parité désactivé (exécution autorisée)."
        )
    elif enforce_parity_when_reference:
        warnings.append(
            "Aucune référence Pine fournie: gate de parité ignoré pour ce run."
        )

    can_run = len(blockers) == 0
    return {
        "schema_version": "pine_execution_gate.v1",
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "strategy_mode": mode,
        "status": "pass" if can_run else "blocked",
        "can_run": can_run,
        "enforce_parity_when_reference": bool(enforce_parity_when_reference),
        "parity_reference_present": has_parity_reference,
        "checks": checks,
        "blockers": blockers,
        "warnings": warnings,
    }

