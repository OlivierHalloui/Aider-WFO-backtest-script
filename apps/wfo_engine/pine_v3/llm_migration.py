"""LLM-assisted Pine -> strategy_spec.v1 migration helpers (P2.1)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Tuple

from expert import LLMConfig, OpenAICompatibleGateway

from .spec import build_strategy_spec_v1_from_pine_text, validate_strategy_spec_v1


def _utc_now_iso() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8", errors="ignore")).hexdigest()


def _extract_first_json_object(raw_text: str) -> Dict[str, Any] | None:
    text = str(raw_text or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    quote = ""
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if ch == quote and (i == 0 or text[i - 1] != "\\"):
                in_string = False
            continue
        if ch in ("'", '"'):
            in_string = True
            quote = ch
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    payload = json.loads(candidate)
                except Exception:
                    return None
                return payload if isinstance(payload, dict) else None
    return None


def build_llm_migration_prompts(
    *,
    pine_text: str,
    source_name: str,
    strategy_id: str,
    precheck_report: Dict[str, Any] | None,
    compatibility_report: Dict[str, Any] | None,
    parser_backend: str,
    baseline_spec: Dict[str, Any],
) -> Tuple[str, str]:
    """Build deterministic system/user prompts for Pine migration assistant."""
    system_prompt = (
        "You are a Pine Script migration assistant.\n"
        "Goal: produce ONE strict JSON object only, no markdown, no prose.\n"
        "Language for any free-text fields: French (professional, concise).\n"
        "Constraints:\n"
        "1) Output must follow strategy_spec.v1 structure.\n"
        "2) Keep schema_version='strategy_spec.v1'.\n"
        "3) Keep strategy.kind='pine_imported'.\n"
        "4) Preserve source metadata and detected capabilities when possible.\n"
        "5) Do not invent unsupported runtime claims.\n"
        "6) Prefer conservative values if uncertain; add warnings instead of guessing.\n"
    )
    payload = {
        "task": "migrate_pine_to_strategy_spec_v1",
        "source_name": str(source_name or ""),
        "strategy_id": str(strategy_id or ""),
        "parser_backend_requested": str(parser_backend or "auto"),
        "precheck_report": precheck_report if isinstance(precheck_report, dict) else {},
        "compatibility_report": compatibility_report if isinstance(compatibility_report, dict) else {},
        "baseline_spec": baseline_spec if isinstance(baseline_spec, dict) else {},
        "pine_source": str(pine_text or ""),
    }
    user_prompt = (
        "Generate a strict JSON object for strategy_spec.v1.\n"
        f"Input payload:\n{json.dumps(payload, ensure_ascii=True, default=str)}"
    )
    return system_prompt, user_prompt


def run_llm_spec_migration(
    *,
    pine_text: str,
    source_name: str,
    strategy_id: str,
    precheck_report: Dict[str, Any] | None,
    compatibility_report: Dict[str, Any] | None,
    parser_backend: str,
    llm_config: LLMConfig,
    gateway: OpenAICompatibleGateway | None = None,
) -> Dict[str, Any]:
    """
    Run LLM-assisted migration and revalidate deterministically.

    Safety policy:
    - baseline deterministic spec is always built first,
    - LLM output is only accepted when `validate_strategy_spec_v1` passes,
    - otherwise fallback to baseline spec.
    """
    baseline_spec = build_strategy_spec_v1_from_pine_text(
        pine_text=pine_text,
        source_name=source_name,
        strategy_id=strategy_id,
        precheck_report=precheck_report,
        compatibility_report=compatibility_report,
        parser_backend=parser_backend,
    )
    baseline_validation = validate_strategy_spec_v1(baseline_spec)

    system_prompt, user_prompt = build_llm_migration_prompts(
        pine_text=pine_text,
        source_name=source_name,
        strategy_id=strategy_id,
        precheck_report=precheck_report,
        compatibility_report=compatibility_report,
        parser_backend=parser_backend,
        baseline_spec=baseline_spec,
    )
    llm = gateway or OpenAICompatibleGateway()

    warnings: list[str] = []
    errors: list[str] = []
    raw_text = ""
    parsed_payload: Dict[str, Any] | None = None
    candidate_validation: Dict[str, Any] | None = None
    status = "error"

    try:
        raw_text = llm.generate(system_prompt=system_prompt, user_prompt=user_prompt, config=llm_config)
        parsed_payload = _extract_first_json_object(raw_text)
        if not isinstance(parsed_payload, dict):
            errors.append("LLM output is not a valid JSON object.")
            status = "fallback"
        else:
            candidate_validation = validate_strategy_spec_v1(parsed_payload)
            if bool(candidate_validation.get("valid", False)):
                status = "ok"
            else:
                status = "fallback"
                errors.extend([str(e) for e in (candidate_validation.get("errors") or [])])
    except Exception as e:  # noqa: BLE001
        status = "error"
        errors.append(f"LLM call failed: {e}")

    accepted_spec = parsed_payload if (status == "ok" and isinstance(parsed_payload, dict)) else baseline_spec
    accepted_validation = (
        candidate_validation
        if (status == "ok" and isinstance(candidate_validation, dict))
        else baseline_validation
    )
    if status != "ok":
        warnings.append("Fallback to deterministic baseline spec.")

    trace = {
        "schema_version": "pine_llm_generation_trace.v1",
        "generated_at_utc": _utc_now_iso(),
        "llm_used": True,
        "provider": str(llm_config.provider or ""),
        "model": str(llm_config.model or ""),
        "base_url": str(llm_config.base_url or ""),
        "temperature": float(llm_config.temperature),
        "max_tokens": int(llm_config.max_tokens),
        "parser_backend_requested": str(parser_backend or "auto"),
        "source_name": str(source_name or ""),
        "strategy_id": str(strategy_id or ""),
        "input_sha256": _sha256_text(pine_text),
        "system_prompt_sha256": _sha256_text(system_prompt),
        "user_prompt_sha256": _sha256_text(user_prompt),
        "llm_output_sha256": _sha256_text(raw_text),
        "status": status,
        "candidate_valid": bool(candidate_validation.get("valid", False))
        if isinstance(candidate_validation, dict)
        else False,
        "accepted_spec_sha256": _sha256_text(json.dumps(accepted_spec, sort_keys=True, ensure_ascii=False)),
        "warnings": warnings,
        "errors": errors,
    }

    return {
        "schema_version": "pine_llm_migration_report.v1",
        "status": status,
        "warnings": warnings,
        "errors": errors,
        "baseline_spec": baseline_spec,
        "baseline_validation": baseline_validation,
        "candidate_spec": parsed_payload,
        "candidate_validation": candidate_validation,
        "accepted_spec": accepted_spec,
        "accepted_validation": accepted_validation,
        "raw_output": raw_text,
        "trace": trace,
    }

