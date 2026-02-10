from __future__ import annotations

from typing import Any, Dict, List, Tuple


class ExpertOutputSchema:
    @staticmethod
    def json_schema_v1() -> Dict[str, Any]:
        return {
            "schema_version": "expert.v1",
            "required_keys": [
                "schema_version",
                "run_id",
                "mode",
                "detail_level",
                "global_assessment",
                "key_findings",
                "recommended_actions",
                "alerts",
                "limitations",
                "disclaimer",
            ],
        }

    @staticmethod
    def validate(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if not isinstance(payload, dict):
            return False, ["Payload is not a JSON object."]

        schema = ExpertOutputSchema.json_schema_v1()
        required_keys = schema.get("required_keys", [])
        for key in required_keys:
            if key not in payload:
                errors.append(f"Missing required key: {key}")

        if payload.get("schema_version") not in {"expert.v1"}:
            errors.append("schema_version must be 'expert.v1'.")

        if "key_findings" in payload and not isinstance(payload.get("key_findings"), list):
            errors.append("key_findings must be a list.")

        if "recommended_actions" in payload and not isinstance(payload.get("recommended_actions"), list):
            errors.append("recommended_actions must be a list.")

        if "alerts" in payload and not isinstance(payload.get("alerts"), list):
            errors.append("alerts must be a list.")

        return len(errors) == 0, errors
