from __future__ import annotations

import json
import re
import time
from typing import Any, Dict

from .models import ExpertInputData, ExpertRequest, ExpertResponse, LLMConfig
from .prompt_builder import ExpertPromptBuilder
from .schema import ExpertOutputSchema


class ExpertAnalyzer:
    def __init__(self, llm_gateway: Any, prompt_builder: ExpertPromptBuilder):
        self.llm_gateway = llm_gateway
        self.prompt_builder = prompt_builder

    def analyze(
        self,
        data: ExpertInputData,
        request: ExpertRequest,
        llm_config: LLMConfig,
        system_prompt_override: str | None = None,
        user_prompt_override: str | None = None,
    ) -> ExpertResponse:
        t0 = time.time()
        warnings = []
        run_id = data.context.run_id

        if isinstance(system_prompt_override, str) and system_prompt_override.strip():
            system_prompt = system_prompt_override.strip()
        else:
            system_prompt = self.prompt_builder.build_system_prompt(request.mode, request.detail_level)

        if isinstance(user_prompt_override, str) and user_prompt_override.strip():
            user_prompt = user_prompt_override.strip()
        else:
            user_prompt = self.prompt_builder.build_user_prompt(
                data=data,
                request=request,
                output_schema=ExpertOutputSchema.json_schema_v1(),
            )

        try:
            raw_text = self.llm_gateway.generate(system_prompt, user_prompt, llm_config)
        except Exception as e:  # noqa: BLE001
            return ExpertResponse(
                run_id=run_id,
                status="error",
                result_json={
                    "schema_version": "expert.v1",
                    "run_id": run_id,
                    "mode": request.mode,
                    "detail_level": request.detail_level,
                    "global_assessment": {},
                    "key_findings": [],
                    "recommended_actions": [],
                    "alerts": [
                        {
                            "type": "other",
                            "severity": "high",
                            "message": f"LLM request failed: {e}",
                        }
                    ],
                    "limitations": ["LLM call failed."],
                    "disclaimer": "Interpretation unavailable due to API error.",
                },
                raw_text="",
                model_info={"provider": llm_config.provider, "model": llm_config.model},
                timings_ms={"total": int((time.time() - t0) * 1000)},
                warnings=[str(e)],
            )

        parsed = self._post_process(raw_text)
        ok, schema_errors = ExpertOutputSchema.validate(parsed)
        if not ok:
            warnings.extend(schema_errors)
            parsed = self._repair_payload(parsed, data, request, schema_errors)
            status = "partial"
        else:
            status = "ok"

        return ExpertResponse(
            run_id=run_id,
            status=status,
            result_json=parsed,
            raw_text=raw_text,
            model_info={"provider": llm_config.provider, "model": llm_config.model},
            timings_ms={"total": int((time.time() - t0) * 1000)},
            warnings=warnings,
        )

    def _post_process(self, llm_text: str) -> Dict[str, Any]:
        text = (llm_text or "").strip()
        if not text:
            return {}

        # Remove fenced blocks if present.
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()

        # Try direct parse.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Fallback: extract the largest JSON object.
        first = text.find("{")
        last = text.rfind("}")
        if first >= 0 and last > first:
            chunk = text[first:last + 1]
            try:
                parsed = json.loads(chunk)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        return {"raw_output": text}

    @staticmethod
    def _repair_payload(
        payload: Dict[str, Any],
        data: ExpertInputData,
        request: ExpertRequest,
        schema_errors: list[str],
    ) -> Dict[str, Any]:
        out = dict(payload) if isinstance(payload, dict) else {}
        out.setdefault("schema_version", "expert.v1")
        out.setdefault("run_id", data.context.run_id)
        out.setdefault("mode", request.mode)
        out.setdefault("detail_level", request.detail_level)
        out.setdefault("global_assessment", {})
        out.setdefault("key_findings", [])
        out.setdefault("recommended_actions", [])
        out.setdefault("alerts", [])
        out.setdefault("limitations", [])
        out.setdefault("disclaimer", "Interpretation generated with missing fields.")
        if schema_errors:
            out["limitations"] = list(out.get("limitations", [])) + [f"schema_validation: {e}" for e in schema_errors]
        return out
