from __future__ import annotations

import datetime as dt
import json
import os
from typing import Dict

from .models import ExpertResponse


class ExpertStorage:
    def save_report(self, response: ExpertResponse, folder: str = "reports/expert") -> str:
        os.makedirs(folder, exist_ok=True)
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(folder, f"expert_{response.run_id}_{ts}.json")
        payload: Dict[str, object] = {
            "saved_at": dt.datetime.utcnow().isoformat() + "Z",
            "run_id": response.run_id,
            "status": response.status,
            "result": response.result_json,
            "raw_text": response.raw_text,
            "model_info": response.model_info,
            "timings_ms": response.timings_ms,
            "warnings": response.warnings,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path

    def save_audit_line(self, response: ExpertResponse, folder: str = "reports/expert") -> str:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "expert_audit.jsonl")
        line = {
            "at": dt.datetime.utcnow().isoformat() + "Z",
            "run_id": response.run_id,
            "status": response.status,
            "provider": response.model_info.get("provider"),
            "model": response.model_info.get("model"),
            "timings_ms": response.timings_ms,
            "warnings_count": len(response.warnings),
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        return path
