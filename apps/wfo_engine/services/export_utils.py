"""Helpers for export/replay metadata generation."""

from __future__ import annotations

import datetime
import os
from typing import Any

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
