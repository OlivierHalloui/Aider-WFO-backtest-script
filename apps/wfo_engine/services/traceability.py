"""Traceability services (git metadata + payload hashing)."""

from __future__ import annotations

import os
import subprocess
from typing import Any

from domain.serialization import sanitize_for_json, sha256_json, utc_now_iso


def safe_git_command(args: list[str], cwd: str | None = None) -> str | None:
    """Execute a git command and return stripped output or None on failure."""
    try:
        output = subprocess.check_output(
            ["git", *args],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return output or None
    except Exception:
        return None


def get_git_traceability_info(cwd: str | None = None) -> dict[str, Any]:
    """Collect basic git metadata for reproducibility."""
    status = safe_git_command(["status", "--porcelain"], cwd=cwd)
    return {
        "branch": safe_git_command(["branch", "--show-current"], cwd=cwd),
        "commit": safe_git_command(["rev-parse", "HEAD"], cwd=cwd),
        "commit_short": safe_git_command(["rev-parse", "--short", "HEAD"], cwd=cwd),
        "remote_origin": safe_git_command(["remote", "get-url", "origin"], cwd=cwd),
        "working_tree_dirty": bool(status) if status is not None else None,
    }


def build_traceability_payload(
    config_snapshot: Any = None,
    results_snapshot: Any = None,
    run_metadata: dict[str, Any] | None = None,
    app_name: str = "ATDMF Strategy Walk-Forward Optimizer",
    cwd: str | None = None,
) -> dict[str, Any]:
    """Build normalized traceability payload used in UI and exports."""
    payload = {
        "generated_at_utc": utc_now_iso(),
        "app_name": app_name,
        "config_sha256": sha256_json(config_snapshot) if config_snapshot is not None else None,
        "results_sha256": sha256_json(results_snapshot) if results_snapshot is not None else None,
        "git": get_git_traceability_info(cwd=cwd or os.getcwd()),
        "run": run_metadata or {},
    }
    return sanitize_for_json(payload)
