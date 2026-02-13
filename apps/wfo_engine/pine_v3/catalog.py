"""Local strategy catalog for Pine V3 (P2.2)."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import tempfile
from typing import Any, Dict, List


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _repo_root_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _catalog_dir(root_dir: str | None = None) -> str:
    base = os.path.abspath(root_dir) if root_dir else _repo_root_dir()
    folder = os.path.join(base, "reports", "pine_catalog")
    os.makedirs(folder, exist_ok=True)
    return folder


def _sources_dir(root_dir: str | None = None) -> str:
    folder = os.path.join(_catalog_dir(root_dir), "sources")
    os.makedirs(folder, exist_ok=True)
    return folder


def _catalog_path(root_dir: str | None = None) -> str:
    return os.path.join(_catalog_dir(root_dir), "catalog.json")


def _safe_filename(name: str) -> str:
    text = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(name or "strategy.pine"))
    return text or "strategy.pine"


def _read_json(path: str) -> Dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="catalog_", suffix=".tmp", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def load_catalog(root_dir: str | None = None) -> Dict[str, Any]:
    path = _catalog_path(root_dir)
    data = _read_json(path)
    if not isinstance(data, dict):
        return {
            "schema_version": "pine_strategy_catalog.v1",
            "updated_at_utc": _utc_now_iso(),
            "entries": [],
        }
    entries = data.get("entries")
    if not isinstance(entries, list):
        data["entries"] = []
    data.setdefault("schema_version", "pine_strategy_catalog.v1")
    data.setdefault("updated_at_utc", _utc_now_iso())
    return data


def save_catalog(catalog: Dict[str, Any], root_dir: str | None = None) -> None:
    payload = catalog if isinstance(catalog, dict) else {}
    payload.setdefault("schema_version", "pine_strategy_catalog.v1")
    payload["updated_at_utc"] = _utc_now_iso()
    if not isinstance(payload.get("entries"), list):
        payload["entries"] = []
    _atomic_write_json(_catalog_path(root_dir), payload)


def _build_entry(
    *,
    strategy_spec: Dict[str, Any],
    strategy_spec_validation: Dict[str, Any] | None,
    precheck_report: Dict[str, Any] | None,
    compatibility_report: Dict[str, Any] | None,
    generation_trace: Dict[str, Any] | None,
    source_text: str | None,
    source_name: str | None,
    source_path: str | None,
    library_files: List[Dict[str, Any]] | None,
    root_dir: str | None = None,
) -> Dict[str, Any]:
    spec = strategy_spec if isinstance(strategy_spec, dict) else {}
    spec_val = strategy_spec_validation if isinstance(strategy_spec_validation, dict) else {}
    pre = precheck_report if isinstance(precheck_report, dict) else {}
    compat = compatibility_report if isinstance(compatibility_report, dict) else {}
    trace = generation_trace if isinstance(generation_trace, dict) else {}
    libs = library_files if isinstance(library_files, list) else []

    strategy = spec.get("strategy") if isinstance(spec.get("strategy"), dict) else {}
    source = spec.get("source") if isinstance(spec.get("source"), dict) else {}
    source_sha1 = str(source.get("source_sha1") or pre.get("source_sha1") or "").strip()
    if not source_sha1 and isinstance(source_text, str):
        source_sha1 = hashlib.sha1(source_text.encode("utf-8", errors="ignore")).hexdigest()

    sid = str(strategy.get("id") or "").strip() or "unknown_strategy"
    sname = str(strategy.get("name") or "").strip()
    src_name = str(source_name or source.get("file_name") or "strategy_source.pine.txt").strip()
    entry_id = f"{sid}:{source_sha1}" if source_sha1 else sid

    source_snapshot_path = ""
    if isinstance(source_text, str) and source_text.strip():
        safe_name = _safe_filename(src_name)
        if not safe_name.lower().endswith((".pine", ".txt")):
            safe_name = f"{safe_name}.txt"
        snapshot_name = f"{source_sha1[:12] if source_sha1 else 'nosha'}_{safe_name}"
        source_snapshot_path = os.path.join(_sources_dir(root_dir), snapshot_name)
        if not os.path.exists(source_snapshot_path):
            with open(source_snapshot_path, "w", encoding="utf-8") as f:
                f.write(source_text)

    ts = _utc_now_iso()
    entry = {
        "entry_id": entry_id,
        "strategy_id": sid,
        "strategy_name": sname,
        "source_name": src_name,
        "source_sha1": source_sha1,
        "source_path": str(source_path or "").strip(),
        "source_snapshot_path": source_snapshot_path,
        "pine_version": source.get("pine_version"),
        "imports_count": int(len(spec.get("imports") or [])) if isinstance(spec.get("imports"), list) else 0,
        "inputs_count": int(len(spec.get("inputs") or [])) if isinstance(spec.get("inputs"), list) else 0,
        "libraries_count": int(len(libs)),
        "compatibility_status": compat.get("status"),
        "compatibility_score": compat.get("compatibility_score"),
        "precheck_status": pre.get("status"),
        "spec_valid": bool(spec_val.get("valid", False)),
        "parser_backend_used": (
            ((spec.get("transcription") or {}).get("parser_backend_used"))
            if isinstance(spec.get("transcription"), dict)
            else None
        ),
        "llm_used": bool(trace.get("llm_used", False)),
        "created_at_utc": ts,
        "updated_at_utc": ts,
        "last_used_at_utc": ts,
    }
    return entry


def upsert_catalog_entry(
    *,
    strategy_spec: Dict[str, Any],
    strategy_spec_validation: Dict[str, Any] | None,
    precheck_report: Dict[str, Any] | None,
    compatibility_report: Dict[str, Any] | None,
    generation_trace: Dict[str, Any] | None,
    source_text: str | None,
    source_name: str | None,
    source_path: str | None,
    library_files: List[Dict[str, Any]] | None,
    root_dir: str | None = None,
) -> Dict[str, Any]:
    catalog = load_catalog(root_dir)
    entries = catalog.get("entries")
    if not isinstance(entries, list):
        entries = []

    incoming = _build_entry(
        strategy_spec=strategy_spec,
        strategy_spec_validation=strategy_spec_validation,
        precheck_report=precheck_report,
        compatibility_report=compatibility_report,
        generation_trace=generation_trace,
        source_text=source_text,
        source_name=source_name,
        source_path=source_path,
        library_files=library_files,
        root_dir=root_dir,
    )

    updated = False
    for idx, row in enumerate(entries):
        if not isinstance(row, dict):
            continue
        if str(row.get("entry_id") or "") == str(incoming.get("entry_id") or ""):
            created = row.get("created_at_utc") or incoming.get("created_at_utc")
            incoming["created_at_utc"] = created
            incoming["updated_at_utc"] = _utc_now_iso()
            incoming["last_used_at_utc"] = row.get("last_used_at_utc") or incoming.get("last_used_at_utc")
            entries[idx] = incoming
            updated = True
            break
    if not updated:
        entries.append(incoming)

    # Newest first for UI.
    entries = sorted(
        [x for x in entries if isinstance(x, dict)],
        key=lambda x: str(x.get("updated_at_utc") or ""),
        reverse=True,
    )
    catalog["entries"] = entries
    save_catalog(catalog, root_dir)
    return incoming


def list_catalog_entries(query: str | None = None, limit: int = 500, root_dir: str | None = None) -> List[Dict[str, Any]]:
    catalog = load_catalog(root_dir)
    entries = [x for x in (catalog.get("entries") or []) if isinstance(x, dict)]
    q = str(query or "").strip().lower()
    if q:
        filtered: List[Dict[str, Any]] = []
        for row in entries:
            hay = " ".join(
                [
                    str(row.get("strategy_id") or ""),
                    str(row.get("strategy_name") or ""),
                    str(row.get("source_name") or ""),
                    str(row.get("source_sha1") or ""),
                ]
            ).lower()
            if q in hay:
                filtered.append(row)
        entries = filtered
    entries = sorted(entries, key=lambda x: str(x.get("updated_at_utc") or ""), reverse=True)
    return entries[: max(1, int(limit))]


def get_catalog_entry(entry_id: str, root_dir: str | None = None) -> Dict[str, Any] | None:
    wanted = str(entry_id or "").strip()
    if not wanted:
        return None
    for row in list_catalog_entries(root_dir=root_dir, limit=5000):
        if str(row.get("entry_id") or "") == wanted:
            return row
    return None


def mark_catalog_entry_used(entry_id: str, root_dir: str | None = None) -> Dict[str, Any] | None:
    wanted = str(entry_id or "").strip()
    if not wanted:
        return None
    catalog = load_catalog(root_dir)
    entries = catalog.get("entries")
    if not isinstance(entries, list):
        return None
    touched = None
    now = _utc_now_iso()
    for row in entries:
        if not isinstance(row, dict):
            continue
        if str(row.get("entry_id") or "") == wanted:
            row["last_used_at_utc"] = now
            row["updated_at_utc"] = now
            touched = row
            break
    if touched is not None:
        catalog["entries"] = sorted(
            [x for x in entries if isinstance(x, dict)],
            key=lambda x: str(x.get("updated_at_utc") or ""),
            reverse=True,
        )
        save_catalog(catalog, root_dir)
    return touched


def load_catalog_source_text(entry: Dict[str, Any] | None) -> str | None:
    row = entry if isinstance(entry, dict) else {}
    candidates = [
        str(row.get("source_path") or "").strip(),
        str(row.get("source_snapshot_path") or "").strip(),
    ]
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            try:
                with open(path, "r", encoding="latin-1", errors="replace") as f:
                    return f.read()
            except Exception:
                continue
    return None

