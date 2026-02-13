"""Tests for Pine V3 strategy catalog (P2.2)."""

from __future__ import annotations

import os
import sys

TEST_DIR = os.path.dirname(__file__)
WFO_ENGINE_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
sys.path.append(WFO_ENGINE_DIR)

from pine_v3.catalog import (
    get_catalog_entry,
    list_catalog_entries,
    load_catalog,
    load_catalog_source_text,
    mark_catalog_entry_used,
    upsert_catalog_entry,
)


def _sample_spec(strategy_id: str = "pine_demo", sha1: str = "abc123") -> dict:
    return {
        "schema_version": "strategy_spec.v1",
        "strategy": {"id": strategy_id, "name": "Demo", "kind": "pine_imported"},
        "source": {"source_sha1": sha1, "file_name": "demo.txt", "pine_version": 6},
        "imports": [],
        "inputs": [],
        "transcription": {"parser_backend_used": "regex"},
    }


def test_catalog_upsert_and_retrieve(tmp_path) -> None:
    root = str(tmp_path)
    spec = _sample_spec()
    entry = upsert_catalog_entry(
        strategy_spec=spec,
        strategy_spec_validation={"valid": True},
        precheck_report={"status": "valid"},
        compatibility_report={"status": "compatible", "compatibility_score": 100},
        generation_trace={"llm_used": False},
        source_text="//@version=6\nstrategy('Demo')\n",
        source_name="demo.txt",
        source_path="",
        library_files=[],
        root_dir=root,
    )
    assert str(entry.get("entry_id", "")).startswith("pine_demo:")

    loaded = load_catalog(root)
    assert loaded.get("schema_version") == "pine_strategy_catalog.v1"
    assert isinstance(loaded.get("entries"), list) and len(loaded["entries"]) == 1

    row = get_catalog_entry(entry.get("entry_id"), root_dir=root)
    assert isinstance(row, dict)
    text = load_catalog_source_text(row)
    assert isinstance(text, str) and "strategy" in text


def test_catalog_query_and_mark_used(tmp_path) -> None:
    root = str(tmp_path)
    upsert_catalog_entry(
        strategy_spec=_sample_spec(strategy_id="pine_a", sha1="sha_a"),
        strategy_spec_validation={"valid": True},
        precheck_report={"status": "valid"},
        compatibility_report={"status": "compatible", "compatibility_score": 95},
        generation_trace={"llm_used": False},
        source_text="a",
        source_name="a.txt",
        source_path="",
        library_files=[],
        root_dir=root,
    )
    upsert_catalog_entry(
        strategy_spec=_sample_spec(strategy_id="pine_b", sha1="sha_b"),
        strategy_spec_validation={"valid": True},
        precheck_report={"status": "valid"},
        compatibility_report={"status": "compatible", "compatibility_score": 90},
        generation_trace={"llm_used": True},
        source_text="b",
        source_name="b.txt",
        source_path="",
        library_files=[],
        root_dir=root,
    )

    rows = list_catalog_entries(query="pine_b", root_dir=root)
    assert len(rows) == 1
    assert rows[0].get("strategy_id") == "pine_b"

    entry_id = rows[0].get("entry_id")
    touched = mark_catalog_entry_used(entry_id, root_dir=root)
    assert isinstance(touched, dict)
    assert str(touched.get("last_used_at_utc") or "").strip() != ""
