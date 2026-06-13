"""Tests for services/run_service.py checkpoint/persistence layer — no VBT dependency."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import stat

import pytest

from services.run_service import (
    _checkpoint_job_state,
    _finalize_run_dir,
    _write_window_result,
    mark_run_resolved,
    scan_orphaned_runs,
)


class TestCheckpointJobState:
    def test_happy_path_writes_json(self, tmp_path):
        run_dir = tmp_path / "run01"
        job_state = {"status": "running", "progress": 0.5, "window": 3}
        _checkpoint_job_state(job_state, run_dir)
        checkpoint = run_dir / "checkpoint.json"
        assert checkpoint.exists()
        data = json.loads(checkpoint.read_text())
        assert data["status"] == "running"
        assert data["progress"] == pytest.approx(0.5)

    def test_atomic_replace_no_tmp_left(self, tmp_path):
        run_dir = tmp_path / "run02"
        _checkpoint_job_state({"status": "running"}, run_dir)
        tmp_files = list(run_dir.glob("*.tmp"))
        assert tmp_files == []

    def test_creates_run_dir_if_missing(self, tmp_path):
        run_dir = tmp_path / "deep" / "nested" / "run"
        assert not run_dir.exists()
        _checkpoint_job_state({"status": "running"}, run_dir)
        assert (run_dir / "checkpoint.json").exists()

    def test_inaccessible_dir_logs_error_not_debug(self, tmp_path, caplog, monkeypatch):
        # Use monkeypatch instead of chmod — chmod is bypassed by root.
        import pathlib

        real_mkdir = pathlib.Path.mkdir

        def failing_mkdir(self, **kwargs):
            raise OSError("Permission denied (simulated)")

        run_dir = tmp_path / "locked" / "run"
        job_state = {}
        monkeypatch.setattr(pathlib.Path, "mkdir", failing_mkdir)
        with caplog.at_level(logging.ERROR, logger="services.run_service"):
            _checkpoint_job_state(job_state, run_dir)
        # Must log at ERROR level, not just DEBUG
        error_records = [r for r in caplog.records if r.levelname == "ERROR"]
        assert error_records, "Expected ERROR log when dir mkdir fails"
        # Must propagate warning to job_state
        assert "warnings" in job_state
        assert any("Checkpoint" in w for w in job_state["warnings"])

    def test_inaccessible_dir_does_not_raise(self, tmp_path, monkeypatch):
        import pathlib

        def failing_mkdir(self, **kwargs):
            raise OSError("Permission denied (simulated)")

        run_dir = tmp_path / "locked2" / "run"
        monkeypatch.setattr(pathlib.Path, "mkdir", failing_mkdir)
        _checkpoint_job_state({}, run_dir)  # must not raise

    def test_write_failure_logs_error(self, tmp_path, caplog, monkeypatch):
        run_dir = tmp_path / "run_write_fail"
        run_dir.mkdir()

        def bad_replace(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", bad_replace)
        job_state = {}
        with caplog.at_level(logging.ERROR, logger="services.run_service"):
            _checkpoint_job_state(job_state, run_dir)
        error_records = [r for r in caplog.records if r.levelname == "ERROR"]
        assert error_records
        assert "warnings" in job_state

    def test_none_job_state_does_not_crash(self, tmp_path):
        run_dir = tmp_path / "run_none"
        _checkpoint_job_state(None, run_dir)  # type: ignore[arg-type]


class TestWriteWindowResult:
    def test_happy_path(self, tmp_path):
        run_dir = tmp_path / "run_w"
        payload = {"evaluations": 100, "speed": 12.5, "window_metrics": {"sharpe": 1.2}}
        _write_window_result(3, payload, run_dir)
        win_file = run_dir / "windows" / "window_0003.json"
        assert win_file.exists()
        data = json.loads(win_file.read_text())
        assert data["window"] == 3
        assert data["evaluations"] == 100

    def test_atomic_replace_no_tmp_left(self, tmp_path):
        run_dir = tmp_path / "run_w2"
        _write_window_result(1, {}, run_dir)
        tmp_files = list((run_dir / "windows").glob("*.tmp"))
        assert tmp_files == []

    def test_write_failure_logs_error_not_debug(self, tmp_path, caplog, monkeypatch):
        run_dir = tmp_path / "run_wfail"
        run_dir.mkdir()
        (run_dir / "windows").mkdir()

        def bad_replace(src, dst):
            raise OSError("no space")

        monkeypatch.setattr(os, "replace", bad_replace)
        with caplog.at_level(logging.ERROR, logger="services.run_service"):
            _write_window_result(0, {}, run_dir)
        error_records = [r for r in caplog.records if r.levelname == "ERROR"]
        assert error_records

    def test_write_failure_does_not_raise(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run_wfail2"
        run_dir.mkdir()
        (run_dir / "windows").mkdir()
        monkeypatch.setattr(os, "replace", lambda s, d: (_ for _ in ()).throw(OSError("fail")))
        _write_window_result(0, {}, run_dir)  # must not raise


def _make_run_dir(root: pathlib.Path, name: str, *, checkpoint: bool = True,
                  completed: bool = False, n_windows: int = 0, snap: dict | None = None):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    if checkpoint:
        payload = {"status": "running", "progress": 0.4, "window": 2,
                   "run_id": name, "started_at_utc": "2026-06-12T10:00:00+00:00"}
        payload.update(snap or {})
        (run_dir / "checkpoint.json").write_text(json.dumps(payload), encoding="utf-8")
    if completed:
        (run_dir / "completed.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    if n_windows:
        win_dir = run_dir / "windows"
        win_dir.mkdir()
        for i in range(n_windows):
            (win_dir / f"window_{i:04d}.json").write_text("{}", encoding="utf-8")
    return run_dir


class TestOrphanedRuns:
    def test_orphan_detected(self, tmp_path):
        _make_run_dir(tmp_path, "run_orphan", n_windows=2)
        orphans = scan_orphaned_runs(tmp_path)
        assert len(orphans) == 1
        o = orphans[0]
        assert o["run_id"] == "run_orphan"
        assert o["windows_completed"] == 2
        assert o["progress"] == pytest.approx(0.4)
        assert o["started_at_utc"] == "2026-06-12T10:00:00+00:00"

    def test_resolved_run_not_reported(self, tmp_path):
        _make_run_dir(tmp_path, "run_done", completed=True)
        assert scan_orphaned_runs(tmp_path) == []

    def test_dir_without_checkpoint_ignored(self, tmp_path):
        _make_run_dir(tmp_path, "run_empty", checkpoint=False)
        assert scan_orphaned_runs(tmp_path) == []

    def test_missing_root_returns_empty(self, tmp_path):
        assert scan_orphaned_runs(tmp_path / "does_not_exist") == []

    def test_corrupt_checkpoint_still_reported(self, tmp_path):
        run_dir = tmp_path / "run_corrupt"
        run_dir.mkdir()
        (run_dir / "checkpoint.json").write_text("{not json", encoding="utf-8")
        orphans = scan_orphaned_runs(tmp_path)
        assert len(orphans) == 1
        assert orphans[0]["run_id"] == "run_corrupt"

    def test_finalize_then_scan_excludes_run(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run_fin")
        _finalize_run_dir(run_dir, "completed", {"run_id": "run_fin", "window": 4})
        marker = json.loads((run_dir / "completed.json").read_text())
        assert marker["status"] == "completed"
        assert marker["window"] == 4
        assert marker["ended_at_utc"]
        assert scan_orphaned_runs(tmp_path) == []

    def test_finalize_none_run_dir_noop(self):
        _finalize_run_dir(None, "completed")  # must not raise

    def test_finalize_write_failure_does_not_raise(self, tmp_path, caplog, monkeypatch):
        run_dir = _make_run_dir(tmp_path, "run_fin_fail")
        monkeypatch.setattr(os, "replace", lambda s, d: (_ for _ in ()).throw(OSError("disk full")))
        with caplog.at_level(logging.ERROR, logger="services.run_service"):
            _finalize_run_dir(run_dir, "completed")
        assert any(r.levelname == "ERROR" for r in caplog.records)

    def test_mark_run_resolved_acknowledges(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run_ack")
        assert len(scan_orphaned_runs(tmp_path)) == 1
        mark_run_resolved(run_dir)
        assert scan_orphaned_runs(tmp_path) == []
        marker = json.loads((run_dir / "completed.json").read_text())
        assert marker["status"] == "acknowledged"

    def test_no_tmp_files_left(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run_tmp")
        _finalize_run_dir(run_dir, "stopped")
        assert list(run_dir.glob("*.tmp")) == []
