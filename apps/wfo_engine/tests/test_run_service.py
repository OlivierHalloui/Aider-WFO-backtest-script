"""Tests for services/run_service.py checkpoint/persistence layer — no VBT dependency."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import stat

import pytest

from services.run_service import _checkpoint_job_state, _write_window_result


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
