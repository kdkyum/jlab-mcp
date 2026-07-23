"""Unit tests for autostart (start_server/wait_for_server plumbing) —
no JupyterLab, SLURM, or uv needed."""

import time
from unittest.mock import MagicMock, patch

import pytest

from jlab_mcp import autostart


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point every file autostart touches at a temp dir."""
    monkeypatch.setattr(autostart, "MODE_FILE", tmp_path / "run-mode")
    monkeypatch.setattr(autostart, "SLURM_OPTS_FILE", tmp_path / "slurm-options")
    monkeypatch.setattr(autostart, "START_LOG", tmp_path / "start.log")
    monkeypatch.setattr(autostart, "_LOCK_FILE", tmp_path / "lifecycle.lock")
    monkeypatch.setattr(
        "jlab_mcp.config.STATUS_FILE", tmp_path / "server-status"
    )
    monkeypatch.setattr(autostart, "_start_proc", None)
    monkeypatch.setattr(autostart, "_last_spawn", 0.0)
    monkeypatch.delenv("JLAB_MCP_RUN_MODE", raising=False)
    return tmp_path


def _write_status(tmp_path, state, **kwargs):
    lines = [f"STATE={state}"]
    lines += [f"{k.upper()}={v}" for k, v in kwargs.items()]
    (tmp_path / "server-status").write_text("\n".join(lines) + "\n")


class TestModePersistence:
    def test_roundtrip(self, sandbox):
        autostart.save_mode("slurm")
        assert autostart.read_saved_mode() == "slurm"

    def test_missing_file(self, sandbox):
        assert autostart.read_saved_mode() is None

    def test_junk_content(self, sandbox):
        autostart.MODE_FILE.write_text("banana\n")
        assert autostart.read_saved_mode() is None

    def test_save_rejects_invalid(self, sandbox):
        with pytest.raises(ValueError):
            autostart.save_mode("cloud")


class TestResolveMode:
    def test_unconfigured_returns_none(self, sandbox):
        assert autostart.resolve_mode() is None

    def test_explicit_wins_and_persists(self, sandbox, monkeypatch):
        monkeypatch.setenv("JLAB_MCP_RUN_MODE", "slurm")
        assert autostart.resolve_mode("local") == "local"
        assert autostart.read_saved_mode() == "local"

    def test_explicit_invalid_raises(self, sandbox):
        with pytest.raises(ValueError):
            autostart.resolve_mode("cloud")

    def test_env_beats_saved(self, sandbox, monkeypatch):
        autostart.save_mode("local")
        monkeypatch.setenv("JLAB_MCP_RUN_MODE", "slurm")
        assert autostart.resolve_mode() == "slurm"

    def test_saved_used_when_no_env(self, sandbox):
        autostart.save_mode("slurm")
        assert autostart.resolve_mode() == "slurm"


class TestSpawnValidation:
    def test_invalid_mode(self, sandbox):
        with pytest.raises(ValueError):
            autostart.spawn_start("cloud")

    def test_invalid_time_limit(self, sandbox):
        with pytest.raises(ValueError):
            autostart.spawn_start("slurm", {"time": "4:00:00; rm -rf /"})

    def test_valid_time_limits(self):
        for t in ("30", "4:00:00", "1-00:00:00"):
            assert autostart._TIME_LIMIT_RE.fullmatch(t)


class TestSlurmOptions:
    def test_validate_normalizes_and_drops_empty(self):
        opts = autostart.validate_slurm_options(
            {"time": "8:00:00", "cpus": 18, "partition": "", "mem_mb": 0}
        )
        assert opts == {"time": "8:00:00", "cpus": "18"}

    def test_validate_rejects_shell_metacharacters(self):
        for key, value in (
            ("partition", "gpu1; rm -rf /"),
            ("gres", "gpu:a100:1 --oops"),
            ("cpus", "18 18"),
        ):
            with pytest.raises(ValueError):
                autostart.validate_slurm_options({key: value})

    def test_validate_rejects_unknown_key(self):
        with pytest.raises(ValueError):
            autostart.validate_slurm_options({"nodes": "2"})

    def test_roundtrip(self, sandbox):
        autostart.save_slurm_options(
            {"time": "1-00:00:00", "partition": "gpu1", "gres": "gpu:a100:1"}
        )
        assert autostart.read_slurm_options() == {
            "time": "1-00:00:00",
            "partition": "gpu1",
            "gres": "gpu:a100:1",
        }

    def test_missing_file(self, sandbox):
        assert autostart.read_slurm_options() == {}

    def test_corrupt_file_ignored(self, sandbox):
        autostart.SLURM_OPTS_FILE.write_text("partition=gpu1; rm -rf /\n")
        assert autostart.read_slurm_options() == {}

    def test_spawn_forwards_env(self, sandbox, monkeypatch):
        captured = {}

        def fake_popen(*args, **kwargs):
            captured.update(kwargs["env"])
            proc = MagicMock()
            proc.pid = 12345
            return proc

        monkeypatch.setattr(autostart.subprocess, "Popen", fake_popen)
        autostart.spawn_start(
            "slurm", {"time": "8:00:00", "cpus": 18, "mem_mb": 64000}
        )
        assert captured["JLAB_MCP_RUN_MODE"] == "slurm"
        assert captured["JLAB_MCP_SLURM_TIME"] == "8:00:00"
        assert captured["JLAB_MCP_SLURM_CPUS"] == "18"
        assert captured["JLAB_MCP_SLURM_MEM"] == "64000"


_SINFO_OUTPUT = """\
interactive*|up|2:00:00|4|144|500000|(null)
gpu|up|1-00:00:00|128|144|500000|gpu:a100:4(S:0-1)
gpu|up|1-00:00:00|2|144|500000|gpu:a100:4
gpu1|up|1-00:00:00|1|144|500000|gpu:a100:4
malformed line without pipes
"""


class TestSurveySlurmPartitions:
    def test_parses_and_merges(self, monkeypatch):
        monkeypatch.setattr(autostart.shutil, "which", lambda _: "/usr/bin/sinfo")
        run_result = MagicMock(returncode=0, stdout=_SINFO_OUTPUT)
        monkeypatch.setattr(
            autostart.subprocess, "run", lambda *a, **k: run_result
        )
        rows = autostart.survey_slurm_partitions()
        by_name = {r["partition"]: r for r in rows}

        assert by_name["interactive"]["default_partition"] is True
        assert by_name["interactive"]["gres"] == ""
        assert by_name["interactive"]["max_time"] == "2:00:00"
        # socket annotation stripped -> the two gpu rows merge, nodes summed
        assert by_name["gpu"]["nodes"] == 130
        assert by_name["gpu"]["gres"] == "gpu:a100:4"
        assert by_name["gpu1"]["nodes"] == 1
        assert len(rows) == 3  # malformed line dropped

    def test_no_sinfo(self, monkeypatch):
        monkeypatch.setattr(autostart.shutil, "which", lambda _: None)
        assert autostart.survey_slurm_partitions() == []

    def test_sinfo_failure(self, monkeypatch):
        monkeypatch.setattr(autostart.shutil, "which", lambda _: "/usr/bin/sinfo")
        monkeypatch.setattr(
            autostart.subprocess, "run",
            lambda *a, **k: MagicMock(returncode=1, stdout=""),
        )
        assert autostart.survey_slurm_partitions() == []


class TestStartInFlight:
    def test_no_proc_no_lock(self, sandbox):
        assert autostart.start_in_flight() is False

    def test_live_proc(self, sandbox, monkeypatch):
        proc = MagicMock()
        proc.poll.return_value = None
        monkeypatch.setattr(autostart, "_start_proc", proc)
        assert autostart.start_in_flight() is True

    def test_dead_proc_reaped(self, sandbox, monkeypatch):
        proc = MagicMock()
        proc.poll.return_value = 1
        monkeypatch.setattr(autostart, "_start_proc", proc)
        assert autostart.start_in_flight() is False
        assert autostart._start_proc is None


class TestPollUntilReady:
    def test_ready_healthy(self, sandbox):
        _write_status(
            sandbox, "ready",
            hostname="node1", port="18500", token="tok", job_id="123",
        )
        client = MagicMock()
        client.health_check.return_value = True
        with patch.object(autostart, "JupyterLabClient", return_value=client):
            result = autostart.poll_until_ready(timeout=5)
        assert result["status"] == "ready"
        assert result["hostname"] == "node1"
        assert result["port"] == 18500
        assert result["job_id"] == "123"

    def test_error_state_terminal_when_idle(self, sandbox):
        _write_status(sandbox, "error", message="sbatch exploded")
        autostart.START_LOG.write_text("boom\n")
        result = autostart.poll_until_ready(timeout=5)
        assert result["status"] == "error"
        assert result["message"] == "sbatch exploded"
        assert "boom" in result["log_tail"]

    def test_error_state_stale_while_start_in_flight(self, sandbox, monkeypatch):
        """A leftover 'error' status must not be reported while a fresh
        start attempt (bootstrap) is still running."""
        _write_status(sandbox, "error", message="old failure")
        proc = MagicMock()
        proc.poll.return_value = None
        monkeypatch.setattr(autostart, "_start_proc", proc)
        result = autostart.poll_until_ready(timeout=3)
        assert result["status"] == "timeout"

    def test_timeout_pending_reports_queue(self, sandbox, monkeypatch):
        _write_status(sandbox, "pending", job_id="777")
        # Recent spawn suppresses the auto-resume path
        monkeypatch.setattr(autostart, "_last_spawn", time.time())
        result = autostart.poll_until_ready(timeout=3)
        assert result["status"] == "timeout"
        assert result["state"] == "pending"
        assert "777" in result["message"]

    def test_pending_resumes_lapsed_queue_wait(self, sandbox, monkeypatch):
        _write_status(sandbox, "pending", job_id="777")
        spawned = []

        def fake_spawn(mode, slurm_opts=None):
            spawned.append((mode, slurm_opts))
            autostart._last_spawn = time.time()

        monkeypatch.setattr(autostart, "spawn_start", fake_spawn)
        autostart.save_slurm_options({"time": "8:00:00"})
        autostart.poll_until_ready(timeout=3)
        assert spawned == [("slurm", {"time": "8:00:00"})]

    def test_no_status_file_times_out(self, sandbox):
        result = autostart.poll_until_ready(timeout=3)
        assert result["status"] == "timeout"
        assert result["state"] == "no_status"
