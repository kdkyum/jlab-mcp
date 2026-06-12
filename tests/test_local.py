"""Unit tests for local mode — no JupyterLab or SLURM needed."""

import os
import signal
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from jlab_mcp.local import (
    _pid_is_jupyter,
    is_local_running,
    start_jupyter_local,
    stop_jupyter_local,
)


class TestIsLocalRunning:
    def test_live_jupyter_pid_is_alive(self):
        with patch("jlab_mcp.local._pid_is_jupyter", return_value=True):
            assert is_local_running(os.getpid()) is True

    def test_live_non_jupyter_pid_is_stale(self):
        """A recycled PID belonging to some other process must be treated
        as a stale status file, not a running server."""
        proc = subprocess.Popen(["sleep", "30"])
        try:
            assert is_local_running(proc.pid) is False
        finally:
            proc.kill()
            proc.wait()

    def test_bogus_pid_is_not_alive(self):
        # PID 2**22 is extremely unlikely to exist
        assert is_local_running(2**22) is False


class TestStopJupyterLocal:
    def test_no_raise_on_dead_pid(self):
        # Should not raise even if PID doesn't exist
        stop_jupyter_local(2**22)

    def test_sigterm_sent_to_jupyter(self):
        with patch("jlab_mcp.local._pid_is_jupyter", return_value=True), \
             patch("jlab_mcp.local.os.kill") as mock_kill:
            stop_jupyter_local(12345)
            mock_kill.assert_called_once_with(12345, signal.SIGTERM)

    def test_never_kills_unrelated_process(self):
        """After PID recycling, stop must not SIGTERM whatever process
        happens to own the recorded PID now."""
        with patch("jlab_mcp.local._pid_is_jupyter", return_value=False), \
             patch("jlab_mcp.local.os.kill") as mock_kill:
            stop_jupyter_local(12345)
            mock_kill.assert_not_called()


class TestPidIsJupyter:
    def test_current_process_is_not_jupyter(self):
        # The test runner is python/pytest, not jupyter
        # (guard: skip if argv happens to contain 'jupyter', e.g. -k filters)
        if any("jupyter" in arg for arg in sys.argv):
            pytest.skip("test runner argv mentions jupyter")
        assert _pid_is_jupyter(os.getpid()) is False

    def test_unreadable_proc_falls_back_to_true(self):
        # PID that doesn't exist -> /proc/<pid>/cmdline unreadable -> trust PID
        assert _pid_is_jupyter(2**22) is True


class TestStartJupyterLocal:
    def test_token_in_env_not_argv(self, tmp_path):
        """The token must travel via JUPYTER_TOKEN (owner-only environ),
        never via argv (world-readable /proc/<pid>/cmdline)."""
        fake_proc = MagicMock()
        fake_proc.pid = 4242
        with patch("jlab_mcp.local.subprocess.Popen",
                   return_value=fake_proc) as mock_popen:
            proc, hostname, port, token = start_jupyter_local()

        assert proc is fake_proc
        cmd = mock_popen.call_args.args[0]
        env = mock_popen.call_args.kwargs["env"]
        assert env["JUPYTER_TOKEN"] == token
        assert not any(token in arg for arg in cmd)
        # Port collisions must fail fast, not silently rebind
        assert "--ServerApp.port_retries=0" in cmd


class TestDetectRunMode:
    def test_env_var_local(self):
        with patch.dict(os.environ, {"JLAB_MCP_RUN_MODE": "local"}):
            from jlab_mcp.config import _detect_run_mode
            assert _detect_run_mode() == "local"

    def test_env_var_slurm(self):
        with patch.dict(os.environ, {"JLAB_MCP_RUN_MODE": "slurm"}):
            from jlab_mcp.config import _detect_run_mode
            assert _detect_run_mode() == "slurm"

    def test_auto_detect_with_sbatch(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JLAB_MCP_RUN_MODE", None)
            with patch("jlab_mcp.config.shutil.which", return_value="/usr/bin/sbatch"):
                from jlab_mcp.config import _detect_run_mode
                assert _detect_run_mode() == "slurm"

    def test_auto_detect_without_sbatch(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JLAB_MCP_RUN_MODE", None)
            with patch("jlab_mcp.config.shutil.which", return_value=None):
                from jlab_mcp.config import _detect_run_mode
                assert _detect_run_mode() == "local"
