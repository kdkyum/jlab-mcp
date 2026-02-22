"""Unit tests for local mode — no JupyterLab or SLURM needed."""

import os
import signal
from unittest.mock import patch

import pytest

from jlab_mcp.local import is_local_running, stop_jupyter_local


class TestIsLocalRunning:
    def test_current_process_is_alive(self):
        assert is_local_running(os.getpid()) is True

    def test_bogus_pid_is_not_alive(self):
        # PID 2**22 is extremely unlikely to exist
        assert is_local_running(2**22) is False


class TestStopJupyterLocal:
    def test_no_raise_on_dead_pid(self):
        # Should not raise even if PID doesn't exist
        stop_jupyter_local(2**22)

    def test_sigterm_sent(self):
        with patch("jlab_mcp.local.os.kill") as mock_kill:
            stop_jupyter_local(12345)
            mock_kill.assert_called_once_with(12345, signal.SIGTERM)


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
