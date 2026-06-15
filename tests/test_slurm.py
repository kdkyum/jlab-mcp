"""Unit tests for slurm.py — parsing and error handling, no SLURM needed."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from jlab_mcp.slurm import (
    SlurmCommandError,
    _advertised_host_expr,
    cancel_job,
    get_job_state,
    is_job_alive,
    is_job_running,
    parse_connection_file,
    parse_sbatch_output,
    parse_squeue_output,
    render_slurm_script,
)


def _completed(rc: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(
        args=[], returncode=rc, stdout=stdout, stderr=stderr
    )


class TestParseSbatchOutput:
    def test_normal_output(self):
        assert parse_sbatch_output("Submitted batch job 12345") == "12345"

    def test_large_job_id(self):
        assert parse_sbatch_output("Submitted batch job 9876543") == "9876543"

    def test_trailing_whitespace(self):
        assert parse_sbatch_output("Submitted batch job 12345\n") == "12345"

    def test_multi_cluster_output(self):
        """Federated SLURM appends 'on cluster <name>' — the job ID is not
        the last token."""
        assert (
            parse_sbatch_output("Submitted batch job 12345 on cluster viper")
            == "12345"
        )

    def test_malformed_output(self):
        with pytest.raises(ValueError):
            parse_sbatch_output("Error: something went wrong")

    def test_empty_output(self):
        with pytest.raises(ValueError):
            parse_sbatch_output("")


class TestGetJobState:
    def test_returns_state(self):
        with patch("jlab_mcp.slurm.subprocess.run",
                   return_value=_completed(0, "RUNNING\n")):
            assert get_job_state("123") == "RUNNING"

    def test_empty_with_rc_zero_means_gone(self):
        with patch("jlab_mcp.slurm.subprocess.run",
                   return_value=_completed(0, "")):
            assert get_job_state("123") == ""

    def test_invalid_job_id_means_gone(self):
        with patch("jlab_mcp.slurm.subprocess.run",
                   return_value=_completed(1, "", "slurm_load_jobs error: Invalid job id specified")):
            assert get_job_state("123") == ""

    def test_transient_failure_then_success(self):
        responses = iter([
            _completed(1, "", "slurm_load_jobs error: Socket timed out"),
            _completed(0, "PENDING\n"),
        ])
        with patch("jlab_mcp.slurm.subprocess.run",
                   side_effect=lambda *a, **k: next(responses)), \
             patch("jlab_mcp.slurm.time.sleep"):
            assert get_job_state("123") == "PENDING"

    def test_persistent_failure_raises(self):
        """A controller outage must NOT look like 'job gone'."""
        with patch("jlab_mcp.slurm.subprocess.run",
                   return_value=_completed(1, "", "Socket timed out")), \
             patch("jlab_mcp.slurm.time.sleep"):
            with pytest.raises(SlurmCommandError):
                get_job_state("123")


class TestJobAliveFailSafe:
    def test_is_job_running_assumes_alive_on_squeue_failure(self):
        with patch("jlab_mcp.slurm.get_job_state",
                   side_effect=SlurmCommandError("squeue down")):
            assert is_job_running("123") is True

    def test_is_job_alive_assumes_alive_on_squeue_failure(self):
        with patch("jlab_mcp.slurm.get_job_state",
                   side_effect=SlurmCommandError("squeue down")):
            assert is_job_alive("123") is True

    def test_is_job_alive_false_when_gone(self):
        with patch("jlab_mcp.slurm.get_job_state", return_value=""):
            assert is_job_alive("123") is False


class TestCancelJob:
    def test_success(self):
        with patch("jlab_mcp.slurm.subprocess.run",
                   return_value=_completed(0)):
            cancel_job("123")  # no raise

    def test_already_gone_is_ok(self):
        with patch("jlab_mcp.slurm.subprocess.run",
                   return_value=_completed(1, "", "scancel: error: Invalid job id 123")):
            cancel_job("123")  # no raise

    def test_failure_raises(self):
        """scancel failing must surface — silently 'cancelling' leaves an
        untracked job burning its allocation."""
        with patch("jlab_mcp.slurm.subprocess.run",
                   return_value=_completed(1, "", "Connection refused")):
            with pytest.raises(SlurmCommandError):
                cancel_job("123")


class TestParseSqueueOutput:
    def test_running_with_node(self):
        assert parse_squeue_output("RUNNING ravg1001") == ("RUNNING", "ravg1001")

    def test_pending_no_node(self):
        state, node = parse_squeue_output("PENDING")
        assert state == "PENDING"
        assert node == ""

    def test_completing(self):
        assert parse_squeue_output("COMPLETING ravg1002") == (
            "COMPLETING",
            "ravg1002",
        )

    def test_empty_output(self):
        state, node = parse_squeue_output("")
        assert state == ""
        assert node == ""

    def test_whitespace(self):
        assert parse_squeue_output("  RUNNING   ravg1001  ") == (
            "RUNNING",
            "ravg1001",
        )


class TestParseConnectionFile:
    def test_normal_file(self, tmp_path):
        conn_file = tmp_path / "test.conn"
        conn_file.write_text("HOSTNAME=ravg1001\nPORT=18500\nTOKEN=abc123\n")
        result = parse_connection_file(conn_file)
        assert result["HOSTNAME"] == "ravg1001"
        assert result["PORT"] == "18500"
        assert result["TOKEN"] == "abc123"

    def test_with_status(self, tmp_path):
        conn_file = tmp_path / "test.conn"
        conn_file.write_text(
            "HOSTNAME=ravg1001\nPORT=18500\nTOKEN=abc123\nSTATUS=starting\n"
        )
        result = parse_connection_file(conn_file)
        assert result["STATUS"] == "starting"
        assert result["HOSTNAME"] == "ravg1001"

    def test_extra_whitespace(self, tmp_path):
        conn_file = tmp_path / "test.conn"
        conn_file.write_text("  HOSTNAME = ravg1001 \n PORT = 18500 \n TOKEN = abc \n")
        result = parse_connection_file(conn_file)
        assert result["HOSTNAME"] == "ravg1001"
        assert result["PORT"] == "18500"
        assert result["TOKEN"] == "abc"

    def test_empty_lines_ignored(self, tmp_path):
        conn_file = tmp_path / "test.conn"
        conn_file.write_text("\nHOSTNAME=ravg1001\n\nPORT=18500\n\nTOKEN=abc123\n\n")
        result = parse_connection_file(conn_file)
        assert len(result) == 3


class TestAdvertisedHostExpr:
    @pytest.mark.parametrize("wildcard", ["0.0.0.0", "::", "", "  "])
    def test_wildcard_advertises_hostname(self, wildcard):
        assert _advertised_host_expr(wildcard) == "$(hostname)"

    def test_concrete_ip_advertises_itself(self):
        assert _advertised_host_expr("10.0.0.5") == "10.0.0.5"

    def test_concrete_ip_is_stripped(self):
        assert _advertised_host_expr("  10.0.0.5  ") == "10.0.0.5"


class TestRenderSlurmScript:
    def _render(self, bind_ip):
        with patch("jlab_mcp.config.SLURM_BIND_IP", bind_ip):
            return render_slurm_script(
                port=18500, token="abc123", connection_file="/tmp/x.conn"
            )

    def test_default_binds_all_interfaces(self):
        script = self._render("0.0.0.0")
        assert "--ip=0.0.0.0" in script
        # Connection file advertises the node hostname (shell-expanded in-job)
        assert "HOSTNAME=$(hostname)" in script

    def test_concrete_ip_binds_and_advertises_itself(self):
        script = self._render("10.0.0.5")
        assert "--ip=10.0.0.5" in script
        assert "HOSTNAME=10.0.0.5" in script
        assert "$(hostname)" not in script
