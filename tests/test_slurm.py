"""Unit tests for slurm.py — parsing logic only, no SLURM needed."""

import tempfile
from pathlib import Path

import pytest

from jlab_mcp.slurm import (
    parse_connection_file,
    parse_sbatch_output,
    parse_squeue_output,
)


class TestParseSbatchOutput:
    def test_normal_output(self):
        assert parse_sbatch_output("Submitted batch job 12345") == "12345"

    def test_large_job_id(self):
        assert parse_sbatch_output("Submitted batch job 9876543") == "9876543"

    def test_trailing_whitespace(self):
        assert parse_sbatch_output("Submitted batch job 12345\n") == "12345"

    def test_malformed_output(self):
        with pytest.raises(ValueError):
            parse_sbatch_output("Error: something went wrong")

    def test_empty_output(self):
        with pytest.raises(ValueError):
            parse_sbatch_output("")


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
