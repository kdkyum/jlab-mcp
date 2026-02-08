"""Integration tests — exercises all MCP tools against a live SLURM JupyterLab.

Requires:
  - SLURM cluster access (sbatch, squeue, scancel)
  - GPU partition available
  - Shared filesystem at /raven/ptmp/

Run with:
  uv run python -m pytest tests/test_tools.py -v -s --timeout=300
"""

import time

import pytest

from jlab_mcp.server import (
    add_markdown,
    edit_cell,
    execute_code,
    sessions,
    shutdown_session,
    start_new_session,
    start_session_continue_notebook,
    start_session_resume_notebook,
)
from jlab_mcp.slurm import get_job_state


@pytest.fixture(scope="module")
def session():
    """Create a shared session for most tests. Cleaned up at end."""
    result = start_new_session(experiment_name="test_integration")
    yield result
    # Cleanup
    try:
        shutdown_session(session_id=result["session_id"])
    except Exception:
        pass


class TestStartNewSession:
    def test_returns_required_fields(self, session):
        assert "session_id" in session
        assert "notebook_path" in session
        assert "job_id" in session
        assert "hostname" in session

    def test_session_registered(self, session):
        assert session["session_id"] in sessions

    def test_notebook_exists(self, session):
        from pathlib import Path

        assert Path(session["notebook_path"]).exists()


class TestExecuteCode:
    def test_print(self, session):
        result = execute_code(
            session_id=session["session_id"], code="print('hello world')"
        )
        assert "hello world" in result

    def test_gpu_available(self, session):
        result = execute_code(
            session_id=session["session_id"],
            code=(
                "import torch\n"
                "print(f'CUDA available: {torch.cuda.is_available()}')\n"
                "print(f'GPU: {torch.cuda.get_device_name(0)}')"
            ),
        )
        assert "CUDA available: True" in result
        assert "A100" in result

    def test_image_output(self, session):
        result = execute_code(
            session_id=session["session_id"],
            code=(
                "import matplotlib\n"
                "matplotlib.use('Agg')\n"
                "import matplotlib.pyplot as plt\n"
                "import numpy as np\n"
                "x = np.linspace(0, 10, 100)\n"
                "plt.figure()\n"
                "plt.plot(x, np.sin(x))\n"
                "plt.title('Test Plot')\n"
                "plt.show()"
            ),
        )
        assert "Image" in result or "base64" in result.lower()

    def test_error_handling(self, session):
        result = execute_code(session_id=session["session_id"], code="1/0")
        assert "ZeroDivisionError" in result

    def test_state_persistence(self, session):
        execute_code(session_id=session["session_id"], code="x_persist = 42")
        result = execute_code(
            session_id=session["session_id"], code="print(x_persist)"
        )
        assert "42" in result

    def test_large_output(self, session):
        result = execute_code(
            session_id=session["session_id"],
            code="for i in range(1000): print(f'line {i}')",
        )
        assert "line 999" in result

    def test_long_running(self, session):
        result = execute_code(
            session_id=session["session_id"],
            code="import time; time.sleep(3); print('done sleeping')",
        )
        assert "done sleeping" in result


class TestEditCell:
    def test_edit_cell(self, session):
        execute_code(session_id=session["session_id"], code="y_edit = 10")
        result = edit_cell(
            session_id=session["session_id"],
            cell_index=-1,
            code="y_edit = 20\nprint(y_edit)",
        )
        assert "20" in result

    def test_edit_cell_negative_index(self, session):
        execute_code(session_id=session["session_id"], code="a_neg = 1")
        execute_code(session_id=session["session_id"], code="b_neg = 2")
        result = edit_cell(
            session_id=session["session_id"],
            cell_index=-2,
            code="a_neg = 100\nprint(a_neg)",
        )
        assert "100" in result


class TestAddMarkdown:
    def test_add_markdown(self, session):
        result = add_markdown(
            session_id=session["session_id"],
            markdown="# Test Section\nThis is a test.",
        )
        assert "cell" in result.lower()


class TestResumeNotebook:
    def test_resume_restores_state(self):
        # Create session with state
        s1 = start_new_session(experiment_name="test_resume")
        execute_code(session_id=s1["session_id"], code="resume_var = 'hello'")
        nb_path = s1["notebook_path"]
        shutdown_session(session_id=s1["session_id"])

        # Resume
        s2 = start_session_resume_notebook(
            experiment_name="test_resume_2", notebook_path=nb_path
        )
        try:
            result = execute_code(
                session_id=s2["session_id"], code="print(resume_var)"
            )
            assert "hello" in result
        finally:
            shutdown_session(session_id=s2["session_id"])


class TestContinueNotebook:
    def test_continue_forks_notebook(self):
        s1 = start_new_session(experiment_name="test_continue")
        execute_code(session_id=s1["session_id"], code="cont_var = 99")
        nb_path = s1["notebook_path"]
        shutdown_session(session_id=s1["session_id"])

        s2 = start_session_continue_notebook(
            experiment_name="test_continue_2", notebook_path=nb_path
        )
        try:
            assert "_continued" in s2["notebook_path"]
            result = execute_code(
                session_id=s2["session_id"],
                code=(
                    "try:\n"
                    "    print(cont_var)\n"
                    "except NameError:\n"
                    "    print('NameError: variable not defined')"
                ),
            )
            assert "NameError" in result
        finally:
            shutdown_session(session_id=s2["session_id"])


class TestShutdownSession:
    def test_shutdown(self):
        s = start_new_session(experiment_name="test_shutdown")
        job_id = s["job_id"]
        result = shutdown_session(session_id=s["session_id"])
        assert "shutdown" in result.lower()
        assert s["session_id"] not in sessions
        # Give SLURM a moment to process cancellation
        time.sleep(2)
        state = get_job_state(job_id)
        assert state == "" or "CANCEL" in state


class TestServerStatus:
    def test_status_resource(self, session):
        from jlab_mcp.server import server_status

        status = server_status()
        assert "active_sessions" in status
        assert status["active_sessions"] >= 1
        assert session["session_id"] in status["sessions"]
