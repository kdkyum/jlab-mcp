"""Integration tests — exercises all MCP tools against a live SLURM JupyterLab.

Requires:
  - SLURM cluster access (sbatch, squeue, scancel)
  - GPU partition available
  - Shared filesystem
  - Run `jlab-mcp start` before running these tests

Run with:
  uv run python -m pytest tests/test_tools.py -v -s --timeout=300
"""

import time

import pytest

from jlab_mcp.server import (
    _cleanup_kernels,
    _server,
    add_markdown as _add_markdown,
    edit_cell as _edit_cell,
    execute_code as _execute_code,
    sessions,
    shutdown_session as _shutdown_session,
    start_new_session as _start_new_session,
    start_session_continue_notebook as _start_session_continue_notebook,
    start_session_resume_notebook as _start_session_resume_notebook,
)
from jlab_mcp.slurm import get_job_state

# FastMCP @mcp.tool() wraps functions into FunctionTool objects.
# Access the underlying callable via .fn
start_new_session = _start_new_session.fn
start_session_resume_notebook = _start_session_resume_notebook.fn
start_session_continue_notebook = _start_session_continue_notebook.fn
execute_code = _execute_code.fn
edit_cell = _edit_cell.fn
add_markdown = _add_markdown.fn
shutdown_session = _shutdown_session.fn


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


@pytest.fixture(scope="module", autouse=True)
def cleanup_kernels_after():
    """Cleanup all kernels after tests. SLURM job is left running."""
    yield
    _cleanup_kernels()


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
        # Result is always a list now
        assert any("hello world" in str(item) for item in result)

    def test_gpu_available(self, session):
        result = execute_code(
            session_id=session["session_id"],
            code=(
                "import torch\n"
                "print(f'CUDA available: {torch.cuda.is_available()}')\n"
                "if torch.cuda.is_available():\n"
                "    print(f'GPU: {torch.cuda.get_device_name(0)}')"
            ),
        )
        text = "".join(str(item) for item in result)
        assert "CUDA available: True" in text

    def test_image_output(self, session):
        from fastmcp.utilities.types import Image

        # First enable inline matplotlib in the kernel
        execute_code(
            session_id=session["session_id"],
            code="%matplotlib inline",
        )
        result = execute_code(
            session_id=session["session_id"],
            code=(
                "import matplotlib.pyplot as plt\n"
                "import numpy as np\n"
                "x = np.linspace(0, 10, 100)\n"
                "fig, ax = plt.subplots()\n"
                "ax.plot(x, np.sin(x))\n"
                "ax.set_title('Test Plot')\n"
                "plt.show()"
            ),
        )
        # When images are present, result is a list with Image objects
        assert isinstance(result, list)
        assert any(isinstance(item, Image) for item in result)

    def test_error_handling(self, session):
        result = execute_code(session_id=session["session_id"], code="1/0")
        text = "".join(str(item) for item in result)
        assert "ZeroDivisionError" in text

    def test_state_persistence(self, session):
        execute_code(session_id=session["session_id"], code="x_persist = 42")
        result = execute_code(
            session_id=session["session_id"], code="print(x_persist)"
        )
        text = "".join(str(item) for item in result)
        assert "42" in text

    def test_large_output(self, session):
        result = execute_code(
            session_id=session["session_id"],
            code="for i in range(1000): print(f'line {i}')",
        )
        text = "".join(str(item) for item in result)
        assert "line 999" in text

    def test_long_running(self, session):
        result = execute_code(
            session_id=session["session_id"],
            code="import time; time.sleep(3); print('done sleeping')",
        )
        text = "".join(str(item) for item in result)
        assert "done sleeping" in text


class TestEditCell:
    def test_edit_cell(self, session):
        execute_code(session_id=session["session_id"], code="y_edit = 10")
        result = edit_cell(
            session_id=session["session_id"],
            cell_index=-1,
            code="y_edit = 20\nprint(y_edit)",
        )
        text = "".join(str(item) for item in result)
        assert "20" in text

    def test_edit_cell_negative_index(self, session):
        execute_code(session_id=session["session_id"], code="a_neg = 1")
        execute_code(session_id=session["session_id"], code="b_neg = 2")
        result = edit_cell(
            session_id=session["session_id"],
            cell_index=-2,
            code="a_neg = 100\nprint(a_neg)",
        )
        text = "".join(str(item) for item in result)
        assert "100" in text


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

        # Resume — reuses the same SLURM job, new kernel
        s2 = start_session_resume_notebook(
            experiment_name="test_resume_2", notebook_path=nb_path
        )
        try:
            result = execute_code(
                session_id=s2["session_id"], code="print(resume_var)"
            )
            text = "".join(str(item) for item in result)
            assert "hello" in text
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
            text = "".join(str(item) for item in result)
            assert "NameError" in text
        finally:
            shutdown_session(session_id=s2["session_id"])


class TestShutdownSession:
    def test_shutdown_only_kills_kernel(self):
        """Shutdown session kills the kernel but keeps the SLURM job alive."""
        s = start_new_session(experiment_name="test_shutdown")
        job_id = s["job_id"]
        result = shutdown_session(session_id=s["session_id"])
        assert "shutdown" in result.lower()
        assert s["session_id"] not in sessions
        # SLURM job should still be running (managed by user)
        time.sleep(2)
        state = get_job_state(job_id)
        assert state == "RUNNING"


class TestSharedServer:
    def test_multiple_sessions_same_job(self):
        """Multiple sessions share the same SLURM job."""
        s1 = start_new_session(experiment_name="test_shared_1")
        s2 = start_new_session(experiment_name="test_shared_2")
        try:
            assert s1["job_id"] == s2["job_id"]
            assert s1["hostname"] == s2["hostname"]
            # Both sessions work independently
            execute_code(session_id=s1["session_id"], code="shared_a = 1")
            execute_code(session_id=s2["session_id"], code="shared_b = 2")
            r1 = execute_code(
                session_id=s1["session_id"], code="print(shared_a)"
            )
            r2 = execute_code(
                session_id=s2["session_id"], code="print(shared_b)"
            )
            assert "1" in "".join(str(item) for item in r1)
            assert "2" in "".join(str(item) for item in r2)
            # Session 1 shouldn't see session 2's variables
            r3 = execute_code(
                session_id=s1["session_id"],
                code=(
                    "try:\n"
                    "    print(shared_b)\n"
                    "except NameError:\n"
                    "    print('isolated')"
                ),
            )
            assert "isolated" in "".join(str(item) for item in r3)
        finally:
            shutdown_session(session_id=s1["session_id"])
            shutdown_session(session_id=s2["session_id"])


class TestServerStatus:
    def test_status_resource(self, session):
        from jlab_mcp.server import server_status as _server_status

        status = _server_status.fn()
        assert "active_sessions" in status
        assert status["active_sessions"] >= 1
        assert session["session_id"] in status["sessions"]
        assert "server" in status
        assert status["server"]["healthy"] is True
