"""Integration tests — exercises all MCP tools against a live JupyterLab.

Works in both modes (the mode is whatever `jlab-mcp start` was run in):
  SLURM: run `jlab-mcp start` first (needs sbatch + a GPU partition)
  local: JLAB_MCP_RUN_MODE=local jlab-mcp start

Run with:
  uv run python -m pytest tests/test_tools.py -v -s --timeout=600

The execution tools are async and require a FastMCP Context — tests drive
them through asyncio.run() with a stub context (report_progress no-op).
"""

import asyncio
import time
from pathlib import Path

import pytest

from jlab_mcp import config
from jlab_mcp.server import (
    _cleanup_kernels,
    add_markdown as _add_markdown,
    delete_cell as _delete_cell,
    edit_cell as _edit_cell,
    edit_markdown as _edit_markdown,
    execute_code as _execute_code,
    ping as _ping,
    run_cell as _run_cell,
    server_status as _server_status,
    sessions,
    shutdown_session as _shutdown_session,
    start_new_notebook as _start_new_notebook,
    start_notebook as _start_notebook,
)

# FastMCP @mcp.tool() wraps functions into FunctionTool objects.
# Access the underlying callable via .fn
start_new_notebook = _start_new_notebook.fn
start_notebook = _start_notebook.fn
execute_code = _execute_code.fn
edit_cell = _edit_cell.fn
edit_markdown = _edit_markdown.fn
delete_cell = _delete_cell.fn
run_cell = _run_cell.fn
add_markdown = _add_markdown.fn
shutdown_session = _shutdown_session.fn
ping = _ping.fn
server_status = _server_status.fn


class _CtxStub:
    """Minimal stand-in for fastmcp.Context (only report_progress is used)."""

    async def report_progress(self, progress=None, total=None):
        pass


CTX = _CtxStub()


def run(coro):
    return asyncio.run(coro)


def execute(session_id, code, cell_index=-1):
    return run(execute_code(
        session_id=session_id, code=code, cell_index=cell_index, ctx=CTX
    ))


def text_of(result):
    return "".join(str(item) for item in result)


@pytest.fixture
def session():
    """A fresh session per test.

    start_new_notebook shuts down ALL existing kernels/sessions (single
    active session contract), so a shared module-scoped session would be
    silently killed by any test that starts its own notebook.
    """
    result = start_new_notebook(experiment_name="test_integration")
    yield result
    try:
        shutdown_session(session_id=result["session_id"])
    except Exception:
        pass


@pytest.fixture(scope="module", autouse=True)
def cleanup_kernels_after():
    """Cleanup all kernels after tests. The JupyterLab server stays alive."""
    yield
    _cleanup_kernels()


class TestStartNewNotebook:
    def test_returns_required_fields(self, session):
        assert "session_id" in session
        assert "notebook_path" in session
        assert "job_id" in session
        assert "hostname" in session

    def test_session_registered(self, session):
        assert session["session_id"] in sessions

    def test_notebook_exists(self, session):
        assert Path(session["notebook_path"]).exists()

    def test_same_name_never_overwrites(self):
        """Reusing an experiment_name must not clobber the old notebook."""
        s1 = start_new_notebook(experiment_name="test_overwrite_check")
        execute(s1["session_id"], "kept = 1")
        p1 = Path(s1["notebook_path"])

        s2 = start_new_notebook(experiment_name="test_overwrite_check")
        p2 = Path(s2["notebook_path"])
        try:
            assert p1 != p2
            assert p1.exists() and p2.exists()
            # Original notebook still has its cell
            from jlab_mcp.notebook import NotebookManager

            assert NotebookManager().get_cell_count(p1) == 1
        finally:
            shutdown_session(session_id=s2["session_id"])


class TestExecuteCode:
    def test_print(self, session):
        result = execute(session["session_id"], "print('hello world')")
        assert any("hello world" in str(item) for item in result)

    @pytest.mark.skipif(
        config.RUN_MODE == "local",
        reason="GPU only expected on SLURM compute nodes",
    )
    def test_gpu_available(self, session):
        result = execute(
            session["session_id"],
            (
                "import torch\n"
                "print(f'CUDA available: {torch.cuda.is_available()}')\n"
                "if torch.cuda.is_available():\n"
                "    print(f'GPU: {torch.cuda.get_device_name(0)}')"
            ),
        )
        assert "CUDA available: True" in text_of(result)

    def test_image_output(self, session):
        from fastmcp.utilities.types import Image

        execute(session["session_id"], "%matplotlib inline")
        result = execute(
            session["session_id"],
            (
                "import matplotlib.pyplot as plt\n"
                "import numpy as np\n"
                "x = np.linspace(0, 10, 100)\n"
                "fig, ax = plt.subplots()\n"
                "ax.plot(x, np.sin(x))\n"
                "ax.set_title('Test Plot')\n"
                "plt.show()"
            ),
        )
        assert isinstance(result, list)
        assert any(isinstance(item, Image) for item in result)

    def test_error_handling(self, session):
        result = execute(session["session_id"], "1/0")
        assert "ZeroDivisionError" in text_of(result)

    def test_state_persistence(self, session):
        execute(session["session_id"], "x_persist = 42")
        result = execute(session["session_id"], "print(x_persist)")
        assert "42" in text_of(result)

    def test_large_output(self, session):
        result = execute(
            session["session_id"],
            "for i in range(1000): print(f'line {i}')",
        )
        assert "line 999" in text_of(result)

    def test_long_running(self, session):
        result = execute(
            session["session_id"],
            "import time; time.sleep(3); print('done sleeping')",
        )
        assert "done sleeping" in text_of(result)

    def test_unknown_session(self):
        with pytest.raises(ValueError, match="Unknown session"):
            execute("nonexistent", "print(1)")


class TestEditCell:
    def test_edit_cell_no_execution(self, session):
        """edit_cell updates source and clears outputs, does not execute."""
        execute(session["session_id"], "y_edit = 10")
        result = edit_cell(
            session_id=session["session_id"],
            cell_index=-1,
            code="y_edit = 20\nprint(y_edit)",
        )
        assert "updated" in result.lower()
        assert "not executed" in result.lower()

    def test_edit_then_run(self, session):
        """edit_cell + run_cell produces correct output."""
        execute(session["session_id"], "placeholder = 1")
        edit_cell(
            session_id=session["session_id"],
            cell_index=-1,
            code="placeholder = 99\nprint(placeholder)",
        )
        result = run(run_cell(
            session_id=session["session_id"], cell_index=-1, ctx=CTX
        ))
        assert "99" in text_of(result)

    def test_edit_markdown_cell_rejected(self, session):
        """edit_cell must refuse markdown cells (outputs would corrupt them)."""
        add_markdown(session_id=session["session_id"], markdown="# heading")
        with pytest.raises(ValueError, match="markdown"):
            edit_cell(
                session_id=session["session_id"], cell_index=-1, code="x = 1"
            )


class TestRunCell:
    def test_run_existing_cell(self, session):
        """run_cell executes an existing cell without modifying source."""
        execute(session["session_id"], "print('run_cell_test')")
        result = run(run_cell(
            session_id=session["session_id"], cell_index=-1, ctx=CTX
        ))
        assert "run_cell_test" in text_of(result)

    def test_run_cell_negative_index(self, session):
        execute(session["session_id"], "rc_a = 1")
        execute(session["session_id"], "print('rc_last')")
        result = run(run_cell(
            session_id=session["session_id"], cell_index=-1, ctx=CTX
        ))
        assert "rc_last" in text_of(result)

    def test_run_markdown_cell_skipped(self, session):
        add_markdown(session_id=session["session_id"], markdown="# md")
        result = run(run_cell(
            session_id=session["session_id"], cell_index=-1, ctx=CTX
        ))
        assert "skipped" in text_of(result).lower()


class TestMarkdownAndDelete:
    def test_add_markdown(self, session):
        result = add_markdown(
            session_id=session["session_id"],
            markdown="# Test Section\nThis is a test.",
        )
        assert "cell" in result.lower()

    def test_edit_markdown(self, session):
        add_markdown(session_id=session["session_id"], markdown="# before")
        result = edit_markdown(
            session_id=session["session_id"], cell_index=-1, markdown="# after"
        )
        assert "updated" in result.lower()

    def test_delete_cell(self, session):
        from jlab_mcp.notebook import NotebookManager

        execute(session["session_id"], "to_delete = 1")
        count_before = NotebookManager().get_cell_count(session["notebook_path"])
        result = delete_cell(session_id=session["session_id"], cell_index=-1)
        assert "deleted" in result.lower()
        count_after = NotebookManager().get_cell_count(session["notebook_path"])
        assert count_after == count_before - 1


class TestExecuteCodeCellIndex:
    def test_insert_at_beginning(self, session):
        """execute_code with cell_index=0 inserts at the beginning."""
        from jlab_mcp.notebook import NotebookManager

        nb_path = session["notebook_path"]
        nb_manager = NotebookManager()

        count_before = nb_manager.get_cell_count(nb_path)
        execute(session["session_id"], "print('inserted_first')", cell_index=0)
        nb = nb_manager.get_notebook(nb_path)
        assert nb.cells[0].source == "print('inserted_first')"
        assert nb_manager.get_cell_count(nb_path) == count_before + 1

    def test_outputs_follow_cell_when_index_shifts(self, session):
        """Outputs land on the executed cell even if cells are inserted
        at lower indices during/after execution (cell-id tracking)."""
        from jlab_mcp.notebook import NotebookManager

        nb_path = session["notebook_path"]
        execute(session["session_id"], "print('tracked_output')")
        nb = NotebookManager().get_notebook(nb_path)
        # The output must be attached to the cell with the matching source
        for cell in nb.cells:
            if cell.source == "print('tracked_output')":
                texts = "".join(
                    o.get("text", "") for o in cell.outputs
                )
                assert "tracked_output" in texts
                break
        else:
            pytest.fail("executed cell not found in notebook")


class TestStartNotebook:
    def test_reuses_live_session(self, session):
        """start_notebook on a notebook with a live kernel returns the SAME
        session with state preserved (documented kernel-reuse contract)."""
        execute(session["session_id"], "reuse_var = 7")
        s2 = start_notebook(notebook_path=session["notebook_path"])
        assert s2["session_id"] == session["session_id"]
        result = execute(s2["session_id"], "print(reuse_var)")
        assert "7" in text_of(result)

    def test_attaches_to_existing_notebook(self):
        """After shutdown, start_notebook opens the notebook on a fresh kernel."""
        s1 = start_new_notebook(experiment_name="test_start_nb")
        execute(s1["session_id"], "start_nb_var = 99")
        nb_path = s1["notebook_path"]
        shutdown_session(session_id=s1["session_id"])

        s2 = start_notebook(notebook_path=nb_path)
        try:
            # Same notebook path, no fork
            assert s2["notebook_path"] == nb_path
            # Fresh kernel — old variables are NOT available
            result = execute(
                s2["session_id"],
                (
                    "try:\n"
                    "    print(start_nb_var)\n"
                    "except NameError:\n"
                    "    print('NameError: variable not defined')"
                ),
            )
            assert "NameError" in text_of(result)
        finally:
            shutdown_session(session_id=s2["session_id"])

    def test_returns_cells(self):
        """start_notebook returns code cells in the response."""
        s1 = start_new_notebook(experiment_name="test_cells_return")
        execute(s1["session_id"], "x = 1")
        execute(s1["session_id"], "y = 2")
        nb_path = s1["notebook_path"]
        shutdown_session(session_id=s1["session_id"])

        s2 = start_notebook(notebook_path=nb_path)
        try:
            assert "cells" in s2
            cells = s2["cells"]
            assert len(cells) >= 2
            sources = [c["source"] for c in cells]
            assert "x = 1" in sources
            assert "y = 2" in sources
        finally:
            shutdown_session(session_id=s2["session_id"])

    def test_rejects_path_outside_notebook_dir(self):
        with pytest.raises((ValueError, FileNotFoundError)):
            start_notebook(notebook_path="/etc/passwd")


class TestSingleSessionContract:
    def test_new_notebook_shuts_down_previous(self):
        """start_new_notebook kills all existing kernels and sessions —
        exactly one active session at a time."""
        sa = start_new_notebook(experiment_name="test_auto_a")
        sb = start_new_notebook(experiment_name="test_auto_b")
        try:
            assert sa["job_id"] == sb["job_id"]
            assert sa["hostname"] == sb["hostname"]
            assert sa["session_id"] not in sessions
            assert sb["session_id"] in sessions
            result = execute(sb["session_id"], "print('only_session')")
            assert "only_session" in text_of(result)
        finally:
            shutdown_session(session_id=sb["session_id"])


class TestShutdownSession:
    def test_shutdown_removes_session(self):
        s = start_new_notebook(experiment_name="test_shutdown")
        result = shutdown_session(session_id=s["session_id"])
        assert "shutdown" in result.lower()
        assert s["session_id"] not in sessions

    def test_server_survives_shutdown(self):
        """Shutdown stops the kernel but the JupyterLab server stays up
        (SLURM job / local process is managed by `jlab-mcp` CLI, not tools)."""
        s = start_new_notebook(experiment_name="test_shutdown_srv")
        job_id = s["job_id"]
        shutdown_session(session_id=s["session_id"])
        time.sleep(2)
        if job_id:  # SLURM mode
            from jlab_mcp.slurm import get_job_state

            assert get_job_state(job_id) == "RUNNING"
        status = run(ping())
        assert status["status"] == "ok"


class TestPingAndStatus:
    def test_ping(self, session):
        result = run(ping())
        assert result["status"] == "ok"
        assert result["healthy"] is True
        assert result["active_sessions"] >= 1

    def test_status_resource(self, session):
        status = run(server_status())
        assert "active_sessions" in status
        assert status["active_sessions"] == 1
        assert session["session_id"] in status["sessions"]
        assert "server" in status
        assert status["server"]["healthy"] is True
