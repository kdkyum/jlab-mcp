import atexit
import base64
import logging
import signal
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.utilities.types import Image

from jlab_mcp import config
from jlab_mcp.jupyter_client import JupyterLabClient
from jlab_mcp.notebook import NotebookManager
from jlab_mcp.slurm import (
    cancel_job,
    cleanup_connection_file,
    is_job_running,
    submit_job,
    wait_for_connection_file,
    wait_for_job_running,
)

logger = logging.getLogger("jlab-mcp")

mcp = FastMCP("jlab-mcp")


# ---------------------------------------------------------------------------
# Shared JupyterLab server (one SLURM job, many kernels)
# ---------------------------------------------------------------------------

@dataclass
class JupyterServer:
    """A JupyterLab instance running on a SLURM compute node."""
    job_id: str
    client: JupyterLabClient
    hostname: str
    connection_file: str


@dataclass
class Session:
    session_id: str
    kernel_id: str
    jupyter_client: JupyterLabClient
    notebook_path: Path
    notebook_manager: NotebookManager


# Global state
sessions: dict[str, Session] = {}
_server: JupyterServer | None = None
_server_lock = threading.Lock()
_server_error: Exception | None = None


def _start_jupyter_server() -> JupyterServer:
    """Submit SLURM job and wait for JupyterLab to be ready."""
    job_id, conn_file, port, token = submit_job()
    logger.info(f"SLURM job {job_id} submitted (port={port}), waiting for compute node...")

    hostname = wait_for_job_running(job_id, timeout=300)
    logger.info(f"SLURM job {job_id} running on {hostname}")

    conn_info = wait_for_connection_file(conn_file, timeout=120)
    client = JupyterLabClient(
        conn_info["HOSTNAME"], int(conn_info["PORT"]), conn_info["TOKEN"]
    )

    logger.info(f"Waiting for JupyterLab at {client.base_url}...")
    _wait_for_jupyter(client, timeout=120)
    logger.info(f"JupyterLab ready at {client.base_url}")

    return JupyterServer(
        job_id=job_id, client=client, hostname=hostname, connection_file=conn_file
    )


def _get_or_start_server() -> JupyterServer:
    """Get the shared JupyterLab server, starting one if needed.

    If the existing server's SLURM job has terminated, starts a new one.
    Thread-safe: only one thread can start the server at a time.
    """
    global _server, _server_error
    with _server_lock:
        # Check if existing server is still alive
        if _server is not None:
            if is_job_running(_server.job_id):
                return _server
            # Server died (walltime, preemption, etc.)
            logger.warning(
                f"JupyterLab server (job {_server.job_id}) terminated. "
                f"All existing sessions are lost. Starting a new server..."
            )
            _cleanup_server_resources()
            # Invalidate all sessions that used the dead server
            sessions.clear()

        # Start a new server
        _server_error = None
        try:
            _server = _start_jupyter_server()
            return _server
        except Exception as e:
            _server_error = e
            raise


def start_jupyter_background():
    """Start SLURM job submission in a background thread.

    Called from __main__.py so the job starts immediately when Claude Code
    launches the MCP server, reducing wait time on first tool call.
    """
    def _bg():
        try:
            _get_or_start_server()
        except Exception as e:
            logger.error(f"Background JupyterLab startup failed: {e}")

    threading.Thread(target=_bg, daemon=True).start()


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def _cleanup_server_resources():
    """Cancel the shared SLURM job and remove connection file."""
    global _server
    if _server is None:
        return
    try:
        cancel_job(_server.job_id)
    except Exception:
        pass
    try:
        cleanup_connection_file(_server.connection_file)
    except Exception:
        pass
    _server = None


def _cleanup_all():
    """Shutdown all kernels and cancel the shared SLURM job."""
    # Shutdown individual kernels
    for sid, session in list(sessions.items()):
        try:
            session.jupyter_client.shutdown_kernel(session.kernel_id)
        except Exception:
            pass
    sessions.clear()
    # Cancel the shared SLURM job
    _cleanup_server_resources()
    logger.info("All sessions and SLURM job cleaned up")


def _signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT by cleaning up then exiting."""
    _cleanup_all()
    raise SystemExit(0)


atexit.register(_cleanup_all)
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_notebook_path(notebook_path: str) -> Path:
    """Validate that notebook_path is within NOTEBOOK_DIR to prevent traversal."""
    nb_path = Path(notebook_path).resolve()
    notebook_dir = config.NOTEBOOK_DIR.resolve()
    if not str(nb_path).startswith(str(notebook_dir)):
        raise ValueError(
            f"Notebook path must be within {notebook_dir}, "
            f"got: {notebook_path}"
        )
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")
    return nb_path


def _format_outputs(outputs: list[dict]) -> list:
    """Format kernel outputs into a list of content blocks.

    Returns a list of text strings and Image objects.  FastMCP converts
    each element into a separate MCP content block (TextContent or
    ImageContent), so this must always return a list — not a bare string.
    """
    parts: list = []
    for out in outputs:
        if out["type"] == "text":
            parts.append(out["content"])
        elif out["type"] == "image":
            image_bytes = base64.b64decode(out["content"])
            parts.append(Image(data=image_bytes, format="png"))
        elif out["type"] == "error":
            tb = "\n".join(out.get("traceback", []))
            parts.append(
                f"Error: {out.get('ename', 'Error')}: "
                f"{out.get('evalue', '')}\n{tb}"
            )
    if not parts:
        return ["(no output)"]
    return parts


def _wait_for_jupyter(client: JupyterLabClient, timeout: int = 120) -> None:
    """Poll until JupyterLab health check passes."""
    start = time.time()
    while time.time() - start < timeout:
        if client.health_check():
            return
        time.sleep(3)
    raise TimeoutError(
        f"JupyterLab not responding at {client.base_url} after {timeout}s"
    )


def _start_kernel(server: JupyterServer) -> str:
    """Start a new kernel on the shared JupyterLab server."""
    kernel_id = server.client.start_kernel()
    time.sleep(2)
    logger.info(f"Kernel {kernel_id} started on {server.hostname}")
    return kernel_id


def _register_session(
    kernel_id: str,
    client: JupyterLabClient,
    nb_path: Path,
    nb_manager: NotebookManager,
) -> Session:
    """Create a Session, register it in the global store, and return it."""
    session_id = str(uuid.uuid4())[:8]
    session = Session(
        session_id=session_id,
        kernel_id=kernel_id,
        jupyter_client=client,
        notebook_path=nb_path,
        notebook_manager=nb_manager,
    )
    sessions[session_id] = session
    return session


def _get_session(session_id: str) -> Session:
    """Look up a session by ID or raise ValueError."""
    if session_id not in sessions:
        raise ValueError(f"Unknown session: {session_id}")
    return sessions[session_id]


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def start_new_session(experiment_name: str) -> dict:
    """Start a new session: start kernel, create notebook.

    Uses the shared JupyterLab server (auto-started on first call).

    Args:
        experiment_name: Name for the experiment/notebook.

    Returns:
        Dict with session_id, notebook_path, job_id, hostname.
    """
    server = _get_or_start_server()
    kernel_id = _start_kernel(server)

    nb_manager = NotebookManager()
    nb_path = nb_manager.create_notebook(experiment_name, config.NOTEBOOK_DIR)

    session = _register_session(kernel_id, server.client, nb_path, nb_manager)
    return {
        "session_id": session.session_id,
        "notebook_path": str(nb_path),
        "job_id": server.job_id,
        "hostname": server.hostname,
    }


@mcp.tool()
def start_session_resume_notebook(
    experiment_name: str, notebook_path: str
) -> dict:
    """Resume a notebook: re-execute all cells to restore kernel state.

    Args:
        experiment_name: Name for this session.
        notebook_path: Path to existing notebook to resume.

    Returns:
        Dict with session_id, notebook_path, job_id, hostname, errors.
    """
    nb_path = _validate_notebook_path(notebook_path)
    server = _get_or_start_server()
    kernel_id = _start_kernel(server)

    nb_manager = NotebookManager()

    def execute_fn(code: str) -> list[dict]:
        return server.client.execute_code(kernel_id, code)

    errors = nb_manager.restore_notebook(nb_path, execute_fn)

    session = _register_session(kernel_id, server.client, nb_path, nb_manager)
    return {
        "session_id": session.session_id,
        "notebook_path": str(nb_path),
        "job_id": server.job_id,
        "hostname": server.hostname,
        "restored_with_errors": errors if errors else None,
    }


@mcp.tool()
def start_session_continue_notebook(
    experiment_name: str, notebook_path: str
) -> dict:
    """Continue a notebook: fork it with fresh kernel (no re-execution).

    Args:
        experiment_name: Name for this session.
        notebook_path: Path to existing notebook to fork.

    Returns:
        Dict with session_id, notebook_path (forked), job_id, hostname.
    """
    nb_path = _validate_notebook_path(notebook_path)
    server = _get_or_start_server()
    kernel_id = _start_kernel(server)

    nb_manager = NotebookManager()
    forked_path = nb_manager.copy_notebook(nb_path, suffix="_continued")

    session = _register_session(
        kernel_id, server.client, forked_path, nb_manager
    )
    return {
        "session_id": session.session_id,
        "notebook_path": str(forked_path),
        "original_notebook": str(nb_path),
        "job_id": server.job_id,
        "hostname": server.hostname,
    }


@mcp.tool()
def execute_code(session_id: str, code: str) -> list:
    """Execute code in the kernel and add cell to notebook.

    Args:
        session_id: Session identifier.
        code: Python code to execute.

    Returns:
        List of text strings and Image objects.
    """
    session = _get_session(session_id)
    outputs = session.jupyter_client.execute_code(session.kernel_id, code)
    session.notebook_manager.add_code_cell(
        session.notebook_path, code, outputs
    )
    return _format_outputs(outputs)


@mcp.tool()
def edit_cell(session_id: str, cell_index: int, code: str) -> list:
    """Edit an existing cell, re-execute it, and update outputs.

    Args:
        session_id: Session identifier.
        cell_index: Cell index (supports negative indexing).
        code: New code for the cell.

    Returns:
        List of text strings and Image objects.
    """
    session = _get_session(session_id)
    outputs = session.jupyter_client.execute_code(session.kernel_id, code)
    session.notebook_manager.edit_cell(
        session.notebook_path, cell_index, code, outputs
    )
    return _format_outputs(outputs)


@mcp.tool()
def add_markdown(session_id: str, markdown: str) -> str:
    """Add a markdown cell to the notebook.

    Args:
        session_id: Session identifier.
        markdown: Markdown content.

    Returns:
        Confirmation with cell index.
    """
    session = _get_session(session_id)
    cell_idx = session.notebook_manager.add_markdown_cell(
        session.notebook_path, markdown
    )
    return f"Added markdown cell at index {cell_idx}"


@mcp.tool()
def shutdown_session(session_id: str) -> str:
    """Shutdown session: stop kernel. The SLURM job stays alive for other sessions.

    Args:
        session_id: Session identifier.

    Returns:
        Confirmation message.
    """
    _get_session(session_id)
    session = sessions.pop(session_id)
    logger.info(f"Shutting down session {session_id} (kernel={session.kernel_id})")
    try:
        session.jupyter_client.shutdown_kernel(session.kernel_id)
    except Exception:
        pass
    logger.info(f"Kernel {session.kernel_id} stopped")

    return (
        f"Session {session_id} shutdown successfully. "
        f"Notebook saved at {session.notebook_path}"
    )


# ---------------------------------------------------------------------------
# MCP Resource
# ---------------------------------------------------------------------------

@mcp.resource("jlab-mcp://server/status")
def server_status() -> dict:
    """Get server status: shared SLURM job, active sessions."""
    server_info = {}
    if _server is not None:
        job_running = False
        try:
            job_running = is_job_running(_server.job_id)
        except Exception:
            pass
        server_info = {
            "job_id": _server.job_id,
            "hostname": _server.hostname,
            "url": _server.client.base_url,
            "job_running": job_running,
        }

    session_info = {}
    for sid, session in sessions.items():
        session_info[sid] = {
            "kernel_id": session.kernel_id,
            "notebook_path": str(session.notebook_path),
        }

    return {
        "server": server_info if server_info else "not started",
        "active_sessions": len(sessions),
        "sessions": session_info,
    }
