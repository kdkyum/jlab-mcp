import atexit
import base64
import logging
import signal
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


@dataclass
class Session:
    session_id: str
    job_id: str
    kernel_id: str
    jupyter_client: JupyterLabClient
    notebook_path: Path
    notebook_manager: NotebookManager
    hostname: str = ""
    connection_file: str = ""


# Global session store
sessions: dict[str, Session] = {}


def _cleanup_session_resources(session: Session) -> None:
    """Shut down kernel, cancel SLURM job, and remove connection file."""
    try:
        session.jupyter_client.shutdown_kernel(session.kernel_id)
    except Exception:
        pass
    try:
        cancel_job(session.job_id)
    except Exception:
        pass
    try:
        if session.connection_file:
            cleanup_connection_file(session.connection_file)
    except Exception:
        pass


def _cleanup_all_sessions():
    """Cancel all SLURM jobs and clean up on exit."""
    if not sessions:
        return
    logger.info(f"Cleaning up {len(sessions)} active session(s)...")
    for sid, session in list(sessions.items()):
        _cleanup_session_resources(session)
        logger.info(f"Cancelled SLURM job {session.job_id} (session {sid})")
    sessions.clear()


def _signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT by cleaning up sessions then exiting."""
    _cleanup_all_sessions()
    raise SystemExit(0)


atexit.register(_cleanup_all_sessions)
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


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


def _format_outputs(outputs: list[dict]) -> list | str:
    """Format kernel outputs into content blocks.

    Returns a list of text strings and Image objects when images are present,
    or a plain string when there are no images.
    """
    parts: list = []
    has_images = False
    for out in outputs:
        if out["type"] == "text":
            parts.append(out["content"])
        elif out["type"] == "image":
            has_images = True
            image_bytes = base64.b64decode(out["content"])
            parts.append(Image(data=image_bytes, format="png"))
        elif out["type"] == "error":
            tb = "\n".join(out.get("traceback", []))
            parts.append(
                f"Error: {out.get('ename', 'Error')}: "
                f"{out.get('evalue', '')}\n{tb}"
            )
    if not parts:
        return "(no output)"
    if has_images:
        return parts
    return "".join(parts)


def _setup_slurm_and_kernel() -> tuple[str, str, JupyterLabClient, str, str]:
    """Submit SLURM job, wait for it, connect, start kernel.

    Returns (job_id, kernel_id, client, hostname, connection_file).
    """
    job_id, conn_file, port, token = submit_job()
    logger.info(f"SLURM job {job_id} submitted (port={port}), waiting for it to start...")

    # Wait for SLURM job to start
    hostname = wait_for_job_running(job_id, timeout=300)
    logger.info(f"SLURM job {job_id} is RUNNING on {hostname}")

    # Wait for connection file
    conn_info = wait_for_connection_file(conn_file, timeout=120)
    actual_host = conn_info["HOSTNAME"]
    actual_port = int(conn_info["PORT"])
    actual_token = conn_info["TOKEN"]

    # Create client and wait for JupyterLab to be ready
    client = JupyterLabClient(actual_host, actual_port, actual_token)
    logger.info(f"Waiting for JupyterLab at {actual_host}:{actual_port}...")
    _wait_for_jupyter(client, timeout=120)
    logger.info(f"JupyterLab ready at {actual_host}:{actual_port}")

    # Start kernel
    kernel_id = client.start_kernel()
    # Give kernel a moment to initialize
    time.sleep(2)
    logger.info(f"Kernel {kernel_id} started")

    return job_id, kernel_id, client, actual_host, conn_file


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


def _register_session(
    job_id: str,
    kernel_id: str,
    client: JupyterLabClient,
    nb_path: Path,
    nb_manager: NotebookManager,
    hostname: str,
    conn_file: str,
) -> Session:
    """Create a Session, register it in the global store, and return it."""
    session_id = str(uuid.uuid4())[:8]
    session = Session(
        session_id=session_id,
        job_id=job_id,
        kernel_id=kernel_id,
        jupyter_client=client,
        notebook_path=nb_path,
        notebook_manager=nb_manager,
        hostname=hostname,
        connection_file=conn_file,
    )
    sessions[session_id] = session
    return session


def _get_session(session_id: str) -> Session:
    """Look up a session by ID or raise ValueError."""
    if session_id not in sessions:
        raise ValueError(f"Unknown session: {session_id}")
    return sessions[session_id]


@mcp.tool()
def start_new_session(experiment_name: str) -> dict:
    """Start a new session: submit SLURM job, start kernel, create notebook.

    Args:
        experiment_name: Name for the experiment/notebook.

    Returns:
        Dict with session_id, notebook_path, job_id, hostname.
    """
    job_id, kernel_id, client, hostname, conn_file = _setup_slurm_and_kernel()

    nb_manager = NotebookManager()
    nb_path = nb_manager.create_notebook(experiment_name, config.NOTEBOOK_DIR)

    session = _register_session(
        job_id, kernel_id, client, nb_path, nb_manager, hostname, conn_file
    )
    return {
        "session_id": session.session_id,
        "notebook_path": str(nb_path),
        "job_id": job_id,
        "hostname": hostname,
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
    job_id, kernel_id, client, hostname, conn_file = _setup_slurm_and_kernel()

    nb_manager = NotebookManager()

    def execute_fn(code: str) -> list[dict]:
        return client.execute_code(kernel_id, code)

    errors = nb_manager.restore_notebook(nb_path, execute_fn)

    session = _register_session(
        job_id, kernel_id, client, nb_path, nb_manager, hostname, conn_file
    )
    return {
        "session_id": session.session_id,
        "notebook_path": str(nb_path),
        "job_id": job_id,
        "hostname": hostname,
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
    job_id, kernel_id, client, hostname, conn_file = _setup_slurm_and_kernel()

    nb_manager = NotebookManager()
    forked_path = nb_manager.copy_notebook(nb_path, suffix="_continued")

    session = _register_session(
        job_id, kernel_id, client, forked_path, nb_manager, hostname, conn_file
    )
    return {
        "session_id": session.session_id,
        "notebook_path": str(forked_path),
        "original_notebook": str(nb_path),
        "job_id": job_id,
        "hostname": hostname,
    }


@mcp.tool()
def execute_code(session_id: str, code: str) -> str:
    """Execute code in the kernel and add cell to notebook.

    Args:
        session_id: Session identifier.
        code: Python code to execute.

    Returns:
        Formatted output string.
    """
    session = _get_session(session_id)
    outputs = session.jupyter_client.execute_code(session.kernel_id, code)
    session.notebook_manager.add_code_cell(
        session.notebook_path, code, outputs
    )
    return _format_outputs(outputs)


@mcp.tool()
def edit_cell(session_id: str, cell_index: int, code: str) -> str:
    """Edit an existing cell, re-execute it, and update outputs.

    Args:
        session_id: Session identifier.
        cell_index: Cell index (supports negative indexing).
        code: New code for the cell.

    Returns:
        Formatted output string.
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
    """Shutdown session: stop kernel and cancel SLURM job.

    Args:
        session_id: Session identifier.

    Returns:
        Confirmation message.
    """
    _get_session(session_id)
    session = sessions.pop(session_id)
    logger.info(f"Shutting down session {session_id} (job={session.job_id}, host={session.hostname})")
    _cleanup_session_resources(session)
    logger.info(f"Kernel {session.kernel_id} stopped, SLURM job {session.job_id} cancelled")

    return (
        f"Session {session_id} shutdown successfully. "
        f"Notebook saved at {session.notebook_path}"
    )


@mcp.resource("jlab-mcp://server/status")
def server_status() -> dict:
    """Get server status: active sessions, job states."""
    active = {}
    for sid, session in sessions.items():
        job_running = False
        try:
            job_running = is_job_running(session.job_id)
        except Exception:
            pass
        active[sid] = {
            "job_id": session.job_id,
            "kernel_id": session.kernel_id,
            "notebook_path": str(session.notebook_path),
            "hostname": session.hostname,
            "job_running": job_running,
        }
    return {
        "active_sessions": len(sessions),
        "sessions": active,
    }
