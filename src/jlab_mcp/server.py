import base64
import json
import logging
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

logger = logging.getLogger("jlab-mcp")

mcp = FastMCP("jlab-mcp")

_STATUS_FILE = config.STATUS_FILE


# ---------------------------------------------------------------------------
# Shared JupyterLab server (managed externally via `jlab-mcp start`)
# ---------------------------------------------------------------------------

@dataclass
class JupyterServer:
    """A JupyterLab instance running on a SLURM compute node or local subprocess."""
    job_id: str
    client: JupyterLabClient
    hostname: str


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


def _read_status_file() -> dict:
    """Read the server-status file written by `jlab-mcp start`."""
    if not _STATUS_FILE.exists():
        return {}
    info = {}
    for line in _STATUS_FILE.read_text().strip().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k] = v
    return info


def _connect_to_server() -> JupyterServer:
    """Connect to JupyterLab using connection info from the status file."""
    info = _read_status_file()
    state = info.get("STATE")

    if not info or state is None:
        raise RuntimeError(
            "No JupyterLab server running. Start one with: jlab-mcp start"
        )
    if state != "ready":
        raise RuntimeError(
            f"JupyterLab not ready (state={state}). "
            "Check progress with: jlab-mcp wait"
        )

    hostname = info["HOSTNAME"]
    port = int(info["PORT"])
    token = info["TOKEN"]
    job_id = info.get("JOB_ID", "")

    client = JupyterLabClient(hostname, port, token)
    if not client.health_check():
        raise RuntimeError(
            "JupyterLab not responding. Restart with: jlab-mcp start"
        )

    return JupyterServer(job_id=job_id, client=client, hostname=hostname)


def _get_or_start_server() -> JupyterServer:
    """Get the shared JupyterLab server, connecting on first call.

    Reads connection info from the status file written by `jlab-mcp start`.
    If the server stops responding, clears cached state and reconnects.
    """
    global _server
    with _server_lock:
        if _server is not None:
            if _server.client.health_check():
                return _server
            logger.warning("JupyterLab server not responding, reconnecting...")
            _server = None
            sessions.clear()

        _server = _connect_to_server()
        logger.info(f"Connected to JupyterLab on {_server.hostname}")
        return _server


# ---------------------------------------------------------------------------
# Cleanup (kernels only — SLURM job is managed by user)
# ---------------------------------------------------------------------------

def _cleanup_kernels():
    """Shutdown all kernels. Does NOT cancel the SLURM job."""
    for sid, session in list(sessions.items()):
        try:
            session.jupyter_client.shutdown_kernel(session.kernel_id)
        except Exception:
            pass
    sessions.clear()
    logger.info("All sessions cleaned up")


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

    Uses the shared JupyterLab server (must be started with `jlab-mcp start`).

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


@mcp.tool(output_schema=None)
def execute_code(session_id: str, code: str) -> list:
    """Execute code in the kernel and add cell to notebook.

    Args:
        session_id: Session identifier.
        code: Python code to execute.

    Returns:
        List of text strings and Image objects.
    """
    session = _get_session(session_id)
    session.jupyter_client.interrupt_kernel(session.kernel_id)
    outputs = session.jupyter_client.execute_code(session.kernel_id, code)
    session.notebook_manager.add_code_cell(
        session.notebook_path, code, outputs
    )
    return _format_outputs(outputs)


@mcp.tool(output_schema=None)
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
    session.jupyter_client.interrupt_kernel(session.kernel_id)
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

    IMPORTANT: Do NOT call this automatically after finishing work.
    The user may want to continue experiments on the same notebook later.
    Only call this when the user explicitly asks to shutdown, or when
    starting a new session for a different notebook.

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


@mcp.tool()
def interrupt_kernel(session_id: str) -> str:
    """Interrupt the running kernel to stop execution.

    Use this when code is taking too long and you want to cancel it
    without shutting down the session. Safe to call on an idle kernel.

    Args:
        session_id: Session identifier.

    Returns:
        Confirmation message.
    """
    session = _get_session(session_id)
    session.jupyter_client.interrupt_kernel(session.kernel_id)
    return f"Kernel interrupted for session {session_id}"


@mcp.tool()
def ping() -> dict:
    """Check if JupyterLab is reachable and responding.

    Lightweight health check — no kernel needed. Use this to verify
    the connection before starting a session.

    Returns:
        Dict with status, hostname, url, active sessions count.
    """
    info = _read_status_file()
    state = info.get("STATE")

    if not info or state is None:
        return {"status": "no_server", "message": "Run jlab-mcp start"}
    if state != "ready":
        return {"status": state, "message": f"Server not ready (state={state})"}

    hostname = info.get("HOSTNAME", "")
    port = info.get("PORT", "")
    token = info.get("TOKEN", "")

    if not (hostname and port and token):
        return {"status": "error", "message": "Incomplete status file"}

    client = JupyterLabClient(hostname, int(port), token)
    healthy = client.health_check()

    return {
        "status": "ok" if healthy else "unreachable",
        "hostname": hostname,
        "url": f"http://{hostname}:{port}",
        "healthy": healthy,
        "active_sessions": len(sessions),
    }


@mcp.tool()
def check_resources(session_id: str) -> dict:
    """Check compute resource usage: CPU, memory, and GPU.

    Runs lightweight commands on the session's kernel to report current
    resource utilization. Use this to monitor memory pressure or GPU usage.

    Args:
        session_id: Session identifier.

    Returns:
        Dict with cpu, memory, and gpu resource information.
    """
    session = _get_session(session_id)

    code = """\
import json as _json

_result = {}

# --- CPU ---
try:
    import os
    _nproc = os.cpu_count()
    _load1, _load5, _load15 = os.getloadavg()
    _result["cpu"] = {
        "count": _nproc,
        "load_1m": round(_load1, 2),
        "load_5m": round(_load5, 2),
        "load_15m": round(_load15, 2),
    }
except Exception as _e:
    _result["cpu"] = {"error": str(_e)}

# --- Memory ---
try:
    with open("/proc/meminfo") as _f:
        _mem = {}
        for _line in _f:
            _parts = _line.split()
            if _parts[0] in ("MemTotal:", "MemAvailable:", "MemFree:"):
                _mem[_parts[0].rstrip(":")] = int(_parts[1])
    _total = _mem.get("MemTotal", 0)
    _avail = _mem.get("MemAvailable", _mem.get("MemFree", 0))
    _used = _total - _avail
    _result["memory"] = {
        "total_mb": round(_total / 1024),
        "used_mb": round(_used / 1024),
        "available_mb": round(_avail / 1024),
        "percent_used": round(_used / _total * 100, 1) if _total else 0,
    }
except Exception as _e:
    _result["memory"] = {"error": str(_e)}

# --- GPU ---
try:
    import subprocess
    _r = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    _gpus = []
    for _line in _r.stdout.strip().splitlines():
        _p = [_x.strip() for _x in _line.split(",")]
        if len(_p) >= 6:
            _gpus.append({
                "index": int(_p[0]),
                "name": _p[1],
                "memory_used_mb": int(_p[2]),
                "memory_total_mb": int(_p[3]),
                "utilization_percent": int(_p[4]),
                "temperature_c": int(_p[5]),
            })
    _result["gpu"] = _gpus if _gpus else {"error": "no GPUs found"}
except FileNotFoundError:
    _result["gpu"] = {"error": "nvidia-smi not found (no GPU)"}
except Exception as _e:
    _result["gpu"] = {"error": str(_e)}

print(_json.dumps(_result))
"""
    session.jupyter_client.interrupt_kernel(session.kernel_id)
    outputs = session.jupyter_client.execute_code(session.kernel_id, code, timeout=30)

    for out in outputs:
        if out["type"] == "text":
            try:
                return json.loads(out["content"].strip())
            except json.JSONDecodeError:
                pass
        if out["type"] == "error":
            return {"error": f"{out.get('ename')}: {out.get('evalue')}"}

    return {"error": "no output from resource check"}


@mcp.tool(output_schema=None)
def execute_scratch(code: str) -> list:
    """Execute code on a temporary kernel for diagnostics (GPU status, etc.).

    Starts a throwaway kernel, runs the code, and shuts it down immediately.
    The code is NOT saved to any notebook and does not affect session state.

    Args:
        code: Python code to execute.

    Returns:
        List of text strings and Image objects.
    """
    server = _get_or_start_server()
    kernel_id = server.client.start_kernel()
    time.sleep(2)
    try:
        outputs = server.client.execute_code(kernel_id, code)
        return _format_outputs(outputs)
    finally:
        try:
            server.client.shutdown_kernel(kernel_id)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# MCP Resource
# ---------------------------------------------------------------------------

@mcp.resource("jlab-mcp://server/status")
def server_status() -> dict:
    """Get server status: shared SLURM job, active sessions."""
    server_info = {}
    if _server is not None:
        healthy = False
        try:
            healthy = _server.client.health_check()
        except Exception:
            pass
        server_info = {
            "job_id": _server.job_id,
            "hostname": _server.hostname,
            "url": _server.client.base_url,
            "healthy": healthy,
        }

    session_info = {}
    for sid, session in sessions.items():
        session_info[sid] = {
            "kernel_id": session.kernel_id,
            "notebook_path": str(session.notebook_path),
        }

    return {
        "server": server_info if server_info else "not connected",
        "active_sessions": len(sessions),
        "sessions": session_info,
    }
