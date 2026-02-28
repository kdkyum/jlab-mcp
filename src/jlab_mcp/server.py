import asyncio
import base64
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastmcp import Context, FastMCP
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
_sessions_lock = threading.Lock()


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
            with _sessions_lock:
                sessions.clear()

        _server = _connect_to_server()
        logger.info(f"Connected to JupyterLab on {_server.hostname}")
        return _server


# ---------------------------------------------------------------------------
# Cleanup (kernels only — SLURM job is managed by user)
# ---------------------------------------------------------------------------

def _cleanup_kernels():
    """Shutdown all kernels. Does NOT cancel the SLURM job."""
    with _sessions_lock:
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
    if not nb_path.is_relative_to(notebook_dir):
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
            # Strip ANSI escape codes from IPython tracebacks
            tb = re.sub(r"\x1b\[[0-9;]*m", "", tb)
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
    with _sessions_lock:
        sessions[session_id] = session
    return session


def _get_session(session_id: str) -> Session:
    """Look up a session by ID or raise ValueError."""
    with _sessions_lock:
        if session_id not in sessions:
            raise ValueError(f"Unknown session: {session_id}")
        return sessions[session_id]


async def _run_with_progress(
    ctx: Context,
    fn,
    *args,
    progress: int = 0,
    total: int = 0,
):
    """Run a blocking function in a thread, sending MCP progress every 15s.

    This keeps the stdio pipe active so Claude Code doesn't consider
    the MCP server dead during long-running kernel executions.

    On cancellation (user presses ESC), the background thread is NOT
    automatically stopped -- callers must handle CancelledError and
    clean up (e.g. close WebSocket, interrupt kernel).

    The *progress* and *total* parameters are forwarded to
    ``ctx.report_progress`` so callers can report cell-level progress
    instead of elapsed seconds.
    """
    task = asyncio.create_task(asyncio.to_thread(fn, *args))
    start = time.time()
    try:
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=15)
            if done:
                break
            report = int(time.time() - start) if total == 0 else progress
            await ctx.report_progress(progress=report, total=total)
        return task.result()
    except asyncio.CancelledError:
        task.cancel()
        raise


def _interrupt_and_close_ws(client: JupyterLabClient, kernel_id: str) -> None:
    """Interrupt a kernel and close its cached WebSocket.

    Used on cancellation to stop the kernel and unblock any background
    thread stuck on ws.recv().
    """
    client._close_ws(kernel_id)
    try:
        client.interrupt_kernel(kernel_id)
    except Exception:
        pass


async def _execute_with_cancellation(
    ctx: Context,
    client: JupyterLabClient,
    kernel_id: str,
    code: str,
    *,
    progress: int = 0,
    total: int = 0,
) -> list[dict]:
    """Execute code on a kernel, handling cancellation cleanly.

    Wraps _run_with_progress and, on CancelledError, interrupts the
    kernel and closes the WebSocket so the background thread unblocks.
    """
    try:
        return await _run_with_progress(
            ctx, client.execute_code, kernel_id, code,
            progress=progress, total=total,
        )
    except asyncio.CancelledError:
        _interrupt_and_close_ws(client, kernel_id)
        raise


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def start_new_notebook(experiment_name: str) -> dict:
    """Start a new session: start kernel, create notebook.

    Uses the shared JupyterLab server (must be started with `jlab-mcp start`).

    Args:
        experiment_name: Name for the experiment/notebook.

    Returns:
        Dict with session_id, notebook_path, job_id, hostname.
    """
    server = _get_or_start_server()
    kernel_id = _start_kernel(server)

    try:
        nb_manager = NotebookManager()
        nb_path = nb_manager.create_notebook(experiment_name, config.NOTEBOOK_DIR)
    except Exception:
        server.client.shutdown_kernel(kernel_id)
        raise

    session = _register_session(kernel_id, server.client, nb_path, nb_manager)
    return {
        "session_id": session.session_id,
        "notebook_path": str(nb_path),
        "job_id": server.job_id,
        "hostname": server.hostname,
    }


@mcp.tool()
def start_notebook(notebook_path: str) -> dict:
    """Open an existing notebook, reusing the kernel if still alive.

    If a session already exists for this notebook and its kernel is
    still running, returns that session (all state preserved).
    Otherwise starts a fresh kernel (no previous state).

    Use shutdown_session + start_notebook to force a kernel restart.

    Args:
        notebook_path: Path to existing notebook.

    Returns:
        Dict with session_id, notebook_path, job_id, hostname.
    """
    nb_path = _validate_notebook_path(notebook_path)
    server = _get_or_start_server()

    # Check for an existing session on this notebook with a live kernel
    with _sessions_lock:
        for session in sessions.values():
            if session.notebook_path == nb_path:
                live_ids = {
                    k["id"] for k in server.client.list_kernels()
                }
                if session.kernel_id in live_ids:
                    return {
                        "session_id": session.session_id,
                        "notebook_path": str(nb_path),
                        "job_id": server.job_id,
                        "hostname": server.hostname,
                    }

    # No live session — start a fresh kernel
    kernel_id = _start_kernel(server)

    try:
        nb_manager = NotebookManager()
        nb_manager.get_notebook(nb_path)  # validate it's a real notebook
    except Exception:
        server.client.shutdown_kernel(kernel_id)
        raise

    session = _register_session(kernel_id, server.client, nb_path, nb_manager)
    return {
        "session_id": session.session_id,
        "notebook_path": str(nb_path),
        "job_id": server.job_id,
        "hostname": server.hostname,
    }


@mcp.tool(output_schema=None)
async def execute_code(session_id: str, code: str, ctx: Context) -> list:
    """Execute code in the kernel and add cell to notebook.

    Args:
        session_id: Session identifier.
        code: Python code to execute.

    Returns:
        List of text strings and Image objects.
    """
    session = _get_session(session_id)
    outputs = await _execute_with_cancellation(
        ctx, session.jupyter_client, session.kernel_id, code
    )
    session.notebook_manager.add_code_cell(
        session.notebook_path, code, outputs
    )
    return _format_outputs(outputs)


@mcp.tool(output_schema=None)
async def edit_cell(session_id: str, cell_index: int, code: str, ctx: Context) -> list:
    """Edit an existing cell, re-execute it, and update outputs.

    Args:
        session_id: Session identifier.
        cell_index: Cell index (supports negative indexing).
        code: New code for the cell.

    Returns:
        List of text strings and Image objects.
    """
    session = _get_session(session_id)
    outputs = await _execute_with_cancellation(
        ctx, session.jupyter_client, session.kernel_id, code
    )
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
    with _sessions_lock:
        if session_id not in sessions:
            raise ValueError(f"Unknown session: {session_id}")
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

    result = {
        "status": "ok" if healthy else "unreachable",
        "hostname": hostname,
        "url": f"http://{hostname}:{port}",
        "healthy": healthy,
        "active_sessions": len(sessions),
    }

    if healthy:
        try:
            kernels = client.list_kernels()
            result["kernels"] = [
                {
                    "id": k["id"],
                    "name": k.get("name", ""),
                    "state": k.get("execution_state", "unknown"),
                    "last_activity": k.get("last_activity", ""),
                }
                for k in kernels
            ]
        except Exception:
            result["kernels"] = []

    return result


@mcp.tool()
def check_resources() -> dict:
    """Check compute resource usage: CPU, memory, and GPU.

    Runs on a throwaway kernel so it does not pollute session state.
    No session required — only needs a running JupyterLab server.

    Returns:
        Dict with cpu, memory, and gpu resource information.
    """
    code = """\
import json, os, subprocess

result = {}

# --- CPU ---
try:
    nproc = os.cpu_count()
    load1, load5, load15 = os.getloadavg()
    result["cpu"] = {
        "count": nproc,
        "load_1m": round(load1, 2),
        "load_5m": round(load5, 2),
        "load_15m": round(load15, 2),
    }
except Exception as e:
    result["cpu"] = {"error": str(e)}

# --- Memory ---
try:
    with open("/proc/meminfo") as f:
        mem = {}
        for line in f:
            parts = line.split()
            if parts[0] in ("MemTotal:", "MemAvailable:", "MemFree:"):
                mem[parts[0].rstrip(":")] = int(parts[1])
    total = mem.get("MemTotal", 0)
    avail = mem.get("MemAvailable", mem.get("MemFree", 0))
    used = total - avail
    result["memory"] = {
        "total_mb": round(total / 1024),
        "used_mb": round(used / 1024),
        "available_mb": round(avail / 1024),
        "percent_used": round(used / total * 100, 1) if total else 0,
    }
except Exception as e:
    result["memory"] = {"error": str(e)}

# --- GPU ---
try:
    r = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    gpus = []
    for line in r.stdout.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 6:
            gpus.append({
                "index": int(p[0]),
                "name": p[1],
                "memory_used_mb": int(p[2]),
                "memory_total_mb": int(p[3]),
                "utilization_percent": int(p[4]),
                "temperature_c": int(p[5]),
            })
    result["gpu"] = gpus if gpus else {"error": "no GPUs found"}
except FileNotFoundError:
    result["gpu"] = {"error": "nvidia-smi not found (no GPU)"}
except Exception as e:
    result["gpu"] = {"error": str(e)}

print(json.dumps(result))
"""
    server = _get_or_start_server()
    kernel_id = server.client.start_kernel()
    time.sleep(2)
    try:
        outputs = server.client.execute_code(kernel_id, code, timeout=30)
    finally:
        try:
            server.client.shutdown_kernel(kernel_id)
        except Exception:
            pass

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
async def execute_scratch(code: str, ctx: Context) -> list:
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
        outputs = await _execute_with_cancellation(
            ctx, server.client, kernel_id, code
        )
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

    with _sessions_lock:
        session_info = {}
        for sid, session in sessions.items():
            session_info[sid] = {
                "kernel_id": session.kernel_id,
                "notebook_path": str(session.notebook_path),
            }
        session_count = len(sessions)

    return {
        "server": server_info if server_info else "not connected",
        "active_sessions": session_count,
        "sessions": session_info,
    }
