"""Local mode: spawn JupyterLab as a subprocess instead of via SLURM."""

import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

from jlab_mcp import config
from jlab_mcp.slurm import generate_token, random_port

logger = logging.getLogger("jlab-mcp.local")


def start_jupyter_local() -> tuple[subprocess.Popen, str, int, str]:
    """Start JupyterLab as a local subprocess.

    Returns (process, hostname, port, token).
    """
    port = random_port()
    token = generate_token()
    hostname = config.LOCAL_BIND_IP

    log_file = config.LOG_DIR / f"jupyter-local-{port}.log"

    # Use the project's .venv Python so kernels have project dependencies
    venv_python = config.PROJECT_DIR / ".venv" / "bin" / "python"
    if venv_python.exists():
        python = str(venv_python)
    else:
        python = sys.executable

    cmd = [
        python,
        "-m",
        "jupyter",
        "lab",
        f"--ip={hostname}",
        f"--port={port}",
        "--no-browser",
        # Fail fast on a port collision instead of silently binding port+1
        # while the status file advertises the original port
        "--ServerApp.port_retries=0",
        "--ServerApp.shutdown_no_activity_timeout=0",
        "--MappingKernelManager.cull_idle_timeout=0",
        "--MappingKernelManager.cull_interval=300",
        "--MappingKernelManager.cull_connected=True",
        f"--notebook-dir={config.SERVER_ROOT_DIR}",
    ]

    # Set VIRTUAL_ENV so JupyterLab picks up the project's venv
    env = os.environ.copy()
    # Token via env, not argv: /proc/<pid>/cmdline is world-readable on
    # multi-user hosts, /proc/<pid>/environ is owner-only
    env["JUPYTER_TOKEN"] = token
    venv_dir = config.PROJECT_DIR / ".venv"
    if venv_dir.exists():
        env["VIRTUAL_ENV"] = str(venv_dir)
        env["PATH"] = f"{venv_dir / 'bin'}:{env.get('PATH', '')}"

    with open(log_file, "w") as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
        )

    logger.info(f"Started JupyterLab (PID {proc.pid}) on {hostname}:{port}")
    return proc, hostname, port, token


def _pid_is_jupyter(pid: int) -> bool:
    """Best-effort check that a PID actually belongs to a jupyter process.

    Status files survive reboots, and PIDs get recycled — without this
    check, `stop` could SIGTERM an unrelated process. Where /proc is not
    available (e.g. macOS), fall back to trusting the PID.
    """
    try:
        cmdline = (Path(f"/proc/{pid}") / "cmdline").read_bytes()
    except OSError:
        return True
    return b"jupyter" in cmdline


def stop_jupyter_local(pid: int) -> None:
    """Stop a local JupyterLab process by PID."""
    if not _pid_is_jupyter(pid):
        logger.warning(
            "PID %d is not a jupyter process (stale status file?) — not killing it",
            pid,
        )
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def is_local_running(pid: int) -> bool:
    """Check if a local JupyterLab process is still alive."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return _pid_is_jupyter(pid)
