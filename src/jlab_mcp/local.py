"""Local mode: spawn JupyterLab as a subprocess instead of via SLURM."""

import logging
import os
import signal
import subprocess
import sys

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

    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "lab",
        f"--ip={hostname}",
        f"--port={port}",
        f"--IdentityProvider.token={token}",
        "--no-browser",
        f"--notebook-dir={config.NOTEBOOK_DIR}",
    ]

    with open(log_file, "w") as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )

    logger.info(f"Started JupyterLab (PID {proc.pid}) on {hostname}:{port}")
    return proc, hostname, port, token


def stop_jupyter_local(pid: int) -> None:
    """Stop a local JupyterLab process by PID."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def is_local_running(pid: int) -> bool:
    """Check if a local JupyterLab process is still alive."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
