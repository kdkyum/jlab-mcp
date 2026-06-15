import hashlib
import importlib.resources
import os
import shutil
from pathlib import Path


def _get_path(env_key: str, default: str, private: bool = False) -> Path:
    value = os.environ.get(env_key, default)
    # Resolve symlinks so paths compare equal everywhere (e.g. /ptmp -> /viper/ptmp2)
    path = Path(value).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    if private:
        # Directories holding tokens (status/connection/log files) must not be
        # group/world readable — parent-dir permissions can't be relied on when
        # these are relocated to shared scratch filesystems.
        os.chmod(path, 0o700)
    return path


def _get_str(env_key: str, default: str) -> str:
    return os.environ.get(env_key, default)


def _get_int(env_key: str, default: int) -> int:
    return int(os.environ.get(env_key, str(default)))


# Base directory
JLAB_MCP_DIR = _get_path("JLAB_MCP_DIR", "~/.jlab-mcp", private=True)

# Notebook storage (defaults to cwd/notebooks)
NOTEBOOK_DIR = _get_path("JLAB_MCP_NOTEBOOK_DIR", str(Path.cwd() / "notebooks"))

# JupyterLab root directory (what the file browser sees; defaults to project root)
SERVER_ROOT_DIR = _get_path("JLAB_MCP_SERVER_ROOT_DIR", str(Path.cwd()))

# SLURM job logs (Jupyter prints tokenized URLs into its logs — keep private)
LOG_DIR = _get_path("JLAB_MCP_LOG_DIR", str(JLAB_MCP_DIR / "logs"), private=True)

# Connection info files
CONNECTION_DIR = _get_path(
    "JLAB_MCP_CONNECTION_DIR", str(JLAB_MCP_DIR / "connections"), private=True
)

# SLURM defaults (all overridable via JLAB_MCP_* env vars)
SLURM_PARTITION = _get_str("JLAB_MCP_SLURM_PARTITION", "gpu")
SLURM_GRES = _get_str("JLAB_MCP_SLURM_GRES", "gpu:1")
SLURM_CPUS = _get_int("JLAB_MCP_SLURM_CPUS", 4)
SLURM_MEM = _get_int("JLAB_MCP_SLURM_MEM", 32000)
SLURM_TIME = _get_str("JLAB_MCP_SLURM_TIME", "4:00:00")

# Address JupyterLab binds to on the compute node (the `--ip` flag). The
# default 0.0.0.0 listens on all interfaces and the connection file advertises
# the node's `$(hostname)` (resolvable from the login node where the MCP server
# runs). Set a concrete IP to bind a single interface — that exact IP is then
# advertised so the MCP server connects to the interface we bound.
SLURM_BIND_IP = _get_str("JLAB_MCP_SLURM_BIND_IP", "0.0.0.0")

# How long `jlab-mcp start` waits for the job to leave the queue before
# giving up (the job stays queued and `jlab-mcp start` resumes waiting).
QUEUE_TIMEOUT = _get_int("JLAB_MCP_QUEUE_TIMEOUT", 300)

# How long to wait for the connection file / JupyterLab health check once
# the job is running, before cancelling the job and reporting an error.
READY_TIMEOUT = _get_int("JLAB_MCP_READY_TIMEOUT", 120)

# Modules to load in SLURM job (space-separated, e.g. "cuda/12.6 anaconda3")
# Set to empty string to skip module loading
SLURM_MODULES = _get_str("JLAB_MCP_SLURM_MODULES", "")

# Port range for JupyterLab
PORT_RANGE = (
    _get_int("JLAB_MCP_PORT_MIN", 18000),
    _get_int("JLAB_MCP_PORT_MAX", 19000),
)

# Working directory — the .venv here is activated on the compute node
PROJECT_DIR = Path.cwd()

# Per-project status file (keyed by project directory)
_project_hash = hashlib.sha256(str(PROJECT_DIR).encode()).hexdigest()[:12]
STATUS_DIR = _get_path(
    "JLAB_MCP_STATUS_DIR",
    str(JLAB_MCP_DIR / "servers" / f"{PROJECT_DIR.name}-{_project_hash}"),
    private=True,
)
STATUS_FILE = STATUS_DIR / "server-status"


def _detect_run_mode() -> str:
    """Detect whether to use SLURM or local mode.

    Checks JLAB_MCP_RUN_MODE env var first, then falls back to
    auto-detection based on whether sbatch is available on PATH.
    """
    env_mode = os.environ.get("JLAB_MCP_RUN_MODE", "").lower()
    if env_mode in ("local", "slurm"):
        return env_mode
    return "slurm" if shutil.which("sbatch") else "local"


RUN_MODE = _detect_run_mode()

# Local mode bind address. JupyterLab and the MCP server run on the same
# host in local mode, so loopback is sufficient; set to 0.0.0.0 explicitly
# if remote access to the JupyterLab UI is needed.
LOCAL_BIND_IP = _get_str("JLAB_MCP_LOCAL_BIND_IP", "127.0.0.1")


def get_template_content() -> str:
    """Read the bundled SLURM template from package data."""
    return importlib.resources.read_text("jlab_mcp", "jupyter_slurm.sh.template")
