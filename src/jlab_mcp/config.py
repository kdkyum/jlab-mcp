import importlib.resources
import os
from pathlib import Path


def _get_path(env_key: str, default: str) -> Path:
    value = os.environ.get(env_key, default)
    path = Path(value).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_str(env_key: str, default: str) -> str:
    return os.environ.get(env_key, default)


def _get_int(env_key: str, default: int) -> int:
    return int(os.environ.get(env_key, str(default)))


# Base directory
JLAB_MCP_DIR = _get_path("JLAB_MCP_DIR", "~/.jlab-mcp")

# Notebook storage (shared FS)
NOTEBOOK_DIR = _get_path("JLAB_MCP_NOTEBOOK_DIR", str(JLAB_MCP_DIR / "notebooks"))

# SLURM job logs
LOG_DIR = _get_path("JLAB_MCP_LOG_DIR", str(JLAB_MCP_DIR / "logs"))

# Connection info files
CONNECTION_DIR = _get_path("JLAB_MCP_CONNECTION_DIR", str(JLAB_MCP_DIR / "connections"))

# SLURM defaults (all overridable via JLAB_MCP_* env vars)
SLURM_PARTITION = _get_str("JLAB_MCP_SLURM_PARTITION", "gpu")
SLURM_GRES = _get_str("JLAB_MCP_SLURM_GRES", "gpu:1")
SLURM_CPUS = _get_int("JLAB_MCP_SLURM_CPUS", 4)
SLURM_MEM = _get_int("JLAB_MCP_SLURM_MEM", 32000)
SLURM_TIME = _get_str("JLAB_MCP_SLURM_TIME", "4:00:00")

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


def get_template_content() -> str:
    """Read the bundled SLURM template from package data."""
    return importlib.resources.read_text("jlab_mcp", "jupyter_slurm.sh.template")
