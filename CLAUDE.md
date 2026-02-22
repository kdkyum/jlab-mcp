# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Install/sync dependencies
uv sync

# Install torch separately (GPU support, not in pyproject.toml)
uv pip install torch --index-url https://download.pytorch.org/whl/cu126

# Run the MCP server
uv run python -m jlab_mcp

# Run all unit tests (no SLURM required)
uv run python -m pytest tests/test_slurm.py tests/test_notebook.py tests/test_image_utils.py tests/test_local.py -v

# Run a single test
uv run python -m pytest tests/test_notebook.py::TestEditCell::test_edit_negative_index -v

# Run integration tests (requires SLURM cluster with GPU partition)
uv run python -m pytest tests/test_tools.py -v -s --timeout=300
```

## Architecture

MCP server (FastMCP, stdio transport) that manages JupyterLab sessions. Supports two modes: **SLURM** (HPC clusters) and **local** (laptops/workstations). Mode is auto-detected (`sbatch` on PATH → SLURM, else local) or set via `JLAB_MCP_RUN_MODE`.

```
Claude Code ↔ stdio ↔ MCP Server ↔ HTTP/WS ↔ JupyterLab (SLURM compute node or local subprocess)
```

Communication happens via:
1. **Status file** — `jlab-mcp start` writes hostname/port/token, MCP server reads it
2. **JupyterLab REST API** — kernel lifecycle (start, stop, list)
3. **Kernel WebSocket** — code execution via Jupyter message protocol v5.3

### Module Roles

- **server.py** — FastMCP server with 7 tools + 1 resource. Maintains a global `sessions: dict[str, Session]` mapping session IDs to `Session` dataclasses (job_id, kernel_id, JupyterLabClient, notebook_path, NotebookManager).
- **slurm.py** — Renders SLURM template, runs `sbatch`/`squeue`/`scancel`, polls for job state and connection file. All SLURM output parsing is string-based (no `jq`).
- **local.py** — Local mode: spawns `jupyter lab` as a subprocess, manages PID-based lifecycle.
- **jupyter_client.py** — `JupyterLabClient` class: REST API calls (`requests`) for kernel management, WebSocket (`websocket-client`) for code execution. Collects outputs (text/image/error) until kernel goes idle.
- **notebook.py** — `NotebookManager` class: creates/edits/saves `.ipynb` files with `nbformat`. Handles output conversion, cell ID generation, and notebook restoration (re-executing all cells).
- **config.py** — All defaults overridable via `JLAB_MCP_*` environment variables. Directories auto-created on import.
- **image_utils.py** — Resizes images >512px maintaining aspect ratio (Pillow). Gracefully returns original bytes on error.

### Session Lifecycle

`start_new_session` → submit sbatch → poll squeue until RUNNING → poll connection file → health check JupyterLab → start kernel → create notebook → return session_id. Shutdown reverses: stop kernel → scancel job.

## Key Constraints

- `nbformat` v5+ requires cell IDs — `clean_notebook()` ensures they exist, never removes them
- `torch` is installed separately via `uv pip install` (not in pyproject.toml dependencies)
- The SLURM template at `templates/jupyter_slurm.sh.template` uses Python `.format()` placeholders (curly braces)
- All SLURM settings (partition, GPU, modules, etc.) are configurable via `JLAB_MCP_*` env vars — see README.md
- `server.py` validates notebook paths against `NOTEBOOK_DIR` to prevent path traversal
- Connection files are created with `umask 077` and cleaned up on session shutdown
