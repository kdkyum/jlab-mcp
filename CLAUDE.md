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

# Run integration tests (requires running `jlab-mcp start` first)
uv run python -m pytest tests/test_tools.py -v -s --timeout=600
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

- **server.py** — FastMCP server with 10 tools + 1 resource. Maintains a global `sessions: dict[str, Session]` mapping session IDs to `Session` dataclasses (kernel_id, JupyterLabClient, notebook_path, NotebookManager). All access to `sessions` dict is guarded by `_sessions_lock`.
- **slurm.py** — Renders SLURM template, runs `sbatch`/`squeue`/`scancel`, polls for job state and connection file. All SLURM output parsing is string-based (no `jq`).
- **local.py** — Local mode: spawns `jupyter lab` as a subprocess, manages PID-based lifecycle.
- **jupyter_client.py** — `JupyterLabClient` class: REST API calls (`requests`) for kernel management, WebSocket (`websocket-client`) for code execution. Caches one WebSocket per kernel (`_ws_cache`) to avoid reconnection storms — critical during `restore_notebook` which executes many cells sequentially. Collects outputs (text/image/error) until kernel goes idle.
- **notebook.py** — `NotebookManager` class: creates/edits/saves `.ipynb` files with `nbformat`. Handles output conversion, cell ID generation, and notebook restoration (re-executing all cells).
- **config.py** — All defaults overridable via `JLAB_MCP_*` environment variables. Directories auto-created on import.
- **image_utils.py** — Resizes images >512px maintaining aspect ratio (Pillow). Gracefully returns original bytes on error.
- **__main__.py** — CLI entry point: `jlab-mcp start [--debug]`, `stop`, `wait`, `status`, or MCP server (no args). Parses `JLAB_MCP_RUN_MODE` for SLURM vs local mode.

### Session Lifecycle

`start_new_notebook` → read status file → health check JupyterLab → start kernel → create notebook → return session_id. `start_notebook` → same but attaches to an existing notebook (no fork, no re-execution). Restart kernel: `shutdown_session` + `start_notebook(same_path)`. The MCP server does **not** manage SLURM — `jlab-mcp start` (run separately) handles job submission and writes the status file. Shutdown: stop kernel only (SLURM job stays alive).

## Key Constraints

- `nbformat` v5+ requires cell IDs — `clean_notebook()` ensures they exist, never removes them
- `torch` is installed separately via `uv pip install` (not in pyproject.toml dependencies)
- The SLURM template at `src/jlab_mcp/jupyter_slurm.sh.template` uses Python `.format()` placeholders (curly braces)
- All SLURM settings (partition, GPU, modules, etc.) are configurable via `JLAB_MCP_*` env vars — see README.md
- `server.py` validates notebook paths against `NOTEBOOK_DIR` to prevent path traversal
- Connection files are created with `umask 077` and cleaned up on session shutdown
- `execute_code`/`edit_cell`/`execute_scratch` are async and use `_run_with_progress()` to send MCP progress every 15s during long executions, keeping the stdio pipe alive
- Tools returning `Image` objects must use `@mcp.tool(output_schema=None)` and return `list` — otherwise FastMCP fails to serialize `Image` as `ImageContent`
- `_sessions_lock` must guard all reads/writes to the `sessions` dict (thread safety for concurrent tool calls)
- Kernel death (OOM, crash) is detected during WebSocket execution via `status: restarting/dead` messages and `WebSocketConnectionClosedException`
- On user cancellation (ESC), `_execute_with_cancellation()` closes the cached WebSocket (unblocks background thread) and interrupts the kernel — do NOT add pre-execution `interrupt_kernel` calls (causes WS reconnection storms)
- `_run_with_progress()` accepts `progress`/`total` kwargs: `total=0` (default) reports elapsed seconds, `total>0` reports caller-provided cell-level progress
