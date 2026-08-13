# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Install/sync dependencies
uv sync

# Install torch separately (GPU support, not in pyproject.toml)
uv pip install torch --index-url https://download.pytorch.org/whl/cu126   # NVIDIA
uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.3 # AMD (MI300A)

# Run the MCP server
uv run python -m jlab_mcp

# Run all unit tests (no SLURM required)
uv run python -m pytest tests/ --ignore=tests/test_tools.py -v

# Run a single test
uv run python -m pytest tests/test_notebook.py::TestEditCell::test_edit_negative_index -v

# Run integration tests (requires running `jlab-mcp start` first)
uv run python -m pytest tests/test_tools.py -v -s --timeout=600
```

## Architecture

MCP server (official `mcp` SDK ≥2.0 `MCPServer`, stdio transport, speaks both the stateless 2026-07-28 protocol and the legacy initialize-handshake protocol) that manages JupyterLab sessions. Supports two modes: **SLURM** (HPC clusters) and **local** (laptops/workstations). Mode is auto-detected (`sbatch` on PATH → SLURM, else local) or set via `JLAB_MCP_RUN_MODE`.

```
Claude Code ↔ stdio ↔ MCP Server ↔ HTTP/WS ↔ JupyterLab (SLURM compute node or local subprocess)
```

Communication happens via:
1. **Status file** — `jlab-mcp start` writes hostname/port/token to a per-project file (`~/.jlab-mcp/servers/{name}-{hash}/server-status`), MCP server reads it. Project is determined by `Path.cwd()` at startup.
2. **JupyterLab REST API** — kernel lifecycle (start, stop, list)
3. **Kernel WebSocket** — code execution via Jupyter message protocol v5.3

### Module Roles

- **server.py** — `MCPServer` (mcp SDK 2.x) with 15 tools + 1 resource. Maintains a global `sessions: dict[str, Session]` mapping session IDs to `Session` dataclasses (kernel_id, JupyterLabClient, notebook_path, NotebookManager). All access to `sessions` dict is guarded by `_sessions_lock`.
- **autostart.py** — Lets the MCP server drive `jlab-mcp start` itself (`start_server`/`wait_for_server` tools): persists the per-project run-mode choice (`STATUS_DIR/run-mode`), spawns a detached bootstrap+start pipeline (own session, stdio redirected to `STATUS_DIR/start.log` — never the MCP stdio pipe), bootstraps bare projects (`uv init --bare` + `uv add jupyterlab ipykernel matplotlib numpy` when `.venv/bin/jupyter-lab` is missing), and polls the status file (`poll_until_ready`). The CLI remains the sole owner of the startup state machine.
- **slurm.py** — Renders SLURM template, runs `sbatch`/`squeue`/`scancel`, polls for job state and connection file. All SLURM output parsing is string-based (no `jq`). A nonzero squeue/scancel exit raises `SlurmCommandError`; `is_job_running`/`is_job_alive` treat that as "assume alive" so transient controller failures never get a healthy job cancelled.
- **local.py** — Local mode: spawns `jupyter lab` as a subprocess, manages PID-based lifecycle. PIDs are verified against `/proc/<pid>/cmdline` before trusting/killing (PID recycling). The token is passed via the `JUPYTER_TOKEN` env var, never argv.
- **jupyter_client.py** — `JupyterLabClient` class: REST API calls (`requests`) for kernel management, WebSocket (`websocket-client`) for code execution. Caches one WebSocket per kernel (`_ws_cache`) to avoid reconnection storms. Executions on the same kernel are serialized via per-kernel locks (`_exec_locks`) — two recv loops on one WS would steal each other's messages. Collects outputs (text/image/error) until kernel goes idle.
- **notebook.py** — `NotebookManager` class: creates/edits/saves `.ipynb` files with `nbformat`. Handles output conversion (stream stdout/stderr, execute_result, images, errors), cell ID generation, and id-based output updates (`get_cell_id`/`update_cell_outputs_by_id`). `create_notebook` never overwrites — name collisions get `_2`, `_3`, … suffixes.
- **config.py** — All defaults overridable via `JLAB_MCP_*` environment variables. Directories auto-created on import (token-bearing dirs with mode 0700); paths are symlink-resolved.
- **image_utils.py** — Resizes images >2576px on the long edge maintaining aspect ratio (Pillow). Gracefully returns original bytes on error.
- **__main__.py** — CLI entry point: `jlab-mcp start [--debug]`, `stop`, `wait`, `status`, or MCP server (no args). Parses `JLAB_MCP_RUN_MODE` for SLURM vs local mode.

### Session Lifecycle

`start_new_notebook` → read status file → health check JupyterLab → shut down ALL existing kernels/sessions (one active session at a time) → start kernel → create notebook → return session_id. `start_notebook` → opens an existing notebook, reusing a live session/kernel for that path if one exists (state preserved), else starts a fresh kernel (no fork, no re-execution); dead sessions for the path are purged. Restart kernel: `shutdown_session` + `start_notebook(same_path)`. The MCP server does **not** own the startup state machine — `jlab-mcp start` handles job submission and writes the status file. It can be run manually, or launched by the agent via the `start_server` tool (spawns it detached; asks the user local-vs-slurm on first use — plus SLURM walltime/resources, grounded in a live `sinfo` partition survey (`survey_slurm_partitions`) included in the `mode_required` response, forwarded as `JLAB_MCP_SLURM_*` env to the child — choices saved in `STATUS_DIR/run-mode` and `STATUS_DIR/slurm-options`; `JLAB_MCP_RUN_MODE` env overrides the mode) followed by `wait_for_server` (progress keepalives; auto-resumes a lapsed SLURM queue wait with the saved options). Shutdown: stop kernel only (SLURM job stays alive). `jlab-mcp start`/`stop` are serialized per project via an flock; the status file is written atomically with mode 0600. A queue-wait timeout leaves the job queued and the status `pending` (rerun `start` or let `wait_for_server` resume); a readiness timeout after the job is running cancels the job.

## Key Constraints

- `nbformat` v5+ requires cell IDs — `clean_notebook()` ensures they exist, never removes them
- `torch` is installed separately via `uv pip install` (not in pyproject.toml dependencies)
- The SLURM template at `src/jlab_mcp/jupyter_slurm.sh.template` uses Python `.format()` placeholders (curly braces)
- All SLURM settings (partition, GPU, modules, etc.) are configurable via `JLAB_MCP_*` env vars — see README.md
- `server.py` validates notebook paths against `NOTEBOOK_DIR` to prevent path traversal
- Connection files are created with `umask 077`, unique per submission (`jupyter-{port}-{token[:8]}.conn`), and removed once the server is ready and on `jlab-mcp stop`/error paths
- Tokens never appear on command lines (world-readable `/proc/<pid>/cmdline`) — they travel via the `JUPYTER_TOKEN` env var and 0600 files only
- `execute_code`/`run_cell`/`execute_scratch` are async and use `_run_with_progress()` to send MCP progress every 15s during long executions, keeping the stdio pipe alive
- Network-bound sync work inside async tools goes through `asyncio.to_thread` — blocking the event loop starves the progress keepalives of concurrent executions
- `execute_code`/`run_cell` pin the target cell by its nbformat id before the long await and write outputs by id (`update_cell_outputs_by_id`) — positional indices go stale if cells are edited during execution; a failed save appends a warning instead of discarding the outputs
- `edit_cell` is synchronous (edit only, no execution) — use `run_cell` afterwards to execute; it rejects non-code cells (writing `outputs` to a markdown cell makes the notebook schema-invalid)
- Tools returning `Image` objects must use `@mcp.tool(structured_output=False)` and return `list` — the SDK then serializes each element as its own content block (`Image` → `ImageContent`)
- The `mcp` SDK is ≥2.0 (protocol 2026-07-28 + legacy era). `@mcp.tool()` returns the original function (tests call tools directly, no `.fn` unwrapping); `Context`/`Image` import from `mcp.server.mcpserver`; `ctx.report_progress` is a no-op when the client sends no `progressToken`
- `_sessions_lock` must guard all reads/writes to the `sessions` dict; `_server` reads outside `_get_or_start_server` snapshot it under `_server_lock` (thread safety for concurrent tool calls)
- Kernel death (OOM, crash) is detected during WebSocket execution via `status: restarting/dead` messages and `WebSocketConnectionClosedException`; the cached WS is evicted on death so buffered death broadcasts can't poison the next execution. A 404 channels handshake returns `KernelGone`
- On user cancellation (ESC), `_execute_with_cancellation()` calls `client.cancel_execution()` (closes the cached WebSocket to unblock the background thread AND sets a per-kernel cancelled flag so the retry loop doesn't re-send the code) then interrupts the kernel — do NOT add pre-execution `interrupt_kernel` calls (causes WS reconnection storms)
- `_run_with_progress()` accepts `progress`/`total` kwargs: `total=0` (default) reports elapsed seconds, `total>0` reports caller-provided cell-level progress
- `start_server` never blocks and never inherits stdio: the bootstrap+`jlab-mcp start` pipeline is spawned with `start_new_session=True` and output redirected to `STATUS_DIR/start.log` (0600) — the MCP server's stdout IS the stdio transport, and in local mode the spawned process is JupyterLab's parent and must outlive the MCP server. `wait_for_server` is the blocking monitor (runs `poll_until_ready` via `_run_with_progress`); it treats `error`/`terminated` states as stale while a start attempt is in flight (`start_in_flight()`)
