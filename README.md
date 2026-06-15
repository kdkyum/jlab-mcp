# jlab-mcp

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that enables Claude Code to execute Python code on GPU compute nodes via JupyterLab running on a SLURM cluster.

Inspired by and adapted from [goodfire-ai/scribe](https://github.com/goodfire-ai/scribe), which provides notebook-based code execution for Claude. This project adapts that approach for HPC/SLURM environments where GPU resources are allocated via job schedulers.

## Architecture

```
Claude Code
    ↕ stdio
MCP Server
    ↕ HTTP/WebSocket
JupyterLab (SLURM compute node or local subprocess)   ← one server, many kernels
    ↕
IPython Kernels (GPU access)
```

JupyterLab runs either on a SLURM compute node (HPC clusters) or as a local subprocess (laptops/workstations). The server is managed separately from the MCP server — you start it with `jlab-mcp start` and it keeps running across Claude Code sessions. All sessions create separate kernels on this shared server. Each project directory gets its own JupyterLab instance — the status file is scoped by a hash of the working directory where `jlab-mcp start` was run.

## Local Mode

On machines without SLURM (laptops, workstations), jlab-mcp automatically runs JupyterLab as a local subprocess. Mode is auto-detected: if `sbatch` is on PATH, SLURM mode is used; otherwise, local mode.

Override with an environment variable:

```bash
export JLAB_MCP_RUN_MODE=local   # force local mode
export JLAB_MCP_RUN_MODE=slurm   # force SLURM mode
```

In local mode, `jlab-mcp start` runs in the **foreground** — press Ctrl+C to stop. The status file uses the same format as SLURM mode, so the MCP server works identically in both modes.

## Setup

```bash
# Install (no git clone needed)
uv tool install git+https://github.com/kdkyum/jlab-mcp.git
```

The SLURM job activates `.venv` in the **current working directory**. Set up your project's venv on the shared filesystem with the compute dependencies:

```bash
cd /shared/fs/my-project
uv venv
uv pip install jupyterlab ipykernel matplotlib numpy
uv pip install torch --index-url https://download.pytorch.org/whl/cu126   # NVIDIA GPUs
# AMD GPUs (e.g. MI300A): use the ROCm wheels instead
# uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.3
```

## Usage

### 1. Start the compute node

In a separate terminal, start the SLURM job:

```bash
jlab-mcp start              # uses default time limit (4h)
jlab-mcp start 24:00:00     # 24 hour time limit
jlab-mcp start 1-00:00:00   # 1 day
```

This submits the job and waits until JupyterLab is ready:

```
SLURM job 24215408 submitted, waiting in queue...
Job running on ravg1011, JupyterLab starting...
JupyterLab ready at http://ravg1011:18432
```

### 2. Use Claude Code

In another terminal, start Claude Code. The MCP server connects to the running JupyterLab automatically.

### 3. Stop when done

```bash
jlab-mcp stop
```

### CLI Commands

| Command | Description |
|---|---|
| `jlab-mcp start [TIME] [--debug]` | Start JupyterLab and wait until ready. In SLURM mode, submits a job and polls until the server responds. In local mode, spawns a subprocess and blocks in the foreground. Optional TIME overrides `JLAB_MCP_SLURM_TIME` (e.g. `24:00:00`). Skips submission if an existing server is still running. |
| `jlab-mcp stop` | Stop JupyterLab. In SLURM mode, runs `scancel`. In local mode, sends SIGTERM to the subprocess. Removes the status file in both cases. |
| `jlab-mcp wait` | Poll the status file from another terminal until the server is ready (up to 10 min). Prints state transitions (`pending → starting → ready`). Useful for scripts or for monitoring `start` progress from a separate shell. |
| `jlab-mcp status` | Print server state, mode, hostname, port, and whether the process/job is alive. Lists active kernels with execution state and last activity time. Queries GPU memory and utilization via `nvidia-smi` on a temporary kernel. |
| `jlab-mcp` | Run MCP server (stdio transport, used by Claude Code — not run manually) |

All commands accept `--debug` to enable verbose logging (status file reads, SLURM parameters, health check attempts, connection file paths) on stderr.

The SLURM job **survives Claude Code restarts**. You only need to run `jlab-mcp start` once per work session.

## Configuration

All settings are configurable via environment variables. No values are hardcoded for a specific cluster.

| Environment Variable | Default | Description |
|---|---|---|
| `JLAB_MCP_DIR` | `~/.jlab-mcp` | Base working directory |
| `JLAB_MCP_NOTEBOOK_DIR` | `./notebooks` | Notebook storage (relative to cwd) |
| `JLAB_MCP_SERVER_ROOT_DIR` | cwd | JupyterLab root directory (what the file browser sees) |
| `JLAB_MCP_LOG_DIR` | `~/.jlab-mcp/logs` | SLURM job logs |
| `JLAB_MCP_STATUS_DIR` | `~/.jlab-mcp/servers/{name}-{hash}` | Per-project status directory (auto-derived from cwd) |
| `JLAB_MCP_CONNECTION_DIR` | `~/.jlab-mcp/connections` | Connection info files |
| `JLAB_MCP_SLURM_PARTITION` | `gpu` | SLURM partition |
| `JLAB_MCP_SLURM_GRES` | `gpu:1` | SLURM generic resource |
| `JLAB_MCP_SLURM_CPUS` | `4` | CPUs per task |
| `JLAB_MCP_SLURM_MEM` | `32000` | Memory in MB |
| `JLAB_MCP_SLURM_TIME` | `4:00:00` | Wall clock time limit |
| `JLAB_MCP_SLURM_BIND_IP` | `0.0.0.0` | Address JupyterLab binds to on the compute node. Default listens on all interfaces and advertises the node's `$(hostname)`; a concrete IP binds that one interface and advertises that exact IP |
| `JLAB_MCP_SLURM_MODULES` | *(empty)* | Space-separated modules to load (e.g. `cuda/12.6`) |
| `JLAB_MCP_QUEUE_TIMEOUT` | `300` | Seconds `start` waits for the job to leave the queue. On timeout the job **stays queued** — rerun `jlab-mcp start` to resume waiting |
| `JLAB_MCP_READY_TIMEOUT` | `120` | Seconds to wait for JupyterLab once the job is running. On timeout the job is cancelled |
| `JLAB_MCP_PORT_MIN` | `18000` | Port range lower bound |
| `JLAB_MCP_PORT_MAX` | `19000` | Port range upper bound |
| `JLAB_MCP_RUN_MODE` | *(auto)* | `local` or `slurm` (auto-detects based on `sbatch` availability) |
| `JLAB_MCP_LOCAL_BIND_IP` | `0.0.0.0` | Address JupyterLab binds to in local mode. Default listens on all interfaces (UI reachable from other hosts / a container host); the same-host MCP server still connects over loopback. Set `127.0.0.1` to restrict JupyterLab to loopback only |

### Example: Cluster with A100 GPUs and CUDA module

```bash
export JLAB_MCP_SLURM_PARTITION=gpu1
export JLAB_MCP_SLURM_GRES=gpu:a100:1
export JLAB_MCP_SLURM_CPUS=18
export JLAB_MCP_SLURM_MEM=125000
export JLAB_MCP_SLURM_TIME=1-00:00:00
export JLAB_MCP_SLURM_MODULES="cuda/12.6"
```

## Claude Code Integration

Add to `~/.claude.json` or project `.mcp.json`:

```json
{
  "mcpServers": {
    "jlab-mcp": {
      "command": "jlab-mcp",
      "env": {
        "JLAB_MCP_SLURM_PARTITION": "gpu1",
        "JLAB_MCP_SLURM_GRES": "gpu:a100:1",
        "JLAB_MCP_SLURM_MODULES": "cuda/12.6"
      }
    }
  }
}
```

The MCP server uses the working directory to find `.venv` for the compute node. Claude Code launches from your project directory, so it picks up the right venv automatically.

## MCP Tools

| Tool | Description |
|---|---|
| `start_new_notebook` | Start kernel on shared server, create empty notebook (never overwrites — duplicate names get `_2`, `_3`, …). Shuts down all previous kernels/sessions first |
| `start_notebook` | Open an existing notebook, reusing its live kernel if one exists (state preserved); otherwise starts a fresh kernel. Returns cell contents |
| `execute_code` | Insert new code cell and execute it (supports positional insertion) |
| `edit_cell` | Edit code cell source only, no execution (clears stale outputs) |
| `run_cell` | Run existing cell without modifying its source |
| `add_markdown` | Add markdown cell to notebook (supports positional insertion) |
| `edit_markdown` | Edit an existing markdown cell's content |
| `delete_cell` | Delete a cell (code or markdown) by index |
| `execute_scratch` | Run code on a utility kernel (no notebook save, no session state) |
| `interrupt_kernel` | Interrupt running execution without shutting down the session |
| `shutdown_session` | Stop kernel (SLURM job stays alive) |
| `ping` | Lightweight health check — verify JupyterLab is reachable (no kernel needed) |
| `check_resources` | Check CPU, memory, and GPU usage on the compute node (no session needed) |

Resource: `jlab-mcp://server/status` — returns shared server info and active sessions.

### Session Lifecycle

- **`start_new_notebook`**: Creates a new kernel and a new notebook. Any previously running kernels/sessions are shut down (one active session at a time)
- **`start_notebook`**: Opens an existing notebook. If a session with a live kernel already exists for it, that session is returned with all state preserved; otherwise a fresh kernel is started
- **Restart kernel**: `shutdown_session` + `start_notebook(same_path)` = fresh kernel on same notebook
- **`shutdown_session`**: Kills the kernel only. The SLURM job keeps running.
- **SLURM job dies**: Next tool call returns an error. Run `jlab-mcp start` to restart.

## Testing

```bash
# Unit tests (no SLURM needed)
uv run python -m pytest tests/ --ignore=tests/test_tools.py -v

# Integration tests (requires running `jlab-mcp start` first)
uv run python -m pytest tests/test_tools.py -v -s --timeout=600
```

## Acknowledgments

This project is inspired by [goodfire-ai/scribe](https://github.com/goodfire-ai/scribe), which provides MCP-based notebook code execution for Claude. The tool interface design, image resizing approach, and notebook management patterns are adapted from scribe for use on HPC/SLURM clusters.

## License

MIT
