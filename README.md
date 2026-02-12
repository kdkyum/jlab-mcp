# jlab-mcp

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that enables Claude Code to execute Python code on GPU compute nodes via JupyterLab running on a SLURM cluster.

Inspired by and adapted from [goodfire-ai/scribe](https://github.com/goodfire-ai/scribe), which provides notebook-based code execution for Claude. This project adapts that approach for HPC/SLURM environments where GPU resources are allocated via job schedulers.

## Architecture

```
Claude Code (login node)
    ↕ stdio
MCP Server (login node)
    ↕ HTTP/WebSocket
JupyterLab (compute node, via sbatch)   ← one SLURM job, many kernels
    ↕
IPython Kernels (GPU access)
```

Login and compute nodes share a filesystem. The SLURM job is managed separately from the MCP server — you start it with `jlab-mcp start` and it keeps running across Claude Code sessions. All sessions create separate kernels on this shared server.

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
uv pip install torch --index-url https://download.pytorch.org/whl/cu126  # GPU support
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
| `jlab-mcp start [TIME]` | Submit SLURM job and wait until ready. Optional TIME overrides `JLAB_MCP_SLURM_TIME` (e.g. `24:00:00`) |
| `jlab-mcp stop` | Cancel the SLURM job |
| `jlab-mcp wait` | Poll status (check from another terminal) |
| `jlab-mcp status` | Print server state, active kernels, and GPU memory |
| `jlab-mcp` | Run MCP server (used by Claude Code, not run manually) |

The SLURM job **survives Claude Code restarts**. You only need to run `jlab-mcp start` once per work session.

## Configuration

All settings are configurable via environment variables. No values are hardcoded for a specific cluster.

| Environment Variable | Default | Description |
|---|---|---|
| `JLAB_MCP_DIR` | `~/.jlab-mcp` | Base working directory |
| `JLAB_MCP_NOTEBOOK_DIR` | `./notebooks` | Notebook storage (relative to cwd) |
| `JLAB_MCP_LOG_DIR` | `~/.jlab-mcp/logs` | SLURM job logs |
| `JLAB_MCP_CONNECTION_DIR` | `~/.jlab-mcp/connections` | Connection info files |
| `JLAB_MCP_SLURM_PARTITION` | `gpu` | SLURM partition |
| `JLAB_MCP_SLURM_GRES` | `gpu:1` | SLURM generic resource |
| `JLAB_MCP_SLURM_CPUS` | `4` | CPUs per task |
| `JLAB_MCP_SLURM_MEM` | `32000` | Memory in MB |
| `JLAB_MCP_SLURM_TIME` | `4:00:00` | Wall clock time limit |
| `JLAB_MCP_SLURM_MODULES` | *(empty)* | Space-separated modules to load (e.g. `cuda/12.6`) |
| `JLAB_MCP_PORT_MIN` | `18000` | Port range lower bound |
| `JLAB_MCP_PORT_MAX` | `19000` | Port range upper bound |

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
| `start_new_session` | Start kernel on shared server, create empty notebook |
| `start_session_resume_notebook` | Resume existing notebook (re-executes all cells to restore state) |
| `start_session_continue_notebook` | Fork notebook with fresh kernel (no re-execution) |
| `execute_code` | Run Python code, append cell to notebook (returns text + images) |
| `edit_cell` | Edit and re-execute a cell (supports negative indexing) |
| `add_markdown` | Add markdown cell to notebook |
| `execute_scratch` | Run code on a utility kernel (no notebook save, no session state) |
| `shutdown_session` | Stop kernel (SLURM job stays alive for other sessions) |

Resource: `jlab-mcp://server/status` — returns shared server info and active sessions.

### Session Lifecycle

- **`start_new_session`**: Creates a new kernel on the shared JupyterLab
- **`shutdown_session`**: Kills the kernel only. The SLURM job keeps running.
- **SLURM job dies**: Next tool call returns an error. Run `jlab-mcp start` to restart.

## Testing

```bash
# Unit tests (no SLURM needed)
uv run python -m pytest tests/test_slurm.py tests/test_notebook.py tests/test_image_utils.py -v

# Integration tests (requires running `jlab-mcp start` first)
uv run python -m pytest tests/test_tools.py -v -s --timeout=600
```

## Acknowledgments

This project is inspired by [goodfire-ai/scribe](https://github.com/goodfire-ai/scribe), which provides MCP-based notebook code execution for Claude. The tool interface design, image resizing approach, and notebook management patterns are adapted from scribe for use on HPC/SLURM clusters.

## License

MIT
