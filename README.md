# jlab-mcp

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that enables Claude Code to execute Python code on GPU compute nodes via JupyterLab running on a SLURM cluster.

Inspired by and adapted from [goodfire-ai/scribe](https://github.com/goodfire-ai/scribe), which provides notebook-based code execution for Claude. This project adapts that approach for HPC/SLURM environments where GPU resources are allocated via job schedulers.

## Architecture

```
Claude Code (login node)
    ↕ stdio
MCP Server (login node)
    ↕ HTTP/WebSocket
JupyterLab (compute node, via sbatch)
    ↕
IPython Kernel (GPU access)
```

Login and compute nodes share a filesystem. The MCP server submits a SLURM job that starts JupyterLab on a compute node, then communicates with it over HTTP/WebSocket. Connection info (hostname, port, token) is exchanged via a file on the shared filesystem.

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
| `start_new_session` | Submit SLURM job, start kernel, create empty notebook |
| `start_session_resume_notebook` | Resume existing notebook (re-executes all cells) |
| `start_session_continue_notebook` | Fork notebook with fresh kernel |
| `execute_code` | Run Python code, append cell to notebook |
| `edit_cell` | Edit and re-execute a cell (supports negative indexing) |
| `add_markdown` | Add markdown cell to notebook |
| `shutdown_session` | Stop kernel, cancel SLURM job, clean up |

Resource: `jlab-mcp://server/status` — returns active sessions and job states.

## Testing

```bash
# Unit tests (no SLURM needed)
uv run python -m pytest tests/test_slurm.py tests/test_notebook.py tests/test_image_utils.py -v

# Integration tests (requires SLURM cluster)
uv run python -m pytest tests/test_tools.py -v -s --timeout=300
```

## Acknowledgments

This project is inspired by [goodfire-ai/scribe](https://github.com/goodfire-ai/scribe), which provides MCP-based notebook code execution for Claude. The tool interface design, image resizing approach, and notebook management patterns are adapted from scribe for use on HPC/SLURM clusters.

## License

MIT
