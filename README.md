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

Login and compute nodes share a filesystem. The MCP server submits a **single SLURM job** that starts JupyterLab on a compute node. All sessions create separate kernels on this shared server. Connection info (hostname, port, token) is exchanged via a file on the shared filesystem.

### Eager Startup

The SLURM job is submitted **immediately** when Claude Code starts the MCP server (in a background thread). By the time you make your first request, the compute node is often already running. Progress is logged to stderr:

```
[jlab-mcp] SLURM job 24215408 submitted (port=18432), waiting for compute node...
[jlab-mcp] SLURM job 24215408 running on ravg1011
[jlab-mcp] Waiting for JupyterLab at http://ravg1011:18432...
[jlab-mcp] JupyterLab ready at http://ravg1011:18432
```

### Server Death Detection

If the SLURM job terminates (walltime, preemption, node failure), the next session start automatically detects this and submits a new SLURM job.

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
| `start_new_session` | Start kernel on shared server, create empty notebook |
| `start_session_resume_notebook` | Resume existing notebook (re-executes all cells to restore state) |
| `start_session_continue_notebook` | Fork notebook with fresh kernel (no re-execution) |
| `execute_code` | Run Python code, append cell to notebook (returns text + images) |
| `edit_cell` | Edit and re-execute a cell (supports negative indexing) |
| `add_markdown` | Add markdown cell to notebook |
| `shutdown_session` | Stop kernel (SLURM job stays alive for other sessions) |

Resource: `jlab-mcp://server/status` — returns shared server info and active sessions.

### Session Lifecycle

- **First `start_new_session`**: Waits for the background SLURM job (usually already running)
- **Subsequent sessions**: Instant — just creates a new kernel on the same server
- **`shutdown_session`**: Kills the kernel only. The SLURM job keeps running.
- **User exits Claude Code**: Signal handler cancels the SLURM job and cleans up all sessions

## Status Line (Optional)

Show the compute node connection status in Claude Code's status bar. Save this script as `~/.claude/statusline.sh`:

```bash
#!/bin/bash
input=$(cat)

extract() {
    echo "$input" | grep -o "\"$1\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" | head -1 | sed 's/.*:.*"\([^"]*\)"/\1/'
}
extract_num() {
    echo "$input" | grep -o "\"$1\"[[:space:]]*:[[:space:]]*[0-9.]*" | head -1 | sed 's/.*:[[:space:]]*//';
}

cwd=$(extract current_dir)
[ -z "$cwd" ] && cwd=$(extract cwd)
dir=$(basename "${cwd:-~}")

branch=""
if [ -n "$cwd" ] && [ -d "$cwd/.git" ]; then
    branch=$(git --no-optional-locks -C "$cwd" rev-parse --abbrev-ref HEAD 2>/dev/null)
fi

model=$(extract display_name | sed -E 's/Claude ([0-9.]+) ([A-Z])[a-z]*/\1\2/g')
remaining=$(extract_num remaining_percentage)

# jlab-mcp: find active connection (skip stopped jobs)
jlab=""
conn_dir="$HOME/.jlab-mcp/connections"
if [ -d "$conn_dir" ]; then
    for f in $(ls -t "$conn_dir"/jupyter-*.conn 2>/dev/null); do
        last_status=$(grep '^STATUS=' "$f" 2>/dev/null | tail -1 | cut -d= -f2)
        [ "$last_status" = "stopped" ] && continue
        host=$(grep '^HOSTNAME=' "$f" 2>/dev/null | cut -d= -f2)
        jlab="${host:-starting}"
        break
    done
fi

s="$dir"
[ -n "$branch" ] && s="$dir:$branch"
[ -n "$model" ] && s="$s | $model"
[ -n "$remaining" ] && s="$s | ctx:$(printf '%.0f' "$remaining")%"
[ -n "$jlab" ] && s="$s | gpu:$jlab"

printf '[%s]' "$s"
```

Then enable it:

```bash
chmod +x ~/.claude/statusline.sh
```

Add to `~/.claude/settings.json`:

```json
{
  "statusLine": {
    "type": "command",
    "command": "~/.claude/statusline.sh"
  }
}
```

This displays the compute node hostname when connected:

```
[my-project:main | 4.6O | ctx:72% | gpu:ravg1011]
```

| Status | Meaning |
|---|---|
| `gpu:ravg1011` | Connected to compute node ravg1011 |
| `gpu:starting` | SLURM job submitted, waiting for JupyterLab |
| *(no gpu tag)* | No active jlab-mcp server |

## Testing

```bash
# Unit tests (no SLURM needed)
uv run python -m pytest tests/test_slurm.py tests/test_notebook.py tests/test_image_utils.py -v

# Integration tests (requires SLURM cluster)
uv run python -m pytest tests/test_tools.py -v -s --timeout=600
```

## Acknowledgments

This project is inspired by [goodfire-ai/scribe](https://github.com/goodfire-ai/scribe), which provides MCP-based notebook code execution for Claude. The tool interface design, image resizing approach, and notebook management patterns are adapted from scribe for use on HPC/SLURM clusters.

## License

MIT
