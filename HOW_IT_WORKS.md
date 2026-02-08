# How jlab-mcp Works: Step-by-Step

## 1. Starting the MCP Server

When Claude Code starts, it reads your `.mcp.json`:

```json
{
  "mcpServers": {
    "jlab-mcp": {
      "command": "jlab-mcp",
      "env": { "JLAB_MCP_SLURM_MODULES": "cuda/12.6" }
    }
  }
}
```

Claude Code spawns the process: `jlab-mcp` (the entry point from `pyproject.toml`).

This runs `__main__.py`:

```python
mcp.run(transport="stdio")
```

The FastMCP server starts and communicates with Claude Code over **stdin/stdout pipes**. It advertises 7 tools + 1 resource. At this point, **nothing is running on any compute node** — the server is just listening on the login node.

## 2. Claude Calls `start_new_session`

When Claude decides it needs to run code (e.g., you ask "train a model"), it calls:

```
start_new_session(experiment_name="my_experiment")
```

Here's what happens inside, step by step:

### 2a. Generate Connection Details

```python
# slurm.py → submit_job()
port = random.randint(18000, 19000)      # e.g. 18432
token = secrets.token_hex(24)             # e.g. "a3f8b2c1..."
connection_file = "~/.jlab-mcp/connections/jupyter-18432.conn"
```

### 2b. Render the SLURM Script

The template (`jupyter_slurm.sh.template`) is filled in:

```bash
#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=125000
...

module purge
module load cuda/12.6

source /shared/fs/my-project/.venv/bin/activate  # ← cwd's .venv

export JUPYTER_TOKEN="a3f8b2c1..."

# Write connection info immediately
umask 077
echo "HOSTNAME=$(hostname)" > ~/.jlab-mcp/connections/jupyter-18432.conn
echo "PORT=18432" >> ...
echo "TOKEN=a3f8b2c1..." >> ...

jupyter lab --no-browser --ip=0.0.0.0 --port=18432 --ServerApp.token="$JUPYTER_TOKEN" ...
```

This is written to a temp file and submitted:

```python
subprocess.run(["sbatch", "/tmp/script.sh"])
# stdout: "Submitted batch job 24215408"
# → job_id = "24215408"
```

### 2c. Wait for SLURM Job to Start (up to 5 min)

```python
# Polls every 3 seconds:
subprocess.run(["squeue", "-j", "24215408", "-h", "-o", "%T %N"])
```

```
Attempt 1: "PENDING "           ← waiting in queue
Attempt 2: "PENDING "
...
Attempt 8: "RUNNING ravg1011"   ← got a node!
```

Now we know: the job is running on `ravg1011`.

### 2d. Wait for Connection File (up to 2 min)

The SLURM script on `ravg1011` writes the connection file to the shared filesystem. The MCP server on the login node polls:

```python
# Polls every 2 seconds:
Path("~/.jlab-mcp/connections/jupyter-18432.conn").exists()  # True!
```

Reads it:

```
HOSTNAME=ravg1011
PORT=18432
TOKEN=a3f8b2c1...
STATUS=starting
```

### 2e. Wait for JupyterLab to Be Ready (up to 2 min)

```python
# Polls every 3 seconds:
requests.get("http://ravg1011:18432/api/status", headers={"Authorization": "token a3f8b2c1..."})
# Eventually returns 200 OK
```

This works because **login and compute nodes are on the same internal network**.

### 2f. Start a Kernel

```python
requests.post("http://ravg1011:18432/api/kernels", json={"name": "python3"})
# Returns: {"id": "kernel-uuid-1234"}
```

JupyterLab starts an IPython kernel process on `ravg1011` with GPU access.

### 2g. Create Notebook and Register Session

```python
notebook = nbformat.v4.new_notebook()
# Saved to: ~/.jlab-mcp/notebooks/my_experiment.ipynb

session = Session(
    session_id="031ec533",
    job_id="24215408",
    kernel_id="kernel-uuid-1234",
    jupyter_client=JupyterLabClient("ravg1011", 18432, "a3f8b2c1..."),
    notebook_path="~/.jlab-mcp/notebooks/my_experiment.ipynb",
)
sessions["031ec533"] = session
```

Returns to Claude:

```json
{"session_id": "031ec533", "notebook_path": "...", "job_id": "24215408", "hostname": "ravg1011"}
```

## 3. Claude Calls `execute_code`

```
execute_code(session_id="031ec533", code="import torch; print(torch.cuda.get_device_name(0))")
```

### 3a. Open WebSocket to Kernel

```python
ws = websocket.create_connection(
    "ws://ravg1011:18432/api/kernels/kernel-uuid-1234/channels?token=a3f8b2c1..."
)
```

### 3b. Send `execute_request` (Jupyter Message Protocol)

```python
ws.send(json.dumps({
    "header": {"msg_id": "abc123", "msg_type": "execute_request"},
    "content": {"code": "import torch; print(torch.cuda.get_device_name(0))"},
    "channel": "shell"
}))
```

### 3c. Receive Messages Until Kernel Goes Idle

```
← {"msg_type": "stream",  "content": {"text": "NVIDIA A100-SXM4-80GB"}}  # stdout
← {"msg_type": "status",  "content": {"execution_state": "idle"}}        # done
```

If the code produces a plot (`plt.show()` with `%matplotlib inline`):

```
← {"msg_type": "display_data", "content": {"data": {"image/png": "iVBOR..."}}}  # base64 PNG
```

### 3d. Process Outputs

- **Text** → returned as string
- **Images** → decoded, resized to 512px max, returned as FastMCP `Image` object → becomes MCP `ImageContent` → **Claude Code sees the actual plot**
- **Errors** → traceback returned as string

### 3e. Save to Notebook

The code and outputs are appended as a cell to `my_experiment.ipynb` on the shared filesystem.

## 4. Shutdown and Cleanup

There are two ways sessions end:

### 4a. Explicit: Claude Calls `shutdown_session`

```python
requests.delete("http://ravg1011:18432/api/kernels/kernel-uuid-1234")  # kill kernel
subprocess.run(["scancel", "24215408"])                                 # cancel SLURM job
Path("~/.jlab-mcp/connections/jupyter-18432.conn").unlink()             # delete token file
```

The compute node is released back to the SLURM pool.

### 4b. Implicit: User Quits Claude Code

When the user exits Claude Code (Ctrl+C, `/exit`, or closes the terminal):

1. Claude Code kills the `jlab-mcp` child process (sends SIGTERM)
2. The MCP server's **signal handler** catches SIGTERM
3. It iterates over all active sessions and runs `scancel` for each SLURM job
4. Connection files are deleted, then the process exits

```python
# Registered at startup:
atexit.register(_cleanup_all_sessions)
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)
```

This prevents orphaned SLURM jobs from wasting GPU time after Claude Code exits.

> **Note:** The only case cleanup won't happen is `kill -9` (SIGKILL), which cannot be intercepted. In that case, the SLURM job runs until its walltime expires.

## Summary Diagram

```
Login Node                          Shared FS                    Compute Node (ravg1011)
──────────                          ─────────                    ────────────────────────

Claude Code
  │ stdin/stdout
  ▼
MCP Server
  │
  ├─ sbatch script.sh ──────────────────────────────────────────► SLURM schedules job
  │                                                                │
  ├─ squeue (poll) ◄─────────────────────────────────────────────── job starts
  │                                                                │
  │                              connection file ◄──────────────── writes hostname/port/token
  ├─ read file ◄─────────────────────┘                             │
  │                                                                │
  ├─ GET /api/status ──────────────────────────────────────────────► JupyterLab ready
  ├─ POST /api/kernels ────────────────────────────────────────────► kernel started
  ├─ WS execute_request ──────────────────────────────────────────► runs code on GPU
  │◄── stream/display_data ◄───────────────────────────────────────┘
  │
  ├─ save cell ──────────────────► notebook.ipynb
  │
  ▼
Claude Code (sees text + images)
```
