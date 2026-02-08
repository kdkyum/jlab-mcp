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
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
start_jupyter_background()   # submit SLURM job immediately
mcp.run(transport="stdio")   # start MCP server
```

Two things happen:

1. **Logging to stderr** — All status messages are written to stderr (stdout is reserved for MCP protocol). Claude Code captures and displays stderr from MCP servers.
2. **Background SLURM submission** — A background thread immediately submits the SLURM job so the compute node is warming up while Claude Code completes its handshake.

The FastMCP server starts and communicates with Claude Code over **stdin/stdout pipes**. It advertises 7 tools + 1 resource. By the time the user makes their first request, the JupyterLab server is often already running.

### What the User Sees (in Claude Code MCP logs)

```
2025-01-15 10:32:01 [jlab-mcp] SLURM job 24215408 submitted (port=18432), waiting for compute node...
2025-01-15 10:32:15 [jlab-mcp] SLURM job 24215408 running on ravg1011
2025-01-15 10:32:15 [jlab-mcp] Waiting for JupyterLab at http://ravg1011:18432...
2025-01-15 10:32:22 [jlab-mcp] JupyterLab ready at http://ravg1011:18432
```

If the SLURM job terminates (e.g., walltime reached):

```
2025-01-15 14:32:01 [jlab-mcp] JupyterLab server (job 24215408) terminated. All existing sessions are lost. Starting a new server...
2025-01-15 14:32:02 [jlab-mcp] SLURM job 24215500 submitted (port=18501), waiting for compute node...
```

## 2. Shared JupyterLab Server

Unlike the old design (one SLURM job per session), **all sessions share a single JupyterLab instance**. The SLURM job runs one JupyterLab server, and each session creates its own kernel on that server. This means:

- No waiting in the SLURM queue for second/third sessions
- Multiple notebooks can run in parallel on the same GPU node
- `shutdown_session` only kills the kernel, not the SLURM job
- The SLURM job is only cancelled when the MCP server exits

### 2a. Background Startup

When `__main__.py` calls `start_jupyter_background()`:

```python
# Runs in a background thread:
def _get_or_start_server():
    job_id, conn_file, port, token = submit_job()
    hostname = wait_for_job_running(job_id, timeout=300)
    conn_info = wait_for_connection_file(conn_file, timeout=120)
    client = JupyterLabClient(hostname, port, token)
    _wait_for_jupyter(client, timeout=120)
    _server = JupyterServer(job_id, client, hostname, conn_file)
```

### 2b. Generate Connection Details

```python
# slurm.py → submit_job()
port = random.randint(18000, 19000)      # e.g. 18432
token = secrets.token_hex(24)             # e.g. "a3f8b2c1..."
connection_file = "~/.jlab-mcp/connections/jupyter-18432.conn"
```

### 2c. Render the SLURM Script

The template (`jupyter_slurm.sh.template`) is filled in:

```bash
#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=125000
...

module purge
module load cuda/12.6

source /shared/fs/my-project/.venv/bin/activate  # cwd's .venv

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
```

### 2d. Wait for SLURM Job to Start (up to 5 min)

```python
# Polls every 3 seconds:
subprocess.run(["squeue", "-j", "24215408", "-h", "-o", "%T %N"])
```

```
Attempt 1: "PENDING "           <- waiting in queue
Attempt 2: "PENDING "
...
Attempt 8: "RUNNING ravg1011"   <- got a node!
```

### 2e. Wait for Connection File (up to 2 min)

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

### 2f. Wait for JupyterLab to Be Ready (up to 2 min)

```python
# Polls every 3 seconds:
requests.get("http://ravg1011:18432/api/status", headers={"Authorization": "token a3f8b2c1..."})
# Eventually returns 200 OK
```

This works because **login and compute nodes are on the same internal network**.

## 3. Claude Calls `start_new_session`

When Claude decides it needs to run code (e.g., you ask "train a model"), it calls:

```
start_new_session(experiment_name="my_experiment")
```

This **does not submit a new SLURM job**. Instead:

1. `_get_or_start_server()` returns the already-running shared server (or waits for the background thread to finish)
2. A new **kernel** is created on the existing JupyterLab:
   ```python
   requests.post("http://ravg1011:18432/api/kernels", json={"name": "python3"})
   # Returns: {"id": "kernel-uuid-1234"}
   ```
3. A new notebook is created on the shared filesystem
4. The session is registered

Returns to Claude:

```json
{"session_id": "031ec533", "notebook_path": "...", "job_id": "24215408", "hostname": "ravg1011"}
```

### Server Death Detection

If the SLURM job terminates (walltime, preemption, node failure), `_get_or_start_server()` detects this via `is_job_running()` and automatically submits a **new** SLURM job. All previous sessions are invalidated since their kernels are gone.

## 4. Claude Calls `execute_code`

```
execute_code(session_id="031ec533", code="import torch; print(torch.cuda.get_device_name(0))")
```

### 4a. Open WebSocket to Kernel

```python
ws = websocket.create_connection(
    "ws://ravg1011:18432/api/kernels/kernel-uuid-1234/channels?token=a3f8b2c1..."
)
```

### 4b. Send `execute_request` (Jupyter Message Protocol)

```python
ws.send(json.dumps({
    "header": {"msg_id": "abc123", "msg_type": "execute_request"},
    "content": {"code": "import torch; print(torch.cuda.get_device_name(0))"},
    "channel": "shell"
}))
```

### 4c. Receive Messages Until Kernel Goes Idle

```
<- {"msg_type": "stream",  "content": {"text": "NVIDIA A100-SXM4-80GB"}}  # stdout
<- {"msg_type": "status",  "content": {"execution_state": "idle"}}        # done
```

If the code produces a plot (`plt.show()` with `%matplotlib inline`):

```
<- {"msg_type": "display_data", "content": {"data": {"image/png": "iVBOR..."}}}  # base64 PNG
```

### 4d. Process Outputs

- **Text** -> returned as string in a list
- **Images** -> decoded, resized to 512px max, returned as FastMCP `Image` object -> becomes MCP `ImageContent` -> **Claude Code sees the actual plot**
- **Errors** -> traceback returned as string in a list

### 4e. Save to Notebook

The code and outputs are appended as a cell to `my_experiment.ipynb` on the shared filesystem.

## 5. Shutdown and Cleanup

### 5a. Explicit: Claude Calls `shutdown_session`

```python
requests.delete("http://ravg1011:18432/api/kernels/kernel-uuid-1234")  # kill kernel only
```

The **kernel** is stopped, but the **SLURM job stays alive** for other sessions. The notebook is preserved.

### 5b. Implicit: User Quits Claude Code

When the user exits Claude Code (Ctrl+C, `/exit`, or closes the terminal):

1. Claude Code kills the `jlab-mcp` child process (sends SIGTERM)
2. The MCP server's **signal handler** catches SIGTERM
3. It shuts down all kernels
4. It cancels the shared SLURM job via `scancel`
5. The connection file is deleted, then the process exits

```python
# Registered at startup:
atexit.register(_cleanup_all)
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)
```

This prevents orphaned SLURM jobs from wasting GPU time after Claude Code exits.

> **Note:** The only case cleanup won't happen is `kill -9` (SIGKILL), which cannot be intercepted. In that case, the SLURM job runs until its walltime expires.

### 5c. Server Death (Walltime / Preemption)

If the SLURM job terminates while the MCP server is still running (e.g., walltime limit reached), the next call to `start_new_session` will:

1. Detect the job is no longer running
2. Log a warning: "JupyterLab server terminated. Starting a new server..."
3. Clear all existing sessions (their kernels are gone)
4. Submit a new SLURM job and wait for it

## Summary Diagram

```
Login Node                          Shared FS                    Compute Node (ravg1011)
----------                          ---------                    -----------------------

Claude Code starts
  | stdin/stdout
  v
MCP Server starts
  |
  |-- [background thread] --------------------------------------------->
  |   sbatch script.sh ------------------------------------------------> SLURM schedules job
  |   squeue (poll) <--------------------------------------------------- job starts
  |                                 connection file <------------------- writes hostname/port/token
  |   read file <-----------------------+                                |
  |   GET /api/status -------------------------------------------------> JupyterLab ready
  |<- server ready! <---------------------------------------------------+
  |
  |-- Claude calls start_new_session
  |   POST /api/kernels -----------------------------------------------> kernel-1 started
  |   create notebook.ipynb
  |
  |-- Claude calls execute_code
  |   WS execute_request ----------------------------------------------> runs code on GPU
  |   <- stream/display_data <-------------------------------------------+
  |   save cell ----------------------> notebook.ipynb
  |
  |-- Claude calls start_new_session (2nd session — no SLURM wait!)
  |   POST /api/kernels -----------------------------------------------> kernel-2 started
  |
  |-- User exits Claude Code
  |   SIGTERM caught
  |   DELETE /api/kernels/kernel-1 ------------------------------------> kernel-1 stopped
  |   DELETE /api/kernels/kernel-2 ------------------------------------> kernel-2 stopped
  |   scancel 24215408 ------------------------------------------------> SLURM job cancelled
  v
Done
```
