# How jlab-mcp Works: Step-by-Step

jlab-mcp supports two modes: **SLURM** (HPC clusters) and **local** (laptops/workstations). Mode is auto-detected — if `sbatch` is on PATH, SLURM mode is used; otherwise, local mode. Override with `JLAB_MCP_RUN_MODE=local|slurm`.

## 1. Start JupyterLab (`jlab-mcp start`)

In a separate terminal, run:

```bash
jlab-mcp start
```

### SLURM Mode

On an HPC cluster, this runs on the **login node** and does the following:

#### 1a. Generate Connection Details

```python
# slurm.py → submit_job()
port = random.randint(18000, 19000)      # e.g. 18432
token = secrets.token_hex(24)             # e.g. "a3f8b2c1..."
connection_file = "~/.jlab-mcp/connections/jupyter-18432.conn"
```

#### 1b. Render the SLURM Script

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

This is submitted via `sbatch`.

#### 1c. Wait for SLURM Job (up to 5 min)

```
SLURM job 24215408 submitted, waiting in queue...
```

Polls `squeue` every 3 seconds until the job state is `RUNNING`.

#### 1d. Wait for Connection File (up to 2 min)

The SLURM script on the compute node writes hostname/port/token to the shared filesystem. The login node polls until the file appears.

#### 1e. Wait for JupyterLab (up to 2 min)

```python
requests.get("http://ravg1011:18432/api/status", headers={"Authorization": "token a3f8b2c1..."})
```

#### 1f. Write Status File

When ready, writes the per-project status file:

```
STATE=ready
JOB_ID=24215408
HOSTNAME=ravg1011
PORT=18432
TOKEN=a3f8b2c1...
```

Terminal output:

```
SLURM job 24215408 submitted, waiting in queue...
Job running on ravg1011, JupyterLab starting...
JupyterLab ready at http://ravg1011:18432
```

### Local Mode

On machines without SLURM, `jlab-mcp start` spawns JupyterLab as a local subprocess:

#### 1a. Generate Port and Token

```python
# local.py → start_jupyter_local()
port = random.randint(18000, 19000)
token = secrets.token_hex(24)
```

#### 1b. Spawn JupyterLab Subprocess

```python
subprocess.Popen([sys.executable, "-m", "jupyter", "lab",
    "--ip=127.0.0.1", "--port=18432",
    "--IdentityProvider.token=a3f8b2c1...",
    "--no-browser", "--notebook-dir=./notebooks"])
```

Logs are written to `~/.jlab-mcp/logs/jupyter-local-18432.log`.

#### 1c. Wait for Health Check (up to 1 min)

Polls `GET /api/status` every second. Also checks `proc.poll()` to detect early crashes.

#### 1d. Write Status File

Same format as SLURM mode, with `MODE=local` and `PID=` instead of `JOB_ID=`:

```
STATE=ready
MODE=local
PID=12345
HOSTNAME=127.0.0.1
PORT=18432
TOKEN=a3f8b2c1...
```

The process stays in the **foreground** — press Ctrl+C to stop.

Terminal output:

```
JupyterLab starting (PID 12345)...
JupyterLab ready at http://127.0.0.1:18432 (PID 12345)
Press Ctrl+C to stop.
```

The MCP server reads this same status file to connect, regardless of mode.

## 2. MCP Server Connects

When Claude Code starts, it reads your `.mcp.json` and spawns `jlab-mcp` (no args = MCP server mode). The MCP server communicates with Claude Code over **stdin/stdout pipes** and advertises 10 tools + 1 resource.

The MCP server **does not manage SLURM** — it only reads the status file written by `jlab-mcp start` to find the running JupyterLab.

### Diagnostic Tools (no session required)

**`ping`** — Lightweight health check that reads the status file and calls `GET /api/status` on JupyterLab. No kernel needed. Returns connection status, hostname, and active session count.

```python
# Reads status file → makes one HTTP GET → returns dict
{"status": "ok", "hostname": "ravg1011", "url": "http://ravg1011:18432", "healthy": true, "active_sessions": 1}
```

### Resource Monitoring

**`check_resources`** — Runs a lightweight script on the session's kernel to report CPU, memory, and GPU usage:

```python
check_resources(session_id="031ec533")
# Returns:
{
    "cpu": {"count": 18, "load_1m": 0.5, "load_5m": 0.3, "load_15m": 0.2},
    "memory": {"total_mb": 125000, "used_mb": 8500, "available_mb": 116500, "percent_used": 6.8},
    "gpu": [{"index": 0, "name": "NVIDIA A100-SXM4-80GB", "memory_used_mb": 2048,
             "memory_total_mb": 81920, "utilization_percent": 35, "temperature_c": 42}]
}
```

This executes on the **same kernel** as your session — it reads `/proc/meminfo` for memory, `os.getloadavg()` for CPU, and `nvidia-smi` for GPU. The code is not saved to the notebook.

## 3. Claude Calls `start_new_session`

```
start_new_session(experiment_name="my_experiment")
```

1. `_get_or_start_server()` reads the status file and connects to JupyterLab
2. A new **kernel** is created:
   ```python
   requests.post("http://ravg1011:18432/api/kernels", json={"name": "python3"})
   # Returns: {"id": "kernel-uuid-1234"}
   ```
3. A new notebook is created on the shared filesystem
4. Returns: `{"session_id": "031ec533", "notebook_path": "...", "hostname": "ravg1011"}`

## 4. Claude Calls `execute_code`

```
execute_code(session_id="031ec533", code="import torch; print(torch.cuda.get_device_name(0))")
```

### 4a. Auto-interrupt Previous Execution

Before sending new code, the kernel is interrupted to cancel any still-running execution from a previously cancelled tool call. This is a no-op on an idle kernel.

```python
requests.post("http://ravg1011:18432/api/kernels/kernel-uuid-1234/interrupt")
```

### 4b. Open WebSocket to Kernel

```python
ws = websocket.create_connection(
    "ws://ravg1011:18432/api/kernels/kernel-uuid-1234/channels?token=a3f8b2c1..."
)
```

### 4c. Send `execute_request` (Jupyter Message Protocol)

```python
ws.send(json.dumps({
    "header": {"msg_id": "abc123", "msg_type": "execute_request"},
    "content": {"code": "import torch; print(torch.cuda.get_device_name(0))"},
    "channel": "shell"
}))
```

### 4d. Receive Messages Until Kernel Goes Idle

```
<- {"msg_type": "stream",  "content": {"text": "NVIDIA A100-SXM4-80GB"}}  # stdout
<- {"msg_type": "status",  "content": {"execution_state": "idle"}}        # done
```

If the code produces a plot (`plt.show()` with `%matplotlib inline`):

```
<- {"msg_type": "display_data", "content": {"data": {"image/png": "iVBOR..."}}}  # base64 PNG
```

#### Kernel Death Detection

If the kernel dies during execution (OOM, segfault, CUDA error), two things can happen on the WebSocket:

1. **WebSocket closes** — the kernel process is gone, JupyterLab closes the connection. `ws.recv()` raises `WebSocketConnectionClosedException`, caught immediately.

2. **Kernel auto-restarts** — JupyterLab sends a broadcast status message (no `parent_header`):
   ```
   <- {"msg_type": "status", "content": {"execution_state": "restarting"}}
   ```
   This is checked **before** the `parent_msg_id` filter so it's not skipped.

Both paths return immediately with a `KernelDied` error:
```
Error: KernelDied: Kernel restarting during execution (likely OOM or crash).
All in-memory state is lost. Start a new session to continue.
```

Without this detection, a kernel death would cause the WebSocket loop to hang silently for up to 300 seconds (the execution timeout), leaving Claude Code with no feedback.

### 4e. Process Outputs

- **Text** -> returned as string in a list
- **Images** -> decoded, resized to 512px max, returned as FastMCP `Image` object -> becomes MCP `ImageContent` -> **Claude Code sees the actual plot**
- **Errors** -> traceback returned as string in a list

### 4f. Save to Notebook

The code and outputs are appended as a cell to `my_experiment.ipynb` on the shared filesystem.

## 5. Shutdown and Cleanup

### 5a. `shutdown_session` — Kills Kernel Only

```python
requests.delete("http://ravg1011:18432/api/kernels/kernel-uuid-1234")
```

The **kernel** is stopped, but the **SLURM job stays alive** for other sessions. The notebook is preserved.

### 5b. User Exits Claude Code

When Claude Code terminates, it kills the MCP server process. The MCP server shuts down active kernels but **does not cancel the SLURM job**. The JupyterLab instance keeps running for the next Claude Code session.

### 5c. `jlab-mcp stop` — Stops JupyterLab

When you're done for the day:

```bash
jlab-mcp stop
```

- **SLURM mode**: runs `scancel` and removes the status file
- **Local mode**: sends `SIGTERM` to the subprocess and removes the status file (or just press Ctrl+C in the `jlab-mcp start` terminal)

### 5d. Kernel Death (OOM / Crash)

If a kernel dies during execution (CUDA OOM, segfault, etc.), JupyterLab auto-restarts the kernel process. The WebSocket detects this immediately via `status: restarting` or connection close and returns a `KernelDied` error to Claude Code. All in-memory state (variables, models, imports) is lost — start a new session to continue.

### 5e. Server Death (Walltime / Preemption / Crash)

If JupyterLab terminates while using Claude Code (SLURM walltime, preemption, or local process crash), the next MCP tool call will fail with:

```
JupyterLab not responding. Restart with: jlab-mcp start
```

Run `jlab-mcp start` in your other terminal to get a new server.

## Summary Diagram

### SLURM Mode

```
Terminal 1 (login node)              Shared FS                Compute Node (ravg1011)
-------------------                  ---------                -----------------------

jlab-mcp start
  |
  | sbatch script.sh ------------------------------------------------> SLURM schedules job
  | squeue (poll) <--------------------------------------------------- job starts
  |                               connection file <------------------- writes hostname/port/token
  | read file <-----------------------+                                |
  | GET /api/status -------------------------------------------------> JupyterLab ready
  | writes server-status
  v
"JupyterLab ready at http://ravg1011:18432"


Terminal 2 (login node)
-------------------

claude (starts Claude Code)
  | stdin/stdout
  v
MCP Server starts
  | reads server-status
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
  |-- User exits Claude Code
  |   MCP server stops (kernels cleaned up, SLURM job stays alive)
  v


Terminal 1
-------------------

jlab-mcp stop
  | scancel 24215408 -------------------------------------------------> SLURM job cancelled
  v
Done
```

### Local Mode

```
Terminal 1                               Local Machine
-------------------                      -------------

jlab-mcp start
  |
  | subprocess.Popen("jupyter lab") --> JupyterLab starts (PID 12345)
  | GET /api/status (poll) <----------- responds when ready
  | writes server-status
  v
"JupyterLab ready at http://127.0.0.1:18432 (PID 12345)"
"Press Ctrl+C to stop."
  | (blocks in foreground)


Terminal 2
-------------------

claude (starts Claude Code)
  | stdin/stdout
  v
MCP Server starts
  | reads server-status (same format, MODE=local)
  |
  |-- Claude calls start_new_session
  |   POST /api/kernels ----------------> kernel-1 started
  |   create notebook.ipynb
  |
  |-- Claude calls execute_code
  |   WS execute_request ----------------> runs code locally
  |   <- stream/display_data <-----------+
  |   save cell --> notebook.ipynb
  |
  |-- User exits Claude Code
  |   MCP server stops (kernels cleaned up, JupyterLab stays alive)
  v


Terminal 1
-------------------

Ctrl+C (or jlab-mcp stop from another terminal)
  | SIGTERM to PID 12345 --> JupyterLab stopped
  v
Done
```
