# How jlab-mcp Works: Step-by-Step

## 1. Start the Compute Node (`jlab-mcp start`)

In a separate terminal, run:

```bash
jlab-mcp start
```

This runs on the **login node** and does the following:

### 1a. Generate Connection Details

```python
# slurm.py → submit_job()
port = random.randint(18000, 19000)      # e.g. 18432
token = secrets.token_hex(24)             # e.g. "a3f8b2c1..."
connection_file = "~/.jlab-mcp/connections/jupyter-18432.conn"
```

### 1b. Render the SLURM Script

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

### 1c. Wait for SLURM Job (up to 5 min)

```
SLURM job 24215408 submitted, waiting in queue...
```

Polls `squeue` every 3 seconds until the job state is `RUNNING`.

### 1d. Wait for Connection File (up to 2 min)

The SLURM script on the compute node writes hostname/port/token to the shared filesystem. The login node polls until the file appears.

### 1e. Wait for JupyterLab (up to 2 min)

```python
requests.get("http://ravg1011:18432/api/status", headers={"Authorization": "token a3f8b2c1..."})
```

### 1f. Write Status File

When ready, writes `~/.jlab-mcp/server-status`:

```
STATE=ready
JOB_ID=24215408
HOSTNAME=ravg1011
PORT=18432
TOKEN=a3f8b2c1...
```

The MCP server reads this file to connect. The statusline script also reads it.

Terminal output:

```
SLURM job 24215408 submitted, waiting in queue...
Job running on ravg1011, JupyterLab starting...
JupyterLab ready at http://ravg1011:18432
```

## 2. MCP Server Connects

When Claude Code starts, it reads your `.mcp.json` and spawns `jlab-mcp` (no args = MCP server mode). The MCP server communicates with Claude Code over **stdin/stdout pipes** and advertises 7 tools + 1 resource.

The MCP server **does not manage SLURM** — it only reads the status file written by `jlab-mcp start` to find the running JupyterLab.

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

### 5a. `shutdown_session` — Kills Kernel Only

```python
requests.delete("http://ravg1011:18432/api/kernels/kernel-uuid-1234")
```

The **kernel** is stopped, but the **SLURM job stays alive** for other sessions. The notebook is preserved.

### 5b. User Exits Claude Code

When Claude Code terminates, it kills the MCP server process. The MCP server shuts down active kernels but **does not cancel the SLURM job**. The JupyterLab instance keeps running for the next Claude Code session.

### 5c. `jlab-mcp stop` — Cancels SLURM Job

When you're done for the day:

```bash
jlab-mcp stop
```

This runs `scancel` and removes the status file.

### 5d. Server Death (Walltime / Preemption)

If the SLURM job terminates while using Claude Code (walltime, preemption), the next MCP tool call will fail with:

```
JupyterLab not responding. Restart with: jlab-mcp start
```

Run `jlab-mcp start` in your other terminal to get a new compute node.

## Summary Diagram

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
