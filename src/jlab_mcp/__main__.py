import logging
import sys
import time

from jlab_mcp import config

_STATUS_FILE = config.STATUS_FILE


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "start":
            _cmd_start()
        elif cmd == "stop":
            _cmd_stop()
        elif cmd == "wait":
            _cmd_wait()
        elif cmd == "status":
            _cmd_status()
        else:
            print(f"Unknown command: {cmd}", file=sys.stderr)
            print("Usage: jlab-mcp [start|stop|wait|status]", file=sys.stderr)
            sys.exit(1)
        return

    # Default: run MCP server (stdio transport)
    from jlab_mcp.server import mcp

    mcp.run(transport="stdio")


# ---------------------------------------------------------------------------
# Status file helpers
# ---------------------------------------------------------------------------

def _write_status(state: str, **kwargs: str) -> None:
    """Write current server state to the status file."""
    lines = [f"STATE={state}"]
    for k, v in kwargs.items():
        lines.append(f"{k.upper()}={v}")
    _STATUS_FILE.write_text("\n".join(lines) + "\n")


def _clear_status() -> None:
    """Remove the status file."""
    _STATUS_FILE.unlink(missing_ok=True)


def _read_status():
    """Read the server-status file, return (state, info_dict)."""
    if not _STATUS_FILE.exists():
        return None, {}
    try:
        text = _STATUS_FILE.read_text()
        info = {}
        for line in text.strip().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                info[k] = v
        return info.get("STATE"), info
    except Exception:
        return None, {}


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def _cmd_start():
    """Submit SLURM job and wait until JupyterLab is ready."""
    from jlab_mcp.jupyter_client import JupyterLabClient
    from jlab_mcp.slurm import (
        is_job_running,
        submit_job,
        wait_for_connection_file,
        wait_for_job_running,
    )

    # Check if already running
    state, info = _read_status()
    if state == "ready":
        job_id = info.get("JOB_ID", "")
        if job_id and is_job_running(job_id):
            hostname = info.get("HOSTNAME", "?")
            print(f"JupyterLab already running on {hostname} (job {job_id})")
            return
        # Stale status file — job no longer running
        _clear_status()
    elif state in ("pending", "starting"):
        job_id = info.get("JOB_ID", "")
        if job_id and not is_job_running(job_id):
            _clear_status()

    # Submit new job
    job_id, conn_file, port, token = submit_job()
    _write_status("pending", job_id=job_id)
    print(f"SLURM job {job_id} submitted, waiting in queue...", flush=True)

    try:
        hostname = wait_for_job_running(job_id, timeout=300)
    except Exception as e:
        _write_status("error", job_id=job_id, message=str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    _write_status("starting", job_id=job_id, hostname=hostname)
    print(f"Job running on {hostname}, JupyterLab starting...", flush=True)

    try:
        conn_info = wait_for_connection_file(conn_file, timeout=120, job_id=job_id)
    except Exception as e:
        _write_status("error", job_id=job_id, hostname=hostname, message=str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    host = conn_info["HOSTNAME"]
    p = conn_info["PORT"]
    tok = conn_info["TOKEN"]

    client = JupyterLabClient(host, int(p), tok)
    print(f"Waiting for JupyterLab at {client.base_url}...", flush=True)

    start_t = time.time()
    while time.time() - start_t < 120:
        if not is_job_running(job_id):
            _write_status("error", job_id=job_id, message="Job cancelled")
            print(f"SLURM job {job_id} is no longer running.", file=sys.stderr)
            sys.exit(1)
        if client.health_check():
            break
        time.sleep(3)
    else:
        _write_status("error", job_id=job_id, message="JupyterLab timeout")
        print(
            f"Timeout: JupyterLab not responding at {client.base_url}",
            file=sys.stderr,
        )
        sys.exit(1)

    _write_status("ready", job_id=job_id, hostname=host, port=p, token=tok)
    print(f"JupyterLab ready at {client.base_url}", flush=True)


def _cmd_stop():
    """Cancel the SLURM job."""
    from jlab_mcp.slurm import cancel_job

    state, info = _read_status()
    job_id = info.get("JOB_ID")
    if not job_id:
        print("No running job found.")
        return

    try:
        cancel_job(job_id)
    except Exception as e:
        print(f"Error cancelling job {job_id}: {e}", file=sys.stderr)
    _clear_status()
    print(f"SLURM job {job_id} cancelled.")


def _cmd_wait():
    """Poll status file until JupyterLab is ready."""
    timeout = 600
    start = time.time()
    last_state = None

    print("Waiting for jlab-mcp compute node...", flush=True)

    while time.time() - start < timeout:
        state, info = _read_status()

        if state != last_state:
            last_state = state
            if state is None:
                print("  No status file yet (run `jlab-mcp start`)", flush=True)
            elif state == "pending":
                job_id = info.get("JOB_ID", "?")
                print(f"  SLURM job {job_id} submitted, waiting in queue...", flush=True)
            elif state == "starting":
                hostname = info.get("HOSTNAME", "?")
                print(f"  Job running on {hostname}, JupyterLab starting...", flush=True)
            elif state == "ready":
                hostname = info.get("HOSTNAME", "?")
                print(f"  Connected to {hostname} - ready!", flush=True)
                return
            elif state == "error":
                msg = info.get("MESSAGE", "unknown error")
                print(f"  Error: {msg}", file=sys.stderr, flush=True)
                sys.exit(1)
            elif state == "terminated":
                print("  Server terminated. Run `jlab-mcp start` to restart.", flush=True)
                sys.exit(1)

        time.sleep(2)

    print("Timeout waiting for compute node.", file=sys.stderr, flush=True)
    sys.exit(1)


def _cmd_status():
    """Print current server status, kernels, and GPU memory."""
    state, info = _read_status()

    if state is None:
        print("No jlab-mcp server running.")
        return

    print(f"State:    {state}")
    if "JOB_ID" in info:
        print(f"Job ID:   {info['JOB_ID']}")
    if "HOSTNAME" in info:
        print(f"Hostname: {info['HOSTNAME']}")
    if "PORT" in info:
        print(f"Port:     {info['PORT']}")

    if state != "ready":
        return

    from jlab_mcp.slurm import is_job_running

    job_id = info.get("JOB_ID", "")
    if job_id:
        running = is_job_running(job_id)
        print(f"Running:  {running}")
        if not running:
            return

    hostname = info.get("HOSTNAME", "")
    port = info.get("PORT", "")
    token = info.get("TOKEN", "")

    if not (hostname and port and token):
        return

    from jlab_mcp.jupyter_client import JupyterLabClient

    try:
        client = JupyterLabClient(hostname, int(port), token)
    except Exception as e:
        print(f"\n(connection error: {e})")
        return

    # --- Kernels ---
    try:
        kernels = client.list_kernels()
        print(f"\nKernels:  {len(kernels)}")
        for k in kernels:
            kid = k.get("id", "?")[:8]
            name = k.get("name", "?")
            kstate = k.get("execution_state", "?")
            last_activity = k.get("last_activity", "?")
            print(f"  {kid}  {name:12s}  {kstate:8s}  last_activity={last_activity}")
    except Exception as e:
        print(f"\nKernels:  (error: {e})")

    # --- GPU memory via temp kernel ---
    _print_gpu_status(client)


def _print_gpu_status(client):
    """Query GPU status by running nvidia-smi on a temporary kernel."""
    gpu_code = (
        "import subprocess\n"
        "r = subprocess.run(\n"
        "    ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',\n"
        "     '--format=csv,noheader,nounits'],\n"
        "    capture_output=True, text=True\n"
        ")\n"
        "print(r.stdout.strip())\n"
    )
    kernel_id = None
    try:
        kernel_id = client.start_kernel()
        time.sleep(2)
        outputs = client.execute_code(kernel_id, gpu_code, timeout=15)
        text = ""
        for out in outputs:
            if out["type"] == "text":
                text += out["content"]
        if text.strip():
            print(f"\nGPU:")
            for line in text.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    idx, name, mem_used, mem_total, util = parts[:5]
                    print(f"  [{idx}] {name}  {mem_used}/{mem_total} MiB  util={util}%")
                else:
                    print(f"  {line}")
        else:
            print(f"\nGPU:      (no output from nvidia-smi)")
    except Exception as e:
        print(f"\nGPU:      (error: {e})")
    finally:
        if kernel_id is not None:
            try:
                client.shutdown_kernel(kernel_id)
            except Exception:
                pass


if __name__ == "__main__":
    main()
