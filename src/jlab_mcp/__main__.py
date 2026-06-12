import fcntl
import logging
import os
import signal
import sys
import time
from contextlib import contextmanager

from jlab_mcp import config

_STATUS_FILE = config.STATUS_FILE
_LOCK_FILE = config.STATUS_DIR / "lifecycle.lock"


@contextmanager
def _lifecycle_lock():
    """Serialize start/stop for this project so two concurrent commands
    can't both submit jobs or race the status file."""
    fd = os.open(_LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                "Another jlab-mcp start/stop is already running for this "
                "project. Follow it with: jlab-mcp wait",
                file=sys.stderr,
            )
            sys.exit(1)
        yield
    finally:
        os.close(fd)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )

    args = sys.argv[1:]
    debug = "--debug" in args
    if debug:
        args.remove("--debug")

    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("jlab-mcp").setLevel(logging.DEBUG)

    if args:
        cmd = args[0]
        if cmd == "start":
            time_limit = args[1] if len(args) > 1 else None
            _cmd_start(time_limit=time_limit, debug=debug)
        elif cmd == "stop":
            with _lifecycle_lock():
                _cmd_stop()
        elif cmd == "wait":
            _cmd_wait()
        elif cmd == "status":
            _cmd_status()
        else:
            print(f"Unknown command: {cmd}", file=sys.stderr)
            print("Usage: jlab-mcp [start|stop|wait|status] [--debug]", file=sys.stderr)
            sys.exit(1)
        return

    # Default: run MCP server (stdio transport)
    from jlab_mcp.server import mcp

    mcp.run(transport="stdio")


# ---------------------------------------------------------------------------
# Status file helpers
# ---------------------------------------------------------------------------

def _write_status(state: str, **kwargs: str) -> None:
    """Write current server state to the status file.

    Atomic (readers polling the file never see a partial write) and 0600
    (the ready state contains the JupyterLab token).
    """
    lines = [f"STATE={state}"]
    for k, v in kwargs.items():
        lines.append(f"{k.upper()}={v}")
    tmp = _STATUS_FILE.with_suffix(".tmp")
    fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, _STATUS_FILE)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


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

def _cmd_start(time_limit: str | None = None, debug: bool = False):
    """Start JupyterLab — dispatches to SLURM or local mode.

    Locking note: the lifecycle lock covers only the startup transition.
    Local mode blocks in the foreground for the server's whole lifetime,
    and holding the lock there would make `jlab-mcp stop` from another
    terminal impossible.
    """
    if config.RUN_MODE == "local":
        _cmd_start_local(debug=debug)
    else:
        with _lifecycle_lock():
            _cmd_start_slurm(time_limit=time_limit, debug=debug)


def _cancel_job_quietly(job_id: str) -> None:
    """Best-effort job cancellation (failure is reported, not raised)."""
    from jlab_mcp.slurm import cancel_job

    try:
        cancel_job(job_id)
    except Exception as e:
        print(f"Warning: could not cancel job {job_id}: {e}", file=sys.stderr)


def _cmd_start_slurm(time_limit: str | None = None, debug: bool = False):
    """Submit SLURM job and wait until JupyterLab is ready."""
    from jlab_mcp.jupyter_client import JupyterLabClient
    from jlab_mcp.slurm import (
        is_job_alive,
        is_job_running,
        submit_job,
    )

    log = logging.getLogger("jlab-mcp")

    if time_limit:
        config.SLURM_TIME = time_limit

    # Check for existing job
    state, info = _read_status()
    if debug:
        log.debug("Status file: state=%s info=%s", state, info)

    if state == "ready":
        job_id = info.get("JOB_ID", "")
        if job_id and is_job_running(job_id):
            # Verify JupyterLab is actually reachable before returning
            hostname = info.get("HOSTNAME", "?")
            port = info.get("PORT", "")
            token = info.get("TOKEN", "")
            if port and token:
                client = JupyterLabClient(hostname, int(port), token)
                if client.health_check():
                    print(f"JupyterLab already running on {hostname} (job {job_id})")
                    return
            # Job running but JupyterLab not responding
            print(f"Job {job_id} on {hostname} not responding, cancelling...", flush=True)
            _cancel_job_quietly(job_id)
        if debug:
            log.debug("Stale status file, clearing")
        _clear_status()

    elif state in ("pending", "starting", "error"):
        # A pending/starting job is resumable; an 'error' state can also
        # carry a live job with intact connection info (e.g. a previous
        # readiness wait failed transiently) — resume rather than throwing
        # away a queued/running allocation.
        job_id = info.get("JOB_ID", "")
        if job_id and is_job_alive(job_id):
            conn_file = info.get("CONN_FILE", "")
            port = info.get("PORT", "")
            token = info.get("TOKEN", "")
            if conn_file and port and token:
                print(f"Resuming wait for job {job_id}...", flush=True)
                _write_status("pending", job_id=job_id, conn_file=conn_file,
                              port=port, token=token)
                _wait_for_slurm_server(job_id, conn_file, int(port), token, debug)
                return
            # Missing connection info — cancel and resubmit
            if debug:
                log.debug("Status file missing connection info, cancelling job %s", job_id)
            _cancel_job_quietly(job_id)
        _clear_status()

    # Submit new job
    if debug:
        log.debug(
            "Submitting SLURM job: partition=%s gpu=%s time=%s",
            config.SLURM_PARTITION, config.SLURM_GRES, config.SLURM_TIME,
        )
    job_id, conn_file, port, token = submit_job()
    _write_status("pending", job_id=job_id, conn_file=conn_file,
                  port=str(port), token=token)
    print(f"SLURM job {job_id} submitted, waiting in queue...", flush=True)
    if debug:
        log.debug("Connection file: %s, token=%s", conn_file, token)

    _wait_for_slurm_server(job_id, conn_file, port, token, debug)


def _fail_running_job(job_id: str, conn_file: str, message: str):
    """A running job failed to become ready: cancel it (so it doesn't burn
    its allocation unreachable), clean up, record the error, and exit."""
    from jlab_mcp.slurm import cleanup_connection_file

    print(f"Cancelling SLURM job {job_id}: {message}", file=sys.stderr)
    _cancel_job_quietly(job_id)
    cleanup_connection_file(conn_file)
    _write_status("error", job_id=job_id, message=message)
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def _wait_for_slurm_server(
    job_id: str, conn_file: str, port: int | str, token: str,
    debug: bool = False,
):
    """Wait for a SLURM job to become a ready JupyterLab server.

    Handles Ctrl+C gracefully — the SLURM job stays alive and
    can be resumed by running ``jlab-mcp start`` again.
    """
    from jlab_mcp.jupyter_client import JupyterLabClient
    from jlab_mcp.slurm import (
        cleanup_connection_file,
        is_job_running,
        wait_for_connection_file,
        wait_for_job_running,
    )

    log = logging.getLogger("jlab-mcp")

    # Clean Ctrl+C: leave status file intact so next `start` can resume
    def _handle_interrupt(signum, frame):
        print("\nInterrupted. SLURM job stays in the queue.", flush=True)
        print("  Resume:  jlab-mcp start", flush=True)
        print("  Cancel:  jlab-mcp stop", flush=True)
        sys.exit(130)

    old_sigint = signal.signal(signal.SIGINT, _handle_interrupt)
    old_sigterm = signal.signal(signal.SIGTERM, _handle_interrupt)
    try:
        # Wait for job to start running (skip if already running)
        if not is_job_running(job_id):
            try:
                hostname = wait_for_job_running(
                    job_id, timeout=config.QUEUE_TIMEOUT
                )
            except TimeoutError as e:
                # The job is still healthily queued — keep the 'pending'
                # status so the next `jlab-mcp start` resumes this job
                # instead of cancelling it and going to the back of the
                # queue. (Long queue waits are routine on busy partitions.)
                print(f"{e}", file=sys.stderr)
                print("Job is still in the queue; status stays 'pending'.", flush=True)
                print("  Resume:  jlab-mcp start", flush=True)
                print("  Cancel:  jlab-mcp stop", flush=True)
                sys.exit(1)
            except Exception as e:
                # Job actually left the queue without running (failed)
                _write_status("error", job_id=job_id, message=str(e))
                cleanup_connection_file(conn_file)
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
            _write_status("starting", job_id=job_id, hostname=hostname,
                          conn_file=conn_file, port=str(port), token=token)
            print(f"Job running on {hostname}, JupyterLab starting...", flush=True)

        # Wait for connection file
        try:
            conn_info = wait_for_connection_file(
                conn_file, timeout=config.READY_TIMEOUT, job_id=job_id
            )
        except TimeoutError as e:
            _fail_running_job(job_id, conn_file, str(e))
        except Exception as e:
            _write_status("error", job_id=job_id, message=str(e))
            cleanup_connection_file(conn_file)
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        host = conn_info["HOSTNAME"]
        p = conn_info["PORT"]
        tok = conn_info["TOKEN"]

        # Wait for JupyterLab health check
        client = JupyterLabClient(host, int(p), tok)
        print(f"Waiting for JupyterLab at {client.base_url}...", flush=True)

        start_t = time.time()
        attempt = 0
        while time.time() - start_t < config.READY_TIMEOUT:
            if not is_job_running(job_id):
                _write_status("error", job_id=job_id, message="Job cancelled")
                cleanup_connection_file(conn_file)
                print(f"SLURM job {job_id} is no longer running.", file=sys.stderr)
                sys.exit(1)
            attempt += 1
            if debug:
                log.debug("Health check attempt %d -> %s", attempt, client.base_url)
            if client.health_check():
                if debug:
                    log.debug("Health check passed after %d attempts", attempt)
                break
            time.sleep(3)
        else:
            _fail_running_job(
                job_id, conn_file,
                f"JupyterLab not responding at {client.base_url}",
            )

        _write_status("ready", job_id=job_id, hostname=host, port=p, token=tok)
        # The connection file has served its purpose (the token now lives in
        # the 0600 status file) — don't leave token-bearing files behind
        cleanup_connection_file(conn_file)
        if debug:
            log.debug("Status file written: state=ready host=%s port=%s", host, p)
        print(f"JupyterLab ready at {client.base_url}", flush=True)
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)


def _cmd_start_local(debug: bool = False):
    """Start JupyterLab as a local subprocess (foreground).

    The lifecycle lock is held only through the startup transition and
    released before the foreground wait — otherwise `jlab-mcp stop` from
    another terminal could never acquire it.
    """
    from jlab_mcp.jupyter_client import JupyterLabClient
    from jlab_mcp.local import is_local_running, start_jupyter_local, stop_jupyter_local

    log = logging.getLogger("jlab-mcp")

    with _lifecycle_lock():
        # Check if already running
        state, info = _read_status()
        if debug:
            log.debug("Status file: state=%s info=%s", state, info)
        if state == "ready" and info.get("MODE") == "local":
            pid = int(info.get("PID", "0"))
            if pid and is_local_running(pid):
                hostname = info.get("HOSTNAME", "?")
                port = info.get("PORT", "?")
                print(f"JupyterLab already running at {hostname}:{port} (PID {pid})")
                return
            if debug:
                log.debug("Stale status file (process not running), clearing")
            _clear_status()

        if debug:
            log.debug("Starting local JupyterLab subprocess")
        proc, hostname, port, token = start_jupyter_local()
        _write_status(
            "starting", mode="local", pid=str(proc.pid),
            hostname=hostname, port=str(port), token=token,
        )
        print(f"JupyterLab starting (PID {proc.pid})...", flush=True)
        if debug:
            log.debug("Subprocess PID=%d, target=%s:%s, token=%s", proc.pid, hostname, port, token)

        # Wait for health check
        client = JupyterLabClient(hostname, port, token)
        start_t = time.time()
        attempt = 0
        try:
            while time.time() - start_t < 60:
                if proc.poll() is not None:
                    # Keep the error status so `jlab-mcp wait` fails fast
                    # instead of polling for a file that never appears
                    _write_status("error", mode="local", message="Process exited early")
                    print(
                        f"JupyterLab process exited with code {proc.returncode}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                attempt += 1
                if debug:
                    log.debug("Health check attempt %d -> %s", attempt, client.base_url)
                if client.health_check():
                    if debug:
                        log.debug("Health check passed after %d attempts", attempt)
                    break
                time.sleep(1)
            else:
                stop_jupyter_local(proc.pid)
                _write_status("error", mode="local", message="JupyterLab timeout")
                print(
                    f"Timeout: JupyterLab not responding at {client.base_url}",
                    file=sys.stderr,
                )
                sys.exit(1)
        except KeyboardInterrupt:
            print("\nInterrupted during startup, stopping JupyterLab...", flush=True)
            stop_jupyter_local(proc.pid)
            _clear_status()
            sys.exit(130)

        _write_status(
            "ready", mode="local", pid=str(proc.pid),
            hostname=hostname, port=str(port), token=token,
        )
        if debug:
            log.debug("Status file written: state=ready pid=%d host=%s port=%s", proc.pid, hostname, port)
        print(f"JupyterLab ready at {client.base_url} (PID {proc.pid})", flush=True)
        print("Press Ctrl+C to stop.", flush=True)

    # Block in foreground, clean up on signal
    def _shutdown(signum, frame):
        print("\nShutting down JupyterLab...", flush=True)
        stop_jupyter_local(proc.pid)
        _clear_status()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        proc.wait()
    except KeyboardInterrupt:
        _shutdown(None, None)
    finally:
        _clear_status()


def _cmd_stop():
    """Stop JupyterLab (SLURM or local)."""
    state, info = _read_status()

    if state is None:
        print("No jlab-mcp server found.")
        return

    mode = info.get("MODE", "slurm")

    if mode == "local":
        from jlab_mcp.local import stop_jupyter_local

        try:
            pid = int(info.get("PID", "0"))
        except (ValueError, TypeError):
            print("Invalid PID in status file, clearing.")
            _clear_status()
            return
        if not pid:
            print("No running local server found.")
            _clear_status()
            return
        stop_jupyter_local(pid)
        _clear_status()
        print(f"Local JupyterLab (PID {pid}) stopped.")
    else:
        from jlab_mcp.slurm import cancel_job, cleanup_connection_file

        job_id = info.get("JOB_ID")
        conn_file = info.get("CONN_FILE", "")
        if not job_id:
            print("No running job found.")
            _clear_status()
            return
        try:
            cancel_job(job_id)
        except Exception as e:
            # Keep the status file so the job stays tracked and stop can
            # be retried — clearing it here would orphan a live job.
            print(f"Error cancelling job {job_id}: {e}", file=sys.stderr)
            print("Status kept — run `jlab-mcp stop` again to retry.", file=sys.stderr)
            sys.exit(1)
        if conn_file:
            cleanup_connection_file(conn_file)
        _clear_status()
        print(f"SLURM job {job_id} cancelled.")


def _cmd_wait():
    """Poll status file until JupyterLab is ready."""
    timeout = 600
    start = time.time()
    last_state = None

    print("Waiting for jlab-mcp server...", flush=True)

    while time.time() - start < timeout:
        state, info = _read_status()

        if state != last_state:
            last_state = state
            if state is None:
                print("  No status file yet (run `jlab-mcp start`)", flush=True)
            elif state == "pending":
                job_id = info.get("JOB_ID", "?")
                print(f"  Job {job_id} submitted, waiting in queue...", flush=True)
            elif state == "starting":
                hostname = info.get("HOSTNAME", "?")
                print(f"  Running on {hostname}, JupyterLab starting...", flush=True)
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

    print("Timeout waiting for server.", file=sys.stderr, flush=True)
    sys.exit(1)


def _cmd_status():
    """Print current server status, kernels, and GPU memory."""
    state, info = _read_status()

    if state is None:
        print("No jlab-mcp server running.")
        return

    mode = info.get("MODE", "slurm")
    print(f"State:    {state}")
    print(f"Mode:     {mode}")
    if "JOB_ID" in info:
        print(f"Job ID:   {info['JOB_ID']}")
    if "PID" in info:
        print(f"PID:      {info['PID']}")
    if "HOSTNAME" in info:
        print(f"Hostname: {info['HOSTNAME']}")
    if "PORT" in info:
        print(f"Port:     {info['PORT']}")
    if "TOKEN" in info:
        print(f"Token:    {info['TOKEN']}")

    if state != "ready":
        return

    # Check if the underlying process/job is alive
    if mode == "local":
        from jlab_mcp.local import is_local_running

        pid = int(info.get("PID", "0"))
        if pid:
            running = is_local_running(pid)
            print(f"Running:  {running}")
            if not running:
                return
    else:
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
