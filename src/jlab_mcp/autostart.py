"""Autostart: let the MCP server launch and monitor `jlab-mcp start` itself.

The CLI (`jlab-mcp start`) stays the single owner of the startup state
machine (job submission, queue resume, status file writes). This module
only adds what the MCP server needs to drive it from tool calls:

* remember the user's run-mode choice (local vs slurm) per project,
* bootstrap a bare project (uv init / uv add jupyterlab) when the
  project has no usable .venv yet,
* spawn `jlab-mcp start` fully detached so it survives the MCP server
  (in local mode that process IS JupyterLab's parent),
* poll the status file so a tool can report readiness to the user.
"""

import fcntl
import logging
import os
import re
import shutil
import subprocess
import sys
import time

from jlab_mcp import config
from jlab_mcp.jupyter_client import JupyterLabClient

logger = logging.getLogger("jlab-mcp.autostart")

VALID_MODES = ("local", "slurm")
MODE_FILE = config.STATUS_DIR / "run-mode"
SLURM_OPTS_FILE = config.STATUS_DIR / "slurm-options"
START_LOG = config.STATUS_DIR / "start.log"
_LOCK_FILE = config.STATUS_DIR / "lifecycle.lock"

# SLURM walltime shapes: "30", "4:00:00", "1-00:00:00", ...
_TIME_LIMIT_RE = re.compile(r"[0-9][0-9:\-]{0,15}")

# Per-option validation + env var each maps onto in the spawned
# `jlab-mcp start` (config.py reads them). Values land in sbatch
# arguments, so the patterns are strict.
_SLURM_OPT_SPECS = {
    "time": (_TIME_LIMIT_RE, "JLAB_MCP_SLURM_TIME"),
    "partition": (re.compile(r"[A-Za-z0-9_.,-]{1,64}"), "JLAB_MCP_SLURM_PARTITION"),
    "gres": (re.compile(r"[A-Za-z0-9:_,/.-]{1,64}"), "JLAB_MCP_SLURM_GRES"),
    "cpus": (re.compile(r"[0-9]{1,4}"), "JLAB_MCP_SLURM_CPUS"),
    "mem_mb": (re.compile(r"[0-9]{1,9}"), "JLAB_MCP_SLURM_MEM"),
}

# Handle to the most recently spawned start process (this MCP server
# process only) — used to reap it and to tell "bootstrap still running"
# apart from "start died". A server restart loses the handle; the flock
# check below still covers the `jlab-mcp start` phase.
_start_proc: subprocess.Popen | None = None
_last_spawn: float = 0.0

# The project .venv is what runs on the compute node (SLURM template) or
# hosts the local subprocess, so a bare project (only .mcp.json) must be
# set up before `jlab-mcp start` can succeed. `jlab-mcp start` itself is
# run with the MCP server's interpreter — the project venv doesn't
# contain jlab-mcp.
_BOOTSTRAP_SCRIPT = """\
set -u
cd "$JLAB_MCP_PROJECT_DIR"
if command -v uv >/dev/null 2>&1; then
    if [ ! -f pyproject.toml ]; then
        echo "[jlab-mcp] no pyproject.toml -- running: uv init --bare"
        uv init --bare || echo "[jlab-mcp] uv init failed; continuing"
    fi
    if [ ! -x .venv/bin/jupyter-lab ]; then
        echo "[jlab-mcp] project venv missing jupyterlab -- running: uv add jupyterlab ipykernel matplotlib numpy"
        uv add jupyterlab ipykernel matplotlib numpy \\
            || echo "[jlab-mcp] uv add failed; continuing (may fall back to the tool environment)"
    fi
else
    echo "[jlab-mcp] uv not found -- skipping project setup"
fi
exec "$JLAB_MCP_PYTHON" -m jlab_mcp start
"""


# ---------------------------------------------------------------------------
# Run-mode persistence
# ---------------------------------------------------------------------------

def read_saved_mode() -> str | None:
    """Return the mode saved for this project, or None if not chosen yet."""
    try:
        mode = MODE_FILE.read_text().strip().lower()
    except OSError:
        return None
    return mode if mode in VALID_MODES else None


def save_mode(mode: str) -> None:
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid run mode: {mode!r} (expected 'local' or 'slurm')")
    MODE_FILE.write_text(mode + "\n")


def resolve_mode(explicit: str | None = None) -> str | None:
    """Resolve the run mode: explicit arg > env var > saved choice > None.

    None means nobody has chosen yet — the caller should ask the user
    rather than silently auto-detecting. An explicit choice is persisted.
    """
    if explicit:
        mode = explicit.strip().lower()
        save_mode(mode)  # raises ValueError on junk
        return mode
    env_mode = os.environ.get("JLAB_MCP_RUN_MODE", "").strip().lower()
    if env_mode in VALID_MODES:
        return env_mode
    return read_saved_mode()


def validate_slurm_options(opts: dict) -> dict:
    """Validate/normalize SLURM options ({time, partition, gres, cpus, mem_mb}).

    Values end up in sbatch parameters, so anything not matching the
    strict per-option pattern is rejected. Empty/zero values are dropped
    (meaning: keep the current default).
    """
    clean = {}
    for key, value in opts.items():
        if key not in _SLURM_OPT_SPECS:
            raise ValueError(f"Unknown SLURM option: {key!r}")
        if value in ("", 0, None):
            continue
        pattern, _ = _SLURM_OPT_SPECS[key]
        text = str(value).strip()
        if not pattern.fullmatch(text):
            raise ValueError(f"Invalid SLURM {key}: {value!r}")
        clean[key] = text
    return clean


def survey_slurm_partitions() -> list[dict]:
    """Live partition survey via sinfo, so the agent can propose concrete
    resource options (partition, GPU type, walltime ceiling) instead of
    guessing. Best effort: returns [] when sinfo is missing or fails.

    Rows with identical configuration are merged (node counts summed);
    socket annotations in gres ("gpu:a100:4(S:0-1)") are stripped.
    """
    if not shutil.which("sinfo"):
        return []
    try:
        r = subprocess.run(
            ["sinfo", "-h", "-o", "%P|%a|%l|%D|%c|%m|%G"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if r.returncode != 0:
        return []

    merged: dict[tuple, dict] = {}
    for line in r.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) != 7:
            continue
        name, avail, timelimit, nodes, cpus, mem, gres = (
            p.strip() for p in parts
        )
        is_default = name.endswith("*")
        name = name.rstrip("*")
        gres = re.sub(r"\(S:[^)]*\)", "", gres)
        if gres == "(null)":
            gres = ""
        key = (name, avail, timelimit, cpus, mem, gres)
        node_count = int(nodes) if nodes.isdigit() else 0
        if key in merged:
            merged[key]["nodes"] += node_count
        else:
            row = {
                "partition": name,
                "available": avail,
                "max_time": timelimit,
                "nodes": node_count,
                "cpus_per_node": cpus,
                "memory_mb_per_node": mem,
                "gres": gres,
            }
            if is_default:
                row["default_partition"] = True
            merged[key] = row
    return list(merged.values())


def read_slurm_options() -> dict:
    """SLURM options saved for this project (empty dict if none)."""
    try:
        text = SLURM_OPTS_FILE.read_text()
    except OSError:
        return {}
    opts = {}
    for line in text.strip().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            opts[k] = v
    try:
        return validate_slurm_options(opts)
    except ValueError:
        return {}


def save_slurm_options(opts: dict) -> None:
    opts = validate_slurm_options(opts)
    lines = [f"{k}={v}" for k, v in sorted(opts.items())]
    SLURM_OPTS_FILE.write_text("\n".join(lines) + "\n" if lines else "")


# ---------------------------------------------------------------------------
# Status / liveness
# ---------------------------------------------------------------------------

def read_status() -> tuple[str | None, dict]:
    """Read the server-status file written by `jlab-mcp start`."""
    if not config.STATUS_FILE.exists():
        return None, {}
    try:
        info = {}
        for line in config.STATUS_FILE.read_text().strip().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                info[k] = v
        return info.get("STATE"), info
    except OSError:
        return None, {}


def _cli_lock_held() -> bool:
    """True if a `jlab-mcp start`/`stop` holds the lifecycle flock right now."""
    try:
        fd = os.open(_LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o600)
    except OSError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def start_in_flight() -> bool:
    """True if a start attempt (bootstrap or `jlab-mcp start`) is running."""
    global _start_proc
    if _start_proc is not None:
        if _start_proc.poll() is None:
            return True
        _start_proc = None  # reap the zombie
    return _cli_lock_held()


def tail_log(lines: int = 40) -> str:
    """Last lines of the start log (bootstrap + `jlab-mcp start` output)."""
    try:
        text = START_LOG.read_text(errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


# ---------------------------------------------------------------------------
# Spawning
# ---------------------------------------------------------------------------

def spawn_start(mode: str, slurm_opts: dict | None = None) -> None:
    """Spawn the bootstrap + `jlab-mcp start` pipeline, fully detached.

    Detached (own session, no stdio inheritance) because:
    * the MCP server's stdout is the MCP stdio transport — a child
      writing there would corrupt the protocol,
    * in local mode this process stays alive as JupyterLab's parent and
      must outlive the MCP server / Claude session.

    slurm_opts ({time, partition, gres, cpus, mem_mb}) are forwarded to
    the child via JLAB_MCP_SLURM_* env vars, overriding the MCP server's
    own environment (.mcp.json defaults) for that submission.
    """
    global _start_proc, _last_spawn
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid run mode: {mode!r}")

    env = os.environ.copy()
    env["JLAB_MCP_RUN_MODE"] = mode
    env["JLAB_MCP_PROJECT_DIR"] = str(config.PROJECT_DIR)
    env["JLAB_MCP_PYTHON"] = sys.executable
    for key, value in validate_slurm_options(slurm_opts or {}).items():
        env[_SLURM_OPT_SPECS[key][1]] = value

    # 0600: `jlab-mcp start --debug` and jupyter itself can print tokens
    fd = os.open(START_LOG, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
    try:
        log_fh = os.fdopen(fd, "ab")
    except Exception:
        os.close(fd)
        raise
    with log_fh:
        _start_proc = subprocess.Popen(
            ["bash", "-c", _BOOTSTRAP_SCRIPT],
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=log_fh,
            env=env,
            cwd=str(config.PROJECT_DIR),
            start_new_session=True,
        )
    _last_spawn = time.time()
    logger.info("Spawned jlab-mcp start (mode=%s, pid=%d)", mode, _start_proc.pid)


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

def poll_until_ready(timeout: float = 600) -> dict:
    """Poll the status file until the server is ready, errored, or timeout.

    Blocking — run it via asyncio.to_thread/_run_with_progress. While a
    start attempt is in flight, 'error'/'terminated' states are treated
    as stale leftovers (the running `jlab-mcp start` clears them). If the
    CLI gave up its queue wait (QUEUE_TIMEOUT) while the SLURM job is
    still queued, the wait is resumed automatically — `jlab-mcp start` on
    a pending job resumes, never resubmits.
    """
    deadline = time.time() + timeout
    state: str | None = None
    info: dict = {}
    while True:
        state, info = read_status()
        if state == "ready":
            hostname = info.get("HOSTNAME", "")
            port = info.get("PORT", "")
            token = info.get("TOKEN", "")
            if hostname and port and token:
                client = JupyterLabClient(hostname, int(port), token)
                if client.health_check():
                    result = {
                        "status": "ready",
                        "mode": info.get("MODE", "slurm"),
                        "hostname": hostname,
                        "port": int(port),
                        "url": f"http://{hostname}:{port}",
                        "message": (
                            "JupyterLab is ready — notify the user, then "
                            "proceed (e.g. start_new_notebook)."
                        ),
                    }
                    if info.get("JOB_ID"):
                        result["job_id"] = info["JOB_ID"]
                    if info.get("PID"):
                        result["pid"] = info["PID"]
                    return result
        elif state in ("error", "terminated") and not start_in_flight():
            return {
                "status": "error",
                "message": info.get("MESSAGE", f"server state: {state}"),
                "log_tail": tail_log(),
                "hint": (
                    "Report this to the user. Calling start_server again "
                    "retries from a clean state."
                ),
            }
        elif (
            state == "pending"
            and not start_in_flight()
            and time.time() - _last_spawn > 30
        ):
            logger.info("Queue wait lapsed with job still pending — resuming")
            spawn_start("slurm", read_slurm_options())

        if time.time() >= deadline:
            break
        time.sleep(2)

    result = {
        "status": "timeout",
        "state": state or "no_status",
        "log_tail": tail_log(),
    }
    if state == "pending":
        result["message"] = (
            f"SLURM job {info.get('JOB_ID', '?')} is still waiting in the "
            "queue (this can take a while on busy partitions). Tell the "
            "user, then call wait_for_server again to keep waiting — the "
            "job stays queued either way."
        )
    else:
        result["message"] = (
            f"Server not ready after {int(timeout)}s (state="
            f"{state or 'no status file'}). Check log_tail; call "
            "wait_for_server to keep waiting or start_server to retry."
        )
    return result
