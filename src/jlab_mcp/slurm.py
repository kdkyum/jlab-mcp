import os
import random
import secrets
import subprocess
import tempfile
import time
from pathlib import Path

from jlab_mcp import config


def _random_port() -> int:
    return random.randint(*config.PORT_RANGE)


def _generate_token() -> str:
    return secrets.token_hex(24)


def _build_module_commands(modules_str: str) -> str:
    """Build shell commands for loading modules."""
    modules_str = modules_str.strip()
    if not modules_str:
        return "# No modules configured"
    lines = ["module purge"]
    for mod in modules_str.split():
        lines.append(f"module load {mod}")
    return "\n".join(lines)


def render_slurm_script(
    port: int,
    token: str,
    connection_file: str,
    notebook_dir: str | None = None,
    log_dir: str | None = None,
    project_dir: str | None = None,
) -> str:
    template_path = config.TEMPLATE_DIR / "jupyter_slurm.sh.template"
    template = template_path.read_text()
    return template.format(
        port=port,
        token=token,
        connection_file=connection_file,
        notebook_dir=notebook_dir or str(config.NOTEBOOK_DIR),
        log_dir=log_dir or str(config.LOG_DIR),
        project_dir=project_dir or str(config.PROJECT_DIR),
        partition=config.SLURM_PARTITION,
        gres=config.SLURM_GRES,
        cpus=config.SLURM_CPUS,
        mem=config.SLURM_MEM,
        time=config.SLURM_TIME,
        module_commands=_build_module_commands(config.SLURM_MODULES),
    )


def parse_sbatch_output(stdout: str) -> str:
    """Parse job ID from sbatch stdout like 'Submitted batch job 12345'."""
    parts = stdout.strip().split()
    if len(parts) >= 4 and parts[0] == "Submitted":
        return parts[-1]
    raise ValueError(f"Cannot parse sbatch output: {stdout!r}")


def parse_squeue_output(stdout: str) -> tuple[str, str]:
    """Parse state and nodelist from squeue output like 'RUNNING ravg1001'."""
    parts = stdout.strip().split()
    if len(parts) >= 2:
        return parts[0], parts[1]
    if len(parts) == 1:
        return parts[0], ""
    return "", ""


def parse_connection_file(path: str | Path) -> dict[str, str]:
    """Parse connection file with KEY=VALUE lines."""
    result = {}
    text = Path(path).read_text()
    for line in text.strip().splitlines():
        line = line.strip()
        if "=" in line:
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    return result


def submit_job(
    notebook_dir: str | None = None,
) -> tuple[str, str, int, str]:
    """Submit a SLURM job for JupyterLab.

    Returns (job_id, connection_file_path, port, token).
    """
    port = _random_port()
    token = _generate_token()
    connection_file = str(config.CONNECTION_DIR / f"jupyter-{port}.conn")

    script_content = render_slurm_script(
        port=port,
        token=token,
        connection_file=connection_file,
        notebook_dir=notebook_dir,
    )

    # Write script to temp file and submit
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, dir=str(config.LOG_DIR)
    ) as f:
        f.write(script_content)
        script_path = f.name

    try:
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True,
            check=True,
        )
        job_id = parse_sbatch_output(result.stdout)
    finally:
        os.unlink(script_path)

    return job_id, connection_file, port, token


def wait_for_job_running(job_id: str, timeout: int = 120) -> str:
    """Wait until SLURM job is RUNNING. Returns the hostname."""
    start = time.time()
    while time.time() - start < timeout:
        result = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T %N"],
            capture_output=True,
            text=True,
        )
        stdout = result.stdout.strip()
        if not stdout:
            # Job no longer in queue — may have failed
            raise RuntimeError(
                f"Job {job_id} disappeared from queue (may have failed)"
            )
        state, node = parse_squeue_output(stdout)
        if state == "RUNNING" and node:
            return node
        time.sleep(3)
    raise TimeoutError(f"Job {job_id} did not start within {timeout}s")


def wait_for_connection_file(
    path: str | Path, timeout: int = 60
) -> dict[str, str]:
    """Wait for connection file to appear and have content."""
    start = time.time()
    while time.time() - start < timeout:
        p = Path(path)
        if p.exists() and p.stat().st_size > 0:
            info = parse_connection_file(p)
            if "HOSTNAME" in info and "PORT" in info and "TOKEN" in info:
                return info
        time.sleep(2)
    raise TimeoutError(f"Connection file {path} not ready within {timeout}s")


def cancel_job(job_id: str) -> None:
    """Cancel a SLURM job."""
    subprocess.run(["scancel", job_id], capture_output=True, text=True)


def get_job_state(job_id: str) -> str:
    """Get current state of a SLURM job."""
    result = subprocess.run(
        ["squeue", "-j", job_id, "-h", "-o", "%T"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def is_job_running(job_id: str) -> bool:
    """Quick check if job is still running."""
    return get_job_state(job_id) == "RUNNING"


def cleanup_connection_file(path: str | Path) -> None:
    """Remove a connection file (contains token)."""
    p = Path(path)
    if p.exists():
        p.unlink()
