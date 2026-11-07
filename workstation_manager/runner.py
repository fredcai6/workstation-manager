"""Run batch jobs as transient systemd user units, so they outlive broker restarts.

The unit's main process is `ws _wrap <id>`. It restores the submitter's
environment, runs the command in its own process group, tees output to the job
log, reads `chunk N/M` and `NN%` progress from it, and records the exit code.
On SIGTERM it passes the signal to the job, waits the job's grace period, then
kills the group and any container labelled `workstation-manager.job=<id>`.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from .config import REPO, Config
from .store import Job, Store

CHUNK_RE = re.compile(r"\bchunk\s+(\d+)\s*(?:/|of)\s*(\d+)", re.I)
PCT_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")
# An exit after one of these is out of GPU memory: rerunning the same command at the
# same declaration fails the same way.
OOM_RE = re.compile(r"RESOURCE_EXHAUSTED|out of memory|hipErrorOutOfMemory|HIP out of memory", re.I)


def label(job_id: int) -> str:
    return f"workstation-manager.job={job_id}"


def save_env(cfg: Config, job_id: int, env: dict) -> None:
    d = cfg.job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "env.json"
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(env, f)


def start(cfg: Config, job: Job) -> None:
    subprocess.run(["systemctl", "--user", "reset-failed", job.unit],
                   capture_output=True)
    cmd = [
        "systemd-run", "--user", "--quiet", "--collect", f"--unit={job.unit}",
        f"--working-directory={job.cwd}",
        "-p", "KillMode=mixed",
        "-p", f"TimeoutStopSec={job.grace_seconds + 30}",
        sys.executable, str(REPO / "ws"), "_wrap", str(job.id),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def stop(job: Job) -> None:
    subprocess.run(["systemctl", "--user", "stop", "--no-block", job.unit],
                   capture_output=True)


def is_active(job: Job) -> bool:
    r = subprocess.run(["systemctl", "--user", "is-active", job.unit],
                       capture_output=True, text=True)
    return r.stdout.strip() in ("active", "activating", "deactivating", "reloading")


def _labelled(job_id: int, all_states: bool) -> list:
    r = subprocess.run(["docker", "ps", "-q", *(["-a"] if all_states else []),
                        "--filter", f"label={label(job_id)}"], capture_output=True, text=True)
    return r.stdout.split()


def stop_labelled_containers(job_id: int, wait_seconds: float = 60) -> None:
    """Stop the job's containers and wait until they are gone, so a rerun of the same
    command cannot collide with a `--rm` container that is still being removed."""
    ids = _labelled(job_id, all_states=False)
    if ids:
        subprocess.run(["docker", "stop", "-t", "20", *ids], capture_output=True)
    deadline = time.time() + wait_seconds
    while (ids := _labelled(job_id, all_states=True)) and time.time() < deadline:
        time.sleep(1)
    if ids:
        subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)


def parse_progress(line: str):
    m = CHUNK_RE.search(line)
    if not m:
        return None
    chunk, chunks = int(m.group(1)), int(m.group(2))
    rest = line[m.end():]
    p = PCT_RE.search(rest) or PCT_RE.search(line[:m.start()])
    pct = float(p.group(1)) if p and 0 <= float(p.group(1)) <= 100 else None
    return chunk, chunks, pct


def wrap(cfg: Config, job_id: int) -> int:
    store = Store(cfg.db_path)
    job = store.get(job_id)
    d = cfg.job_dir(job_id)
    env = json.loads((d / "env.json").read_text())
    env.update(WS_JOB_ID=str(job_id), WS_JOB_LABEL=label(job_id),
               WS_JOB_VRAM_MIB=str(job.vram_mib), PYTHONUNBUFFERED="1")
    log = open(d / "log", "ab", buffering=0)
    log.write(f"\n== workstation-manager: start {time.strftime('%F %T')} attempt "
              f"{job.failures + job.interruptions + 1}\n".encode())
    proc = subprocess.Popen(job.argv, cwd=job.cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True)
    stopping = threading.Event()

    def on_term(signum, frame):
        if stopping.is_set():
            return
        stopping.set()
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        def reap():
            deadline = time.time() + job.grace_seconds
            while time.time() < deadline and proc.poll() is None:
                time.sleep(0.5)
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        threading.Thread(target=reap, daemon=True).start()

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    oom = False
    for raw in iter(proc.stdout.readline, b""):
        log.write(raw)
        line = raw.decode(errors="replace")
        oom = oom or bool(OOM_RE.search(line))
        got = parse_progress(line)
        if got:
            store.progress(job_id, chunk=got[0], chunks=got[1], pct=got[2])
    code = proc.wait()
    stop_labelled_containers(job_id)
    log.write(f"== workstation-manager: exit {code} {time.strftime('%F %T')}\n".encode())
    (d / "result.json").write_text(json.dumps(
        {"exit_code": code, "at": time.time(), "oom": oom and code != 0}))
    return 0


def take_result(cfg: Config, job_id: int):
    p = cfg.job_dir(job_id) / "result.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    p.rename(p.with_name(f"result-{int(data['at'])}.json"))
    return data
