"""Run batch jobs as transient systemd user units, so they outlive broker restarts.

The unit's main process is `ws _wrap <id>`. It restores the submitter's
environment, pins itself to the job's cores (the command inherits them), runs the
command in its own process group, tees output to the job log, reads `chunk N/M` and
`NN%` progress from it, and records the exit code. The unit's MemoryMax is the
job's declared memory, with no swap; a kill at that limit is recorded as such.
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

from . import cpu
from .config import REPO, Config
from .store import Job, Store

CHUNK_RE = re.compile(r"\bchunk\s+(\d+)\s*(?:/|of)\s*(\d+)", re.I)
PCT_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")
# An exit after one of these is out of GPU memory: rerunning the same command at the
# same declaration fails the same way.
OOM_RE = re.compile(r"RESOURCE_EXHAUSTED|out of memory|hipErrorOutOfMemory|HIP out of memory", re.I)


def label(job: Job) -> str:
    return f"{'gpu-manager' if job.legacy else 'workstation-manager'}.job={job.id}"


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
        # A kill at MemoryMax takes the command, not the unit: the wrapper records it.
        "-p", "OOMPolicy=continue",
        # The wrapper reads the same config (and so the same state) as this broker.
        *([f"--setenv=WORKSTATION_MANAGER_CONFIG={os.environ['WORKSTATION_MANAGER_CONFIG']}"]
          if os.environ.get("WORKSTATION_MANAGER_CONFIG") else []),
        *(["-p", f"MemoryMax={job.mem_mib}M", "-p", "MemorySwapMax=0"]
          if job.mem_mib else []),
        sys.executable, str(REPO / "ws"), "_wrap", str(job.id),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def stop(job: Job) -> None:
    subprocess.run(["systemctl", "--user", "stop", "--no-block", job.unit],
                   capture_output=True)


def unit_result(job: Job) -> str:
    """What the journal says ended the unit, e.g. "status=200/CHDIR". The unit itself
    is collected once it ends, so systemd no longer knows."""
    r = subprocess.run(["journalctl", "--user", "-u", job.unit, "-n", "20", "--no-pager", "-o", "cat"],
                       capture_output=True, text=True)
    for line in reversed(r.stdout.splitlines()):
        m = re.search(r"code=\w+, status=(\S+)", line)
        if m:
            return f"status={m.group(1)}"
    return "no unit status in the journal"


def is_active(job: Job) -> bool:
    r = subprocess.run(["systemctl", "--user", "is-active", job.unit],
                       capture_output=True, text=True)
    return r.stdout.strip() in ("active", "activating", "deactivating", "reloading")


def _labelled(job_label: str, all_states: bool) -> list:
    r = subprocess.run(["docker", "ps", "-q", *(["-a"] if all_states else []),
                        "--filter", f"label={job_label}"], capture_output=True, text=True)
    return r.stdout.split()


def stop_labelled_containers(job_label: str, wait_seconds: float = 60) -> None:
    """Stop the job's containers and wait until they are gone, so a rerun of the same
    command cannot collide with a `--rm` container that is still being removed."""
    ids = _labelled(job_label, all_states=False)
    if ids:
        subprocess.run(["docker", "stop", "-t", "20", *ids], capture_output=True)
    deadline = time.time() + wait_seconds
    while (ids := _labelled(job_label, all_states=True)) and time.time() < deadline:
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
    env.update(WS_JOB_ID=str(job_id), WS_JOB_LABEL=label(job),
               WS_JOB_VRAM_MIB=str(job.vram_mib), WS_JOB_MEM_MIB=str(job.mem_mib),
               PYTHONUNBUFFERED="1")
    if job.cpus:
        cpu.pin_process(job.cpus)
        env = cpu.thread_env(env, job.cores, job.cpus)
    oom_kills_before = cpu.own_oom_kills()
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

    # The job's containers run under Docker, outside this unit: note their ids while they
    # run, so a kill at their --memory cap can be told from an ordinary failure.
    started = time.time()
    containers: set = set()
    done = threading.Event()

    # Their CPU time and memory, last seen while they ran: a job that runs in a container
    # uses almost nothing in its own unit.
    c_cpu: dict = {}
    c_peak: dict = {}
    c_together = [0]  # most memory the containers held at once (MiB)

    def watch_containers():
        while not done.wait(5):
            ids = cpu.container_ids(label(job))
            containers.update(ids)
            now_mib = 0
            for cid in ids:
                cg = cpu.container_cgroup(cid)
                used, peak = cpu.cgroup_usage(cg)
                if used is not None:
                    c_cpu[cid] = used
                if peak is not None:
                    c_peak[cid] = max(peak, c_peak.get(cid, 0))
                now_mib += cpu.memory_current_mib(cg) or 0
            c_together[0] = max(c_together[0], now_mib)
    threading.Thread(target=watch_containers, daemon=True).start()

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
    done.set()
    unit_cpu, unit_peak = cpu.cgroup_usage(cpu.own_cgroup())
    # Containers' peak: what they were seen holding together, or the largest one's own
    # peak if that is more (summing peaks would overstate containers run one after another).
    c_mib = max(c_together[0], max(c_peak.values(), default=0)) if c_peak else None
    usage = {
        "wall_seconds": round(time.time() - started, 1),
        "cpu_seconds": round((unit_cpu or 0) + sum(c_cpu.values()), 1) if unit_cpu is not None or c_cpu else None,
        "peak_mib": (unit_peak or 0) + (c_mib or 0) if unit_peak is not None or c_mib else None,
        "container_peak_mib": c_mib,
    }
    stop_labelled_containers(label(job))
    mem_oom = code != 0 and (cpu.own_oom_kills() > oom_kills_before
                             or cpu.kernel_oom_kills(started, containers))
    if mem_oom:
        log.write(f"== workstation-manager: killed at its memory limit ({job.mem_mib} MiB)\n".encode())
    log.write(f"== workstation-manager: exit {code} {time.strftime('%F %T')}\n".encode())
    (d / "result.json").write_text(json.dumps(
        {"exit_code": code, "at": time.time(), "oom": oom and code != 0, "mem_oom": mem_oom,
         "usage": usage}))
    return 0


def take_result(cfg: Config, job_id: int):
    p = cfg.job_dir(job_id) / "result.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    p.rename(p.with_name(f"result-{int(data['at'])}.json"))
    return data
