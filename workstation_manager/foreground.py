"""Foreground work: `ws run` and pytest, which their caller's own process runs.

The caller queues a job of kind run or pytest, waits until the broker grants it N
physical cores and M of memory, pins itself to them (children inherit the pinning),
runs, and reports how it ended; the broker also ends one whose process has gone.
`ws run` and the pytest hook (pytest_hook/ws_pytest.py) are the two callers.

Work already inside a job takes no second one: it is accounted for.
"""

from __future__ import annotations

import getpass
import os
import socket
import sys
import time
from typing import Optional, Sequence, TextIO

from . import cpu
from .config import Config
from .store import Job, Store, describe

STALE_SECONDS = 60.0


def held() -> bool:
    """Whether this process already runs inside a job (batch or foreground)."""
    return bool(os.environ.get("WS_JOB_ID"))


def poke(cfg: Config) -> None:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.sendto(b"w", str(cfg.wake_path))
    except OSError:
        pass  # the broker's next tick sees it anyway


def broker_up(cfg: Config, store: Store) -> bool:
    d = store.get_kv("daemon")
    return d is not None and time.time() - d["at"] < STALE_SECONDS


def limits(cfg: Config):
    """(cores, MiB) the largest request can ever get."""
    cores = len(cpu.topology()) - cfg.reserve_cores
    total, _ = cpu.meminfo()
    return max(1, cores), total - cfg.mem_margin_mib


def acquire(cfg: Config, kind: str, name: str, cores: int, mem_mib: int,
            scope: Optional[str] = None, wait_limit: Optional[float] = None,
            out: TextIO = sys.stderr, priority: int = 0,
            expect_seconds: Optional[int] = None, argv: Sequence[str] = ()) -> Optional[Job]:
    """Queue foreground work and wait until it is granted. None when the broker is not
    running or the wait ran past `wait_limit`: the caller then runs unpinned."""
    if not cfg.db_path.exists():
        return None
    store = Store(cfg.db_path)
    if not broker_up(cfg, store):
        print("ws: broker not running; running outside the queue", file=out)
        return None
    max_cores, max_mem = limits(cfg)
    cores, mem_mib = min(cores, max_cores), min(mem_mib, max_mem)
    owner = os.environ.get("WS_OWNER") or getpass.getuser()
    pid = os.getpid()
    job_id = store.request(kind, name, owner, pid, cpu.process_start(pid), cores, mem_mib,
                           os.getcwd(), scope, argv=argv, priority=priority,
                           expect_seconds=expect_seconds)
    poke(cfg)
    start = time.time()
    said = False
    told = None                     # the last reason printed, so a change is printed once
    try:
        while True:
            job = store.get(job_id)
            if job.status == "running":
                if said:
                    print(f"ws: #{job_id} got {describe(cores, mem_mib, 0)} after "
                          f"{time.time() - start:.0f}s", file=out)
                return job
            waited = time.time() - start
            if not said and waited > 1.0:
                print(f"ws: #{job_id} waiting for {describe(cores, mem_mib, 0)} ({ahead(store, job)}); "
                      "`ws status` shows who holds them", file=out)
                said = True
            if said and kind == "run":
                why = ((store.get_kv("daemon") or {}).get("waiting_on") or {}).get(job.key)
                if why and why != told:
                    print(f"ws: held up by: {why}", file=out)
                    told = why
            if wait_limit is not None and waited > wait_limit:
                store.finish(job_id, None)
                print(f"ws: waited {waited:.0f}s; running outside the queue", file=out)
                return None
            if not broker_up(cfg, store):
                store.finish(job_id, None)
                print("ws: broker stopped; running outside the queue", file=out)
                return None
            time.sleep(0.2)
    except BaseException:
        store.finish(job_id, None)
        poke(cfg)
        raise


def ahead(store: Store, job: Job) -> str:
    running = store.jobs(("running",))
    bits = [f"{len(running)} job(s) running"]
    if job.fast:
        before = [j for j in store.jobs(("queued",), ("pytest",)) if j.id < job.id]
        if before:
            bits.append(f"{len(before)} pytest run(s) ahead")
    else:
        # A `ws run` waits in the queue it shares with batch jobs.
        from .scheduler import queue_order
        q = queue_order([j for j in store.jobs(("queued",)) if not j.fast])
        n = next((i for i, h in enumerate(q) if h.key == job.key), 0)
        if n:
            bits.append(f"{n} ahead of it in the queue")
    return ", ".join(bits)


def enter(job: Job) -> None:
    """Pin this process to the job's cores and tell its children about it."""
    cpu.pin_process(job.cpus or ())
    os.environ.update(cpu.thread_env(os.environ, job.cores, job.cpus or ()))
    os.environ["WS_JOB_ID"] = str(job.id)


def release(cfg: Config, job: Optional[Job], exit_code: Optional[int]) -> None:
    """The work has ended with `exit_code` (None: unknown)."""
    if job is None:
        return
    Store(cfg.db_path).finish(job.id, exit_code)
    poke(cfg)
