"""Leases: foreground work that holds cores and memory while it runs.

A process asks the broker for N physical cores and M of memory, waits until they
are granted, pins itself to them (children inherit the pinning) and releases them
when it ends; the broker also ends a lease whose process has gone. `ws run` and the
pytest hook (pytest_hook/ws_pytest.py) are the two callers.

Work already inside a job or a lease takes no second lease: it is accounted for.
"""

from __future__ import annotations

import getpass
import os
import socket
import sys
import time
from typing import Optional, TextIO

from . import cpu
from .config import Config
from .store import Lease, Store, describe

STALE_SECONDS = 60.0


def held() -> bool:
    """Whether this process already runs inside a job or a lease."""
    return bool(os.environ.get("WS_JOB_ID") or os.environ.get("WS_LEASE"))


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
            unit: Optional[str] = None, wait_limit: Optional[float] = None,
            out: TextIO = sys.stderr) -> Optional[Lease]:
    """Wait for a lease. None when the broker is not running or the wait ran past
    `wait_limit`: the caller then runs unpinned."""
    if not cfg.db_path.exists():
        return None
    store = Store(cfg.db_path)
    if not broker_up(cfg, store):
        print("ws: broker not running; running without a core lease", file=out)
        return None
    max_cores, max_mem = limits(cfg)
    cores, mem_mib = min(cores, max_cores), min(mem_mib, max_mem)
    owner = os.environ.get("WS_OWNER") or getpass.getuser()
    pid = os.getpid()
    lease_id = store.request_lease(kind, name, owner, pid, cpu.process_start(pid), cores,
                                   mem_mib, os.getcwd(), unit)
    poke(cfg)
    start = time.time()
    said = False
    try:
        while True:
            lease = store.lease(lease_id)
            if lease.status == "granted":
                if said:
                    print(f"ws: got {describe(cores, mem_mib, 0)} after "
                          f"{time.time() - start:.0f}s", file=out)
                return lease
            waited = time.time() - start
            if not said and waited > 1.0:
                print(f"ws: waiting for {describe(cores, mem_mib, 0)} ({ahead(store, lease)}); "
                      "`ws status` shows who holds them", file=out)
                said = True
            if wait_limit is not None and waited > wait_limit:
                store.end_lease(lease_id)
                print(f"ws: waited {waited:.0f}s; running without a core lease", file=out)
                return None
            if not broker_up(cfg, store):
                store.end_lease(lease_id)
                print("ws: broker stopped; running without a core lease", file=out)
                return None
            time.sleep(0.2)
    except BaseException:
        store.end_lease(lease_id)
        poke(cfg)
        raise


def ahead(store: Store, lease: Lease) -> str:
    before = [l for l in store.leases(("waiting",)) if l.id < lease.id]
    held_by = store.leases(("granted",))
    bits = [f"{len(held_by)} lease(s) and {len(store.jobs(('running',)))} job(s) running"]
    if before:
        bits.append(f"{len(before)} lease(s) ahead")
    return ", ".join(bits)


def enter(lease: Lease) -> None:
    """Pin this process to the lease and tell its children about it."""
    cpu.pin_process(lease.cpus or ())
    os.environ.update(cpu.thread_env(os.environ, lease.cores, lease.cpus or ()))
    os.environ["WS_LEASE"] = str(lease.id)


def release(cfg: Config, lease: Optional[Lease]) -> None:
    if lease is None:
        return
    Store(cfg.db_path).end_lease(lease.id)
    poke(cfg)
