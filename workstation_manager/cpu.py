"""The host's CPU cores and memory, as the broker sees them.

A core here is a physical core: both SMT siblings go to the same holder, so two
managed jobs never share one. Cores are listed alternating between L3 groups
(CCDs), which is the order the 2026-10-15 throughput sweep found best for
memory-bound fits (scratch/v35/perf/RESULTS_20261015.md).
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SYS_CPU = Path("/sys/devices/system/cpu")


@dataclasses.dataclass(frozen=True)
class Core:
    index: int                 # position in allocation order
    cpus: Tuple[int, ...]      # logical CPUs (SMT siblings)
    l3: int


def parse_cpulist(text: str) -> List[int]:
    out: List[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def format_cpulist(cpus: Iterable[int]) -> str:
    return ",".join(str(c) for c in sorted(cpus))


def topology(root: Path = SYS_CPU) -> List[Core]:
    """Physical cores, alternating between L3 groups."""
    groups: Dict[int, List[Tuple[int, ...]]] = {}
    seen = set()
    for c in parse_cpulist((root / "online").read_text()):
        d = root / f"cpu{c}"
        sib = tuple(parse_cpulist((d / "topology/thread_siblings_list").read_text()))
        if sib in seen:
            continue
        seen.add(sib)
        try:
            l3 = min(parse_cpulist((d / "cache/index3/shared_cpu_list").read_text()))
        except OSError:
            l3 = 0
        groups.setdefault(l3, []).append(sib)
    lists = [sorted(groups[k]) for k in sorted(groups)]
    order: List[Tuple[int, int]] = []
    for i in range(max((len(l) for l in lists), default=0)):
        for g, l in enumerate(lists):
            if i < len(l):
                order.append((g, i))
    return [Core(n, lists[g][i], g) for n, (g, i) in enumerate(order)]


class Meter:
    """Busy fraction of each physical core, smoothed over about `tau` seconds. A core
    counts as busy as its busiest sibling."""

    def __init__(self, cores: Sequence[Core], tau: float = 30.0, stat: Path = Path("/proc/stat")):
        self.cores = list(cores)
        self.tau = tau
        self.stat = stat
        self.busy = [0.0] * len(self.cores)
        self.prev: Optional[Dict[int, Tuple[int, int]]] = None
        self.prev_at = 0.0
        self.primed = False

    def _read(self) -> Dict[int, Tuple[int, int]]:
        out = {}
        for line in self.stat.read_text().splitlines():
            if not line.startswith("cpu") or line.startswith("cpu "):
                continue
            f = line.split()
            v = [int(x) for x in f[1:]]
            idle = v[3] + (v[4] if len(v) > 4 else 0)
            out[int(f[0][3:])] = (sum(v[:8]), idle)
        return out

    def sample(self, now: Optional[float] = None) -> List[float]:
        now = time.time() if now is None else now
        if self.prev is not None and now - self.prev_at < 1.0:
            return self.busy  # too soon for a meaningful delta
        cur = self._read()
        if self.prev is not None:
            # The first delta is taken whole: starting from zero would read a busy
            # machine as idle for the first `tau` seconds after a restart.
            w = 1.0 if not self.primed else min(1.0, (now - self.prev_at) / self.tau)
            self.primed = True
            for i, core in enumerate(self.cores):
                fr = 0.0
                for c in core.cpus:
                    if c in cur and c in self.prev:
                        dt = cur[c][0] - self.prev[c][0]
                        di = cur[c][1] - self.prev[c][1]
                        if dt > 0:
                            fr = max(fr, 1.0 - di / dt)
                self.busy[i] += w * (fr - self.busy[i])
        self.prev, self.prev_at = cur, now
        return self.busy


def meminfo(path: Path = Path("/proc/meminfo")) -> Tuple[int, int]:
    """(MemTotal, MemAvailable) in MiB."""
    vals = {}
    for line in path.read_text().splitlines():
        k, _, rest = line.partition(":")
        if k in ("MemTotal", "MemAvailable"):
            vals[k] = int(rest.split()[0]) // 1024
    return vals["MemTotal"], vals["MemAvailable"]


def unit_cgroup(unit: str) -> Optional[Path]:
    r = subprocess.run(["systemctl", "--user", "show", "-p", "ControlGroup", "--value", unit],
                       capture_output=True, text=True)
    cg = r.stdout.strip()
    return Path("/sys/fs/cgroup" + cg) if cg else None


def memory_current_mib(cgroup: Optional[Path]) -> Optional[int]:
    try:
        return int((cgroup / "memory.current").read_text()) // (1024 * 1024)
    except (OSError, TypeError, ValueError):
        return None


def rss_mib(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def own_oom_kills() -> int:
    """oom_kill count of the cgroup this process runs in (0 if unreadable)."""
    try:
        cg = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
        for line in Path("/sys/fs/cgroup" + cg + "/memory.events").read_text().splitlines():
            k, v = line.split()
            if k == "oom_kill":
                return int(v)
    except (OSError, IndexError, ValueError):
        pass
    return 0


def pin_process(cpus: Iterable[int], pid: int = 0) -> None:
    """Set the affinity of every thread of `pid` (0: this process); children inherit it."""
    cpus = set(cpus)
    base = Path(f"/proc/{pid or os.getpid()}/task")
    try:
        tids = [int(t) for t in os.listdir(base)]
    except OSError:
        tids = [pid]
    for tid in tids:
        try:
            os.sched_setaffinity(tid, cpus)
        except OSError:
            pass  # thread exited


def process_start(pid: int) -> Optional[int]:
    """Start time of `pid` in clock ticks since boot, to tell a pid apart from its reuse."""
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return None


def alive(pid: int, start: Optional[int]) -> bool:
    now = process_start(pid)
    return now is not None and (start is None or now == start)


def thread_env(env: dict, cores: int, cpus: Sequence[int]) -> dict:
    """What a pinned process is told: its CPUs and core count, and thread-pool sizes
    matching its cores unless the caller set them."""
    env = dict(env)
    env["WS_JOB_CPUS"] = format_cpulist(cpus)
    env["WS_JOB_CORES"] = str(cores)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(k, str(cores))
    return env


def container_ids(job_label: str) -> List[str]:
    """Full ids of the running containers carrying a job's label."""
    r = subprocess.run(["docker", "ps", "-q", "--no-trunc", "--filter", f"label={job_label}"],
                       capture_output=True, text=True)
    return r.stdout.split()


def container_memory_mib(job_label: str) -> int:
    """Memory in use by a job's containers. They run under Docker, outside the job's unit."""
    total = 0
    for cid in container_ids(job_label):
        total += memory_current_mib(Path(f"/sys/fs/cgroup/system.slice/docker-{cid}.scope")) or 0
    return total


def kernel_oom_kills(since: float, ids: Iterable[str]) -> bool:
    """Whether the kernel log records an OOM kill in one of these containers since `since`."""
    ids = [i for i in ids if i]
    if not ids:
        return False
    r = subprocess.run(["journalctl", "-k", "--no-pager", "-o", "cat", "--since", f"@{int(since)}"],
                       capture_output=True, text=True)
    return any("oom" in l.lower() and any(f"docker-{i}" in l for i in ids) for l in r.stdout.splitlines())
