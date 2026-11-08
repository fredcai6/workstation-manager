"""One scheduling decision: what to start and stop, given the card's current state.

Pure: no I/O, so every rule is testable. The daemon gathers a View, applies the
returned actions, and asks again next tick.

Rules, in order:
1. `ws take` stops everything and starts nothing.
2. With preemption on, a waiting interactive request stops batch, newest first,
   until the interactive pair fits, then starts it.
3. Queued jobs start first-come first-served when they fit in both the measured
   free VRAM and the declared budget. A later job may start ahead of the head only
   if the head would still fit once the interactive pair is gone.
4. If the head is still waiting, the pair is stopped for it once no interactive
   request has arrived for `hold_minutes`.
5. With nothing waiting, the pair comes back up when it fits, with the wanted model.
6. When the wanted model differs from the loaded one, the llm is recreated with it
   once it fits; meanwhile no batch job starts that would keep it from fitting.
7. In a reserve window the pair stays up with the wanted (default) model: it is
   brought up even with jobs queued, never stopped for batch, and a job starts
   only if it fits beside it. Running jobs are left alone.
8. An on-demand service starts when a request waits for it and it fits, ahead of
   queued batch (a caller is waiting). It stops once idle for its limit, or at once
   when idle and its room is needed by a model swap or the head of the queue.
9. Cores and memory are resources like VRAM. Every job and lease declares physical
   cores and peak memory and starts only when both fit the declared budget (at least
   `reserve` cores are always left to unmanaged work) and what is measured: memory
   that running holders have declared but not yet used counts against
   MemAvailable, and for batch jobs cores busy outside managed work count against
   the room too. A lease has a caller waiting and skips that last check, and
   `lease_reserve` cores are kept for leases: batch never takes them. A job gets whole physical cores, least busy first
   in CCD-alternating order. The rules above about the card apply to jobs that
   declare VRAM; a CPU-only job neither waits for the pair nor sends it away.
10. A lease (a foreground pytest or `ws run`: a caller is waiting) is granted ahead
   of queued batch, first come first served, even during a take: it holds no VRAM.
11. Priority: the queue is in priority order (highest first, then first come). When
   the first job waiting for the card cannot fit, running jobs of lower priority are
   stopped and requeued, newest first, until it would, if stopping them is enough.
   Jobs that are already stopping count as gone. Not during a reserve window.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .store import Job, Lease

CROWDED_GRACE_SECONDS = 10.0

# ("start_job", id[, cpus]) | ("grant_lease", id, cpus) | ("stop_job", id, reason) |
# ("stop_pair",) | ("start_pair", model) | ("swap_model", model) | ("start_service", name) |
# ("stop_service", name). start_job carries cpus when the View has a CpuView.
Action = Tuple


@dataclasses.dataclass
class CpuView:
    """Physical cores in allocation order (each a tuple of logical CPUs), how busy
    each is (0-1, smoothed), and how many are always left to unmanaged work."""
    cores: Sequence[Tuple[int, ...]]
    busy: Sequence[float]
    reserve: int = 2
    lease_reserve: int = 0          # cores batch never takes, so a pytest need not wait for it


@dataclasses.dataclass
class MemView:
    total_mib: int
    available_mib: int
    margin_mib: int
    unrealized_mib: int = 0         # declared by running holders, not yet in use


@dataclasses.dataclass
class Svc:
    """An on-demand service as the scheduler sees it."""
    name: str
    state: str                      # up | starting | stopping | down
    vram_mib: int
    waiting: int = 0                # requests waiting for it
    in_flight: int = 0
    idle_seconds: float = 0.0       # since it last served a request
    idle_limit: float = 120.0


@dataclasses.dataclass
class View:
    now: float
    queued: Sequence[Job]           # in queue order
    active: Sequence[Job]           # running or stopping
    pair: str                       # up | starting | stopping | down
    last_interactive_at: float
    vram_total_mib: Optional[int]
    vram_used_mib: Optional[int]
    take: bool = False
    demand_pending: bool = False    # an interactive request is waiting at the proxy
    loaded: Optional[str] = None    # model the llm container runs (None: unknown)
    want: Optional[str] = None      # model it should run (None: the only/first one)
    services: Sequence[Svc] = ()
    reserved: bool = False          # inside a reserve window
    cpu: Optional[CpuView] = None   # None: cores are not scheduled (tests of card rules)
    mem: Optional[MemView] = None   # None: memory is not scheduled
    leases: Sequence[Lease] = ()    # waiting and granted, in request order


def cores_of(h) -> int:
    """Cores a holder takes. A legacy job declared none: like any unmanaged process,
    its load shows up as unmanaged instead."""
    return 0 if getattr(h, "legacy", False) else h.cores


class Host:
    """Room on the cores and in memory, updated as the decision hands it out."""

    def __init__(self, v: View):
        self.cpu, self.mem = v.cpu, v.mem
        granted = [l for l in v.leases if l.status == "granted"]
        holders = list(v.active) + granted
        self.cores_committed = sum(cores_of(h) for h in holders)
        self.lease_cores = sum(l.cores for l in granted)
        self.mem_committed = sum(h.mem_mib for h in holders)
        if self.cpu:
            where = {c: i for i, core in enumerate(self.cpu.cores) for c in core}
            self.assigned = {where[c] for h in holders for c in (h.cpus or ()) if c in where}
            unmanaged = sum(b for i, b in enumerate(self.cpu.busy) if i not in self.assigned)
            self.cores_cap = len(self.cpu.cores) - self.cpu.reserve
            # What batch may take: the reserve or the unmanaged load, whichever is more.
            self.unmanaged_over = max(0, round(unmanaged) - self.cpu.reserve)
        if self.mem:
            self.mem_cap = self.mem.total_mib - self.mem.margin_mib
            self.mem_room = min(self.mem_cap - self.mem_committed,
                                self.mem.available_mib - self.mem.margin_mib - self.mem.unrealized_mib)

    def fits(self, h, lease: bool = False) -> bool:
        if self.cpu and cores_of(h):
            room = self.cores_cap - self.cores_committed
            if not lease:
                room -= self.unmanaged_over + max(0, self.cpu.lease_reserve - self.lease_cores)
            if h.cores > room:
                return False
        return not self.mem or h.mem_mib <= self.mem_room

    def leaves_room_for(self, h, head) -> bool:
        """Whether starting `h` still lets `head` fit once running work is gone."""
        if self.cpu and cores_of(h) and \
                self.cores_cap - (self.cores_committed + cores_of(h)) < cores_of(head):
            return False
        if self.mem and h.mem_mib and self.mem_cap - (self.mem_committed + h.mem_mib) < head.mem_mib:
            return False
        return True

    def take(self, h) -> Optional[List[int]]:
        """Commit `h`; the logical CPUs it gets (None when cores are not scheduled)."""
        self.cores_committed += cores_of(h)
        if isinstance(h, Lease):
            self.lease_cores += h.cores
        self.mem_committed += h.mem_mib
        if self.mem:
            self.mem_room -= h.mem_mib
        if not self.cpu:
            return None
        if not cores_of(h):
            return []
        free = [i for i in range(len(self.cpu.cores)) if i not in self.assigned]
        free.sort(key=lambda i: (self.cpu.busy[i] >= 0.5, i))
        got = free[:h.cores]
        self.assigned.update(got)
        return sorted(c for i in got for c in self.cpu.cores[i])


def decide(v: View, margin_mib: int, vram: Union[int, Dict[str, int]], hold_minutes: float,
           preempt_on_interactive: bool) -> List[Action]:
    """`vram` is the pair's footprint per model (or one number for a single model)."""
    if isinstance(vram, int):
        vram = {"default": vram}
    want = v.want or next(iter(vram))
    want_mib = vram[want]
    held_mib = vram.get(v.loaded, want_mib) if v.loaded else want_mib

    actions: List[Action] = []
    running = [j for j in v.active if j.status == "running"]
    host = Host(v)

    pending = [l for l in v.leases if l.status == "waiting"]
    for lease in list(pending):
        if not host.fits(lease, lease=True):
            continue
        if lease is not pending[0] and not host.leaves_room_for(lease, pending[0]):
            continue
        actions.append(("grant_lease", lease.id, host.take(lease) or []))
        pending.remove(lease)

    if v.take:
        actions += [("stop_job", j.id, "requeue") for j in running]
        if v.pair in ("up", "starting"):
            actions.append(("stop_pair",))
        return actions

    if v.vram_total_mib is None or v.vram_used_mib is None:
        return actions  # fail closed: start nothing on an unreadable card

    cap = v.vram_total_mib - margin_mib
    held = [sv for sv in v.services if sv.state != "down"]
    committed = sum(j.vram_mib for j in v.active) + sum(sv.vram_mib for sv in held)
    pair_holds = v.pair in ("up", "starting", "stopping")
    free = v.vram_total_mib - v.vram_used_mib - margin_mib

    if preempt_on_interactive and v.demand_pending and v.pair == "down":
        for j in sorted(running, key=lambda j: j.started_at or 0, reverse=True):
            if cap - committed >= want_mib:
                break
            actions.append(("stop_job", j.id, "requeue"))
            committed -= j.vram_mib
        if not actions and free >= want_mib:
            actions.append(("start_pair", want))
        return actions

    # In a reserve window the pair's room is spoken for even while it is down.
    pair_mib = held_mib if pair_holds else (want_mib if v.reserved else 0)
    # Switching models: recreate the llm with the wanted one once it fits. Until it
    # fits, no batch job may start that would keep it from ever fitting.
    swap_blocked = False
    if v.pair == "up" and v.loaded is not None and v.loaded != want:
        if cap - committed >= want_mib and free + held_mib >= want_mib:
            actions.append(("swap_model", want))
            free += held_mib - want_mib
            pair_mib = want_mib
        else:
            swap_blocked = True

    room = cap - committed - pair_mib
    head = v.queued[0] if v.queued else None
    # On-demand services: start for a waiting caller; stop when idle and either past
    # the idle limit or in the way of a swap or of the head of the queue.
    for sv in v.services:
        if sv.state == "down" and sv.waiting and sv.vram_mib <= min(room, free):
            actions.append(("start_service", sv.name))
            committed += sv.vram_mib
            room -= sv.vram_mib
            free -= sv.vram_mib
    for sv in held:
        if sv.state != "up" or sv.waiting or sv.in_flight:
            continue
        crowding = swap_blocked or (head is not None and head.vram_mib > min(room, free))
        # A short grace even when crowded: back-to-back requests leave gaps of
        # milliseconds, and a caller mid-batch should not lose the server between them.
        if sv.idle_seconds >= sv.idle_limit or (crowding and sv.idle_seconds >= CROWDED_GRACE_SECONDS):
            actions.append(("stop_service", sv.name))
    started = {a[1] for a in actions if a[0] == "start_service"}
    # A service still waiting for room: no batch job may start that keeps it out.
    svc_need = max((sv.vram_mib for sv in v.services
                    if sv.state == "down" and sv.waiting and sv.name not in started), default=0)

    waiting = list(v.queued)
    for job in v.queued:
        if job.vram_mib > min(room, free) or not host.fits(job):
            continue
        # Whoever is first still waiting is protected, also once the original head starts.
        first = waiting[0]
        if job is not first:
            if job.vram_mib and cap - (committed + job.vram_mib) < first.vram_mib:
                continue
            if not host.leaves_room_for(job, first):
                continue
        if job.vram_mib and swap_blocked and cap - (committed + job.vram_mib) < want_mib:
            continue
        if job.vram_mib and svc_need and cap - (committed + job.vram_mib) - pair_mib < svc_need:
            continue
        cpus = host.take(job)
        actions.append(("start_job", job.id) if cpus is None else ("start_job", job.id, cpus))
        waiting.remove(job)
        committed += job.vram_mib
        room -= job.vram_mib
        free -= job.vram_mib

    # Only jobs that need the card keep the pair away or send it off.
    waiting = [j for j in waiting if j.vram_mib > 0]
    if waiting and not v.reserved:
        actions += _make_room(v, waiting[0], cap - committed - pair_mib)
    if v.reserved:
        if v.pair == "down" and cap - committed >= want_mib and free >= want_mib:
            actions.append(("start_pair", want))
    elif waiting:
        first = waiting[0]
        idle_min = (v.now - v.last_interactive_at) / 60.0
        if v.pair == "up" and not swap_blocked and first.vram_mib <= cap - committed \
                and idle_min >= hold_minutes and ("swap_model", want) not in actions:
            actions.append(("stop_pair",))
    elif v.pair == "down" and cap - committed >= want_mib and free >= want_mib:
        actions.append(("start_pair", want))
    return actions


def _make_room(v: View, first: Job, room_mib: int) -> List[Action]:
    """Stop lower-priority running jobs, newest first, so `first` fits on the card."""
    room_mib += sum(j.vram_mib for j in v.active if j.status == "stopping")
    if first.vram_mib <= room_mib:
        return []
    below = [j for j in v.active if j.status == "running" and j.vram_mib
             and j.priority < first.priority]
    if first.vram_mib > room_mib + sum(j.vram_mib for j in below):
        return []  # stopping them would not be enough
    actions: List[Action] = []
    for j in sorted(below, key=lambda j: j.started_at or 0, reverse=True):
        if first.vram_mib <= room_mib:
            break
        actions.append(("stop_job", j.id, "requeue"))
        room_mib += j.vram_mib
    return actions
