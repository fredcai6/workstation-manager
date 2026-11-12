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
   `reserve` cores are always left to unmanaged work) and what is measured:
   MemAvailable less the margin (memory running holders declared but have not yet
   used is not counted again), and for batch jobs cores busy outside managed work
   count against the room too. A lease has a caller waiting and skips that last check, and
   `lease_reserve` cores (default none) are kept for pytest: batch never takes them. A job gets whole physical cores, least busy first
   in CCD-alternating order. The rules above about the card apply to jobs that
   declare VRAM; a CPU-only job neither waits for the pair nor sends it away.
10. pytest runs are the fast lane: granted ahead of the queue, first come first served,
   even during a take; they count declarations only and may use the `lease_reserve`
   cores and the `reserve` ones (sharing them with the desktop for a few minutes). A `ws run` waits in the queue with jobs: same order, same room checks, and
   it is granted where a job would start (it runs in its caller's terminal).
11. Priority: the queue is in priority order (highest first, then first come). When
   the first in line cannot fit, on the card, in memory or on the cores, running jobs
   of lower priority are stopped and requeued, newest first, until it would, if
   stopping them is enough, and no more than it needs.
   Jobs that are already stopping count as gone. Not during a reserve window.
12. Estimates: every job and `ws run` says how long it should run (--expect); the daemon
   passes the learned upper (P80) run time instead (estimates.py: the guess corrected by
   the record, by where and what the run is and how loaded the box is). A later
   job that does not leave room for the first in line may still start if its estimate
   has it done before the first in line could start (project(): holders end at their
   estimates). If it runs past its
   estimate while the first in line waits, it is stopped and requeued, newest first,
   when that makes room. A `ws run` goes ahead this way only if it is small (rule 13):
   it cannot be stopped, but it ends soon.
   Nothing else is ever stopped for running over.
13. Small lane: `small_lane` cores are kept for small work (at most that many cores,
   planned to end within `small_seconds`). Small work fills them first, then the rest;
   other work never takes them. Small work that fits in the lane may start ahead of a
   bigger first in line, since it takes nothing that one could use.
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
    small_lane: int = 0             # cores only small work may take (it may use the rest too)
    small_seconds: float = 0.0      # small: planned to end within this, on <= small_lane cores


@dataclasses.dataclass
class MemView:
    total_mib: int
    available_mib: int
    margin_mib: int
    unrealized_mib: int = 0         # declared by running holders, not yet in use (shown, not counted)


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


def is_small(h, cpu: Optional[CpuView]) -> bool:
    """Small work moves along on the small lane: few cores, planned to end soon."""
    return bool(cpu and cpu.small_lane and not getattr(h, "fast", False)
                and 0 < cores_of(h) <= cpu.small_lane
                and getattr(h, "expect_seconds", None) and h.expect_seconds <= cpu.small_seconds)


def _gb(mib: float) -> str:
    return f"{mib / 1024:.1f} GB"


class Host:
    """Room on the cores and in memory, updated as the decision hands it out."""

    def __init__(self, v: View):
        self.cpu, self.mem = v.cpu, v.mem
        granted = [l for l in v.leases if l.status == "granted"]
        holders = list(v.active) + granted
        self.cores_committed = sum(cores_of(h) for h in holders)
        # Cores held by pytest runs, the fast lane `lease_reserve` is kept for.
        self.fast_cores = sum(l.cores for l in granted if l.fast)
        # Cores held by small work: it fills the small lane first, then the rest.
        self.small_cores = sum(cores_of(h) for h in holders if is_small(h, v.cpu))
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
            # Declarations must fit the budget, and the new one must fit what is free now.
            # Memory running holders declared but have not used is not held back as well:
            # each is capped at its declaration, and most peak near half of it.
            self.mem_room = min(self.mem_cap - self.mem_committed,
                                self.mem.available_mib - self.mem.margin_mib)

    def fits(self, h) -> bool:
        return self.shortfall(h) is None

    def core_room(self, h) -> int:
        """Cores `h` may take now. A pytest lease counts declarations only and may use the
        `reserve` cores too: it is short and rare, so sharing them with the desktop for a
        few minutes costs little. Everything in the queue backs off from unmanaged load
        and leaves the `reserve` and any `lease_reserve` cores alone."""
        if getattr(h, "fast", False):
            return len(self.cpu.cores) - self.cores_committed
        room = (self.cores_cap - self.cores_committed - self.unmanaged_over
                - max(0, self.cpu.lease_reserve - self.fast_cores))
        if not is_small(h, self.cpu):       # the small lane's free cores are not for it
            room -= self.lane_free()
        return room

    def lane_free(self) -> int:
        return max(0, self.cpu.small_lane - self.small_cores) if self.cpu else 0

    def shortfall(self, h) -> Optional[str]:
        """Why `h` does not fit now (None when it does), in words for the status page.
        A pytest lease (the fast lane) counts declarations only and may use the cores
        kept for it; everything in the queue also backs off from unmanaged load."""
        fast = getattr(h, "fast", False)
        if self.cpu and cores_of(h):
            room = self.core_room(h)
            if h.cores > room:
                extra = f", {self.unmanaged_over} busy with unmanaged work" \
                    if not fast and self.unmanaged_over else ""
                return f"cores: needs {h.cores}, {max(0, room)} free{extra}"
        if self.mem and h.mem_mib > self.mem_room:
            return f"memory: needs {_gb(h.mem_mib)}, {_gb(max(0, self.mem_room))} free"
        return None

    def leaves_room_for(self, h, head) -> bool:
        """Whether starting `h` still lets `head` fit once running work is gone. Small
        work that fits in the small lane takes nothing a bigger `head` could use."""
        lane_only = self.cpu and is_small(h, self.cpu) and not is_small(head, self.cpu) \
            and cores_of(h) <= self.lane_free()
        if self.cpu and cores_of(h) and not lane_only and \
                self.cores_cap - (self.cores_committed + cores_of(h)) < cores_of(head):
            return False
        if self.mem and h.mem_mib and self.mem_cap - (self.mem_committed + h.mem_mib) < head.mem_mib:
            return False
        return True

    def take(self, h) -> Optional[List[int]]:
        """Commit `h`; the logical CPUs it gets (None when cores are not scheduled)."""
        self.cores_committed += cores_of(h)
        if is_small(h, self.cpu):
            self.small_cores += cores_of(h)
        if getattr(h, "fast", False):
            self.fast_cores += h.cores
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
           preempt_on_interactive: bool, why: Optional[Dict[str, str]] = None) -> List[Action]:
    """`vram` is the pair's footprint per model (or one number for a single model).
    `why`, when given, is filled with what keeps each waiting job and lease from
    starting ("job:N" / "lease:N" -> reason). It changes no decision."""
    why = {} if why is None else why
    if isinstance(vram, int):
        vram = {"default": vram}
    want = v.want or next(iter(vram))
    want_mib = vram[want]
    held_mib = vram.get(v.loaded, want_mib) if v.loaded else want_mib

    actions: List[Action] = []
    running = [j for j in v.active if j.status == "running"]
    host = Host(v)

    # The fast lane: pytest runs, first come first served, ahead of the queue.
    pending = [l for l in v.leases if l.status == "waiting" and l.fast]
    for lease in list(pending):
        short = host.shortfall(lease)
        if short:
            why[f"lease:{lease.id}"] = short
            continue
        if lease is not pending[0] and not host.leaves_room_for(lease, pending[0]):
            why[f"lease:{lease.id}"] = f"held back for lease #{pending[0].id} (first in line)"
            continue
        actions.append(("grant_lease", lease.id, host.take(lease) or []))
        pending.remove(lease)

    # The queue: jobs and `ws run` commands together, highest priority first, then first come.
    queue = queue_order(list(v.queued) + [l for l in v.leases if l.status == "waiting" and not l.fast])

    if v.take:
        why.update({h.key: "operator take: nothing starts until `ws release`" for h in queue})
        actions += [("stop_job", j.id, "requeue") for j in running]
        if v.pair in ("up", "starting"):
            actions.append(("stop_pair",))
        return actions

    if v.vram_total_mib is None or v.vram_used_mib is None:
        why.update({h.key: "the card is unreadable; nothing starts" for h in queue})
        return actions  # fail closed: start nothing on an unreadable card

    cap = v.vram_total_mib - margin_mib
    held = [sv for sv in v.services if sv.state != "down"]
    committed = sum(j.vram_mib for j in v.active) + sum(sv.vram_mib for sv in held)
    pair_holds = v.pair in ("up", "starting", "stopping")
    free = v.vram_total_mib - v.vram_used_mib - margin_mib

    if preempt_on_interactive and v.demand_pending and v.pair == "down":
        why.update({h.key: "the interactive pair is coming back first" for h in queue})
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
    head = queue[0] if queue else None
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

    waiting = list(queue)
    started_now: List = []
    shadows: Dict[str, Optional[float]] = {}
    for job in queue:
        key = job.key
        if job.vram_mib > min(room, free):
            whose = " (interactive pair up)" if pair_holds else (" (reserve window)" if v.reserved else "")
            why[key] = f"VRAM: needs {_gb(job.vram_mib)}, {_gb(max(0, min(room, free)))} free{whose}"
            continue
        short = host.shortfall(job)
        if short:
            why[key] = short
            continue
        # Whoever is first still waiting is protected, also once the original head starts.
        first = waiting[0]
        backfill = False
        if job is not first:
            if (job.vram_mib and cap - (committed + job.vram_mib) < first.vram_mib) \
                    or not host.leaves_room_for(job, first):
                # It may still go ahead if its estimate has it done before the first in
                # line could start anyway. A job, which can be stopped if it runs over, or
                # small work (a `ws run` too): if it runs over, the first waits at most a little.
                if first.key not in shadows:
                    shadows[first.key] = project(v, margin_mib, [first], started_now)[first.key][0]
                shadow = shadows[first.key]
                if not ((isinstance(job, Job) or is_small(job, v.cpu)) and job.expect_seconds
                        and shadow is not None
                        and v.now + job.expect_seconds <= shadow):
                    why[key] = f"held back for {label(first)} (first in line)"
                    continue
                backfill = True
        if job.vram_mib and swap_blocked and cap - (committed + job.vram_mib) < want_mib:
            why[key] = "waiting for the LLM model switch"
            continue
        if job.vram_mib and svc_need and cap - (committed + job.vram_mib) - pair_mib < svc_need:
            why[key] = "waiting for an on-demand service to start"
            continue
        cpus = host.take(job)
        if isinstance(job, Lease):
            actions.append(("grant_lease", job.id, cpus or []))
        elif backfill:
            actions.append(("start_job", job.id, cpus, "backfill"))
        else:
            actions.append(("start_job", job.id) if cpus is None else ("start_job", job.id, cpus))
        waiting.remove(job)
        started_now.append(job)
        committed += job.vram_mib
        room -= job.vram_mib
        free -= job.vram_mib

    # A job that went ahead on its estimate and has run past it makes way for the first in line.
    if waiting and not v.take:
        actions += _overrun_stops(v, waiting[0], host, cap - committed - pair_mib, why)

    # The first in line stops running jobs of lower priority until it fits.
    if waiting and not v.reserved:
        gone = {a[1] for a in actions if a[0] == "stop_job"}
        made = _make_room(v, waiting[0], host, cap - committed - pair_mib, gone)
        if made:
            why[waiting[0].key] = "stopping lower-priority " + \
                ", ".join(f"#{a[1]}" for a in made) + " to make room"
        actions += made

    # Only jobs that need the card keep the pair away or send it off.
    waiting = [j for j in waiting if j.vram_mib > 0]
    if v.reserved:
        if v.pair == "down" and cap - committed >= want_mib and free >= want_mib:
            actions.append(("start_pair", want))
    elif waiting:
        first = waiting[0]
        idle_min = (v.now - v.last_interactive_at) / 60.0
        if v.pair == "up" and not swap_blocked and first.vram_mib <= cap - committed \
                and idle_min >= hold_minutes and ("swap_model", want) not in actions:
            actions.append(("stop_pair",))
        elif v.pair == "up" and idle_min < hold_minutes and f"job:{first.id}" in why:
            why[f"job:{first.id}"] += f"; the pair is held for {hold_minutes - idle_min:.0f} more min"
    elif v.pair == "down" and cap - committed >= want_mib and free >= want_mib:
        actions.append(("start_pair", want))
    return actions


def queue_order(items):
    """Highest priority first, then first come."""
    return sorted(items, key=lambda h: (-h.priority, h.queued_at, h.key))


def label(h) -> str:
    return f"#{h.id}" if isinstance(h, Job) else f"ws run #{h.id}"


OVERDUE_GRACE_SECONDS = 600     # a holder past its estimate is assumed to end this much later


def expected_end(h, now: float) -> Tuple[Optional[float], bool]:
    """(When a running job or lease should end, whether it is past its estimate); None
    when it gave none. From the estimate alone: chunk progress is no clock, since a job
    that resumes from saved work jumps straight to a late chunk."""
    start = getattr(h, "started_at", None) or getattr(h, "granted_at", None)
    if start is None or not h.expect_seconds:
        return None, False
    est = start + h.expect_seconds
    if est < now:
        return now + OVERDUE_GRACE_SECONDS, True
    return est, False


def project(v: View, margin_mib: int, items=None, extra_running=()) -> Dict[str, Tuple]:
    """When each waiting item should start and end ((start, end); None where unknown),
    and each running holder end ((None, end)): holders end at their expected end, and
    the queue starts in order as the declared budget frees up. Measured load, the
    interactive pair and pytest are left out: it is an estimate, not a promise."""
    need = lambda h: (cores_of(h) if v.cpu else 0, h.mem_mib if v.mem else 0, h.vram_mib)
    holders = list(v.active) + [l for l in v.leases if l.status == "granted" and not l.fast]
    # The room as measured now, plus what holders free as they end: unmanaged work (the
    # desktop, agents, services) is taken to stay as it is.
    host = Host(v)
    held_cores = sum(cores_of(h) for h in holders)
    held_mem = sum(h.mem_mib for h in holders)
    budget = [
        (host.cores_cap - host.unmanaged_over - v.cpu.lease_reserve) if v.cpu else float("inf"),
        (min(host.mem_cap, host.mem_room + host.mem_committed) - (host.mem_committed - held_mem))
        if v.mem else float("inf"),
        (v.vram_total_mib - margin_mib) if v.vram_total_mib else float("inf"),
    ]
    del held_cores
    out: Dict[str, Tuple] = {}
    lane = v.cpu.small_lane if v.cpu else 0
    sm = lambda h: cores_of(h) if v.cpu and is_small(h, v.cpu) else 0
    INF = float("inf")
    # Everything placed on the timeline: (start, end, need, small-lane cores). A holder
    # with no estimate holds its room for good.
    placed = []
    for h in holders:
        end, over = expected_end(h, v.now)
        out[h.key] = (None, end, over)
        placed.append((v.now, end if end is not None else INF, need(h), sm(h)))
    for h in extra_running:         # started this tick
        placed.append((v.now, v.now + h.expect_seconds if h.expect_seconds else INF, need(h), sm(h)))

    def fits_at(t: float, dur: float, n, small_h: int) -> bool:
        """Whether `n` fits from t for dur beside what is placed (checked wherever use rises)."""
        for tau in [t] + [a for a, _, _, _ in placed if t < a < t + dur]:
            use = [0.0, 0.0, 0.0]
            small = 0
            for a, b, nn, ss in placed:
                if a <= tau < b:
                    use = [u + x for u, x in zip(use, nn)]
                    small += ss
            room = list(budget)
            if not small_h:             # other work leaves the small lane's free cores alone
                room[0] -= max(0, lane - small)
            if any(x > r - u for x, r, u in zip(n, room, use)):
                return False
        return True

    floor = v.now                   # bigger work starts in queue order; small work need not
    queue = queue_order(list(v.queued) + [l for l in v.leases if l.status == "waiting" and not l.fast]) \
        if items is None else list(items)
    for h in queue:
        n, small_h = need(h), sm(h)
        dur = h.expect_seconds or INF
        start = None
        for t in sorted({floor if not small_h else v.now} |
                        {b for _, b, _, _ in placed if b != INF and b > (v.now if small_h else floor)}):
            if fits_at(t, dur, n, small_h):
                start = t
                break
        if start is None:           # waits on something with no estimate
            out[h.key] = (None, None, False)
            continue
        end = start + h.expect_seconds if h.expect_seconds else None
        out[h.key] = (start, end, False)
        placed.append((start, end if end is not None else INF, n, small_h))
        if not small_h:
            floor = start
    return out


def _overrun_stops(v: View, first, host: "Host", vram_room: int, why: Dict[str, str]) -> List[Action]:
    """Stop jobs that started ahead of `first` on their estimate and have run past it,
    newest first, if that makes room for `first` in the declared budget."""
    over = [j for j in v.active if j.status == "running" and j.backfill and j.expect_seconds
            and j.started_at and v.now > j.started_at + j.expect_seconds]
    if not over:
        return []
    cores = (host.cores_cap - host.cores_committed) if host.cpu else 10 ** 6
    mem = (host.mem_cap - host.mem_committed) if host.mem else 10 ** 9
    vram = vram_room + sum(j.vram_mib for j in v.active if j.status == "stopping")
    fits = lambda: cores_of(first) <= cores and first.mem_mib <= mem and first.vram_mib <= vram
    if fits():
        return []
    stops = []
    for j in sorted(over, key=lambda j: j.started_at, reverse=True):
        stops.append(("stop_job", j.id, "requeue"))
        cores, mem, vram = cores + cores_of(j), mem + j.mem_mib, vram + j.vram_mib
        if fits():
            why[first.key] = "stopping " + ", ".join(f"#{a[1]}" for a in stops) + \
                " (went ahead on its estimate and ran over) to make room"
            return stops
    return []


def _make_room(v: View, first, host: "Host", vram_room: int, gone=frozenset()) -> List[Action]:
    """Stop lower-priority running jobs, newest first, so `first` fits on the card, in
    memory and on the cores. Jobs already stopping (or in `gone`, stopped this tick)
    count as freed. Stops no more than it needs: one that `first` would fit without is dropped."""
    # Room `first` sees on each resource: VRAM, memory, cores.
    room = [vram_room,
            host.mem_room if host.mem else 10 ** 9,
            host.core_room(first) if host.cpu else 10 ** 6]
    need = [first.vram_mib, first.mem_mib if host.mem else 0, cores_of(first) if host.cpu else 0]
    held = lambda j: [j.vram_mib, j.mem_mib if host.mem else 0, cores_of(j) if host.cpu else 0]
    for j in v.active:
        if j.status == "stopping" or j.id in gone:
            room = [r + h for r, h in zip(room, held(j))]
    short = lambda: [n > r for n, r in zip(need, room)]
    if not any(short()):
        return []
    below = [j for j in v.active if j.status == "running" and j.id not in gone
             and j.priority < first.priority]
    total = [r + sum(held(j)[k] for j in below) for k, r in enumerate(room)]
    if any(n > t for n, t in zip(need, total)):
        return []  # stopping them would not be enough
    stops = []
    for j in sorted(below, key=lambda j: j.started_at or 0, reverse=True):
        if not any(short()):
            break
        stops.append(j)
        room = [r + h for r, h in zip(room, held(j))]
    # Keep only the stops it needs: a newer small job passed on the way is spared.
    for j in list(stops):
        less = [r - h for r, h in zip(room, held(j))]
        if all(n <= r for n, r in zip(need, less)):
            stops.remove(j)
            room = less
    return [("stop_job", j.id, "requeue") for j in stops]
