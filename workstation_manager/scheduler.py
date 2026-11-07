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
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .store import Job

CROWDED_GRACE_SECONDS = 10.0

Action = Tuple  # ("start_job", id) | ("stop_job", id, reason) | ("stop_pair",) | ("start_pair", model) | ("swap_model", model) | ("start_service", name) | ("stop_service", name)


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
        if job.vram_mib > min(room, free):
            continue
        if job is not head and head in waiting and cap - (committed + job.vram_mib) < head.vram_mib:
            continue
        if swap_blocked and cap - (committed + job.vram_mib) < want_mib:
            continue
        if svc_need and cap - (committed + job.vram_mib) - pair_mib < svc_need:
            continue
        actions.append(("start_job", job.id))
        waiting.remove(job)
        committed += job.vram_mib
        room -= job.vram_mib
        free -= job.vram_mib

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
