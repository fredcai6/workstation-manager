"""Broker state as one dict (for `ws status --json` and the status page) and as text."""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, List, Optional

from .config import Config
from .store import ACTIVE, Store, describe


def log_activity(cfg: Config, job_id: int) -> Dict[str, Any]:
    """When the job last wrote output, and what. A fit can compile for most of an
    hour before its first chunk line; this is what shows it is alive meanwhile."""
    p = cfg.job_dir(job_id) / "log"
    try:
        with open(p, "rb") as f:
            f.seek(max(0, p.stat().st_size - 4096))
            tail = f.read().decode(errors="replace").splitlines()
        last = next((l.strip() for l in reversed(tail) if l.strip()), None)
        return {"log_at": p.stat().st_mtime, "log_last": last[:200] if last else None}
    except OSError:
        return {"log_at": None, "log_last": None}


def snapshot(cfg: Config, store: Store, now: float = None) -> Dict[str, Any]:
    now = time.time() if now is None else now
    d = store.get_kv("daemon")
    last = store.get_kv("last_interactive_at")
    hold_left = None
    if last is not None:
        hold_left = max(0.0, cfg.hold_minutes * 60 - (now - last))
    return {
        "at": now,
        "broker": None if d is None else {**d, "age_seconds": now - d["at"]},
        "take": bool(store.get_kv("take", False)),
        "hold_minutes": cfg.hold_minutes,
        "last_interactive_at": last,
        "hold_seconds_left": hold_left,
        "running": [{**j.to_dict(), **log_activity(cfg, j.id)} for j in store.jobs(ACTIVE)],
        "queued": [j.to_dict() for j in store.jobs(("queued",))],
        "leases": [{**dataclasses.asdict(l), "fast": l.fast} for l in store.leases()],
        "queue": queue_view(store),
        "recent": [j.to_dict() for j in store.recent(10)],
        "events": store.events(now - 86400),
    }


def queue_view(store: Store) -> List[Dict[str, Any]]:
    """The queue as the scheduler orders it: jobs and waiting `ws run` commands."""
    from .scheduler import queue_order
    items = queue_order(store.jobs(("queued",)) + [l for l in store.leases(("waiting",)) if not l.fast])
    out = []
    for h in items:
        if hasattr(h, "submitted_at"):
            out.append({**h.to_dict(), "kind": "job", "key": h.key, "label": f"#{h.id} {h.name}"})
        else:
            out.append({**dataclasses.asdict(h), "kind": "run", "key": h.key, "vram_mib": 0,
                        "submitted_at": h.requested_at, "interruptions": 0, "failures": 0,
                        "legacy": False, "label": f"ws run #{h.id} {h.name}"})
    return out


def runs_held(s: Dict[str, Any]) -> List[Dict[str, Any]]:
    """`ws run` commands that hold their cores: running work, like jobs."""
    return [l for l in s.get("leases", ()) if l["status"] == "granted" and not l.get("fast")]


def estimate(s: Dict[str, Any], key: str, started: Optional[float], expect: Optional[int]) -> str:
    """e.g. "expected 2.0h, ends ~14:10" or "over its 1.0h estimate by 40m"."""
    now = s["at"]
    proj = (((s.get("broker") or {}).get("projection")) or {}).get(key)
    if not expect:
        return "no estimate"
    if started and now > started + expect:
        return f"over its {_ago(expect)} estimate by {_ago(now - started - expect)}"
    end = proj[1] if proj and proj[1] else (started + expect if started else None)
    return f"expected {_ago(expect)}" + (f", ends ~{time.strftime('%H:%M', time.localtime(end))}" if end else "")


def core_holders(s: Dict[str, Any]) -> List[Optional[str]]:
    """For each physical core, who holds it: "#N name", "lease #N name", or None."""
    cpu = ((s.get("broker") or {}).get("cpu")) or {}
    cores = cpu.get("cores") or []
    where = {c: i for i, core in enumerate(cores) for c in core}
    out: List[Optional[str]] = [None] * len(cores)
    for j in s["running"]:
        for c in j.get("cpus") or ():
            if c in where:
                out[where[c]] = f"#{j['id']} {j['name']}"
    for l in s.get("leases", ()):
        if l["status"] == "granted":
            for c in l.get("cpus") or ():
                if c in where:
                    out[where[c]] = f"lease #{l['id']} {l['name']}"
    return out


def host_summary(s: Dict[str, Any]) -> Dict[str, Any]:
    """Cores held and the unmanaged load on the rest; memory declared and available."""
    b = s.get("broker") or {}
    cpu, mem = b.get("cpu") or {}, b.get("mem") or {}
    holders = core_holders(s)
    busy = cpu.get("busy") or []
    granted = [l for l in s.get("leases", ()) if l["status"] == "granted"]
    declared = sum(j.get("mem_mib") or 0 for j in s["running"]) + sum(l["mem_mib"] for l in granted)
    room = None
    if mem.get("total_mib") and mem.get("available_mib") is not None:
        margin = mem.get("margin_mib") or 0
        # As the scheduler sees it: the declared budget and what is measured, less
        # memory holders have declared but not yet used.
        room = max(0, min(mem["total_mib"] - margin - declared,
                          mem["available_mib"] - margin - (mem.get("unrealized_mib") or 0)))
    return {
        "mem_room_mib": room,
        "cores": len(holders),
        "held": sum(1 for h in holders if h),
        "unmanaged_busy": sum(b for b, h in zip(busy, holders) if not h),
        "reserve": cpu.get("reserve"),
        "mem_total_mib": mem.get("total_mib"),
        "mem_available_mib": mem.get("available_mib"),
        "mem_declared_mib": sum(j.get("mem_mib") or 0 for j in s["running"])
                            + sum(l["mem_mib"] for l in granted),
    }


def _ago(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"{seconds // 60}m"
    return f"{seconds / 3600:.1f}h"


def _progress(j: Dict[str, Any]) -> str:
    bits = []
    if j.get("chunks"):
        bits.append(f"chunk {j['chunk']} of {j['chunks']}")
    if j.get("pct") is not None:
        bits.append(f"{j['pct']:.0f}%")
    return ", ".join(bits) or "no progress reported"


def render_text(s: Dict[str, Any]) -> str:
    now = s["at"]
    out = ["", "Broker:"]
    b = s["broker"]
    if b is None:
        return "\n".join(out + ["  not running (no state yet)"])
    stale = b["age_seconds"] > 60
    why = b.get("waiting_on") or {}
    out.append(f"  {'STALE' if stale else 'running'} (last tick {_ago(b['age_seconds'])} ago)")
    if s["take"]:
        out.append("  TAKEN by operator: nothing starts until `ws release`")
    pair = b["pair"]
    line = f"  Interactive pair: {pair}"
    if b.get("model"):
        line += f" ({b['model']}" + (f", switching to {b['want']}" if b.get("want") not in (None, b["model"]) else "") + ")"
    if s["last_interactive_at"]:
        line += f"; last interactive request {_ago(now - s['last_interactive_at'])} ago"
    if pair == "up" and s["queued"]:
        left = s["hold_seconds_left"] or 0
        line += "; batch may take the card now" if left <= 0 else f"; batch may take the card in {_ago(left)}"
    out.append(line)
    if b.get("waiting"):
        out.append(f"  {b['waiting']} request(s) waiting for the pair"
                   f" ({b.get('waiting_interactive', 0)} interactive)")
    if b.get("reserved"):
        out.append("  Reserve window: the pair stays up; big jobs wait")
    if b.get("services"):
        out.append("  On-demand: " + ", ".join(f"{n} {st}" for n, st in b["services"].items()))
    h = host_summary(s)
    if h["cores"]:
        out.append(f"  Cores: {h['held']} of {h['cores']} held; ~{h['unmanaged_busy']:.1f} busy with "
                   f"unmanaged work; {h['reserve']} always left to it")
    if h["mem_total_mib"]:
        out.append(f"  Memory: {h['mem_available_mib'] / 1024:.1f} GB available of "
                   f"{h['mem_total_mib'] / 1024:.1f}; {h['mem_declared_mib'] / 1024:.1f} GB declared by holders"
                   + (f"; room for new work {h['mem_room_mib'] / 1024:.1f} GB" if h.get("mem_room_mib") is not None else ""))
    fast = [l for l in s.get("leases", ()) if l.get("fast")]
    if fast:
        out.append("  pytest (the fast lane):")
    for l in fast:
        when = l["granted_at"] if l["status"] == "granted" else l["requested_at"]
        out.append(f"    lease #{l['id']} {l['name']}  {describe(l['cores'], l['mem_mib'], 0)}  "
                   f"{'held' if l['status'] == 'granted' else 'waiting'} {_ago(now - when)}"
                   + (f"  [{why[f'lease:{l['id']}']}]" if l["status"] != "granted"
                      and f"lease:{l['id']}" in why else ""))
    runs = runs_held(s)
    out.append("  Running:" if s["running"] or runs else "  Running: none")
    for j in s["running"]:
        out.append(f"    #{j['id']} {j['name']}  {describe(j.get('cores') or 1, j.get('mem_mib') or 0, j['vram_mib'], j.get('legacy'))}  {j['status']}"
                   f"  {_progress(j)}  started {_ago(now - (j['started_at'] or now))} ago"
                   + (f", priority {j['priority']}" if j.get("priority") else "")
                   + f"; {estimate(s, f'job:{j['id']}', j['started_at'], j.get('expect_seconds'))}"
                   + ("; went ahead on its estimate" if j.get("backfill") else ""))
        if j.get("log_at"):
            out.append(f"      last output {_ago(now - j['log_at'])} ago: {j['log_last'] or ''}")
    for l in runs:
        out.append(f"    ws run #{l['id']} {l['name']}  {describe(l['cores'], l['mem_mib'], 0)}  "
                   f"running {_ago(now - l['granted_at'])}"
                   + (f"  cpus {','.join(map(str, l['cpus']))}" if l.get("cpus") else "")
                   + (f", priority {l['priority']}" if l.get("priority") else "")
                   + f"; {estimate(s, f'lease:{l['id']}', l['granted_at'], l.get('expect_seconds'))}")
    q = s.get("queue", ())
    out.append("  Queue (jobs and ws run, in the order they start):" if q else "  Queue: empty")
    proj = b.get("projection") or {}
    for n, j in enumerate(q, 1):
        extra = f", {j['interruptions']} interruption(s)" if j["interruptions"] else ""
        p = proj.get(j["key"])
        when = f", starts ~{time.strftime('%H:%M', time.localtime(p[0]))}" if p and p[0] and p[0] > now + 60 else ""
        exp = f", expected {_ago(j['expect_seconds'])}" if j.get("expect_seconds") else ", no estimate"
        out.append(f"    {n}. [{j.get('priority') or 0:+d}] {j['label']}  "
                   f"{describe(j.get('cores') or 1, j.get('mem_mib') or 0, j['vram_mib'], j.get('legacy'))}  waiting "
                   f"{_ago(now - j['submitted_at'])}{exp}{when}{extra}")
        if j["key"] in why:
            out.append(f"       held up by: {why[j['key']]}")
    return "\n".join(out)
