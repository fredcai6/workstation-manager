"""Broker state as one dict (for `ws status --json` and the status page) and as text."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from .config import Config
from .store import ACTIVE, Job, Store, describe

RECENT = 30                     # finished batch jobs and `ws run` commands on the page
RECENT_TEXT = 10                # ... and in `ws status`


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
        # Running work and the queue, pytest apart: it is the fast lane.
        "running": [view(j, cfg) for j in store.jobs(ACTIVE) if not j.fast],
        "queue": queue_view(store),
        "pytest": [view(j) for j in store.jobs(("queued",) + ACTIVE, ("pytest",))],
        "recent": [view(j) for j in store.recent(RECENT)],
        "pytest_day": store.tally("pytest", now - 86400),
        "events": store.events(now - 86400),
    }


def view(j: Job, cfg: Optional[Config] = None) -> Dict[str, Any]:
    """A job as the page and `ws status --json` show it; a running batch job with its log."""
    d = {**j.to_dict(), "key": j.key, "tag": j.tag, "in_terminal": j.in_terminal,
         "label": f"#{j.id} {j.name}" + (f" ({j.tag})" if j.tag else "")}
    if cfg is not None and not j.in_terminal:
        d.update(log_activity(cfg, j.id))
    return d


def queue_view(store: Store) -> List[Dict[str, Any]]:
    """The queue as the scheduler orders it: batch jobs and `ws run` commands."""
    from .scheduler import queue_order
    return [view(j) for j in queue_order([j for j in store.jobs(("queued",)) if not j.fast])]


def outcome(j: Dict[str, Any]) -> str:
    """e.g. "done", "failed (exit 1)", "ended (no exit code)"."""
    if j["status"] == "ended":
        return "ended (no exit code)"
    return j["status"] + ("" if j.get("exit_code") in (None, 0) else f" (exit {j['exit_code']})")


def pytest_day(s: Dict[str, Any]) -> str:
    """e.g. "77 pytest runs in the last day: 70 done, 5 failed, 2 ended"; "" when none."""
    t = s.get("pytest_day") or {}
    n = sum(t.values())
    if not n:
        return ""
    order = ("done", "failed", "ended", "cancelled")
    return f"{n} pytest run{'s' if n != 1 else ''} in the last day: " + ", ".join(
        f"{t[k]} {k}" for k in order if t.get(k))


def expect_of(s: Dict[str, Any], key: str, given: Optional[int]) -> Tuple[Optional[int], str]:
    """The run time the scheduler plans on for `key` (the learned upper one, P80, where
    there is a model), and a note with the typical one and the author's guess."""
    got = ((s.get("broker") or {}).get("estimates") or {}).get(key)
    if got:
        typical, plan = got
        return plan, f" (learned: typically {_ago(typical)}" + (f"; --expect {_ago(given)}" if given else "") + ")"
    return given, ""


def estimate(s: Dict[str, Any], key: str, started: Optional[float], expect: Optional[int]) -> str:
    """e.g. "expected 2.0h, ends ~14:10" or "over its 1.0h estimate by 40m"."""
    now = s["at"]
    proj = (((s.get("broker") or {}).get("projection")) or {}).get(key)
    expect, note = expect_of(s, key, expect)
    if not expect:
        return "no estimate"
    if started and now > started + expect:
        return f"over its {_ago(expect)} estimate{note} by {_ago(now - started - expect)}"
    end = proj[1] if proj and proj[1] else (started + expect if started else None)
    return f"expected {_ago(expect)}{note}" + (f", ends ~{time.strftime('%H:%M', time.localtime(end))}" if end else "")


def core_holders(s: Dict[str, Any]) -> List[Optional[str]]:
    """For each physical core, who holds it ("#N name"), or None."""
    cpu = ((s.get("broker") or {}).get("cpu")) or {}
    cores = cpu.get("cores") or []
    where = {c: i for i, core in enumerate(cores) for c in core}
    out: List[Optional[str]] = [None] * len(cores)
    for j in holders(s):
        for c in j.get("cpus") or ():
            if c in where:
                out[where[c]] = f"#{j['id']} {j['name']}"
    return out


def holders(s: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Everything holding cores and memory: running work and running pytest runs."""
    return list(s["running"]) + [j for j in s.get("pytest", ()) if j["status"] == "running"]


def host_summary(s: Dict[str, Any]) -> Dict[str, Any]:
    """Cores held and the unmanaged load on the rest; memory declared and available."""
    b = s.get("broker") or {}
    cpu, mem = b.get("cpu") or {}, b.get("mem") or {}
    held_by = core_holders(s)
    busy = cpu.get("busy") or []
    charged = b.get("mem_charged") or {}
    # Counted as the scheduler counts it: its name's largest peak, padded, where known.
    declared = sum(charged.get(j["key"], j.get("mem_mib") or 0) for j in holders(s))
    room = None
    if mem.get("total_mib") and mem.get("available_mib") is not None:
        margin = mem.get("margin_mib") or 0
        # As the scheduler sees it: the declared budget and what is measured.
        room = max(0, min(mem["total_mib"] - margin - declared, mem["available_mib"] - margin))
    return {
        "mem_room_mib": room,
        "cores": len(held_by),
        "held": sum(1 for h in held_by if h),
        "unmanaged_busy": sum(b for b, h in zip(busy, held_by) if not h),
        "reserve": cpu.get("reserve"),
        "mem_total_mib": mem.get("total_mib"),
        "mem_available_mib": mem.get("available_mib"),
        "mem_declared_mib": sum(j.get("mem_mib") or 0 for j in holders(s)),
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
    if pair == "up" and s["queue"]:
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
    fast = s.get("pytest", ())
    if fast:
        out.append("  pytest (the fast lane):")
    for j in fast:
        held = j["status"] == "running"
        when = j["started_at"] if held else j["submitted_at"]
        out.append(f"    #{j['id']} {j['name']}  {describe(j['cores'], j['mem_mib'], 0)}  "
                   f"{'running' if held else 'waiting'} {_ago(now - when)}"
                   + (f"  [{why[j['key']]}]" if not held and j["key"] in why else ""))
    out.append("  Running:" if s["running"] else "  Running: none")
    for j in s["running"]:
        out.append(f"    {j['label']}  {describe(j.get('cores') or 1, j.get('mem_mib') or 0, j['vram_mib'], j.get('legacy'))}  {j['status']}"
                   + ("" if j["in_terminal"] else f"  {_progress(j)}")
                   + f"  started {_ago(now - (j['started_at'] or now))} ago"
                   + (f"  cpus {','.join(map(str, j['cpus']))}" if j["in_terminal"] and j.get("cpus") else "")
                   + (f", priority {j['priority']}" if j.get("priority") else "")
                   + f"; {estimate(s, j['key'], j['started_at'], j.get('expect_seconds'))}"
                   + ("; went ahead on its estimate" if j.get("backfill") else ""))
        if j.get("log_at"):
            out.append(f"      last output {_ago(now - j['log_at'])} ago: {j['log_last'] or ''}")
    q = s.get("queue", ())
    out.append("  Queue (batch jobs and ws run, in the order they start):" if q else "  Queue: empty")
    proj = b.get("projection") or {}
    for n, j in enumerate(q, 1):
        extra = f", {j['interruptions']} interruption(s)" if j["interruptions"] else ""
        p = proj.get(j["key"])
        when = f", starts ~{time.strftime('%H:%M', time.localtime(p[0]))}" if p and p[0] and p[0] > now + 60 else ""
        e, note = expect_of(s, j["key"], j.get("expect_seconds"))
        exp = f", expected {_ago(e)}{note}" if e else ", no estimate"
        out.append(f"    {n}. [{j.get('priority') or 0:+d}] {j['label']}  "
                   f"{describe(j.get('cores') or 1, j.get('mem_mib') or 0, j['vram_mib'], j.get('legacy'))}  waiting "
                   f"{_ago(now - j['submitted_at'])}{exp}{when}{extra}")
        if j["key"] in why:
            out.append(f"       held up by: {why[j['key']]}")
    recent = s.get("recent", ())[:RECENT_TEXT]
    if recent:
        out.append("  Finished:")
    for j in recent:
        ran = (j["ended_at"] or now) - (j["started_at"] or j["ended_at"] or now)
        out.append(f"    {j['label']}  {outcome(j)}  ended {_ago(now - (j['ended_at'] or now))} ago"
                   + (f", ran {_ago(ran)}" if j["started_at"] else ", never started"))
    day = pytest_day(s)
    if day:
        out.append(f"  {day}")
    return "\n".join(out)
