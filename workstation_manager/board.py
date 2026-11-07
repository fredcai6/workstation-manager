"""Broker state as one dict (for `ws status --json` and the status page) and as text."""

from __future__ import annotations

import time
from typing import Any, Dict

from .config import Config
from .store import ACTIVE, Store


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
        "recent": [j.to_dict() for j in store.recent(10)],
        "events": store.events(now - 86400),
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
    out.append("  Running:" if s["running"] else "  Running: none")
    for j in s["running"]:
        out.append(f"    #{j['id']} {j['name']}  {j['vram_mib'] / 1024:.1f} GB  {j['status']}"
                   f"  {_progress(j)}  started {_ago(now - (j['started_at'] or now))} ago"
                   + (f"  ({j['owner']})" if j["owner"] else ""))
        if j.get("log_at"):
            out.append(f"      last output {_ago(now - j['log_at'])} ago: {j['log_last'] or ''}")
    out.append("  Queue:" if s["queued"] else "  Queue: empty")
    for j in s["queued"]:
        extra = f", {j['interruptions']} interruption(s)" if j["interruptions"] else ""
        out.append(f"    #{j['id']} {j['name']}  {j['vram_mib'] / 1024:.1f} GB  waiting "
                   f"{_ago(now - j['submitted_at'])}{extra}"
                   + (f"  ({j['owner']})" if j["owner"] else ""))
    return "\n".join(out)
