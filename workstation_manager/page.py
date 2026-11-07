"""The status page: one self-contained HTML file rendered from the broker snapshot.

The daemon writes it every 30 s into each of `page_dirs`; point one at a static
site's directory to read it remotely.
"""

from __future__ import annotations

import html
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from .board import _ago, core_holders, host_summary
from .store import describe

# Reference palette (dataviz skill): categorical slots in fixed order, status
# roles reserved, light and dark surfaces.
CSS = """
:root{--surface:#fcfcfb;--panel:#f4f3f0;--track:#e6e5e0;--ink:#0b0b0b;--ink2:#52514e;--ink3:#7a7974;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--other:#a8a7a1;
--good:#0ca30c;--warning:#fab219;--critical:#d03b3b;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--panel:#242422;--track:#383835;--ink:#f0efec;
--ink2:#c3c2b7;--ink3:#8f8e86;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--other:#6a6964;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--ink);
font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:880px;margin:0 auto;padding:20px 16px 48px}
h1{font-size:20px;margin:0}h2{font-size:15px;margin:28px 0 8px;color:var(--ink2);font-weight:600}
.sub{color:var(--ink2);font-size:13px;margin-top:2px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:16px}
.tile{background:var(--panel);border-radius:8px;padding:12px 14px}
.tile .k{color:var(--ink2);font-size:13px}.tile .v{font-size:22px;font-weight:600;margin-top:2px}
.tile .n{color:var(--ink2);font-size:13px;margin-top:4px}
.bar{display:flex;gap:2px;height:14px;margin-top:10px}
.bar span{display:block;height:100%}.bar span:first-child{border-radius:4px 0 0 4px}
.bar span:last-child{border-radius:0 4px 4px 0}.bar span:only-child{border-radius:4px}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;margin-top:8px;font-size:13px;color:var(--ink2)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.meter{height:8px;background:var(--track);border-radius:4px;overflow:hidden;margin:8px 0 4px}
.meter b{display:block;height:100%;background:var(--s1);border-radius:4px}
.job{background:var(--panel);border-radius:8px;padding:12px 14px;margin-bottom:10px}
.job .t{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}
.job .name{font-weight:600}.muted{color:var(--ink2);font-size:13px}
.log{font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink2);margin-top:6px;
overflow-wrap:anywhere}
table{width:100%;border-collapse:collapse;font-size:14px}
th{text-align:left;color:var(--ink2);font-weight:500;font-size:13px;border-bottom:1px solid var(--track);padding:4px 8px 4px 0}
td{padding:6px 8px 6px 0;border-bottom:1px solid var(--track);vertical-align:top}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.st{white-space:nowrap}.st::before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;
margin-right:6px;vertical-align:1px;background:var(--ink3)}
.st.good::before{background:var(--good)}.st.warning::before{background:var(--warning)}
.st.critical::before{background:var(--critical)}
.empty{color:var(--ink2);font-size:14px}
.cores{display:grid;grid-template-columns:repeat(auto-fit,minmax(18px,1fr));gap:3px;margin-top:10px}
.cores span{display:block;height:22px;border-radius:3px;background:var(--track);position:relative;overflow:hidden}
.cores span b{position:absolute;left:0;right:0;bottom:0;background:var(--ink3)}
"""

STATUS = {"done": "good", "failed": "critical", "cancelled": "", "running": "good",
          "stopping": "warning", "queued": ""}
PAIR = {"up": ("good", "up"), "starting": ("warning", "starting"),
        "stopping": ("warning", "stopping"), "down": ("", "down")}


def _e(x) -> str:
    return html.escape("" if x is None else str(x))


def _gb(mib) -> str:
    return "?" if mib is None else f"{mib / 1024:.1f} GB"


def _clock(t) -> str:
    return time.strftime("%a %H:%M", time.localtime(t)) if t else ""


def _job_colour(i: int) -> str:
    """Jobs take categorical slots 2-4 in fixed order (slot 1 is the interactive pair)."""
    return ["var(--s2)", "var(--s3)", "var(--s4)"][i] if i < 3 else "var(--other)"


def _usage(s: Dict[str, Any], pair_vram_mib: int) -> str:
    """Who holds the card's VRAM: the pair, each job by its declaration, the rest."""
    b = s["broker"] or {}
    total, used = b.get("vram_total_mib"), b.get("vram_used_mib")
    if not total or used is None:
        return '<div class="n">VRAM unreadable</div>'
    parts: List = []
    pair_mib = 0
    if b.get("pair") in ("up", "starting", "stopping"):
        pair_mib = min(used, pair_vram_mib)
        parts.append((f"Interactive {b.get('model') or 'LLM'} + ASR", pair_mib, "var(--s1)"))
    for i, j in enumerate(s["running"]):
        parts.append((f"#{j['id']} {j['name']}", j["vram_mib"], _job_colour(i)))
    other = used - sum(p[1] for p in parts)
    if other > 256:
        parts.append(("other / not yet allocated", other, "var(--other)"))
    elif other < 0:  # a job declared more than it holds yet
        pass
    held = sum(p[1] for p in parts)
    free = max(0, total - max(used, held))
    segs = "".join(
        f'<span style="width:{100 * m / total:.2f}%;background:{c}" title="{_e(n)}: {_gb(m)}"></span>'
        for n, m, c in parts if m > 0)
    segs += f'<span style="width:{100 * free / total:.2f}%;background:var(--track)" title="free: {_gb(free)}"></span>'
    legend = "".join(f'<span><i style="background:{c}"></i>{_e(n)} {_gb(m)}</span>' for n, m, c in parts if m > 0)
    legend += f'<span><i style="background:var(--track)"></i>free {_gb(free)}</span>'
    return f'<div class="bar" role="img" aria-label="VRAM by holder">{segs}</div><div class="legend">{legend}</div>'


def _holder_colours(s: Dict[str, Any]) -> Dict[str, str]:
    """Running jobs keep their slots from the VRAM bar; leases share slot 1's hue."""
    out = {f"#{j['id']} {j['name']}": _job_colour(i) for i, j in enumerate(s["running"])}
    for l in s.get("leases", ()):
        out[f"lease #{l['id']} {l['name']}"] = "var(--s1)"
    return out


def _cores(s: Dict[str, Any]) -> str:
    """One cell per physical core: the holder's colour, or grey filled to how busy
    unmanaged work keeps it."""
    holders = core_holders(s)
    if not holders:
        return '<div class="n">cores not reported yet</div>'
    busy = ((s["broker"] or {}).get("cpu") or {}).get("busy") or [0.0] * len(holders)
    colours = _holder_colours(s)
    cores = ((s["broker"] or {}).get("cpu") or {}).get("cores") or [[i] for i in range(len(holders))]
    cells, legend = [], {}
    for i, h in enumerate(holders):
        where = "CPUs " + ",".join(map(str, cores[i]))
        if h:
            c = colours.get(h, "var(--other)")
            legend[h] = c
            cells.append(f'<span style="background:{c}" title="{where}: {_e(h)}"></span>')
        else:
            cells.append(f'<span title="{where}: unmanaged, {100 * busy[i]:.0f}% busy">'
                         f'<b style="height:{100 * busy[i]:.0f}%"></b></span>')
    leg = "".join(f'<span><i style="background:{c}"></i>{_e(n)}</span>' for n, c in legend.items())
    leg += '<span><i style="background:var(--ink3)"></i>unmanaged load</span>'
    return f'<div class="cores" role="img" aria-label="cores by holder">{"".join(cells)}</div><div class="legend">{leg}</div>'


def _memory(s: Dict[str, Any], h: Dict[str, Any]) -> str:
    total, avail = h["mem_total_mib"], h["mem_available_mib"]
    if not total:
        return '<div class="n">memory not reported yet</div>'
    colours = _holder_colours(s)
    parts = [(f"#{j['id']} {j['name']}", j.get("mem_mib") or 0) for j in s["running"]]
    parts += [(f"lease #{l['id']} {l['name']}", l["mem_mib"]) for l in s.get("leases", ())
              if l["status"] == "granted"]
    parts = [(n, m, colours.get(n, "var(--other)")) for n, m in parts if m]
    used = total - avail
    other = max(0, used - sum(m for _, m, _ in parts))
    parts.append(("everything else in use", other, "var(--other)"))
    free = max(0, total - sum(m for _, m, _ in parts))
    segs = "".join(f'<span style="width:{100 * m / total:.2f}%;background:{c}" title="{_e(n)}: {_gb(m)}"></span>'
                   for n, m, c in parts if m > 0)
    segs += f'<span style="width:{100 * free / total:.2f}%;background:var(--track)" title="free: {_gb(free)}"></span>'
    legend = "".join(f'<span><i style="background:{c}"></i>{_e(n)} {_gb(m)}</span>' for n, m, c in parts if m > 0)
    legend += f'<span><i style="background:var(--track)"></i>free {_gb(free)}</span>'
    return (f'<div class="bar" role="img" aria-label="memory by holder (declared)">{segs}</div>'
            f'<div class="legend">{legend}</div>')


def _progress(j, colour: str) -> str:
    pct = j.get("pct")
    if pct is None and j.get("chunks"):
        pct = 100.0 * (j["chunk"] - 1) / j["chunks"] if j["chunk"] else 0.0
    label = []
    if j.get("chunks"):
        label.append(f"chunk {j['chunk']} of {j['chunks']}")
    if j.get("pct") is not None:
        label.append(f"{j['pct']:.0f}%")
    text = ", ".join(label) or "no progress reported yet"
    meter = (f'<div class="meter" role="progressbar" aria-valuenow="{pct:.0f}" aria-valuemin="0" '
             f'aria-valuemax="100"><b style="width:{pct:.1f}%;background:{colour}"></b></div>') if pct is not None else ""
    return meter + f'<div class="muted">{_e(text)}</div>'


def render(s: Dict[str, Any], pair_vram_mib: int = 30000) -> str:
    now = s["at"]
    b = s["broker"]
    out = []
    if b is None:
        body = '<p class="empty">The broker has not run yet.</p>'
    else:
        stale = b["age_seconds"] > 60
        pcls, ptxt = PAIR.get(b["pair"], ("", b["pair"]))
        hold = ""
        if b["pair"] == "up" and s["queued"]:
            left = s["hold_seconds_left"] or 0
            hold = "batch may take the card now" if left <= 0 else f"batch may take the card in {_ago(left)}"
        elif b["pair"] == "down":
            hold = "card lent to batch; requests wait"
        elif s["last_interactive_at"]:
            hold = f"last interactive request {_ago(now - s['last_interactive_at'])} ago"
        busy = b.get("busy_pct")
        out.append('<div class="tiles">')
        out.append(f'<div class="tile" style="grid-column:1/-1"><div class="k">VRAM, {_e(b.get("card") or "GPU")}</div>'
                   f'<div class="v">{_gb(b.get("vram_used_mib"))} of {_gb(b.get("vram_total_mib"))}</div>'
                   f'<div class="n">GPU busy {"?" if busy is None else f"{busy}%"}</div>{_usage(s, pair_vram_mib)}</div>')
        h = host_summary(s)
        out.append(f'<div class="tile" style="grid-column:1/-1"><div class="k">Cores</div>'
                   f'<div class="v">{h["held"]} of {h["cores"]} held</div>'
                   f'<div class="n">~{h["unmanaged_busy"]:.1f} busy with unmanaged work · '
                   f'{_e(h["reserve"])} always left to it</div>{_cores(s)}</div>')
        out.append(f'<div class="tile" style="grid-column:1/-1"><div class="k">Memory</div>'
                   f'<div class="v">{_gb(h["mem_available_mib"])} available of {_gb(h["mem_total_mib"])}</div>'
                   f'<div class="n">holders shown at what they declared</div>{_memory(s, h)}</div>')
        out.append(f'<div class="tile"><div class="k">Interactive tier</div>'
                   f'<div class="v"><span class="st {pcls}">{_e(ptxt)}</span></div>'
                   f'<div class="n">{_e(b.get("model") or "")}'
                   f'{" → " + _e(b["want"]) if b.get("want") not in (None, b.get("model")) else ""}</div>'
                   f'<div class="n">{_e(hold)}</div></div>')
        out.append(f'<div class="tile"><div class="k">Batch</div><div class="v">{len(s["running"])} running</div>'
                   f'<div class="n">{len(s["queued"])} waiting</div></div>')
        out.append(f'<div class="tile"><div class="k">Broker</div><div class="v">'
                   f'<span class="st {"critical" if stale else "good"}">{"stale" if stale else "running"}</span></div>'
                   f'<div class="n">{"TAKEN by operator · " if s["take"] else ""}last tick {_ago(b["age_seconds"])} ago</div></div>')
        out.append("</div>")

        out.append("<h2>Running</h2>")
        if not s["running"]:
            out.append('<p class="empty">Nothing running.</p>')
        for i, j in enumerate(s["running"]):
            log = ""
            if j.get("log_at"):
                log = f'<div class="log">last output {_ago(now - j["log_at"])} ago: {_e(j.get("log_last"))}</div>'
            out.append(
                f'<div class="job"><div class="t"><span class="name">#{j["id"]} {_e(j["name"])}</span>'
                f'<span class="muted">{_e(describe(j.get("cores") or 1, j.get("mem_mib") or 0, j["vram_mib"], j.get("legacy")))} · started {_ago(now - (j["started_at"] or now))} ago'
                f'{" · " + _e(j["owner"]) if j["owner"] else ""}</span></div>'
                f'{_progress(j, _job_colour(i))}{log}</div>')

        out.append("<h2>Leases (pytest, ws run)</h2>")
        if not s.get("leases"):
            out.append('<p class="empty">None.</p>')
        else:
            out.append('<table><tr><th>Lease</th><th>Holds</th><th>State</th><th>Owner</th></tr>')
            for l in s["leases"]:
                when = l["granted_at"] if l["status"] == "granted" else l["requested_at"]
                state = f'{"held" if l["status"] == "granted" else "waiting"} {_ago(now - when)}'
                out.append(f'<tr><td>#{l["id"]} {_e(l["name"])}</td><td>{_e(describe(l["cores"], l["mem_mib"], 0))}</td>'
                           f'<td><span class="st {"good" if l["status"] == "granted" else ""}">{state}</span></td>'
                           f'<td>{_e(l["owner"])}</td></tr>')
            out.append("</table>")

        out.append("<h2>Queue</h2>")
        if not s["queued"]:
            out.append('<p class="empty">Empty.</p>')
        else:
            out.append('<table><tr><th>Job</th><th>Needs</th><th>Waiting</th><th>Notes</th><th>Owner</th></tr>')
            for j in s["queued"]:
                notes = []
                if j["interruptions"]:
                    notes.append(f"{j['interruptions']} interruption(s)")
                if j["failures"]:
                    notes.append(f"{j['failures']} failed attempt(s)")
                if j.get("chunks"):
                    notes.append(f"last at chunk {j['chunk']} of {j['chunks']}")
                out.append(f'<tr><td>#{j["id"]} {_e(j["name"])}</td><td>{_e(describe(j.get("cores") or 1, j.get("mem_mib") or 0, j["vram_mib"], j.get("legacy")))}</td>'
                           f'<td>{_ago(now - j["submitted_at"])}</td><td>{_e(", ".join(notes))}</td>'
                           f'<td>{_e(j["owner"])}</td></tr>')
            out.append("</table>")

        out.append("<h2>Finished</h2>")
        if not s["recent"]:
            out.append('<p class="empty">None yet.</p>')
        else:
            out.append('<table><tr><th>Job</th><th>Result</th><th>Ended</th><th class="num">Ran</th></tr>')
            for j in s["recent"]:
                ran = _ago((j["ended_at"] or now) - (j["started_at"] or j["ended_at"] or now))
                out.append(f'<tr><td>#{j["id"]} {_e(j["name"])}</td><td><span class="st {STATUS.get(j["status"], "")}">'
                           f'{_e(j["status"])}{"" if j["exit_code"] in (None, 0) else f" (exit {j["exit_code"]})"}</span></td>'
                           f'<td>{_clock(j["ended_at"])}</td><td class="num">{ran}</td></tr>')
            out.append("</table>")

        out.append("<h2>Last 24 hours</h2>")
        ev = list(reversed(s["events"]))[:40]
        if not ev:
            out.append('<p class="empty">No events.</p>')
        else:
            out.append("<table>")
            for e in ev:
                who = f"#{e['job_id']} " if e["job_id"] else ""
                out.append(f'<tr><td class="muted" style="width:90px">{_clock(e["at"])}</td><td>{_e(who + e["what"])}</td></tr>')
            out.append("</table>")
        body = "".join(out)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<meta http-equiv="refresh" content="30"><meta name="robots" content="noindex">'
            f'<title>Workstation status</title><style>{CSS}</style></head><body><main>'
            f'<h1>Workstation status</h1><div class="sub">Updated {time.strftime("%a %b %d, %H:%M:%S", time.localtime(now))}'
            f' · refreshes every 30 s</div>{body}</main></body></html>')


def publish(text: str, dirs) -> None:
    for d in dirs:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / ".index.html.tmp"
        tmp.write_text(text)
        os.chmod(tmp, 0o644)
        tmp.replace(d / "index.html")
