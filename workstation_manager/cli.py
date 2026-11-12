"""Command-line entry point.

  ws status [--json]                       card snapshot plus broker state
  ws submit --cores 4 --mem 8G [--vram 18G] [--name N] -- CMD...
  ws run [--cores 4] --mem 8G [--name N] -- CMD...   in the foreground, once it fits
  ws queue                                 broker state only
  ws cancel ID
  ws priority ID N
  ws progress --chunk 2/4 [--pct 50]       from inside a job (uses $WS_JOB_ID)
  ws log ID                                path of a job's log
  ws take / ws release                     operator override
  ws daemon                                the broker itself (systemd user unit)
"""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
from typing import Optional

from . import board, config
from .collector import collect
from .renderer import render_json, render_text
from .store import Store


def parse_size(text: str) -> int:
    """MiB from a size like 18G or 4500M (a bare number is GB)."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([GgMm])?i?[Bb]?\s*", text)
    if not m:
        raise argparse.ArgumentTypeError(f"a size like 18G or 4500M, got {text!r}")
    value = float(m.group(1))
    return int(value * 1024) if (m.group(2) or "G").upper() == "G" else int(value)


def parse_duration(text: str) -> int:
    """Seconds from a duration like 2h, 90m, 1h30m or 45s (a bare number is minutes)."""
    t = text.strip().lower()
    if re.fullmatch(r"\d+(?:\.\d+)?", t):
        return int(float(t) * 60)
    m = re.fullmatch(r"(?:(\d+(?:\.\d+)?)h)?\s*(?:(\d+(?:\.\d+)?)m)?\s*(?:(\d+)s)?", t)
    if not t or not m or not any(m.groups()):
        raise argparse.ArgumentTypeError(f"a duration like 2h, 90m or 1h30m, got {text!r}")
    h, mi, se = (float(g) if g else 0 for g in m.groups())
    return int(h * 3600 + mi * 60 + se)


def _expect(args, cmd: str) -> Optional[int]:
    """--expect, else $WS_EXPECT (so a launcher script passes it through); None and a
    message when neither is given."""
    if args.expect is not None:
        return args.expect
    env = os.environ.get("WS_EXPECT")
    if env:
        try:
            return parse_duration(env)
        except argparse.ArgumentTypeError as exc:
            print(f"ws {cmd}: WS_EXPECT: {exc}", file=sys.stderr)
            return None
    print(f"ws {cmd}: --expect is required: how long it should run, e.g. --expect 2h "
          "(or set WS_EXPECT). It is an estimate: nothing is stopped for running over.",
          file=sys.stderr)
    return None


def _ago_text(seconds: float) -> str:
    return f"{seconds / 3600:.1f}h" if seconds >= 5400 else f"{seconds / 60:.0f}m"


def default_name(argv) -> str:
    """The folder it runs in and the script, e.g. "cfb-oracle: two_sided_c_game.py":
    an interpreter's name ("python") says nothing about whose work it is."""
    prog = os.path.basename(argv[0])
    script = next((os.path.basename(a) for a in argv[1:]
                   if not a.startswith("-") and re.search(r"\.(py|sh|R|jl)$", a)), None)
    if script is None and prog in ("python", "python3", "bash", "sh", "uv") and len(argv) > 1:
        script = next((a for a in argv[1:] if not a.startswith("-")), None)
    return f"{os.path.basename(os.getcwd())}: {script or prog}"


def past_runs(store, name: str) -> str:
    """What earlier runs of this name took and used, to guess --expect and --mem from."""
    rows = store.db.execute(
        "SELECT ended_at - started_at AS took FROM jobs WHERE name=? AND status='done'"
        " AND started_at IS NOT NULL ORDER BY ended_at DESC LIMIT 5", (name,)).fetchall()
    took = sorted(r["took"] for r in rows if r["took"] and r["took"] > 0)
    peaks = sorted(r["peak_mib"] for r in store.db.execute(
        "SELECT peak_mib FROM attempts WHERE name=? AND outcome IN ('done','ended')"
        " AND peak_mib IS NOT NULL ORDER BY ended_at DESC LIMIT 5", (name,)).fetchall())
    use = store.db.execute(
        "SELECT cores, cpu_seconds / (ended_at - started_at) AS busy FROM attempts WHERE name=?"
        " AND outcome IN ('done','ended') AND cores > 0 AND cpu_seconds IS NOT NULL"
        " AND ended_at - started_at > 60 ORDER BY ended_at DESC LIMIT 5", (name,)).fetchall()
    if not took and not peaks and not use:
        return ""
    bits = []
    if took:
        bits.append(f"took {_ago_text(took[0])}" + (f"-{_ago_text(took[-1])}" if len(took) > 1 else ""))
    if peaks:
        bits.append(f"peak {peaks[0] / 1024:.1f}" + (f"-{peaks[-1] / 1024:.1f}" if len(peaks) > 1 else "") + " GB")
    if use:
        bits.append(core_use(use[0]["cores"], max(r["busy"] for r in use)))
    return f"earlier runs of {name} ({max(len(took), len(peaks), len(use))}): " + ", ".join(bits)


def core_use(cores: int, busy: float) -> str:
    """How busy a run kept its cores (each has two hardware threads), and what that suggests."""
    per = busy / cores
    text = f"kept {busy:.1f} threads busy on {cores} core{'s' if cores > 1 else ''}"
    if busy < cores - 0.5:          # at least half a core's worth never used
        return text + f" (about {max(1, round(busy))} would do)"
    if per < 1.2:
        return text + " (one per core: if it is compute-bound, size its pool from $WS_JOB_THREADS)"
    return text


def _store(cfg):
    return Store(cfg.db_path)


def cmd_status(args, cfg) -> int:
    if not math.isfinite(args.budget) or args.budget <= 0:
        print(f"gpu: error: --budget must be a finite positive number, got {args.budget!r}",
              file=sys.stderr)
        return 2
    result = collect(budget_seconds=args.budget)
    have_broker = cfg.db_path.exists()
    if args.as_json:
        data = json.loads(render_json(result))
        if have_broker:
            data["broker"] = board.snapshot(cfg, _store(cfg))
        print(json.dumps(data, indent=2))
    else:
        print(render_text(result))
        if have_broker:
            print(board.render_text(board.snapshot(cfg, _store(cfg))))
    return 0


def cmd_queue(args, cfg) -> int:
    s = board.snapshot(cfg, _store(cfg))
    print(json.dumps(s, indent=2) if args.as_json else board.render_text(s))
    return 0


def _command(args, verb):
    argv = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not argv:
        print(f"ws {verb}: give the command after --", file=sys.stderr)
    return argv


def _never_fits(cfg, cores, mem_mib, verb) -> bool:
    from .lease import limits
    max_cores, max_mem = limits(cfg)
    if cores < 1 or cores > max_cores:
        print(f"ws {verb}: --cores {cores} can never fit (1 to {max_cores}: the physical "
              f"cores less {cfg.reserve_cores} reserved)", file=sys.stderr)
        return True
    if mem_mib > max_mem:
        print(f"ws {verb}: {mem_mib} MiB can never fit ({max_mem} MiB: memory less the "
              f"{cfg.mem_margin_mib} MiB margin)", file=sys.stderr)
        return True
    return False


def cmd_submit(args, cfg) -> int:
    argv = _command(args, "submit")
    if not argv:
        return 2
    if args.mem is None:
        print("ws submit: --mem is required (peak memory, e.g. 8G)", file=sys.stderr)
        return 2
    expect = _expect(args, "submit")
    if expect is None:
        return 2
    if _never_fits(cfg, args.cores, args.mem, "submit"):
        return 2
    if args.vram:
        from .daemon import pick_card
        card = pick_card(collect(budget_seconds=2.0).devices, cfg.pci_id)
        total = card.vram_total_bytes // (1024 * 1024) if card and card.vram_total_bytes else None
        if total is not None and args.vram > total - cfg.margin_mib:
            print(f"ws submit: {args.vram} MiB VRAM can never fit ({total} MiB card, "
                  f"{cfg.margin_mib} MiB margin)", file=sys.stderr)
            return 2
    from . import runner
    from .store import describe
    store = _store(cfg)
    name = args.name or default_name(argv)
    owner = args.owner or os.environ.get("WS_OWNER") or getpass.getuser()
    job_id = store.submit(name, owner, args.vram, argv, os.getcwd(),
                          cfg.default_retries if args.retries is None else args.retries,
                          cfg.default_grace_seconds if args.grace is None else args.grace,
                          cores=args.cores, mem_mib=args.mem,
                          priority=args.priority, expect_seconds=expect)
    runner.save_env(cfg, job_id, dict(os.environ))
    from .lease import poke
    poke(cfg)
    print(f"submitted job #{job_id} ({name}, {describe(args.cores, args.mem or 0, args.vram)}, "
          f"expected {_ago_text(expect)}); log: {cfg.job_dir(job_id) / 'log'}")
    hint = past_runs(store, name)
    if hint:
        print(hint)
    return 0


def cmd_run(args, cfg) -> int:
    """Wait for cores and memory, then run the command here, pinned, with its memory
    capped in a scope of its own. Inside a job or a lease it just runs."""
    from . import lease as leases
    argv = _command(args, "run")
    if not argv:
        return 2
    if leases.held():
        os.execvp(argv[0], argv)
    if _never_fits(cfg, args.cores, args.mem, "run"):
        return 2
    expect = _expect(args, "run")
    if expect is None:
        return 2
    unit = f"ws-run-{os.getpid()}"
    name = args.name or default_name(argv)
    if cfg.db_path.exists():
        hint = past_runs(_store(cfg), name)
        if hint:
            print(f"ws: {hint}", file=sys.stderr)
    got = leases.acquire(cfg, "run", name, args.cores, args.mem, unit=unit,
                         priority=args.priority, expect_seconds=expect)
    cmd = argv
    if got is not None:
        leases.enter(got)
        cmd = ["systemd-run", "--user", "--scope", "--quiet", "--collect", f"--unit={unit}",
               "-p", f"MemoryMax={args.mem}M", "-p", "MemorySwapMax=0", "--", *argv]
    try:
        # Ctrl-C reaches the command through the terminal; a TERM or HUP sent to us is passed on.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        proc = subprocess.Popen(cmd)
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, lambda s, f: proc.send_signal(s))
        code = proc.wait()
    finally:
        leases.release(cfg, got)
    if code == -signal.SIGKILL and got is not None:
        print(f"ws run: the command was killed; if it ran past --mem {args.mem}M, "
              "raise --mem", file=sys.stderr)
    return 128 - code if code < 0 else code


def cmd_cancel(args, cfg) -> int:
    store = _store(cfg)
    job = store.get(args.id)
    if job is None or job.status in ("done", "failed", "cancelled"):
        print(f"ws cancel: no live job #{args.id}", file=sys.stderr)
        return 1
    store.update(job.id, stop_request="cancel")
    print(f"cancelling job #{job.id} ({job.status})")
    return 0


def cmd_priority(args, cfg) -> int:
    store = _store(cfg)
    job = store.get(args.id)
    if job is None or job.status in ("done", "failed", "cancelled"):
        print(f"ws priority: no live job #{args.id}", file=sys.stderr)
        return 1
    store.update(job.id, priority=args.priority)
    store.event(job.id, f"priority {job.priority} -> {args.priority}")
    from .lease import poke
    poke(cfg)
    print(f"job #{job.id} ({job.name}, {job.status}) now has priority {args.priority}")
    return 0


def cmd_progress(args, cfg) -> int:
    job_id = args.job or os.environ.get("WS_JOB_ID") or os.environ.get("GPU_JOB_ID")
    if not job_id:
        print("ws progress: not inside a job (no $WS_JOB_ID); pass --job", file=sys.stderr)
        return 2
    chunk = chunks = None
    if args.chunk:
        m = re.fullmatch(r"(\d+)/(\d+)", args.chunk)
        if not m:
            print("ws progress: --chunk N/M", file=sys.stderr)
            return 2
        chunk, chunks = int(m.group(1)), int(m.group(2))
    _store(cfg).progress(int(job_id), chunk=chunk, chunks=chunks, pct=args.pct, note=args.note)
    return 0


def cmd_take(args, cfg) -> int:
    store = _store(cfg)
    store.set_kv("take", True)
    store.event(None, "operator take")
    print("taken: running jobs are stopped and requeued, the interactive pair is stopped, "
          "nothing starts until `ws release`")
    return 0


def cmd_release(args, cfg) -> int:
    store = _store(cfg)
    store.set_kv("take", False)
    store.event(None, "operator release")
    print("released")
    return 0


def cmd_log(args, cfg) -> int:
    print(cfg.job_dir(args.id) / "log")
    return 0


def cmd_daemon(args, cfg) -> int:
    from .daemon import Daemon
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    Daemon(cfg).run()
    return 0


def cmd_wrap(args, cfg) -> int:
    from . import runner
    return runner.wrap(cfg, args.id)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ws", description="Workstation manager: queue and leases for the cores, memory and GPU, and the interactive tier.")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("status", help="Card snapshot plus broker state.")
    p.add_argument("--json", action="store_true", dest="as_json")
    p.add_argument("--budget", type=float, default=5.0, metavar="SECONDS",
                   help="Cooperative collection budget in seconds (default 5.0).")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("queue", help="Broker state only.")
    p.add_argument("--json", action="store_true", dest="as_json")
    p.set_defaults(fn=cmd_queue)

    p = sub.add_parser("submit", help="Queue a batch job.")
    p.add_argument("--cores", type=int, default=1, help="Physical cores (default 1).")
    p.add_argument("--mem", type=parse_size,
                   help="Peak memory, e.g. 8G (required); the job is killed past it.")
    p.add_argument("--vram", type=parse_size, default=0, help="Peak VRAM, e.g. 18G (default none).")
    p.add_argument("--name")
    p.add_argument("--owner", help="Who submitted it (default $WS_OWNER or user).")
    p.add_argument("--retries", type=int, help="Retries after a failing exit (default 2).")
    p.add_argument("--grace", type=int, help="Seconds between SIGTERM and SIGKILL (default 60).")
    p.add_argument("--priority", type=int, default=0,
                   help="Higher goes first and stops lower running jobs for the card (default 0).")
    p.add_argument("--expect", type=parse_duration,
                   help="How long it should run, e.g. 2h or 90m (required; or $WS_EXPECT). An estimate: "
                        "it places the job on the page's timeline and lets it go ahead of the first in "
                        "line when it should finish before that one could start.")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_submit)

    p = sub.add_parser("run", help="Run a command in the foreground once its cores and memory fit.")
    p.add_argument("--cores", type=int, default=1, help="Physical cores (default 1).")
    p.add_argument("--mem", type=parse_size, required=True,
                   help="Peak memory, e.g. 8G; the command is killed past it.")
    p.add_argument("--expect", type=parse_duration,
                   help="How long it should run, e.g. 20m (required; or $WS_EXPECT).")
    p.add_argument("--priority", type=int, default=0,
                   help="Its place in the queue it shares with jobs (default 0).")
    p.add_argument("--name")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("cancel", help="Cancel a queued or running job.")
    p.add_argument("id", type=int)
    p.set_defaults(fn=cmd_cancel)

    p = sub.add_parser("priority", help="Set a queued or running job's priority.")
    p.add_argument("id", type=int)
    p.add_argument("priority", type=int)
    p.set_defaults(fn=cmd_priority)

    p = sub.add_parser("progress", help="Report progress from inside a job.")
    p.add_argument("--chunk", help="N/M")
    p.add_argument("--pct", type=float)
    p.add_argument("--note")
    p.add_argument("--job", type=int)
    p.set_defaults(fn=cmd_progress)

    p = sub.add_parser("log", help="Print the path of a job's log.")
    p.add_argument("id", type=int)
    p.set_defaults(fn=cmd_log)

    sub.add_parser("take", help="Operator override: stop everything.").set_defaults(fn=cmd_take)
    sub.add_parser("release", help="End an operator take.").set_defaults(fn=cmd_release)
    sub.add_parser("daemon", help="Run the broker.").set_defaults(fn=cmd_daemon)
    p = sub.add_parser("_wrap")
    p.add_argument("id", type=int)
    p.set_defaults(fn=cmd_wrap)

    args = parser.parse_args(argv)
    if not getattr(args, "fn", None):
        parser.print_help()
        return 1
    return args.fn(args, config.load())


if __name__ == "__main__":
    sys.exit(main())
