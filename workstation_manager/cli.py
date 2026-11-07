"""Command-line entry point.

  ws status [--json]                    card snapshot plus broker state
  ws submit --vram 18G [--name N] -- CMD...
  ws queue                              broker state only
  ws cancel ID
  ws progress --chunk 2/4 [--pct 50]    from inside a job (uses $WS_JOB_ID)
  ws log ID                             path of a job's log
  ws take / ws release                 operator override
  ws daemon                             the broker itself (systemd user unit)
"""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import math
import os
import re
import sys

from . import board, config
from .collector import collect
from .renderer import render_json, render_text
from .store import Store


def parse_vram(text: str) -> int:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([GgMm])?i?[Bb]?\s*", text)
    if not m:
        raise argparse.ArgumentTypeError(f"VRAM like 18G or 4500M, got {text!r}")
    value = float(m.group(1))
    return int(value * 1024) if (m.group(2) or "G").upper() == "G" else int(value)


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


def cmd_submit(args, cfg) -> int:
    argv = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not argv:
        print("ws submit: give the command after --", file=sys.stderr)
        return 2
    from .daemon import pick_card
    card = pick_card(collect(budget_seconds=2.0).devices, cfg.pci_id)
    total = card.vram_total_bytes // (1024 * 1024) if card and card.vram_total_bytes else None
    if total is not None and args.vram > total - cfg.margin_mib:
        print(f"ws submit: {args.vram} MiB can never fit ({total} MiB card, "
              f"{cfg.margin_mib} MiB margin)", file=sys.stderr)
        return 2
    from . import runner
    store = _store(cfg)
    name = args.name or os.path.basename(argv[0])
    owner = args.owner or os.environ.get("WS_OWNER") or getpass.getuser()
    job_id = store.submit(name, owner, args.vram, argv, os.getcwd(),
                          cfg.default_retries if args.retries is None else args.retries,
                          cfg.default_grace_seconds if args.grace is None else args.grace)
    runner.save_env(cfg, job_id, dict(os.environ))
    print(f"submitted job #{job_id} ({name}, {args.vram} MiB); log: {cfg.job_dir(job_id) / 'log'}")
    return 0


def cmd_cancel(args, cfg) -> int:
    store = _store(cfg)
    job = store.get(args.id)
    if job is None or job.status in ("done", "failed", "cancelled"):
        print(f"ws cancel: no live job #{args.id}", file=sys.stderr)
        return 1
    store.update(job.id, stop_request="cancel")
    print(f"cancelling job #{job.id} ({job.status})")
    return 0


def cmd_progress(args, cfg) -> int:
    job_id = args.job or os.environ.get("WS_JOB_ID")
    if not job_id:
        print("ws progress: not inside a gpu job (no $WS_JOB_ID); pass --job", file=sys.stderr)
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
    parser = argparse.ArgumentParser(prog="ws", description="GPU manager: batch queue and interactive tier for one AMD card.")
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
    p.add_argument("--vram", type=parse_vram, required=True, help="Peak VRAM, e.g. 18G.")
    p.add_argument("--name")
    p.add_argument("--owner", help="Who submitted it (default $WS_OWNER or user).")
    p.add_argument("--retries", type=int, help="Retries after a failing exit (default 2).")
    p.add_argument("--grace", type=int, help="Seconds between SIGTERM and SIGKILL (default 60).")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_submit)

    p = sub.add_parser("cancel", help="Cancel a queued or running job.")
    p.add_argument("id", type=int)
    p.set_defaults(fn=cmd_cancel)

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
