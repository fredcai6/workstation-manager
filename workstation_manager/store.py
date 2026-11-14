"""Broker state in one SQLite file. The CLI writes requests; the daemon acts on them.

Every piece of work is a job, whatever runs it: `ws submit` (kind batch), `ws run`
(kind run) and pytest (kind pytest) share one table, one id sequence, one set of
states (queued, running, stopping, done, failed, cancelled; `ended` for foreground
work whose process went away without saying how it ended) and one history.

Rationale: the scheduler already treats them as one queue with one set of room
checks; the one real difference is who owns the process. The broker starts a batch
job as a systemd unit, so it can stop, requeue and retry it; a `ws run` or pytest
run is its caller's own process, which the broker only grants cores and watches.
`Job.in_terminal` carries that difference, nothing else does.
Rejected: separate jobs and leases tables. Their ids overlapped (#768 beside
`ws run #1326`), foreground results were never recorded, and lease rows were
deleted after a day, so run times of `ws run` were learned from one day's runs.
See: #24.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    owner TEXT,
    vram_mib INTEGER NOT NULL,
    argv TEXT NOT NULL,
    cwd TEXT NOT NULL,
    retries INTEGER NOT NULL,
    grace_seconds INTEGER NOT NULL,
    status TEXT NOT NULL,
    submitted_at REAL NOT NULL,
    started_at REAL,
    ended_at REAL,
    failures INTEGER NOT NULL DEFAULT 0,
    interruptions INTEGER NOT NULL DEFAULT 0,
    exit_code INTEGER,
    stop_request TEXT,
    chunk INTEGER,
    chunks INTEGER,
    pct REAL,
    progress_at REAL,
    note TEXT,
    cores INTEGER NOT NULL DEFAULT 1,
    mem_mib INTEGER NOT NULL DEFAULT 0,
    cpus TEXT,
    legacy INTEGER NOT NULL DEFAULT 0,
    priority INTEGER NOT NULL DEFAULT 0,
    expect_seconds INTEGER,
    backfill INTEGER NOT NULL DEFAULT 0,
    kind TEXT NOT NULL DEFAULT 'batch',
    pid INTEGER,
    pid_start INTEGER,
    scope TEXT
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (
    at REAL NOT NULL, job_id INTEGER, what TEXT NOT NULL
);
-- One row per run of a job (each attempt): what it declared and what it used.
CREATE TABLE IF NOT EXISTS attempts (
    kind TEXT NOT NULL,             -- job (see below)
    ref INTEGER NOT NULL,           -- the job id
    name TEXT,
    started_at REAL, ended_at REAL NOT NULL,
    outcome TEXT NOT NULL,          -- done | failed | retry | requeued | interrupted | cancelled | ended
    exit_code INTEGER,
    cores INTEGER, mem_mib INTEGER, vram_mib INTEGER, priority INTEGER,
    cpu_seconds REAL,               -- unit plus containers
    peak_mib INTEGER,               -- unit plus containers
    container_peak_mib INTEGER,
    expect_seconds INTEGER
);
-- kind is 'job' (ref is a jobs id); 'lease' rows are foreground runs recorded before
-- they were jobs, whose lease row is gone.
CREATE INDEX IF NOT EXISTS attempts_ref ON attempts (kind, ref);
-- The whole box once a minute, and what holds up each waiting job.
CREATE TABLE IF NOT EXISTS samples (
    at REAL NOT NULL,
    cores_held INTEGER, unmanaged_busy REAL,
    mem_available_mib INTEGER, mem_declared_mib INTEGER, mem_unrealized_mib INTEGER,
    vram_used_mib INTEGER, gpu_busy_pct INTEGER, pair TEXT, model TEXT,
    running INTEGER, queued INTEGER,           -- batch jobs
    leases_held INTEGER, leases_waiting INTEGER,  -- `ws run` and pytest, running and waiting
    waiting_on TEXT                 -- JSON: "job:N" -> reason (before the merge also "lease:N")
);
"""

ACTIVE = ("running", "stopping")
ENDED = ("done", "failed", "cancelled", "ended")
KINDS = ("batch", "run", "pytest")
# Columns added after the first schema, with their definitions, for older databases.
JOB_COLUMNS = {"cores": "INTEGER NOT NULL DEFAULT 1", "mem_mib": "INTEGER NOT NULL DEFAULT 0",
               "cpus": "TEXT", "legacy": "INTEGER NOT NULL DEFAULT 0",
               "priority": "INTEGER NOT NULL DEFAULT 0", "expect_seconds": "INTEGER",
               "backfill": "INTEGER NOT NULL DEFAULT 0",
               "kind": "TEXT NOT NULL DEFAULT 'batch'", "pid": "INTEGER", "pid_start": "INTEGER",
               "scope": "TEXT"}
ATTEMPT_COLUMNS = {"expect_seconds": "INTEGER"}


@dataclasses.dataclass
class Job:
    id: int
    name: str
    owner: Optional[str]
    vram_mib: int
    argv: List[str]
    cwd: str
    retries: int
    grace_seconds: int
    status: str
    submitted_at: float
    started_at: Optional[float]
    ended_at: Optional[float]
    failures: int
    interruptions: int
    exit_code: Optional[int]
    stop_request: Optional[str]
    chunk: Optional[int]
    chunks: Optional[int]
    pct: Optional[float]
    progress_at: Optional[float]
    note: Optional[str]
    cores: int = 1                  # physical cores
    mem_mib: int = 0                # peak memory, enforced as the unit's MemoryMax
    cpus: Optional[List[int]] = None  # logical CPUs it was given when started
    # Submitted under the gpu-manager names (before the rename, or through the `gpu`
    # command, retired 2026-11-11): kept so those rows still read as they ran (gpu-job-N,
    # VRAM only). No new job is submitted this way.
    legacy: bool = False
    # Higher goes first in the queue, and stops running jobs below it that keep it off the card.
    priority: int = 0
    expect_seconds: Optional[int] = None  # how long it should run (an estimate; never enforced)
    backfill: bool = False          # started ahead of the first in line, on its estimate
    kind: str = "batch"             # batch (`ws submit`) | run (`ws run`) | pytest
    # Foreground work (run, pytest): the caller's process, and the scope `ws run` puts
    # its command in.
    pid: Optional[int] = None
    pid_start: Optional[int] = None
    scope: Optional[str] = None

    @property
    def key(self) -> str:
        return f"job:{self.id}"

    @property
    def queued_at(self) -> float:
        return self.submitted_at

    @property
    def in_terminal(self) -> bool:
        """Its caller's own process runs it (`ws run`, pytest): the broker grants it cores
        and watches it, but cannot stop, requeue or retry it."""
        return self.kind != "batch"

    @property
    def fast(self) -> bool:
        """pytest: short, a caller blocked on it. It skips the queue; `ws run` waits in it."""
        return self.kind == "pytest"

    @property
    def unit(self) -> Optional[str]:
        if self.in_terminal:
            return self.scope
        return f"{'gpu' if self.legacy else 'ws'}-job-{self.id}"

    @property
    def tag(self) -> str:
        """How it runs, for the status page: "" for batch."""
        return {"run": "ws run", "pytest": "pytest"}.get(self.kind, "")

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def describe(cores: int, mem_mib: int, vram_mib: int, legacy: bool = False) -> str:
    if legacy:
        return f"{vram_mib / 1024:.1f} GB VRAM; cores and memory undeclared"
    bits = [f"{cores} core{'s' if cores != 1 else ''}", f"{mem_mib / 1024:.1f} GB"]
    if vram_mib:
        bits.append(f"{vram_mib / 1024:.1f} GB VRAM")
    return ", ".join(bits)


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=30, isolation_level=None,
                                  check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        have = {r["name"] for r in self.db.execute("PRAGMA table_info(jobs)")}
        for col, ddl in JOB_COLUMNS.items():
            if col not in have:
                self.db.execute(f"ALTER TABLE jobs ADD COLUMN {col} {ddl}")
                if col == "legacy":  # every job so far was submitted under the old names
                    self.db.execute("UPDATE jobs SET legacy=1")
        have = {r["name"] for r in self.db.execute("PRAGMA table_info(attempts)")}
        for col, ddl in ATTEMPT_COLUMNS.items():
            if col not in have:
                self.db.execute(f"ALTER TABLE attempts ADD COLUMN {col} {ddl}")
        self._absorb_leases()

    def _absorb_leases(self) -> None:
        """Move the rows of a pre-merge `leases` table into jobs, numbered after the
        last job (job ids stay, since running units and logs are named by them), and
        point their attempts at the new ids. The old table is kept as leases_premerge.
        A process still running the old code recreates an empty `leases` table when it
        opens the store; rows that reach it are absorbed too, and it is dropped."""
        if not self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='leases'").fetchone():
            return
        cols = {r["name"] for r in self.db.execute("PRAGMA table_info(leases)")}
        get = lambda r, c, d=None: r[c] if c in cols else d
        self.db.execute("BEGIN IMMEDIATE")
        try:
            rows = self.db.execute("SELECT * FROM leases ORDER BY id").fetchall()
            for r in rows:
                status = {"waiting": "queued", "granted": "running"}.get(r["status"])
                if status is None:  # done: how it ended was never recorded
                    status = "ended" if r["granted_at"] else "cancelled"
                cur = self.db.execute(
                    "INSERT INTO jobs (name, owner, vram_mib, argv, cwd, retries, grace_seconds, status,"
                    " submitted_at, started_at, ended_at, cores, mem_mib, cpus, priority, expect_seconds,"
                    " kind, pid, pid_start, scope) VALUES (?,?,0,'[]',?,0,0,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (r["name"], r["owner"], r["cwd"] or "", status, r["requested_at"], r["granted_at"],
                     r["ended_at"], r["cores"], r["mem_mib"], r["cpus"], get(r, "priority", 0),
                     get(r, "expect_seconds"), r["kind"], r["pid"], r["pid_start"], r["unit"]))
                self.db.execute("UPDATE attempts SET kind='job', ref=? WHERE kind='lease' AND ref=?",
                                (cur.lastrowid, r["id"]))
            if self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table'"
                               " AND name='leases_premerge'").fetchone():
                self.db.execute("DROP TABLE leases")
            else:
                self.db.execute("ALTER TABLE leases RENAME TO leases_premerge")
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    # -- jobs -------------------------------------------------------------
    def submit(self, name, owner, vram_mib, argv, cwd, retries, grace_seconds,
               now=None, cores=1, mem_mib=0, legacy=False, priority=0, expect_seconds=None) -> int:
        now = time.time() if now is None else now
        cur = self.db.execute(
            "INSERT INTO jobs (name, owner, vram_mib, argv, cwd, retries, grace_seconds,"
            " status, submitted_at, cores, mem_mib, legacy, priority, expect_seconds)"
            " VALUES (?,?,?,?,?,?,?, 'queued', ?,?,?,?,?,?)",
            (name, owner, vram_mib, json.dumps(argv), cwd, retries, grace_seconds, now,
             cores, mem_mib, int(legacy), priority, expect_seconds))
        self.event(cur.lastrowid, f"submitted ({describe(cores, mem_mib, vram_mib)})", now)
        return cur.lastrowid

    def _job(self, row) -> Job:
        d = dict(row)
        d["argv"] = json.loads(d["argv"])
        d["cpus"] = json.loads(d["cpus"]) if d.get("cpus") else None
        d["legacy"] = bool(d.get("legacy"))
        d["backfill"] = bool(d.get("backfill"))
        return Job(**d)

    def get(self, job_id: int) -> Optional[Job]:
        row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._job(row) if row else None

    def jobs(self, statuses=None, kinds=None) -> List[Job]:
        where, args = [], []
        for col, vals in (("status", statuses), ("kind", kinds)):
            if vals:
                where.append(f"{col} IN ({','.join('?' * len(vals))})")
                args += list(vals)
        q = "SELECT * FROM jobs" + (" WHERE " + " AND ".join(where) if where else "") \
            + " ORDER BY priority DESC, submitted_at, id"
        return [self._job(r) for r in self.db.execute(q, args).fetchall()]

    def recent(self, limit=10, kinds=("batch", "run")) -> List[Job]:
        rows = self.db.execute(
            "SELECT * FROM jobs WHERE status IN ('done','failed','cancelled','ended') AND kind IN (%s)"
            " ORDER BY ended_at DESC LIMIT ?" % ",".join("?" * len(kinds)), (*kinds, limit)).fetchall()
        return [self._job(r) for r in rows]

    def tally(self, kind: str, since: float) -> Dict[str, int]:
        """How the work of `kind` that ended since `since` ended: status -> count."""
        return {r[0]: r[1] for r in self.db.execute(
            "SELECT status, COUNT(*) FROM jobs WHERE kind=? AND ended_at>=? GROUP BY status",
            (kind, since)).fetchall()}

    def update(self, job_id: int, **fields) -> None:
        if "cpus" in fields and fields["cpus"] is not None:
            fields["cpus"] = json.dumps(fields["cpus"])
        cols = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    def progress(self, job_id: int, chunk=None, chunks=None, pct=None, note=None,
                 now=None) -> None:
        fields = {"progress_at": time.time() if now is None else now}
        for k, v in (("chunk", chunk), ("chunks", chunks), ("pct", pct), ("note", note)):
            if v is not None:
                fields[k] = v
        self.update(job_id, **fields)

    # -- foreground work: `ws run` and pytest, run by their caller's own process --
    def request(self, kind, name, owner, pid, pid_start, cores, mem_mib, cwd, scope=None,
                argv=(), now=None, priority=0, expect_seconds=None) -> int:
        """Queue foreground work; its caller waits for `grant` and runs it itself."""
        cur = self.db.execute(
            "INSERT INTO jobs (name, owner, vram_mib, argv, cwd, retries, grace_seconds, status,"
            " submitted_at, cores, mem_mib, priority, expect_seconds, kind, pid, pid_start, scope)"
            " VALUES (?,?,0,?,?,0,0,'queued',?,?,?,?,?,?,?,?,?)",
            (name, owner, json.dumps(list(argv)), cwd, time.time() if now is None else now,
             cores, mem_mib, priority, expect_seconds, kind, pid, pid_start, scope))
        return cur.lastrowid

    def grant(self, job_id: int, cpus, now=None) -> None:
        self.db.execute("UPDATE jobs SET status='running', started_at=?, cpus=? WHERE id=?"
                        " AND status='queued' AND kind!='batch'",
                        (time.time() if now is None else now, json.dumps(list(cpus)), job_id))

    def finish(self, job_id: int, exit_code: Optional[int], now=None) -> None:
        """Foreground work ends: done or failed by its exit code; `ended` when its
        process went away without one; cancelled if it was never granted (its caller
        stopped waiting, or ran without the queue)."""
        now = time.time() if now is None else now
        self.db.execute(
            "UPDATE jobs SET ended_at=?, exit_code=?, status=CASE"
            " WHEN status='queued' THEN 'cancelled' WHEN ? IS NULL THEN 'ended'"
            " WHEN ?=0 THEN 'done' ELSE 'failed' END"
            " WHERE id=? AND status IN ('queued','running') AND kind!='batch'",
            (now, exit_code, exit_code, exit_code, job_id))

    def unrecorded(self, since: float) -> List[Job]:
        """Foreground work that ended since `since` and has no attempt row yet."""
        rows = self.db.execute(
            "SELECT * FROM jobs j WHERE kind!='batch' AND status IN ('done','failed','cancelled','ended')"
            " AND ended_at>=? AND NOT EXISTS (SELECT 1 FROM attempts a WHERE a.kind='job' AND a.ref=j.id)"
            " ORDER BY id", (since,)).fetchall()
        return [self._job(r) for r in rows]

    # -- key/value and history ---------------------------------------------
    def get_kv(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_kv(self, key: str, value) -> None:
        self.db.execute("INSERT INTO kv (key, value) VALUES (?, ?) ON CONFLICT(key)"
                        " DO UPDATE SET value=excluded.value", (key, json.dumps(value)))

    def event(self, job_id, what, now=None) -> None:
        self.db.execute("INSERT INTO events (at, job_id, what) VALUES (?,?,?)",
                        (time.time() if now is None else now, job_id, what))

    def attempt(self, kind: str, ref: int, outcome: str, ended_at: float, usage=None, **fields) -> None:
        usage = usage or {}
        row = dict(kind=kind, ref=ref, outcome=outcome, ended_at=ended_at,
                   cpu_seconds=usage.get("cpu_seconds"), peak_mib=usage.get("peak_mib"),
                   container_peak_mib=usage.get("container_peak_mib"), **fields)
        self.db.execute(f"INSERT INTO attempts ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                        tuple(row.values()))

    def attempts(self, kind: str, ref: int) -> List[Dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM attempts WHERE kind=? AND ref=? ORDER BY ended_at",
                               (kind, ref)).fetchall()
        return [dict(r) for r in rows]

    def sample(self, at: float, **fields) -> None:
        row = dict(at=at, **fields)
        self.db.execute(f"INSERT INTO samples ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                        tuple(row.values()))

    def prune_samples(self, before: float) -> None:
        self.db.execute("DELETE FROM samples WHERE at<?", (before,))

    def events(self, since: float) -> List[Dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM events WHERE at>=? ORDER BY at",
                               (since,)).fetchall()
        return [dict(r) for r in rows]
