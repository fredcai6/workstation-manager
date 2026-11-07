"""Broker state in one SQLite file. The CLI writes requests; the daemon acts on them."""

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
    cpus TEXT
);
CREATE TABLE IF NOT EXISTS leases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    owner TEXT,
    pid INTEGER NOT NULL,
    pid_start INTEGER,
    cores INTEGER NOT NULL,
    mem_mib INTEGER NOT NULL,
    cwd TEXT,
    unit TEXT,
    status TEXT NOT NULL,
    requested_at REAL NOT NULL,
    granted_at REAL,
    ended_at REAL,
    cpus TEXT
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (
    at REAL NOT NULL, job_id INTEGER, what TEXT NOT NULL
);
"""

ACTIVE = ("running", "stopping")
# Columns added after the first schema, with their definitions, for older databases.
JOB_COLUMNS = {"cores": "INTEGER NOT NULL DEFAULT 1", "mem_mib": "INTEGER NOT NULL DEFAULT 0",
               "cpus": "TEXT"}


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

    @property
    def unit(self) -> str:
        return f"ws-job-{self.id}"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Lease:
    id: int
    kind: str                       # pytest | run
    name: str
    owner: Optional[str]
    pid: int
    pid_start: Optional[int]
    cores: int
    mem_mib: int
    cwd: Optional[str]
    unit: Optional[str]             # the scope `ws run` puts its command in
    status: str                     # waiting | granted | done
    requested_at: float
    granted_at: Optional[float]
    ended_at: Optional[float]
    cpus: Optional[List[int]]


def describe(cores: int, mem_mib: int, vram_mib: int) -> str:
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

    # -- jobs -------------------------------------------------------------
    def submit(self, name, owner, vram_mib, argv, cwd, retries, grace_seconds,
               now=None, cores=1, mem_mib=0) -> int:
        now = time.time() if now is None else now
        cur = self.db.execute(
            "INSERT INTO jobs (name, owner, vram_mib, argv, cwd, retries, grace_seconds,"
            " status, submitted_at, cores, mem_mib) VALUES (?,?,?,?,?,?,?, 'queued', ?,?,?)",
            (name, owner, vram_mib, json.dumps(argv), cwd, retries, grace_seconds, now,
             cores, mem_mib))
        self.event(cur.lastrowid, f"submitted ({describe(cores, mem_mib, vram_mib)})", now)
        return cur.lastrowid

    def _job(self, row) -> Job:
        d = dict(row)
        d["argv"] = json.loads(d["argv"])
        d["cpus"] = json.loads(d["cpus"]) if d.get("cpus") else None
        return Job(**d)

    def get(self, job_id: int) -> Optional[Job]:
        row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._job(row) if row else None

    def jobs(self, statuses=None) -> List[Job]:
        if statuses:
            q = "SELECT * FROM jobs WHERE status IN (%s) ORDER BY submitted_at, id" % (
                ",".join("?" * len(statuses)))
            rows = self.db.execute(q, tuple(statuses)).fetchall()
        else:
            rows = self.db.execute("SELECT * FROM jobs ORDER BY submitted_at, id").fetchall()
        return [self._job(r) for r in rows]

    def recent(self, limit=10) -> List[Job]:
        rows = self.db.execute(
            "SELECT * FROM jobs WHERE status IN ('done','failed','cancelled')"
            " ORDER BY ended_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._job(r) for r in rows]

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

    # -- leases: foreground work (pytest, `ws run`) holding cores while it runs --
    def request_lease(self, kind, name, owner, pid, pid_start, cores, mem_mib, cwd,
                      unit=None, now=None) -> int:
        cur = self.db.execute(
            "INSERT INTO leases (kind, name, owner, pid, pid_start, cores, mem_mib, cwd, unit,"
            " status, requested_at) VALUES (?,?,?,?,?,?,?,?,?, 'waiting', ?)",
            (kind, name, owner, pid, pid_start, cores, mem_mib, cwd, unit,
             time.time() if now is None else now))
        return cur.lastrowid

    def lease(self, lease_id: int) -> Optional[Lease]:
        row = self.db.execute("SELECT * FROM leases WHERE id=?", (lease_id,)).fetchone()
        return self._lease(row) if row else None

    def leases(self, statuses=("waiting", "granted")) -> List[Lease]:
        q = "SELECT * FROM leases WHERE status IN (%s) ORDER BY requested_at, id" % (
            ",".join("?" * len(statuses)))
        return [self._lease(r) for r in self.db.execute(q, tuple(statuses)).fetchall()]

    def _lease(self, row) -> Lease:
        d = dict(row)
        d["cpus"] = json.loads(d["cpus"]) if d.get("cpus") else None
        return Lease(**d)

    def grant_lease(self, lease_id: int, cpus, now=None) -> None:
        self.db.execute("UPDATE leases SET status='granted', granted_at=?, cpus=? WHERE id=?"
                        " AND status='waiting'",
                        (time.time() if now is None else now, json.dumps(list(cpus)), lease_id))

    def end_lease(self, lease_id: int, now=None) -> None:
        self.db.execute("UPDATE leases SET status='done', ended_at=? WHERE id=? AND status!='done'",
                        (time.time() if now is None else now, lease_id))

    def prune_leases(self, before: float) -> None:
        self.db.execute("DELETE FROM leases WHERE status='done' AND ended_at<?", (before,))

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

    def events(self, since: float) -> List[Dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM events WHERE at>=? ORDER BY at",
                               (since,)).fetchall()
        return [dict(r) for r in rows]
