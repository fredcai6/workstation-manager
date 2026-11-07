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
    note TEXT
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (
    at REAL NOT NULL, job_id INTEGER, what TEXT NOT NULL
);
"""

ACTIVE = ("running", "stopping")


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

    @property
    def unit(self) -> str:
        return f"ws-job-{self.id}"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=30, isolation_level=None,
                                  check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    # -- jobs -------------------------------------------------------------
    def submit(self, name, owner, vram_mib, argv, cwd, retries, grace_seconds,
               now=None) -> int:
        now = time.time() if now is None else now
        cur = self.db.execute(
            "INSERT INTO jobs (name, owner, vram_mib, argv, cwd, retries, grace_seconds,"
            " status, submitted_at) VALUES (?,?,?,?,?,?,?, 'queued', ?)",
            (name, owner, vram_mib, json.dumps(argv), cwd, retries, grace_seconds, now))
        self.event(cur.lastrowid, f"submitted ({vram_mib} MiB)", now)
        return cur.lastrowid

    def _job(self, row) -> Job:
        d = dict(row)
        d["argv"] = json.loads(d["argv"])
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
        cols = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    def progress(self, job_id: int, chunk=None, chunks=None, pct=None, note=None,
                 now=None) -> None:
        fields = {"progress_at": time.time() if now is None else now}
        for k, v in (("chunk", chunk), ("chunks", chunks), ("pct", pct), ("note", note)):
            if v is not None:
                fields[k] = v
        self.update(job_id, **fields)

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
