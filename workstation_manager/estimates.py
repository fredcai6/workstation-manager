"""Learned run times: what a run will take, given its author's --expect and the record.

Agents guess (`--expect`); finished runs say how good the guesses were. The model is
on the log of actual / guessed:

    log(actual / expect) = mu + beta * (load - load_mean) + gamma * (log expect - mean)
                           + a[where] + b[stem] + c[name] + e

- mu: how far off guesses are overall (on 11-12, runs took a median 0.31x their guess).
- gamma: long guesses are padded more than short ones (gamma < 0 shrinks them more).
- load: cores busy on the box during the run (held plus unmanaged, 0-12), from the
  per-minute samples; busy boxes run memory-bound work slower.
- where: the directory the work runs from (its script's, else its cwd); stem: the name
  without a trailing run number ("e0-2024a" -> "e0"); name: the exact name.
- Each level is shrunk toward zero by n / (n + SHRINK), so one odd run moves little.

Stdlib only, closed form (backfitted shrunken means): ~100s of runs, refit every few
minutes. `predict` gives the typical time and an upper one (P80) the scheduler plans
on, so a run that goes ahead on its estimate is rarely stopped for running over.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SHRINK = 3.0            # pseudo-runs at zero effect per group
Z80 = 0.8416            # standard normal 80th percentile
MIN_RUNS = 5            # below this, guesses are used as given
PASSES = 4              # backfitting sweeps


@dataclasses.dataclass
class Run:
    """One finished run: what was guessed, what it took, where and how loaded."""
    name: str
    where: str
    expect: float
    took: float
    load: Optional[float] = None
    ended: Optional[float] = None


@dataclasses.dataclass
class Model:
    n: int
    mu: float
    beta: float
    load_mean: float
    sigma: float
    effects: Dict[str, Dict[str, Tuple[float, int]]]   # level -> key -> (effect, runs)
    gamma: float = 0.0
    guess_mean: float = 0.0

    def summary(self) -> str:
        return (f"{self.n} runs: guesses x{math.exp(self.mu):.2f} overall, "
                f"x{math.exp(self.beta):.3f} per busy core, x{math.exp(self.gamma):.2f} per e-fold"
                f" of guess, spread x{math.exp(self.sigma):.2f}")


def stem(name: str) -> str:
    """The name without a trailing run label: "generator-anatomy-3" -> "generator-anatomy",
    "e0-2024a" -> "e0", "cfb-csta-weekly-new" stays."""
    parts = re.split(r"[-_ ]", name)
    while len(parts) > 1 and re.search(r"\d", parts[-1]):
        parts.pop()
    return "-".join(parts)


def where_of(argv: Sequence[str], cwd: str) -> str:
    """The directory the work comes from: its first absolute script, else its cwd."""
    for a in argv[:3]:
        if a.startswith("/") and not a.startswith(("/usr/", "/bin/")) and "/." not in a:
            return os.path.dirname(a)
    return cwd


def _levels(r: Run) -> Dict[str, str]:
    return {"where": r.where, "stem": stem(r.name), "name": r.name}


def _fit(y, base, load, lv, guess_slope=True):
    """Backfitted shrunken means on y = log(observed / base): an overall shift, slopes on
    load (None: none) and on log base, and effects by where, stem and name."""
    loads = [v for v in load if v is not None]
    load_mean = sum(loads) / len(loads) if loads else 0.0
    x = [(v - load_mean) if v is not None else 0.0 for v in load]
    guess_mean = sum(math.log(b) for b in base) / len(base)
    g = [math.log(b) - guess_mean for b in base]
    effects: Dict[str, Dict[str, Tuple[float, int]]] = {k: {} for k in ("where", "stem", "name")}
    mu = beta = gamma = 0.0
    eff = lambda i, skip=None: sum(effects[k].get(lv[i][k], (0.0, 0))[0] for k in effects if k != skip)
    # Shrunk slopes: a few pseudo-points of zero effect, so a thin record moves them little.
    slope = lambda xs, r: sum(a * b for a, b in zip(xs, r)) / (sum(a * a for a in xs) + 4.0)
    for _ in range(PASSES):
        mu = sum(y[i] - beta * x[i] - gamma * g[i] - eff(i) for i in range(len(y))) / len(y)
        if loads:
            beta = slope(x, [y[i] - mu - gamma * g[i] - eff(i) for i in range(len(y))])
        if guess_slope:
            gamma = slope(g, [y[i] - mu - beta * x[i] - eff(i) for i in range(len(y))])
        for k in effects:
            sums: Dict[str, List[float]] = {}
            for i in range(len(y)):
                sums.setdefault(lv[i][k], []).append(y[i] - mu - beta * x[i] - gamma * g[i] - eff(i, k))
            effects[k] = {grp: (sum(v) / (len(v) + SHRINK), len(v)) for grp, v in sums.items()}
    resid = [y[i] - mu - beta * x[i] - gamma * g[i] - eff(i) for i in range(len(y))]
    sigma = math.sqrt(sum(e * e for e in resid) / max(1, len(resid) - 1))
    return Model(len(y), mu, beta, load_mean, sigma, effects, gamma, guess_mean)


def fit(runs: Iterable[Run], guess_slope: bool = True) -> Optional[Model]:
    runs = [r for r in runs if r.expect and r.expect > 0 and r.took and r.took > 0]
    if len(runs) < MIN_RUNS:
        return None
    return _fit([math.log(r.took / r.expect) for r in runs], [r.expect for r in runs],
                [r.load for r in runs], [_levels(r) for r in runs], guess_slope)


def predict(m: Optional[Model], name: str, where: str, expect: Optional[float],
            load: Optional[float]) -> Optional[Tuple[float, float]]:
    """(typical seconds, P80 seconds) for a run, or None without a guess or a model."""
    if not m or not expect:
        return None
    mean, known = _mean(m, name, where, expect)
    if load is not None:
        mean += m.beta * (load - m.load_mean)
    # Less known about this run (no runs of its name or stem): a wider upper estimate.
    spread = m.sigma * math.sqrt(1 + 1 / (known + 1))
    return math.exp(mean), math.exp(mean + Z80 * spread)


def _mean(m: Model, name: str, where: str, base: float) -> Tuple[float, int]:
    """The model's log prediction for a run (without load), and how many runs of its
    name or stem it has seen."""
    lv = {"where": where, "stem": stem(name), "name": name}
    mean = math.log(base) + m.mu + m.gamma * (math.log(base) - m.guess_mean) \
        + sum(m.effects[k].get(g, (0.0, 0))[0] for k, g in lv.items())
    known = max(m.effects["name"].get(name, (0, 0))[1], m.effects["stem"].get(lv["stem"], (0, 0))[1])
    return mean, known


# -- memory: what a run peaks at, given what it declared ----------------------------
#
# Same model on log(peak / declared), without load (memory does not depend on it). The
# scheduler charges a run its P95 peak, not its declaration: declarations are padded
# (the median run peaks at half), and each run is still killed at its declaration, so
# the charge never goes above it. A run with no runs of its name or stem behind it gets
# a wider P95, so new work is charged near what it declared.

Z95 = 1.6449


@dataclasses.dataclass
class MemRun:
    name: str
    where: str
    declared: float             # MiB
    peak: float                 # MiB


def fit_memory(runs: Iterable[MemRun]) -> Optional[Model]:
    runs = [r for r in runs if r.declared > 0 and r.peak > 0]
    if len(runs) < MIN_RUNS:
        return None
    # A peak at or past the declaration was cut off there (killed): it says "at least".
    return _fit([math.log(min(r.peak, r.declared) / r.declared) for r in runs], [r.declared for r in runs],
                [None] * len(runs), [{"where": r.where, "stem": stem(r.name), "name": r.name} for r in runs])


def memory_charge(m: Optional[Model], name: str, where: str, declared: Optional[int]) -> Optional[int]:
    """MiB the scheduler counts for a run: its learned P95 peak, at most its declaration."""
    if not m or not declared:
        return None
    mean, known = _mean(m, name, where, declared)
    spread = m.sigma * math.sqrt(1 + 1 / (known + 1))
    return int(min(declared, math.ceil(math.exp(mean + Z95 * spread))))


def memory_runs_from(db) -> List[MemRun]:
    """Peaks of finished attempts. Cancelled and requeued attempts stopped part way, so
    their peaks may be short of a full run's; they are left out."""
    rows = db.execute(
        "SELECT a.name, a.mem_mib, a.peak_mib, j.argv, j.cwd, l.cwd FROM attempts a"
        " LEFT JOIN jobs j ON a.kind='job' AND j.id=a.ref"
        " LEFT JOIN leases l ON a.kind='lease' AND l.id=a.ref"
        " WHERE a.peak_mib > 0 AND a.mem_mib > 0 AND a.outcome IN ('done','failed','retry','ended')"
        " AND a.ended_at - a.started_at > 60").fetchall()
    return [MemRun(name, where_of(json.loads(argv) if argv else [], jcwd or lcwd or ""), mem, peak)
            for name, mem, peak, argv, jcwd, lcwd in rows]


def runs_from(db, samples: bool = True) -> List[Run]:
    """Finished runs from the broker's database: jobs (attempts summed, so a resumed run
    is one run) and `ws run` leases (grant to end). Failed, cancelled and given-up runs
    say nothing about how long a full run takes, so they are left out."""
    rows = db.execute(
        "SELECT j.name, j.argv, j.cwd, j.expect_seconds AS expect,"
        " COALESCE((SELECT SUM(a.ended_at - a.started_at) FROM attempts a WHERE a.kind='job'"
        "  AND a.ref=j.id AND a.started_at IS NOT NULL), j.ended_at - j.started_at) AS took,"
        " COALESCE((SELECT MIN(a.started_at) FROM attempts a WHERE a.kind='job' AND a.ref=j.id),"
        "  j.started_at) AS t0, j.ended_at AS t1"
        " FROM jobs j WHERE j.status='done' AND j.started_at IS NOT NULL AND j.expect_seconds > 0"
        " UNION ALL SELECT name, NULL, cwd, expect_seconds, ended_at - granted_at, granted_at, ended_at"
        " FROM leases WHERE kind='run' AND status='done' AND granted_at IS NOT NULL"
        " AND expect_seconds > 0").fetchall()
    out = []
    for name, argv, cwd, expect, took, t0, t1 in rows:
        load = None
        if samples and t0 and t1:
            r = db.execute("SELECT AVG(cores_held + COALESCE(unmanaged_busy, 0)) FROM samples"
                           " WHERE at BETWEEN ? AND ?", (t0, t1)).fetchone()
            load = r[0] if r and r[0] is not None else None
        out.append(Run(name, where_of(json.loads(argv) if argv else [], cwd or ""), expect, took, load, t1))
    return out
