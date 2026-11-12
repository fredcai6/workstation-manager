import dataclasses
import sqlite3
import tempfile
import unittest
from pathlib import Path

from workstation_manager import cpu
from workstation_manager.scheduler import CpuView, MemView, View, decide
from workstation_manager.store import Lease, Store

from tests.test_scheduler import MARGIN, PAIR, TOTAL, job

# The 9900X as topology() lists it: 12 physical cores alternating between CCDs.
CORES = [(0, 12), (6, 18), (1, 13), (7, 19), (2, 14), (8, 20),
         (3, 15), (9, 21), (4, 16), (10, 22), (5, 17), (11, 23)]
GB = 1024


def cjob(id, cores=1, mem=2 * GB, vram=0, status="queued", cpus=None):
    return dataclasses.replace(job(id, vram, status=status), cores=cores, mem_mib=mem, cpus=cpus)


def lease(id, cores=1, mem=2 * GB, status="waiting", cpus=None):
    return Lease(id=id, kind="pytest", name=f"l{id}", owner=None, pid=1, pid_start=None,
                 cores=cores, mem_mib=mem, cwd="/", unit=None, status=status,
                 requested_at=float(id), granted_at=None, ended_at=None, cpus=cpus)


def hview(queued=(), active=(), leases=(), busy=None, avail=50 * GB, unrealized=0,
          pair="up", used=None, reserve=2, lease_reserve=0, **kw):
    if used is None:
        used = (12000 if pair == "up" else 0) + sum(j.vram_mib for j in active)
    return View(now=10_000.0, queued=list(queued), active=list(active), pair=pair,
                last_interactive_at=10_000.0, vram_total_mib=TOTAL, vram_used_mib=used,
                cpu=CpuView(CORES, busy or [0.0] * 12, reserve, lease_reserve),
                mem=MemView(60 * GB, avail, 8 * GB, unrealized), leases=list(leases), **kw)


def run(v):
    return decide(v, MARGIN, 12500, 30, False)


class TestCores(unittest.TestCase):
    def test_cpu_job_gets_whole_cores_alternating_ccds(self):
        self.assertEqual(run(hview(queued=[cjob(1, cores=4)])),
                         [("start_job", 1, [0, 1, 6, 7, 12, 13, 18, 19])])

    def test_busy_cores_are_passed_over(self):
        busy = [0.9] + [0.0] * 11
        self.assertEqual(run(hview(queued=[cjob(1)], busy=busy)), [("start_job", 1, [6, 18])])

    def test_reserve_is_left_to_unmanaged_work(self):
        acts = run(hview(queued=[cjob(1, cores=10), cjob(2)]))
        self.assertEqual([a[:2] for a in acts], [("start_job", 1)])

    def test_unmanaged_load_beyond_the_reserve_shrinks_the_room(self):
        busy = [1.0] * 7 + [0.0] * 5     # seven cores busy with work nobody declared
        acts = run(hview(queued=[cjob(1, cores=6), cjob(2, cores=5)], busy=busy))
        self.assertEqual(acts, [])       # 5 left: the head waits, and the 5-core job behind it too
        busy = [1.0] * 6 + [0.0] * 6
        acts = run(hview(queued=[cjob(1, cores=6)], busy=busy))
        self.assertEqual(acts, [("start_job", 1, sorted(c for core in CORES[6:] for c in core))])

    def test_held_cores_are_not_double_booked(self):
        running = cjob(1, cores=2, status="running", cpus=[0, 12, 6, 18])
        self.assertEqual(run(hview(queued=[cjob(2)], active=[running])),
                         [("start_job", 2, [1, 13])])

    def test_head_waiting_for_cores_is_not_overtaken(self):
        running = cjob(1, cores=8, status="running", cpus=[c for core in CORES[:8] for c in core])
        acts = run(hview(queued=[cjob(2, cores=4), cjob(3, cores=2)], active=[running]))
        self.assertEqual(acts, [])


class TestMemory(unittest.TestCase):
    def test_declared_budget_blocks(self):
        running = cjob(1, mem=40 * GB, status="running", cpus=[0, 12])
        self.assertEqual(run(hview(queued=[cjob(2, mem=16 * GB)], active=[running])), [])

    def test_measured_available_blocks(self):
        self.assertEqual(run(hview(queued=[cjob(1, mem=16 * GB)], avail=20 * GB)), [])

    def test_memory_declared_but_not_yet_used_is_not_held_back_twice(self):
        # 11-12: runs peak at a median 50% of what they declare; holding the unused part
        # back on top of the budget kept an 18 GB job waiting with ~23 GB to spare.
        acts = run(hview(queued=[cjob(1, mem=18 * GB)], avail=30 * GB, unrealized=14 * GB))
        self.assertEqual([a[:2] for a in acts], [("start_job", 1)])
        # The declared budget still binds: 40 GB declared leaves 12 of the 52 GB budget.
        running = cjob(9, mem=40 * GB, status="running", cpus=[0, 12])
        self.assertEqual(run(hview(queued=[cjob(1, mem=18 * GB)], active=[running], avail=30 * GB)), [])


class TestCpuJobsAndTheCard(unittest.TestCase):
    def test_cpu_job_waiting_for_cores_does_not_send_the_pair_away(self):
        running = cjob(1, cores=10, status="running", cpus=[c for core in CORES[:10] for c in core])
        v = hview(queued=[cjob(2, cores=4)], active=[running])
        v.last_interactive_at = 0.0          # hold long over
        self.assertEqual(run(v), [])

    def test_pair_comes_back_while_cpu_jobs_wait(self):
        running = cjob(1, cores=10, status="running", cpus=[c for core in CORES[:10] for c in core])
        acts = run(hview(queued=[cjob(2, cores=4)], active=[running], pair="down"))
        self.assertEqual(acts, [("start_pair", "default")])

    def test_cpu_job_runs_beside_a_gpu_head_waiting_for_the_card(self):
        acts = run(hview(queued=[cjob(1, vram=25000), cjob(2, cores=2)]))
        self.assertEqual([a[:2] for a in acts], [("start_job", 2)])

    def test_but_not_on_the_cores_the_gpu_head_needs(self):
        acts = run(hview(queued=[cjob(1, cores=6, vram=25000), cjob(2, cores=5)]))
        self.assertEqual(acts, [])


class TestLeases(unittest.TestCase):
    def test_lease_is_granted_despite_unmanaged_load(self):
        acts = run(hview(queued=[cjob(1)], leases=[lease(7)], busy=[1.0] * 12))
        self.assertEqual(acts, [("grant_lease", 7, [0, 12])])


    def test_lease_goes_ahead_of_queued_batch(self):
        acts = run(hview(queued=[cjob(1, cores=10)], leases=[lease(7)]))
        self.assertEqual(acts, [("grant_lease", 7, [0, 12])])

    def test_batch_leaves_the_lease_cores(self):
        acts = run(hview(queued=[cjob(1, cores=9)], lease_reserve=2))
        self.assertEqual(acts, [])
        acts = run(hview(queued=[cjob(1, cores=8)], leases=[lease(7, cores=3)], lease_reserve=2))
        self.assertEqual([a[:2] for a in acts], [("grant_lease", 7)])   # 10 - 3 = 7 left for batch
        acts = run(hview(queued=[cjob(1, cores=8)], leases=[lease(7, cores=2)], lease_reserve=2))
        self.assertEqual([a[:2] for a in acts], [("grant_lease", 7), ("start_job", 1)])

    def test_pytest_shares_the_reserve_core(self):
        # 11-12: pytest held cores 1.7% of the time; it shares the desktop's core instead.
        full = cjob(8, cores=11, status="running", cpus=[])
        acts = run(hview(active=[full], leases=[lease(7)], reserve=1))
        self.assertEqual([a[:2] for a in acts], [("grant_lease", 7)])
        acts = run(hview(active=[full], leases=[lease(7, cores=2)], reserve=1))
        self.assertEqual(acts, [])          # one core left: a 2-core run waits
        # The queue still leaves the reserve alone.
        self.assertEqual(run(hview(queued=[cjob(1, cores=12)], reserve=1)), [])
        self.assertEqual([a[:2] for a in run(hview(queued=[cjob(1, cores=11)], reserve=1))],
                         [("start_job", 1)])

    def test_granted_lease_holds_its_cores(self):
        acts = run(hview(queued=[cjob(1)], leases=[lease(7, status="granted", cpus=[0, 12])]))
        self.assertEqual(acts, [("start_job", 1, [6, 18])])

    def test_leases_are_granted_during_a_take(self):
        acts = run(hview(leases=[lease(7)], take=True))
        self.assertIn(("grant_lease", 7, [0, 12]), acts)
        self.assertIn(("stop_pair",), acts)

    def test_small_lease_does_not_overtake_a_big_one(self):
        running = cjob(1, cores=10, status="running", cpus=[c for core in CORES[:10] for c in core])
        acts = run(hview(active=[running], leases=[lease(7, cores=4), lease(8, cores=1)]))
        self.assertEqual(acts, [])


class TestTopology(unittest.TestCase):
    def test_alternates_l3_groups_and_pairs_siblings(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "online").write_text("0-7\n")
            for c in range(8):
                core = c % 4
                sib = f"{core},{core + 4}"
                l3 = "0-1,4-5" if core < 2 else "2-3,6-7"
                (root / f"cpu{c}/topology").mkdir(parents=True)
                (root / f"cpu{c}/cache/index3").mkdir(parents=True)
                (root / f"cpu{c}/topology/thread_siblings_list").write_text(sib)
                (root / f"cpu{c}/cache/index3/shared_cpu_list").write_text(l3)
            self.assertEqual([c.cpus for c in cpu.topology(root)], [(0, 4), (2, 6), (1, 5), (3, 7)])

    def test_meter_counts_a_core_as_busy_as_its_busiest_sibling(self):
        with tempfile.TemporaryDirectory() as d:
            stat = Path(d) / "stat"
            def write(busy0, busy1):
                stat.write_text("cpu 0 0 0 0 0 0 0 0\n"
                                f"cpu0 {busy0} 0 0 {1000 - busy0} 0 0 0 0\n"
                                f"cpu1 {busy1} 0 0 {1000 - busy1} 0 0 0 0\n")
            m = cpu.Meter([cpu.Core(0, (0, 1), 0)], tau=10.0, stat=stat)
            write(0, 0)
            m.sample(now=0.0)
            stat.write_text("cpu 0 0 0 0 0 0 0 0\ncpu0 0 0 0 2000 0 0 0 0\ncpu1 500 0 0 1500 0 0 0 0\n")
            self.assertAlmostEqual(m.sample(now=10.0)[0], 0.5)  # first delta whole: sibling 1 half busy
            stat.write_text("cpu 0 0 0 0 0 0 0 0\ncpu0 0 0 0 3000 0 0 0 0\ncpu1 500 0 0 2500 0 0 0 0\n")
            self.assertAlmostEqual(m.sample(now=15.0)[0], 0.25)  # then smoothed: halfway to idle


class TestStore(unittest.TestCase):
    def test_older_database_gains_the_new_columns(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "broker.db"
            db = sqlite3.connect(path)
            db.execute("CREATE TABLE jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,"
                       " owner TEXT, vram_mib INTEGER NOT NULL, argv TEXT NOT NULL, cwd TEXT NOT NULL,"
                       " retries INTEGER NOT NULL, grace_seconds INTEGER NOT NULL, status TEXT NOT NULL,"
                       " submitted_at REAL NOT NULL, started_at REAL, ended_at REAL,"
                       " failures INTEGER NOT NULL DEFAULT 0, interruptions INTEGER NOT NULL DEFAULT 0,"
                       " exit_code INTEGER, stop_request TEXT, chunk INTEGER, chunks INTEGER, pct REAL,"
                       " progress_at REAL, note TEXT)")
            db.execute("INSERT INTO jobs (name, vram_mib, argv, cwd, retries, grace_seconds, status,"
                       " submitted_at) VALUES ('old', 18000, '[\"x\"]', '/', 2, 60, 'running', 0)")
            db.commit()
            db.close()
            j = Store(path).get(1)
            self.assertEqual((j.cores, j.mem_mib, j.cpus), (1, 0, None))
            # submitted under the old names, it keeps them
            self.assertEqual((j.legacy, j.unit), (True, "gpu-job-1"))
            from workstation_manager import runner
            self.assertEqual(runner.label(j), "gpu-manager.job=1")
            new = Store(path).get(Store(path).submit("n", None, 0, ["x"], "/", 2, 60, mem_mib=1024))
            self.assertEqual((new.legacy, new.unit, runner.label(new)),
                             (False, "ws-job-2", "workstation-manager.job=2"))

    def test_lease_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            s = Store(Path(d) / "broker.db")
            lid = s.request_lease("pytest", "p", None, 123, 456, 2, 2048, "/")
            s.grant_lease(lid, [0, 12, 6, 18])
            self.assertEqual(s.lease(lid).cpus, [0, 12, 6, 18])
            s.end_lease(lid)
            self.assertEqual(s.leases(), [])


if __name__ == "__main__":
    unittest.main()


class TestSubmitCommand(unittest.TestCase):
    def test_ws_submit_needs_mem(self):
        import os, subprocess, sys
        from workstation_manager.config import REPO
        with tempfile.TemporaryDirectory() as d:
            conf = Path(d) / "config.toml"
            conf.write_text(f'state_dir = "{d}/state"\n')
            env = dict(os.environ, WORKSTATION_MANAGER_CONFIG=str(conf))
            r = subprocess.run([sys.executable, str(REPO / "ws"), "submit", "--", "true"],
                               env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)
            self.assertIn("--mem is required", r.stderr)


class TestWhyWaiting(unittest.TestCase):
    """The reasons decide() gives for what it did not start, for the status page."""

    def why(self, v):
        out = {}
        acts = decide(v, MARGIN, 12500, 30, False, out)
        return acts, out

    def test_each_resource_names_itself(self):
        _, why = self.why(hview(queued=[cjob(1, mem=16 * GB)], avail=20 * GB))
        self.assertEqual(why, {"job:1": "memory: needs 16.0 GB, 12.0 GB free"})
        _, why = self.why(hview(queued=[cjob(1, cores=11)]))
        self.assertEqual(why, {"job:1": "cores: needs 11, 10 free"})
        _, why = self.why(hview(queued=[cjob(1, vram=25 * GB)]))
        self.assertTrue(why["job:1"].startswith("VRAM: needs 25.0 GB"), why)
        self.assertIn("pair", why["job:1"])

    def test_a_later_job_held_back_for_the_first(self):
        acts, why = self.why(hview(queued=[cjob(1, cores=9), cjob(2, cores=2)], busy=[0.0] * 12,
                                   active=[cjob(3, cores=2, status="running", cpus=[0, 12, 6, 18])]))
        self.assertEqual(acts, [])
        self.assertEqual(why["job:2"], "held back for #1 (first in line)")

    def test_started_jobs_and_granted_leases_have_no_reason(self):
        acts, why = self.why(hview(queued=[cjob(1)], leases=[lease(7)]))
        self.assertEqual(len(acts), 2)
        self.assertEqual(why, {})

    def test_reasons_change_no_decision(self):
        v = hview(queued=[cjob(1, mem=16 * GB), cjob(2)], leases=[lease(7, mem=30 * GB)], avail=20 * GB)
        self.assertEqual(run(v), self.why(v)[0])


def rlease(id, cores=1, mem=2 * GB, priority=0, expect=None, requested=None, status="waiting", cpus=None):
    """A `ws run` lease: it waits in the queue with jobs."""
    return dataclasses.replace(lease(id, cores, mem, status, cpus), kind="run", priority=priority,
                               expect_seconds=expect,
                               requested_at=float(id) if requested is None else requested)


def held(id, cores, start=9_000.0, expect=None, backfill=False, mem=2 * GB, vram=0):
    """A job running on the first `cores` physical cores."""
    cpus = [c for core in CORES[:cores] for c in core]
    return dataclasses.replace(cjob(id, cores=cores, mem=mem, vram=vram, status="running", cpus=cpus),
                               started_at=start, expect_seconds=expect, backfill=backfill)


class TestOneQueue(unittest.TestCase):
    """Jobs and `ws run` share one queue; pytest keeps its fast lane."""

    def test_ws_run_takes_its_place_by_priority(self):
        busy = dataclasses.replace(held(9, 6), priority=5)   # not stopped: order only
        acts = run(hview(queued=[cjob(1, cores=2)], leases=[rlease(7, cores=2, priority=5)],
                         active=[busy], lease_reserve=2))
        self.assertEqual([a[:2] for a in acts], [("grant_lease", 7)])
        acts = run(hview(queued=[dataclasses.replace(cjob(1, cores=2), priority=9)],
                         leases=[rlease(7, cores=2, priority=5)], active=[busy], lease_reserve=2))
        self.assertEqual([a[:2] for a in acts], [("start_job", 1)])

    def test_ws_run_is_held_back_for_an_earlier_job(self):
        why = {}
        acts = decide(hview(queued=[cjob(1, cores=8)], active=[held(9, 4)],
                                                   leases=[rlease(7, cores=2, requested=50_000)],
                                                   lease_reserve=2), MARGIN, 12500, 30, False, why)
        self.assertEqual(acts, [])
        self.assertEqual(why["lease:7"], "held back for #1 (first in line)")

    def test_only_pytest_uses_the_kept_cores_and_skips_unmanaged_load(self):
        acts = run(hview(leases=[rlease(7, cores=9)], lease_reserve=2))
        self.assertEqual(acts, [])
        acts = run(hview(leases=[lease(8, cores=9)], lease_reserve=2))
        self.assertEqual([a[:2] for a in acts], [("grant_lease", 8)])
        acts = run(hview(leases=[rlease(7, cores=2)], busy=[1.0] * 12))
        self.assertEqual(acts, [])


class TestEstimates(unittest.TestCase):
    """A later job goes ahead of the first in line only if it should finish before that one
    could start, and makes way if it runs over."""

    def setUp(self):
        # 8 cores for batch; a 4-core job runs until 13_600 (1h estimate from 10_000).
        self.busy = held(9, 4, start=10_000.0, expect=3600)
        self.head = cjob(1, cores=8)

    def go(self, later, active=None):
        return run(hview(queued=[self.head, later], active=active or [self.busy], lease_reserve=2))

    def test_short_job_fills_the_gap(self):
        later = dataclasses.replace(cjob(2, cores=2), expect_seconds=1800)
        (act,) = self.go(later)
        self.assertEqual((act[0], act[1], len(act[2]), act[3]), ("start_job", 2, 4, "backfill"))

    def test_too_long_or_unestimated_job_waits(self):
        self.assertEqual(self.go(dataclasses.replace(cjob(2, cores=2), expect_seconds=7200)), [])
        self.assertEqual(self.go(cjob(2, cores=2)), [])
        unknown = dataclasses.replace(self.busy, expect_seconds=None)
        self.assertEqual(self.go(dataclasses.replace(cjob(2, cores=2), expect_seconds=60), [unknown]), [])

    def test_ws_run_never_jumps_on_an_estimate(self):
        acts = run(hview(queued=[self.head], leases=[rlease(7, cores=2, expect=60, requested=50_000)],
                         active=[self.busy], lease_reserve=2))
        self.assertEqual(acts, [])

    def test_job_that_went_ahead_and_ran_over_makes_way(self):
        over = held(5, 4, start=5_000.0, expect=1800, backfill=True)
        busy = dataclasses.replace(self.busy, cpus=[c for core in CORES[4:8] for c in core])
        acts = run(hview(queued=[self.head], active=[busy, over], lease_reserve=2))
        self.assertEqual(acts, [])  # stopping it alone would not make 8 cores
        acts = run(hview(queued=[cjob(1, cores=4)], active=[busy, over], lease_reserve=2))
        self.assertEqual(acts, [("stop_job", 5, "requeue")])
        not_over = dataclasses.replace(over, expect_seconds=99_999)
        self.assertEqual(run(hview(queued=[cjob(1, cores=4)], active=[busy, not_over], lease_reserve=2)), [])

    def test_projection_is_a_timeline(self):
        from workstation_manager.scheduler import project
        a = dataclasses.replace(cjob(1, cores=8), expect_seconds=3600)
        b = dataclasses.replace(cjob(2, cores=4), expect_seconds=600)
        c = cjob(3, cores=8)
        d = cjob(4, cores=8)
        p = project(hview(queued=[a, b, c, d], active=[self.busy], lease_reserve=2), MARGIN)
        self.assertEqual(p["job:9"][1], 13_600.0)
        self.assertEqual(p["job:1"][:2], (13_600.0, 17_200.0))
        self.assertEqual(p["job:2"][:2], (17_200.0, 17_800.0))
        self.assertEqual(p["job:3"][:2], (17_800.0, None))
        self.assertEqual(p["job:4"][:2], (None, None))  # behind a job with no estimate

    def test_end_comes_from_the_estimate_not_chunk_progress(self):
        # 11-11: #363 resumed at "chunk 2 of 2" after 9 minutes; it was not half done.
        from workstation_manager.scheduler import expected_end
        j = dataclasses.replace(self.busy, started_at=9_500.0, chunk=2, chunks=2)
        self.assertEqual(expected_end(j, 10_000.0), (13_100.0, False))
        self.assertEqual(expected_end(dataclasses.replace(j, expect_seconds=None), 10_000.0), (None, False))
        late = dataclasses.replace(self.busy, started_at=1_000.0, chunk=None, chunks=None)
        self.assertEqual(expected_end(late, 10_000.0), (10_600.0, True))


class TestPriorityAcrossResources(unittest.TestCase):
    """Priority stops lower-priority running jobs for memory and cores, as for the card."""

    def go(self, queued, active, leases=(), why=None, **kw):
        return decide(hview(queued=queued, active=active, leases=leases, **kw),
                      MARGIN, 12500, 30, False, why)

    def hot(self, id=1, **kw):
        return dataclasses.replace(cjob(id, **kw), priority=10)

    def test_memory(self):
        low = held(5, 1, mem=40 * GB)
        why = {}
        acts = self.go([self.hot(mem=16 * GB)], [low], why=why)
        self.assertEqual(acts, [("stop_job", 5, "requeue")])
        self.assertEqual(why["job:1"], "stopping lower-priority #5 to make room")

    def test_cores(self):
        low = held(5, 8)
        self.assertEqual(self.go([self.hot(cores=4)], [low]), [("stop_job", 5, "requeue")])

    def test_newest_first_and_only_what_is_short(self):
        old = held(5, 1, start=1_000.0, mem=20 * GB)
        new_cpu = dataclasses.replace(held(6, 1, start=2_000.0, mem=1 * GB), cpus=[6, 18])
        new_mem = dataclasses.replace(held(7, 1, start=3_000.0, mem=20 * GB), cpus=[1, 13])
        # 52 GB budget, 41 committed: needs 20 GB, so one 20 GB job goes, the newest.
        acts = self.go([self.hot(mem=20 * GB)], [old, new_cpu, new_mem])
        self.assertEqual(acts, [("stop_job", 7, "requeue")])
        # A newer small job is passed over: stopping it is not needed.
        newest_cpu = dataclasses.replace(new_cpu, started_at=4_000.0)
        acts = self.go([self.hot(mem=20 * GB)], [old, newest_cpu, new_mem])
        self.assertEqual(acts, [("stop_job", 7, "requeue")])

    def test_equal_or_higher_priority_and_not_enough_stop_nothing(self):
        low = held(5, 1, mem=40 * GB)
        self.assertEqual(self.go([cjob(1, mem=16 * GB)], [low]), [])
        self.assertEqual(self.go([self.hot(mem=16 * GB)], [dataclasses.replace(low, priority=10)]), [])
        # Unmanaged memory use leaves too little even with the 12 GB job gone (2 + 12 < 16).
        small = held(5, 1, mem=12 * GB)
        self.assertEqual(self.go([self.hot(mem=16 * GB)], [small], avail=10 * GB), [])

    def test_stopping_jobs_count_as_gone(self):
        low = dataclasses.replace(held(5, 1, mem=40 * GB), status="stopping")
        self.assertEqual(self.go([self.hot(mem=16 * GB)], [low]), [])

    def test_not_in_a_reserve_window(self):
        low = held(5, 1, mem=40 * GB)
        self.assertEqual(self.go([self.hot(mem=16 * GB)], [low], reserved=True), [])

    def test_ws_run_with_priority_makes_room(self):
        low = held(5, 1, mem=40 * GB)
        acts = self.go([], [low], leases=[rlease(7, mem=16 * GB, priority=10)])
        self.assertEqual(acts, [("stop_job", 5, "requeue")])

    def test_overrun_stop_is_not_repeated(self):
        over = held(5, 4, start=5_000.0, expect=1800, backfill=True)
        busy = held(9, 4, start=10_000.0, expect=3600)
        busy = dataclasses.replace(busy, cpus=[c for core in CORES[4:8] for c in core])
        acts = self.go([self.hot(cores=4)], [busy, over], lease_reserve=2)
        self.assertEqual(acts, [("stop_job", 5, "requeue")])
