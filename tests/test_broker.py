"""Runner, proxy and daemon bookkeeping, without systemd, Docker or a GPU."""

import http.client
import http.server
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from workstation_manager import config, daemon, proxy, runner
from workstation_manager.config import REPO, Config
from workstation_manager.store import Store


class Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.cfg = Config(state_dir=self.dir / "state")
        self.store = Store(self.cfg.db_path)

    def tearDown(self):
        self._tmp.cleanup()

    def submit(self, argv, **kw):
        args = dict(name="t", owner="test", vram_mib=1000, argv=argv, cwd=str(self.dir),
                    retries=2, grace_seconds=3)
        args.update(kw)
        jid = self.store.submit(**args)
        runner.save_env(self.cfg, jid, {"PATH": os.environ["PATH"]})
        return jid


class TestProgressParse(unittest.TestCase):
    def test_tennis_style_line(self):
        self.assertEqual(runner.parse_progress("19:47:00   chunk 34/144  rows 1078  628s"),
                         (34, 144, None))

    def test_chunk_with_percent(self):
        self.assertEqual(runner.parse_progress("chunk 2 of 4, 50%"), (2, 4, 50.0))

    def test_no_progress(self):
        self.assertIsNone(runner.parse_progress("loss 0.42 at 99% confidence"))


class TestWrap(Tmp):
    script = (
        "import signal, sys, time\n"
        "signal.signal(signal.SIGTERM, lambda *a: (open('got-term','w').write('1'), sys.exit(143)))\n"
        "print('chunk 1/3', flush=True)\n"
        "print('chunk 2/3 50%', flush=True)\n"
        "time.sleep(float(sys.argv[1]))\n"
    )

    def test_records_progress_and_exit(self):
        jid = self.submit([sys.executable, "-c", self.script, "0"])
        with mock.patch.object(runner, "stop_labelled_containers"):
            runner.wrap(self.cfg, jid)
        job = self.store.get(jid)
        self.assertEqual((job.chunk, job.chunks, job.pct), (2, 3, 50.0))
        self.assertEqual(runner.take_result(self.cfg, jid)["exit_code"], 0)
        self.assertIsNone(runner.take_result(self.cfg, jid), "result is consumed once")
        log = (self.cfg.job_dir(jid) / "log").read_text()
        self.assertIn("chunk 1/3", log)

    def test_result_carries_usage(self):
        jid = self.submit([sys.executable, "-c", "x = bytearray(50 * 2**20); print('ok')"])
        with mock.patch.object(runner, "stop_labelled_containers"):
            runner.wrap(self.cfg, jid)
        u = runner.take_result(self.cfg, jid)["usage"]
        self.assertGreaterEqual(u["wall_seconds"], 0)
        self.assertIn("peak_mib", u)  # None outside a unit with a readable cgroup

    def test_oom_line_marks_the_result(self):
        script = "import sys; print('RESOURCE_EXHAUSTED: Out of memory while trying to allocate', flush=True); sys.exit(1)"
        jid = self.submit([sys.executable, "-c", script])
        with mock.patch.object(runner, "stop_labelled_containers"):
            runner.wrap(self.cfg, jid)
        self.assertTrue(runner.take_result(self.cfg, jid)["oom"])

    def test_waits_for_container_removal_then_forces_it(self):
        seen = {"n": 0}
        def fake_labelled(job_label, all_states):
            seen["n"] += 1
            return ["abc"]  # never goes away on its own
        with mock.patch.object(runner, "_labelled", side_effect=fake_labelled), \
             mock.patch.object(runner.subprocess, "run") as run:
            runner.stop_labelled_containers("workstation-manager.job=1", wait_seconds=1.5)
        cmds = [c.args[0][:3] for c in run.call_args_list]
        self.assertIn(["docker", "stop", "-t"], cmds)
        self.assertIn(["docker", "rm", "-f"], cmds)
        self.assertGreater(seen["n"], 2)

    def test_sigterm_reaches_the_job(self):
        jid = self.submit([sys.executable, "-c", self.script, "30"])
        conf = self.dir / "config.toml"
        conf.write_text(f'state_dir = "{self.cfg.state_dir}"\n')
        env = dict(os.environ, WORKSTATION_MANAGER_CONFIG=str(conf))
        p = subprocess.Popen([sys.executable, str(REPO / "ws"), "_wrap", str(jid)], env=env)
        deadline = time.time() + 10
        while self.store.get(jid).chunks is None and time.time() < deadline:
            time.sleep(0.1)
        p.send_signal(signal.SIGTERM)
        p.wait(timeout=15)
        self.assertTrue((self.dir / "got-term").exists())
        self.assertEqual(runner.take_result(self.cfg, jid)["exit_code"], 143)


class TestReconcile(Tmp):
    def reconcile(self, jid, result, **fields):
        self.store.update(jid, **{"status": "running", "started_at": 1.0, **fields})
        d = daemon.Daemon(self.cfg, store=self.store)
        with mock.patch.object(runner, "is_active", return_value=False), \
             mock.patch.object(runner, "take_result", return_value=result):
            d.reconcile_jobs()
        return self.store.get(jid)

    def test_clean_exit_is_done(self):
        self.assertEqual(self.reconcile(self.submit(["x"]), {"exit_code": 0}).status, "done")

    def test_failure_retries_then_fails(self):
        jid = self.submit(["x"], retries=1)
        job = self.reconcile(jid, {"exit_code": 1})
        self.assertEqual((job.status, job.failures), ("queued", 1))
        job = self.reconcile(jid, {"exit_code": 1})
        self.assertEqual((job.status, job.failures), ("failed", 2))

    def test_out_of_memory_fails_without_retry(self):
        job = self.reconcile(self.submit(["x"]), {"exit_code": 1, "oom": True})
        self.assertEqual((job.status, job.failures), ("failed", 1))
        self.assertIn("out of GPU memory", self.store.events(0)[-1]["what"])

    def test_lost_unit_is_an_interruption_not_a_failure(self):
        job = self.reconcile(self.submit(["x"]), None)
        self.assertEqual((job.status, job.interruptions, job.failures), ("queued", 1, 0))

    def test_unit_that_never_ran_fails_after_its_retries(self):
        # 2026-11-08: a job whose directory was deleted restarted 3,044 times, every 5 s.
        jid = self.submit(["x"], retries=1)
        with mock.patch.object(runner, "unit_result", return_value="status=200/CHDIR"):
            for _ in range(2):
                job = self.reconcile(jid, None, started_at=time.time() - 3)
        self.assertEqual((job.status, job.failures, job.interruptions), ("failed", 2, 0))
        self.assertIn("200/CHDIR", self.store.events(0)[-1]["what"])

    def test_job_whose_directory_is_gone_is_not_started(self):
        jid = self.submit(["x"], cwd=str(self.dir / "gone"))
        d = daemon.Daemon(self.cfg, store=self.store)
        with mock.patch.object(runner, "start") as start:
            d.apply([("start_job", jid)])
        start.assert_not_called()
        self.assertEqual(self.store.get(jid).status, "failed")

    def test_each_attempt_is_recorded_with_what_it_used(self):
        jid = self.submit(["x"], cores=2, mem_mib=16384, retries=1)
        usage = {"wall_seconds": 7200, "cpu_seconds": 11520, "peak_mib": 10445}
        self.reconcile(jid, {"exit_code": 1, "usage": usage})
        self.reconcile(jid, {"exit_code": 0, "usage": usage})
        rows = self.store.attempts("job", jid)
        self.assertEqual([r["outcome"] for r in rows], ["retry", "done"])
        self.assertEqual((rows[1]["peak_mib"], rows[1]["cpu_seconds"], rows[1]["mem_mib"]),
                         (10445, 11520, 16384))
        self.assertEqual(self.store.events(0)[-1]["what"],
                         "used 1.6 of 2 cores, peak 10.2 of 16.0 GB over 2.0h")

    def test_requeue_keeps_place_in_line(self):
        first = self.submit(["x"])
        self.submit(["y"])
        job = self.reconcile(first, {"exit_code": 143}, stop_request="requeue")
        self.assertEqual(job.status, "queued")
        self.assertEqual(self.store.jobs(("queued",))[0].id, first)

    def test_cancel(self):
        job = self.reconcile(self.submit(["x"]), {"exit_code": 143}, stop_request="cancel")
        self.assertEqual(job.status, "cancelled")


class Backend(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(3):
            self.wfile.write(f"data: {i} {body.decode()}\n\n".encode())
            self.wfile.flush()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")


class TestProxy(unittest.TestCase):
    def setUp(self):
        self.backend = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        threading.Thread(target=self.backend.serve_forever, daemon=True).start()
        self.bport = self.backend.server_address[1]
        self.gate = proxy.Gate(queue_wait_seconds=10)
        self.front = proxy.serve(self.gate, ["127.0.0.1"], 0, self.bport)[0]
        self.port = self.front.server_address[1]

    def tearDown(self):
        self.front.shutdown()
        self.backend.shutdown()

    def post(self, out, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        c.request("POST", "/v1/chat/completions", body=b"hi",
                  headers={"Content-Type": "application/json", **(headers or {})})
        r = c.getresponse()
        out.append((r.status, r.read().decode()))

    def test_health_is_503_while_down(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        c.request("GET", "/health")
        self.assertEqual(c.getresponse().status, 503)

    def test_request_waits_for_pair_then_streams_through(self):
        out = []
        t = threading.Thread(target=self.post, args=(out,))
        t.start()
        time.sleep(0.5)
        self.assertEqual(out, [], "request must wait while the pair is down")
        self.gate.set_ok(self.bport, True)
        t.join(10)
        self.assertEqual(out[0][0], 200)
        self.assertEqual(out[0][1].count("data:"), 3)
        self.assertIn("hi", out[0][1])
        self.assertEqual(self.gate.in_flight, 0)

    def test_caller_that_hangs_up_is_dropped_not_sent_later(self):
        hits = []
        orig = Backend.do_POST

        def counting(handler):
            hits.append(1)
            orig(handler)
        Backend.do_POST = counting
        try:
            sock = socket.create_connection(("127.0.0.1", self.port))
            sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                         b"Content-Length: 2\r\n\r\nhi")
            deadline = time.time() + 5
            while self.gate.waiting == 0 and time.time() < deadline:
                time.sleep(0.05)
            self.assertEqual(self.gate.waiting, 1)
            sock.close()
            deadline = time.time() + 5
            while self.gate.waiting and time.time() < deadline:
                time.sleep(0.1)
            self.assertEqual(self.gate.waiting, 0, "hung-up caller must leave the queue")
            self.gate.set_ok(self.bport, True)
            time.sleep(0.5)
            self.assertEqual(hits, [], "a dropped request must never reach the backend")
        finally:
            Backend.do_POST = orig

    def post_model(self, model, out):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        c.request("POST", "/v1/chat/completions", body=json.dumps({"model": model}).encode(),
                  headers={"Content-Type": "application/json"})
        out.append(c.getresponse().status)

    def test_model_named_in_query_or_body(self):
        self.assertEqual(proxy.requested_model(b"", "/props?model=Qwen3.8-27B-Q4_K_M"), "Qwen3.8-27B-Q4_K_M")
        self.assertEqual(proxy.requested_model(b'{"model": "local"}', "/v1/chat/completions"), "local")
        self.assertIsNone(proxy.requested_model(b"", "/props"))

    def test_named_model_waits_for_it_and_local_takes_what_is_loaded(self):
        self.front.shutdown()
        self.front = proxy.serve(self.gate, ["127.0.0.1"], 0, self.bport,
                                 resolve=lambda m: "qwen" if m in ("qwen", "Qwen3.8-27B-Q4_K_M") else None)[0]
        self.port = self.front.server_address[1]
        self.gate.set_ok(self.bport, True)
        self.gate.set_loaded("dory")
        local, named = [], []
        self.post_model("local", local)
        self.assertEqual(local, [200], "an unnamed model is served by whatever is loaded")
        t = threading.Thread(target=self.post_model, args=("Qwen3.8-27B-Q4_K_M", named))
        t.start()
        time.sleep(0.5)
        self.assertEqual(named, [], "a request naming qwen waits while dory is loaded")
        self.assertEqual(dict(self.gate.waiting_for), {"qwen": 1})
        self.gate.set_loaded("qwen")
        t.join(10)
        self.assertEqual(named, [200])
        self.assertIn("qwen", self.gate.asked_at)

    def test_only_requests_marked_as_assistant_use_hold_the_pair(self):
        self.gate.set_ok(self.bport, True)
        self.post([])
        self.assertEqual(self.gate.last_interactive_at, 0.0)
        self.post([], headers={"X-GPU-Hold": "1"})
        self.assertGreater(self.gate.last_interactive_at, 0.0)
        self.assertFalse(proxy.holds_pair({}))
        self.assertFalse(proxy.holds_pair({"X-GPU-Hold": "0"}))
        self.assertTrue(proxy.holds_pair({"X-GPU-Hold": "1"}))

    def test_drain_waits_for_in_flight(self):
        self.gate.set_ok(self.bport, True)
        self.gate.in_flight = 1
        threading.Timer(0.5, self.gate.done).start()
        t0 = time.time()
        self.assertEqual(self.gate.close_and_drain(5), 0)
        self.assertGreater(time.time() - t0, 0.4)
        self.gate.reopen()



class TestWantedModel(Tmp):
    def wanted(self, asked, waiting=None, loaded="dory"):
        cfg = Config(state_dir=self.cfg.state_dir, default_model="dory", models={
            "dory": config.Model("dory", None, 12500), "qwen": config.Model("qwen", None, 27500)})
        d = daemon.Daemon(cfg, store=self.store)
        d.gate.asked_at, d.gate.loaded = dict(asked), loaded
        d.gate.waiting_for.update(waiting or {})
        return d.wanted_model(10_000.0)

    def test_default_when_nothing_asked(self):
        self.assertEqual(self.wanted({}), "dory")

    def test_named_model_held_for_the_hold(self):
        self.assertEqual(self.wanted({"qwen": 10_000.0 - 29 * 60}), "qwen")
        self.assertEqual(self.wanted({"qwen": 10_000.0 - 31 * 60}), "dory")

    def test_most_recent_name_wins_so_models_do_not_flap(self):
        self.assertEqual(self.wanted({"qwen": 9_000.0, "dory": 9_500.0}), "dory")

    def test_waiting_requests_decide_and_loaded_wins_ties(self):
        self.assertEqual(self.wanted({}, waiting={"qwen": 1}), "qwen")
        self.assertEqual(self.wanted({}, waiting={"qwen": 1, "dory": 1}, loaded="dory"), "dory")


class TestObservePair(Tmp):
    def test_pair_state_opens_the_backend_ports_the_proxy_checks(self):
        d = daemon.Daemon(self.cfg, store=self.store)
        with mock.patch("workstation_manager.interactive.state", return_value="up"):
            d.observe_pair()
        self.assertTrue(d.gate.ok[self.cfg.llm_backend])
        self.assertTrue(d.gate.ok[self.cfg.asr_backend])


if __name__ == "__main__":
    unittest.main()


class TestQueueOrder(unittest.TestCase):
    def test_priority_then_first_come(self):
        import tempfile
        from pathlib import Path
        from workstation_manager.store import Store
        with tempfile.TemporaryDirectory() as d:
            st = Store(Path(d) / "b.db")
            a = st.submit("a", None, 0, ["x"], "/", 0, 60, now=1.0, mem_mib=1)
            b = st.submit("b", None, 0, ["x"], "/", 0, 60, now=2.0, mem_mib=1, priority=10)
            c = st.submit("c", None, 0, ["x"], "/", 0, 60, now=3.0, mem_mib=1)
            st.update(c, priority=10)
            self.assertEqual([j.id for j in st.jobs(("queued",))], [b, c, a])


class TestUnitCgroup(unittest.TestCase):
    def test_bare_name_finds_a_ws_run_scope(self):
        # `ws run` leases record "ws-run-N"; the unit is ws-run-N.scope. Reading it as a
        # service found no cgroup, so the lease's memory read as its wrapper's 20 MB.
        from workstation_manager import cpu
        seen = []

        def show(cmd, **kw):
            seen.append(cmd[-1])
            out = "/app.slice/ws-run-7.scope" if cmd[-1] == "ws-run-7.scope" else ""
            return subprocess.CompletedProcess(cmd, 0, stdout=out + "\n", stderr="")
        with mock.patch.object(cpu.subprocess, "run", side_effect=show):
            self.assertEqual(cpu.unit_cgroup("ws-run-7"), Path("/sys/fs/cgroup/app.slice/ws-run-7.scope"))
        self.assertEqual(seen, ["ws-run-7.service", "ws-run-7.scope"])


class TestPastEstimates(Tmp):
    def done_job(self, name, attempts, ended):
        jid = self.store.submit(name=name, owner="t", vram_mib=0, argv=["x"], cwd="/", retries=2,
                                grace_seconds=3, expect_seconds=7200)
        t = ended - sum(attempts)
        for i, took in enumerate(attempts):
            last = i == len(attempts) - 1
            self.store.attempt("job", jid, "done" if last else "requeued", t + took, started_at=t)
            t += took
        self.store.update(jid, status="done", started_at=ended - attempts[-1], ended_at=ended)
        return jid

    def test_longest_of_the_last_runs_with_attempts_summed(self):
        self.done_job("fit", [600], 1000.0)
        self.done_job("fit", [300, 400], 2000.0)      # requeued once: 700 s in all
        for i in range(5):
            self.done_job("fit", [100], 3000.0 + i)    # the five newest push the others out
        self.done_job("other", [50], 1500.0)
        self.assertEqual(self.store.past_seconds(), {"fit": 100.0, "other": 50.0})
        self.assertEqual(self.store.past_seconds(last=7)["fit"], 700.0)

    def test_ws_run_counts_from_grant_and_pytest_and_give_ups_do_not(self):
        lid = self.store.request_lease("run", "spike: x.py", "t", 1, None, 1, 1024, "/", now=10.0)
        self.store.grant_lease(lid, [0], now=100.0)
        self.store.end_lease(lid, now=400.0)
        gave_up = self.store.request_lease("run", "never", "t", 1, None, 1, 1024, "/", now=10.0)
        self.store.end_lease(gave_up, now=900.0)
        py = self.store.request_lease("pytest", "tests", "t", 1, None, 1, 1024, "/", now=10.0)
        self.store.grant_lease(py, [0], now=20.0)
        self.store.end_lease(py, now=80.0)
        self.assertEqual(self.store.past_seconds(), {"spike: x.py": 300.0})

    def test_status_shows_the_estimate_in_use(self):
        from workstation_manager.board import estimate, expect_of
        s = {"at": 1000.0, "broker": {"estimates": {"job:3": 600}}}
        self.assertEqual(expect_of(s, "job:3", 7200), (600, " (past runs; --expect 2.0h)"))
        self.assertEqual(expect_of(s, "job:4", 7200), (7200, ""))
        self.assertTrue(estimate(s, "job:3", 900.0, 7200).startswith("expected 10m (past runs"))


class TestThreads(Tmp):
    def test_job_is_told_its_threads(self):
        from workstation_manager.cpu import thread_env
        env = thread_env({}, 2, [0, 6, 12, 18])
        self.assertEqual((env["WS_JOB_CORES"], env["WS_JOB_THREADS"], env["OMP_NUM_THREADS"]),
                         ("2", "4", "2"))

    def test_past_runs_say_how_busy_the_cores_were(self):
        from workstation_manager.cli import past_runs
        for name, cores, cpu in (("wide", 4, 4 * 600 * 0.55), ("one", 4, 4 * 600 * 0.95),
                                 ("both", 2, 2 * 600 * 1.9), ("pair", 2, 600 * 1.2)):
            self.store.attempt("job", 1, "done", 1600.0, started_at=1000.0, cores=cores,
                               usage={"cpu_seconds": cpu}, name=name)
        self.assertIn("kept 2.2 threads busy on 4 cores (about 2 would do)", past_runs(self.store, "wide"))
        self.assertIn("one per core", past_runs(self.store, "one"))
        self.assertIn("kept 1.2 threads busy on 2 cores (about 1 would do)", past_runs(self.store, "pair"))
        self.assertTrue(past_runs(self.store, "both").endswith("kept 3.8 threads busy on 2 cores"))


class TestCliEstimates(unittest.TestCase):
    def test_durations(self):
        from workstation_manager.cli import parse_duration
        for text, secs in (("2h", 7200), ("90m", 5400), ("1h30m", 5400), ("45s", 45), ("20", 1200),
                           ("1.5h", 5400)):
            self.assertEqual(parse_duration(text), secs, text)
        for bad in ("", "soon", "2d"):
            with self.assertRaises(Exception):
                parse_duration(bad)

    def test_default_name_says_whose_work_it_is(self):
        from workstation_manager.cli import default_name
        with mock.patch("os.getcwd", return_value="/home/t/projects/wt/cfb-oracle"):
            self.assertEqual(default_name(["/x/.venv/bin/python", "research/two_sided_c_game.py", "passers"]),
                             "cfb-oracle: two_sided_c_game.py")
            self.assertEqual(default_name(["bash", "scratch/pass4/t1.sh"]), "cfb-oracle: t1.sh")
            self.assertEqual(default_name(["make", "fit"]), "cfb-oracle: make")
