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
        self.store.update(jid, status="running", started_at=1.0, **fields)
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
