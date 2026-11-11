"""End to end on this machine: real systemd units, real pinning, a real pytest run
through the hook. The broker is a private one (its own state dir, no proxy); the
card and the interactive pair are stubbed. Skipped without a systemd user manager."""

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from workstation_manager import config, daemon
from workstation_manager.config import REPO


def have_systemd() -> bool:
    return shutil.which("systemd-run") is not None and subprocess.run(
        ["systemctl", "--user", "is-system-running"], capture_output=True).returncode in (0, 1)


@unittest.skipUnless(have_systemd(), "needs a systemd user manager")
class TestLive(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.conf = d / "config.toml"
        self.conf.write_text(f'state_dir = "{d / "state"}"\npage_dirs = []\nmem_margin_mib = 1024\n')
        self.env = dict(os.environ, WORKSTATION_MANAGER_CONFIG=str(self.conf))
        # The pytest lease this suite runs under sets the thread counts for its own core.
        for k in ("WS_JOB_ID", "WS_LEASE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "MKL_NUM_THREADS"):
            self.env.pop(k, None)
        self.cfg = config.load(self.conf)
        self.patches = [mock.patch.dict(os.environ, WORKSTATION_MANAGER_CONFIG=str(self.conf)),
                        mock.patch.object(daemon.interactive, "state", return_value="up"),
                        mock.patch.object(daemon.interactive, "loaded_model", return_value="default"),
                        mock.patch.object(daemon, "read_card", return_value=("card", 32624, 12000, 0))]
        for p in self.patches:
            p.start()
        self.d = daemon.Daemon(self.cfg)
        # The machine's real load is not this test's business: every core reads idle.
        self.d.meter.sample = lambda now=None: [0.0] * len(self.d.cores)
        self.stop = threading.Event()
        self.ticker = threading.Thread(target=self.tick_loop, daemon=True)
        self.ticker.start()

    def tick_loop(self):
        while not self.stop.is_set():
            self.d.tick()
            time.sleep(0.3)

    def tearDown(self):
        self.stop.set()
        self.ticker.join()
        for p in self.patches:
            p.stop()
        for j in self.d.store.jobs():
            subprocess.run(["systemctl", "--user", "stop", j.unit], capture_output=True)
        self.tmp.cleanup()

    def ws(self, *args, **kw):
        return subprocess.run([sys.executable, str(REPO / "ws"), *args], env=self.env,
                              capture_output=True, text=True, cwd=self.tmp.name, **kw)

    def wait_job(self, jid, timeout=60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            j = self.d.store.get(jid)
            if j.status in ("done", "failed"):
                return j
            time.sleep(0.3)
        self.fail(f"job {jid} still {self.d.store.get(jid).status}")

    def test_job_runs_pinned_to_its_cores(self):
        r = self.ws("submit", "--cores", "2", "--mem", "256M", "--name", "pin", "--",
                    sys.executable, "-c",
                    "import os; print('cpus', sorted(os.sched_getaffinity(0)), os.environ['OMP_NUM_THREADS'])")
        self.assertEqual(r.returncode, 0, r.stderr)
        j = self.wait_job(1)
        self.assertEqual(j.status, "done")
        log = (self.cfg.job_dir(1) / "log").read_text()
        self.assertIn(f"cpus {sorted(j.cpus)} 2", log)
        self.assertEqual(len(j.cpus), 4)  # two physical cores, both siblings each

    def test_legacy_job_keeps_the_old_names_and_runs_unpinned(self):
        r = subprocess.run([sys.executable, str(REPO / "gpu"), "submit", "--vram", "1G", "--name", "old",
                            "--", sys.executable, "-c",
                            "import os; print('env', os.environ['GPU_JOB_ID'], os.environ['GPU_JOB_LABEL'],"
                            " len(os.sched_getaffinity(0)) == os.cpu_count())"],
                           env=self.env, capture_output=True, text=True, cwd=self.tmp.name)
        self.assertEqual(r.returncode, 0, r.stderr)
        j = self.wait_job(1)
        self.assertEqual((j.status, j.unit), ("done", "gpu-job-1"))
        self.assertIn("env 1 gpu-manager.job=1 True", (self.cfg.job_dir(1) / "log").read_text())

    def test_job_past_its_memory_is_killed_and_not_retried(self):
        r = self.ws("submit", "--mem", "200M", "--name", "hog", "--",
                    sys.executable, "-c", "x = bytearray(600 * 1024 * 1024); x[::4096] = b'1' * len(x[::4096])")
        self.assertEqual(r.returncode, 0, r.stderr)
        j = self.wait_job(1)
        self.assertEqual((j.status, j.failures), ("failed", 1))
        events = [e["what"] for e in self.d.store.events(0)]
        self.assertTrue(any("memory limit" in e for e in events), events)

    def test_pytest_run_waits_for_and_holds_a_lease(self):
        t = Path(self.tmp.name) / "test_probe.py"
        t.write_text("import os\n"
                     "def test_pinned():\n"
                     "    assert os.environ.get('WS_LEASE')\n"
                     "    assert len(os.sched_getaffinity(0)) == 2, os.sched_getaffinity(0)\n")
        env = dict(self.env, PYTHONPATH=str(REPO / "pytest_hook"), PYTEST_PLUGINS="ws_pytest")
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(t)],
                           env=env, capture_output=True, text=True, cwd=self.tmp.name, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        leases = self.d.store.leases(("done",))
        self.assertEqual(len(leases), 1)
        self.assertEqual(leases[0].kind, "pytest")
        self.assertIsNotNone(leases[0].granted_at)

    @unittest.skipUnless(shutil.which("docker") and subprocess.run(
        ["docker", "image", "inspect", "alpine:latest"], capture_output=True).returncode == 0,
        "needs docker and alpine:latest")
    def test_container_killed_at_its_memory_cap_is_not_retried(self):
        sh = Path(self.tmp.name) / "hog.sh"
        sh.write_text('#!/bin/sh\nexec docker run --rm --label "$WS_JOB_LABEL" --cpuset-cpus "$WS_JOB_CPUS" '
                      '--memory "${WS_JOB_MEM_MIB}m" --memory-swap "${WS_JOB_MEM_MIB}m" '
                      'alpine:latest sh -c "sleep 8; tail /dev/zero"\n')
        sh.chmod(0o755)
        r = self.ws("submit", "--mem", "100M", "--name", "chog", "--", str(sh))
        self.assertEqual(r.returncode, 0, r.stderr)
        j = self.wait_job(1, timeout=90)
        self.assertEqual((j.status, j.failures), ("failed", 1))
        events = [e["what"] for e in self.d.store.events(0)]
        self.assertTrue(any("memory limit" in e for e in events), events)

    def test_ws_run_holds_a_lease_and_caps_memory(self):
        r = self.ws("run", "--cores", "1", "--mem", "200M", "--", sys.executable, "-c",
                    "import os; print(len(os.sched_getaffinity(0)), os.environ['WS_LEASE'])", timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.split()[0], "2")
        r = self.ws("run", "--mem", "200M", "--", sys.executable, "-c",
                    "x = bytearray(600 * 1024 * 1024); x[::4096] = b'1' * len(x[::4096])", timeout=60)
        self.assertEqual(r.returncode, 137, r.stderr)
        self.assertIn("raise --mem", r.stderr)


if __name__ == "__main__":
    unittest.main()
