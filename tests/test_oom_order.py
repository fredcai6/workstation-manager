import os
import subprocess
import unittest

from workstation_manager import cli


class TestWsRunOomScore(unittest.TestCase):
    def test_ws_run_child_is_first_to_be_killed(self):
        out = subprocess.run(["cat", "/proc/self/oom_score_adj"], preexec_fn=cli._prefer_oom_kill,
                             capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(out, "800")
        self.assertNotEqual(open(f"/proc/{os.getpid()}/oom_score_adj").read().strip(), "800")


if __name__ == "__main__":
    unittest.main()
