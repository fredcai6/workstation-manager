"""Tests for workstation_manager.renderer — no sysfs, hand-built snapshots."""
import json
import sys
import unittest


from workstation_manager.snapshot import CollectionResult, DeviceSnapshot, ProcessInfo
from workstation_manager.renderer import render_text, render_json


def _dev(pci="0000:03:00.0", vram_total=8*1024**3, vram_used=1*1024**3,
         vram_free=7*1024**3, busy=42, processes=None, notes=None):
    return DeviceSnapshot(
        path="/sys/class/drm/card0",
        label="Radeon RX 6800",
        pci_id=pci,
        vram_total_bytes=vram_total,
        vram_used_bytes=vram_used,
        vram_free_bytes=vram_free,
        gpu_busy_percent=busy,
        processes=processes or [],
        notes=notes or [],
    )


def _result(devices, exhausted=False, notes=None):
    return CollectionResult(
        devices=devices,
        budget_exhausted=exhausted,
        notes=notes or [],
    )


class TestRenderText(unittest.TestCase):

    def test_zero_values_shown_not_unavailable(self):
        d = _dev(vram_total=8*1024**3, vram_used=0, vram_free=8*1024**3, busy=0)
        out = render_text(_result([d]))
        self.assertIn("0.0 MiB", out)   # vram_used=0 → "0.0 MiB", not "unavailable"
        self.assertIn("0%", out)         # busy=0 → "0%", not "unavailable"
        self.assertNotIn("unavailable", out)

    def test_none_values_shown_as_unavailable(self):
        d = _dev(vram_total=None, vram_used=None, vram_free=None, busy=None)
        out = render_text(_result([d]))
        self.assertIn("unavailable", out)

    def test_budget_exhausted_shows_warning(self):
        d = _dev()
        out = render_text(_result([d], exhausted=True))
        self.assertIn("WARNING", out)

    def test_no_warning_when_not_exhausted(self):
        d = _dev()
        out = render_text(_result([d], exhausted=False))
        self.assertNotIn("WARNING", out)

    def test_process_shown(self):
        p = ProcessInfo(pid=1234, comm="myapp")
        d = _dev(processes=[p])
        out = render_text(_result([d]))
        self.assertIn("1234", out)
        self.assertIn("myapp", out)

    def test_process_comm_none_shown_explicitly(self):
        p = ProcessInfo(pid=5678, comm=None)
        d = _dev(processes=[p])
        out = render_text(_result([d]))
        self.assertIn("5678", out)
        self.assertIn("unavailable", out)

    def test_no_devices_message(self):
        out = render_text(_result([]))
        self.assertIn("No AMD GPU", out)

    def test_device_note_surfaced(self):
        d = _dev(notes=["vram_total: unavailable"])
        out = render_text(_result([d]))
        self.assertIn("vram_total: unavailable", out)


class TestRenderJson(unittest.TestCase):

    def _parse(self, result):
        return json.loads(render_json(result))

    def test_none_fields_serialise_as_null(self):
        d = _dev(vram_total=None, vram_used=None, vram_free=None, busy=None)
        obj = self._parse(_result([d]))
        dev = obj["devices"][0]
        self.assertIsNone(dev["vram_total_bytes"])
        self.assertIsNone(dev["vram_used_bytes"])
        self.assertIsNone(dev["vram_free_bytes"])
        self.assertIsNone(dev["gpu_busy_percent"])

    def test_zero_values_are_zero_not_null(self):
        d = _dev(vram_total=8*1024**3, vram_used=0, vram_free=8*1024**3, busy=0)
        obj = self._parse(_result([d]))
        dev = obj["devices"][0]
        self.assertEqual(dev["vram_used_bytes"], 0)
        self.assertEqual(dev["gpu_busy_percent"], 0)

    def test_budget_exhausted_in_json(self):
        obj = self._parse(_result([_dev()], exhausted=True))
        self.assertTrue(obj["budget_exhausted"])

    def test_process_comm_none_is_null(self):
        p = ProcessInfo(pid=1234, comm=None)
        d = _dev(processes=[p])
        obj = self._parse(_result([d]))
        proc = obj["devices"][0]["processes"][0]
        self.assertEqual(proc["pid"], 1234)
        self.assertIsNone(proc["comm"])

    def test_valid_json_structure(self):
        d = _dev()
        obj = self._parse(_result([d]))
        self.assertIn("devices", obj)
        self.assertIn("budget_exhausted", obj)
        self.assertIn("notes", obj)
        dev = obj["devices"][0]
        for key in ("path", "label", "pci_id", "vram_total_bytes",
                    "vram_used_bytes", "vram_free_bytes",
                    "gpu_busy_percent", "processes", "notes"):
            self.assertIn(key, dev)


class TestCLIBudgetValidation(unittest.TestCase):
    """CLI must reject non-finite / non-positive budget values."""

    def _cli(self, args):
        from workstation_manager.cli import main
        return main(args)

    def test_budget_inf_rejected(self):
        import io
        import sys
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            rc = self._cli(["status", "--budget", "inf"])
            err = sys.stderr.getvalue()
        finally:
            sys.stderr = old_stderr
        self.assertEqual(rc, 2, "exit code must be 2 for invalid budget")
        self.assertIn("error", err.lower())

    def test_budget_negative_rejected(self):
        import io
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            rc = self._cli(["status", "--budget", "-1"])
            err = sys.stderr.getvalue()
        finally:
            sys.stderr = old_stderr
        self.assertEqual(rc, 2)
        self.assertIn("error", err.lower())

    def test_budget_zero_rejected(self):
        import io
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            rc = self._cli(["status", "--budget", "0"])
            err = sys.stderr.getvalue()
        finally:
            sys.stderr = old_stderr
        self.assertEqual(rc, 2)
        self.assertIn("error", err.lower())


if __name__ == "__main__":
    unittest.main()
