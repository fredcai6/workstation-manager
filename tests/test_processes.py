
import os, sys, tempfile, unittest

from workstation_manager.collector import collect, _pci_id_for_entry, _is_amd, _vram_free
from workstation_manager.snapshot import ProcessInfo


def _write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def amd_card(drm, name, pci, *, vram_total="8589934592", vram_used="1073741824",
             gpu_busy="42", product="Radeon RX 6800",
             skip_vram=False, skip_busy=False):
    card = os.path.join(drm, name)
    dev  = os.path.join(card, "device")
    os.makedirs(dev, exist_ok=True)
    _write(os.path.join(dev, "vendor"), "0x1002\n")
    _write(os.path.join(dev, "uevent"),
           f"PCI_SLOT_NAME={pci}\nPCI_ID=1002:687F\n")
    _write(os.path.join(dev, "product_name"), product + "\n")
    if not skip_vram:
        if vram_total is not None:
            _write(os.path.join(dev, "mem_info_vram_total"), vram_total + "\n")
        if vram_used is not None:
            _write(os.path.join(dev, "mem_info_vram_used"),  vram_used  + "\n")
    if not skip_busy and gpu_busy is not None:
        _write(os.path.join(dev, "gpu_busy_percent"), gpu_busy + "\n")
    os.makedirs(os.path.join(dev, "drm", "renderD128"), exist_ok=True)
    return card


def run(drm, proc=None, budget=5.0):
    if proc is None:
        proc = tempfile.mkdtemp()
    return collect(drm_root=drm, proc_root=proc, budget_seconds=budget)

import time
from unittest.mock import patch

PCI = "0000:03:00.0"


def fdinfo_for(pci):
    return (
        "pos:\t0\nflags:\t02\n"
        "drm-driver:\tamdgpu\n"
        f"drm-pdev:\t{pci}\n"
        "drm-client-id:\t12345\n"
    )


def _proc(proc_root, pid, comm, fdinfo):
    pid_dir = os.path.join(proc_root, str(pid))
    fdi_dir = os.path.join(pid_dir, "fdinfo")
    os.makedirs(fdi_dir, exist_ok=True)
    _write(os.path.join(pid_dir, "comm"), comm + "\n")
    _write(os.path.join(fdi_dir, "5"), fdinfo)


class TestProcessAttribution(unittest.TestCase):

    def test_fdinfo_match_via_drm_pdev(self):
        with tempfile.TemporaryDirectory() as tmp:
            drm  = os.path.join(tmp, "drm")
            proc = os.path.join(tmp, "proc")
            amd_card(drm, "card0", PCI)
            _proc(proc, 1234, "myapp", fdinfo_for(PCI))
            r = run(drm, proc)
            self.assertEqual(len(r.devices[0].processes), 1)
            self.assertEqual(r.devices[0].processes[0].pid, 1234)
            self.assertEqual(r.devices[0].processes[0].comm, "myapp")

    def test_fdinfo_wrong_device_not_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            drm  = os.path.join(tmp, "drm")
            proc = os.path.join(tmp, "proc")
            amd_card(drm, "card0", PCI)
            _proc(proc, 1234, "other", fdinfo_for("0000:05:00.0"))
            r = run(drm, proc)
            self.assertEqual(len(r.devices[0].processes), 0)

    def test_fdinfo_dir_permission_denied(self):
        """EACCES on fdinfo dir must be reported as incomplete visibility."""
        with tempfile.TemporaryDirectory() as tmp:
            drm  = os.path.join(tmp, "drm")
            proc = os.path.join(tmp, "proc")
            amd_card(drm, "card0", PCI)
            _proc(proc, 1234, "blocked", fdinfo_for(PCI))
            fdi_dir = os.path.join(proc, "1234", "fdinfo")
            from unittest.mock import patch
            real_listdir = os.listdir

            def denied(path):
                if path == fdi_dir:
                    raise PermissionError("fixture denied")
                return real_listdir(path)

            with patch("workstation_manager.collector.os.listdir", side_effect=denied):
                r = run(drm, proc)
            self.assertEqual(len(r.devices[0].processes), 0)
            self.assertTrue(any("permission denied" in n for n in r.devices[0].notes), r.devices[0].notes)

    def test_individual_fdinfo_file_permission_denied(self):
        """EACCES on individual fdinfo file -> perm_denied note recorded."""
        with tempfile.TemporaryDirectory() as tmp:
            drm  = os.path.join(tmp, "drm")
            proc = os.path.join(tmp, "proc")
            amd_card(drm, "card0", PCI)
            _proc(proc, 1234, "myapp", fdinfo_for(PCI))
            fdinfo_file = os.path.join(proc, "1234", "fdinfo", "5")
            os.chmod(fdinfo_file, 0o000)
            try:
                r = run(drm, proc)
                notes = r.devices[0].notes
                self.assertTrue(
                    any("unreadable" in n or "partial" in n for n in notes),
                    f"Expected partial-visibility note, got: {notes}",
                )
            finally:
                os.chmod(fdinfo_file, 0o644)

    def test_vanished_fdinfo_file_skipped(self):
        """ENOENT on individual fdinfo file -> silently ignored."""
        with tempfile.TemporaryDirectory() as tmp:
            drm  = os.path.join(tmp, "drm")
            proc = os.path.join(tmp, "proc")
            amd_card(drm, "card0", PCI)
            # Process dir exists but fdinfo file will disappear before read.
            pid_dir = os.path.join(proc, "1234")
            fdi_dir = os.path.join(pid_dir, "fdinfo")
            os.makedirs(fdi_dir)
            _write(os.path.join(pid_dir, "comm"), "runner\n")
            # Write a stub, then remove it before collect() runs.
            stub = os.path.join(fdi_dir, "5")
            _write(stub, fdinfo_for(PCI))
            os.remove(stub)
            # Should not raise; process list will be empty.
            r = run(drm, proc)
            self.assertEqual(len(r.devices[0].processes), 0)

    def test_comm_unreadable_pid_still_recorded(self):
        """Vanished comm -> comm=None, ProcessInfo still present."""
        with tempfile.TemporaryDirectory() as tmp:
            drm  = os.path.join(tmp, "drm")
            proc = os.path.join(tmp, "proc")
            amd_card(drm, "card0", PCI)
            _proc(proc, 1234, "runner", fdinfo_for(PCI))
            comm_path = os.path.join(proc, "1234", "comm")
            os.remove(comm_path)
            r = run(drm, proc)
            procs = r.devices[0].processes
            self.assertEqual(len(procs), 1)
            self.assertEqual(procs[0].pid, 1234)
            self.assertIsNone(procs[0].comm)

    def test_budget_exhausted_during_process_scan(self):
        """Budget expiry during proc scan -> budget_exhausted=True, VRAM/busy retained."""
        with tempfile.TemporaryDirectory() as tmp:
            drm  = os.path.join(tmp, "drm")
            proc = os.path.join(tmp, "proc")
            amd_card(drm, "card0", PCI,
                     vram_total="8589934592", vram_used="1073741824",
                     gpu_busy="55")
            _proc(proc, 1234, "runner", fdinfo_for(PCI))

            # Use a counter clock that starts just before and then crosses deadline.
            calls = [0]
            base = time.monotonic()

            def fake_clock():
                calls[0] += 1
                # First few calls: before deadline (device discovery + VRAM + busy).
                # After 4 calls: past deadline (inside process scan).
                if calls[0] <= 4:
                    return base
                return base + 10.0   # way past any deadline

            with patch("workstation_manager.collector.time") as mock_time:
                mock_time.monotonic = fake_clock
                r = collect(drm_root=drm, proc_root=proc, budget_seconds=0.001)

            self.assertTrue(r.budget_exhausted)
            self.assertEqual(len(r.devices), 1)
            d = r.devices[0]
            # VRAM and busy must still be present (read before budget expired).
            self.assertIsNotNone(d.vram_total_bytes)
            self.assertIsNotNone(d.gpu_busy_percent)


if __name__ == "__main__":
    unittest.main()


class TestTwoDeviceBudgetExhaustion(unittest.TestCase):
    """With two devices, budget exhaustion during process scan of device 1
    must still produce a DeviceSnapshot with VRAM/busy for device 2."""

    def test_two_devices_both_have_metrics_after_budget_exhaustion(self):
        with tempfile.TemporaryDirectory() as tmp:
            drm  = os.path.join(tmp, "drm")
            proc = os.path.join(tmp, "proc")
            amd_card(drm, "card0", "0000:03:00.0",
                     vram_total="8589934592", vram_used="1073741824",
                     gpu_busy="55")
            amd_card(drm, "card1", "0000:04:00.0",
                     vram_total="4294967296", vram_used="536870912",
                     gpu_busy="10")
            _proc(proc, 1234, "runner", fdinfo_for("0000:03:00.0"))

            # Clock that lets pass-1 (VRAM+busy for both devices) complete,
            # then expires during pass-2 (process scanning for card0).
            pass1_calls = {"n": 0}
            base = time.monotonic()

            def fake_clock():
                pass1_calls["n"] += 1
                # Pass 1 reads: budget.exhausted() checks + _read_vram/_read_busy
                # We allow 8 calls before the deadline fires.
                if pass1_calls["n"] <= 8:
                    return base
                return base + 10.0  # past deadline

            with patch("workstation_manager.collector.time") as mock_time:
                mock_time.monotonic = fake_clock
                r = collect(drm_root=drm, proc_root=proc, budget_seconds=0.001)

            self.assertTrue(r.budget_exhausted)
            self.assertEqual(len(r.devices), 2,
                             "Both devices must appear even when budget exhausted")
            for d in r.devices:
                self.assertIsNotNone(
                    d.vram_total_bytes,
                    f"vram_total_bytes must be set for {d.pci_id}",
                )
                self.assertIsNotNone(
                    d.gpu_busy_percent,
                    f"gpu_busy_percent must be set for {d.pci_id}",
                )
