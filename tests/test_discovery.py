
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


class TestDiscovery(unittest.TestCase):

    def test_pci_id_from_uevent(self):
        with tempfile.TemporaryDirectory() as tmp:
            drm = os.path.join(tmp, "drm")
            amd_card(drm, "card0", "0000:03:00.0")
            self.assertEqual(_pci_id_for_entry(os.path.join(drm, "card0")),
                             "0000:03:00.0")

    def test_pci_id_unknown_when_no_uevent(self):
        with tempfile.TemporaryDirectory() as tmp:
            card = os.path.join(tmp, "card0")
            os.makedirs(os.path.join(card, "device"))
            self.assertEqual(_pci_id_for_entry(card), "unknown")

    def test_is_amd_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            drm = os.path.join(tmp, "drm")
            amd_card(drm, "card0", "0000:03:00.0")
            self.assertTrue(_is_amd(os.path.join(drm, "card0")))

    def test_is_amd_false_nvidia(self):
        with tempfile.TemporaryDirectory() as tmp:
            card = os.path.join(tmp, "card0")
            dev = os.path.join(card, "device")
            os.makedirs(dev)
            _write(os.path.join(dev, "vendor"), "0x10de\n")
            self.assertFalse(_is_amd(card))

    def test_single_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            drm = os.path.join(tmp, "drm")
            amd_card(drm, "card0", "0000:03:00.0")
            r = run(drm)
            self.assertEqual(len(r.devices), 1)
            self.assertEqual(r.devices[0].pci_id, "0000:03:00.0")

    def test_two_distinct_devices(self):
        with tempfile.TemporaryDirectory() as tmp:
            drm = os.path.join(tmp, "drm")
            amd_card(drm, "card0", "0000:03:00.0", product="GPU A")
            amd_card(drm, "card1", "0000:04:00.0", product="GPU B")
            r = run(drm)
            self.assertEqual(len(r.devices), 2)
            self.assertEqual({d.pci_id for d in r.devices},
                             {"0000:03:00.0", "0000:04:00.0"})

    def test_alias_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            drm = os.path.join(tmp, "drm")
            amd_card(drm, "card0", "0000:03:00.0")
            amd_card(drm, "card1", "0000:03:00.0")   # same PCI id = alias
            r = run(drm)
            self.assertEqual(len(r.devices), 1)

    def test_unknown_pci_not_merged(self):
        """Two cards both resolving to unknown PCI are kept separate."""
        with tempfile.TemporaryDirectory() as tmp:
            drm = os.path.join(tmp, "drm")
            for name in ("card0", "card1"):
                card = os.path.join(drm, name)
                dev  = os.path.join(card, "device")
                os.makedirs(dev)
                _write(os.path.join(dev, "vendor"), "0x1002\n")
                # No uevent -> pci_id == "unknown"
                _write(os.path.join(dev, "mem_info_vram_total"), "4096\n")
                _write(os.path.join(dev, "mem_info_vram_used"),  "0\n")
                _write(os.path.join(dev, "gpu_busy_percent"),    "0\n")
            r = run(drm)
            self.assertEqual(len(r.devices), 2)

    def test_non_amd_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            drm = os.path.join(tmp, "drm")
            nvidia = os.path.join(drm, "card0", "device")
            os.makedirs(nvidia)
            _write(os.path.join(nvidia, "vendor"), "0x10de\n")
            r = run(drm)
            self.assertEqual(len(r.devices), 0)

    def test_missing_drm_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = run(os.path.join(tmp, "nope"))
            self.assertEqual(r.devices, [])
            self.assertTrue(any("unavailable" in n for n in r.notes))


if __name__ == "__main__":
    unittest.main()


class TestEmptyVsUnavailableDiscovery(unittest.TestCase):
    """Empty dir (exists) vs non-existent dir must produce different notes."""

    def test_empty_dir_no_unavailable_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            drm = os.path.join(tmp, "drm")
            os.makedirs(drm)
            r = run(drm)
            self.assertEqual(r.devices, [])
            self.assertFalse(r.budget_exhausted)
            # An empty real directory is not an error — no "unavailable" note.
            self.assertFalse(any("unavailable" in n for n in r.notes))

    def test_nonexistent_dir_produces_unavailable_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = run(os.path.join(tmp, "nope"))
            self.assertEqual(r.devices, [])
            self.assertTrue(any("unavailable" in n for n in r.notes))

    def test_empty_and_missing_notes_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            drm = os.path.join(tmp, "drm")
            os.makedirs(drm)
            r_empty = run(drm)
            r_missing = run(os.path.join(tmp, "nope"))
            self.assertNotEqual(list(r_empty.notes), list(r_missing.notes))
