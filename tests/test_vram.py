
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


class TestVramFree(unittest.TestCase):
    def test_normal(self):
        self.assertEqual(_vram_free(8*1024**3, 1*1024**3), 7*1024**3)
    def test_zero_used(self):
        self.assertEqual(_vram_free(8*1024**3, 0), 8*1024**3)
    def test_equal(self):
        self.assertEqual(_vram_free(1024, 1024), 0)
    def test_none_total(self):
        self.assertIsNone(_vram_free(None, 1024))
    def test_none_used(self):
        self.assertIsNone(_vram_free(1024, None))
    def test_used_exceeds_total(self):
        self.assertIsNone(_vram_free(1024, 2048))
    def test_negative_used(self):
        self.assertIsNone(_vram_free(1024, -1))
    def test_negative_total(self):
        self.assertIsNone(_vram_free(-1, 0))


class TestVramCollect(unittest.TestCase):

    def test_zero_usage_is_measured_not_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            drm = os.path.join(tmp, "drm")
            amd_card(drm, "card0", "0000:03:00.0", vram_used="0", gpu_busy="0")
            d = run(drm).devices[0]
            self.assertEqual(d.vram_used_bytes, 0)
            self.assertEqual(d.gpu_busy_percent, 0)
            self.assertEqual(d.vram_free_bytes, d.vram_total_bytes)

    def test_invalid_vram_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            drm = os.path.join(tmp, "drm")
            amd_card(drm, "card0", "0000:03:00.0",
                     vram_total="not_a_number", vram_used="bad")
            d = run(drm).devices[0]
            self.assertIsNone(d.vram_total_bytes)
            self.assertIsNone(d.vram_used_bytes)
            self.assertIsNone(d.vram_free_bytes)
            self.assertTrue(any("invalid" in n for n in d.notes))

    def test_missing_vram_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            drm = os.path.join(tmp, "drm")
            amd_card(drm, "card0", "0000:03:00.0", skip_vram=True)
            d = run(drm).devices[0]
            self.assertIsNone(d.vram_total_bytes)
            self.assertIsNone(d.vram_used_bytes)
            self.assertIsNone(d.vram_free_bytes)
            self.assertTrue(any("vram_total: unavailable" in n for n in d.notes))

    def test_negative_total_becomes_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            drm = os.path.join(tmp, "drm")
            amd_card(drm, "card0", "0000:03:00.0",
                     vram_total="-1", vram_used="0")
            d = run(drm).devices[0]
            self.assertIsNone(d.vram_total_bytes)
            self.assertTrue(any("vram_total" in n for n in d.notes))

    def test_negative_used_becomes_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            drm = os.path.join(tmp, "drm")
            amd_card(drm, "card0", "0000:03:00.0",
                     vram_total="4096", vram_used="-1")
            d = run(drm).devices[0]
            self.assertIsNone(d.vram_used_bytes)
            self.assertTrue(any("vram_used" in n for n in d.notes))

    def test_used_greater_than_total_free_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            drm = os.path.join(tmp, "drm")
            amd_card(drm, "card0", "0000:03:00.0",
                     vram_total="1024", vram_used="2048")
            d = run(drm).devices[0]
            self.assertEqual(d.vram_total_bytes, 1024)
            self.assertEqual(d.vram_used_bytes, 2048)
            self.assertIsNone(d.vram_free_bytes)

    def test_valid_free_computed(self):
        with tempfile.TemporaryDirectory() as tmp:
            drm = os.path.join(tmp, "drm")
            amd_card(drm, "card0", "0000:03:00.0",
                     vram_total="8192", vram_used="2048")
            d = run(drm).devices[0]
            self.assertEqual(d.vram_free_bytes, 6144)


if __name__ == "__main__":
    unittest.main()
