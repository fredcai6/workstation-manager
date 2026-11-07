
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


class TestBusy(unittest.TestCase):

    def _d(self, gpu_busy, skip_busy=False):
        with tempfile.TemporaryDirectory() as tmp:
            drm = os.path.join(tmp, "drm")
            amd_card(drm, "card0", "0000:03:00.0",
                     gpu_busy=gpu_busy, skip_busy=skip_busy)
            return run(drm).devices[0]

    def test_busy_zero(self):
        d = self._d("0")
        self.assertEqual(d.gpu_busy_percent, 0)

    def test_busy_100(self):
        d = self._d("100")
        self.assertEqual(d.gpu_busy_percent, 100)

    def test_busy_mid(self):
        d = self._d("57")
        self.assertEqual(d.gpu_busy_percent, 57)

    def test_busy_over_100_is_none(self):
        d = self._d("101")
        self.assertIsNone(d.gpu_busy_percent)
        self.assertTrue(any("invalid" in n for n in d.notes))

    def test_busy_negative_is_none(self):
        d = self._d("-1")
        self.assertIsNone(d.gpu_busy_percent)
        self.assertTrue(any("invalid" in n for n in d.notes))

    def test_busy_missing_file(self):
        d = self._d(None, skip_busy=True)
        self.assertIsNone(d.gpu_busy_percent)
        self.assertTrue(any("gpu_busy_percent: unavailable" in n for n in d.notes))

    def test_busy_unparseable(self):
        d = self._d("not_a_number")
        self.assertIsNone(d.gpu_busy_percent)
        self.assertTrue(any("invalid" in n for n in d.notes))

    def test_busy_permission_denied(self):
        with tempfile.TemporaryDirectory() as tmp:
            drm = os.path.join(tmp, "drm")
            amd_card(drm, "card0", "0000:03:00.0", gpu_busy="50")
            busy_path = os.path.join(drm, "card0", "device", "gpu_busy_percent")
            os.chmod(busy_path, 0o000)
            try:
                d = run(drm).devices[0]
                self.assertIsNone(d.gpu_busy_percent)
                self.assertTrue(any("gpu_busy_percent" in n for n in d.notes))
            finally:
                os.chmod(busy_path, 0o644)


if __name__ == "__main__":
    unittest.main()
