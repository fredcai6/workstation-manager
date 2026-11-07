import time
import unittest

from workstation_manager.config import Config


def at(hh, mm):
    t = list(time.localtime())
    t[3], t[4], t[5] = hh, mm, 0
    return time.mktime(tuple(t))


class TestReserveWindows(unittest.TestCase):
    def test_inside_and_outside_a_morning_window(self):
        c = Config(reserve_windows=("07:30-08:00",))
        self.assertTrue(c.reserved(at(7, 30)))
        self.assertTrue(c.reserved(at(7, 59)))
        self.assertFalse(c.reserved(at(8, 0)))
        self.assertFalse(c.reserved(at(7, 29)))

    def test_window_across_midnight(self):
        c = Config(reserve_windows=("23:30-00:30",))
        self.assertTrue(c.reserved(at(23, 45)))
        self.assertTrue(c.reserved(at(0, 15)))
        self.assertFalse(c.reserved(at(1, 0)))

    def test_no_windows(self):
        self.assertFalse(Config().reserved(at(7, 45)))


if __name__ == "__main__":
    unittest.main()
