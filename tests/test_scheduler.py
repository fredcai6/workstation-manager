import unittest

from workstation_manager.scheduler import Svc, View, decide
from workstation_manager.store import Job

TOTAL = 32624
PAIR = 30000
MARGIN = 1024


def job(id, vram, status="queued", submitted=0.0, started=None, priority=0):
    return Job(priority=priority, id=id, name=f"j{id}", owner=None, vram_mib=vram, argv=["x"], cwd="/",
               retries=2, grace_seconds=60, status=status, submitted_at=submitted,
               started_at=started, ended_at=None, failures=0, interruptions=0,
               exit_code=None, stop_request=None, chunk=None, chunks=None, pct=None,
               progress_at=None, note=None)


def view(queued=(), active=(), pair="up", idle_min=0.0, used=None, **kw):
    if used is None:
        used = (29391 if pair in ("up", "starting", "stopping") else 0) + sum(j.vram_mib for j in active)
    return View(now=10_000.0, queued=list(queued), active=list(active), pair=pair,
                last_interactive_at=10_000.0 - idle_min * 60, vram_total_mib=TOTAL,
                vram_used_mib=used, **kw)


def run(v, preempt=False):
    return decide(v, MARGIN, PAIR, 30, preempt)


class TestScheduler(unittest.TestCase):
    def test_idle_card_with_empty_queue_brings_pair_up(self):
        self.assertEqual(run(view(pair="down")), [("start_pair", "default")])

    def test_nothing_to_do_when_pair_up_and_queue_empty(self):
        self.assertEqual(run(view()), [])

    def test_big_job_waits_out_the_hold(self):
        self.assertEqual(run(view(queued=[job(1, 18000)], idle_min=10)), [])

    def test_big_job_takes_card_after_hold(self):
        self.assertEqual(run(view(queued=[job(1, 18000)], idle_min=31)), [("stop_pair",)])

    def test_job_starts_once_pair_is_down(self):
        self.assertEqual(run(view(queued=[job(1, 18000)], pair="down")), [("start_job", 1)])

    def test_job_waits_while_pair_stopping(self):
        self.assertEqual(run(view(queued=[job(1, 18000)], pair="stopping", idle_min=60)), [])

    def test_small_job_runs_beside_pair(self):
        # 32624 - 1024 - 30000 = 1600 MiB of room beside the pair
        self.assertEqual(run(view(queued=[job(1, 1500)])), [("start_job", 1)])

    def test_measured_usage_blocks_even_if_declared_fits(self):
        self.assertEqual(run(view(queued=[job(1, 1500)], used=31000)), [])

    def test_fifo_two_jobs_fill_card(self):
        acts = run(view(queued=[job(1, 15000), job(2, 15000), job(3, 4000)], pair="down"))
        self.assertEqual(acts, [("start_job", 1), ("start_job", 2)])

    def test_backfill_cannot_starve_head(self):
        # Head (#1, 31000) waits for the hold; #2 fits beside the pair but would
        # leave the head no room once the pair goes, so it does not jump ahead.
        acts = run(view(queued=[job(1, 31000), job(2, 1500)], idle_min=5))
        self.assertEqual(acts, [])

    def test_backfill_allowed_when_head_still_fits(self):
        acts = run(view(queued=[job(1, 20000), job(2, 1500)], idle_min=5))
        self.assertEqual(acts, [("start_job", 2)])

    def test_pair_stays_down_while_job_waits_on_running_batch(self):
        running = job(9, 20000, status="running", started=1.0)
        acts = run(view(queued=[job(1, 20000)], active=[running], pair="down"))
        self.assertEqual(acts, [])

    def test_pair_returns_when_queue_drains_and_room(self):
        running = job(9, 1000, status="running", started=1.0)
        self.assertEqual(run(view(active=[running], pair="down")), [("start_pair", "default")])

    def test_pair_stays_down_beside_big_running_job(self):
        running = job(9, 18000, status="running", started=1.0)
        self.assertEqual(run(view(active=[running], pair="down")), [])

    def test_interactive_demand_without_preemption_changes_nothing(self):
        running = job(9, 18000, status="running", started=1.0)
        acts = run(view(active=[running], pair="down", demand_pending=True))
        self.assertEqual(acts, [])

    def test_interactive_demand_with_preemption_evicts_newest_first(self):
        old = job(8, 10000, status="running", started=1.0)
        new = job(9, 10000, status="running", started=2.0)
        acts = run(view(active=[old, new], pair="down", demand_pending=True), preempt=True)
        self.assertEqual(acts, [("stop_job", 9, "requeue"), ("stop_job", 8, "requeue")])

    def test_take_stops_everything(self):
        running = job(9, 1000, status="running", started=1.0)
        acts = run(view(queued=[job(1, 500)], active=[running], take=True))
        self.assertEqual(acts, [("stop_job", 9, "requeue"), ("stop_pair",)])

    # -- model switching (Dory 12.5 GB resident, Qwen 26.5 GB on request) ------------
    MODELS = {"dory": 12500, "qwen": 26500}

    def mv(self, **kw):
        kw.setdefault("used", None)
        loaded = kw.pop("loaded", "dory")
        want = kw.pop("want", "dory")
        services = kw.pop("services", ())
        reserved = kw.pop("reserved", False)
        v = view(**kw)
        v.loaded, v.want, v.services, v.reserved = loaded, want, services, reserved
        if kw["used"] is None:
            v.vram_used_mib = (self.MODELS[loaded] if v.pair != "down" else 0) + sum(j.vram_mib for j in v.active)
        return decide(v, MARGIN, self.MODELS, 30, False)

    def test_request_for_other_model_swaps_when_it_fits(self):
        self.assertEqual(self.mv(want="qwen"), [("swap_model", "qwen")])

    def test_swap_waits_for_batch_and_blocks_jobs_that_would_starve_it(self):
        running = job(9, 10000, status="running", started=1.0)
        # 31600 cap - 10000 batch < 26500 qwen: wait; a 2 GB job would make it worse
        self.assertEqual(self.mv(want="qwen", active=[running], queued=[job(1, 2000)]), [])

    def test_job_that_leaves_room_for_the_swap_still_starts(self):
        self.assertEqual(self.mv(want="qwen", queued=[job(1, 4000)]),
                         [("swap_model", "qwen"), ("start_job", 1)])

    def test_back_to_default(self):
        self.assertEqual(self.mv(loaded="qwen", want="dory"), [("swap_model", "dory")])

    def test_pair_returns_with_wanted_model(self):
        self.assertEqual(self.mv(pair="down", want="qwen"), [("start_pair", "qwen")])

    def test_unknown_loaded_model_is_left_alone(self):
        v = view()
        v.loaded, v.want = None, "dory"
        self.assertEqual(decide(v, MARGIN, self.MODELS, 30, False), [])

    # -- on-demand services (an 11.5 GB OCR server) -----------------------------------
    def sv(self, state="down", **kw):
        return Svc("ocr", state, 11500, **kw)

    def test_waiting_service_starts_beside_default_model(self):
        self.assertEqual(self.mv(services=[self.sv(waiting=1)]), [("start_service", "ocr")])

    def test_waiting_service_starts_ahead_of_queued_batch(self):
        acts = self.mv(services=[self.sv(waiting=1)], queued=[job(1, 10000)])
        self.assertEqual(acts, [("start_service", "ocr")])

    def test_batch_that_would_keep_a_waiting_service_out_does_not_start(self):
        running = job(9, 9000, status="running", started=1.0)
        # 31600 - 9000 - 12500 = 10100 free: ocr waits; a 2 GB job would make it worse
        acts = self.mv(services=[self.sv(waiting=1)], active=[running], queued=[job(1, 2000)])
        self.assertEqual(acts, [])

    def test_service_waits_while_big_model_loaded(self):
        self.assertEqual(self.mv(loaded="qwen", want="qwen", services=[self.sv(waiting=1)]), [])

    def test_idle_service_stops_after_its_limit(self):
        s = self.sv("up", idle_seconds=130, idle_limit=120)
        self.assertEqual(self.mv(services=[s]), [("stop_service", "ocr")])

    def test_recently_used_service_stays(self):
        self.assertEqual(self.mv(services=[self.sv("up", idle_seconds=30)]), [])

    def test_idle_service_gives_way_to_a_model_swap(self):
        s = self.sv("up", idle_seconds=15)
        v = self.mv(want="qwen", services=[s], used=12500 + 11500)
        self.assertEqual(v, [("stop_service", "ocr")])

    def test_service_between_back_to_back_requests_keeps_its_room(self):
        s = self.sv("up", idle_seconds=2)
        self.assertEqual(self.mv(want="qwen", services=[s], used=12500 + 11500), [])

    def test_busy_service_is_not_stopped_for_a_swap(self):
        s = self.sv("up", in_flight=1, idle_seconds=0)
        self.assertEqual(self.mv(want="qwen", services=[s], used=12500 + 11500), [])

    # -- reserve window -------------------------------------------------------------
    def test_window_keeps_the_pair_up_past_the_hold(self):
        self.assertEqual(self.mv(queued=[job(1, 25000)], idle_min=90, reserved=True), [])

    def test_window_brings_the_pair_up_even_with_jobs_queued(self):
        self.assertEqual(self.mv(pair="down", queued=[job(1, 25000)], reserved=True),
                         [("start_pair", "dory")])

    def test_window_lets_a_job_start_that_fits_beside_the_pair(self):
        acts = self.mv(pair="down", queued=[job(1, 10000)], reserved=True)
        self.assertEqual(acts, [("start_job", 1), ("start_pair", "dory")])

    def test_window_leaves_a_running_big_job_alone(self):
        running = job(9, 25000, status="running", started=1.0)
        self.assertEqual(self.mv(pair="down", active=[running], reserved=True), [])

    def test_unreadable_card_starts_nothing(self):
        v = view(queued=[job(1, 500)], pair="down")
        v.vram_used_mib = None
        self.assertEqual(run(v), [])


if __name__ == "__main__":
    unittest.main()


class TestPriority(unittest.TestCase):
    def test_priority_job_stops_lower_running_job_for_the_card(self):
        low = job(1, 18000, status="running", started=1.0)
        hot = job(2, 30000, priority=10)
        self.assertEqual(run(view(queued=[hot], active=[low], pair="down")),
                         [("stop_job", 1, "requeue")])

    def test_stops_newest_first_and_only_as_many_as_needed(self):
        a = job(1, 10000, status="running", started=1.0)
        b = job(2, 10000, status="running", started=2.0)
        hot = job(3, 18000, priority=10)
        self.assertEqual(run(view(queued=[hot], active=[a, b], pair="down")),
                         [("stop_job", 2, "requeue")])

    def test_stopping_jobs_count_as_gone(self):
        low = job(1, 18000, status="stopping", started=1.0)
        hot = job(2, 30000, priority=10)
        self.assertEqual(run(view(queued=[hot], active=[low], pair="down")), [])

    def test_equal_priority_is_not_stopped(self):
        low = job(1, 18000, status="running", started=1.0)
        self.assertEqual(run(view(queued=[job(2, 30000)], active=[low], pair="down")), [])

    def test_higher_running_job_is_left_alone(self):
        top = job(1, 18000, status="running", started=1.0, priority=20)
        hot = job(2, 30000, priority=10)
        self.assertEqual(run(view(queued=[hot], active=[top], pair="down")), [])

    def test_not_in_a_reserve_window(self):
        low = job(1, 18000, status="running", started=1.0)
        hot = job(2, 30000, priority=10)
        acts = run(view(queued=[hot], active=[low], pair="down", reserved=True))
        self.assertNotIn(("stop_job", 1, "requeue"), acts)

    def test_next_waiting_job_is_protected_once_the_head_starts(self):
        a, b, c = job(1, 17000, priority=10), job(2, 30000, priority=10), job(3, 13000)
        acts = run(view(queued=[a, b, c], pair="down"))
        self.assertIn(("start_job", 1), acts)
        self.assertNotIn(("start_job", 3), acts)
