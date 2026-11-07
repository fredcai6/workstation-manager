"""The broker loop: observe the card, ask the scheduler, apply its actions."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import os

from . import board, interactive, page, proxy, runner, scheduler
from .collector import collect
from .config import Config
from .store import ACTIVE, Store

log = logging.getLogger("workstation-manager")


def pick_card(devices, pci_id):
    """The configured card, or the AMD card with the most VRAM."""
    if pci_id:
        return next((d for d in devices if d.pci_id == pci_id), None)
    return max(devices, key=lambda d: d.vram_total_bytes or 0, default=None)


def read_card(cfg: Config):
    """(label, total MiB, used MiB, busy %) for the compute card; None where unreadable."""
    d = pick_card(collect(budget_seconds=2.0).devices, cfg.pci_id)
    if d is None:
        return None, None, None, None
    mib = lambda b: None if b is None else b // (1024 * 1024)
    return d.label, mib(d.vram_total_bytes), mib(d.vram_used_bytes), d.gpu_busy_percent


class Daemon:
    def __init__(self, cfg: Config, store: Optional[Store] = None, gate: Optional[proxy.Gate] = None):
        self.cfg = cfg
        self.store = store or Store(cfg.db_path)
        self.gate = gate or proxy.Gate(cfg.queue_wait_seconds)
        self.pair = "down"
        self.pair_action: Optional[threading.Thread] = None
        # On-demand services: state, the thread starting/stopping one, when it came up.
        self.svc_state = {name: "down" for name in cfg.services}
        self.svc_action: dict = {}
        self.svc_up_at: dict = {}
        self.page_at = 0.0
        # The hold runs from the owner's last assistant request, across restarts.
        self.gate.last_interactive_at = self.store.get_kv("last_interactive_at", 0.0)

    # -- reconcile --------------------------------------------------------
    def reconcile_jobs(self) -> None:
        for job in self.store.jobs(ACTIVE):
            if runner.is_active(job):
                continue
            res = runner.take_result(self.cfg, job.id)
            now = time.time()
            if res is None:
                # Unit gone with no recorded exit: power loss, OOM-killed wrapper, reboot.
                self.store.update(job.id, status="queued", interruptions=job.interruptions + 1,
                                  stop_request=None)
                self.store.event(job.id, "interrupted without an exit record; requeued")
            elif job.stop_request == "cancel":
                self.store.update(job.id, status="cancelled", ended_at=now,
                                  exit_code=res["exit_code"], stop_request=None)
                self.store.event(job.id, "cancelled")
            elif job.stop_request == "requeue":
                self.store.update(job.id, status="queued", interruptions=job.interruptions + 1,
                                  exit_code=res["exit_code"], stop_request=None)
                self.store.event(job.id, "stopped for the card; requeued")
            elif res["exit_code"] == 0:
                self.store.update(job.id, status="done", ended_at=now, exit_code=0)
                self.store.event(job.id, "done")
            elif res.get("oom"):
                self.store.update(job.id, status="failed", ended_at=now,
                                  failures=job.failures + 1, exit_code=res["exit_code"])
                self.store.event(job.id, f"out of GPU memory at {job.vram_mib} MiB declared; "
                                         "not retried (resubmit with more --vram)")
            elif job.failures < job.retries:
                self.store.update(job.id, status="queued", failures=job.failures + 1,
                                  exit_code=res["exit_code"])
                self.store.event(job.id, f"exit {res['exit_code']}; retry {job.failures + 1}/{job.retries}")
            else:
                self.store.update(job.id, status="failed", ended_at=now,
                                  failures=job.failures + 1, exit_code=res["exit_code"])
                self.store.event(job.id, f"exit {res['exit_code']}; failed")
        for job in self.store.jobs(("queued",)):
            if job.stop_request == "cancel":
                self.store.update(job.id, status="cancelled", ended_at=time.time(), stop_request=None)
                self.store.event(job.id, "cancelled while queued")

    def observe_pair(self) -> None:
        if self.pair_action and self.pair_action.is_alive():
            return  # starting or stopping; the thread owns the state
        self.pair = interactive.state(self.cfg)
        self.gate.set_loaded(interactive.loaded_model(self.cfg))
        up = self.pair == "up"
        self.gate.set_ok(self.cfg.llm_backend, up)
        self.gate.set_ok(self.cfg.asr_backend, up)

    def observe_services(self) -> None:
        for name, svc in self.cfg.services.items():
            t = self.svc_action.get(name)
            if t and t.is_alive():
                continue
            st = interactive.service_state(self.cfg, svc)
            if st == "up" and self.svc_state[name] != "up":
                self.svc_up_at[name] = time.time()
            self.svc_state[name] = st
            self.gate.set_ok(svc.backend, st == "up")

    # -- actions ----------------------------------------------------------
    def _pair_thread(self, fn, label: str) -> None:
        def run():
            try:
                fn()
                self.store.event(None, label)
            except Exception as exc:  # recorded, retried next tick
                self.store.event(None, f"{label} failed: {exc}")
                log.exception(label)
        self.pair_action = threading.Thread(target=run, daemon=True)
        self.pair_action.start()

    def stop_pair(self) -> None:
        self.pair = "stopping"
        self.gate.set_ok(self.cfg.llm_backend, False)
        self.gate.set_ok(self.cfg.asr_backend, False)

        def go():
            left = self.gate.close_and_drain(self.cfg.drain_seconds)
            if left:
                self.store.event(None, f"stopping pair with {left} request(s) still in flight")
            try:
                interactive.stop(self.cfg)
            finally:
                self.gate.reopen()
        self._pair_thread(go, "interactive pair stopped")

    def start_pair(self, model: str) -> None:
        self.pair = "starting"
        self._pair_thread(lambda: interactive.start(self.cfg, model),
                          f"interactive pair started with {model}")

    def swap_model(self, model: str) -> None:
        """Let in-flight requests finish, then recreate the llm with `model`."""
        self.pair = "starting"
        self.gate.set_ok(self.cfg.llm_backend, False)

        def go():
            left = self.gate.close_and_drain(self.cfg.drain_seconds)
            if left:
                self.store.event(None, f"switching llm with {left} request(s) still in flight")
            try:
                interactive.swap(self.cfg, model)
            finally:
                self.gate.reopen()
        self._pair_thread(go, f"llm switched to {model}")

    def _service_thread(self, name: str, fn, label: str) -> None:
        def run():
            try:
                fn()
                self.store.event(None, label)
            except Exception as exc:
                self.store.event(None, f"{label} failed: {exc}")
                log.exception(label)
        t = threading.Thread(target=run, daemon=True)
        self.svc_action[name] = t
        t.start()

    def start_service(self, name: str) -> None:
        svc = self.cfg.services[name]
        self.svc_state[name] = "starting"
        self._service_thread(name, lambda: interactive.start_service(self.cfg, svc),
                             f"service {name} started")

    def stop_service(self, name: str) -> None:
        svc = self.cfg.services[name]
        self.svc_state[name] = "stopping"
        self.gate.set_ok(svc.backend, False)  # later requests wait for the next start
        self._service_thread(name, lambda: interactive.stop_service(self.cfg, svc),
                             f"service {name} stopped")

    def services_view(self, now: float):
        out = []
        for name, svc in self.cfg.services.items():
            last = max(self.gate.used_at.get(svc.backend, 0.0), self.svc_up_at.get(name, 0.0))
            out.append(scheduler.Svc(
                name, self.svc_state[name], svc.vram_mib,
                waiting=self.gate.waiting_on[svc.backend], in_flight=self.gate.busy_on[svc.backend],
                idle_seconds=now - last if last else 0.0, idle_limit=svc.idle_seconds))
        return out

    def wanted_model(self, now: float) -> str:
        """The model asked for by waiting requests, else the one most recently asked
        for by name within the hold, else the default. A service waiting for room
        outranks the hold: the pair goes back to the default to make that room."""
        default = self.cfg.default
        if self.cfg.reserved(now):
            return default  # requests for another model wait for the window to end
        waiting = {m: n for m, n in self.gate.waiting_for.items() if n > 0}
        if waiting:
            if self.gate.loaded in waiting:
                return self.gate.loaded
            return max(waiting, key=waiting.get)
        if any(self.svc_state[n] == "down" and self.gate.waiting_on[s.backend]
               for n, s in self.cfg.services.items()):
            return default
        recent = {m: t for m, t in self.gate.asked_at.items()
                  if now - t < self.cfg.hold_minutes * 60}
        return max(recent, key=recent.get) if recent else default

    def apply(self, actions) -> None:
        for act in actions:
            kind = act[0]
            if kind == "start_job":
                job = self.store.get(act[1])
                try:
                    runner.start(self.cfg, job)
                except Exception as exc:
                    self.store.event(job.id, f"could not start: {exc}")
                    log.exception("start %s", job.id)
                    continue
                self.store.update(job.id, status="running", started_at=time.time(), stop_request=None)
                self.store.event(job.id, "started")
            elif kind == "stop_job":
                job = self.store.get(act[1])
                self.store.update(job.id, status="stopping", stop_request=act[2])
                runner.stop(job)
                self.store.event(job.id, f"stopping ({act[2]})")
            elif kind == "stop_pair":
                self.stop_pair()
            elif kind == "start_pair":
                self.start_pair(act[1])
            elif kind == "swap_model":
                self.swap_model(act[1])
            elif kind == "start_service":
                self.start_service(act[1])
            elif kind == "stop_service":
                self.stop_service(act[1])

    # -- loop -------------------------------------------------------------
    def tick(self) -> None:
        self.reconcile_jobs()
        self.observe_pair()
        self.observe_services()
        # Cancels of running jobs go out before scheduling.
        for job in self.store.jobs(("running",)):
            if job.stop_request == "cancel":
                self.store.update(job.id, status="stopping")
                runner.stop(job)
        card, total, used, busy = read_card(self.cfg)
        now = time.time()
        want = self.wanted_model(now)
        self.store.set_kv("last_interactive_at", self.gate.last_interactive_at)
        view = scheduler.View(
            now=time.time(),
            queued=[j for j in self.store.jobs(("queued",)) if j.stop_request != "cancel"],
            active=self.store.jobs(ACTIVE),
            pair=self.pair,
            last_interactive_at=self.gate.last_interactive_at,
            vram_total_mib=total, vram_used_mib=used,
            take=bool(self.store.get_kv("take", False)),
            demand_pending=self.gate.waiting_interactive > 0,
            loaded=self.gate.loaded, want=want,
            services=self.services_view(now),
            reserved=self.cfg.reserved(now),
        )
        pair_busy = self.pair_action is not None and self.pair_action.is_alive()
        vram = {m.name: m.vram_mib for m in self.cfg.model_table().values()}
        actions = scheduler.decide(view, self.cfg.margin_mib, vram,
                                   self.cfg.hold_minutes, self.cfg.preempt_on_interactive)
        if pair_busy:
            actions = [a for a in actions if a[0] not in ("stop_pair", "start_pair", "swap_model")]
        svc_busy = {n for n, t in self.svc_action.items() if t.is_alive()}
        actions = [a for a in actions if not (a[0] in ("start_service", "stop_service") and a[1] in svc_busy)]
        self.apply(actions)
        self.store.set_kv("daemon", {"at": time.time(), "pair": self.pair,
                                     "vram_total_mib": total, "vram_used_mib": used,
                                     "busy_pct": busy, "card": card, "in_flight": self.gate.in_flight, "waiting": self.gate.waiting,
                                     "waiting_interactive": self.gate.waiting_interactive,
                                     "model": self.gate.loaded, "want": want,
                                     "waiting_for": dict(self.gate.waiting_for),
                                     "services": {n: self.svc_state[n] for n in self.cfg.services},
                                     "reserved": self.cfg.reserved(now)})

    def publish_page(self) -> None:
        if time.time() - self.page_at < self.cfg.page_seconds:
            return
        self.page_at = time.time()
        try:
            table = self.cfg.model_table()
            pair_mib = table[self.gate.loaded].vram_mib if self.gate.loaded in table else table[self.cfg.default].vram_mib
            text = page.render(board.snapshot(self.cfg, self.store), pair_mib)
            page.publish(text, [os.path.expanduser(d) for d in self.cfg.page_dirs])
        except Exception:
            log.exception("status page")

    def run(self) -> None:
        proxy.serve(self.gate, self.cfg.listen, self.cfg.llm_port, self.cfg.llm_backend, self.cfg.resolve)
        proxy.serve(self.gate, self.cfg.listen, self.cfg.asr_port, self.cfg.asr_backend)
        for svc in self.cfg.services.values():
            proxy.serve(self.gate, self.cfg.listen, svc.port, svc.backend)
        self.store.event(None, "broker started")
        while True:
            try:
                self.tick()
                self.publish_page()
            except Exception:
                log.exception("tick")
            time.sleep(self.cfg.tick_seconds)
