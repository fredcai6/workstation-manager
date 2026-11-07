"""The broker loop: observe the card, the cores and memory, ask the scheduler, apply its actions."""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from typing import Optional

from . import board, cpu, interactive, page, proxy, runner, scheduler
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
        self.cores = cpu.topology()
        self.meter = cpu.Meter(self.cores)
        self.wake = threading.Event()
        self.cgroups: dict = {}  # unit -> cgroup path
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
            elif res.get("mem_oom"):
                self.store.update(job.id, status="failed", ended_at=now,
                                  failures=job.failures + 1, exit_code=res["exit_code"])
                self.store.event(job.id, f"killed at its memory limit ({job.mem_mib} MiB declared); "
                                         "not retried (resubmit with more --mem)")
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

    def reconcile_leases(self) -> None:
        for lease in self.store.leases():
            if not cpu.alive(lease.pid, lease.pid_start):
                self.store.end_lease(lease.id)
        self.store.prune_leases(time.time() - 86400)

    def _cgroup(self, unit: str):
        if unit not in self.cgroups:
            path = cpu.unit_cgroup(unit)
            if path is None or not path.exists():
                return None
            self.cgroups[unit] = path
        return self.cgroups[unit]

    def host_view(self, active, leases):
        """The cores and memory as the scheduler sees them, and memory that holders have
        declared but not yet used (it is about to leave MemAvailable)."""
        busy = self.meter.sample()
        total, available = cpu.meminfo()
        unrealized = 0
        for j in active:
            if j.mem_mib:
                used = (cpu.memory_current_mib(self._cgroup(j.unit)) or 0) \
                    + cpu.container_memory_mib(runner.label(j))
                unrealized += max(0, j.mem_mib - used)
        for l in leases:
            if l.status != "granted":
                continue
            used = cpu.memory_current_mib(self._cgroup(l.unit)) if l.unit else None
            unrealized += max(0, l.mem_mib - (used if used is not None else cpu.rss_mib(l.pid)))
        live = {j.unit for j in active} | {l.unit for l in leases if l.unit}
        self.cgroups = {u: p for u, p in self.cgroups.items() if u in live}
        return (scheduler.CpuView([c.cpus for c in self.cores], list(busy), self.cfg.reserve_cores,
                                   self.cfg.lease_reserve_cores),
                scheduler.MemView(total, available, self.cfg.mem_margin_mib, unrealized))

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
            if kind == "grant_lease":
                self.store.grant_lease(act[1], act[2])
            elif kind == "start_job":
                job = self.store.get(act[1])
                if len(act) > 2:
                    self.store.update(job.id, cpus=act[2])
                    job = self.store.get(job.id)
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
        self.reconcile_leases()
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
        active = self.store.jobs(ACTIVE)
        leases = self.store.leases()
        cpu_view, mem_view = self.host_view(active, leases)
        view = scheduler.View(
            now=time.time(),
            queued=[j for j in self.store.jobs(("queued",)) if j.stop_request != "cancel"],
            active=active,
            pair=self.pair,
            last_interactive_at=self.gate.last_interactive_at,
            vram_total_mib=total, vram_used_mib=used,
            take=bool(self.store.get_kv("take", False)),
            demand_pending=self.gate.waiting_interactive > 0,
            loaded=self.gate.loaded, want=want,
            services=self.services_view(now),
            reserved=self.cfg.reserved(now),
            cpu=cpu_view, mem=mem_view, leases=leases,
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
                                     "reserved": self.cfg.reserved(now),
                                     "cpu": {"cores": [list(c) for c in cpu_view.cores],
                                             "busy": [round(b, 3) for b in cpu_view.busy],
                                             "reserve": cpu_view.reserve,
                                             "lease_reserve": cpu_view.lease_reserve},
                                     "mem": {"total_mib": mem_view.total_mib,
                                             "available_mib": mem_view.available_mib,
                                             "margin_mib": mem_view.margin_mib,
                                             "unrealized_mib": mem_view.unrealized_mib}})

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
        self.listen_for_wakes()
        # One reading of the cores before the first decision, so a restart on a busy
        # machine does not see idle cores.
        self.meter.sample()
        time.sleep(1.5)
        self.store.event(None, "broker started")
        while True:
            try:
                self.tick()
                self.publish_page()
            except Exception:
                log.exception("tick")
            self.wake.wait(self.cfg.tick_seconds)
            self.wake.clear()

    def listen_for_wakes(self) -> None:
        """A lease request pokes the wake socket, so it is granted without waiting a tick."""
        path = self.cfg.wake_path
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.bind(str(path))

        def loop():
            while True:
                sock.recv(16)
                self.wake.set()
        threading.Thread(target=loop, daemon=True).start()
