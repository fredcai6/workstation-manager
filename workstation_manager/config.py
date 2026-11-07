"""Broker configuration: defaults, overridden by ~/.config/workstation-manager/config.toml."""

from __future__ import annotations

import dataclasses
import os
import time
import tomllib
from pathlib import Path
from typing import Dict, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
MIB = 1024 * 1024
PATHS = ("state_dir", "compose_file", "env_file")


@dataclasses.dataclass(frozen=True)
class Model:
    """One LLM the interactive pair can run. `env_file` holds the pair's full settings
    with this LLM (LLM_IMAGE, LLM_MODEL_FILE, flags, and the ASR's); `vram_mib` is
    what the pair measures with it."""
    name: str
    env_file: Optional[Path]
    vram_mib: int
    aliases: Tuple[str, ...] = ()

    @property
    def model_file(self) -> Optional[str]:
        """LLM_MODEL_FILE from the env file: how the running container is recognised."""
        if not self.env_file or not Path(self.env_file).exists():
            return None
        for line in Path(self.env_file).read_text().splitlines():
            if line.startswith("LLM_MODEL_FILE="):
                return line.split("=", 1)[1].strip()
        return None


@dataclasses.dataclass(frozen=True)
class Service:
    """A GPU server started on demand: the broker brings it up when a request arrives
    on `port` and stops it once nothing has used it for `idle_seconds`, or sooner when
    the room is needed. It runs `compose_service` from the compose file (profile of the
    same name) with `env_file`, and listens on 127.0.0.1:`backend`."""
    name: str
    port: int
    backend: int
    vram_mib: int
    compose_service: str
    env_file: Optional[Path] = None
    idle_seconds: float = 120.0


@dataclasses.dataclass(frozen=True)
class Config:
    state_dir: Path = Path.home() / ".local/state/workstation-manager"
    # The compute card's PCI address; unset means the AMD card with the most VRAM.
    pci_id: Optional[str] = None
    # Headroom kept free on top of every declared footprint.
    margin_mib: int = 1024
    # The pair's VRAM when no [models] table is configured.
    pair_vram_mib: int = 30000
    # LLMs the pair can switch between ([models.NAME] tables). A request naming one
    # (by name or alias) loads it; anything else is served by whatever is loaded.
    # The default is resident whenever no other model was asked for within hold_minutes.
    models: Dict[str, Model] = dataclasses.field(default_factory=dict)
    default_model: Optional[str] = None
    # On-demand servers ([services.NAME] tables), e.g. an OCR model used in bursts.
    services: Dict[str, Service] = dataclasses.field(default_factory=dict)
    # Batch may take the card once no interactive request has arrived for this long.
    hold_minutes: float = 30.0
    # Daily windows ("HH:MM-HH:MM", local time) when the interactive pair is kept up
    # with the default model: no job starts that wouldn't fit beside it, the pair is
    # not stopped for batch, and requests for another model wait for the window to
    # end. Running jobs are left alone.
    reserve_windows: Tuple[str, ...] = ()
    # Off: an interactive request waits for running batch work instead of evicting it.
    preempt_on_interactive: bool = False
    compose_file: Path = REPO / "interactive/docker-compose.yml"
    # Env file passed to `docker compose` for the pair (model files, paths, aliases).
    env_file: Optional[Path] = None
    # Public ports the proxy owns, and where the servers listen behind it.
    llm_port: int = 8005
    asr_port: int = 8006
    llm_backend: int = 18005
    asr_backend: int = 18006
    # Addresses the proxy listens on. Loopback callers count as batch; add the Docker
    # bridge gateways that interactive clients (the gate, host.docker.internal) arrive on.
    listen: Tuple[str, ...] = ("127.0.0.1",)
    # How long a caller's request waits for the servers before a 503.
    queue_wait_seconds: float = 6 * 3600
    # Time an in-flight request gets to finish before the pair is stopped under it.
    drain_seconds: float = 120.0
    tick_seconds: float = 5.0
    # Directories the status page is written to (add a static site's directory to publish it).
    page_dirs: Tuple[str, ...] = ("~/.local/state/workstation-manager/page",)
    page_seconds: float = 30.0
    default_retries: int = 2
    default_grace_seconds: int = 60
    # Physical cores never handed to managed work: left for the desktop, agents, the
    # interactive pair and this broker. Cores busy with unmanaged work also count
    # against the room, so this is a floor.
    reserve_cores: int = 2
    # Physical cores batch jobs never take, kept for pytest runs and `ws run`.
    lease_reserve_cores: int = 2
    # Memory kept free on top of every declared footprint (MemAvailable headroom).
    mem_margin_mib: int = 8192
    # What a pytest run asks for unless it says otherwise (WS_PYTEST_CORES /
    # WS_PYTEST_MEM, or -n N with xdist); -n auto asks for pytest_auto_cores.
    lease_cores: int = 1
    lease_mem_mib: int = 2048
    pytest_auto_cores: int = 4
    # A pytest run that has waited this long runs anyway, unpinned, and says so.
    lease_wait_seconds: float = 900.0

    @property
    def db_path(self) -> Path:
        return self.state_dir / "broker.db"

    def model_table(self) -> Dict[str, Model]:
        if self.models:
            return self.models
        return {"default": Model("default", self.env_file, self.pair_vram_mib)}

    @property
    def default(self) -> str:
        return self.default_model or next(iter(self.model_table()))

    def resolve(self, requested: Optional[str]) -> Optional[str]:
        """The registered model a request names, or None if it names none."""
        if not requested:
            return None
        for m in self.model_table().values():
            if requested == m.name or requested in m.aliases:
                return m.name
        return None

    def reserved(self, when: float) -> bool:
        """Whether `when` (epoch seconds) falls in a reserve window, in local time."""
        t = time.localtime(when)
        minute = t.tm_hour * 60 + t.tm_min
        for w in self.reserve_windows:
            start, end = (int(h) * 60 + int(m) for h, m in (p.split(":") for p in w.split("-")))
            if (start <= minute < end) if start <= end else (minute >= start or minute < end):
                return True
        return False

    @property
    def wake_path(self) -> Path:
        """Datagram socket the broker listens on, so a new request is seen at once."""
        return self.state_dir / "wake.sock"

    def job_dir(self, job_id: int) -> Path:
        return self.state_dir / "jobs" / str(job_id)


def load(path: Optional[Path] = None) -> Config:
    path = path or Path(os.environ.get(
        "WORKSTATION_MANAGER_CONFIG", Path.home() / ".config/workstation-manager/config.toml"))
    if not Path(path).exists():
        return Config()
    raw = tomllib.loads(Path(path).read_text())
    fields = {f.name: f for f in dataclasses.fields(Config)}
    kwargs = {}
    for key, value in raw.items():
        if key not in fields:
            raise ValueError(f"unknown config key {key!r} in {path}")
        if key == "models":
            kwargs[key] = {
                name: Model(name, Path(m["env_file"]).expanduser() if m.get("env_file") else None,
                            int(m["vram_mib"]), tuple(m.get("aliases", ())))
                for name, m in value.items()}
            continue
        if key == "services":
            kwargs[key] = {
                name: Service(name, int(v["port"]), int(v["backend"]), int(v["vram_mib"]),
                              v.get("compose_service", name),
                              Path(v["env_file"]).expanduser() if v.get("env_file") else None,
                              float(v.get("idle_seconds", 120)))
                for name, v in value.items()}
            continue
        default = getattr(Config, key, None)
        if key in PATHS:
            value = Path(value).expanduser()
        elif isinstance(default, tuple):
            value = tuple(value)
        kwargs[key] = value
    return Config(**kwargs)
