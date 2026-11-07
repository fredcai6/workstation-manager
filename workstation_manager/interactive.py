"""The interactive pair (27B on llm, transcriber on asr): health, start, stop."""

from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from typing import Optional

from .config import Config


def _compose(cfg: Config, *args: str, model: Optional[str] = None) -> subprocess.CompletedProcess:
    env_file = cfg.model_table()[model or cfg.default].env_file
    env = ["--env-file", str(env_file)] if env_file else []
    return subprocess.run(
        ["docker", "compose", "-f", str(cfg.compose_file), *env,
         "--profile", "llm", "--profile", "asr", *args],
        capture_output=True, text=True)


def healthy(port: int, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def running(cfg: Config) -> bool:
    r = _compose(cfg, "ps", "--status", "running", "-q", "llm", "asr")
    return bool(r.stdout.split())


def state(cfg: Config) -> str:
    """up when both answer /health, starting when containers run but are not ready."""
    if healthy(cfg.llm_backend) and healthy(cfg.asr_backend):
        return "up"
    return "starting" if running(cfg) else "down"


def loaded_model(cfg: Config) -> Optional[str]:
    """Which registered model the llm container is set to run, from its --model flag."""
    r = subprocess.run(["docker", "inspect", "-f", "{{json .Config.Cmd}}", "interactive-llm-1"],
                       capture_output=True, text=True)
    if r.returncode:
        return None
    try:
        cmd = json.loads(r.stdout)
        path = cmd[cmd.index("--model") + 1]
    except (ValueError, IndexError, TypeError):
        return None
    for m in cfg.model_table().values():
        if m.model_file and os.path.basename(path) == m.model_file:
            return m.name
    return None


def start(cfg: Config, model: Optional[str] = None) -> None:
    r = _compose(cfg, "up", "-d", "--no-build", "llm", "asr", model=model)
    if r.returncode:
        raise RuntimeError(f"compose up failed: {r.stderr.strip()[-400:]}")


def swap(cfg: Config, model: str) -> None:
    """Recreate only the llm container with `model`; the transcriber keeps running."""
    r = _compose(cfg, "up", "-d", "--no-build", "llm", model=model)
    if r.returncode:
        raise RuntimeError(f"compose up llm failed: {r.stderr.strip()[-400:]}")


def stop(cfg: Config) -> None:
    r = _compose(cfg, "stop", "llm", "asr")
    if r.returncode:
        raise RuntimeError(f"compose stop failed: {r.stderr.strip()[-400:]}")


# -- on-demand services --------------------------------------------------------------
def _service_compose(cfg: Config, svc, *args: str) -> subprocess.CompletedProcess:
    env_file = svc.env_file or cfg.model_table()[cfg.default].env_file
    env = ["--env-file", str(env_file)] if env_file else []
    return subprocess.run(
        ["docker", "compose", "-f", str(cfg.compose_file), *env,
         "--profile", svc.compose_service, *args],
        capture_output=True, text=True)


def service_state(cfg: Config, svc) -> str:
    if healthy(svc.backend):
        return "up"
    r = _service_compose(cfg, svc, "ps", "--status", "running", "-q", svc.compose_service)
    return "starting" if r.stdout.split() else "down"


def start_service(cfg: Config, svc) -> None:
    r = _service_compose(cfg, svc, "up", "-d", "--no-build", svc.compose_service)
    if r.returncode:
        raise RuntimeError(f"compose up {svc.compose_service} failed: {r.stderr.strip()[-400:]}")


def stop_service(cfg: Config, svc) -> None:
    r = _service_compose(cfg, svc, "stop", svc.compose_service)
    if r.returncode:
        raise RuntimeError(f"compose stop {svc.compose_service} failed: {r.stderr.strip()[-400:]}")
