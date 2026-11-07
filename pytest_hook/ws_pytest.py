"""pytest plugin: a pytest run waits for its cores from the workstation manager.

Loaded in every venv through PYTEST_PLUGINS=ws_pytest with this directory on
PYTHONPATH (set in ~/.bashrc; see the README). Before collection, the run asks the
broker for a lease, waits until it is granted, and pins itself (and its xdist
workers) to those cores; the lease ends with the run.

It asks for 1 core and 2 GB unless told otherwise: `-n N` (xdist) asks for N cores,
`-n auto` for the broker's pytest_auto_cores and then runs that many workers;
WS_PYTEST_CORES and WS_PYTEST_MEM (e.g. 8G) override. WS_PYTEST=0 or
`-p no:ws_pytest` skips it. Runs inside a job or lease, xdist workers,
--collect-only and --help take no lease. Any failure here prints one line and the
run goes ahead unpinned.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_REPO = Path(os.path.realpath(__file__)).parent.parent
_state = {"lease": None, "cfg": None}


def _package():
    if "workstation_manager" in sys.modules:
        return sys.modules["workstation_manager"]
    init = _REPO / "workstation_manager" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "workstation_manager", init, submodule_search_locations=[str(init.parent)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["workstation_manager"] = mod
    spec.loader.exec_module(mod)
    return mod


def _skip(config) -> bool:
    if os.environ.get("WS_PYTEST", "1") == "0" or os.environ.get("PYTEST_XDIST_WORKER"):
        return True
    if os.environ.get("WS_JOB_ID") or os.environ.get("WS_LEASE"):
        return True
    opt = config.option
    return bool(getattr(opt, "collectonly", False) or getattr(opt, "help", False)
                or getattr(opt, "version", False) or getattr(opt, "markers", False)
                or getattr(opt, "showfixtures", False))


def _acquire(config, cores=None) -> None:
    if _state["lease"] is not None or _state["cfg"] is not None or _skip(config):
        return
    _package()
    from workstation_manager import config as wsconfig, lease as wslease
    from workstation_manager.cli import parse_size
    cfg = wsconfig.load()
    _state["cfg"] = cfg
    n = getattr(config.option, "numprocesses", None)
    auto = n in ("auto", "logical")
    if cores is None:
        if os.environ.get("WS_PYTEST_CORES"):
            cores = int(os.environ["WS_PYTEST_CORES"])
        elif auto:
            cores = cfg.pytest_auto_cores
        elif isinstance(n, int) and n > 0:
            cores = n
        else:
            cores = cfg.lease_cores
    mem = parse_size(os.environ["WS_PYTEST_MEM"]) if os.environ.get("WS_PYTEST_MEM") \
        else cfg.lease_mem_mib
    name = "pytest " + os.path.basename(os.getcwd())
    got = wslease.acquire(cfg, "pytest", name, cores, mem, wait_limit=cfg.lease_wait_seconds)
    if got is None:
        return
    wslease.enter(got)
    _state["lease"] = got
    if auto:
        config.option.numprocesses = got.cores


def _guarded(config, cores=None) -> None:
    try:
        _acquire(config, cores)
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # never fail a test run over its lease
        print(f"ws: no core lease ({type(exc).__name__}: {exc}); running unpinned",
              file=sys.stderr)


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config):
    _guarded(config)


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_auto_num_workers(config):
    """xdist asks how many workers `-n auto` means: as many as the lease's cores."""
    _guarded(config)
    lease = _state["lease"]
    return lease.cores if lease is not None else None


def pytest_unconfigure(config):
    lease, cfg = _state["lease"], _state["cfg"]
    _state["lease"] = None
    if lease is not None:
        try:
            from workstation_manager import lease as wslease
            wslease.release(cfg, lease)
        except Exception:
            pass  # the broker ends it once this process is gone
