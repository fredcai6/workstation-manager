"""pytest plugin: a pytest run waits for its cores from the workstation manager.

Loaded in every venv through PYTEST_PLUGINS=ws_pytest with this directory on
PYTHONPATH (set in ~/.bashrc; see the README). Before collection, the run queues a
job of kind pytest, waits until it is granted, and pins itself (and its xdist
workers) to those cores; the job ends with the run, with pytest's exit status.

It asks for 1 core and 2 GB unless told otherwise: `-n N` (xdist) asks for N cores,
`-n auto` for the broker's pytest_auto_cores and then runs that many workers;
WS_PYTEST_CORES and WS_PYTEST_MEM (e.g. 8G) override. WS_PYTEST=0 or
`-p no:ws_pytest` skips it. Runs inside a job, xdist workers, --collect-only and
--help queue nothing. Any failure here prints one line and the
run goes ahead unpinned.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_REPO = Path(os.path.realpath(__file__)).parent.parent
_state = {"job": None, "cfg": None, "exit": None}


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
    if os.environ.get("WS_JOB_ID"):
        return True
    opt = config.option
    return bool(getattr(opt, "collectonly", False) or getattr(opt, "help", False)
                or getattr(opt, "version", False) or getattr(opt, "markers", False)
                or getattr(opt, "showfixtures", False))


def _acquire(config, cores=None) -> None:
    if _state["job"] is not None or _state["cfg"] is not None or _skip(config):
        return
    _package()
    from workstation_manager import config as wsconfig, foreground
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
            cores = cfg.pytest_cores
    mem = parse_size(os.environ["WS_PYTEST_MEM"]) if os.environ.get("WS_PYTEST_MEM") \
        else cfg.pytest_mem_mib
    name = "pytest " + os.path.basename(os.getcwd())
    got = foreground.acquire(cfg, "pytest", name, cores, mem, wait_limit=cfg.pytest_wait_seconds,
                             argv=sys.argv[1:])
    if got is None:
        return
    foreground.enter(got)
    _state["job"] = got
    if auto:
        config.option.numprocesses = got.cores


def _guarded(config, cores=None) -> None:
    try:
        _acquire(config, cores)
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # never fail a test run over the queue
        print(f"ws: not queued ({type(exc).__name__}: {exc}); running unpinned",
              file=sys.stderr)


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config):
    _guarded(config)


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_auto_num_workers(config):
    """xdist asks how many workers `-n auto` means: as many as the job's cores."""
    _guarded(config)
    job = _state["job"]
    return job.cores if job is not None else None


def pytest_sessionfinish(session, exitstatus):
    _state["exit"] = int(exitstatus)


def pytest_unconfigure(config):
    job, cfg = _state["job"], _state["cfg"]
    _state["job"] = None
    if job is not None:
        try:
            from workstation_manager import foreground
            foreground.release(cfg, job, _state["exit"])
        except Exception:
            pass  # the broker ends it once this process is gone
