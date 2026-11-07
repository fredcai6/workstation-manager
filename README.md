# workstation-manager

Shares this workstation between the work on it. A broker hands out physical CPU
cores, memory and the AMD GPU: batch jobs wait in a queue, foreground work (pytest
runs, `ws run`) takes a lease, and each runs pinned to its own cores with its memory
declared. The GPU also has an always-wanted interactive tier (a local LLM and a
speech-recognition server): the broker lends the card to batch when the tier is idle
and fronts the LLM/ASR ports so callers wait instead of failing while it is lent
out. Python standard library only; Linux, AMDGPU, Docker and systemd user units.

Why cores: on this 12-core Ryzen 9900X, memory-bound fits ran fastest as four
single-core jobs pinned to alternating CCDs (1201 passes/h). The usual overloaded
state reached 750-900, and pinning to SMT siblings cost 35% (2026-10-15 sweep,
`scratch/v35/perf/RESULTS_20261015.md`). Why memory: on 2026-10-30, 21 fits at
~2.6 GB each ran the 59 GB box out of memory and restarted it.

## Using it

```
ws status [--json]                    card, cores, memory, leases, queue
ws submit --cores 4 --mem 8G [--vram 18G] --name NAME -- CMD...   queue a job
ws run --cores 4 --mem 8G -- CMD...   run here, in the foreground, once it fits
ws queue [--json]                     broker state only
ws cancel ID
ws log ID                             path of a job's log
ws take / ws release                  operator override: stop everything / resume
```

`--cores` counts physical cores (default 1); each comes with both of its SMT
siblings. `--mem` is peak memory and is required: it is enforced (MemoryMax, no
swap), so a job or command that goes past it is killed, and a killed job is not
retried. `--vram` (default none) makes it a GPU job.

A submitted job runs from the directory it was submitted in, with the submitter's
environment. `ws run` runs the command directly, with its terminal, and returns its
exit code. Inside a job or a lease, `ws run` just runs the command: that work is
already accounted for.

`gpu` is the old command, kept only while worktree scripts still call
`~/projects/gpu-manager/gpu submit` (the folder is a symlink to this one). A job it
submits, like every job submitted before the rename, keeps the old names: unit
`gpu-job-N`, label `gpu-manager.job=N`, `GPU_JOB_*` env next to `WS_JOB_*`. It is
counted as one core and no memory, and is neither pinned nor capped. Delete `gpu`,
the symlink and `Job.legacy` once nothing calls them.

Every pinned process gets `WS_JOB_CPUS` (its logical CPUs, e.g. `0,6,12,18`) and
`WS_JOB_CORES`. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` are
set to the core count unless already set. Size worker pools from `WS_JOB_CORES`.

### pytest

Every pytest run on the machine takes a lease. `PYTEST_PLUGINS=ws_pytest`, with
`pytest_hook/` on `PYTHONPATH` (both set in `~/.bashrc`), loads
`pytest_hook/ws_pytest.py` in every venv. Before collection, the run waits for its
cores and pins itself to them, and its xdist workers inherit them. The lease ends
with the run.

- It asks for 1 core and 2 GB. `-n N` asks for N cores, and `-n auto` asks for
  `pytest_auto_cores` (4) and runs that many workers. `WS_PYTEST_CORES=3` and
  `WS_PYTEST_MEM=8G` override.
- A run that waits prints one line saying so; `ws status` shows who holds the cores.
  After `lease_wait_seconds` (15 min) it runs anyway, unpinned, and says so. When
  the broker is not running it runs unpinned at once.
- No lease is taken for `--collect-only`, `--help`, xdist workers, runs inside a
  job or lease, or with `WS_PYTEST=0` or `-p no:ws_pytest`. Memory is declared for
  accounting only.

### The job contract

- **Save progress about every 10 minutes and resume from it.** A job can be stopped
  at any moment (operator take, reboot, power loss) and is simply rerun with the
  same command.
- **Stop on SIGTERM.** The job gets its grace period (`--grace`, default 60 s)
  before SIGKILL.
- **Declare cores, memory and VRAM** with `--cores`, `--mem` and `--vram`. A job
  starts only when the declared budget and the measurements both have room.
- **Containers:** add `--label "$WS_JOB_LABEL"` to `docker run`, so a stopped job
  cannot leave a container holding VRAM. Also add `--cpuset-cpus "$WS_JOB_CPUS"`
  and `--memory "${WS_JOB_MEM_MIB}m"`: a container runs under Docker, not in the
  job's unit, so the job's pinning and memory cap do not reach it.
- **Report progress** by printing `chunk N/M` (optionally with `NN%`) on a line, or
  by calling `ws progress --chunk N/M --pct P`. Keep chunks under 4 hours.
- **Failures:** a failing exit is retried up to `--retries` times (default 2).
  Running out of declared memory or VRAM is not retried. A stop, a reboot or a power
  loss is an interruption: the job goes back to the front of the queue, and it
  doesn't count as a failure.

The status page shows the card and who holds its VRAM, each core and who holds it
(or how busy unmanaged work keeps it), memory by holder, the leases, each running
job's progress and last output line, the queue, finished jobs and the last day of
events. The broker rewrites it every 30 s into each directory in `page_dirs`.

## How cores and memory are shared

- **Physical cores, alternating CCDs.** A holder gets whole cores, least busy first,
  in the order 0, 6, 1, 7, ... (each with its sibling, 12, 18, 13, 19, ...).
  Pinning is CPU affinity only: nothing stops unmanaged processes from running on a
  held core.
- **Room.** `reserve_cores` (2) are never handed out: they are left for the desktop,
  agents, the interactive pair and the broker. `lease_reserve_cores` (2) are kept
  for leases: batch never takes them. A batch job also counts cores that unmanaged
  work keeps busy (smoothed over about 30 s) against its room, so it waits rather
  than piling onto an overloaded box. A lease has a caller waiting and counts
  declarations only.
- **Memory.** A holder starts only when its declaration fits both the budget
  (MemTotal less `mem_margin_mib`, 8 GB) and what is measured: MemAvailable less the
  margin, less memory running holders have declared but not yet used.
- **Order.** Leases go first, first come first served, even during a take: they hold
  no VRAM. Then jobs, first come first served. A later job may start ahead of the
  head only if the head would still fit once running work is gone, in every
  resource the later job uses. A CPU-only job neither waits for the pair nor sends
  it away.
- **Wake-ups.** A new lease or job pokes the broker's socket (`wake.sock` in the
  state dir), so a lease is granted on the next tick, about a second later, instead
  of after the 5 s wait.

## How the card is shared

- **The interactive pair:** `interactive-llm-1` (:8005) and `interactive-asr-1`
  (:8006), serving chat, voice notes and tutoring clients. The model, image and
  flags come from the env file (`LLM_IMAGE`, `LLM_MODEL_FILE`, `LLM_EXTRA_ARGS`, ...);
  set `pair_vram_mib` to what the pair measures.
- **Batch takes the card** once no assistant request (`X-GPU-Hold: 1`) has arrived for 30 minutes.
  The broker lets in-flight requests finish (up to 2 minutes), stops the pair, and
  runs jobs first come, first served. A later job may start beside others only if it
  doesn't delay the job at the head of the queue.
- **The pair comes back** when the queue is empty and it fits.
- **Switching models:** with `[models.*]` in the config, the LLM slot can run any of
  several models. The default stays resident. A request that names another model
  (by name or alias, in the JSON `"model"` field or as `?model=` on a GET such as
  `/props`) waits while the broker swaps it in, and that model stays until no
  request has named a different one for `hold_minutes`. Requests naming no
  registered model (e.g. `"local"`) are served by whatever is loaded. Only the LLM
  is recreated; the transcriber keeps running.
- **Reserve windows:** `reserve_windows = ["07:30-08:00"]` keeps the pair up with the
  default model at times you expect to use it (local time). No job starts that wouldn't
  fit beside it, the pair isn't stopped for batch, and requests naming another model
  wait for the window to end. Running jobs are left alone: a fit that holds the whole
  card keeps it, and the pair comes up once there's room.
- **On-demand services:** a `[services.NAME]` table gives a GPU server its own port
  (e.g. an OCR model on :8007, `ocr:8007` through the gate) without holding VRAM
  between uses. The first request starts it (the compose service of the same name,
  under its own profile) and waits until it is up. It stops after `idle_seconds`
  without a request, or after 10 s idle when its room is needed by a model swap or
  the head of the queue. A waiting request ends a model's hold, so the slot goes
  back to the default to make room. It starts ahead of queued batch, and no batch
  job starts that would keep a waiting one out. Its callers don't reset the hold.
  Measured: a cold start plus one page is ~13 s; handing the room back to the 27B
  is ~12 s.
- **No preemption:** an assistant request does not evict batch work
  (`preempt_on_interactive = false`).
- **Requests wait:** any request to :8005/:8006 waits while the pair is down, and
  goes through once it is back. `/health` answers 503 at once instead of waiting.
- **Only assistant use holds the card:** a request sent with `X-GPU-Hold: 1` (the owner
  talking to the assistant) resets the 30-minute hold; nothing else does, whatever its
  origin. The hold runs from the last such request and survives broker restarts.

Measured on a 32 GB Radeon AI PRO R9700, with Qwen3-ASR 1.7B as the transcriber:

| LLM | ASR context | pair VRAM | room for batch beside it | cold load |
|---|---|---|---|---|
| Qwen3.8 27B Q4_K_M, 64k context, 16 slots | 32k | 29.2 GB | ~1.5 GB | ~22 s |
| Nemotron-Labs Dory 1.0 4B Q5_K_M, 32k context, 4 slots | 32k | 14.9 GB | ~16 GB | ~3 s |
| the same Dory | 8k | 12.0 GB | ~19 GB | ~3 s |

The ASR context only has to hold one request. A 2-minute chunk is ~1.6k audio tokens
(about 13 per second), plus the transcript (at most ~0.6k for very fast speech) and any
biasing vocabulary (~1.8k tokens for 4,000 characters); the worst case measured 3.7k of 8k.

`docker pause` frees no VRAM, so a paused job still blocks the card.

## Layout

| Path | What |
|---|---|
| `workstation_manager/scheduler.py` | the scheduling rules, as one pure function |
| `workstation_manager/daemon.py` | the broker loop (`ws daemon`) |
| `workstation_manager/cpu.py` | core topology, the load meter, memory, pinning |
| `workstation_manager/lease.py` | leases: what `ws run` and the pytest hook wait for |
| `pytest_hook/ws_pytest.py` | the pytest plugin |
| `gpu` | temporary: the old command, for callers not yet moved |
| `workstation_manager/proxy.py` | the :8005/:8006 front door |
| `workstation_manager/runner.py` | jobs as transient systemd user units (`ws-job-N`) |
| `workstation_manager/board.py`, `page.py` | broker state for `ws status` and the status page |
| `workstation_manager/store.py` | state in `~/.local/state/workstation-manager/broker.db` |
| `workstation_manager/config.py` | defaults; override in `~/.config/workstation-manager/config.toml` |
| `interactive/docker-compose.yml` | the pair, plus `gate`, which holds the `llm`/`asr` names on Journal's network |
| `interactive/check-journal.sh` | end-to-end chat and voice-note check through Journal |
| `deploy/workstation-manager.service` | systemd user unit |
| `deploy/config.example.toml` | every setting, for `~/.config/workstation-manager/config.toml` |

## Setup

```
ln -s "$PWD/ws" ~/.local/bin/ws
cp deploy/config.example.toml ~/.config/workstation-manager/config.toml   # then edit
ln -s "$PWD/deploy/workstation-manager.service" ~/.config/systemd/user/
systemctl --user enable --now workstation-manager
loginctl enable-linger "$USER"     # so the broker and jobs run without a login
# every pytest run takes a lease (in ~/.bashrc, after the interactive guard):
export PYTHONPATH="$HOME/projects/workstation-manager/pytest_hook${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_PLUGINS=ws_pytest
```

Start the pair once with `docker compose -f interactive/docker-compose.yml
--env-file ENV --profile llm --profile asr up -d --no-build llm asr gate`
(build first if you have no `docker-llm` image); the broker manages it from then on.

## Operating

```
systemctl --user status workstation-manager        # broker
journalctl --user -u workstation-manager -f
systemctl --user restart workstation-manager       # jobs keep running; they are their own units
```

Restart the pair by hand only with the broker in a take, or it will undo you:
`ws take`, do the work, `ws release`.

## Tests

```
python3 -m pytest -q tests
```

`tests/test_live.py` runs real systemd units, pinning and a pytest run through the
hook against a private broker (its own state dir, the card stubbed).
