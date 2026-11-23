"""The suite never drives the machine's interactive tier.

A test daemon's tick can decide to start, stop or swap the pair. Through the real
`interactive` module that is `docker compose up` against this checkout's compose file
with the test's config, and the project is named `interactive` either way, so it
replaces the live containers: on 2026-11-22 a run of TestTickPublishesModelAndSamples
recreated interactive-llm-1 from a worktree with no env file (Qwen defaults, an empty
./models), and it crash-looped. Every compose call made under test fails instead.
"""

from workstation_manager import interactive


def _refuse(*args, **kwargs):
    raise RuntimeError("tests must not run docker compose on the live interactive tier")


interactive._compose = _refuse
interactive._service_compose = _refuse
