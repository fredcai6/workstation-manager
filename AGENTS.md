# GPU manager

README.md describes the broker and the job contract. The scheduling rules live in
`workstation_manager/scheduler.py`; change them there and in `tests/test_scheduler.py`.

The broker is live (`systemctl --user status workstation-manager`). After changing broker
code, run `python3 -m pytest -q tests`, then `systemctl --user restart workstation-manager`.
Jobs survive the restart. After changing `interactive/docker-compose.yml`, run
`ws take`, then `docker compose ... up -d --no-build`, then
`interactive/check-journal.sh`, then `ws release`.
