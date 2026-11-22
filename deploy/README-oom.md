# Who gets killed when memory runs out

Queue work saves progress and resumes; the desktop and the broker don't. So when the box runs out of memory, the kill should land on a job.

| | kernel `oom_score_adj` | systemd-oomd |
|---|---|---|
| `ws submit` jobs (`ws-job-*.service`) | 800 | default |
| `ws run` commands | 800 | default |
| other user units, terminals, agents | 200 (user-unit default) | default |
| broker (`workstation-manager.service`) | 100 | avoid |
| desktop (`session.slice`) | 100 (gnome-shell) | avoid |

systemd-oomd kills by memory pressure on `user@1000.service` (50% for 20 s). It respects `avoid` here because these cgroups and the monitored one share an owner (systemd.resource-control(5)).

**Not covered:** Docker containers run under `system.slice` with score 0. Their own `--memory` cap holds them, but in a box-wide OOM the kernel weighs them below user processes of the same size. Add `--oom-score-adj 800` to a job's `docker run` to put it with the other jobs.

Install the session drop-in once:

```bash
ln -s "$PWD/deploy/session.slice.d" ~/.config/systemd/user/session.slice.d
systemctl --user daemon-reload
```

Rationale: the three box-wide OOM kills of 11-16 hit terminal Python outside the queue, and nothing decided which process went. Rejected: negative scores for the desktop, which need root.
