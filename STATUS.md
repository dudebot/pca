# KILL-FIRST verdict — 2026-07-07

## VERDICT: LAUNCH-READY

The pre-registered kill experiment (Phase-3 bootstrap-to-convergence with
issue-#7 exhaustive leaf completion) is **fully implemented, unit-tested,
integration-smoked, and launchable with one command** — but was NOT run
here. It could not be: the box is at load ~14/8 cores with 13GB disk free,
the GPU is owned by another job, and the minimum single-round signal is
~5h on an idle box (measured throughput is ~5x worse under current
contention). That exceeds the 25-minute CPU cap by more than an order of
magnitude at any valid scale — round-0 training alone (required before any
model-guided search exists) is ~1.5-2.5h CPU.

Everything lives UNCOMMITTED on branch `kill-experiment-2026-07-07`.

## One-command launch

```bash
./run_kill_experiment.sh                          # rental / GPU box
CUDA_VISIBLE_DEVICES= ./run_kill_experiment.sh    # idle CPU box, ~1-2 days
```

Rental spec, cost table, and clone-to-verdict commands: README.md
"RUNBOOK" section. Recommended: RTX 4090-class spot with 8+ vCPUs, ~$2-4
total (H100 as originally specced: ~$6-18, buys nothing for a 280K-param
MLP except `--max-hidden 1024` headroom).

## Pre-registered kill rule (registered 2026-07-07, before any run)

No numeric threshold existed anywhere in the repo or issues; the closest
verbatim commitment is issue #6's key metric: *"Can the model solve
depth-5+ tasks that neither random search nor brute force (120s timeout)
can reach?"* The registered rule (also in `run_kill_experiment.sh` header):

**KILL** if, after the loop with `--exhaustive-radius 3` runs to
escalation exhaustion or 10 rounds, the solved frontier contains **zero
depth>=6 tasks** that `timeout 120 ./enumerate <task> -d 8 --no-branch`
cannot also solve. Depth 6 rather than issue #6's depth 5 because R=3
completion hands the model depth 5-6 nearly for free (the model navigates
only depth 2-3), so depth 5 no longer tests the model. `--no-branch` so
the brute-force baseline searches the same branchless space as the model
(branch mode explores a bigger space and would time out more easily,
biasing toward a false KEEP).

## Threshold decision tree for when results land

`kill_report/crosscheck.json` → `beyond_brute_force` = N:

- **N == 0 → KILLED.** Write the tombstone: leaf completion could not turn
  a depth-4-frontier model into a beyond-brute-force solver. Keep
  `kill_report/`, `logs/loop.log`, and the A/B summary as calibration
  artifacts; the branch dies with a clean conscience. Do not iterate on
  R=4, bigger models, or more rounds — those are exactly the escalations
  the loop already burned through before stopping.
- **N >= 1 and frontier advanced >=2 depths past the brute-force-solvable
  set with monotone new-solutions per round (`logs/loop.log` round
  summaries) → strong KEEP / SURPRISE.** Next capped investment (max ~1
  agent-day): raise search `max_depth` past the hardcoded 8 (issue #7
  targets 12-15+), rerun the loop, and re-verify the depth-7-8 claim from
  issue #7 that this experiment's R=0 A/B arm will have tested.
- **N >= 1 but weaker (no 2-depth advance or non-monotone) → marginal
  KEEP.** One more loop run at `MAX_ROUNDS=20` before any new code.
- **A/B sanity gate (`kill_report/ab_summary.json`):** if R=3 does not
  beat R=0 on solve rate at equal budget, the completion isn't paying for
  itself — treat any N >= 1 as suspect and inspect whether the "wins" came
  from brute force alone (model prefix depth <= 2).

## What was implemented and verified here (this session)

- `src/search_state.c/.h`: `search_exhaustive_complete()` — IDDFS 0..R
  with static per-depth OEP tables cleared per call, leaf check at every
  depth, shortest-completion guarantee; plus
  `search_exhaustive_states_explored()` instrumentation. Builds clean.
- `python/search_eval.py`: ctypes bindings + `exhaustive_complete()`.
- `python/bootstrap.py`: trigger at `depth >= max_depth - R`, verified-
  suffix replay, heuristic-dead-end handling (counted against budget),
  depth-dominance-aware `visited_oep` (Codex fix), per-task and aggregate
  instrumentation, `--exhaustive-radius` (default 3).
- `python/run_bootstrap.py`: `--exhaustive-radius` pass-through.
- `python/test_exhaustive.py`: 9 manifest tasks (d*=1..3) — root
  completion finds exactly-optimal-length solutions, partial-state
  completion works, `found=False` below optimal depth, no OEP state leak
  across calls, python replay solves. **9/9 pass, 0 failures.**
- Integration smoke (random-weights 64-dim checkpoint, 8 easy tasks,
  budget 20, max_depth 4): R=3 solved 8/8 (2 via exhaustive completion,
  304,220 states, ~34ms total) vs R=0 solved 6/8. Plumbing verified
  end-to-end including record writing; not a model-quality result.
- Measured C completion cost from root: d*=3 tasks ≈ 3.4-8.3M states in
  300-730ms (~11M states/s single-thread under load) — above issue #7's
  10-150ms/node estimate; the instrumentation exists to watch this.

## Numbers observed (all from this session)

| Measurement | Value |
|---|---|
| Unit tests | 9/9 pass |
| Smoke A/B (random model, budget 20) | R=3: 8/8 solved vs R=0: 6/8 |
| Exhaustive throughput (1 thread, loaded box) | ~11M states/s |
| R=3 completion from depth-1 trigger states | 14-20ms/call |
| Box state at decision time | load 13.9/8 cores, 13GB free disk |

## Files on branch `kill-experiment-2026-07-07` (uncommitted)

- `src/search_state.c`, `src/search_state.h` — C implementation
- `python/search_eval.py`, `python/bootstrap.py`, `python/run_bootstrap.py` — integration
- `python/test_exhaustive.py` — unit tests
- `run_kill_experiment.sh` — the one command
- `README.md` — RUNBOOK section (rental instructions)
- `STATUS.md` — this file
