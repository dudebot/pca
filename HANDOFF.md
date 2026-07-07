# pca — Handoff (2026-07-07)

> Synthesized on Opus 4.8 after the Fable quota expired at the handoff step. The kill
> experiment was implemented, unit-tested, and smoke-passed on Fable; `STATUS.md` is the
> authoritative verdict artifact on disk. This file summarizes it.

## Verdict — LAUNCH-READY (not run here, by design)

The pre-registered kill experiment — Phase-3 bootstrap-to-convergence with issue-#7
exhaustive leaf completion — is **fully implemented, unit-tested, integration-smoked, and
launchable with one command**, but was correctly **not run** in this session. It couldn't
be: the box was saturated (load ~14/8 cores, GPU owned by the tractus job, 13GB disk),
and the minimum single-round signal is ~1.5–2.5h CPU on an *idle* box — an order of
magnitude past the 25-minute sprint cap. So it was made launch-ready instead of run blind.
This is the honest outcome: pca gets a real go/no-go for ~$2–4 of spot compute whenever
you want it, with the verdict rule fixed in advance so the answer can't be motivated.

## Verified state snapshot

- Phase-2's headline result is real and committed: value-network beats random (51% vs 31% Success@8), with artifacts. The emulator builds and passes fib=55 today. An Adam-vs-Muon ablation is recoverable from git history.
- Phase-3 bootstrap never converged (frontier stuck at depth 4; last commit was mid-crash-debug) — which is exactly the open question the kill experiment settles.
- The kill experiment code (implementation + `run_kill_experiment.sh` + README RUNBOOK) is **uncommitted** on branch `kill-experiment-2026-07-07`. Unit-tested and smoke-passed; not run at scale.

## Pre-registered kill rule (registered 2026-07-07, before any run)

No numeric threshold existed in the repo, so one was registered up front (also in the
`run_kill_experiment.sh` header): **KILL** if, after the loop with `--exhaustive-radius 3`
runs to escalation-exhaustion or 10 rounds, the solved frontier contains **zero depth≥6
tasks** that `timeout 120 ./enumerate <task> -d 8 --no-branch` cannot also solve. (Depth 6
rather than issue #6's depth 5, because radius-3 completion hands the model depth 5–6
nearly for free — so depth 6 is the honest bar for "the learned search beats brute force.")
SURPRISE (keep) if it clears depth≥6 beyond brute-force reach.

## What this run changed

- Branch `kill-experiment-2026-07-07` (uncommitted): the bootstrap+leaf-completion implementation, `run_kill_experiment.sh`, README RUNBOOK, and `STATUS.md`. No commits, no pushes.

## One-command launch

```bash
./run_kill_experiment.sh                          # rental / GPU box
CUDA_VISIBLE_DEVICES= ./run_kill_experiment.sh    # idle CPU box, ~1-2 days
```

Rental spec: an RTX 4090-class spot with 8+ vCPUs, **~$2–4 total**. (The originally-imagined
H100 at ~$6–18 buys nothing for a 280K-param MLP except `--max-hidden 1024` headroom —
don't over-rent.)

## Open decisions for you — rapid-fire

1. **Run the kill experiment, or archive on the strategic read?** Hypothesis: **run it once** — it's cheap ($2–4), the code is done, and a real convergence number is worth more than a vibes-archive. But: the strategic case for archiving regardless is strong (the endgame is AlphaDev-with-a-datacenter, and prompt→assembly is being won from the LLM+compiler side). So: run it for the clean-conscience number, then archive either way.
2. **Keep the emulator + CPU-prefix/GPU-suffix CUDA solver regardless of verdict?** Hypothesis: **YES** — those are genuinely reusable artifacts independent of whether the superoptimizer thesis lives.
3. **If it SURPRISEs (clears depth≥6)?** Then and only then invest the next capped step (STATUS.md names it). Don't pre-commit effort to a project the strategy section already outflanks.

## Next actions

1. Read `STATUS.md`.
2. Review the branch (`git status` / `git diff`); commit the record.
3. When you have an idle box or want to spend $2–4: fire `./run_kill_experiment.sh`, let it run, read the verdict it writes, archive.

## Landmines

- **Untracked branch work is unrecoverable** — commit before any `git clean`.
- **Don't over-rent** — this is a tiny model; an H100 is wasted money here.
- The kill rule is pre-registered *specifically so the answer can't be motivated after the fact.* If the frontier lands at depth 5, that's a KILL by the registered rule (depth 6 was the bar) — resist the urge to relitigate the threshold post-hoc.
