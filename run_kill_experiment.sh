#!/usr/bin/env bash
# run_kill_experiment.sh — Phase-3 kill experiment, one command (issue #7 + #6).
#
# Pre-registered kill rule (registered 2026-07-07, before any run):
#   KILL if, after the bootstrap loop with --exhaustive-radius 3 runs to
#   escalation exhaustion or MAX_ROUNDS (whichever first), the solved-task
#   frontier contains ZERO depth>=6 tasks that a 120s branchless brute force
#   (timeout 120 ./enumerate <task> -d 8 --no-branch) cannot also solve.
#   KEEP if >=1 such task. Strong KEEP if the frontier advances >=2 depths
#   past the brute-force-solvable set with monotone new-solutions per round.
#   (Depth 6, not 5: R=3 completion hands the model depth 5-6 nearly for
#   free, so depth 5 no longer tests the model.)
#
# Local CPU-only usage (GPU owned by another job):
#   CUDA_VISIBLE_DEVICES= ./run_kill_experiment.sh
# GPU rental usage: see RUNBOOK section in README.md. Just:
#   ./run_kill_experiment.sh
#
# Tunables (env vars):
#   MAX_ROUNDS=10  EXH_RADIUS=3  NEG_STRIDE=8  AB_MAX_TASKS=0(=all)
#   SKIP_DATA_REGEN=0  SKIP_AB=0  BRUTE_TIMEOUT=120
set -euo pipefail
cd "$(dirname "$0")"

: "${MAX_ROUNDS:=10}"
: "${EXH_RADIUS:=3}"
: "${NEG_STRIDE:=8}"
: "${AB_MAX_TASKS:=0}"
: "${SKIP_DATA_REGEN:=0}"
: "${SKIP_AB:=0}"
: "${BRUTE_TIMEOUT:=120}"

mkdir -p logs kill_report checkpoints

echo "== [1/7] deps + build =="
python3 -c 'import torch; print("torch", torch.__version__, "cuda:", torch.cuda.is_available())'
python3 -c 'import tensorboard' 2>/dev/null || pip install --user tensorboard
make all libpca.so
make test

echo "== [2/7] unit tests: exhaustive leaf completion =="
python3 python/test_exhaustive.py

echo "== [3/7] regenerate exhaustive training data (states-only) =="
# edges_*.bin are 83% of bytes and read by zero python code — delete them.
if [ "$SKIP_DATA_REGEN" != 1 ]; then
    ./tools/gen_dataset.sh data/synthetic data/train 6 "$NEG_STRIDE" 30 \
        2>&1 | tee logs/gen_train.log
    rm -f data/train/edges_*.bin
    ./tools/gen_dataset.sh data/synthetic_deep data/train_deep 8 "$NEG_STRIDE" 120 \
        2>&1 | tee logs/gen_train_deep.log
    rm -f data/train_deep/edges_*.bin
fi

echo "== [4/7] round-0 training (skipped if checkpoints/value_v2.pt exists) =="
CKPT=checkpoints/value_v2.pt
if [ ! -f "$CKPT" ]; then
    python3 python/build_manifest.py 2>&1 | tee logs/manifest0.log
    python3 python/train.py \
        --epochs 10 --batch-size 2048 --lr 3e-4 --pos-weight 50 --workers 4 \
        --hidden-dim 256 --trunk-dim 128 \
        --save "$CKPT" --run-name round0_h256_e10 \
        2>&1 | tee logs/train0.log
fi
[ -f "$CKPT" ] || { echo "FATAL: round-0 training produced no checkpoint"; exit 1; }

echo "== [5/7] A/B control: R=0 vs R=$EXH_RADIUS, same checkpoint (issue #7 test plan) =="
if [ "$SKIP_AB" != 1 ]; then
    AB_ARGS=()
    [ "$AB_MAX_TASKS" != 0 ] && AB_ARGS=(--max-tasks "$AB_MAX_TASKS")
    python3 python/bootstrap.py --task-dir data/synthetic_deep \
        --output-dir data/ab_r0 --checkpoint "$CKPT" \
        --budget 500 --max-depth 8 --neg-stride "$NEG_STRIDE" \
        --exhaustive-radius 0 "${AB_ARGS[@]}" \
        2>&1 | tee logs/ab_r0.log
    python3 python/bootstrap.py --task-dir data/synthetic_deep \
        --output-dir data/ab_r3 --checkpoint "$CKPT" \
        --budget 500 --max-depth 8 --neg-stride "$NEG_STRIDE" \
        --exhaustive-radius "$EXH_RADIUS" "${AB_ARGS[@]}" \
        2>&1 | tee logs/ab_r3.log
    python3 - <<'PYEOF'
import glob, json, os
def summarize(d):
    depths, lens = {}, []
    for f in glob.glob(os.path.join(d, 'states_*.bin')):
        if os.path.getsize(f) < 576: continue
        rec = open(f, 'rb').read(576)
        od = rec[22] + rec[23]
        depths[od] = depths.get(od, 0) + 1
    return {'solved': sum(depths.values()), 'depth_dist': depths}
out = {'R0': summarize('data/ab_r0'), 'R3': summarize('data/ab_r3')}
json.dump(out, open('kill_report/ab_summary.json', 'w'), indent=2, sort_keys=True)
print(json.dumps(out, indent=2, sort_keys=True))
PYEOF
    # A/B arms must not leak into loop training data
    rm -rf data/ab_r0 data/ab_r3
fi

echo "== [6/7] bootstrap loop to convergence (max $MAX_ROUNDS rounds, R=$EXH_RADIUS) =="
python3 python/run_bootstrap.py \
    --task-dir data/synthetic_deep \
    --max-rounds "$MAX_ROUNDS" \
    --initial-budget 500 --initial-hidden 256 \
    --exhaustive-radius "$EXH_RADIUS" \
    2>&1 | tee logs/loop.log

echo "== [7/7] brute-force cross-check on depth>=6 solves =="
python3 - "$BRUTE_TIMEOUT" <<'PYEOF'
import glob, json, os, subprocess, sys
brute_timeout = int(sys.argv[1])
boot = 'data/train_boot'
results = []
for f in sorted(glob.glob(os.path.join(boot, 'states_*.bin'))):
    if os.path.getsize(f) < 576:
        continue
    rec = open(f, 'rb').read(576)
    od = rec[22] + rec[23]  # depth + budget_left = optimal depth found
    if od < 6:
        continue
    name = os.path.basename(f)[len('states_'):-len('.bin')]
    task = os.path.join('data', 'synthetic_deep', name + '.json')
    if not os.path.exists(task):
        results.append({'task': name, 'depth': od, 'brute': 'missing-json'})
        continue
    try:
        p = subprocess.run(['./enumerate', task, '-d', '8', '--no-branch'],
                           capture_output=True, timeout=brute_timeout)
        brute = 'solved' if p.returncode == 0 else 'failed'
    except subprocess.TimeoutExpired:
        brute = 'timeout'
    results.append({'task': name, 'depth': od, 'brute': brute})
beyond = [r for r in results if r['brute'] in ('timeout', 'failed')]
report = {
    'brute_timeout_s': brute_timeout,
    'depth6plus_solved_by_loop': len(results),
    'beyond_brute_force': len(beyond),
    'beyond_tasks': beyond,
    'all': results,
}
json.dump(report, open('kill_report/crosscheck.json', 'w'), indent=2)
print(f"depth>=6 tasks solved by loop:        {len(results)}")
print(f"  ... that {brute_timeout}s brute force CANNOT solve: {len(beyond)}")
print()
if beyond:
    print("VERDICT: KEEP — model-guided search reached programs brute force cannot.")
    print("(Check kill_report/ + STATUS.md decision tree for strong-KEEP criteria.)")
else:
    print("VERDICT: KILL — every 'frontier' solve is brute-force reachable.")
    print("(Per pre-registered rule; see STATUS.md for the tombstone protocol.)")
PYEOF

echo
echo "Reports: kill_report/ab_summary.json, kill_report/crosscheck.json, logs/"
