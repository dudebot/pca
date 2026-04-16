#!/usr/bin/env python3
"""
run_bootstrap.py — Autonomous recursive self-improvement loop.

Trains a value network from scratch, then uses it to solve progressively
harder tasks, recording solutions as new training data. The depth frontier
advances each cycle as the model learns from its own discoveries.

Loop:
  1. Train initial model from scratch on existing exhaustive data (if no checkpoint)
  2. Model-guided best-first search on unsolved tasks → record solutions
  3. Analyze depth frontier — what's the deepest depth being solved?
  4. Rebuild manifest with all data sources (exhaustive + bootstrap)
  5. Retrain on combined data

Adaptive escalation (when no new tasks solved):
  - First: increase search budget (500 → 1000 → ... → 10000 expansions)
  - Then: increase training epochs (10 → 20 → ... → 50)
  - Finally: scale up model (hidden 256 → 512 → ... → 1024)

Adaptive task generation (when unsolved tasks run low):
  - Generates new synthetic tasks at the depth frontier
  - Kernel length targets frontier_depth to frontier_depth+3
  - Lower depths remain valuable — shallow data teaches "promising early moves"
    that transfer upward (a depth-5 solution starts at depth 0 like everything else)

Usage:
  python run_bootstrap.py --task-dir data/synthetic_deep --max-rounds 10 --initial-hidden 3072
"""

import argparse
import hashlib
import json
import glob
import os
import subprocess
import sys
import time
from torch.utils.tensorboard import SummaryWriter


def analyze_frontier(output_dir):
    """Analyze solved tasks by optimal depth. Returns {depth: count}."""
    depths = {}
    for f in glob.glob(os.path.join(output_dir, 'states_*.bin')):
        if os.path.getsize(f) < 576:
            continue
        with open(f, 'rb') as fh:
            rec = fh.read(576)
        d = rec[22] + rec[23]  # depth + budget_left = optimal_depth
        depths[d] = depths.get(d, 0) + 1
    return depths


def count_solved(output_dir):
    """Count solved tasks by checking for non-empty state files."""
    files = glob.glob(os.path.join(output_dir, 'states_*.bin'))
    return sum(1 for f in files if os.path.getsize(f) > 0)


def count_total_records(output_dir):
    """Count total records across all state files."""
    total = 0
    for f in glob.glob(os.path.join(output_dir, 'states_*.bin')):
        total += os.path.getsize(f) // 576
    return total


def read_manifest_depths(base_dir):
    """Read depth distribution from manifest."""
    manifest_path = os.path.join(base_dir, 'data', 'manifest.json')
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path) as f:
        manifest = json.load(f)
    depths = {}
    for t in manifest['tasks']:
        d = t.get('optimal_depth')
        if d is not None:
            depths[d] = depths.get(d, 0) + 1
    return depths


def load_splits(base_dir):
    """Load data/splits.json, or None if it doesn't exist."""
    path = os.path.join(base_dir, 'data', 'splits.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_splits(base_dir, splits_data):
    """Write data/splits.json."""
    path = os.path.join(base_dir, 'data', 'splits.json')
    with open(path, 'w') as f:
        json.dump(splits_data, f, indent=2)


def seed_splits(base_dir):
    """Create initial splits from manifest — 85% training, 15% frontier.
    Uses hash-based ordering for determinism regardless of file order."""
    manifest_path = os.path.join(base_dir, 'data', 'manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)
    solved = [t for t in manifest['tasks'] if t['solved']]
    # Sort by name hash — deterministic and order-independent
    solved.sort(key=lambda t: hashlib.sha256(t['name'].encode()).hexdigest())
    n_frontier = max(1, len(solved) // 7)  # ~15%
    tasks = {}
    for t in solved[:-n_frontier]:
        tasks[t['name']] = 'training'
    for t in solved[-n_frontier:]:
        tasks[t['name']] = 'frontier'
    return {'round': 0, 'tasks': tasks}


def graduate_frontier(splits_data):
    """Move all frontier tasks → training. Returns count graduated."""
    n = 0
    for name in list(splits_data['tasks']):
        if splits_data['tasks'][name] == 'frontier':
            splits_data['tasks'][name] = 'training'
            n += 1
    return n


def tag_new_solutions(splits_data, task_dir, boot_dir):
    """Tag newly solved bootstrap tasks as frontier. Returns count tagged."""
    task_files = sorted(glob.glob(os.path.join(task_dir, '*.json')))
    n = 0
    for tf in task_files:
        name = os.path.basename(tf).replace('.json', '')
        if name in splits_data['tasks']:
            continue
        bin_path = os.path.join(boot_dir, f'states_{name}.bin')
        if os.path.exists(bin_path) and os.path.getsize(bin_path) > 0:
            splits_data['tasks'][name] = 'frontier'
            n += 1
    return n


def run_cmd(cmd, desc):
    """Run a command with logging."""
    print(f"\n{'='*70}")
    print(f"  {desc}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'='*70}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    dt = time.time() - t0
    print(f"  [{dt:.0f}s, exit={result.returncode}]")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description='Autonomous bootstrap loop: search → record → retrain → repeat')
    parser.add_argument('--base-dir', default=None)
    parser.add_argument('--task-dir', default='data/synthetic_deep',
                        help='Tasks to solve')
    parser.add_argument('--max-rounds', type=int, default=10,
                        help='Maximum bootstrap rounds')
    parser.add_argument('--initial-budget', type=int, default=500,
                        help='Initial search budget per task')
    parser.add_argument('--max-budget', type=int, default=10000,
                        help='Maximum search budget per task')
    parser.add_argument('--initial-epochs', type=int, default=10,
                        help='Initial training epochs')
    parser.add_argument('--max-epochs', type=int, default=50,
                        help='Maximum training epochs')
    parser.add_argument('--initial-hidden', type=int, default=256,
                        help='Initial hidden dim')
    parser.add_argument('--max-hidden', type=int, default=1024,
                        help='Maximum hidden dim')
    parser.add_argument('--checkpoint', default=None,
                        help='Starting checkpoint (default: checkpoints/value_v2.pt)')
    parser.add_argument('--run-name', default=None,
                        help='TensorBoard run name (default: auto)')
    args = parser.parse_args()

    if args.base_dir is None:
        args.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    task_dir = os.path.join(args.base_dir, args.task_dir)
    total_tasks = len(glob.glob(os.path.join(task_dir, '*.json')))

    # State tracking
    budget = args.initial_budget
    epochs = args.initial_epochs
    hidden_dim = args.initial_hidden
    trunk_dim = hidden_dim // 2
    checkpoint = (args.checkpoint or
                  os.path.join(args.base_dir, 'checkpoints', 'value_v2.pt'))
    if not os.path.isabs(checkpoint):
        checkpoint = os.path.join(args.base_dir, checkpoint)
    boot_dir = os.path.join(args.base_dir, 'data', 'train_boot')
    prev_solved = 0

    # TensorBoard for round-level frontier metrics
    run_name = args.run_name or f'bootstrap_h{hidden_dim}'
    tb_dir = os.path.join(args.base_dir, 'runs', run_name)
    writer = SummaryWriter(tb_dir)

    print(f"Bootstrap loop: {total_tasks} tasks, max {args.max_rounds} rounds")
    print(f"Starting: budget={budget}, epochs={epochs}, hidden={hidden_dim}")
    print(f"TensorBoard: {tb_dir}")

    # ── Ensure splits.json exists ──
    splits_data = load_splits(args.base_dir)
    if splits_data is None:
        # Build manifest (old-style) so we can scan what's solved
        run_cmd([sys.executable, 'python/build_manifest.py'],
                "Build initial manifest")
        splits_data = seed_splits(args.base_dir)
        save_splits(args.base_dir, splits_data)
        n_t = sum(1 for v in splits_data['tasks'].values() if v == 'training')
        n_f = sum(1 for v in splits_data['tasks'].values() if v == 'frontier')
        print(f"  Seeded splits.json: {n_t} training, {n_f} frontier")

    # ── Round 0: Train initial model from scratch ──
    if not os.path.exists(checkpoint):
        # Rebuild manifest with splits.json
        run_cmd([sys.executable, 'python/build_manifest.py'],
                "Round 0: Build manifest with splits")

        run_cmd([
            sys.executable, 'python/train.py',
            '--epochs', str(epochs),
            '--batch-size', '2048',
            '--lr', '3e-4',
            '--pos-weight', '50',
            '--workers', '4',
            '--hidden-dim', str(hidden_dim),
            '--trunk-dim', str(trunk_dim),
            '--save', os.path.relpath(checkpoint, args.base_dir),
            '--run-name', f'round0_h{hidden_dim}_e{epochs}',
        ], f"Round 0: Train initial {hidden_dim}-dim model from scratch")

        if not os.path.exists(checkpoint):
            print("ERROR: Initial training failed. Exiting.")
            return

    for round_num in range(1, args.max_rounds + 1):
        print(f"\n{'#'*70}")
        print(f"  ROUND {round_num}")
        print(f"  budget={budget}, epochs={epochs}, "
              f"hidden={hidden_dim}, trunk={trunk_dim}")
        print(f"{'#'*70}")

        # ── Step 1: Search with current model ──
        run_cmd([
            sys.executable, 'python/bootstrap.py',
            '--task-dir', args.task_dir,
            '--output-dir', 'data/train_boot',
            '--checkpoint', os.path.relpath(checkpoint, args.base_dir),
            '--budget', str(budget),
            '--max-depth', '8',
            '--neg-stride', '8',
        ], f"Round {round_num}: Model-guided search (budget={budget})")

        cur_solved = count_solved(boot_dir)
        cur_records = count_total_records(boot_dir)
        new_solved = cur_solved - prev_solved

        print(f"\n  Solved: {cur_solved}/{total_tasks} "
              f"(+{new_solved} this round, {cur_records:,} records)")

        # ── Tag new solutions + conditionally graduate old frontier ──
        old_frontier = {name for name, s in splits_data['tasks'].items()
                        if s == 'frontier'}
        n_tagged = tag_new_solutions(splits_data, task_dir, boot_dir)

        if n_tagged > 0:
            # New frontier exists — graduate old frontier to training
            for name in old_frontier:
                splits_data['tasks'][name] = 'training'
            print(f"  Graduated {len(old_frontier)} → training, "
                  f"tagged {n_tagged} new frontier")
        elif old_frontier:
            print(f"  No new solutions, keeping {len(old_frontier)} "
                  f"frontier as val")

        splits_data['round'] = round_num
        save_splits(args.base_dir, splits_data)

        if cur_solved >= total_tasks:
            print(f"\n  All {total_tasks} tasks solved!")

        # ── Adaptive task generation ──
        # Analyze depth frontier: what's the deepest depth we're solving?
        boot_depths = analyze_frontier(boot_dir)
        if boot_depths:
            frontier = max(boot_depths.keys())
            frontier_rate = boot_depths[frontier]
            print(f"\n  Depth frontier: {boot_depths}")
            print(f"  Deepest solved: depth {frontier}")

            # Generate tasks at and just beyond the frontier
            # Kernel length ≈ optimal_depth + 1 (solver often finds shortcuts)
            gen_min = max(frontier, 4)
            gen_max = min(frontier + 3, 10)

            # Count how many unsolved tasks we have at frontier depths
            unsolved = total_tasks - cur_solved
            if unsolved < 100 or cur_solved >= total_tasks:
                n_new = 500
                print(f"  Generating {n_new} tasks (kernel {gen_min}-{gen_max}) "
                      f"to push past depth {frontier}")
                run_cmd([
                    os.path.join(args.base_dir, 'gen_tasks'),
                    task_dir,
                    '-n', str(n_new),
                    '-s', str(1337 + round_num * 1000),
                    '--min-len', str(gen_min),
                    '--max-len', str(gen_max),
                ], f"Generate frontier tasks (kernel {gen_min}-{gen_max})")
                total_tasks = len(glob.glob(os.path.join(task_dir, '*.json')))

        # ── Decide strategy ──
        dims_changed = False
        if new_solved == 0:
            # No progress — escalate
            if budget < args.max_budget:
                budget = min(budget * 2, args.max_budget)
                print(f"  No new solutions → increasing budget to {budget}")
            elif epochs < args.max_epochs:
                epochs = min(epochs + 10, args.max_epochs)
                print(f"  Budget maxed → increasing epochs to {epochs}")
            elif hidden_dim < args.max_hidden:
                hidden_dim = min(hidden_dim * 2, args.max_hidden)
                trunk_dim = hidden_dim // 2
                dims_changed = True
                print(f"  Epochs maxed → scaling model to hidden={hidden_dim}")
            else:
                print(f"  All strategies exhausted. Stopping.")
                break
        else:
            print(f"  Progress! {new_solved} new solutions.")

        # ── Step 2: Rebuild manifest to include bootstrap data ──
        run_cmd([
            sys.executable, 'python/build_manifest.py',
        ], f"Round {round_num}: Rebuild manifest")

        # ── Log depth distribution to TensorBoard ──
        all_depths = read_manifest_depths(args.base_dir)
        total_solved_tasks = sum(all_depths.values())
        if total_solved_tasks > 0:
            for d, count in sorted(all_depths.items()):
                writer.add_scalar(f'depth_fraction/depth_{d}',
                                  count / total_solved_tasks, round_num)
                writer.add_scalar(f'depth_count/depth_{d}', count, round_num)
        writer.add_scalar('frontier/solved_total', cur_solved, round_num)
        writer.add_scalar('frontier/solved_new', new_solved, round_num)
        writer.add_scalar('frontier/records_total', cur_records, round_num)
        writer.flush()

        # ── Step 3: Retrain on combined data (warm-start from previous) ──
        new_ckpt = os.path.join(args.base_dir, 'checkpoints',
                                f'value_round{round_num}.pt')

        train_cmd = [
            sys.executable, 'python/train.py',
            '--epochs', str(epochs),
            '--batch-size', '2048',
            '--lr', '3e-4',
            '--pos-weight', '50',
            '--workers', '4',
            '--hidden-dim', str(hidden_dim),
            '--trunk-dim', str(trunk_dim),
            '--save', os.path.relpath(new_ckpt, args.base_dir),
            '--run-name', f'round{round_num}_h{hidden_dim}_e{epochs}',
        ]
        if os.path.exists(checkpoint) and not dims_changed:
            train_cmd += ['--init-from', os.path.relpath(checkpoint, args.base_dir)]
        run_cmd(train_cmd,
                f"Round {round_num}: Retrain (epochs={epochs}, hidden={hidden_dim})")

        if os.path.exists(new_ckpt):
            checkpoint = new_ckpt
            print(f"  New checkpoint: {checkpoint}")
        else:
            print(f"  WARNING: Training failed, keeping old checkpoint")

        prev_solved = cur_solved

    # ── Final report ──
    writer.close()
    print(f"\n{'='*70}")
    print(f"Bootstrap complete after {round_num} rounds")
    print(f"Final: {cur_solved}/{total_tasks} tasks solved")
    print(f"Final checkpoint: {checkpoint}")
    print(f"Total bootstrap records: {count_total_records(boot_dir):,}")


if __name__ == '__main__':
    main()
