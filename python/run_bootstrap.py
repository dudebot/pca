#!/usr/bin/env python3
"""
run_bootstrap.py — Autonomous recursive self-improvement loop.

Strategy:
  1. Search unsolved tasks with current model → record solutions
  2. If solve rate drops → retrain on combined data
  3. If retrain doesn't help → train longer (more epochs)
  4. If still stuck → scale up model (bigger hidden dims)
  5. Repeat until all tasks solved or budget exhausted

Usage:
  python run_bootstrap.py --task-dir data/synthetic_deep --max-rounds 10
"""

import argparse
import json
import glob
import os
import subprocess
import sys
import time


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
    checkpoint = os.path.join(args.base_dir, 'checkpoints', 'value_v1.pt')
    boot_dir = os.path.join(args.base_dir, 'data', 'train_boot')
    prev_solved = 0

    print(f"Bootstrap loop: {total_tasks} tasks, max {args.max_rounds} rounds")
    print(f"Starting: budget={budget}, epochs={epochs}, hidden={hidden_dim}")

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

        if cur_solved >= total_tasks:
            print("\n  ALL TASKS SOLVED!")
            break

        # ── Decide strategy ──
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

        # ── Step 3: Retrain on combined data ──
        new_ckpt = os.path.join(args.base_dir, 'checkpoints',
                                f'value_round{round_num}.pt')

        run_cmd([
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
        ], f"Round {round_num}: Retrain (epochs={epochs}, hidden={hidden_dim})")

        if os.path.exists(new_ckpt):
            checkpoint = new_ckpt
            print(f"  New checkpoint: {checkpoint}")
        else:
            print(f"  WARNING: Training failed, keeping old checkpoint")

        prev_solved = cur_solved

    # ── Final report ──
    print(f"\n{'='*70}")
    print(f"Bootstrap complete after {round_num} rounds")
    print(f"Final: {cur_solved}/{total_tasks} tasks solved")
    print(f"Final checkpoint: {checkpoint}")
    print(f"Total bootstrap records: {count_total_records(boot_dir):,}")


if __name__ == '__main__':
    main()
