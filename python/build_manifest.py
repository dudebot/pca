#!/usr/bin/env python3
"""
build_manifest.py — Create data/manifest.json from synthetic tasks + training data.

Freezes task_id → task_path → optimal_depth → num_inputs → train/val/test split.
Extracts optimal_depth from binary state records.
Stratifies split by (num_inputs, optimal_depth).
"""

import json
import glob
import os
import struct
import random
import sys
from collections import defaultdict

RECORD_SIZE = 576

def get_optimal_depth(states_path):
    """Read first record from states file, return optimal_depth = depth + budget_left."""
    fsize = os.path.getsize(states_path)
    if fsize < RECORD_SIZE:
        return None
    with open(states_path, 'rb') as f:
        rec = f.read(RECORD_SIZE)
    # Header: <QQIHBBBBbBB = state_hash(8), oep_hash(8), task_id(4),
    #          parent_action(2), depth(1), budget_left(1), live(1),
    #          can_finish(1), remaining(1), pad(2)
    # depth at offset 22, budget_left at offset 23
    depth = rec[22]
    budget_left = rec[23]
    return depth + budget_left


def count_records(states_path):
    """Count state records in a file."""
    return os.path.getsize(states_path) // RECORD_SIZE


def stratified_split(items, key_fn, train=0.70, val=0.15, test=0.15, seed=42):
    """Split items into train/val/test stratified by key_fn."""
    rng = random.Random(seed)

    # Group by stratum
    strata = defaultdict(list)
    for item in items:
        strata[key_fn(item)].append(item)

    splits = {'train': [], 'val': [], 'test': []}

    for key in sorted(strata.keys()):
        group = strata[key]
        rng.shuffle(group)
        n = len(group)
        n_val = max(1, round(n * val)) if n >= 3 else 0
        n_test = max(1, round(n * test)) if n >= 3 else 0
        n_train = n - n_val - n_test

        # If group too small, put everything in train
        if n < 3:
            splits['train'].extend(group)
        else:
            splits['train'].extend(group[:n_train])
            splits['val'].extend(group[n_train:n_train + n_val])
            splits['test'].extend(group[n_train + n_val:])

    return splits


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    task_dir = os.path.join(base_dir, 'data', 'synthetic')
    train_dir = os.path.join(base_dir, 'data', 'train')
    out_path = os.path.join(base_dir, 'data', 'manifest.json')

    # Get sorted task list (same order as gen_dataset.sh)
    task_files = sorted(glob.glob(os.path.join(task_dir, '*.json')))
    print(f"Found {len(task_files)} task files")

    tasks = []
    solved_tasks = []

    for task_id, task_path in enumerate(task_files):
        with open(task_path) as f:
            spec = json.load(f)

        rel_path = os.path.relpath(task_path, base_dir)
        num_inputs = len(spec['input_ports'])
        num_tests = len(spec['tests'])

        # Check for training data
        states_path = os.path.join(train_dir, f'states_{task_id:06d}.bin')
        if os.path.exists(states_path) and os.path.getsize(states_path) >= RECORD_SIZE:
            optimal_depth = get_optimal_depth(states_path)
            n_records = count_records(states_path)
            solved = True
        else:
            optimal_depth = None
            n_records = 0
            solved = False

        entry = {
            'task_id': task_id,
            'name': spec['name'],
            'path': rel_path,
            'num_inputs': num_inputs,
            'num_tests': num_tests,
            'solved': solved,
            'optimal_depth': optimal_depth,
            'n_records': n_records,
        }
        tasks.append(entry)

        if solved:
            solved_tasks.append(entry)

    print(f"  {len(solved_tasks)} solved, {len(tasks) - len(solved_tasks)} unsolved")

    # Stratified split of solved tasks by (num_inputs, optimal_depth)
    splits = stratified_split(
        solved_tasks,
        key_fn=lambda t: (t['num_inputs'], t['optimal_depth']),
        train=0.70, val=0.15, test=0.15,
        seed=42,
    )

    # Assign splits
    split_map = {}
    for split_name, split_tasks in splits.items():
        for t in split_tasks:
            split_map[t['task_id']] = split_name

    for t in tasks:
        t['split'] = split_map.get(t['task_id'], 'unsolved')

    # Summary stats
    depth_dist = defaultdict(int)
    split_counts = defaultdict(int)
    split_records = defaultdict(int)
    for t in tasks:
        split_counts[t['split']] += 1
        split_records[t['split']] += t['n_records']
        if t['optimal_depth'] is not None:
            depth_dist[t['optimal_depth']] += 1

    print(f"\nSplit distribution:")
    for s in ['train', 'val', 'test', 'unsolved']:
        print(f"  {s:>8s}: {split_counts[s]:4d} tasks, "
              f"{split_records[s]:>10,d} records")

    print(f"\nOptimal depth distribution:")
    for d in sorted(depth_dist.keys()):
        print(f"  depth {d}: {depth_dist[d]} tasks")

    # Write manifest
    manifest = {
        'version': 1,
        'total_tasks': len(tasks),
        'solved_tasks': len(solved_tasks),
        'splits': {
            'train': split_counts['train'],
            'val': split_counts['val'],
            'test': split_counts['test'],
        },
        'tasks': tasks,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to {out_path}")


if __name__ == '__main__':
    main()
