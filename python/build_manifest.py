#!/usr/bin/env python3
"""
build_manifest.py — Create data/manifest.json from all task/data sources.

Scans multiple (task_dir, data_dir) pairs. Each data_dir uses its own
0-based task_ids (matching gen_dataset.sh output). The manifest stores
the actual state file path so the loader doesn't need to reconstruct it.
"""

import json
import glob
import os
import random
from collections import defaultdict

RECORD_SIZE = 576


def get_optimal_depth(states_path):
    """Read first record, return optimal_depth = depth + budget_left."""
    if os.path.getsize(states_path) < RECORD_SIZE:
        return None
    with open(states_path, 'rb') as f:
        rec = f.read(RECORD_SIZE)
    return rec[22] + rec[23]


def count_records(states_path):
    return os.path.getsize(states_path) // RECORD_SIZE


def stratified_split(items, key_fn, train=0.70, val=0.15, test=0.15, seed=42):
    rng = random.Random(seed)
    strata = defaultdict(list)
    for item in items:
        strata[key_fn(item)].append(item)

    splits = {'train': [], 'val': [], 'test': []}
    for key in sorted(strata.keys()):
        group = strata[key]
        rng.shuffle(group)
        n = len(group)
        if n < 3:
            splits['train'].extend(group)
        else:
            n_val = max(1, round(n * val))
            n_test = max(1, round(n * test))
            n_train = n - n_val - n_test
            splits['train'].extend(group[:n_train])
            splits['val'].extend(group[n_train:n_train + n_val])
            splits['test'].extend(group[n_train + n_val:])
    return splits


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(base_dir, 'data', 'manifest.json')

    # All (task_dir, data_dir) sources. Each data_dir has 0-based task IDs.
    sources = [
        ('data/synthetic', 'data/train'),
        ('data/synthetic_deep', 'data/train_deep'),
        ('data/synthetic_deep', 'data/train_boot'),
    ]

    tasks = []
    seen_names = set()
    global_id = 0

    for task_rel, data_rel in sources:
        task_dir = os.path.join(base_dir, task_rel)
        data_dir = os.path.join(base_dir, data_rel)
        if not os.path.isdir(task_dir):
            continue

        task_files = sorted(glob.glob(os.path.join(task_dir, '*.json')))

        for local_id, task_path in enumerate(task_files):
            name = os.path.basename(task_path).replace('.json', '')

            # Deduplicate: if this task was already added from another source,
            # check if THIS source has a deeper/better solution
            if name in seen_names:
                # Find existing entry and check if new source has data
                if not os.path.isdir(data_dir):
                    continue
                states_path = os.path.join(data_dir, f'states_{local_id:06d}.bin')
                if not os.path.exists(states_path) or os.path.getsize(states_path) < RECORD_SIZE:
                    continue
                # This source has data for a duplicate task — update existing entry
                # if the new data has more records (deeper exploration)
                new_depth = get_optimal_depth(states_path)
                new_records = count_records(states_path)
                for t in tasks:
                    if t['name'] == name:
                        if new_records > t['n_records']:
                            t['optimal_depth'] = new_depth
                            t['n_records'] = new_records
                            t['states_path'] = os.path.relpath(states_path, base_dir)
                            t['solved'] = True
                        break
                continue

            seen_names.add(name)

            with open(task_path) as f:
                spec = json.load(f)

            # Check for state data in this source's data_dir
            solved = False
            optimal_depth = None
            n_records = 0
            states_rel = None

            if os.path.isdir(data_dir):
                states_path = os.path.join(data_dir, f'states_{local_id:06d}.bin')
                if os.path.exists(states_path) and os.path.getsize(states_path) >= RECORD_SIZE:
                    optimal_depth = get_optimal_depth(states_path)
                    n_records = count_records(states_path)
                    solved = True
                    states_rel = os.path.relpath(states_path, base_dir)

            tasks.append({
                'task_id': global_id,
                'name': name,
                'path': os.path.relpath(task_path, base_dir),
                'num_inputs': len(spec['input_ports']),
                'num_tests': len(spec['tests']),
                'solved': solved,
                'optimal_depth': optimal_depth,
                'n_records': n_records,
                'states_path': states_rel,
            })
            global_id += 1

    solved_tasks = [t for t in tasks if t['solved']]
    print(f"Found {len(tasks)} unique tasks, {len(solved_tasks)} solved")

    # Stratified split
    splits = stratified_split(
        solved_tasks,
        key_fn=lambda t: (t['num_inputs'], t['optimal_depth']),
    )
    split_map = {}
    for split_name, split_tasks in splits.items():
        for t in split_tasks:
            split_map[t['task_id']] = split_name
    for t in tasks:
        t['split'] = split_map.get(t['task_id'], 'unsolved')

    # Summary
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

    manifest = {
        'version': 2,
        'total_tasks': len(tasks),
        'solved_tasks': len(solved_tasks),
        'splits': {s: split_counts[s] for s in ['train', 'val', 'test']},
        'tasks': tasks,
    }
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {out_path}")


if __name__ == '__main__':
    main()
