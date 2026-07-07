#!/usr/bin/env python3
"""
bootstrap.py — Recursive self-improvement loop for PCA Phase 3.

Uses the trained value network to guide search on unsolved tasks,
records solutions as training data, retrains, repeats.

Usage:
  # One iteration: search → record → retrain
  python bootstrap.py --task-dir data/synthetic_deep --output-dir data/train_boot \
                      --checkpoint checkpoints/value_v1.pt --budget 1000

  # Full loop (N iterations)
  python bootstrap.py --task-dir data/synthetic_deep --output-dir data/train_boot \
                      --checkpoint checkpoints/value_v1.pt --budget 1000 --iterations 5
"""

import argparse
import ctypes
import heapq
import json
import os
import struct
import sys
import time
import glob
import numpy as np
import torch

from model import ValueNetwork
from dataset import bits_encode_array, load_task_io, RECORD_SIZE, OFF_REGS, OFF_FLAGS
from search_eval import (PCALib, SearchState, FeatureExtractor,
                         MAX_TESTS, PCA_NUM_REGS)


# ── Recording search ────────────────────────────────────────────────────────

def search_and_record(lib, ctx, root, max_depth, budget, task_id,
                      model, feat_extractor, device, neg_stride=8,
                      r_exhaustive=0):
    """
    Model-guided best-first search that records all visited states.

    When r_exhaustive > 0, nodes within r_exhaustive of max_depth are
    handed to brute-force IDDFS leaf completion (issue #7) instead of
    model-guided expansion.

    Returns:
        solved: bool
        solution_depth: int or None
        states: list of (SearchState, can_finish, remaining, parent_action)
        stats: dict of exhaustive-completion instrumentation
    """
    # Track parent relationships for backtracing optimal path
    # Each entry: (state, parent_idx, action_taken)
    nodes = [(root, -1, 0xFFFF)]  # root has no parent
    heap = [(0.0, 0, 0)]  # (priority, counter, node_idx)
    counter = 1
    expansions = 0
    # Depth-dominance-aware OEP dedup: oep_hash -> shallowest depth expanded.
    # Only skip if a same-or-shallower visit exists (a shallower state has
    # more remaining budget and must not be blocked by a deeper failure).
    visited_oep = {}
    solution_node = None
    stats = {'exhaustive_calls': 0, 'exhaustive_states_explored': 0,
             'exhaustive_time_ms': 0.0, 'exhaustive_solved': 0}

    while heap and expansions < budget:
        _, _, node_idx = heapq.heappop(heap)
        state, _, _ = nodes[node_idx]

        oep = lib.oep_hash(ctx, state)
        seen_depth = visited_oep.get(oep)
        if seen_depth is not None and seen_depth <= state.depth:
            continue
        visited_oep[oep] = state.depth

        expansions += 1

        if state.depth >= max_depth:
            continue

        # Exhaustive leaf completion (issue #7): within R of max_depth,
        # brute force finishes the job instead of model-guided expansion.
        if r_exhaustive > 0 and state.depth >= max_depth - r_exhaustive:
            t0 = time.time()
            found, sol_len, sol_insns = lib.exhaustive_complete(
                ctx, state, max_depth)
            stats['exhaustive_calls'] += 1
            stats['exhaustive_states_explored'] += \
                lib.exhaustive_states_explored()
            stats['exhaustive_time_ms'] += (time.time() - t0) * 1000.0
            if found:
                stats['exhaustive_solved'] = 1
                # Replay verified suffix via lib.step() into nodes
                cur = state
                parent_idx = node_idx
                for insn in sol_insns:
                    child = lib.step(ctx, cur, insn)
                    assert child is not None, \
                        "exhaustive completion replay step failed"
                    idx = len(nodes)
                    nodes.append((child, parent_idx, insn))
                    parent_idx = idx
                    cur = child
                solution_node = parent_idx
                break
            else:
                # Heuristic (not proven) dead end: OEP hash ignores
                # liveness. Already counted against budget above.
                continue

        candidates = lib.gen_candidates(state)

        children = []
        child_actions = []
        for insn in candidates:
            child = lib.step(ctx, state, insn)
            if child is None:
                continue

            if lib.leaf_check(ctx, child):
                # Found a solution! Record the solving child
                child_idx = len(nodes)
                nodes.append((child, node_idx, insn))
                solution_node = child_idx
                break

            if child.depth < max_depth:
                children.append(child)
                child_actions.append(insn)

        if solution_node is not None:
            break

        if not children:
            continue

        # Score children with model
        batch = feat_extractor.extract_batch(children)
        with torch.no_grad():
            reach_logit, _, _ = model(
                batch['per_test'].to(device),
                batch['live'].to(device),
                batch['depth'].to(device),
            )
        scores = torch.sigmoid(reach_logit.squeeze(1)).cpu().numpy()

        for child, action, score in zip(children, child_actions, scores):
            idx = len(nodes)
            nodes.append((child, node_idx, action))
            heapq.heappush(heap, (-float(score), counter, idx))
            counter += 1

    if solution_node is None:
        return False, None, [], stats

    # Backtrace optimal path
    optimal_set = set()
    idx = solution_node
    solution_depth = nodes[idx][0].depth
    while idx >= 0:
        optimal_set.add(idx)
        _, parent_idx, _ = nodes[idx]
        idx = parent_idx

    # Build labeled records
    # NOTE (issue #7 v1): ancestors above an exhaustive-completion trigger
    # state carry heuristic labels (first path found), same quality as
    # current bootstrap labels; the exhaustive suffix itself is verified.
    records = []
    neg_count = 0
    for i, (state, parent_idx, action) in enumerate(nodes):
        if i in optimal_set:
            can_finish = 1
            remaining = solution_depth - state.depth
            records.append((state, can_finish, remaining, action))
        else:
            neg_count += 1
            if neg_count % neg_stride == 0:
                records.append((state, 0, -1, action))

    return True, solution_depth, records, stats


def write_records(records, task_id, max_depth, ctx, lib, output_path):
    """Write state records in the binary format matching dataset_state_record_t."""
    with open(output_path, 'wb') as f:
        for state, can_finish, remaining, action in records:
            # Build 576-byte record
            buf = bytearray(RECORD_SIZE)

            # state_hash (uint64 at 0)
            key_hash = lib.lib.search_state_key_hash64(
                ctypes.byref(ctx), ctypes.byref(state))
            struct.pack_into('<Q', buf, 0, key_hash & 0xFFFFFFFFFFFFFFFF)

            # oep_hash (uint64 at 8)
            oep_hash = lib.oep_hash(ctx, state)
            struct.pack_into('<Q', buf, 8, oep_hash)

            # task_id (uint32 at 16)
            struct.pack_into('<I', buf, 16, task_id)

            # parent_action (uint16 at 20)
            struct.pack_into('<H', buf, 20, action & 0xFFFF)

            # depth (uint8 at 22)
            buf[22] = state.depth

            # budget_left (uint8 at 23)
            buf[23] = max(0, max_depth - state.depth)

            # live (uint8 at 24)
            buf[24] = state.live

            # can_finish_optimally (uint8 at 25)
            buf[25] = can_finish

            # remaining_steps (int8 at 26)
            buf[26] = remaining & 0xFF  # int8 as unsigned byte

            # regs[32][8] (uint16 at 30)
            for t in range(MAX_TESTS):
                for r in range(PCA_NUM_REGS):
                    struct.pack_into('<H', buf, 30 + (t * PCA_NUM_REGS + r) * 2,
                                    state.regs[t][r])

            # flags[32] (uint8 at 542)
            for t in range(MAX_TESTS):
                buf[542 + t] = state.flags[t]

            f.write(buf)

    return len(records)


# ── Bootstrap iteration ─────────────────────────────────────────────────────

def run_bootstrap_iteration(lib, task_files, model, device, num_tests,
                            output_dir, max_depth, budget, neg_stride,
                            task_id_offset=0, r_exhaustive=0):
    """Run model-guided search on a list of tasks, record solutions."""
    os.makedirs(output_dir, exist_ok=True)

    solved = 0
    failed = 0
    total_records = 0
    depth_dist = {}
    exh_totals = {'exhaustive_calls': 0, 'exhaustive_states_explored': 0,
                  'exhaustive_time_ms': 0.0, 'exhaustive_solved': 0}

    for i, task_path in enumerate(task_files):
        task_id = task_id_offset + i
        name = os.path.basename(task_path).replace('.json', '')

        # Skip already-solved tasks
        existing = os.path.join(output_dir, f'states_{name}.bin')
        if os.path.exists(existing) and os.path.getsize(existing) > 0:
            solved += 1
            continue

        spec = lib.load_task(task_path)
        ctx = lib.init_ctx(spec)
        root = lib.init_root(ctx)

        if lib.leaf_check(ctx, root):
            solved += 1
            continue

        task_io = load_task_io(task_path)
        feat_extractor = FeatureExtractor(task_io, num_tests=num_tests)

        t0 = time.time()
        ok, sol_depth, records, stats = search_and_record(
            lib, ctx, root, max_depth, budget, task_id,
            model, feat_extractor, device, neg_stride=neg_stride,
            r_exhaustive=r_exhaustive)
        dt = time.time() - t0

        for k in exh_totals:
            exh_totals[k] += stats[k]

        exh_str = ""
        if stats['exhaustive_calls']:
            exh_str = (f" exh:{stats['exhaustive_calls']}c/"
                       f"{stats['exhaustive_time_ms']:.0f}ms")

        if ok:
            states_path = os.path.join(output_dir, f'states_{name}.bin')
            n = write_records(records, task_id, sol_depth, ctx, lib, states_path)
            total_records += n
            solved += 1
            depth_dist[sol_depth] = depth_dist.get(sol_depth, 0) + 1
            via = "exh" if stats['exhaustive_solved'] else "model"
            status = f"d*={sol_depth} {n:5d} records ({via})"
        else:
            failed += 1
            status = "FAIL"

        print(f"  [{i+1:3d}/{len(task_files)}] {name[:35]:35s}  "
              f"{status:25s}  [{dt:.1f}s]{exh_str}")

    return {
        'solved': solved,
        'failed': failed,
        'total_records': total_records,
        'depth_dist': depth_dist,
        'exhaustive': exh_totals,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Bootstrap: model-guided search → record → retrain')
    parser.add_argument('--base-dir', default=None)
    parser.add_argument('--task-dir', required=True,
                        help='Directory of task JSONs to attempt')
    parser.add_argument('--output-dir', default='data/train_boot',
                        help='Where to write recorded state files')
    parser.add_argument('--checkpoint', default='checkpoints/value_v1.pt')
    parser.add_argument('--lib', default='libpca.so')
    parser.add_argument('--budget', type=int, default=1000,
                        help='Max node expansions per task')
    parser.add_argument('--max-depth', type=int, default=8)
    parser.add_argument('--num-tests', type=int, default=10)
    parser.add_argument('--neg-stride', type=int, default=8,
                        help='Record every Nth negative state')
    parser.add_argument('--exhaustive-radius', type=int, default=3,
                        help='Brute-force leaf completion within R of '
                             'max_depth (0 = disabled)')
    parser.add_argument('--max-tasks', type=int, default=None)
    parser.add_argument('--task-id-offset', type=int, default=0,
                        help='Starting task_id for recorded states')
    args = parser.parse_args()

    if args.base_dir is None:
        args.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load library
    lib_path = os.path.join(args.base_dir, args.lib)
    lib = PCALib(lib_path)

    # Load model
    ckpt_path = os.path.join(args.base_dir, args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get('args', {})
    model = ValueNetwork(
        max_depth=args.max_depth,
        num_tests=args.num_tests,
        hidden_dim=ckpt_args.get('hidden_dim', 256),
        trunk_dim=ckpt_args.get('trunk_dim', 128),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} params from {ckpt_path}")

    # Load task files
    task_dir = os.path.join(args.base_dir, args.task_dir)
    task_files = sorted(glob.glob(os.path.join(task_dir, '*.json')))
    if args.max_tasks:
        task_files = task_files[:args.max_tasks]
    print(f"Tasks: {len(task_files)} from {task_dir}")

    output_dir = os.path.join(args.base_dir, args.output_dir)
    print(f"Output: {output_dir}")
    print(f"Budget: {args.budget} expansions, max_depth={args.max_depth}, "
          f"exhaustive_radius={args.exhaustive_radius}")
    print("-" * 70)

    result = run_bootstrap_iteration(
        lib, task_files, model, device, args.num_tests,
        output_dir, args.max_depth, args.budget, args.neg_stride,
        task_id_offset=args.task_id_offset,
        r_exhaustive=args.exhaustive_radius)

    print(f"\n{'='*70}")
    print(f"Solved: {result['solved']}, Failed: {result['failed']}")
    print(f"Total records: {result['total_records']:,}")
    if result['depth_dist']:
        print("Depth distribution:")
        for d in sorted(result['depth_dist']):
            print(f"  depth {d}: {result['depth_dist'][d]} tasks")
    exh = result['exhaustive']
    if exh['exhaustive_calls']:
        print(f"Exhaustive completion: {exh['exhaustive_calls']:,} calls, "
              f"{exh['exhaustive_states_explored']:,} states, "
              f"{exh['exhaustive_time_ms']/1000.0:.1f}s total, "
              f"{exh['exhaustive_solved']} tasks solved via completion")


if __name__ == '__main__':
    main()
