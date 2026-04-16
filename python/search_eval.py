#!/usr/bin/env python3
"""
search_eval.py — Best-first search evaluator for PCA Phase 2.

Compares model-guided search vs random baseline on held-out tasks.
Reports Success@{8,16,32,64} — did search find optimal program within K expansions?

Uses ctypes bindings to libpca.so for exact C search_state logic.
"""

import argparse
import ctypes
import heapq
import json
import os
import sys
import time
import numpy as np
import torch

from model import ValueNetwork
from dataset import bits_encode_array, load_task_io

# ── C struct definitions (must match search_state.h / spec.h / pca.h) ────────

MAX_TESTS = 32
PCA_NUM_REGS = 8
MAX_IO_PORTS = 8
MAX_CANDIDATES = 4096


class TestCase(ctypes.Structure):
    _fields_ = [
        ('inputs', ctypes.c_uint16 * MAX_IO_PORTS),
        ('outputs', ctypes.c_uint16 * MAX_IO_PORTS),
    ]


class TaskSpec(ctypes.Structure):
    _fields_ = [
        ('name', ctypes.c_char * 64),
        ('max_cycles', ctypes.c_uint64),
        ('input_ports', ctypes.c_uint8 * MAX_IO_PORTS),
        ('num_inputs', ctypes.c_int),
        ('output_ports', ctypes.c_uint8 * MAX_IO_PORTS),
        ('num_outputs', ctypes.c_int),
        ('tests', TestCase * MAX_TESTS),
        ('num_tests', ctypes.c_int),
    ]


class SearchCtx(ctypes.Structure):
    _fields_ = [
        ('task', TaskSpec),
        ('num_tests', ctypes.c_uint8),
        ('num_inputs', ctypes.c_uint8),
        ('num_outputs', ctypes.c_uint8),
        ('input_regs', ctypes.c_uint8 * MAX_IO_PORTS),
        ('output_port', ctypes.c_uint8),
    ]


class SearchState(ctypes.Structure):
    _fields_ = [
        ('regs', (ctypes.c_uint16 * PCA_NUM_REGS) * MAX_TESTS),
        ('flags', ctypes.c_uint8 * MAX_TESTS),
        ('live', ctypes.c_uint8),
        ('depth', ctypes.c_uint8),
    ]


# ── Library wrapper ──────────────────────────────────────────────────────────

class PCALib:
    """Wrapper around libpca.so C functions."""

    def __init__(self, lib_path):
        self.lib = ctypes.cdll.LoadLibrary(lib_path)

        # task_load(const char *path, task_spec_t *spec) -> int
        self.lib.task_load.argtypes = [ctypes.c_char_p, ctypes.POINTER(TaskSpec)]
        self.lib.task_load.restype = ctypes.c_int

        # search_ctx_init(search_ctx_t *ctx, const task_spec_t *task) -> int
        self.lib.search_ctx_init.argtypes = [
            ctypes.POINTER(SearchCtx), ctypes.POINTER(TaskSpec)]
        self.lib.search_ctx_init.restype = ctypes.c_int

        # search_state_init_root(const search_ctx_t *ctx, search_state_t *s)
        self.lib.search_state_init_root.argtypes = [
            ctypes.POINTER(SearchCtx), ctypes.POINTER(SearchState)]
        self.lib.search_state_init_root.restype = None

        # search_state_step(ctx, src, insn, dst) -> int
        self.lib.search_state_step.argtypes = [
            ctypes.POINTER(SearchCtx), ctypes.POINTER(SearchState),
            ctypes.c_uint16, ctypes.POINTER(SearchState)]
        self.lib.search_state_step.restype = ctypes.c_int

        # search_state_oep_hash64(ctx, s) -> uint64
        self.lib.search_state_oep_hash64.argtypes = [
            ctypes.POINTER(SearchCtx), ctypes.POINTER(SearchState)]
        self.lib.search_state_oep_hash64.restype = ctypes.c_uint64

        # search_state_key_hash64(ctx, s) -> uint64
        self.lib.search_state_key_hash64.argtypes = [
            ctypes.POINTER(SearchCtx), ctypes.POINTER(SearchState)]
        self.lib.search_state_key_hash64.restype = ctypes.c_uint64

        # search_state_leaf_outputs(ctx, s) -> uint8
        self.lib.search_state_leaf_outputs.argtypes = [
            ctypes.POINTER(SearchCtx), ctypes.POINTER(SearchState)]
        self.lib.search_state_leaf_outputs.restype = ctypes.c_uint8

        # search_gen_candidates_branchless(out, max, s) -> int
        self.lib.search_gen_candidates_branchless.argtypes = [
            ctypes.POINTER(ctypes.c_uint16), ctypes.c_int,
            ctypes.POINTER(SearchState)]
        self.lib.search_gen_candidates_branchless.restype = ctypes.c_int

    def load_task(self, path):
        spec = TaskSpec()
        rc = self.lib.task_load(path.encode(), ctypes.byref(spec))
        if rc != 0:
            raise RuntimeError(f"Failed to load task: {path}")
        return spec

    def init_ctx(self, spec):
        ctx = SearchCtx()
        rc = self.lib.search_ctx_init(ctypes.byref(ctx), ctypes.byref(spec))
        if rc != 0:
            raise RuntimeError("Failed to init search context")
        return ctx

    def init_root(self, ctx):
        root = SearchState()
        self.lib.search_state_init_root(ctypes.byref(ctx), ctypes.byref(root))
        return root

    def step(self, ctx, src, insn):
        dst = SearchState()
        ok = self.lib.search_state_step(
            ctypes.byref(ctx), ctypes.byref(src), insn, ctypes.byref(dst))
        return dst if ok else None

    def oep_hash(self, ctx, state):
        return self.lib.search_state_oep_hash64(
            ctypes.byref(ctx), ctypes.byref(state))

    def leaf_check(self, ctx, state):
        return self.lib.search_state_leaf_outputs(
            ctypes.byref(ctx), ctypes.byref(state))

    def gen_candidates(self, state):
        buf = (ctypes.c_uint16 * MAX_CANDIDATES)()
        n = self.lib.search_gen_candidates_branchless(buf, MAX_CANDIDATES,
                                                      ctypes.byref(state))
        return [buf[i] for i in range(n)]


# ── Feature extraction ───────────────────────────────────────────────────────

class FeatureExtractor:
    """Convert search states to model input tensors."""

    def __init__(self, task_io, num_tests=10):
        self.num_tests = num_tests
        nt = num_tests
        actual = min(task_io['num_tests'], nt)

        # Pre-compute task I/O bits (constant for all states of this task)
        inp_bits = bits_encode_array(task_io['inputs'][:actual])
        out_bits = bits_encode_array(task_io['outputs'][:actual])
        self.test_in = np.zeros((nt, 32), dtype=np.float32)
        self.test_out = np.zeros((nt, 16), dtype=np.float32)
        self.test_in[:actual] = inp_bits.reshape(actual, -1)
        self.test_out[:actual] = out_bits.reshape(actual, -1)

    def extract_batch(self, states):
        """Convert list of SearchState objects to batched tensors."""
        B = len(states)
        nt = self.num_tests

        per_test_all = np.zeros((B, nt, 180), dtype=np.float32)
        live_all = np.zeros((B, 8), dtype=np.float32)
        depth_all = np.zeros(B, dtype=np.float32)

        for i, s in enumerate(states):
            # Extract regs as numpy array
            regs = np.zeros((nt, PCA_NUM_REGS), dtype=np.uint16)
            for t in range(nt):
                for r in range(PCA_NUM_REGS):
                    regs[t, r] = s.regs[t][r]

            regs_bits = bits_encode_array(regs)  # (nt, 8, 16)

            # Flags
            flags = np.zeros((nt, 4), dtype=np.float32)
            for t in range(nt):
                f = s.flags[t]
                flags[t, 0] = (f >> 0) & 1
                flags[t, 1] = (f >> 1) & 1
                flags[t, 2] = (f >> 2) & 1
                flags[t, 3] = (f >> 3) & 1

            # Concatenate: input(32) + target(16) + regs(128) + flags(4) = 180
            per_test_all[i] = np.concatenate([
                self.test_in,
                self.test_out,
                regs_bits.reshape(nt, -1),
                flags,
            ], axis=1)

            # Live bits
            for b in range(8):
                live_all[i, b] = (s.live >> b) & 1

            depth_all[i] = s.depth

        return {
            'per_test': torch.from_numpy(per_test_all),
            'live': torch.from_numpy(live_all),
            'depth': torch.from_numpy(depth_all),
        }


# ── Search ───────────────────────────────────────────────────────────────────

def best_first_search(lib, ctx, root, max_depth, budget,
                      model=None, feat_extractor=None, device=None, rng=None):
    """
    Best-first search using model scores or random priorities.

    Returns (solved_at_expansion, total_expansions) or (None, total_expansions).
    """
    # Heap: (-priority, counter, state_idx)
    # state_idx indexes into `states` list
    states = [root]
    heap = [(0.0, 0, 0)]  # root gets priority 0 (expanded first)
    counter = 1
    expansions = 0
    visited_oep = set()

    while heap and expansions < budget:
        _, _, state_idx = heapq.heappop(heap)
        state = states[state_idx]

        # OEP dedup: skip if semantically equivalent state already expanded
        oep = lib.oep_hash(ctx, state)
        if oep in visited_oep:
            continue
        visited_oep.add(oep)

        expansions += 1

        if state.depth >= max_depth:
            continue

        # Generate candidates
        candidates = lib.gen_candidates(state)

        # Apply each candidate, check for solutions
        children = []
        for insn in candidates:
            child = lib.step(ctx, state, insn)
            if child is None:
                continue

            # Leaf check: does this child solve the task?
            if lib.leaf_check(ctx, child):
                return expansions, expansions

            # Only keep children that can still reach max_depth
            if child.depth < max_depth:
                children.append(child)

        if not children:
            continue

        # Score children
        if model is not None and feat_extractor is not None:
            # Batch score with model
            batch = feat_extractor.extract_batch(children)
            with torch.no_grad():
                reach_logit, _, _ = model(
                    batch['per_test'].to(device),
                    batch['live'].to(device),
                    batch['depth'].to(device),
                )
            scores = torch.sigmoid(reach_logit.squeeze(1)).cpu().numpy()
        else:
            # Random priorities
            scores = rng.random(len(children))

        # Add children to heap (negate score for min-heap)
        for child, score in zip(children, scores):
            idx = len(states)
            states.append(child)
            heapq.heappush(heap, (-float(score), counter, idx))
            counter += 1

    return None, expansions


# ── Main evaluation ──────────────────────────────────────────────────────────

def evaluate_task(lib, task_path, optimal_depth, model, device, num_tests,
                  max_budget=64, n_random_trials=5):
    """Evaluate one task: model-guided vs random search."""
    spec = lib.load_task(task_path)
    ctx = lib.init_ctx(spec)
    root = lib.init_root(ctx)

    # Check if root already solves it (depth-0 solution)
    if lib.leaf_check(ctx, root):
        return {
            'model_solved_at': 0,
            'model_expansions': 0,
            'random_solved_at': [0] * n_random_trials,
            'optimal_depth': optimal_depth,
        }

    task_io = load_task_io(task_path)
    feat_extractor = FeatureExtractor(task_io, num_tests=num_tests)

    # Model-guided search
    model_solved, model_expansions = best_first_search(
        lib, ctx, root, optimal_depth, max_budget,
        model=model, feat_extractor=feat_extractor, device=device)

    # Random baseline (average over multiple trials)
    random_solved_list = []
    for trial in range(n_random_trials):
        rng = np.random.RandomState(seed=42 + trial)
        root_copy = lib.init_root(ctx)  # fresh root each trial
        r_solved, _ = best_first_search(
            lib, ctx, root_copy, optimal_depth, max_budget, rng=rng)
        random_solved_list.append(r_solved)

    return {
        'model_solved_at': model_solved,
        'model_expansions': model_expansions,
        'random_solved_at': random_solved_list,
        'optimal_depth': optimal_depth,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate model-guided search vs random on held-out tasks')
    parser.add_argument('--base-dir', default=None)
    parser.add_argument('--checkpoint', default='checkpoints/value_v1.pt')
    parser.add_argument('--lib', default='libpca.so')
    parser.add_argument('--split', default='test',
                        help='Which manifest split to evaluate')
    parser.add_argument('--max-budget', type=int, default=64,
                        help='Maximum node expansions per task')
    parser.add_argument('--n-random', type=int, default=5,
                        help='Number of random baseline trials per task')
    parser.add_argument('--num-tests', type=int, default=10)
    parser.add_argument('--max-depth', type=int, default=8)
    parser.add_argument('--max-tasks', type=int, default=None,
                        help='Limit number of tasks to evaluate')
    args = parser.parse_args()

    if args.base_dir is None:
        args.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load library
    lib_path = os.path.join(args.base_dir, args.lib)
    lib = PCALib(lib_path)
    print(f"Loaded {lib_path}")

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
    if 'val_metrics' in ckpt:
        vm = ckpt['val_metrics']
        print(f"  Checkpoint val: P={vm.get('precision',0):.3f} "
              f"R={vm.get('recall',0):.3f} F1={vm.get('f1',0):.3f}")

    # Load manifest
    manifest_path = os.path.join(args.base_dir, 'data', 'manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)

    tasks = [t for t in manifest['tasks']
             if t['split'] == args.split and t['solved']]
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]

    print(f"\nEvaluating {len(tasks)} {args.split}-split tasks "
          f"(budget={args.max_budget}, {args.n_random} random trials)")
    print("-" * 70)

    # Evaluate each task
    budgets = [8, 16, 32, 64]
    model_success = {k: 0 for k in budgets}
    random_success = {k: 0.0 for k in budgets}
    total = 0

    for i, task_info in enumerate(tasks):
        task_path = os.path.join(args.base_dir, task_info['path'])
        optimal_depth = task_info['optimal_depth']

        t0 = time.time()
        result = evaluate_task(
            lib, task_path, optimal_depth, model, device,
            args.num_tests, max_budget=args.max_budget,
            n_random_trials=args.n_random)
        dt = time.time() - t0

        total += 1

        # Compute Success@K
        for k in budgets:
            if result['model_solved_at'] is not None and result['model_solved_at'] <= k:
                model_success[k] += 1

            random_hits = sum(
                1 for r in result['random_solved_at']
                if r is not None and r <= k)
            random_success[k] += random_hits / args.n_random

        # Progress
        m_str = (f"model={result['model_solved_at']}"
                 if result['model_solved_at'] is not None else "model=FAIL")
        r_solved = [r for r in result['random_solved_at'] if r is not None]
        r_str = (f"random={np.mean(r_solved):.0f}"
                 if r_solved else "random=FAIL")
        print(f"  [{i+1:3d}/{len(tasks)}] {task_info['name'][:30]:30s}  "
              f"d*={optimal_depth}  {m_str:15s}  {r_str:15s}  [{dt:.2f}s]")

    # Summary
    print("\n" + "=" * 70)
    print(f"Success@K on {total} {args.split}-split tasks:")
    print(f"{'K':>4s}  {'Model':>8s}  {'Random':>8s}  {'Lift':>8s}")
    print("-" * 34)
    for k in budgets:
        m_rate = model_success[k] / max(1, total)
        r_rate = random_success[k] / max(1, total)
        lift = m_rate - r_rate
        print(f"{k:4d}  {m_rate:8.1%}  {r_rate:8.1%}  {lift:+8.1%}")

    # Save results
    results = {
        'split': args.split,
        'total_tasks': total,
        'max_budget': args.max_budget,
        'n_random_trials': args.n_random,
        'checkpoint': args.checkpoint,
        'model_success': {str(k): model_success[k] for k in budgets},
        'random_success': {str(k): random_success[k] for k in budgets},
    }
    results_path = os.path.join(args.base_dir, 'data', 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
