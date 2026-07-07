#!/usr/bin/env python3
"""
test_exhaustive.py — Unit tests for search_exhaustive_complete (issue #7).

Covers the issue #7 test plan items that don't need a trained model:
  1. C function finds known solutions from root and partial states
  2. IDDFS returns shortest completion (length == manifest optimal_depth)
  3. found=False when no completion exists within max_depth
  4. OEP table clearing doesn't leak state between calls
  5. Replay of the returned instruction path leaf-checks in Python

Usage: python3 python/test_exhaustive.py [--max-per-depth 3]
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search_eval import PCALib

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def replay_solves(lib, ctx, state, insns):
    """Replay instruction path via lib.step, return True if final state solves."""
    cur = state
    for insn in insns:
        cur = lib.step(ctx, cur, insn)
        if cur is None:
            return False
    return bool(lib.leaf_check(ctx, cur))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-per-depth', type=int, default=3)
    args = ap.parse_args()

    lib = PCALib(os.path.join(BASE, 'libpca.so'))

    with open(os.path.join(BASE, 'data', 'manifest.json')) as f:
        manifest = json.load(f)

    # Pick solved tasks at kernel depths 1..3 (root-level exhaustive at
    # depth 4+ is a full enumerate run — too slow for a unit test).
    by_depth = {}
    for t in manifest['tasks']:
        d = t.get('optimal_depth')
        if t.get('solved') and d in (1, 2, 3):
            by_depth.setdefault(d, []).append(t)

    failures = 0
    checked = 0
    prev = None  # (task_path, sol_len) for cross-call leak check

    for d in sorted(by_depth):
        for t in by_depth[d][:args.max_per_depth]:
            path = os.path.join(BASE, t['path'])
            spec = lib.load_task(path)
            ctx = lib.init_ctx(spec)
            root = lib.init_root(ctx)
            checked += 1

            # --- Test 1+2: root completion finds shortest solution ---
            t0 = time.time()
            found, sol_len, insns = lib.exhaustive_complete(ctx, root, d)
            dt = (time.time() - t0) * 1000
            states = lib.exhaustive_states_explored()
            ok = found and sol_len == d and replay_solves(lib, ctx, root, insns)
            print(f"  [{'OK' if ok else 'FAIL'}] {t['name'][:40]:40s} "
                  f"d*={d} found={found} len={sol_len} "
                  f"states={states} [{dt:.1f}ms]")
            if not ok:
                failures += 1
                continue

            # --- Test 3: no completion within max_depth = d-1 ---
            found2, sol_len2, _ = lib.exhaustive_complete(ctx, root, d - 1)
            if found2:
                print(f"  [FAIL] {t['name']}: found len-{sol_len2} completion "
                      f"within {d-1} < optimal {d}")
                failures += 1

            # --- Test 1b: partial-state completion (step 1 insn, finish) ---
            if d >= 2:
                mid = lib.step(ctx, root, insns[0])
                assert mid is not None
                found3, sol_len3, insns3 = lib.exhaustive_complete(ctx, mid, d)
                ok3 = (found3 and sol_len3 <= d - 1
                       and replay_solves(lib, ctx, mid, insns3))
                if not ok3:
                    print(f"  [FAIL] {t['name']}: partial-state completion "
                          f"found={found3} len={sol_len3}")
                    failures += 1

            # --- Test 4: OEP leak check (re-run previous task, same result) ---
            if prev is not None:
                p_path, p_depth, p_len = prev
                p_spec = lib.load_task(p_path)
                p_ctx = lib.init_ctx(p_spec)
                p_root = lib.init_root(p_ctx)
                f4, l4, _ = lib.exhaustive_complete(p_ctx, p_root, p_depth)
                if not f4 or l4 != p_len:
                    print(f"  [FAIL] OEP leak: {p_path} re-run gave "
                          f"found={f4} len={l4}, expected len={p_len}")
                    failures += 1
            prev = (path, d, sol_len)

    print(f"\n{checked} tasks checked, {failures} failures")
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
