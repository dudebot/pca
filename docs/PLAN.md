# PCA: Prompt → Assembly via Learned Computational Proprioception

## Context

We have a working PCA-16 emulator (25-opcode ISA, 8 registers, 256 words program/data memory, ~338M insns/sec). Three perspectives have been synthesized (Claude, GPT Pro, Codex) into a concrete research plan. The goal: train a system that develops "computational proprioception" — learned intuition for machine state — and uses it to synthesize correct assembly programs from specifications.

The user's hardware target is an RTX 5070 (12GB VRAM). The approach must validate cheaply before scaling.

## Architecture Overview

```
Spec (I/O examples) ──→ Policy Network ──→ Instruction ──→ VM Step ──→ Updated State
       ↑                      ↑                                           │
       │                 Value Network ←──────────────────────────────────┘
       │                      │
       └── CEGIS verifier ←───┘ (counterexample tests)
```

Two-phase synthesis (per Codex):
1. **Correctness**: Neural-guided A* search, policy as prior, value as heuristic
2. **Optimization**: STOKE-style stochastic rewrites preserving correctness

## What Exists

- `src/pca.h` — ISA definitions, full VM state struct (266-element observation)
- `src/vm.c` — Execution engine, step/run/dump, I/O callbacks
- `src/asm.c` — Two-pass assembler + disassembler
- `src/main.c` — CLI with trace, disassemble, quiet modes
- `programs/` — fib.s (12 insns), echo.s (3 insns), pid.s (78 insns)
- `Makefile` — builds `pca` binary

## Build Plan

### Phase 1: Synthesis Infrastructure (tonight)

**1a. Task specification format**
- File: `tasks/spec.h`, `tasks/spec.c`
- A task = list of (input_ports[], expected_output_ports[], max_cycles)
- JSON loader for task files
- Task library: `tasks/*.json` with graduated difficulty

**1b. Batch VM runner**
- File: `src/batch.h`, `src/batch.c`
- Run N VMs in parallel (C threads or just sequential-but-fast)
- Evaluate a program against a task spec (all test cases)
- Return: pass/fail per test, total cycles, final state
- This is the evaluator API that all search methods use

**1c. Exhaustive solver for tiny programs**
- File: `tools/enumerate.c`
- Iterative-deepening DFS over instruction sequences
- Fixed I/O scaffold: IN instructions first, OUT+HLT last, enumerate middle
- Pruning (critical for feasibility):
  - Register liveness: only read registers that have been written
  - Skip r0 as destination (hardwired to zero)
  - Commutativity: ADD r1,r2,r3 = ADD r1,r3,r2, only try canonical order
  - Skip no-ops: MOV r1,r1, ADD r1,r1,r0, etc.
  - Skip dead writes: don't write to a reg that's immediately overwritten
  - Early termination: if state matches a previously-seen state, prune
- Feasibility analysis:
  - Raw branching factor per instruction: ~32K
  - With liveness+pruning: ~500-1000 effective choices
  - At 500/step, 3 middle insns = 125M candidates = ~4sec at 33M evals/sec
  - 4 middle insns = 62.5B = ~30min (feasible overnight)
  - 5+ middle insns: needs observational equivalence pruning (OEP)
- Target: find optimal solutions for programs up to ~8 total instructions
- Output: dataset of (task, correct_program, all_intermediate_states)
- This generates perfect training data for the value/policy networks

**1d. Task library** (graduated difficulty, interesting algorithmic variety)

Tier 1 — pure arithmetic, no branches:
- `negate.json` — negate input (~3 insns)
- `double.json` — multiply by 2 (~3-4 insns)
- `add.json` — add two inputs from ports 0+1 (~4 insns)
- `square.json` — square input (~4 insns)

Tier 2 — conditional logic, branches:
- `abs.json` — absolute value (~5-6 insns, needs branch)
- `relu.json` — max(0, x), the ML activation (~4-5 insns)
- `max.json` — max of two inputs (~6-7 insns)
- `sign.json` — sign function: -1, 0, or 1 (~7-8 insns)
- `clamp.json` — clamp to 0-255 (~8-10 insns)
- `saturating_add.json` — add, clamp to 0xFFFF on overflow (~6-8 insns)

Tier 3 — loops and memory:
- `collatz.json` — one Collatz step: even→x/2, odd→3x+1 (~8-10 insns)
- `is_power2.json` — is input a power of 2? bit trick (~5-6 insns)
- `popcount.json` — count set bits (~10-12 insns, loop)
- `gcd.json` — Euclidean algorithm (~10-14 insns)
- `fib.json` — fibonacci(n) (~12 insns, loop + accumulator)

Each task JSON: 8-16 test cases including edge cases (0, 1, -1, max, min)

**1e. Random search baseline**
- File: `tools/random_search.c`
- Generate random valid programs, evaluate against spec
- Measure pass@K for K = 100, 1000, 10000, 100000
- This is the baseline that everything else must beat

**1f. Python bindings**
- File: `python/pca.py`
- ctypes wrapper around the C library (compile as .so)
- `PcaVM` class with step/run/dump/load
- `evaluate(program, task_spec) → score`
- Numpy-friendly state extraction (vm.state → np.array)
- This is the bridge to PyTorch for Phase 2

### Phase 2: Learning (needs GPU, post-tonight)

**2a. Dataset generation**
- Run exhaustive solver on all Phase 1 tasks
- For each solve trajectory, record:
  - State at each step (266-dim vector)
  - Instruction chosen
  - Instructions remaining to solution
  - Whether this prefix extends to any correct solution
- Format: HDF5 or numpy arrays

**2b. Value network**
- Input: state vector (266) + task embedding (from I/O examples)
- Output: P(this state leads to correct solution), estimated remaining instructions
- Architecture: MLP (266 → 512 → 256 → 128 → 2), ~500K params
- Training: supervised on exhaustive solver trajectories
- Loss: BCE for correctness probability, MSE for remaining instructions

**2c. Policy network**
- Input: same as value network
- Output: distribution over next instruction (softmax over valid encodings)
- Architecture: same MLP backbone, different head
- Training: supervised on optimal instruction choices from exhaustive data
- Can share backbone with value network (two-headed)

**2d. A* search harness**
- Priority queue of partial programs
- Score = policy_logprob + value_estimate (like AlphaGo's PUCT)
- Expand: try top-K instructions by policy score
- Evaluate: run each extension, get value estimate
- Prune: discard states with value < threshold
- Terminate: when program passes all tests or budget exhausted

### Phase 3: Verification & Optimization (future)

**3a. CEGIS loop**
- When a candidate passes all known tests:
  - Generate adversarial inputs (random, boundary, symbolic)
  - If any fail, add as new test case, continue search
  - If all pass, declare correct

**3b. STOKE-style superoptimization**
- Given a correct program, try random rewrites:
  - Swap two instructions
  - Replace instruction with random alternative
  - Delete an instruction (shift others)
  - Insert a NOP (test if removable)
- Keep rewrite if: still correct AND shorter/faster
- Run for N iterations, return shortest correct variant

**3c. Curriculum learning**
- Start with 4-instruction tasks, train until >90% solve rate
- Add 8-instruction tasks, retrain
- Add 12-instruction tasks, retrain
- Eventually: 32-64 instruction tasks with A* search

### Phase 4: Prompt Interface (future)

**4a. Natural language → spec**
- Use an LLM to convert natural language to I/O test cases
- "Write a program that doubles its input" → [(in=5, out=10), (in=0, out=0), ...]

**4b. End-to-end pipeline**
- prompt → spec → search → correct program → optimize → output assembly

## Tonight's Focus

1. **Task spec format + evaluator** — JSON I/O specs, C evaluator
2. **Full task library** — all tiers above with rich test cases
3. **Exhaustive solver** — the big one: iterative deepening DFS with pruning,
   find provably optimal programs for tier 1 and tier 2 tasks
4. **Verify** — every found solution runs correctly, is minimal length

## Verification (tonight)

- [ ] All C code compiles clean with -Wall -Wextra -Wpedantic
- [ ] `make test` still passes (existing programs unbroken)
- [ ] Task spec loader correctly parses JSON task files
- [ ] Batch evaluator correctly scores known-good programs (fib→55)
- [ ] Exhaustive solver finds correct negate/double/add programs ≤ 6 instructions
- [ ] Exhaustive solver solutions actually produce correct output when run
- [ ] Random search pass@K provides baseline numbers for comparison
- [ ] Python bindings can load a program, run it, extract state as numpy array
- [ ] Enumerated solutions are minimal (no shorter correct program exists)

## Codex Review

Codex was at capacity when I tried to get a review. Plan proceeds based on
prior three-way synthesis (Claude + GPT Pro + Codex ISA critique). Will
re-submit for review when capacity frees up. The critical self-identified
risk is exhaustive solver feasibility at >8 instructions — mitigated by
starting with trivially short tasks and treating >8 as a Phase 2 problem
(neural-guided search).

## Key Files to Create

```
tasks/
  spec.h          — task spec structs and loader API  
  spec.c          — JSON parser for task files
  add.json        — add two inputs
  max.json        — max of two
  abs.json        — absolute value
  double.json     — multiply by 2
  negate.json     — negate input
  clamp.json      — clamp to 0-255
  fib.json        — fibonacci (stretch)
tools/
  enumerate.c     — exhaustive solver
  random_search.c — baseline random search
python/
  pca.py          — ctypes bindings
src/
  batch.h         — batch evaluator API
  batch.c         — batch evaluator implementation
```

## Key Files to Modify

```
Makefile          — add new targets (enumerate, random_search, libpca.so)
src/pca.h         — no changes needed (API is sufficient)
```
