# From Exhaustive Search to Learned Intuition

### PCA: A Machine That Writes Optimal Assembly

---

## A Machine Re-derived a Bit Trick

Given only input-output examples for "is this number a power of 2?", a brute-force solver explored the space of all 7-instruction programs on a custom ISA and found this:

```asm
IN   r1, 0        ; read input
NEG  r2, r1        ; r2 = -x (two's complement)
AND  r2, r1, r2    ; r2 = x & (-x)  — isolate lowest set bit
BZ   +1            ; if zero, skip (handles x=0)
DIV  r5, r2, r1    ; r5 = lowest_bit / x — equals 1 iff power of 2
OUT  r5, 0         ; output result
HLT
```

7 instructions. **Provably optimal** — every shorter program was tested and none are correct. The solver independently discovered `x & (-x)`, the same bit trick humans use.

This is what happens when you can search *exhaustively*. No heuristics, no approximations — just enumerate every possible program and keep the shortest correct one.

**The question PCA asks: can a neural network learn to predict these solutions without searching?**

---

## The Approach

**PCA** (Predictive Coding in Assembly) is a three-stage system:

1. **Exhaustive search** on a custom 16-bit ISA produces *provably perfect* training data — 25.4 million labeled search states across 778 solved tasks
2. **Value network** trained on these states learns to predict which partial programs lead to correct solutions (F1 = 0.920 on held-out states from training tasks)
3. **Neural-guided A\* search** (next) will use the value function to solve tasks too large for brute force

The core insight: exhaustive search on small problems generates *perfect supervision* for learning. No reward shaping, no approximations — every recorded semantic state has a ground-truth label derived from complete enumeration.

This is related to superoptimizers like STOKE, but differs in scope and method. STOKE operates on real x86 programs using stochastic search. PCA works on a much smaller ISA (16-bit, 25 opcodes) but achieves *exhaustive* search — guaranteeing optimality — and uses the exhaustive search data to train a neural search guide. The bet is that the bootstrap technique generalizes: learn search heuristics where exhaustive data is available, then apply them where it isn't.

---

## PCA-16: A Searchable ISA

A purpose-built 16-bit instruction set — 25 opcodes, 8 registers, fixed-width encoding. Small enough for exhaustive search, rich enough for arithmetic, logic, branches, and I/O.

A typical solved program is 4-8 instructions. At depth 8 with ~500 legal candidates per position, the raw search space is **~500^8 ≈ 3.9 × 10^21**. Pruning reduces actual exploration by orders of magnitude — the solver typically evaluates millions to billions of candidates, not trillions.

### How Pruning Tames the Search Space

| Technique | Effect |
|---|---|
| **Register liveness** | Only read registers that have been written → ~70% branch reduction |
| **Commutativity** | ADD r1,r2,r3 = ADD r1,r3,r2 → skip symmetric duplicates |
| **Observational equivalence (OEP)** | Hash register states across all tests; prune prefixes that produce identical semantic states |
| **Dead instruction elimination** | Skip instructions that provably do nothing: ADD r1,r1,r0, etc. |

---

## The Solver: Hybrid CPU-GPU

```
CPU (DFS with pruning)              GPU (brute-force suffix)
┌─────────────────────┐            ┌─────────────────────┐
│ Iterative deepening │            │ 3,787 candidates    │
│ OEP pruning         │──batch──→  │ per position        │
│ Liveness tracking   │  prefixes  │ Full VM per thread  │
│ Branch generation   │            │ Atomic solution     │
│                     │←─results── │ reporting           │
└─────────────────────┘            └─────────────────────┘
```

The CPU explores the tree with smart pruning, emitting batches of partial programs. The GPU brute-forces all suffix completions in parallel.

| Task | Optimal | CPU | Hybrid (5070 Ti) | Speedup |
|---|---|---|---|---|
| abs(x) | 6 insns | <0.01s | 0.02s | — |
| max(a,b) | 7 insns | 0.3s | 0.06s | 5× |
| saturating_add | 8 insns | 12s | 1.2s | 10× |
| is_power2 | 7 insns | 70s | 3.3s | 21× |

*Every solution is proven optimal: all shorter programs have been exhaustively tested and eliminated.*

---

## Phase 2: Learning from Perfect Data

States visited during exhaustive search receive ground-truth labels:

- **Is this state on an optimal path?** (binary)
- **How many steps remain to a solution?** (0..D*)
- **Which instructions maintain optimality?** (action set)

### Training Data

| | |
|---|---|
| **Task corpus** | 1,000 synthetic tasks (sampled branchless programs, filtered for interestingness) |
| **Solved** | 778 tasks within depth-6 budget |
| **Total states** | 25.4M labeled search states |
| **Positive rate** | ~1% (states on optimal trajectories) |
| **Dataset size** | 81 GB |

### State Representation

The network sees **multi-test register states** — register values across *all* test cases simultaneously:

```
Test 0: r1=5,    r2=65531, ...  flags=N     (input=5, target=65531)
Test 1: r1=0,    r2=0,     ...  flags=Z     (input=0, target=0)
Test 2: r1=100,  r2=65436, ...  flags=N     (input=100, target=65436)
```

The network must learn that "r2 is always the negation of r1 across all tests" — a semantic property that predicts whether this prefix extends to a correct program.

### Architecture

280,000 parameters. Trains in minutes on a single GPU.

Each test case is encoded independently by a shared MLP (180 → 256 → 256) over bit-level features: input bits, target bits, current register bits, and flags. Test encodings are aggregated via mean + max pooling (→ 512 dims), concatenated with global features (live mask, depth, budget), and fed through a trunk (522 → 256 → 128) to two heads: **reachability** (binary logit) and **distance** (classification over 0..8 remaining steps).

Bit-level encoding matters — the ISA is full of bitwise operations (AND, SHL, ASR), and representing registers as 16 binary features lets the network reason about bit structure directly.

---

## Preliminary Results: Same-Task Value Prediction

Trained on 142K states from 50 tasks. Validation is a random held-out split of *states from the same 50 tasks* — this measures same-task state discrimination, not cross-task generalization.

| Metric | Value |
|---|---|
| **Precision** | 97.5% — "this state leads to a solution" is almost always correct |
| **Recall** | 86.1% — finds 86% of optimal-path states |
| **F1** | 0.920 |

The dataset has 99:1 class imbalance — only ~1% of explored states are on optimal paths. A trivial "always predict negative" baseline would achieve 99% accuracy but 0% precision. The network learns genuine structure within the training task distribution.

**What this doesn't show yet:** whether the network generalizes to *unseen tasks*, and whether value-guided search actually reduces node expansions vs random ordering. Both are necessary for the core thesis. Full dataset (25.4M records, 778 tasks) is generated; task-held-out evaluation and search integration are next.

---

## What's Next

**The milestone:** *Success@64* — given a held-out task, can the value network guide best-first search to find the optimal program within 64 node expansions, where random search cannot?

### Roadmap

| Step | What | Status |
|---|---|---|
| Train on full 25M dataset | Expect stronger P/R/F1 | Ready |
| Offline search evaluation | Success@K vs random baseline | Next |
| Policy head | Predict next instruction distribution | Designed |
| Neural A* search | Policy × value guided tree search | Designed |
| Branch support | Extend state rep for conditionals | Future |
| Curriculum learning | 4 → 8 → 16 → 32+ instruction programs | Future |
| CEGIS verification | Adversarial test generation for correctness proofs | Future |

### The Vision

A system that synthesizes correct assembly from I/O specifications — not by exhaustive search, but by learning search heuristics from exhaustive data. Narrow today (branchless, depth-6, toy ISA). If the value network beats random search on held-out tasks, that's a strong feasibility result — and the same bootstrap technique applies to any domain where small instances are exhaustively solvable.

---

*All solver solutions are provably optimal and CPU-verified. All training labels are ground truth from exhaustive enumeration.*
