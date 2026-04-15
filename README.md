# pca

**Predictive coding in assembly.**

The brain doesn't passively receive input — it runs a prediction engine. Perception is the residual: the error between what you expected and what arrived. Minimize the prediction error and you've understood the signal. That's [predictive coding](https://en.wikipedia.org/wiki/Predictive_coding).

A PID controller does the same thing. It holds a setpoint (prediction), reads a sensor (observation), computes the error (surprise), and drives an actuator to minimize it. Same loop, different substrate. One runs on neurons, the other on a register machine.

This repo explores that intersection — control theory, neuroscience, and bare-metal programming on hardware simple enough to hold in your head. The long game: prompt → fully performant assembly, via learned computational proprioception.

## What's here

### PCA-16 emulator

A custom 16-bit ISA and high-performance emulator designed for AI-driven program synthesis.

```
make
./pca programs/fib.s        # assemble, disassemble, run, dump state
./pca programs/fib.s -q     # quiet: print r1 only
./pca programs/fib.s -t     # trace: print every instruction
./pca programs/pid.s -c 1000  # run PID controller for 1000 cycles
```

~338M instructions/sec. A fibonacci-length program evaluates in ~20μs, meaning millions of candidate programs per second for search/training.

### ISA: PCA-16

25 opcodes, fixed 16-bit encoding, 8 registers (`r0` hardwired to zero), 256 words each of program and data memory. Harvard architecture, ZNVC flags, 16-entry hardware call stack.

| Category | Instructions |
|----------|-------------|
| Arithmetic | `ADD SUB MUL DIV MOD` |
| Bitwise | `AND OR XOR SHL SHR ASR` |
| Compare | `CMP` (sets ZNVC flags) |
| Memory | `LD ST` |
| Immediate | `LDI LUI ADDI` |
| Branch | `BR` with conditions: `BRA BZ BNZ BLT BGE BLTU BGEU` |
| Control | `JMP CALL RET` |
| I/O | `IN OUT` (8 ports) |
| System | `NOP HLT` |
| Pseudo | `MOV NEG BEQ BNE BN BNN` |

Instruction encoding:
```
Type R: [op:5][rd:3][rs:3][rt:3][pad:2]   — 3-register ALU ops
Type I: [op:5][rd:3][imm:8]               — immediates, branches, I/O
```

Design rationale: the ISA is small enough for tractable search (25 opcodes × 8 registers = bounded branching factor) but expressive enough for real algorithms (native multiply, signed arithmetic, proper overflow flags). The 6502's lack of MUL required 150-cycle shift-and-add routines; PCA-16 does it in one instruction.

### Programs

- `programs/fib.s` — Fibonacci(10) = 55 in 12 instructions, 77 cycles
- `programs/echo.s` — Minimal I/O loop (3 instructions)
- `programs/pid.s` — Full PID controller with 8.8 fixed-point math, integral windup clamping (78 instructions)
- `pid.asm` — Reference: the same PID on a 6502, for comparison

### Source

- `src/pca.h` — ISA definitions, VM state (fully exposed for "proprioception")
- `src/vm.c` — Execution engine (flat switch dispatch)
- `src/asm.c` — Two-pass assembler + disassembler
- `src/main.c` — CLI

## Phase 1: Exhaustive solver

Given only input-output examples, the solver enumerates every possible program and keeps the shortest correct one. No heuristics, no approximations — just brute-force search with enough pruning to make it tractable.

Nine hand-crafted tasks solved with provably optimal programs:

| Task | Optimal length | Description |
|------|---------------|-------------|
| negate | 4 insns | Two's complement negation |
| double | 4 insns | Multiply by 2 |
| square | 4 insns | Square the input |
| add | 5 insns | Add two inputs |
| abs | 6 insns | Absolute value |
| relu | 6 insns | max(0, x) |
| max | 7 insns | Maximum of two inputs |
| saturating_add | 8 insns | Add with overflow clamp to 0xFFFF |
| is_power2 | 8 insns | Bit trick: x & (x-1) == 0 |

"Provably optimal" means every shorter program was exhaustively tested and none are correct. The solver independently rediscovered `x & (-x)` for power-of-two detection — the same bit trick humans use.

**Pruning** makes this feasible. Raw branching factor per instruction is ~32K. Register liveness analysis, commutativity canonicalization, observational equivalence pruning (OEP), and dead instruction elimination reduce the effective branching factor to ~500-1000 per position. At depth 8 that's still ~4×10²¹ raw candidates, but the pruned tree is orders of magnitude smaller.

**Hybrid CPU-GPU solver.** The CPU explores the search tree with smart pruning (iterative deepening, liveness tracking, branch generation), emitting batches of partial programs. The GPU brute-forces all suffix completions in parallel. On an RTX 5070 Ti, this yields 85-520× speedup over CPU-only, depending on the task.

### Synthetic task generation

Beyond the nine hand-crafted tasks, the solver generated 1,000 synthetic tasks by sampling random branchless programs and deriving I/O specifications from their behavior. 949 of these were solved within the depth budget, producing 25.4 million labeled search states — perfect supervision for Phase 2.

## Phase 2: Learning from perfect data

The thesis: exhaustive search on small programs generates ground-truth training data with no reward shaping and no approximations. A value network trained on this data learns to predict which partial programs lead to correct solutions — and then guides search on tasks too large for brute force.

### Value network

280K parameters. Each test case is encoded independently by a shared MLP over bit-level features (input bits, target bits, current register bits, flags), aggregated via mean + max pooling, and fed through a trunk to two heads: **reachability** (does this partial program extend to a correct solution?) and **distance** (how many instructions remain?).

Bit-level encoding matters — the ISA is full of bitwise operations, and representing 16-bit registers as 16 binary features lets the network reason about bit structure directly.

Trained on 17.8M states from 663 tasks. Cross-task validation F1 on 143 held-out tasks: **0.522** (precision 45%, recall 62%). Same-task F1 reaches 0.920 — the gap is the generalization challenge.

### Does it beat random search?

Yes. The milestone question was whether a learned value function can guide best-first search better than random ordering. On 143 held-out test tasks:

| Expansions (K) | Model | Random | Lift |
|---|---|---|---|
| 8 | 51.0% | 30.9% | +20.1% |
| 16 | 58.0% | 37.2% | +20.8% |
| 32 | 62.9% | 48.4% | +14.5% |
| 64 | 74.8% | 62.2% | +12.6% |

The model solved 4 depth-3 tasks that random search could not solve in any of 5 trials. At tight budgets (K=8), the model finds solutions 65% more often than random. The advantage narrows at larger budgets as random search eventually stumbles into solutions — but the point is that the value network provides *useful search guidance* learned purely from exhaustive data on other tasks.

## Current status

**Proven:** Depth 1-3 branchless tasks. The solver-to-learner pipeline works end to end — exhaustive search produces perfect data, the value network learns cross-task structure from it, and guided search outperforms random on held-out tasks.

**Next:** Curriculum learning to deeper tasks. The model has never seen positive examples at depth 4+. The plan is to generate synthetic tasks with longer optimal programs (depth 4-6), retrain with combined data, and bootstrap upward — use the depth-D model to help generate depth-D+1 data. The same technique should extend to branch support (CMP/BR instructions) and eventually to program lengths where exhaustive search is infeasible.

The bet is that bootstrap generalizes: learn search heuristics where exhaustive data is available, then apply them where it isn't.

## The premise

If the reason we abandoned assembly was that humans can't manage the complexity — and machines can manage complexity differently than humans — what happens when you give a system *computational proprioception*? Not reasoning about machine state through text, but a learned sensory modality for the reachable state space itself.

The VM state is a 266-element vector: 8 registers + 256 memory words + PC + SP. That's small enough for a neural network to digest as a direct observation. The action space is the instruction set. The reward is program correctness. The question is whether a system trained on this loop develops the kind of intuition for computational structure that lets it write assembly no human would consider — an AlphaGo for programs.

A PID controller is the simplest version of the predictive coding loop: sense, predict, correct. Start here.
