# PCA — Predictive Coding in Assembly

## What this is

AI-driven program synthesis on a custom 16-bit ISA (PCA-16). The system
exhaustively solves tiny tasks to generate perfect training data, then
(planned) trains a value/policy network to guide search on harder tasks.

Read `docs/PLAN.md` for the full research roadmap.
Read `docs/CUDA_SPEC.md` for GPU solver design and performance targets.

## Build

```bash
make          # builds: pca (emulator CLI), enumerate (CPU solver)
make test     # run emulator tests
make synth    # run CPU solver on tier 1 tasks

# GPU solver (requires CUDA toolkit):
nvcc -O3 -arch=native -o gpu_enumerate gpu/evaluate.cu tasks/spec.c src/vm.c
```

## Repo layout

```
src/           PCA-16 emulator (pca.h, vm.c, asm.c, main.c)
tasks/         Task specs: spec.h/spec.c (loader) + *.json (I/O test cases)
tools/         CPU exhaustive solver (enumerate.c)
gpu/           CUDA exhaustive solver (evaluate.cu) — untested, needs GPU
programs/      Hand-written PCA-16 assembly (fib.s, echo.s, pid.s)
docs/          PLAN.md (roadmap), CUDA_SPEC.md (GPU design)
pid.asm        Reference 6502 PID controller (historical comparison)
```

## Current state

CPU solver verified optimal programs for 9 tasks:
negate(4), double(4), square(4), add(5), abs(6), relu(6), max(7),
saturating_add(8), is_power2(8).
CUDA solver compiles and runs. Tested on RTX 2060: ~1.4B candidates/sec.

Known parity gaps between CPU and GPU solvers:
- GPU lacks iterative deepening (fixed depth only, may find non-optimal)
- GPU lacks liveness pruning and OEP (explores more dead-end candidates)
- GPU lacks branch instruction generation (can't solve tasks needing conditionals)
- GPU output is raw hex (no disassembly)

Next steps: fix GPU/CPU parity, benchmark on RTX 5070, persist solutions
to file (see issue #1), then Python bindings and value network training.

## ISA quick reference

25 opcodes, 8 registers (r0=0), 16-bit data, 256 words program+data memory.
Fixed 16-bit instruction encoding. Type R: [op:5][rd:3][rs:3][rt:3][pad:2].
Type I: [op:5][rd:3][imm:8]. ZNVC flags. See src/pca.h for full spec.

## Task format

JSON files in tasks/. Each defines input/output ports and test cases:
```json
{"name":"add", "input_ports":[0,1], "output_ports":[0],
 "tests":[{"in":[3,5],"out":[8]}, ...]}
```

## Key design decisions

- GPT Pro feedback: factor policy as P(op)×P(dst|op)×P(src|...), not flat softmax
- GPT Pro feedback: use sublanguage restriction (task-relevant opcodes) to reduce search
- Codex feedback: cap training horizon at 32-64 instructions, curriculum up from there
- Codex feedback: use best-first/A* search, not vanilla MCTS; add CEGIS for verification
- The milestone: "Can a learned value function beat random search at extending partial programs?"
