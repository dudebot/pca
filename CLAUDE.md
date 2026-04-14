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

CPU solver verified optimal programs for 9 tasks (4-8 instructions each).
CUDA solver written but not yet tested (no GPU on dev laptop).

The immediate next step is building and benchmarking the CUDA solver on a
machine with an RTX 5070. Target: 500M candidates/sec (700x over CPU).
After that: Python bindings, value network training, A* search.

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
