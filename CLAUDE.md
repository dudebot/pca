# PCA — Predictive Coding in Assembly

## What this is

AI-driven program synthesis on a custom 16-bit ISA (PCA-16). Exhaustively
solve tiny tasks to generate perfect training data, train a value network
to guide search on harder tasks, bootstrap up to program lengths that
brute force can't reach.

Read `docs/STATUS.md` for current progress and next steps.
Read `docs/CUDA_SPEC.md` for GPU solver design.

## Phase Map

| Phase | Goal | Status |
|-------|------|--------|
| 1. Solver | Exhaustive CPU+GPU search, prove optimality | Done |
| 2. Learning | Value network trained on exhaustive data, beats random search | **Active** |
| 3. Bootstrap | Curriculum learning: depth D model helps generate depth D+1 data | Next |
| 4. Branches | Extend to conditional execution (CMP/BR) | Future |
| 5. Verification | CEGIS: adversarial test generation for correctness proofs | Future |

## Build

```bash
make              # builds: pca, enumerate, gen_tasks
make libpca.so    # shared lib for Python ctypes bindings
make test         # run emulator tests

# GPU hybrid solver (requires CUDA 12.8+):
make gpu_hybrid

# Training:
pip install -r requirements.txt
python python/train.py --epochs 10 --batch-size 2048 --lr 3e-4

# Evaluation:
python python/search_eval.py --checkpoint checkpoints/value_v1.pt --split test
```

## Repo layout

```
src/               PCA-16 emulator (pca.h, vm.c, asm.c, main.c)
src/search_state.c Branchless search state abstraction for Phase 2
tasks/             Task specs: spec.h/spec.c + *.json (I/O test cases)
tools/             enumerate.c (CPU solver), gen_tasks.c (synthetic task gen),
                   gen_dataset.sh (batch solve + record driver)
gpu/               CUDA solver (evaluate.cu, wavefront.cu)
python/            dataset.py, model.py, train.py, search_eval.py, build_manifest.py
data/synthetic/    1000 synthetic task JSONs (in git)
data/train/        Binary state records, 81GB (gitignored, regenerable)
data/manifest.json Task manifest with train/val/test splits (in git)
checkpoints/       Model checkpoints (gitignored)
runs/              TensorBoard logs (gitignored)
programs/          Hand-written PCA-16 assembly (fib.s, echo.s, pid.s)
docs/              STATUS.md (progress), CUDA_SPEC.md (GPU design)
```

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

- Factor policy as P(op)×P(dst|op)×P(src|...), not flat softmax
- Sublanguage restriction (task-relevant opcodes) to reduce branching factor
- Cap training horizon at 32-64 instructions, curriculum up from there
- Best-first/A* search, not vanilla MCTS; add CEGIS for verification later
- The milestone: "Can a learned value function beat random search?" — **Yes (Success@8: 51% vs 31%)**
