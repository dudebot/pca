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

## The premise

If the reason we abandoned assembly was that humans can't manage the complexity — and machines can manage complexity differently than humans — what happens when you give a system *computational proprioception*? Not reasoning about machine state through text, but a learned sensory modality for the reachable state space itself.

The VM state is a 266-element vector: 8 registers + 256 memory words + PC + SP. That's small enough for a neural network to digest as a direct observation. The action space is the instruction set. The reward is program correctness. The question is whether a system trained on this loop develops the kind of intuition for computational structure that lets it write assembly no human would consider — an AlphaGo for programs.

A PID controller is the simplest version of the predictive coding loop: sense, predict, correct. Start here.
