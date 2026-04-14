# CUDA Exhaustive Solver — Design Spec

## The problem

The CPU solver hits a wall at kernel depth 4 (~76 minutes for is_power2,
3.5B candidates at 0.7M candidates/sec). Depth 5 would take ~16 days.
The bottleneck is NOT evaluation (the VM is fast) — it's the 90% overhead
of recursive enumeration, liveness tracking, and OEP hash lookups.

The fix: move BOTH generation and evaluation to GPU. Each thread
independently generates one candidate program from a flat index,
evaluates it against all test cases, and reports pass/fail.
No recursion, no shared mutable state, no OEP hash tables.

## Target hardware

**RTX 5070 (Blackwell, GB205)**
- 48 SMs, 6144 CUDA cores
- 2048 threads/SM max → 98,304 concurrent threads
- 65,536 registers/SM (32-bit) → ~20 regs/thread at full occupancy
- 12 GB GDDR7, ~672 GB/s bandwidth
- 48 MB L2 cache
- Compute capability 12.0
- Boost clock: ~2512 MHz

## Architecture

```
GPU thread i:
  1. Decompose i into instruction indices for each kernel position
  2. Look up actual instruction encodings from candidate table
  3. Build program: [IN scaffold] + [kernel] + [OUT rX] + [HLT]
  4. For each test case:
     a. Init registers to zero
     b. Set input port values
     c. Execute program (inline switch, unrolled if short)
     d. Check output port against expected
  5. If all tests pass, atomicAdd to solution counter,
     write program to solution buffer
```

### Candidate table

Pre-computed on CPU, uploaded to GPU constant memory.

For v1 (flat indexing, no liveness):
- One candidate array per kernel position, same for all programs
- ~800 entries per position (all valid Type R + Type I instructions)
- 800 × 2 bytes × 8 positions = 12.8 KB → fits in constant memory (64 KB)

For v2 (liveness-aware):
- One candidate array per (position, liveness_mask) pair
- 256 liveness masks × 8 positions = 2048 tables
- ~500 entries avg × 2 bytes = 1 KB per table → 2 MB total
- Stored in global memory, cached in L2 (48 MB)

### Index decomposition

Total candidates at depth K with C candidates per position: C^K.
Each of these K-instruction candidate sets gets tried with each
of 7 output registers (r1-r7), so total programs = C^K × 7.

Thread i decodes:
```
out_reg = (i % 7) + 1
idx = i / 7
for pos = 0..K-1:
    insn_idx = idx % C[pos]
    idx = idx / C[pos]
    kernel[pos] = candidate_table[pos][insn_idx]
```

### VM execution on GPU

Minimal inline VM. No function calls, no memory array (tier 1-2 only),
no call stack. Just registers + flags + PC.

```cuda
__device__ int run_program(
    const uint16_t *prog, int prog_len,
    const uint16_t *port_in, int n_in,
    const uint16_t *port_out_expected, int n_out,
    int max_steps)
{
    uint16_t r[8] = {0};
    uint8_t flags = 0;
    uint8_t pc = 0;

    // set input ports (write to registers directly since
    // IN instructions at start copy ports to regs)
    // Actually: just execute the full program including INs.

    for (int step = 0; step < max_steps && pc < prog_len; step++) {
        uint16_t insn = prog[pc];
        uint8_t op = (insn >> 11) & 0x1F;
        uint8_t rd = (insn >> 8) & 0x07;
        uint8_t rs = (insn >> 5) & 0x07;
        uint8_t rt = (insn >> 2) & 0x07;
        uint8_t imm = insn & 0xFF;
        int8_t simm = (int8_t)imm;
        pc++;

        switch (op) {
        case 0x01: return -1;  // HLT → check output
        case 0x02: r[rd] = r[rs] + r[rt]; break;  // ADD
        case 0x03: r[rd] = r[rs] - r[rt]; break;  // SUB
        // ... (full switch, 25 cases)
        case 0x17: r[rd] = port_in[imm & 7]; break;  // IN
        case 0x18: /* OUT: handled at HLT */ break;
        }
        r[0] = 0;
        // flags computed inline where needed
    }
    return 0;  // didn't halt = fail
}
```

For maximum throughput, the switch should use `__forceinline__` and
the loop should be `#pragma unroll` for known-short programs.

## Performance estimates

### Per-thread cost

For a 5-instruction program with 8 test cases:
- Program fetch: 5 × 16-bit from constant/L1 cache → ~5 cycles
- Decode: 5 × ~4 ops = 20 ops
- Execute: 5 × ~8 ops = 40 ops (worst case: flags computation)
- Test loop overhead: 8 × ~10 ops = 80 ops
- Total: ~200 ops per candidate (8 tests, 5 instructions)
- At IPC ~1.0: ~200 cycles per thread = ~80 ns at 2.5 GHz

### Throughput

```
Concurrent threads:  48 SMs × 2048 threads/SM = 98,304
Cycles per thread:   ~200
Candidates/wave:     98,304 / 8 = 12,288 (8 threads per candidate for tests)
                     OR 98,304 (1 thread does all 8 tests sequentially)
Time per wave:       200 cycles / 2.5 GHz = 80 ns
Waves per second:    12.5M
```

**One thread per candidate (all tests sequential):**
- 98,304 candidates/wave × 12.5M waves/sec = impractical (200 cycles dominates)
- More accurately: 98,304 / (200 cycles / 2.5 GHz) = but many warps are in flight
- Effective: ~98,304 threads × 2.5G cycles/sec / 200 cycles = **1.23B candidates/sec**

**One thread per (candidate × test):**
- 98,304 threads / 8 tests = 12,288 candidates in flight
- Same computation but better for short programs
- Effective: similar or slightly lower due to reduction overhead

### Target performance (RTX 5070)

| Metric | Conservative | Target | Optimistic |
|--------|-------------|--------|------------|
| VM steps/sec | 5B | 20B | 100B |
| Candidates/sec (5-insn, 8 tests) | 125M | 500M | 2.5B |
| Candidates/sec (8-insn, 8 tests) | 50M | 200M | 1B |

### Comparison: CPU laptop vs GPU desktop

| Task | CPU (0.7M/s) | GPU target (500M/s) | Speedup |
|------|-------------|---------------------|---------|
| negate (d=1, 449) | instant | instant | - |
| abs (d=3, 30M) | 17 sec | 0.06 sec | 280x |
| max (d=3, 284M) | 175 sec | 0.6 sec | 290x |
| is_power2 (d=4, 3.5B) | 76 min | 7 sec | 650x |
| depth 5 (~1T) | ~16 days | ~33 min | 700x |
| depth 6 (~500T) | ~22 years | ~12 days | 700x |

### With sublanguage restriction (200 candidates/step)

| Depth | Candidates | GPU @ 500M/s | Enables |
|-------|-----------|--------------|---------|
| 4 | 1.6B | 3 sec | all tier 2 |
| 5 | 320B | 10 min | tier 3 (collatz, gcd) |
| 6 | 64T | 1.5 days | complex algorithms |
| 7 | 12.8P | 296 days | needs neural guidance |

Depth 5 with sublanguage restriction on GPU is the sweet spot:
10 minutes to exhaustively solve any task with optimal ≤ 7 instructions.

## Implementation plan

### Files

```
gpu/
  evaluate.cu     — CUDA kernel: flat-index generation + evaluation
  host.cu         — Host driver: launch kernels, collect results
  candidate.h     — Candidate table generation (shared with CPU)
  Makefile.gpu     — CUDA build (nvcc)
```

### Dependencies

- CUDA Toolkit 12.x (for compute capability 12.0)
- nvcc compiler
- No other libraries needed

### Build

```bash
nvcc -O3 -arch=sm_120 -o gpu_enumerate gpu/evaluate.cu gpu/host.cu
```

Or if the 5070 uses a different SM version (might be sm_100 or sm_120
depending on Blackwell mapping):
```bash
nvcc -O3 -arch=native -o gpu_enumerate gpu/evaluate.cu gpu/host.cu
```

### Phases

**Phase 1: Flat-index evaluator (v1)**
- Fixed candidate table per position (no liveness)
- One CUDA thread per candidate
- Each thread does all test cases sequentially
- Launch grid: total_candidates / 256 blocks, 256 threads/block
- Target: 500M candidates/sec

**Phase 2: Liveness-aware generation (v2)**
- Candidate tables indexed by liveness mask
- Each thread computes liveness as it decodes instructions
- Tighter candidate sets → less wasted work
- Target: same throughput, but 3-5x fewer total candidates

**Phase 3: Multi-kernel pipeline**
- Stream multiple kernel depths concurrently
- Use CUDA streams for overlap
- CPU pre-processes next batch while GPU runs current
- Marginal improvement but nice for long runs

### Verification

After GPU finds solutions, verify them on CPU:
1. Disassemble the solution
2. Run it through the CPU emulator on all test cases
3. Confirm match

This catches any GPU-specific bugs (floating-point-style issues
don't apply since we're all integer, but memory race conditions
in the atomic solution write could).

### How to know if the implementation is off

If measured throughput is:
- **< 50M candidates/sec**: something is very wrong (memory bottleneck, occupancy issue, branch divergence killing perf). Profile with nsight.
- **50M - 200M**: acceptable but likely occupancy or divergence issue.
- **200M - 1B**: good, close to target.
- **> 1B**: excellent, likely memory-bound.

The most likely perf killers, in order:
1. Register spill to local memory (check with --ptxas-options=-v)
2. Warp divergence in the switch dispatch
3. Low occupancy from shared memory or register pressure
4. Global memory traffic from reading candidate tables
