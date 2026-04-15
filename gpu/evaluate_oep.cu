/*
 * evaluate_oep.cu — CUDA exhaustive solver with OEP for PCA-16
 *
 * Same flat-index approach as evaluate.cu, with observational equivalence
 * pruning (OEP) added inside the kernel. Each thread incrementally
 * maintains per-test register state as it decodes instructions, hashing
 * the combined state at each position. If another thread already produced
 * the same state (same hash in a global table via atomicCAS), the thread
 * bails — any suffix that works for one works for the other.
 *
 * OEP is only applied for the first OEP_MAX_POS kernel positions and
 * only for straight-line (non-branch) prefixes. After a BR instruction,
 * OEP is disabled for the remainder of the decode.
 *
 * Build:
 *   nvcc -O3 -arch=native -o gpu_enumerate_oep gpu/evaluate_oep.cu tasks/spec.c src/vm.c
 */

#include <cuda_runtime.h>

#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../tasks/spec.h"

#define MAX_CANDIDATES_PER_POS 4096
#define MAX_KERNEL_DEPTH 8
#define MAX_PROG_LEN 16
#define MAX_GPU_SOLUTIONS 1024
#define THREADS_PER_BLOCK 256

/* --- OEP configuration --- */
#define OEP_MAX_POS     3                       /* OEP for kernel positions 0..2 */
#define OEP_TABLE_BITS  23                      /* 8M entries per position */
#define OEP_TABLE_SIZE  (1ULL << OEP_TABLE_BITS)
#define OEP_TABLE_MASK  (OEP_TABLE_SIZE - 1)
#define OEP_NUM_PROBES  8

__constant__ uint16_t d_candidates[MAX_CANDIDATES_PER_POS];
__constant__ int d_num_candidates;

__constant__ uint16_t d_test_in[MAX_TESTS][MAX_IO_PORTS];
__constant__ uint16_t d_test_out[MAX_TESTS][MAX_IO_PORTS];
__constant__ int d_num_tests;
__constant__ int d_num_inputs;
__constant__ int d_num_outputs;
__constant__ uint8_t d_input_ports[MAX_IO_PORTS];
__constant__ uint8_t d_output_ports[MAX_IO_PORTS];
__constant__ int d_max_steps;

__constant__ uint16_t d_scaffold[PCA_NUM_REGS];
__constant__ int d_scaffold_len;
__constant__ uint8_t d_initial_live;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "%s:%d: CUDA error: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err__));          \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

typedef struct {
    uint16_t words[MAX_PROG_LEN];
    int len;
    int passed_tests;
} host_solution_t;

__device__ __forceinline__
int reg_is_live(uint8_t live, uint8_t reg)
{
    return reg == 0 || (live & (uint8_t)(1u << reg));
}

/* --- OEP: execute one instruction on a register state (inline) --- */
__device__ __forceinline__
void oep_exec_one(uint16_t insn, uint16_t *r, uint8_t *flags)
{
    uint8_t op  = INSN_OP(insn);
    uint8_t rd  = INSN_RD(insn);
    uint8_t rs  = INSN_RS(insn);
    uint8_t rt  = INSN_RT(insn);
    uint8_t imm = INSN_IMM8(insn);
    int8_t simm = INSN_SIMM8(insn);

    switch (op) {
    case OP_ADD: {
        uint32_t full = (uint32_t)r[rs] + (uint32_t)r[rt];
        uint16_t res = (uint16_t)full;
        *flags = 0;
        if (res == 0) *flags |= FLAG_Z;
        if (res & 0x8000) *flags |= FLAG_N;
        if (full > 0xFFFF) *flags |= FLAG_C;
        if (~(r[rs] ^ r[rt]) & (r[rs] ^ res) & 0x8000) *flags |= FLAG_V;
        r[rd] = res;
        break;
    }
    case OP_SUB: {
        uint16_t a = r[rs], b = r[rt], res = a - b;
        *flags = 0;
        if (res == 0) *flags |= FLAG_Z;
        if (res & 0x8000) *flags |= FLAG_N;
        if (a >= b) *flags |= FLAG_C;
        if ((a ^ b) & (a ^ res) & 0x8000) *flags |= FLAG_V;
        r[rd] = res;
        break;
    }
    case OP_MUL: {
        uint32_t prod = (uint32_t)r[rs] * (uint32_t)r[rt];
        r[rd] = (uint16_t)prod;
        *flags = 0;
        if (r[rd] == 0) *flags |= FLAG_Z;
        if (r[rd] & 0x8000) *flags |= FLAG_N;
        if (prod > 0xFFFF) *flags |= FLAG_C;
        break;
    }
    case OP_DIV:
        if (r[rt] == 0) { *flags = 0xFF; return; } /* fault sentinel */
        r[rd] = (uint16_t)((int16_t)r[rs] / (int16_t)r[rt]);
        *flags = 0;
        if (r[rd] == 0) *flags |= FLAG_Z;
        if (r[rd] & 0x8000) *flags |= FLAG_N;
        break;
    case OP_MOD:
        if (r[rt] == 0) { *flags = 0xFF; return; }
        r[rd] = (uint16_t)((int16_t)r[rs] % (int16_t)r[rt]);
        *flags = 0;
        if (r[rd] == 0) *flags |= FLAG_Z;
        if (r[rd] & 0x8000) *flags |= FLAG_N;
        break;
    case OP_AND:
        r[rd] = r[rs] & r[rt];
        *flags = (r[rd] == 0 ? FLAG_Z : 0) | (r[rd] & 0x8000 ? FLAG_N : 0);
        break;
    case OP_OR:
        r[rd] = r[rs] | r[rt];
        *flags = (r[rd] == 0 ? FLAG_Z : 0) | (r[rd] & 0x8000 ? FLAG_N : 0);
        break;
    case OP_XOR:
        r[rd] = r[rs] ^ r[rt];
        *flags = (r[rd] == 0 ? FLAG_Z : 0) | (r[rd] & 0x8000 ? FLAG_N : 0);
        break;
    case OP_SHL:
        r[rd] = r[rs] << (r[rt] & 0xF);
        *flags = (r[rd] == 0 ? FLAG_Z : 0) | (r[rd] & 0x8000 ? FLAG_N : 0);
        break;
    case OP_SHR:
        r[rd] = r[rs] >> (r[rt] & 0xF);
        *flags = (r[rd] == 0 ? FLAG_Z : 0) | (r[rd] & 0x8000 ? FLAG_N : 0);
        break;
    case OP_ASR:
        r[rd] = (uint16_t)((int16_t)r[rs] >> (r[rt] & 0xF));
        *flags = (r[rd] == 0 ? FLAG_Z : 0) | (r[rd] & 0x8000 ? FLAG_N : 0);
        break;
    case OP_CMP: {
        uint16_t a = r[rd], b = r[rs], res = a - b;
        *flags = 0;
        if (res == 0) *flags |= FLAG_Z;
        if (res & 0x8000) *flags |= FLAG_N;
        if (a >= b) *flags |= FLAG_C;
        if ((a ^ b) & (a ^ res) & 0x8000) *flags |= FLAG_V;
        break;
    }
    case OP_LDI:
        r[rd] = imm;
        *flags = 0; /* LDI doesn't set flags in cpu VM, keep consistent */
        break;
    case OP_ADDI: {
        uint16_t ext = (uint16_t)(int16_t)simm;
        uint32_t full = (uint32_t)r[rd] + (uint32_t)ext;
        uint16_t res = (uint16_t)full;
        *flags = 0;
        if (res == 0) *flags |= FLAG_Z;
        if (res & 0x8000) *flags |= FLAG_N;
        if (full > 0xFFFF) *flags |= FLAG_C;
        if (~(r[rd] ^ ext) & (r[rd] ^ res) & 0x8000) *flags |= FLAG_V;
        r[rd] = res;
        break;
    }
    default:
        break;
    }
    r[0] = 0;
}

/* --- OEP: FNV-1a 64-bit hash of all test register states + flags --- */
__device__ __forceinline__
uint64_t oep_hash_states(uint16_t oep_regs[][PCA_NUM_REGS],
                         uint8_t *oep_flags, int num_tests)
{
    uint64_t h = 14695981039346656037ULL; /* FNV-1a offset basis */
    for (int t = 0; t < num_tests; t++) {
        for (int r = 0; r < PCA_NUM_REGS; r++) {
            h ^= oep_regs[t][r] & 0xFF;  h *= 1099511628211ULL;
            h ^= oep_regs[t][r] >> 8;    h *= 1099511628211ULL;
        }
        h ^= oep_flags[t]; h *= 1099511628211ULL;
    }
    return h;
}

/* --- OEP: atomic insert-if-absent with linear probing --- */
__device__ __forceinline__
int oep_check_insert(unsigned long long *table, uint64_t hash)
{
    unsigned long long key = (unsigned long long)(hash | 1ULL);
    unsigned long long slot = ((unsigned long long)(hash >> 1)) & OEP_TABLE_MASK;

    for (int probe = 0; probe < OEP_NUM_PROBES; probe++) {
        unsigned long long idx = (slot + probe) & OEP_TABLE_MASK;
        unsigned long long old = atomicCAS(&table[idx], 0ULL, key);
        if (old == 0ULL) return 0; /* new state, inserted */
        if (old == key) return 1;  /* already seen, prune */
    }
    return 0; /* table full at this slot, don't prune (conservative) */
}

__device__ __forceinline__
int vm_run(const uint16_t *prog, int prog_len,
           const uint16_t *port_in, uint16_t *port_out, int max_steps)
{
    uint16_t r[PCA_NUM_REGS] = {0};
    uint16_t ports[PCA_NUM_PORTS] = {0};
    uint8_t flags = 0;
    uint8_t pc = 0;

#pragma unroll
    for (int i = 0; i < PCA_NUM_PORTS; i++) {
        ports[i] = port_in[i];
    }

    for (int step = 0; step < max_steps; step++) {
        if (pc >= prog_len) return PCA_FAULT;

        uint16_t insn = prog[pc];
        uint8_t op  = INSN_OP(insn);
        uint8_t rd  = INSN_RD(insn);
        uint8_t rs  = INSN_RS(insn);
        uint8_t rt  = INSN_RT(insn);
        uint8_t imm = INSN_IMM8(insn);
        int8_t simm = INSN_SIMM8(insn);

        pc++;

        switch (op) {
        case OP_NOP:
            break;
        case OP_HLT:
#pragma unroll
            for (int i = 0; i < PCA_NUM_PORTS; i++) {
                port_out[i] = ports[i];
            }
            return PCA_HALTED;
        case OP_ADD: {
            uint32_t full = (uint32_t)r[rs] + (uint32_t)r[rt];
            uint16_t res = (uint16_t)full;
            flags = 0;
            if (res == 0) flags |= FLAG_Z;
            if (res & 0x8000) flags |= FLAG_N;
            if (full > 0xFFFF) flags |= FLAG_C;
            if (~(r[rs] ^ r[rt]) & (r[rs] ^ res) & 0x8000) flags |= FLAG_V;
            r[rd] = res;
            break;
        }
        case OP_SUB: {
            uint16_t a = r[rs];
            uint16_t b = r[rt];
            uint16_t res = a - b;
            flags = 0;
            if (res == 0) flags |= FLAG_Z;
            if (res & 0x8000) flags |= FLAG_N;
            if (a >= b) flags |= FLAG_C;
            if ((a ^ b) & (a ^ res) & 0x8000) flags |= FLAG_V;
            r[rd] = res;
            break;
        }
        case OP_MUL: {
            uint32_t prod = (uint32_t)r[rs] * (uint32_t)r[rt];
            r[rd] = (uint16_t)prod;
            flags = 0;
            if (r[rd] == 0) flags |= FLAG_Z;
            if (r[rd] & 0x8000) flags |= FLAG_N;
            if (prod > 0xFFFF) flags |= FLAG_C;
            break;
        }
        case OP_DIV:
            if (r[rt] == 0) return PCA_FAULT;
            r[rd] = (uint16_t)((int16_t)r[rs] / (int16_t)r[rt]);
            flags = 0;
            if (r[rd] == 0) flags |= FLAG_Z;
            if (r[rd] & 0x8000) flags |= FLAG_N;
            break;
        case OP_MOD:
            if (r[rt] == 0) return PCA_FAULT;
            r[rd] = (uint16_t)((int16_t)r[rs] % (int16_t)r[rt]);
            flags = 0;
            if (r[rd] == 0) flags |= FLAG_Z;
            if (r[rd] & 0x8000) flags |= FLAG_N;
            break;
        case OP_AND:
            r[rd] = r[rs] & r[rt];
            flags = (r[rd] == 0 ? FLAG_Z : 0) | (r[rd] & 0x8000 ? FLAG_N : 0);
            break;
        case OP_OR:
            r[rd] = r[rs] | r[rt];
            flags = (r[rd] == 0 ? FLAG_Z : 0) | (r[rd] & 0x8000 ? FLAG_N : 0);
            break;
        case OP_XOR:
            r[rd] = r[rs] ^ r[rt];
            flags = (r[rd] == 0 ? FLAG_Z : 0) | (r[rd] & 0x8000 ? FLAG_N : 0);
            break;
        case OP_SHL:
            r[rd] = r[rs] << (r[rt] & 0xF);
            flags = (r[rd] == 0 ? FLAG_Z : 0) | (r[rd] & 0x8000 ? FLAG_N : 0);
            break;
        case OP_SHR:
            r[rd] = r[rs] >> (r[rt] & 0xF);
            flags = (r[rd] == 0 ? FLAG_Z : 0) | (r[rd] & 0x8000 ? FLAG_N : 0);
            break;
        case OP_ASR:
            r[rd] = (uint16_t)((int16_t)r[rs] >> (r[rt] & 0xF));
            flags = (r[rd] == 0 ? FLAG_Z : 0) | (r[rd] & 0x8000 ? FLAG_N : 0);
            break;
        case OP_CMP: {
            uint16_t a = r[rd];
            uint16_t b = r[rs];
            uint16_t res = a - b;
            flags = 0;
            if (res == 0) flags |= FLAG_Z;
            if (res & 0x8000) flags |= FLAG_N;
            if (a >= b) flags |= FLAG_C;
            if ((a ^ b) & (a ^ res) & 0x8000) flags |= FLAG_V;
            break;
        }
        case OP_LDI:
            r[rd] = imm;
            break;
        case OP_LUI:
            r[rd] = (r[rd] & 0x00FF) | ((uint16_t)imm << 8);
            break;
        case OP_ADDI: {
            uint16_t ext = (uint16_t)(int16_t)simm;
            uint32_t full = (uint32_t)r[rd] + (uint32_t)ext;
            uint16_t res = (uint16_t)full;
            flags = 0;
            if (res == 0) flags |= FLAG_Z;
            if (res & 0x8000) flags |= FLAG_N;
            if (full > 0xFFFF) flags |= FLAG_C;
            if (~(r[rd] ^ ext) & (r[rd] ^ res) & 0x8000) flags |= FLAG_V;
            r[rd] = res;
            break;
        }
        case OP_BR: {
            int take = 0;
            switch (rd) {
            case COND_AL:  take = 1; break;
            case COND_EQ:  take = (flags & FLAG_Z) != 0; break;
            case COND_NE:  take = (flags & FLAG_Z) == 0; break;
            case COND_LT:  take = ((flags & FLAG_N) != 0) != ((flags & FLAG_V) != 0); break;
            case COND_GE:  take = ((flags & FLAG_N) != 0) == ((flags & FLAG_V) != 0); break;
            case COND_LTU: take = (flags & FLAG_C) == 0; break;
            case COND_GEU: take = (flags & FLAG_C) != 0; break;
            default: return PCA_FAULT;
            }
            if (take) pc = (uint8_t)((int)pc + simm);
            break;
        }
        case OP_IN:
            r[rd] = ports[imm & 7];
            break;
        case OP_OUT:
            ports[imm & 7] = r[rd];
            break;
        case OP_LD:
        case OP_ST:
        case OP_JMP:
        case OP_CALL:
        case OP_RET:
            return PCA_FAULT;
        default:
            return PCA_FAULT;
        }
        r[0] = 0;
    }

    return PCA_TIMEOUT;
}

__global__
void evaluate_kernel(
    uint64_t base_gid,
    uint64_t launch_candidates,
    int kernel_depth,
    uint8_t output_port,
    uint16_t *d_solutions,
    int *d_solution_lens,
    int *d_num_solutions,
    int max_solutions,
    unsigned long long *oep_table_0,
    unsigned long long *oep_table_1,
    unsigned long long *oep_table_2)
{
    uint64_t local_gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (local_gid >= launch_candidates) return;

    uint64_t gid = base_gid + local_gid;
    int nc = d_num_candidates;
    int scaffold_len = d_scaffold_len;
    int num_tests = d_num_tests;

    int out_reg = (int)(gid % (PCA_NUM_REGS - 1)) + 1;
    uint64_t idx = gid / (PCA_NUM_REGS - 1);

    uint16_t prog[MAX_PROG_LEN];
    int plen = 0;
    uint8_t live = d_initial_live;
    int skip_live = 0;
    int skip_oep = 0;

    /* OEP: per-test register state, maintained incrementally */
    uint16_t oep_regs[MAX_TESTS][PCA_NUM_REGS];
    uint8_t  oep_flags[MAX_TESTS];

    /* Initialize OEP state from scaffold (IN instructions) */
    for (int t = 0; t < num_tests; t++) {
        for (int r = 0; r < PCA_NUM_REGS; r++) oep_regs[t][r] = 0;
        for (int i = 0; i < scaffold_len; i++)
            oep_regs[t][i + 1] = d_test_in[t][i];
        oep_flags[t] = 0;
    }

    /* OEP table pointers indexed by position */
    unsigned long long *oep_tables[OEP_MAX_POS];
    oep_tables[0] = oep_table_0;
    oep_tables[1] = oep_table_1;
    oep_tables[2] = oep_table_2;

#pragma unroll
    for (int i = 0; i < PCA_NUM_REGS; i++) {
        if (i < scaffold_len) prog[plen++] = d_scaffold[i];
    }

    for (int k = 0; k < kernel_depth; k++) {
        int ci = (int)(idx % (uint64_t)nc);
        idx /= (uint64_t)nc;
        uint16_t insn = d_candidates[ci];
        uint8_t op = INSN_OP(insn);
        uint8_t rd = INSN_RD(insn);
        uint8_t rs = INSN_RS(insn);
        uint8_t rt = INSN_RT(insn);

        if (op == OP_BR) {
            int8_t off = INSN_SIMM8(insn);
            if (off < 1 || off > kernel_depth - k - 1) return;
            skip_live = 1;
            skip_oep = 1;
        } else if (!skip_live) {
            switch (op) {
            case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_MOD:
            case OP_AND: case OP_OR:  case OP_XOR: case OP_SHL: case OP_SHR:
            case OP_ASR:
                if (!reg_is_live(live, rs) || !reg_is_live(live, rt)) return;
                if (rd != 0) live |= (uint8_t)(1u << rd);
                break;
            case OP_CMP:
                if (!reg_is_live(live, rd) || !reg_is_live(live, rs)) return;
                break;
            case OP_LDI:
                if (rd != 0) live |= (uint8_t)(1u << rd);
                break;
            case OP_ADDI:
                if (!reg_is_live(live, rd)) return;
                break;
            default:
                return;
            }
        } else {
            if (op >= OP_ADD && op <= OP_ASR) {
                if (rd != 0) live |= (uint8_t)(1u << rd);
            } else if (op == OP_LDI) {
                if (rd != 0) live |= (uint8_t)(1u << rd);
            } else if (op != OP_CMP && op != OP_ADDI) {
                return;
            }
        }

        /* OEP: execute instruction on all test states and check hash */
        if (!skip_oep && k < OEP_MAX_POS && oep_tables[k] != NULL) {
            int faulted = 0;
            for (int t = 0; t < num_tests; t++) {
                oep_exec_one(insn, oep_regs[t], &oep_flags[t]);
                if (oep_flags[t] == 0xFF) { faulted = 1; break; }
            }
            if (faulted) return; /* prefix causes fault on some test → can't solve */

            uint64_t h = oep_hash_states(oep_regs, oep_flags, num_tests);
            if (oep_check_insert(oep_tables[k], h)) return; /* pruned */
        }

        prog[plen++] = insn;
    }

    if (!skip_live && !reg_is_live(live, (uint8_t)out_reg)) return;
    prog[plen++] = ENCODE_I(OP_OUT, out_reg, output_port);
    prog[plen++] = ENCODE_R(OP_HLT, 0, 0, 0);

    int passed = 0;
    for (int t = 0; t < d_num_tests; t++) {
        uint16_t port_in[PCA_NUM_PORTS] = {0};
        uint16_t port_out[PCA_NUM_PORTS];

#pragma unroll
        for (int i = 0; i < MAX_IO_PORTS; i++) {
            if (i < d_num_inputs) {
                port_in[d_input_ports[i]] = d_test_in[t][i];
            }
        }

        int status = vm_run(prog, plen, port_in, port_out, d_max_steps);
        if (status != PCA_HALTED) {
            break;
        }

        int ok = 1;
#pragma unroll
        for (int i = 0; i < MAX_IO_PORTS; i++) {
            if (i < d_num_outputs &&
                port_out[d_output_ports[i]] != d_test_out[t][i]) {
                ok = 0;
            }
        }
        if (!ok) {
            break;
        }
        passed++;
    }

    if (passed == d_num_tests) {
        /* Read before atomic to avoid overflowing the counter when
         * billions of threads find solutions (e.g. depth 4+). */
        if (*(volatile int *)d_num_solutions >= max_solutions) return;
        int slot = atomicAdd(d_num_solutions, 1);
        if (slot < max_solutions) {
            for (int i = 0; i < plen; i++) {
                d_solutions[slot * MAX_PROG_LEN + i] = prog[i];
            }
            d_solution_lens[slot] = plen;
        }
    }
}

static int gen_candidate_table(uint16_t *table, int max)
{
    int n = 0;

    static const uint8_t alu_ops[] = {
        OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD,
        OP_AND, OP_OR,  OP_XOR, OP_SHL, OP_SHR, OP_ASR
    };
    static const int commut[] = {1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0};

    for (int oi = 0; oi < 11; oi++) {
        uint8_t op = alu_ops[oi];
        for (int rd = 1; rd < PCA_NUM_REGS && n < max; rd++) {
            for (int rs = 0; rs < PCA_NUM_REGS && n < max; rs++) {
                int rt0 = commut[oi] ? rs : 0;
                for (int rt = rt0; rt < PCA_NUM_REGS && n < max; rt++) {
                    if (rs == 0 && rt == 0) continue;
                    if (op == OP_ADD && rs == rd && rt == 0) continue;
                    if (op == OP_ADD && rt == rd && rs == 0) continue;
                    if (op == OP_SUB && rs == rd && rt == 0) continue;
                    if (op == OP_MUL && (rs == 0 || rt == 0)) continue;
                    if ((op == OP_DIV || op == OP_MOD) && rt == 0) continue;
                    if ((op == OP_SHL || op == OP_SHR || op == OP_ASR) && rt == 0) continue;
                    table[n++] = ENCODE_R(op, rd, rs, rt);
                }
            }
        }
    }

    for (int rd = 0; rd < PCA_NUM_REGS && n < max; rd++) {
        for (int rs = 0; rs < PCA_NUM_REGS && n < max; rs++) {
            if (rd == 0 && rs == 0) continue;
            if (rd == rs) continue;
            table[n++] = ENCODE_R(OP_CMP, rd, rs, 0);
        }
    }

    static const uint8_t imms[] = {0, 1, 2, 3, 4, 5, 8, 16, 32, 64, 128, 255};
    for (int rd = 1; rd < PCA_NUM_REGS && n < max; rd++) {
        for (int i = 0; i < (int)(sizeof(imms) / sizeof(imms[0])) && n < max; i++) {
            table[n++] = ENCODE_I(OP_LDI, rd, imms[i]);
        }
    }

    static const int8_t addis[] = {-1, 1, -2, 2, -3, 3};
    for (int rd = 1; rd < PCA_NUM_REGS && n < max; rd++) {
        for (int i = 0; i < (int)(sizeof(addis) / sizeof(addis[0])) && n < max; i++) {
            table[n++] = ENCODE_I(OP_ADDI, rd, (uint8_t)addis[i]);
        }
    }

    /* BR cond, +offset (forward only, conditional only, no COND_AL) */
    for (int cond = 1; cond <= 6 && n < max; cond++) {
        for (int off = 1; off < MAX_KERNEL_DEPTH && n < max; off++) {
            table[n++] = ENCODE_I(OP_BR, cond, (uint8_t)(int8_t)off);
        }
    }

    return n;
}

static int u64_mul_overflow(uint64_t a, uint64_t b, uint64_t *out)
{
    if (a != 0 && b > UINT64_MAX / a) {
        return 1;
    }
    *out = a * b;
    return 0;
}

static void upload_task_constants(const task_spec_t *spec)
{
    uint16_t h_test_in[MAX_TESTS][MAX_IO_PORTS] = {{0}};
    uint16_t h_test_out[MAX_TESTS][MAX_IO_PORTS] = {{0}};
    uint16_t h_scaffold[PCA_NUM_REGS] = {0};
    uint8_t initial_live = 0;

    for (int t = 0; t < spec->num_tests; t++) {
        for (int i = 0; i < spec->num_inputs; i++) {
            h_test_in[t][i] = spec->tests[t].in[i];
        }
        for (int i = 0; i < spec->num_outputs; i++) {
            h_test_out[t][i] = spec->tests[t].out[i];
        }
    }

    for (int i = 0; i < spec->num_inputs; i++) {
        h_scaffold[i] = ENCODE_I(OP_IN, i + 1, spec->input_ports[i]);
        initial_live |= (uint8_t)(1u << (i + 1));
    }

    CUDA_CHECK(cudaMemcpyToSymbol(d_test_in, h_test_in, sizeof(h_test_in)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_test_out, h_test_out, sizeof(h_test_out)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_tests, &spec->num_tests, sizeof(spec->num_tests)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_inputs, &spec->num_inputs, sizeof(spec->num_inputs)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_outputs, &spec->num_outputs, sizeof(spec->num_outputs)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_input_ports, spec->input_ports, sizeof(spec->input_ports)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_output_ports, spec->output_ports, sizeof(spec->output_ports)));

    int max_steps = (int)spec->max_cycles;
    CUDA_CHECK(cudaMemcpyToSymbol(d_max_steps, &max_steps, sizeof(max_steps)));

    int scaffold_len = spec->num_inputs;
    CUDA_CHECK(cudaMemcpyToSymbol(d_scaffold, h_scaffold, sizeof(h_scaffold)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_scaffold_len, &scaffold_len, sizeof(scaffold_len)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_initial_live, &initial_live, sizeof(initial_live)));
}

static void print_program_words(const uint16_t *prog, int len)
{
    for (int i = 0; i < len; i++) {
        printf("    %02d: 0x%04X\n", i, prog[i]);
    }
}

static double elapsed_seconds(const struct timespec *start, const struct timespec *end)
{
    return (double)(end->tv_sec - start->tv_sec) +
           (double)(end->tv_nsec - start->tv_nsec) * 1e-9;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: gpu_enumerate <task.json> <max_kernel_depth> [max_solutions]\n");
        return 1;
    }

    const char *task_path = argv[1];
    int max_depth = atoi(argv[2]);
    int max_solutions = (argc >= 4) ? atoi(argv[3]) : 32;
    if (max_depth < 0 || max_depth > MAX_KERNEL_DEPTH) {
        fprintf(stderr, "max_kernel_depth must be in [0, %d]\n", MAX_KERNEL_DEPTH);
        return 1;
    }
    if (max_solutions <= 0 || max_solutions > MAX_GPU_SOLUTIONS) {
        fprintf(stderr, "max_solutions must be in [1, %d]\n", MAX_GPU_SOLUTIONS);
        return 1;
    }

    task_spec_t spec;
    if (task_load(task_path, &spec) < 0) {
        return 1;
    }
    if (spec.num_inputs > PCA_NUM_REGS - 1) {
        fprintf(stderr, "task requires %d inputs, but only %d scaffold registers are available\n",
                spec.num_inputs, PCA_NUM_REGS - 1);
        return 1;
    }
    if (spec.num_tests <= 0 || spec.num_tests > MAX_TESTS) {
        fprintf(stderr, "task has invalid test count: %d\n", spec.num_tests);
        return 1;
    }
    if (spec.num_outputs <= 0 || spec.num_outputs > MAX_IO_PORTS) {
        fprintf(stderr, "task has invalid output count: %d\n", spec.num_outputs);
        return 1;
    }
    if (spec.num_outputs != 1) {
        fprintf(stderr, "multi-output tasks are not supported by the current OUT+HLT program template\n");
        return 1;
    }
    if (spec.max_cycles > INT32_MAX) {
        fprintf(stderr, "task max_cycles exceeds GPU VM limit: %" PRIu64 "\n", spec.max_cycles);
        return 1;
    }
    if (spec.num_inputs + max_depth + 2 > MAX_PROG_LEN) {
        fprintf(stderr, "program length exceeds MAX_PROG_LEN=%d\n", MAX_PROG_LEN);
        return 1;
    }

    uint16_t h_candidates[MAX_CANDIDATES_PER_POS];
    int nc = gen_candidate_table(h_candidates, MAX_CANDIDATES_PER_POS);
    if (nc <= 0) {
        fprintf(stderr, "candidate generation failed\n");
        return 1;
    }

    CUDA_CHECK(cudaMemcpyToSymbol(d_candidates, h_candidates, nc * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_candidates, &nc, sizeof(nc)));
    upload_task_constants(&spec);

    uint16_t *d_solutions = NULL;
    int *d_solution_lens = NULL;
    int *d_num_solutions = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_solutions,
                          (size_t)max_solutions * MAX_PROG_LEN * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_solution_lens, (size_t)max_solutions * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_num_solutions, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_num_solutions, 0, sizeof(int)));

    /* OEP tables: one per prefix position, 8M entries × 8 bytes = 64MB each */
    unsigned long long *d_oep_tables[OEP_MAX_POS] = {NULL};
    size_t oep_table_bytes = OEP_TABLE_SIZE * sizeof(unsigned long long);
    for (int p = 0; p < OEP_MAX_POS; p++) {
        CUDA_CHECK(cudaMalloc((void **)&d_oep_tables[p], oep_table_bytes));
    }
    printf("OEP: %d tables × %zu MB = %zu MB\n",
           OEP_MAX_POS, oep_table_bytes / (1024 * 1024),
           (size_t)OEP_MAX_POS * oep_table_bytes / (1024 * 1024));

    int device = 0;
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    printf("PCA-16 CUDA exhaustive solver\n");
    printf("  task: %s (%s)\n", task_path, spec.name);
    printf("  gpu: %s\n", props.name);
    printf("  tests: %d  inputs: %d  outputs: %d  max_cycles: %" PRIu64 "\n",
           spec.num_tests, spec.num_inputs, spec.num_outputs, spec.max_cycles);
    printf("  candidates/position: %d  max kernel depth: %d\n", nc, max_depth);
    printf("  target throughput reference: 500M candidates/sec\n");

    const uint64_t max_blocks_per_launch = 1u << 20;
    const uint64_t max_candidates_per_launch = max_blocks_per_launch * THREADS_PER_BLOCK;

    struct timespec t0;
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int found_depth = -1;
    int total_hits = 0;
    int stored = 0;
    uint64_t total_candidates_all_depths = 0;
    host_solution_t *solutions = NULL;

    for (int depth = 0; depth <= max_depth; depth++) {
        uint64_t total_candidates = 1;
        for (int i = 0; i < depth; i++) {
            if (u64_mul_overflow(total_candidates, (uint64_t)nc, &total_candidates)) {
                fprintf(stderr, "candidate count overflow while computing nc^depth\n");
                free(solutions);
                CUDA_CHECK(cudaFree(d_solutions));
                CUDA_CHECK(cudaFree(d_solution_lens));
                CUDA_CHECK(cudaFree(d_num_solutions));
                return 1;
            }
        }
        if (u64_mul_overflow(total_candidates, (uint64_t)(PCA_NUM_REGS - 1), &total_candidates)) {
            fprintf(stderr, "candidate count overflow while applying output-register choices\n");
            free(solutions);
            CUDA_CHECK(cudaFree(d_solutions));
            CUDA_CHECK(cudaFree(d_solution_lens));
            CUDA_CHECK(cudaFree(d_num_solutions));
            return 1;
        }
        total_candidates_all_depths += total_candidates;

        printf("\n--- kernel depth %d (total program length %d) ---\n",
               depth, spec.num_inputs + depth + 2);
        printf("  total programs: %" PRIu64 "\n", total_candidates);

        CUDA_CHECK(cudaMemset(d_num_solutions, 0, sizeof(int)));

        /* Clear OEP tables for this depth */
        for (int p = 0; p < OEP_MAX_POS && p < depth; p++)
            CUDA_CHECK(cudaMemset(d_oep_tables[p], 0, oep_table_bytes));

        struct timespec depth_start;
        struct timespec depth_end;
        clock_gettime(CLOCK_MONOTONIC, &depth_start);

        uint64_t launched = 0;
        while (launched < total_candidates) {
            uint64_t remaining = total_candidates - launched;
            uint64_t launch_candidates =
                remaining < max_candidates_per_launch ? remaining : max_candidates_per_launch;
            int blocks = (int)((launch_candidates + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

            evaluate_kernel<<<blocks, THREADS_PER_BLOCK>>>(
                launched,
                launch_candidates,
                depth,
                spec.output_ports[0],
                d_solutions,
                d_solution_lens,
                d_num_solutions,
                max_solutions,
                d_oep_tables[0],
                d_oep_tables[1],
                d_oep_tables[2]);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            launched += launch_candidates;
        }

        clock_gettime(CLOCK_MONOTONIC, &depth_end);
        double depth_secs = elapsed_seconds(&depth_start, &depth_end);

        CUDA_CHECK(cudaMemcpy(&total_hits, d_num_solutions, sizeof(int), cudaMemcpyDeviceToHost));
        stored = total_hits < max_solutions ? total_hits : max_solutions;

        printf("  elapsed: %.3f sec\n", depth_secs);
        if (depth_secs > 0.0) {
            printf("  throughput: %.3f M candidates/sec\n",
                   (double)total_candidates / depth_secs / 1e6);
        }
        printf("  solutions reported: %d", total_hits);
        if (total_hits > stored) {
            printf(" (stored first %d)", stored);
        }
        printf("\n");

        if (total_hits > 0) {
            found_depth = depth;
            solutions =
                (host_solution_t *)calloc((size_t)(stored > 0 ? stored : 1), sizeof(host_solution_t));
            if (!solutions) {
                fprintf(stderr, "out of memory while collecting solutions\n");
                CUDA_CHECK(cudaFree(d_solutions));
                CUDA_CHECK(cudaFree(d_solution_lens));
                CUDA_CHECK(cudaFree(d_num_solutions));
                return 1;
            }

            if (stored > 0) {
                uint16_t *flat_words =
                    (uint16_t *)malloc((size_t)stored * MAX_PROG_LEN * sizeof(uint16_t));
                int *lengths = (int *)malloc((size_t)stored * sizeof(int));
                if (!flat_words || !lengths) {
                    fprintf(stderr, "out of memory while downloading solutions\n");
                    free(flat_words);
                    free(lengths);
                    free(solutions);
                    CUDA_CHECK(cudaFree(d_solutions));
                    CUDA_CHECK(cudaFree(d_solution_lens));
                    CUDA_CHECK(cudaFree(d_num_solutions));
                    return 1;
                }

                CUDA_CHECK(cudaMemcpy(flat_words, d_solutions,
                                      (size_t)stored * MAX_PROG_LEN * sizeof(uint16_t),
                                      cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(lengths, d_solution_lens,
                                      (size_t)stored * sizeof(int),
                                      cudaMemcpyDeviceToHost));

                for (int i = 0; i < stored; i++) {
                    solutions[i].len = lengths[i];
                    memcpy(solutions[i].words,
                           &flat_words[i * MAX_PROG_LEN],
                           (size_t)lengths[i] * sizeof(uint16_t));
                    solutions[i].passed_tests =
                        task_evaluate(&spec, solutions[i].words, solutions[i].len);
                }

                free(flat_words);
                free(lengths);
            }

            break;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = elapsed_seconds(&t0, &t1);

    if (found_depth >= 0) {
        printf("\nOptimal solution length: %d instructions (kernel depth %d)\n",
               spec.num_inputs + found_depth + 2, found_depth);
        for (int i = 0; i < stored; i++) {
            printf("\nSolution #%d  len=%d  cpu_verify=%d/%d %s\n",
                   i + 1,
                   solutions[i].len,
                   solutions[i].passed_tests,
                   spec.num_tests,
                   solutions[i].passed_tests == spec.num_tests ? "OK" : "FAIL");
            print_program_words(solutions[i].words, solutions[i].len);
        }
    } else {
        printf("\nNo solutions found up to kernel depth %d.\n", max_depth);
    }

    printf("\nTotal: %" PRIu64 " candidates in %.3f sec\n",
           total_candidates_all_depths, secs);

    free(solutions);
    CUDA_CHECK(cudaFree(d_solutions));
    CUDA_CHECK(cudaFree(d_solution_lens));
    CUDA_CHECK(cudaFree(d_num_solutions));
    for (int p = 0; p < OEP_MAX_POS; p++)
        CUDA_CHECK(cudaFree(d_oep_tables[p]));

    return (found_depth >= 0) ? 0 : 1;
}
