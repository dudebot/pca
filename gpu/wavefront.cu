/*
 * wavefront.cu — CUDA wavefront-expansion solver for PCA-16
 *
 * Instead of launching one thread per possible program (flat-index),
 * this solver expands the search tree level by level:
 *
 *   expand_kernel(depth 0): scaffold → ~200 unique prefixes
 *   expand_kernel(depth 1): 200 × candidates → ~50K unique prefixes
 *   expand_kernel(depth 2): 50K × candidates → ~2M unique prefixes
 *   leaf_kernel: 2M × 7 output regs → check solutions directly
 *
 * OEP (observational equivalence pruning) happens at each level via
 * atomicCAS hash table. Pruned prefixes never generate children.
 *
 * Build:
 *   nvcc -O3 -arch=native -o gpu_wavefront gpu/wavefront.cu tasks/spec.c src/vm.c
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

/* Wavefront config */
#define WF_MAX_TESTS    16              /* enough for all current tasks */
#define WF_MAX_FRONTIER (6 * 1024 * 1024) /* 6M entries max per level */

/* OEP config */
#define OEP_TABLE_BITS  23
#define OEP_TABLE_SIZE  (1ULL << OEP_TABLE_BITS)
#define OEP_TABLE_MASK  (OEP_TABLE_SIZE - 1)
#define OEP_NUM_PROBES  8

/* --- constant memory --- */

__constant__ uint16_t d_candidates[MAX_CANDIDATES_PER_POS];
__constant__ int d_num_candidates;

__constant__ uint16_t d_test_in[MAX_TESTS][MAX_IO_PORTS];
__constant__ uint16_t d_test_out[MAX_TESTS][MAX_IO_PORTS];
__constant__ int d_num_tests;
__constant__ int d_num_inputs;
__constant__ int d_num_outputs;
__constant__ uint8_t d_input_ports[MAX_IO_PORTS];
__constant__ uint8_t d_output_ports[MAX_IO_PORTS];

__constant__ uint16_t d_scaffold[PCA_NUM_REGS];
__constant__ int d_scaffold_len;
__constant__ uint8_t d_initial_live;
__constant__ int d_max_steps;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "%s:%d: CUDA error: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err__));          \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

/* --- wavefront entry --- */

typedef struct {
    uint16_t regs[WF_MAX_TESTS][PCA_NUM_REGS]; /* per-test register state */
    uint8_t  flags[WF_MAX_TESTS];               /* per-test flags */
    uint8_t  live;                              /* live register mask */
    uint8_t  num_insns;                         /* instructions so far */
    uint16_t insns[MAX_KERNEL_DEPTH];           /* for solution reconstruction */
} wf_entry_t;

/* --- device helpers --- */

__device__ __forceinline__
int reg_is_live(uint8_t live, uint8_t reg)
{
    return reg == 0 || (live & (uint8_t)(1u << reg));
}

/* Execute one instruction on a register state. Does NOT modify flags for
 * LDI (matching the real VM behavior — Codex caught this bug). */
__device__ __forceinline__
void wf_exec_one(uint16_t insn, uint16_t *r, uint8_t *flags)
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
        if (r[rt] == 0) { *flags = 0xFF; return; }
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
        /* LDI does NOT modify flags in the real VM */
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

/* FNV-1a 64-bit hash of all test register states + flags */
__device__ __forceinline__
uint64_t wf_hash_states(uint16_t regs[][PCA_NUM_REGS],
                        uint8_t *flags, int num_tests)
{
    uint64_t h = 14695981039346656037ULL;
    for (int t = 0; t < num_tests; t++) {
        for (int r = 0; r < PCA_NUM_REGS; r++) {
            h ^= regs[t][r] & 0xFF;  h *= 1099511628211ULL;
            h ^= regs[t][r] >> 8;    h *= 1099511628211ULL;
        }
        h ^= flags[t]; h *= 1099511628211ULL;
    }
    return h;
}

/* Atomic insert-if-absent with linear probing */
__device__ __forceinline__
int wf_oep_check(unsigned long long *table, uint64_t hash)
{
    unsigned long long key = (unsigned long long)(hash | 1ULL);
    unsigned long long slot = ((unsigned long long)(hash >> 1)) & OEP_TABLE_MASK;

    for (int probe = 0; probe < OEP_NUM_PROBES; probe++) {
        unsigned long long idx = (slot + probe) & OEP_TABLE_MASK;
        unsigned long long old = atomicCAS(&table[idx], 0ULL, key);
        if (old == 0ULL) return 0; /* new state */
        if (old == key) return 1;  /* duplicate, prune */
    }
    return 0; /* table full at slot, conservative: don't prune */
}

/* --- VM for suffix kernel (search-relevant subset of ISA) --- */

__device__ __forceinline__
int vm_run(const uint16_t *prog, int prog_len,
           const uint16_t *port_in, uint16_t *port_out, int max_steps)
{
    uint16_t r[PCA_NUM_REGS] = {0};
    uint16_t ports[PCA_NUM_PORTS] = {0};
    uint8_t flags = 0;
    uint8_t pc = 0;

    for (int i = 0; i < PCA_NUM_PORTS; i++) ports[i] = port_in[i];

    for (int step = 0; step < max_steps; step++) {
        if (pc >= prog_len) return PCA_FAULT;
        uint16_t insn = prog[pc];
        uint8_t op = INSN_OP(insn), rd = INSN_RD(insn);
        uint8_t rs = INSN_RS(insn), rt = INSN_RT(insn);
        uint8_t imm = INSN_IMM8(insn);
        int8_t simm = INSN_SIMM8(insn);
        pc++;
        switch (op) {
        case OP_NOP: break;
        case OP_HLT:
            for (int i = 0; i < PCA_NUM_PORTS; i++) port_out[i] = ports[i];
            return PCA_HALTED;
        case OP_ADD: {
            uint32_t f = (uint32_t)r[rs]+(uint32_t)r[rt]; uint16_t res=(uint16_t)f;
            flags=0; if(!res)flags|=FLAG_Z; if(res&0x8000)flags|=FLAG_N;
            if(f>0xFFFF)flags|=FLAG_C;
            if(~(r[rs]^r[rt])&(r[rs]^res)&0x8000)flags|=FLAG_V;
            r[rd]=res; break;}
        case OP_SUB: {
            uint16_t a=r[rs],b=r[rt],res=a-b; flags=0;
            if(!res)flags|=FLAG_Z; if(res&0x8000)flags|=FLAG_N;
            if(a>=b)flags|=FLAG_C; if((a^b)&(a^res)&0x8000)flags|=FLAG_V;
            r[rd]=res; break;}
        case OP_MUL: {
            uint32_t p=(uint32_t)r[rs]*(uint32_t)r[rt]; r[rd]=(uint16_t)p;
            flags=0; if(!r[rd])flags|=FLAG_Z; if(r[rd]&0x8000)flags|=FLAG_N;
            if(p>0xFFFF)flags|=FLAG_C; break;}
        case OP_DIV:
            if(!r[rt])return PCA_FAULT;
            r[rd]=(uint16_t)((int16_t)r[rs]/(int16_t)r[rt]);
            flags=0; if(!r[rd])flags|=FLAG_Z; if(r[rd]&0x8000)flags|=FLAG_N; break;
        case OP_MOD:
            if(!r[rt])return PCA_FAULT;
            r[rd]=(uint16_t)((int16_t)r[rs]%(int16_t)r[rt]);
            flags=0; if(!r[rd])flags|=FLAG_Z; if(r[rd]&0x8000)flags|=FLAG_N; break;
        case OP_AND: r[rd]=r[rs]&r[rt]; flags=(r[rd]==0?FLAG_Z:0)|(r[rd]&0x8000?FLAG_N:0); break;
        case OP_OR:  r[rd]=r[rs]|r[rt]; flags=(r[rd]==0?FLAG_Z:0)|(r[rd]&0x8000?FLAG_N:0); break;
        case OP_XOR: r[rd]=r[rs]^r[rt]; flags=(r[rd]==0?FLAG_Z:0)|(r[rd]&0x8000?FLAG_N:0); break;
        case OP_SHL: r[rd]=r[rs]<<(r[rt]&0xF); flags=(r[rd]==0?FLAG_Z:0)|(r[rd]&0x8000?FLAG_N:0); break;
        case OP_SHR: r[rd]=r[rs]>>(r[rt]&0xF); flags=(r[rd]==0?FLAG_Z:0)|(r[rd]&0x8000?FLAG_N:0); break;
        case OP_ASR: r[rd]=(uint16_t)((int16_t)r[rs]>>(r[rt]&0xF)); flags=(r[rd]==0?FLAG_Z:0)|(r[rd]&0x8000?FLAG_N:0); break;
        case OP_CMP: {
            uint16_t a=r[rd],b=r[rs],res=a-b; flags=0;
            if(!res)flags|=FLAG_Z; if(res&0x8000)flags|=FLAG_N;
            if(a>=b)flags|=FLAG_C; if((a^b)&(a^res)&0x8000)flags|=FLAG_V; break;}
        case OP_LDI: r[rd]=imm; break;
        case OP_LUI: r[rd]=(r[rd]&0x00FF)|((uint16_t)imm<<8); break;
        case OP_ADDI: {
            uint16_t ext=(uint16_t)(int16_t)simm;
            uint32_t f=(uint32_t)r[rd]+(uint32_t)ext; uint16_t res=(uint16_t)f;
            flags=0; if(!res)flags|=FLAG_Z; if(res&0x8000)flags|=FLAG_N;
            if(f>0xFFFF)flags|=FLAG_C;
            if(~(r[rd]^ext)&(r[rd]^res)&0x8000)flags|=FLAG_V;
            r[rd]=res; break;}
        case OP_BR: {
            int take=0;
            switch(rd){
            case COND_AL:take=1;break; case COND_EQ:take=(flags&FLAG_Z)!=0;break;
            case COND_NE:take=(flags&FLAG_Z)==0;break;
            case COND_LT:take=((flags&FLAG_N)!=0)!=((flags&FLAG_V)!=0);break;
            case COND_GE:take=((flags&FLAG_N)!=0)==((flags&FLAG_V)!=0);break;
            case COND_LTU:take=(flags&FLAG_C)==0;break;
            case COND_GEU:take=(flags&FLAG_C)!=0;break;
            default:return PCA_FAULT;}
            if(take)pc=(uint8_t)((int)pc+simm); break;}
        case OP_IN: r[rd]=ports[imm&7]; break;
        case OP_OUT: ports[imm&7]=r[rd]; break;
        default: return PCA_FAULT;
        }
        r[0]=0;
    }
    return PCA_TIMEOUT;
}

/* --- expand kernel ---
 * Each thread takes one (frontier_entry, candidate) pair.
 * Applies liveness check, executes instruction on all test states,
 * OEP hashes, and if unique, appends to output frontier. */

__global__
void expand_kernel(
    const wf_entry_t *input,
    int num_input,
    wf_entry_t *output,
    int *output_count,
    int max_output,
    unsigned long long *oep_table,
    int nc)
{
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int n_entries = (num_input == 0) ? 1 : num_input;
    uint64_t total = (uint64_t)n_entries * (uint64_t)nc;
    if (gid >= total) return;

    int entry_idx = (int)(gid / (uint64_t)nc);
    int cand_idx  = (int)(gid % (uint64_t)nc);
    int num_tests = d_num_tests;

    /* Load or initialize entry */
    wf_entry_t entry;
    if (num_input == 0) {
        for (int t = 0; t < num_tests && t < WF_MAX_TESTS; t++) {
            for (int r = 0; r < PCA_NUM_REGS; r++) entry.regs[t][r] = 0;
            for (int i = 0; i < d_scaffold_len; i++)
                entry.regs[t][i + 1] = d_test_in[t][i];
            entry.flags[t] = 0;
        }
        entry.live = d_initial_live;
        entry.num_insns = 0;
    } else {
        entry = input[entry_idx];
    }

    /* Decode candidate */
    uint16_t insn = d_candidates[cand_idx];
    uint8_t op = INSN_OP(insn);
    uint8_t rd = INSN_RD(insn);
    uint8_t rs = INSN_RS(insn);
    uint8_t rt = INSN_RT(insn);

    /* Skip branches — wavefront is straight-line only */
    if (op == OP_BR) return;

    /* Liveness check */
    switch (op) {
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_MOD:
    case OP_AND: case OP_OR:  case OP_XOR: case OP_SHL: case OP_SHR:
    case OP_ASR:
        if (!reg_is_live(entry.live, rs) || !reg_is_live(entry.live, rt)) return;
        break;
    case OP_CMP:
        if (!reg_is_live(entry.live, rd) || !reg_is_live(entry.live, rs)) return;
        break;
    case OP_LDI:
        break; /* no source registers */
    case OP_ADDI:
        if (!reg_is_live(entry.live, rd)) return;
        break;
    default:
        return;
    }

    /* Execute instruction on all test states */
    for (int t = 0; t < num_tests && t < WF_MAX_TESTS; t++) {
        wf_exec_one(insn, entry.regs[t], &entry.flags[t]);
        if (entry.flags[t] == 0xFF) return; /* fault (e.g. div by zero) */
    }

    /* Update liveness */
    if (op != OP_CMP && rd != 0)
        entry.live |= (uint8_t)(1u << rd);

    /* OEP: hash combined state, check for duplicate */
    uint64_t h = wf_hash_states(entry.regs, entry.flags, num_tests);
    if (wf_oep_check(oep_table, h)) return;

    /* Survived! Append to output frontier */
    int slot = atomicAdd(output_count, 1);
    if (slot < max_output) {
        entry.insns[(int)entry.num_insns] = insn;
        entry.num_insns++;
        output[slot] = entry;
    }
}

/* --- leaf kernel ---
 * Each thread takes one (frontier_entry, output_register) pair.
 * Checks if the register value matches expected output on all tests.
 * No VM execution needed — the register state is already computed. */

__global__
void leaf_kernel(
    const wf_entry_t *frontier,
    int num_entries,
    uint16_t *d_solutions,
    int *d_solution_lens,
    int *d_num_solutions,
    int max_solutions)
{
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)num_entries * (PCA_NUM_REGS - 1);
    if (gid >= total) return;

    int entry_idx = (int)(gid / (PCA_NUM_REGS - 1));
    int out_reg   = (int)(gid % (PCA_NUM_REGS - 1)) + 1;
    int num_tests = d_num_tests;

    const wf_entry_t *e = &frontier[entry_idx];

    /* Check output register is live */
    if (!(e->live & (1u << out_reg))) return;

    /* Check all tests: does register value match expected output? */
    for (int t = 0; t < num_tests && t < WF_MAX_TESTS; t++) {
        if (e->regs[t][out_reg] != d_test_out[t][0])
            return;
    }

    /* All tests pass — record solution */
    if (*(volatile int *)d_num_solutions >= max_solutions) return;
    int slot = atomicAdd(d_num_solutions, 1);
    if (slot < max_solutions) {
        int plen = 0;
        int scaffold_len = d_scaffold_len;
        /* Scaffold */
        for (int i = 0; i < scaffold_len; i++)
            d_solutions[slot * MAX_PROG_LEN + plen++] = d_scaffold[i];
        /* Kernel instructions */
        for (int i = 0; i < e->num_insns; i++)
            d_solutions[slot * MAX_PROG_LEN + plen++] = e->insns[i];
        /* OUT + HLT */
        d_solutions[slot * MAX_PROG_LEN + plen++] =
            ENCODE_I(OP_OUT, out_reg, d_output_ports[0]);
        d_solutions[slot * MAX_PROG_LEN + plen++] =
            ENCODE_R(OP_HLT, 0, 0, 0);
        d_solution_lens[slot] = plen;
    }
}

/* --- candidate table generation (same as evaluate.cu) --- */

/* --- suffix kernel ---
 * Brute-force remaining depth from wavefront prefixes. Includes branches.
 * Each thread: load prefix from frontier, decode suffix from flat index,
 * build full program, run VM on all tests. */

__global__
void suffix_kernel(
    const wf_entry_t *frontier,
    int num_entries,
    int suffix_depth,
    uint64_t suffix_space,       /* nc^suffix_depth */
    uint8_t output_port,
    uint16_t *d_solutions,
    int *d_solution_lens,
    int *d_num_solutions,
    int max_solutions,
    uint64_t base_gid,
    uint64_t launch_count)
{
    uint64_t local_gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (local_gid >= launch_count) return;
    uint64_t gid = base_gid + local_gid;

    int nc = d_num_candidates;
    int num_tests = d_num_tests;
    int scaffold_len = d_scaffold_len;

    /* Decompose gid → (entry_idx, suffix_idx, out_reg) */
    int out_reg = (int)(gid % (PCA_NUM_REGS - 1)) + 1;
    uint64_t rem = gid / (PCA_NUM_REGS - 1);
    int entry_idx = (int)(rem / suffix_space);
    uint64_t suffix_idx = rem % suffix_space;

    const wf_entry_t *e = &frontier[entry_idx];

    /* Build full program: scaffold + prefix + suffix + OUT + HLT */
    uint16_t prog[MAX_PROG_LEN];
    int plen = 0;

    for (int i = 0; i < scaffold_len; i++)
        prog[plen++] = d_scaffold[i];
    for (int i = 0; i < e->num_insns; i++)
        prog[plen++] = e->insns[i];

    /* Decode suffix from flat index (with liveness + branch validation) */
    uint8_t live = e->live;
    int skip_live = 0;
    for (int k = 0; k < suffix_depth; k++) {
        int ci = (int)(suffix_idx % (uint64_t)nc);
        suffix_idx /= (uint64_t)nc;
        uint16_t insn = d_candidates[ci];
        uint8_t op = INSN_OP(insn);
        uint8_t rd = INSN_RD(insn);
        uint8_t rs = INSN_RS(insn);
        uint8_t rt = INSN_RT(insn);

        if (op == OP_BR) {
            int8_t off = INSN_SIMM8(insn);
            if (off < 1 || off > suffix_depth - k - 1) return;
            skip_live = 1;
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
        prog[plen++] = insn;
    }

    if (!skip_live && !reg_is_live(live, (uint8_t)out_reg)) return;
    prog[plen++] = ENCODE_I(OP_OUT, out_reg, output_port);
    prog[plen++] = ENCODE_R(OP_HLT, 0, 0, 0);

    /* Run full VM on all tests */
    int passed = 0;
    for (int t = 0; t < num_tests; t++) {
        uint16_t port_in[PCA_NUM_PORTS] = {0};
        uint16_t port_out[PCA_NUM_PORTS];
        for (int i = 0; i < d_num_inputs; i++)
            port_in[d_input_ports[i]] = d_test_in[t][i];
        int status = vm_run(prog, plen, port_in, port_out, d_max_steps);
        if (status != PCA_HALTED) break;
        if (port_out[d_output_ports[0]] != d_test_out[t][0]) break;
        passed++;
    }

    if (passed == num_tests) {
        if (*(volatile int *)d_num_solutions >= max_solutions) return;
        int slot = atomicAdd(d_num_solutions, 1);
        if (slot < max_solutions) {
            for (int i = 0; i < plen; i++)
                d_solutions[slot * MAX_PROG_LEN + i] = prog[i];
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
                    if ((op == OP_SHL || op == OP_SHR || op == OP_ASR) && rt == 0)
                        continue;
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
        for (int i = 0; i < (int)(sizeof(addis) / sizeof(addis[0])) && n < max;
             i++) {
            table[n++] = ENCODE_I(OP_ADDI, rd, (uint8_t)addis[i]);
        }
    }

    /* BR with forward offsets (conditional only, no COND_AL).
     * Wavefront expand_kernel skips these; suffix_kernel uses them. */
    for (int cond = 1; cond <= 6 && n < max; cond++) {
        for (int off = 1; off < MAX_KERNEL_DEPTH && n < max; off++) {
            table[n++] = ENCODE_I(OP_BR, cond, (uint8_t)(int8_t)off);
        }
    }

    return n;
}

static void upload_task_constants(const task_spec_t *spec)
{
    uint16_t h_test_in[MAX_TESTS][MAX_IO_PORTS] = {{0}};
    uint16_t h_test_out[MAX_TESTS][MAX_IO_PORTS] = {{0}};
    uint16_t h_scaffold[PCA_NUM_REGS] = {0};
    uint8_t initial_live = 0;

    for (int t = 0; t < spec->num_tests; t++) {
        for (int i = 0; i < spec->num_inputs; i++)
            h_test_in[t][i] = spec->tests[t].in[i];
        for (int i = 0; i < spec->num_outputs; i++)
            h_test_out[t][i] = spec->tests[t].out[i];
    }

    for (int i = 0; i < spec->num_inputs; i++) {
        h_scaffold[i] = ENCODE_I(OP_IN, i + 1, spec->input_ports[i]);
        initial_live |= (uint8_t)(1u << (i + 1));
    }

    CUDA_CHECK(cudaMemcpyToSymbol(d_test_in, h_test_in, sizeof(h_test_in)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_test_out, h_test_out, sizeof(h_test_out)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_tests, &spec->num_tests,
                                  sizeof(spec->num_tests)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_inputs, &spec->num_inputs,
                                  sizeof(spec->num_inputs)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_outputs, &spec->num_outputs,
                                  sizeof(spec->num_outputs)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_input_ports, spec->input_ports,
                                  sizeof(spec->input_ports)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_output_ports, spec->output_ports,
                                  sizeof(spec->output_ports)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_scaffold, h_scaffold, sizeof(h_scaffold)));

    int scaffold_len = spec->num_inputs;
    CUDA_CHECK(
        cudaMemcpyToSymbol(d_scaffold_len, &scaffold_len, sizeof(scaffold_len)));
    CUDA_CHECK(
        cudaMemcpyToSymbol(d_initial_live, &initial_live, sizeof(initial_live)));

    int max_steps = (int)spec->max_cycles;
    CUDA_CHECK(cudaMemcpyToSymbol(d_max_steps, &max_steps, sizeof(max_steps)));
}

static double elapsed_seconds(const struct timespec *start,
                              const struct timespec *end)
{
    return (double)(end->tv_sec - start->tv_sec) +
           (double)(end->tv_nsec - start->tv_nsec) * 1e-9;
}

/* --- main --- */

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
                "Usage: gpu_wavefront <task.json> <max_kernel_depth> "
                "[max_solutions] [wavefront_depth]\n");
        return 1;
    }

    const char *task_path = argv[1];
    int max_depth = atoi(argv[2]);
    int max_solutions = (argc >= 4) ? atoi(argv[3]) : 32;
    int user_wf_depth = (argc >= 5) ? atoi(argv[4]) : 1;
    if (max_depth < 0 || max_depth > MAX_KERNEL_DEPTH) {
        fprintf(stderr, "max_kernel_depth must be in [0, %d]\n",
                MAX_KERNEL_DEPTH);
        return 1;
    }

    task_spec_t spec;
    if (task_load(task_path, &spec) < 0)
        return 1;
    if (spec.num_tests > WF_MAX_TESTS) {
        fprintf(stderr, "task has %d tests, wavefront max is %d\n",
                spec.num_tests, WF_MAX_TESTS);
        return 1;
    }
    if (spec.num_outputs != 1) {
        fprintf(stderr, "multi-output tasks not supported\n");
        return 1;
    }
    if (spec.num_inputs > PCA_NUM_REGS - 1) {
        fprintf(stderr, "task requires %d inputs, max is %d\n",
                spec.num_inputs, PCA_NUM_REGS - 1);
        return 1;
    }
    if (spec.num_inputs + max_depth + 2 > MAX_PROG_LEN) {
        fprintf(stderr, "program length %d exceeds MAX_PROG_LEN=%d\n",
                spec.num_inputs + max_depth + 2, MAX_PROG_LEN);
        return 1;
    }

    uint16_t h_candidates[MAX_CANDIDATES_PER_POS];
    int nc = gen_candidate_table(h_candidates, MAX_CANDIDATES_PER_POS);

    CUDA_CHECK(cudaMemcpyToSymbol(d_candidates, h_candidates,
                                  nc * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_candidates, &nc, sizeof(nc)));
    upload_task_constants(&spec);

    /* Allocate ping-pong frontier buffers */
    wf_entry_t *d_front_a = NULL, *d_front_b = NULL;
    size_t front_bytes = (size_t)WF_MAX_FRONTIER * sizeof(wf_entry_t);
    CUDA_CHECK(cudaMalloc((void **)&d_front_a, front_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_front_b, front_bytes));

    int *d_front_count = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_front_count, sizeof(int)));

    /* OEP table */
    unsigned long long *d_oep = NULL;
    size_t oep_bytes = OEP_TABLE_SIZE * sizeof(unsigned long long);
    CUDA_CHECK(cudaMalloc((void **)&d_oep, oep_bytes));

    /* Solution storage */
    uint16_t *d_solutions = NULL;
    int *d_solution_lens = NULL;
    int *d_num_solutions = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_solutions,
                          (size_t)max_solutions * MAX_PROG_LEN * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_solution_lens,
                          (size_t)max_solutions * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_num_solutions, sizeof(int)));

    int device = 0;
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    printf("PCA-16 CUDA wavefront solver\n");
    printf("  task: %s (%s)\n", task_path, spec.name);
    printf("  gpu: %s\n", props.name);
    printf("  tests: %d  inputs: %d  outputs: %d\n",
           spec.num_tests, spec.num_inputs, spec.num_outputs);
    printf("  candidates/position: %d (incl. BR)  max depth: %d\n",
           nc, max_depth);
    printf("  frontier cap: %d entries  (%.0f MB × 2 buffers)\n",
           WF_MAX_FRONTIER,
           (double)front_bytes / (1024.0 * 1024.0));
    printf("  OEP table: %llu entries (%.0f MB)\n",
           (unsigned long long)OEP_TABLE_SIZE,
           (double)oep_bytes / (1024.0 * 1024.0));

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int found_depth = -1;

    for (int target_depth = 0; target_depth <= max_depth; target_depth++) {
        printf("\n=== target kernel depth %d (program length %d) ===\n",
               target_depth, spec.num_inputs + target_depth + 2);

        wf_entry_t *front_in  = NULL;
        wf_entry_t *front_out = d_front_a;
        int front_count = 0;

        /* Wavefront expand first N levels (straight-line only, OEP pruned).
         * Suffix kernel handles remaining depth with branches.
         * Higher wf_depth = more OEP pruning but misses solutions with
         * branches in the wavefront positions. */
        int wf_depth = target_depth < user_wf_depth ? target_depth : user_wf_depth;
        int suffix_depth = target_depth - wf_depth;

        for (int d = 0; d < wf_depth; d++) {
            CUDA_CHECK(cudaMemset(d_oep, 0, oep_bytes));
            CUDA_CHECK(cudaMemset(d_front_count, 0, sizeof(int)));

            int n_entries = (d == 0) ? 1 : front_count;
            uint64_t total_threads = (uint64_t)n_entries * (uint64_t)nc;
            int blocks = (int)((total_threads + THREADS_PER_BLOCK - 1) /
                               THREADS_PER_BLOCK);

            struct timespec ds, de;
            clock_gettime(CLOCK_MONOTONIC, &ds);

            expand_kernel<<<blocks, THREADS_PER_BLOCK>>>(
                front_in, (d == 0) ? 0 : front_count,
                front_out, d_front_count, WF_MAX_FRONTIER,
                d_oep, nc);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            clock_gettime(CLOCK_MONOTONIC, &de);

            int raw_count = 0;
            CUDA_CHECK(cudaMemcpy(&raw_count, d_front_count, sizeof(int),
                                  cudaMemcpyDeviceToHost));
            front_count = raw_count < WF_MAX_FRONTIER ? raw_count : WF_MAX_FRONTIER;

            printf("  expand depth %d: %d → %d unique prefixes "
                   "(%.3f sec, %" PRIu64 " threads)\n",
                   d, n_entries, front_count,
                   elapsed_seconds(&ds, &de), total_threads);

            if (raw_count > WF_MAX_FRONTIER) {
                fprintf(stderr, "  WARNING: frontier overflow at depth %d "
                        "(%d found, capped at %d)\n",
                        d, raw_count, WF_MAX_FRONTIER);
            }
            if (front_count == 0) {
                printf("  no surviving prefixes — skipping\n");
                break;
            }

            /* Ping-pong */
            front_in  = front_out;
            front_out = (front_out == d_front_a) ? d_front_b : d_front_a;
        }

        if (wf_depth > 0 && front_count == 0) continue;

        CUDA_CHECK(cudaMemset(d_num_solutions, 0, sizeof(int)));

        /* Create initial frontier entry for depth-0 targets */
        if (wf_depth == 0) {
            wf_entry_t scaffold_entry;
            memset(&scaffold_entry, 0, sizeof(scaffold_entry));
            for (int t = 0; t < spec.num_tests; t++) {
                for (int i = 0; i < spec.num_inputs; i++)
                    scaffold_entry.regs[t][i + 1] = spec.tests[t].in[i];
            }
            scaffold_entry.live = 0;
            for (int i = 0; i < spec.num_inputs; i++)
                scaffold_entry.live |= (uint8_t)(1u << (i + 1));
            scaffold_entry.num_insns = 0;

            CUDA_CHECK(cudaMemcpy(d_front_a, &scaffold_entry,
                                  sizeof(wf_entry_t),
                                  cudaMemcpyHostToDevice));
            front_in = d_front_a;
            front_count = 1;
        }

        struct timespec ls, le;
        clock_gettime(CLOCK_MONOTONIC, &ls);

        if (suffix_depth == 0) {
            /* Pure wavefront: just check output registers */
            uint64_t leaf_threads = (uint64_t)front_count * (PCA_NUM_REGS - 1);
            int leaf_blocks = (int)((leaf_threads + THREADS_PER_BLOCK - 1) /
                                    THREADS_PER_BLOCK);
            leaf_kernel<<<leaf_blocks, THREADS_PER_BLOCK>>>(
                front_in, front_count,
                d_solutions, d_solution_lens, d_num_solutions, max_solutions);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            clock_gettime(CLOCK_MONOTONIC, &le);
            printf("  leaf: %d prefixes × %d out_regs (%.3f sec)\n",
                   front_count, PCA_NUM_REGS - 1, elapsed_seconds(&ls, &le));
        } else {
            /* Hybrid: brute-force remaining depth with branches */
            uint64_t suf_space = 1;
            int overflow = 0;
            for (int i = 0; i < suffix_depth; i++) {
                if (suf_space > UINT64_MAX / (uint64_t)nc) { overflow = 1; break; }
                suf_space *= (uint64_t)nc;
            }
            if (!overflow && front_count > 0) {
                uint64_t tmp = (uint64_t)front_count * suf_space;
                if (tmp / suf_space != (uint64_t)front_count) overflow = 1;
                else {
                    uint64_t tmp2 = tmp * (PCA_NUM_REGS - 1);
                    if (tmp2 / (PCA_NUM_REGS - 1) != tmp) overflow = 1;
                }
            }
            if (overflow) {
                fprintf(stderr, "  suffix search space overflow at depth %d "
                        "— increase wavefront depth or use hybrid solver\n",
                        target_depth);
                continue;
            }
            uint64_t total_suf = (uint64_t)front_count * suf_space *
                                 (PCA_NUM_REGS - 1);

            /* Chunked launch */
            const uint64_t max_per_launch = (uint64_t)(1u << 20) * THREADS_PER_BLOCK;
            uint64_t launched = 0;
            while (launched < total_suf) {
                uint64_t remaining = total_suf - launched;
                uint64_t chunk = remaining < max_per_launch ? remaining : max_per_launch;
                int blocks = (int)((chunk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

                suffix_kernel<<<blocks, THREADS_PER_BLOCK>>>(
                    front_in, front_count, suffix_depth, suf_space,
                    spec.output_ports[0],
                    d_solutions, d_solution_lens, d_num_solutions,
                    max_solutions,
                    launched, chunk);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());

                launched += chunk;
            }

            clock_gettime(CLOCK_MONOTONIC, &le);
            printf("  suffix: %d prefixes × %" PRIu64 " suffixes × %d out_regs "
                   "= %" PRIu64 " threads (%.3f sec)\n",
                   front_count, suf_space, PCA_NUM_REGS - 1, total_suf,
                   elapsed_seconds(&ls, &le));
        }

        int total_hits = 0;
        CUDA_CHECK(cudaMemcpy(&total_hits, d_num_solutions, sizeof(int),
                              cudaMemcpyDeviceToHost));

        printf("  solutions: %d\n", total_hits);

        if (total_hits > 0) {
            found_depth = target_depth;
            int stored = total_hits < max_solutions ? total_hits : max_solutions;

            /* Download and verify solutions */
            uint16_t *flat =
                (uint16_t *)malloc((size_t)stored * MAX_PROG_LEN * sizeof(uint16_t));
            int *lens = (int *)malloc((size_t)stored * sizeof(int));
            CUDA_CHECK(cudaMemcpy(flat, d_solutions,
                                  (size_t)stored * MAX_PROG_LEN * sizeof(uint16_t),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(lens, d_solution_lens,
                                  (size_t)stored * sizeof(int),
                                  cudaMemcpyDeviceToHost));

            printf("\nOptimal solution length: %d instructions (kernel depth %d)\n",
                   spec.num_inputs + found_depth + 2, found_depth);

            for (int i = 0; i < stored; i++) {
                int plen = lens[i];
                int passed = task_evaluate(&spec, &flat[i * MAX_PROG_LEN], plen);
                printf("\nSolution #%d  len=%d  cpu_verify=%d/%d %s\n",
                       i + 1, plen, passed, spec.num_tests,
                       passed == spec.num_tests ? "OK" : "FAIL");
                for (int j = 0; j < plen; j++)
                    printf("    %02d: 0x%04X\n", j, flat[i * MAX_PROG_LEN + j]);
            }

            free(flat);
            free(lens);
            break;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double total_secs = elapsed_seconds(&t0, &t1);

    if (found_depth < 0)
        printf("\nNo solutions found up to kernel depth %d.\n", max_depth);

    printf("\nTotal elapsed: %.3f sec\n", total_secs);

    CUDA_CHECK(cudaFree(d_front_a));
    CUDA_CHECK(cudaFree(d_front_b));
    CUDA_CHECK(cudaFree(d_front_count));
    CUDA_CHECK(cudaFree(d_oep));
    CUDA_CHECK(cudaFree(d_solutions));
    CUDA_CHECK(cudaFree(d_solution_lens));
    CUDA_CHECK(cudaFree(d_num_solutions));

    return (found_depth >= 0) ? 0 : 1;
}
