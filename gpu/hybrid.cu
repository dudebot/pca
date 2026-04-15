/*
 * hybrid.cu — Hybrid CPU-GPU solver for PCA-16
 *
 * CPU does DFS with OEP through split_depth (handles branches correctly).
 * At split_depth, emits prefix to GPU batch buffer.
 * GPU brute-forces remaining suffix depth with full candidate table.
 *
 * Build:
 *   nvcc -O3 -arch=native -o gpu_hybrid gpu/hybrid.cu tasks/spec.c src/vm.c src/asm.c
 *
 * Usage:
 *   gpu_hybrid <task.json> <max_kernel_depth> [max_solutions] [split_depth] [batch_size]
 */

#include <cuda_runtime.h>

#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../src/pca.h"
#include "../tasks/spec.h"

#define MAX_CANDIDATES_PER_POS 4096
#define MAX_KERNEL_DEPTH 8
#define MAX_PROG_LEN 16
#define MAX_GPU_SOLUTIONS 1024
#define THREADS_PER_BLOCK 256

/* Default batch size — tunable via CLI */
#define DEFAULT_BATCH_SIZE 4096

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
__constant__ int d_max_steps;

__constant__ uint16_t d_scaffold[PCA_NUM_REGS];
__constant__ int d_scaffold_len;

/* --- prefix record (CPU → GPU) --- */

typedef struct {
    uint16_t insns[MAX_KERNEL_DEPTH];
    uint8_t  num_insns;
    uint8_t  live;
} prefix_t;

/* --- device helpers --- */

__device__ __forceinline__
int reg_is_live(uint8_t live, uint8_t reg)
{
    return reg == 0 || (live & (uint8_t)(1u << reg));
}

/* --- VM (search-relevant ISA subset) --- */

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
            uint32_t f=(uint32_t)r[rs]+(uint32_t)r[rt]; uint16_t res=(uint16_t)f;
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

/* --- suffix kernel ---
 * Each thread: load prefix from batch, decode suffix from flat index,
 * build full program (scaffold + prefix + suffix + OUT + HLT),
 * run VM on all tests. */

__global__
void suffix_kernel(
    const prefix_t *batch,
    int batch_count,
    int suffix_depth,
    uint64_t suffix_space,
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

    int out_reg = (int)(gid % (PCA_NUM_REGS - 1)) + 1;
    uint64_t rem = gid / (PCA_NUM_REGS - 1);
    int prefix_idx = (int)(rem / suffix_space);
    uint64_t suffix_idx = rem % suffix_space;

    if (prefix_idx >= batch_count) return;
    const prefix_t *p = &batch[prefix_idx];

    /* Build program: scaffold + prefix + suffix + OUT + HLT */
    uint16_t prog[MAX_PROG_LEN];
    int plen = 0;

    for (int i = 0; i < scaffold_len; i++)
        prog[plen++] = d_scaffold[i];
    for (int i = 0; i < p->num_insns; i++)
        prog[plen++] = p->insns[i];

    /* Decode suffix with liveness + branch validation */
    uint8_t live = p->live;
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
            default: return;
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

    /* Run VM on all tests */
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

/* --- GPU candidate table (includes BR) --- */

static int gen_gpu_candidates(uint16_t *table, int max)
{
    int n = 0;
    static const uint8_t alu_ops[] = {
        OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD,
        OP_AND, OP_OR,  OP_XOR, OP_SHL, OP_SHR, OP_ASR
    };
    static const int commut[] = {1,0,1,0,0,1,1,1,0,0,0};

    for (int oi = 0; oi < 11; oi++) {
        uint8_t op = alu_ops[oi];
        for (int rd = 1; rd < PCA_NUM_REGS && n < max; rd++)
            for (int rs = 0; rs < PCA_NUM_REGS && n < max; rs++) {
                int rt0 = commut[oi] ? rs : 0;
                for (int rt = rt0; rt < PCA_NUM_REGS && n < max; rt++) {
                    if (rs == 0 && rt == 0) continue;
                    if (op==OP_ADD && rs==rd && rt==0) continue;
                    if (op==OP_ADD && rt==rd && rs==0) continue;
                    if (op==OP_SUB && rs==rd && rt==0) continue;
                    if (op==OP_MUL && (rs==0||rt==0)) continue;
                    if ((op==OP_DIV||op==OP_MOD) && rt==0) continue;
                    if ((op==OP_SHL||op==OP_SHR||op==OP_ASR) && rt==0) continue;
                    table[n++] = ENCODE_R(op, rd, rs, rt);
                }
            }
    }
    for (int rd = 0; rd < PCA_NUM_REGS && n < max; rd++)
        for (int rs = 0; rs < PCA_NUM_REGS && n < max; rs++) {
            if (rd==0 && rs==0) continue;
            if (rd==rs) continue;
            table[n++] = ENCODE_R(OP_CMP, rd, rs, 0);
        }
    static const uint8_t imms[] = {0,1,2,3,4,5,8,16,32,64,128,255};
    for (int rd = 1; rd < PCA_NUM_REGS && n < max; rd++)
        for (int i = 0; i < (int)(sizeof(imms)/sizeof(imms[0])) && n < max; i++)
            table[n++] = ENCODE_I(OP_LDI, rd, imms[i]);
    static const int8_t addis[] = {-1,1,-2,2,-3,3};
    for (int rd = 1; rd < PCA_NUM_REGS && n < max; rd++)
        for (int i = 0; i < (int)(sizeof(addis)/sizeof(addis[0])) && n < max; i++)
            table[n++] = ENCODE_I(OP_ADDI, rd, (uint8_t)addis[i]);
    for (int cond = 1; cond <= 6 && n < max; cond++)
        for (int off = 1; off < MAX_KERNEL_DEPTH && n < max; off++)
            table[n++] = ENCODE_I(OP_BR, cond, (uint8_t)(int8_t)off);

    return n;
}

/* ================================================================
 * CPU-side: modified enumerator that emits prefixes at split_depth
 * ================================================================ */

/* OEP (copied from enumerate.c — same algorithm) */
#define OEP_TABLE_BITS 20
#define OEP_TABLE_SIZE_CPU (1 << OEP_TABLE_BITS)
#define OEP_TABLE_MASK_CPU (OEP_TABLE_SIZE_CPU - 1)

typedef struct {
    uint32_t *tables[16];
    uint64_t  pruned;
    uint64_t  saturated;  /* times insertion failed (table full at slot) */
    int       enabled;
    int       table_bits;
} oep_state_t;

static int oep_check_and_insert(oep_state_t *oep, int depth, uint32_t hash)
{
    if (!oep->enabled) return 0;
    int tsize = 1 << oep->table_bits;
    int tmask = tsize - 1;
    if (!oep->tables[depth]) {
        oep->tables[depth] = (uint32_t *)calloc(tsize, sizeof(uint32_t));
        if (!oep->tables[depth]) return 0;
    }
    uint32_t key = hash | 1;
    uint32_t idx = hash & tmask;
    for (int probe = 0; probe < 32; probe++) {
        uint32_t slot = oep->tables[depth][(idx + probe) & tmask];
        if (slot == 0) {
            oep->tables[depth][(idx + probe) & tmask] = key;
            return 0;
        }
        if (slot == key) { oep->pruned++; return 1; }
    }
    oep->saturated++;
    return 0;
}

static void oep_clear_depth(oep_state_t *oep, int depth)
{
    if (oep->tables[depth])
        memset(oep->tables[depth], 0, (size_t)(1 << oep->table_bits) * sizeof(uint32_t));
}

static void oep_free(oep_state_t *oep)
{
    for (int i = 0; i < 16; i++) free(oep->tables[i]);
}

/* CPU candidate generation (with liveness, same as enumerate.c) */
static const uint8_t ldi_vals[] = {0,1,2,3,4,5,8,16,32,64,128,255};
#define N_LDI ((int)(sizeof(ldi_vals)/sizeof(ldi_vals[0])))
static const int8_t addi_vals[] = {-1,1,-2,2,-3,3};
#define N_ADDI ((int)(sizeof(addi_vals)/sizeof(addi_vals[0])))

static int cpu_gen_candidates(uint16_t *out, int max, uint8_t live,
                              int pos, int kernel_depth, int use_branches)
{
    int n = 0;
    static const uint8_t alu_ops[] = {
        OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD,
        OP_AND, OP_OR,  OP_XOR, OP_SHL, OP_SHR, OP_ASR
    };
    static const int commut[] = {1,0,1,0,0,1,1,1,0,0,0};
    int n_alu = 11;

    for (int oi = 0; oi < n_alu; oi++) {
        uint8_t op = alu_ops[oi];
        for (int rd = 1; rd < 8 && n < max; rd++)
            for (int rs = 0; rs < 8 && n < max; rs++) {
                if (rs && !(live & (1 << rs))) continue;
                int rt0 = commut[oi] ? rs : 0;
                for (int rt = rt0; rt < 8 && n < max; rt++) {
                    if (rt && !(live & (1 << rt))) continue;
                    if (rs==0 && rt==0) continue;
                    if (op==OP_ADD && rs==(int)rd && rt==0) continue;
                    if (op==OP_ADD && rt==(int)rd && rs==0) continue;
                    if (op==OP_SUB && rs==(int)rd && rt==0) continue;
                    if (op==OP_MUL && (rs==0||rt==0)) continue;
                    if ((op==OP_DIV||op==OP_MOD) && rt==0) continue;
                    if ((op==OP_SHL||op==OP_SHR||op==OP_ASR) && rt==0) continue;
                    out[n++] = ENCODE_R(op, rd, rs, rt);
                }
            }
    }
    for (int rd = 0; rd < 8 && n < max; rd++) {
        if (rd && !(live & (1 << rd))) continue;
        for (int rs = 0; rs < 8 && n < max; rs++) {
            if (rs && !(live & (1 << rs))) continue;
            if (rd==0 && rs==0) continue;
            if (rd==rs) continue;
            out[n++] = ENCODE_R(OP_CMP, rd, rs, 0);
        }
    }
    if (use_branches) {
        int max_fwd = kernel_depth - pos - 1;
        if (max_fwd > 0)
            for (int cond = 1; cond <= 6; cond++)
                for (int off = 1; off <= max_fwd && n < max; off++)
                    out[n++] = ENCODE_I(OP_BR, cond, (uint8_t)(int8_t)off);
    }
    for (int rd = 1; rd < 8 && n < max; rd++)
        for (int i = 0; i < N_LDI && n < max; i++)
            out[n++] = ENCODE_I(OP_LDI, rd, ldi_vals[i]);
    for (int rd = 1; rd < 8 && n < max; rd++) {
        if (!(live & (1 << rd))) continue;
        for (int i = 0; i < N_ADDI && n < max; i++)
            out[n++] = ENCODE_I(OP_ADDI, rd, (uint8_t)addi_vals[i]);
    }
    return n;
}

/* --- hybrid solver state --- */

typedef struct {
    task_spec_t task;
    uint16_t program[PCA_PROG_SIZE];
    int num_inputs;
    int use_branches;
    oep_state_t oep;
    uint64_t cpu_tried;

    /* GPU interface */
    int split_depth;
    int target_depth;
    prefix_t *h_batch;        /* host batch buffer */
    prefix_t *d_batch;        /* device batch buffer */
    int batch_count;
    int batch_capacity;
    uint64_t prefixes_emitted;

    /* GPU resources */
    int gpu_nc;
    uint64_t suffix_space;
    uint8_t output_port;
    uint16_t *d_solutions;
    int *d_solution_lens;
    int *d_num_solutions;
    int max_solutions;
    int solutions_found;
} hybrid_t;

static void hybrid_flush_batch(hybrid_t *h)
{
    if (h->batch_count == 0) return;

    /* Upload batch to GPU */
    cudaMemcpy(h->d_batch, h->h_batch,
               (size_t)h->batch_count * sizeof(prefix_t),
               cudaMemcpyHostToDevice);

    int suffix_depth = h->target_depth - h->split_depth;
    uint64_t total = (uint64_t)h->batch_count * h->suffix_space * (PCA_NUM_REGS - 1);

    const uint64_t max_per_launch = (uint64_t)(1u << 20) * THREADS_PER_BLOCK;
    uint64_t launched = 0;
    while (launched < total) {
        uint64_t rem = total - launched;
        uint64_t chunk = rem < max_per_launch ? rem : max_per_launch;
        int blocks = (int)((chunk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

        suffix_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            h->d_batch, h->batch_count, suffix_depth, h->suffix_space,
            h->output_port,
            h->d_solutions, h->d_solution_lens, h->d_num_solutions,
            h->max_solutions, launched, chunk);
        cudaDeviceSynchronize();

        launched += chunk;
    }

    /* Check if any solutions found */
    int hits = 0;
    cudaMemcpy(&hits, h->d_num_solutions, sizeof(int), cudaMemcpyDeviceToHost);
    h->solutions_found = hits;

    h->batch_count = 0;
}

static void hybrid_emit_prefix(hybrid_t *h, uint8_t live)
{
    prefix_t *p = &h->h_batch[h->batch_count];
    int kernel_start = h->num_inputs;
    p->num_insns = (uint8_t)h->split_depth;
    p->live = live;
    for (int i = 0; i < h->split_depth; i++)
        p->insns[i] = h->program[kernel_start + i];

    h->batch_count++;
    h->prefixes_emitted++;

    if (h->batch_count >= h->batch_capacity)
        hybrid_flush_batch(h);
}

/* CPU DFS — identical to enumerate.c but emits at split_depth */
static void hybrid_enumerate(hybrid_t *h, int pos, int kernel_depth,
                             uint8_t live)
{
    if (h->solutions_found > 0) return;

    int kernel_start = h->num_inputs;

    /* At split_depth: emit prefix to GPU instead of recursing */
    if (pos == h->split_depth) {
        hybrid_emit_prefix(h, live);
        return;
    }

    /* At full depth (CPU-only fallback when split >= target): try output */
    if (pos == kernel_depth) {
        /* not used in hybrid mode */
        return;
    }

    uint16_t cands[32768];
    int nc = cpu_gen_candidates(cands, 32768, live, pos, kernel_depth,
                                h->use_branches);

    int prog_pos = kernel_start + pos;
    for (int i = 0; i < nc; i++) {
        if (h->solutions_found > 0) return;

        h->program[prog_pos] = cands[i];
        h->cpu_tried++;

        uint8_t new_live = live;
        uint8_t op = INSN_OP(cands[i]);
        uint8_t rd = INSN_RD(cands[i]);
        if (op != OP_CMP && op != OP_BR && op != OP_NOP && op != OP_HLT &&
            op != OP_ST && rd != 0) {
            new_live |= (1 << rd);
        }

        /* OEP */
        if (h->oep.enabled && op != OP_BR) {
            int partial_len = prog_pos + 1;
            h->program[partial_len] = ENCODE_R(OP_HLT, 0, 0, 0);

            uint32_t hash = 2166136261u;
            for (int t = 0; t < h->task.num_tests; t++) {
                pca_vm_t vm;
                pca_init(&vm);
                pca_load(&vm, h->program, partial_len + 1);
                for (int p2 = 0; p2 < h->task.num_inputs; p2++)
                    vm.ports[h->task.input_ports[p2]] = h->task.tests[t].in[p2];
                pca_run(&vm, 100);
                for (int r = 0; r < PCA_NUM_REGS; r++) {
                    hash ^= vm.r[r] & 0xFF; hash *= 16777619u;
                    hash ^= vm.r[r] >> 8;   hash *= 16777619u;
                }
                hash ^= vm.flags; hash *= 16777619u;
            }

            if (oep_check_and_insert(&h->oep, pos, hash))
                continue;
        }

        hybrid_enumerate(h, pos + 1, kernel_depth, new_live);
    }
}

/* --- upload constants --- */

static void upload_task_constants(const task_spec_t *spec)
{
    uint16_t h_test_in[MAX_TESTS][MAX_IO_PORTS] = {{0}};
    uint16_t h_test_out[MAX_TESTS][MAX_IO_PORTS] = {{0}};
    uint16_t h_scaffold[PCA_NUM_REGS] = {0};

    for (int t = 0; t < spec->num_tests; t++) {
        for (int i = 0; i < spec->num_inputs; i++)
            h_test_in[t][i] = spec->tests[t].in[i];
        for (int i = 0; i < spec->num_outputs; i++)
            h_test_out[t][i] = spec->tests[t].out[i];
    }
    for (int i = 0; i < spec->num_inputs; i++)
        h_scaffold[i] = ENCODE_I(OP_IN, i + 1, spec->input_ports[i]);

    cudaMemcpyToSymbol(d_test_in, h_test_in, sizeof(h_test_in));
    cudaMemcpyToSymbol(d_test_out, h_test_out, sizeof(h_test_out));
    cudaMemcpyToSymbol(d_num_tests, &spec->num_tests, sizeof(spec->num_tests));
    cudaMemcpyToSymbol(d_num_inputs, &spec->num_inputs, sizeof(spec->num_inputs));
    cudaMemcpyToSymbol(d_num_outputs, &spec->num_outputs, sizeof(spec->num_outputs));
    cudaMemcpyToSymbol(d_input_ports, spec->input_ports, sizeof(spec->input_ports));
    cudaMemcpyToSymbol(d_output_ports, spec->output_ports, sizeof(spec->output_ports));
    cudaMemcpyToSymbol(d_scaffold, h_scaffold, sizeof(h_scaffold));

    int scaffold_len = spec->num_inputs;
    cudaMemcpyToSymbol(d_scaffold_len, &scaffold_len, sizeof(scaffold_len));

    int max_steps = (int)spec->max_cycles;
    cudaMemcpyToSymbol(d_max_steps, &max_steps, sizeof(max_steps));
}

static double elapsed_sec(const struct timespec *a, const struct timespec *b)
{
    return (double)(b->tv_sec - a->tv_sec) +
           (double)(b->tv_nsec - a->tv_nsec) * 1e-9;
}

/* --- main --- */

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
                "Usage: gpu_hybrid <task.json> <max_kernel_depth> "
                "[max_solutions] [split_depth] [batch_size]\n");
        return 1;
    }

    const char *task_path = argv[1];
    int max_depth = atoi(argv[2]);
    int max_solutions = (argc >= 4) ? atoi(argv[3]) : 32;
    int user_split = (argc >= 5) ? atoi(argv[4]) : -1; /* auto */
    int batch_size = (argc >= 6) ? atoi(argv[5]) : DEFAULT_BATCH_SIZE;

    task_spec_t spec;
    if (task_load(task_path, &spec) < 0) return 1;
    if (spec.num_outputs != 1) {
        fprintf(stderr, "multi-output tasks not supported\n");
        return 1;
    }
    if (spec.num_inputs > PCA_NUM_REGS - 1) {
        fprintf(stderr, "too many inputs\n");
        return 1;
    }
    if (spec.num_inputs + max_depth + 2 > MAX_PROG_LEN) {
        fprintf(stderr, "program too long\n");
        return 1;
    }

    /* GPU setup */
    uint16_t h_candidates[MAX_CANDIDATES_PER_POS];
    int gpu_nc = gen_gpu_candidates(h_candidates, MAX_CANDIDATES_PER_POS);

    cudaMemcpyToSymbol(d_candidates, h_candidates, gpu_nc * sizeof(uint16_t));
    cudaMemcpyToSymbol(d_num_candidates, &gpu_nc, sizeof(gpu_nc));
    upload_task_constants(&spec);

    /* Solution storage */
    uint16_t *d_solutions;
    int *d_solution_lens, *d_num_solutions;
    cudaMalloc((void **)&d_solutions,
               (size_t)max_solutions * MAX_PROG_LEN * sizeof(uint16_t));
    cudaMalloc((void **)&d_solution_lens, (size_t)max_solutions * sizeof(int));
    cudaMalloc((void **)&d_num_solutions, sizeof(int));

    /* Batch buffers */
    prefix_t *h_batch = (prefix_t *)malloc((size_t)batch_size * sizeof(prefix_t));
    prefix_t *d_batch;
    cudaMalloc((void **)&d_batch, (size_t)batch_size * sizeof(prefix_t));

    int device = 0;
    cudaDeviceProp props;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);

    printf("PCA-16 hybrid CPU-GPU solver\n");
    printf("  task: %s (%s)\n", task_path, spec.name);
    printf("  gpu: %s\n", props.name);
    printf("  tests: %d  inputs: %d  outputs: %d\n",
           spec.num_tests, spec.num_inputs, spec.num_outputs);
    printf("  gpu candidates: %d  batch size: %d\n", gpu_nc, batch_size);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int found_depth = -1;

    for (int target_depth = 0; target_depth <= max_depth; target_depth++) {
        /* Auto split: leave 1 level for GPU unless user overrides */
        int split = (user_split >= 0) ? user_split : (target_depth > 0 ? target_depth - 1 : 0);
        if (split > target_depth) split = target_depth;
        int suffix_depth = target_depth - split;

        /* Compute suffix space with overflow check */
        uint64_t suffix_space = 1;
        int overflow = 0;
        for (int i = 0; i < suffix_depth; i++) {
            if (suffix_space > UINT64_MAX / (uint64_t)gpu_nc) { overflow = 1; break; }
            suffix_space *= (uint64_t)gpu_nc;
        }
        if (overflow) {
            fprintf(stderr, "  suffix overflow at depth %d — increase split\n",
                    target_depth);
            continue;
        }

        printf("\n=== depth %d (split: CPU %d + GPU %d) ===\n",
               target_depth, split, suffix_depth);

        /* Reset GPU solutions */
        cudaMemset(d_num_solutions, 0, sizeof(int));

        /* Set up hybrid state */
        hybrid_t h;
        memset(&h, 0, sizeof(h));
        h.task = spec;
        h.num_inputs = spec.num_inputs;
        h.use_branches = 1;
        h.oep.enabled = 1;
        h.oep.table_bits = 22;  /* 4M entries = 16 MB — 4x bigger than enumerate.c */
        h.split_depth = split;
        h.target_depth = target_depth;
        h.h_batch = h_batch;
        h.d_batch = d_batch;
        h.batch_count = 0;
        h.batch_capacity = batch_size;
        h.gpu_nc = gpu_nc;
        h.suffix_space = suffix_space;
        h.output_port = spec.output_ports[0];
        h.d_solutions = d_solutions;
        h.d_solution_lens = d_solution_lens;
        h.d_num_solutions = d_num_solutions;
        h.max_solutions = max_solutions;
        h.solutions_found = 0;

        /* Set up scaffold */
        uint8_t initial_live = 0;
        for (int i = 0; i < spec.num_inputs; i++) {
            h.program[i] = ENCODE_I(OP_IN, i + 1, spec.input_ports[i]);
            initial_live |= (1 << (i + 1));
        }

        /* Clear OEP tables */
        for (int d = 0; d < 16; d++) oep_clear_depth(&h.oep, d);

        struct timespec ds, de;
        clock_gettime(CLOCK_MONOTONIC, &ds);

        if (suffix_depth == 0) {
            /* Pure CPU: split == target, just try output at each prefix */
            /* For this case, we need try_output logic. Handle via GPU with
             * suffix_depth=0: each prefix × 7 output regs, trivial kernel.
             * Actually simpler: just emit all prefixes and let GPU check outputs.
             * But suffix_space=1 with suffix_depth=0 means no suffix insns,
             * just OUT+HLT. The suffix kernel handles this correctly. */
            h.suffix_space = 1; /* no suffix instructions */
        }

        hybrid_enumerate(&h, 0, target_depth, initial_live);

        /* Flush remaining batch */
        if (h.batch_count > 0 && h.solutions_found == 0)
            hybrid_flush_batch(&h);

        clock_gettime(CLOCK_MONOTONIC, &de);

        printf("  cpu: %" PRIu64 " nodes, %" PRIu64 " prefixes emitted, "
               "OEP pruned: %" PRIu64 ", OEP saturated: %" PRIu64 "\n",
               h.cpu_tried, h.prefixes_emitted, h.oep.pruned, h.oep.saturated);
        printf("  elapsed: %.3f sec\n", elapsed_sec(&ds, &de));

        /* Check results */
        int hits = 0;
        cudaMemcpy(&hits, d_num_solutions, sizeof(int), cudaMemcpyDeviceToHost);

        if (hits > 0) {
            found_depth = target_depth;
            int stored = hits < max_solutions ? hits : max_solutions;

            uint16_t *flat = (uint16_t *)malloc(
                (size_t)stored * MAX_PROG_LEN * sizeof(uint16_t));
            int *lens = (int *)malloc((size_t)stored * sizeof(int));
            cudaMemcpy(flat, d_solutions,
                       (size_t)stored * MAX_PROG_LEN * sizeof(uint16_t),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(lens, d_solution_lens,
                       (size_t)stored * sizeof(int),
                       cudaMemcpyDeviceToHost);

            printf("\nOptimal solution length: %d instructions (kernel depth %d)\n",
                   spec.num_inputs + target_depth + 2, target_depth);

            for (int i = 0; i < stored; i++) {
                int plen = lens[i];
                int passed = task_evaluate(&spec, &flat[i * MAX_PROG_LEN], plen);
                printf("\nSolution #%d  len=%d  cpu_verify=%d/%d %s\n",
                       i + 1, plen, passed, spec.num_tests,
                       passed == spec.num_tests ? "OK" : "FAIL");
                char buf[64];
                for (int j = 0; j < plen; j++) {
                    pca_disasm(flat[i * MAX_PROG_LEN + j], buf, sizeof(buf));
                    printf("    %02X: %04X  %s\n", j,
                           flat[i * MAX_PROG_LEN + j], buf);
                }
            }
            free(flat);
            free(lens);
            break;
        }

        printf("  solutions: 0\n");
        oep_free(&h.oep);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("\nTotal elapsed: %.3f sec\n", elapsed_sec(&t0, &t1));

    free(h_batch);
    cudaFree(d_batch);
    cudaFree(d_solutions);
    cudaFree(d_solution_lens);
    cudaFree(d_num_solutions);

    return (found_depth >= 0) ? 0 : 1;
}
