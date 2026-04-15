/*
 * search_state.c — Branchless search state for Phase 2 dataset generation
 *
 * Tracks multi-test register states through DFS without re-executing the
 * full VM at each node. CPU analogue of wf_exec_one from wavefront.cu.
 */

#include "search_state.h"
#include <string.h>
#include <stdio.h>

/* --- context init --- */

int search_ctx_init(search_ctx_t *ctx, const task_spec_t *task)
{
    if (task->num_tests > MAX_TESTS) return -1;
    if (task->num_outputs != 1) return -1;  /* Phase 2 v0: single output */
    if (task->num_inputs > PCA_NUM_REGS - 1) return -1;

    memset(ctx, 0, sizeof(*ctx));
    ctx->task = *task;
    ctx->num_tests = (uint8_t)task->num_tests;
    ctx->num_inputs = (uint8_t)task->num_inputs;
    ctx->num_outputs = (uint8_t)task->num_outputs;
    for (int i = 0; i < task->num_inputs; i++)
        ctx->input_regs[i] = (uint8_t)(i + 1);
    ctx->output_port = task->output_ports[0];
    return 0;
}

/* --- root state --- */

void search_state_init_root(const search_ctx_t *ctx, search_state_t *s)
{
    memset(s, 0, sizeof(*s));
    for (int t = 0; t < ctx->num_tests; t++) {
        for (int i = 0; i < ctx->num_inputs; i++)
            s->regs[t][i + 1] = ctx->task.tests[t].in[i];
    }
    s->live = 0;
    for (int i = 0; i < ctx->num_inputs; i++)
        s->live |= (uint8_t)(1u << (i + 1));
    s->depth = 0;
}

/* --- branchless instruction execution --- */

int search_state_step(const search_ctx_t *ctx,
                      const search_state_t *src,
                      uint16_t insn,
                      search_state_t *dst)
{
    uint8_t op  = INSN_OP(insn);
    uint8_t rd  = INSN_RD(insn);
    uint8_t rs  = INSN_RS(insn);
    uint8_t rt  = INSN_RT(insn);
    uint8_t imm = INSN_IMM8(insn);
    int8_t simm = INSN_SIMM8(insn);

    /* reject non-branchless opcodes */
    switch (op) {
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_MOD:
    case OP_AND: case OP_OR:  case OP_XOR: case OP_SHL: case OP_SHR:
    case OP_ASR: case OP_LDI: case OP_ADDI:
        break;
    default:
        return 0;
    }

    *dst = *src;
    dst->depth = src->depth + 1;

    for (int t = 0; t < ctx->num_tests; t++) {
        uint16_t *r = dst->regs[t];
        uint8_t  *f = &dst->flags[t];
        uint16_t a, b, res;
        uint32_t wide;

        switch (op) {
        case OP_ADD:
            a = r[rs]; b = r[rt];
            wide = (uint32_t)a + (uint32_t)b;
            res = (uint16_t)wide;
            *f = 0;
            if (!res) *f |= FLAG_Z;
            if (res & 0x8000) *f |= FLAG_N;
            if (wide > 0xFFFF) *f |= FLAG_C;
            if (~(a ^ b) & (a ^ res) & 0x8000) *f |= FLAG_V;
            r[rd] = res;
            break;

        case OP_SUB:
            a = r[rs]; b = r[rt];
            res = a - b;
            *f = 0;
            if (!res) *f |= FLAG_Z;
            if (res & 0x8000) *f |= FLAG_N;
            if (a >= b) *f |= FLAG_C;
            if ((a ^ b) & (a ^ res) & 0x8000) *f |= FLAG_V;
            r[rd] = res;
            break;

        case OP_MUL:
            wide = (uint32_t)r[rs] * (uint32_t)r[rt];
            res = (uint16_t)wide;
            *f = 0;
            if (!res) *f |= FLAG_Z;
            if (res & 0x8000) *f |= FLAG_N;
            if (wide > 0xFFFF) *f |= FLAG_C;
            r[rd] = res;
            break;

        case OP_DIV:
            if (!r[rt]) return 0;  /* fault on any test → reject */
            r[rd] = (uint16_t)((int16_t)r[rs] / (int16_t)r[rt]);
            *f = 0;
            if (!r[rd]) *f |= FLAG_Z;
            if (r[rd] & 0x8000) *f |= FLAG_N;
            break;

        case OP_MOD:
            if (!r[rt]) return 0;
            r[rd] = (uint16_t)((int16_t)r[rs] % (int16_t)r[rt]);
            *f = 0;
            if (!r[rd]) *f |= FLAG_Z;
            if (r[rd] & 0x8000) *f |= FLAG_N;
            break;

        case OP_AND:
            r[rd] = r[rs] & r[rt];
            *f = (r[rd] == 0 ? FLAG_Z : 0) | (r[rd] & 0x8000 ? FLAG_N : 0);
            break;

        case OP_OR:
            r[rd] = r[rs] | r[rt];
            *f = (r[rd] == 0 ? FLAG_Z : 0) | (r[rd] & 0x8000 ? FLAG_N : 0);
            break;

        case OP_XOR:
            r[rd] = r[rs] ^ r[rt];
            *f = (r[rd] == 0 ? FLAG_Z : 0) | (r[rd] & 0x8000 ? FLAG_N : 0);
            break;

        case OP_SHL:
            r[rd] = r[rs] << (r[rt] & 0xF);
            *f = (r[rd] == 0 ? FLAG_Z : 0) | (r[rd] & 0x8000 ? FLAG_N : 0);
            break;

        case OP_SHR:
            r[rd] = r[rs] >> (r[rt] & 0xF);
            *f = (r[rd] == 0 ? FLAG_Z : 0) | (r[rd] & 0x8000 ? FLAG_N : 0);
            break;

        case OP_ASR:
            r[rd] = (uint16_t)((int16_t)r[rs] >> (r[rt] & 0xF));
            *f = (r[rd] == 0 ? FLAG_Z : 0) | (r[rd] & 0x8000 ? FLAG_N : 0);
            break;

        case OP_LDI:
            r[rd] = imm;
            /* LDI does NOT modify flags (matches real VM) */
            break;

        case OP_ADDI: {
            uint16_t ext = (uint16_t)(int16_t)simm;
            wide = (uint32_t)r[rd] + (uint32_t)ext;
            res = (uint16_t)wide;
            *f = 0;
            if (!res) *f |= FLAG_Z;
            if (res & 0x8000) *f |= FLAG_N;
            if (wide > 0xFFFF) *f |= FLAG_C;
            if (~(r[rd] ^ ext) & (r[rd] ^ res) & 0x8000) *f |= FLAG_V;
            r[rd] = res;
            break;
        }
        }

        r[0] = 0;  /* r0 always zero */
    }

    /* update liveness */
    if (op != OP_CMP && rd != 0)
        dst->live = src->live | (uint8_t)(1u << rd);

    return 1;
}

/* --- hashing (FNV-1a 64-bit) --- */

static uint64_t fnv64(uint64_t h, uint8_t byte)
{
    h ^= byte;
    h *= 0x00000100000001B3ULL;
    return h;
}

uint64_t search_state_oep_hash64(const search_ctx_t *ctx,
                                 const search_state_t *s)
{
    uint64_t h = 0xCBF29CE484222325ULL;
    for (int t = 0; t < ctx->num_tests; t++) {
        for (int r = 0; r < PCA_NUM_REGS; r++) {
            h = fnv64(h, (uint8_t)(s->regs[t][r] & 0xFF));
            h = fnv64(h, (uint8_t)(s->regs[t][r] >> 8));
        }
        h = fnv64(h, s->flags[t]);
    }
    return h;
}

uint64_t search_state_key_hash64(const search_ctx_t *ctx,
                                 const search_state_t *s)
{
    uint64_t h = search_state_oep_hash64(ctx, s);
    h = fnv64(h, s->live);
    h = fnv64(h, s->depth);
    return h;
}

/* --- leaf check --- */

uint8_t search_state_leaf_outputs(const search_ctx_t *ctx,
                                  const search_state_t *s)
{
    uint8_t result = 0;
    for (int r = 1; r < PCA_NUM_REGS; r++) {
        if (!(s->live & (1u << r))) continue;
        int ok = 1;
        for (int t = 0; t < ctx->num_tests; t++) {
            if (s->regs[t][r] != ctx->task.tests[t].out[0]) {
                ok = 0;
                break;
            }
        }
        if (ok) result |= (uint8_t)(1u << r);
    }
    return result;
}

/* --- serialization --- */

void search_state_serialize(const search_ctx_t *ctx,
                            const search_state_t *s,
                            uint32_t task_id,
                            uint16_t parent_action,
                            uint8_t target_depth,
                            uint8_t can_finish_optimally,
                            int8_t remaining_steps,
                            dataset_state_record_t *out)
{
    memset(out, 0, sizeof(*out));
    out->state_hash = search_state_key_hash64(ctx, s);
    out->oep_hash = search_state_oep_hash64(ctx, s);
    out->task_id = task_id;
    out->parent_action = parent_action;
    out->depth = s->depth;
    out->budget_left = target_depth - s->depth;
    out->live = s->live;
    out->can_finish_optimally = can_finish_optimally;
    out->remaining_steps = remaining_steps;

    for (int t = 0; t < ctx->num_tests; t++) {
        for (int r = 0; r < PCA_NUM_REGS; r++)
            out->regs[t][r] = s->regs[t][r];
        out->flags[t] = s->flags[t];
    }
}

/* --- branchless candidate generation --- */

/* LDI immediates */
static const uint8_t ldi_vals[] = {0, 1, 2, 3, 4, 5, 8, 16, 32, 64, 128, 255};
#define N_LDI ((int)(sizeof(ldi_vals)/sizeof(ldi_vals[0])))

/* ADDI signed immediates */
static const int8_t addi_vals[] = {-1, 1, -2, 2, -3, 3};
#define N_ADDI ((int)(sizeof(addi_vals)/sizeof(addi_vals[0])))

int search_gen_candidates_branchless(uint16_t *out, int max,
                                     const search_state_t *s)
{
    int n = 0;
    uint8_t live = s->live;

    /* ALU ops: rd = rs OP rt */
    static const uint8_t alu_ops[] = {
        OP_ADD, OP_SUB, OP_MUL,
        OP_AND, OP_OR,  OP_XOR, OP_SHL, OP_SHR, OP_ASR
    };
    /* commutative: ADD, MUL, AND, OR, XOR */
    static const int commut[] = {1, 0, 1, 1, 1, 1, 0, 0, 0};
    int n_alu = (int)(sizeof(alu_ops) / sizeof(alu_ops[0]));

    for (int oi = 0; oi < n_alu; oi++) {
        uint8_t op = alu_ops[oi];
        for (int rd = 1; rd < PCA_NUM_REGS && n < max; rd++) {
            for (int rs = 0; rs < PCA_NUM_REGS && n < max; rs++) {
                if (rs && !(live & (1 << rs))) continue;
                int rt0 = commut[oi] ? rs : 0;
                for (int rt = rt0; rt < PCA_NUM_REGS && n < max; rt++) {
                    if (rt && !(live & (1 << rt))) continue;

                    if (rs == 0 && rt == 0) continue;
                    if (op == OP_ADD && rs == rd && rt == 0) continue;
                    if (op == OP_ADD && rt == rd && rs == 0) continue;
                    if (op == OP_SUB && rs == rd && rt == 0) continue;
                    if (op == OP_MUL && (rs == 0 || rt == 0)) continue;
                    if ((op == OP_SHL || op == OP_SHR || op == OP_ASR) && rt == 0) continue;

                    out[n++] = ENCODE_R(op, rd, rs, rt);
                }
            }
        }
    }

    /* LDI rd, imm8 */
    for (int rd = 1; rd < PCA_NUM_REGS && n < max; rd++)
        for (int i = 0; i < N_LDI && n < max; i++)
            out[n++] = ENCODE_I(OP_LDI, rd, ldi_vals[i]);

    /* ADDI rd, simm8 (only for live registers) */
    for (int rd = 1; rd < PCA_NUM_REGS && n < max; rd++) {
        if (!(live & (1 << rd))) continue;
        for (int i = 0; i < N_ADDI && n < max; i++)
            out[n++] = ENCODE_I(OP_ADDI, rd, (uint8_t)addi_vals[i]);
    }

    return n;
}

/* --- debug --- */

void search_state_dump(const search_ctx_t *ctx,
                       const search_state_t *s, FILE *fp)
{
    fprintf(fp, "depth=%d live=0x%02X\n", s->depth, s->live);
    for (int t = 0; t < ctx->num_tests; t++) {
        fprintf(fp, "  test[%d] flags=%02X regs=[", t, s->flags[t]);
        for (int r = 0; r < PCA_NUM_REGS; r++)
            fprintf(fp, "%s%u", r ? "," : "", s->regs[t][r]);
        fprintf(fp, "]\n");
    }
}
