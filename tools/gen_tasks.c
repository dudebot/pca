#define _POSIX_C_SOURCE 199309L
/*
 * gen_tasks.c — Synthetic task generator for Phase 2 training
 *
 * Samples random branchless PCA-16 programs, executes them on a probe bank,
 * filters for interestingness, and emits task JSON files.
 *
 * Usage:
 *   gen_tasks <output_dir> [-n count] [-s seed] [--verify]
 */

#include "../src/pca.h"
#include "../src/search_state.h"
#include "../tasks/spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* --- configuration --- */

#define MAX_KERNEL_LEN   10

/* Globals set from CLI --min-len / --max-len */
static int g_min_len = 2;
static int g_max_len = 6;
#define MAX_PROBES_UNARY 192   /* 128 random + 64 edge cases */
#define MAX_PROBES_BIN   320   /* 256 random + 64 paired edge cases */
#define MAX_PROBES       MAX_PROBES_BIN
#define NUM_TESTS        10    /* tests per emitted task */
#define MAX_CYCLES       32

/* --- xoshiro256** PRNG --- */

typedef struct {
    uint64_t s[4];
} rng_t;

static uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t rng_next(rng_t *rng) {
    uint64_t *s = rng->s;
    uint64_t result = rotl64(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl64(s[3], 45);
    return result;
}

static void rng_seed(rng_t *rng, uint64_t seed) {
    /* splitmix64 to seed xoshiro */
    for (int i = 0; i < 4; i++) {
        seed += 0x9E3779B97F4A7C15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        rng->s[i] = z ^ (z >> 31);
    }
}

static uint16_t rng_u16(rng_t *rng) {
    return (uint16_t)(rng_next(rng) & 0xFFFF);
}

static int rng_int(rng_t *rng, int lo, int hi) {
    /* inclusive [lo, hi] */
    uint64_t range = (uint64_t)(hi - lo + 1);
    return lo + (int)(rng_next(rng) % range);
}

static double rng_f64(rng_t *rng) {
    return (rng_next(rng) >> 11) * 0x1.0p-53;
}

/* --- edge case constants --- */

static const uint16_t edge_values[] = {
    0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32,
    63, 64, 127, 128, 255, 256, 0x7FFF, 0x8000, 0xFFFF
};
#define N_EDGES ((int)(sizeof(edge_values)/sizeof(edge_values[0])))

/* --- probe bank generation --- */

typedef struct {
    uint16_t inputs[MAX_PROBES][2];  /* up to 2 inputs */
    uint16_t outputs[MAX_PROBES];
    int count;
} probe_bank_t;

static void gen_probes_unary(rng_t *rng, probe_bank_t *pb)
{
    pb->count = 0;
    /* random probes */
    for (int i = 0; i < 128 && pb->count < MAX_PROBES; i++) {
        pb->inputs[pb->count][0] = rng_u16(rng);
        pb->inputs[pb->count][1] = 0;
        pb->count++;
    }
    /* edge case probes */
    for (int i = 0; i < N_EDGES && pb->count < MAX_PROBES; i++) {
        pb->inputs[pb->count][0] = edge_values[i];
        pb->inputs[pb->count][1] = 0;
        pb->count++;
    }
}

static void gen_probes_binary(rng_t *rng, probe_bank_t *pb)
{
    pb->count = 0;
    /* random probes */
    for (int i = 0; i < 256 && pb->count < MAX_PROBES; i++) {
        pb->inputs[pb->count][0] = rng_u16(rng);
        pb->inputs[pb->count][1] = rng_u16(rng);
        pb->count++;
    }
    /* edge case probes */
    for (int i = 0; i < N_EDGES && pb->count < MAX_PROBES; i++) {
        pb->inputs[pb->count][0] = edge_values[i];
        pb->inputs[pb->count][1] = edge_values[i % N_EDGES];
        pb->count++;
    }
    for (int i = 0; i < N_EDGES && pb->count < MAX_PROBES; i++) {
        pb->inputs[pb->count][0] = edge_values[i];
        pb->inputs[pb->count][1] = edge_values[(i + 7) % N_EDGES];
        pb->count++;
    }
}

/* --- weighted instruction sampling --- */

/* Opcode weights for sampling:
 * ADD/SUB: 0.25, AND/OR/XOR: 0.25, SHL/SHR/ASR: 0.15, LDI: 0.15, ADDI: 0.15, MUL: 0.05 */

typedef enum {
    CAT_ADD_SUB, CAT_LOGIC, CAT_SHIFT, CAT_LDI, CAT_ADDI, CAT_MUL,
    CAT_COUNT
} op_cat_t;

static const double cat_weights[] = {0.25, 0.25, 0.15, 0.15, 0.15, 0.05};

static op_cat_t sample_category(rng_t *rng)
{
    double r = rng_f64(rng);
    double cum = 0;
    for (int i = 0; i < CAT_COUNT; i++) {
        cum += cat_weights[i];
        if (r < cum) return (op_cat_t)i;
    }
    return CAT_ADD_SUB;
}

static uint16_t sample_instruction(rng_t *rng, uint8_t live, int num_live)
{
    (void)num_live;
    for (int attempt = 0; attempt < 100; attempt++) {
        op_cat_t cat = sample_category(rng);
        int rd = rng_int(rng, 1, 7);

        switch (cat) {
        case CAT_ADD_SUB: {
            uint8_t op = rng_int(rng, 0, 1) ? OP_ADD : OP_SUB;
            int rs = rng_int(rng, 0, 7);
            int rt = rng_int(rng, 0, 7);
            if (rs && !(live & (1 << rs))) continue;
            if (rt && !(live & (1 << rt))) continue;
            if (rs == 0 && rt == 0) continue;
            if (op == OP_ADD && rs == rd && rt == 0) continue;
            if (op == OP_ADD && rt == rd && rs == 0) continue;
            if (op == OP_SUB && rs == rd && rt == 0) continue;
            return ENCODE_R(op, rd, rs, rt);
        }
        case CAT_LOGIC: {
            static const uint8_t logic_ops[] = {OP_AND, OP_OR, OP_XOR};
            uint8_t op = logic_ops[rng_int(rng, 0, 2)];
            int rs = rng_int(rng, 0, 7);
            int rt = rng_int(rng, 0, 7);
            if (rs && !(live & (1 << rs))) continue;
            if (rt && !(live & (1 << rt))) continue;
            if (rs == 0 && rt == 0) continue;
            return ENCODE_R(op, rd, rs, rt);
        }
        case CAT_SHIFT: {
            static const uint8_t shift_ops[] = {OP_SHL, OP_SHR, OP_ASR};
            uint8_t op = shift_ops[rng_int(rng, 0, 2)];
            int rs = rng_int(rng, 0, 7);
            int rt = rng_int(rng, 1, 7);  /* rt=0 is nop */
            if (rs && !(live & (1 << rs))) continue;
            if (!(live & (1 << rt))) continue;
            return ENCODE_R(op, rd, rs, rt);
        }
        case CAT_LDI: {
            static const uint8_t ldi_vals[] = {0,1,2,3,4,5,8,16,32,64,128,255};
            uint8_t imm = ldi_vals[rng_int(rng, 0, 11)];
            return ENCODE_I(OP_LDI, rd, imm);
        }
        case CAT_ADDI: {
            if (!(live & (1 << rd))) continue;
            static const int8_t addi_vals[] = {-1,1,-2,2,-3,3};
            int8_t simm = addi_vals[rng_int(rng, 0, 5)];
            return ENCODE_I(OP_ADDI, rd, (uint8_t)simm);
        }
        case CAT_MUL: {
            int rs = rng_int(rng, 1, 7);  /* rs=0 gives 0 */
            int rt = rng_int(rng, 1, 7);
            if (!(live & (1 << rs))) continue;
            if (!(live & (1 << rt))) continue;
            return ENCODE_R(OP_MUL, rd, rs, rt);
        }
        default: break;
        }
    }
    /* fallback: LDI r1, 1 */
    return ENCODE_I(OP_LDI, 1, 1);
}

/* --- backward slice for output register selection --- */

static double backward_slice_coverage(const uint16_t *kernel, int len, int out_reg)
{
    uint8_t needed = (uint8_t)(1u << out_reg);
    int used = 0;

    for (int k = len - 1; k >= 0; k--) {
        uint8_t op = INSN_OP(kernel[k]);
        uint8_t rd = INSN_RD(kernel[k]);
        uint8_t rs = INSN_RS(kernel[k]);
        uint8_t rt = INSN_RT(kernel[k]);

        if (!(needed & (1u << rd))) continue;

        used++;
        switch (op) {
        case OP_LDI:
            needed &= ~(uint8_t)(1u << rd);
            break;
        case OP_ADDI:
            /* rd depends on old rd — keep it needed */
            break;
        default:
            /* binary op: rd depends on rs, rt */
            needed &= ~(uint8_t)(1u << rd);
            if (rs) needed |= (uint8_t)(1u << rs);
            if (rt) needed |= (uint8_t)(1u << rt);
            break;
        }
    }

    return len > 0 ? (double)used / (double)len : 0.0;
}

/* --- interestingness filters --- */

static int is_interesting(const probe_bank_t *pb, int num_inputs)
{
    int nprobes = pb->count;

    /* Count distinct outputs */
    uint16_t seen[MAX_PROBES];
    int n_distinct = 0;
    for (int i = 0; i < nprobes; i++) {
        int found = 0;
        for (int j = 0; j < n_distinct; j++) {
            if (seen[j] == pb->outputs[i]) { found = 1; break; }
        }
        if (!found && n_distinct < MAX_PROBES)
            seen[n_distinct++] = pb->outputs[i];
    }
    if (n_distinct < 4) return 0;  /* too few distinct outputs */

    /* Check against simple baselines */
    typedef struct { const char *name; int match; } baseline_t;
    baseline_t baselines[] = {
        {"zero", 0}, {"one", 0}, {"x", 0}, {"y", 0},
        {"x+1", 0}, {"x-1", 0}, {"x+y", 0}, {"x-y", 0},
        {"y-x", 0}, {"x&y", 0}, {"x|y", 0}, {"x^y", 0},
        {NULL, 0}
    };

    for (int i = 0; i < nprobes; i++) {
        uint16_t x = pb->inputs[i][0];
        uint16_t y = num_inputs > 1 ? pb->inputs[i][1] : 0;
        uint16_t out = pb->outputs[i];

        if (out == 0) baselines[0].match++;
        if (out == 1) baselines[1].match++;
        if (out == x) baselines[2].match++;
        if (out == y && num_inputs > 1) baselines[3].match++;
        if (out == (uint16_t)(x + 1)) baselines[4].match++;
        if (out == (uint16_t)(x - 1)) baselines[5].match++;
        if (out == (uint16_t)(x + y)) baselines[6].match++;
        if (out == (uint16_t)(x - y)) baselines[7].match++;
        if (out == (uint16_t)(y - x)) baselines[8].match++;
        if (out == (uint16_t)(x & y)) baselines[9].match++;
        if (out == (uint16_t)(x | y)) baselines[10].match++;
        if (out == (uint16_t)(x ^ y)) baselines[11].match++;
    }

    int threshold = (int)(nprobes * 0.95);
    for (int i = 0; baselines[i].name; i++) {
        if (baselines[i].match > threshold) return 0;
    }

    return 1;
}

/* --- task generation --- */

typedef struct {
    uint16_t kernel[MAX_KERNEL_LEN];
    int kernel_len;
    int num_inputs;
    int out_reg;
    double slice_coverage;
} gen_program_t;

static int generate_one_task(rng_t *rng, gen_program_t *prog, probe_bank_t *pb)
{
    /* Sample num_inputs */
    prog->num_inputs = rng_f64(rng) < 0.4 ? 1 : 2;

    /* Sample kernel length (configurable via g_min_len/g_max_len) */
    prog->kernel_len = g_min_len + rng_int(rng, 0, g_max_len - g_min_len);

    /* Probe bank register state — separate from search_state_t (which is
     * limited to MAX_TESTS=32). We need up to MAX_PROBES (320) entries. */
    static uint16_t probe_regs[MAX_PROBES][PCA_NUM_REGS];
    memset(probe_regs, 0, sizeof(probe_regs));

    if (prog->num_inputs == 1)
        gen_probes_unary(rng, pb);
    else
        gen_probes_binary(rng, pb);

    /* Init register state from probes */
    uint8_t live = 0;
    for (int p = 0; p < pb->count; p++) {
        for (int i = 0; i < prog->num_inputs; i++)
            probe_regs[p][i + 1] = pb->inputs[p][i];
    }
    for (int i = 0; i < prog->num_inputs; i++)
        live |= (uint8_t)(1u << (i + 1));

    /* Sample instructions */
    int num_live = prog->num_inputs;
    for (int k = 0; k < prog->kernel_len; k++) {
        uint16_t insn = sample_instruction(rng, live, num_live);
        prog->kernel[k] = insn;

        uint8_t op = INSN_OP(insn);
        uint8_t rd = INSN_RD(insn);
        uint8_t rs = INSN_RS(insn);
        uint8_t rt = INSN_RT(insn);
        uint8_t imm = INSN_IMM8(insn);
        int8_t simm = INSN_SIMM8(insn);

        for (int p = 0; p < pb->count; p++) {
            uint16_t *reg = probe_regs[p];
            uint16_t res;

            switch (op) {
            case OP_ADD: reg[rd] = reg[rs] + reg[rt]; break;
            case OP_SUB: reg[rd] = reg[rs] - reg[rt]; break;
            case OP_MUL: reg[rd] = (uint16_t)((uint32_t)reg[rs] * reg[rt]); break;
            case OP_AND: reg[rd] = reg[rs] & reg[rt]; break;
            case OP_OR:  reg[rd] = reg[rs] | reg[rt]; break;
            case OP_XOR: reg[rd] = reg[rs] ^ reg[rt]; break;
            case OP_SHL: reg[rd] = reg[rs] << (reg[rt] & 0xF); break;
            case OP_SHR: reg[rd] = reg[rs] >> (reg[rt] & 0xF); break;
            case OP_ASR: reg[rd] = (uint16_t)((int16_t)reg[rs] >> (reg[rt] & 0xF)); break;
            case OP_LDI: reg[rd] = imm; break;
            case OP_ADDI:
                res = (uint16_t)(reg[rd] + (uint16_t)(int16_t)simm);
                reg[rd] = res;
                break;
            default: break;
            }
            reg[0] = 0;
        }

        if (op != OP_CMP && rd != 0)
            live |= (uint8_t)(1u << rd);
        if (rd != 0) num_live++;
    }

    /* Choose output register: best backward slice coverage among live regs */
    prog->out_reg = 0;
    prog->slice_coverage = 0;
    for (int reg = 1; reg < PCA_NUM_REGS; reg++) {
        if (!(live & (1u << reg))) continue;
        double cov = backward_slice_coverage(prog->kernel, prog->kernel_len, reg);
        if (cov > prog->slice_coverage) {
            prog->slice_coverage = cov;
            prog->out_reg = reg;
        }
    }
    if (prog->out_reg == 0 || prog->slice_coverage < 0.5) return 0;

    /* Extract outputs from probes */
    for (int p = 0; p < pb->count; p++)
        pb->outputs[p] = probe_regs[p][prog->out_reg];

    /* Interestingness check */
    if (!is_interesting(pb, prog->num_inputs)) return 0;

    return 1;
}

/* --- emit task JSON --- */

static void emit_task_json(const char *dir, int task_num,
                           const gen_program_t *prog, const probe_bank_t *pb,
                           rng_t *rng)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/syn_%dinp_len%d_%06d.json",
             dir, prog->num_inputs, prog->kernel_len, task_num);

    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }

    /* Select NUM_TESTS probes, stratified by output value */
    int selected[NUM_TESTS];
    int nsel = 0;

    /* Always include some edge case probes (they're at the end of the bank) */
    if (pb->count > 128) {
        selected[nsel++] = 128;  /* first edge case */
        selected[nsel++] = 129;
    }

    /* Fill remaining with random unique probes */
    while (nsel < NUM_TESTS) {
        int idx = rng_int(rng, 0, pb->count - 1);
        int dup = 0;
        for (int j = 0; j < nsel; j++)
            if (selected[j] == idx) { dup = 1; break; }
        if (!dup) selected[nsel++] = idx;
    }

    fprintf(f, "{\n");
    fprintf(f, "    \"name\": \"syn_%dinp_len%d_%06d\",\n",
            prog->num_inputs, prog->kernel_len, task_num);
    fprintf(f, "    \"max_cycles\": %d,\n", MAX_CYCLES);
    fprintf(f, "    \"input_ports\": [");
    for (int i = 0; i < prog->num_inputs; i++)
        fprintf(f, "%s%d", i ? ", " : "", i);
    fprintf(f, "],\n");
    fprintf(f, "    \"output_ports\": [0],\n");
    fprintf(f, "    \"tests\": [\n");
    for (int i = 0; i < NUM_TESTS; i++) {
        int idx = selected[i];
        fprintf(f, "        {\"in\": [");
        for (int j = 0; j < prog->num_inputs; j++)
            fprintf(f, "%s%u", j ? ", " : "", pb->inputs[idx][j]);
        fprintf(f, "], \"out\": [%u]}", pb->outputs[idx]);
        fprintf(f, "%s\n", i < NUM_TESTS - 1 ? "," : "");
    }
    fprintf(f, "    ]\n}\n");
    fclose(f);
}

/* --- main --- */

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: gen_tasks <output_dir> [-n count] [-s seed] [--verify] [--min-len N] [--max-len N]\n");
        return 1;
    }

    const char *output_dir = argv[1];
    int target_count = 1000;
    uint64_t seed = 42;
    int verify = 0;
    int cli_min_len = 0;  /* 0 = use default weighted sampling */
    int cli_max_len = 0;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            target_count = atoi(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc)
            seed = (uint64_t)atoll(argv[++i]);
        else if (strcmp(argv[i], "--verify") == 0)
            verify = 1;
        else if (strcmp(argv[i], "--min-len") == 0 && i + 1 < argc)
            cli_min_len = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-len") == 0 && i + 1 < argc)
            cli_max_len = atoi(argv[++i]);
    }

    if (cli_min_len > 0) g_min_len = cli_min_len;
    if (cli_max_len > 0) g_max_len = cli_max_len;
    if (g_max_len > MAX_KERNEL_LEN) g_max_len = MAX_KERNEL_LEN;
    if (g_min_len > g_max_len) g_min_len = g_max_len;

    rng_t rng;
    rng_seed(&rng, seed);

    printf("Generating %d synthetic tasks (seed=%llu, len=%d..%d)...\n",
           target_count, (unsigned long long)seed, g_min_len, g_max_len);

    int generated = 0;
    int attempted = 0;
    int rejected_slice = 0;
    int rejected_interest = 0;
    int rejected_verify = 0;

    probe_bank_t pb;
    gen_program_t prog;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    while (generated < target_count && attempted < target_count * 50) {
        attempted++;

        if (attempted % 1000 == 0) {
            printf("  attempt %d: gen=%d slice_rej=%d interest_rej=%d verify_rej=%d\n",
                   attempted, generated, rejected_slice, rejected_interest, rejected_verify);
            fflush(stdout);
        }

        if (!generate_one_task(&rng, &prog, &pb)) {
            if (prog.out_reg == 0 || prog.slice_coverage < 0.5)
                rejected_slice++;
            else
                rejected_interest++;
            continue;
        }

        /* Optional: verify with solver that optimal depth is in range */
        if (verify) {
            /* Build a temporary task spec and solve it */
            task_spec_t tmp;
            memset(&tmp, 0, sizeof(tmp));
            snprintf(tmp.name, sizeof(tmp.name), "verify_%d", generated);
            tmp.max_cycles = MAX_CYCLES;
            tmp.num_inputs = prog.num_inputs;
            tmp.num_outputs = 1;
            for (int i = 0; i < prog.num_inputs; i++)
                tmp.input_ports[i] = (uint8_t)i;
            tmp.output_ports[0] = 0;

            /* Use 10 random probes as tests */
            tmp.num_tests = NUM_TESTS < MAX_TESTS ? NUM_TESTS : MAX_TESTS;
            for (int t = 0; t < tmp.num_tests; t++) {
                int idx = rng_int(&rng, 0, pb.count - 1);
                for (int j = 0; j < prog.num_inputs; j++)
                    tmp.tests[t].in[j] = pb.inputs[idx][j];
                tmp.tests[t].out[0] = pb.outputs[idx];
            }

            /* Quick branchless solve check: just verify it's solvable at
             * the generating kernel length (it should be by construction) */
            search_ctx_t ctx;
            if (search_ctx_init(&ctx, &tmp) != 0) {
                rejected_verify++;
                continue;
            }

            /* Build the generating program and verify it solves the task */
            uint16_t full_prog[32];
            int plen = 0;
            for (int i = 0; i < prog.num_inputs; i++)
                full_prog[plen++] = ENCODE_I(OP_IN, i + 1, i);
            for (int k = 0; k < prog.kernel_len; k++)
                full_prog[plen++] = prog.kernel[k];
            full_prog[plen++] = ENCODE_I(OP_OUT, prog.out_reg, 0);
            full_prog[plen++] = ENCODE_R(OP_HLT, 0, 0, 0);

            int passed = task_evaluate(&tmp, full_prog, plen);
            if (passed != tmp.num_tests) {
                rejected_verify++;
                continue;
            }
        }

        emit_task_json(output_dir, generated, &prog, &pb, &rng);
        generated++;

        if (generated % 10 == 0) {
            printf("  %d/%d generated (%d attempted, %.1f%% accept)\n",
                   generated, target_count, attempted,
                   100.0 * generated / attempted);
            fflush(stdout);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

    printf("\nDone: %d tasks generated in %.2f sec\n", generated, elapsed);
    printf("  attempted: %d\n", attempted);
    printf("  rejected (slice < 0.5): %d\n", rejected_slice);
    printf("  rejected (not interesting): %d\n", rejected_interest);
    if (verify)
        printf("  rejected (verify failed): %d\n", rejected_verify);
    printf("  acceptance rate: %.1f%%\n",
           attempted > 0 ? 100.0 * generated / attempted : 0);

    return 0;
}
