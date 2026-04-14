#define _POSIX_C_SOURCE 199309L
/*
 * enumerate.c — Exhaustive program synthesis for PCA-16
 *
 * Iterative-deepening DFS over instruction sequences.
 * Finds provably shortest correct programs for a given task spec.
 *
 * Program structure:
 *   IN r1, port0           \
 *   IN r2, port1  (maybe)   | scaffold (fixed)
 *   <kernel[0..K-1]>        | enumerated
 *   OUT rX, output_port     | try each live register
 *   HLT                    /
 *
 * Usage:
 *   enumerate <task.json> [-d max_depth] [--no-branch] [-v]
 */

#include "../src/pca.h"
#include "../tasks/spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* --- configuration --- */

#define MAX_SOLUTIONS   100
#define MAX_CANDIDATES  32768

/* --- observational equivalence pruning (OEP) ---
 *
 * After executing the partial program on the first test input,
 * hash the register file + flags. If the same state was reached
 * by a different instruction sequence at the same depth, the
 * remaining instructions will produce the same output — prune.
 *
 * Uses a simple open-addressing hash table per depth level.
 */
#define OEP_TABLE_BITS 20
#define OEP_TABLE_SIZE (1 << OEP_TABLE_BITS)
#define OEP_TABLE_MASK (OEP_TABLE_SIZE - 1)

typedef struct {
    uint32_t *tables[16];  /* one hash table per kernel depth, lazily allocated */
    uint64_t  pruned;
    int       enabled;
} oep_state_t;

/* Returns 1 if this state was already seen (should prune), 0 if new. */
static int oep_check_and_insert(oep_state_t *oep, int depth, uint32_t hash)
{
    if (!oep->enabled) return 0;
    if (!oep->tables[depth]) {
        oep->tables[depth] = calloc(OEP_TABLE_SIZE, sizeof(uint32_t));
        if (!oep->tables[depth]) return 0;
    }

    /* sentinel: 0 means empty, so we avoid hash==0 */
    uint32_t key = hash | 1;
    uint32_t idx = hash & OEP_TABLE_MASK;

    for (int probe = 0; probe < 32; probe++) {
        uint32_t slot = oep->tables[depth][(idx + probe) & OEP_TABLE_MASK];
        if (slot == 0) {
            oep->tables[depth][(idx + probe) & OEP_TABLE_MASK] = key;
            return 0;  /* new state */
        }
        if (slot == key) {
            oep->pruned++;
            return 1;  /* seen before */
        }
    }
    return 0;  /* table full at this slot, don't prune (conservative) */
}

static void oep_clear_depth(oep_state_t *oep, int depth)
{
    if (oep->tables[depth])
        memset(oep->tables[depth], 0, OEP_TABLE_SIZE * sizeof(uint32_t));
}

static void oep_free(oep_state_t *oep)
{
    for (int i = 0; i < 16; i++)
        free(oep->tables[i]);
}

/* immediates to try with LDI */
static const uint8_t ldi_vals[] = {0, 1, 2, 3, 4, 5, 8, 16, 32, 64, 128, 255};
#define N_LDI ((int)(sizeof(ldi_vals)/sizeof(ldi_vals[0])))

/* signed immediates to try with ADDI */
static const int8_t addi_vals[] = {-1, 1, -2, 2, -3, 3};
#define N_ADDI ((int)(sizeof(addi_vals)/sizeof(addi_vals[0])))

/* --- solver state --- */

typedef struct {
    task_spec_t task;
    uint16_t program[PCA_PROG_SIZE];
    int num_inputs;

    uint16_t solutions[MAX_SOLUTIONS][PCA_PROG_SIZE];
    int      solution_lens[MAX_SOLUTIONS];
    int      num_solutions;

    uint64_t tried;
    int      use_branches;
    int      verbose;

    oep_state_t oep;
} solver_t;

/* --- candidate generation --- */

static int gen_candidates(uint16_t *out, int max, uint8_t live,
                          int pos, int kernel_depth, int use_branches)
{
    int n = 0;

    /* ALU ops: rd = rs OP rt */
    static const uint8_t alu_ops[] = {
        OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD,
        OP_AND, OP_OR,  OP_XOR, OP_SHL, OP_SHR, OP_ASR
    };
    static const int commut[] = {1,0,1,0,0, 1,1,1, 0,0,0};
    int n_alu = (int)(sizeof(alu_ops)/sizeof(alu_ops[0]));

    for (int oi = 0; oi < n_alu; oi++) {
        uint8_t op = alu_ops[oi];
        for (int rd = 1; rd < 8 && n < max; rd++) {
            for (int rs = 0; rs < 8 && n < max; rs++) {
                if (rs && !(live & (1 << rs))) continue;
                int rt0 = commut[oi] ? rs : 0;
                for (int rt = rt0; rt < 8 && n < max; rt++) {
                    if (rt && !(live & (1 << rt))) continue;

                    /* prune pointless instructions */
                    if (rs == 0 && rt == 0) continue;
                    /* ADD rd,rd,r0 = nop for rd */
                    if (op == OP_ADD && rs == (int)rd && rt == 0) continue;
                    if (op == OP_ADD && rt == (int)rd && rs == 0) continue;
                    /* SUB rd,rd,r0 = nop */
                    if (op == OP_SUB && rs == (int)rd && rt == 0) continue;
                    /* MUL with r0 = always 0 */
                    if (op == OP_MUL && (rs == 0 || rt == 0)) continue;
                    /* DIV/MOD by r0 = fault */
                    if ((op == OP_DIV || op == OP_MOD) && rt == 0) continue;
                    /* SHL/SHR/ASR by r0 = nop */
                    if ((op == OP_SHL || op == OP_SHR || op == OP_ASR) && rt == 0) continue;

                    out[n++] = ENCODE_R(op, rd, rs, rt);
                }
            }
        }
    }

    /* CMP rd, rs (sets flags for branches) */
    for (int rd = 0; rd < 8 && n < max; rd++) {
        if (rd && !(live & (1 << rd))) continue;
        for (int rs = 0; rs < 8 && n < max; rs++) {
            if (rs && !(live & (1 << rs))) continue;
            if (rd == 0 && rs == 0) continue;
            if (rd == rs) continue;  /* CMP r,r always zero — rarely useful */
            out[n++] = ENCODE_R(OP_CMP, rd, rs, 0);
        }
    }

    /* BR with forward offsets (conditional only, no COND_AL) */
    if (use_branches) {
        int max_fwd = kernel_depth - pos - 1;  /* max useful forward skip */
        if (max_fwd > 0) {
            for (int cond = 1; cond <= 6; cond++) {
                for (int off = 1; off <= max_fwd && n < max; off++) {
                    out[n++] = ENCODE_I(OP_BR, cond, (uint8_t)(int8_t)off);
                }
            }
        }
    }

    /* LDI rd, imm8 */
    for (int rd = 1; rd < 8 && n < max; rd++) {
        for (int i = 0; i < N_LDI && n < max; i++)
            out[n++] = ENCODE_I(OP_LDI, rd, ldi_vals[i]);
    }

    /* ADDI rd, simm8 (only for live registers) */
    for (int rd = 1; rd < 8 && n < max; rd++) {
        if (!(live & (1 << rd))) continue;
        for (int i = 0; i < N_ADDI && n < max; i++)
            out[n++] = ENCODE_I(OP_ADDI, rd, (uint8_t)addi_vals[i]);
    }

    return n;
}

/* --- enumeration core --- */

static void try_output(solver_t *s, uint8_t live, int kernel_end)
{
    if (s->num_solutions >= MAX_SOLUTIONS) return;

    for (int r = 1; r < 8; r++) {
        if (!(live & (1 << r))) continue;

        s->program[kernel_end] = ENCODE_I(OP_OUT, r, s->task.output_ports[0]);
        s->program[kernel_end + 1] = ENCODE_R(OP_HLT, 0, 0, 0);
        int plen = kernel_end + 2;

        s->tried++;

        if (task_evaluate(&s->task, s->program, plen) == s->task.num_tests) {
            /* found a solution! */
            int idx = s->num_solutions++;
            memcpy(s->solutions[idx], s->program, plen * sizeof(uint16_t));
            s->solution_lens[idx] = plen;
        }
    }
}

static void enumerate_kernel(solver_t *s, int pos, int kernel_depth,
                             uint8_t live)
{
    if (s->num_solutions >= MAX_SOLUTIONS) return;

    int kernel_start = s->num_inputs;

    if (pos == kernel_depth) {
        try_output(s, live, kernel_start + kernel_depth);
        return;
    }

    uint16_t cands[MAX_CANDIDATES];
    int nc = gen_candidates(cands, MAX_CANDIDATES, live,
                            pos, kernel_depth, s->use_branches);

    int prog_pos = kernel_start + pos;
    for (int i = 0; i < nc; i++) {
        if (s->num_solutions >= MAX_SOLUTIONS) return;

        s->program[prog_pos] = cands[i];

        /* update liveness (conservative: only add, never remove) */
        uint8_t new_live = live;
        uint8_t op = INSN_OP(cands[i]);
        uint8_t rd = INSN_RD(cands[i]);
        /* instructions that write to rd */
        if (op != OP_CMP && op != OP_BR && op != OP_NOP && op != OP_HLT &&
            op != OP_ST && rd != 0) {
            new_live |= (1 << rd);
        }

        /* observational equivalence pruning:
         * run partial program on ALL test inputs, hash combined register state.
         * two prefixes that produce identical states on every input are
         * truly equivalent — any suffix that works for one works for the other. */
        if (s->oep.enabled && op != OP_BR) {
            int partial_len = prog_pos + 1;
            s->program[partial_len] = ENCODE_R(OP_HLT, 0, 0, 0);

            uint32_t h = 2166136261u;
            for (int t = 0; t < s->task.num_tests; t++) {
                pca_vm_t vm;
                pca_init(&vm);
                pca_load(&vm, s->program, partial_len + 1);
                for (int p = 0; p < s->task.num_inputs; p++)
                    vm.ports[s->task.input_ports[p]] = s->task.tests[t].in[p];
                pca_run(&vm, 100);

                /* fold this test's register state into hash */
                for (int r = 0; r < PCA_NUM_REGS; r++) {
                    h ^= vm.r[r] & 0xFF;  h *= 16777619u;
                    h ^= vm.r[r] >> 8;    h *= 16777619u;
                }
                h ^= vm.flags; h *= 16777619u;
            }

            if (oep_check_and_insert(&s->oep, pos, h)) {
                continue;
            }
        }

        enumerate_kernel(s, pos + 1, kernel_depth, new_live);
    }
}

/* --- solution display --- */

static void print_solution(solver_t *s, int idx)
{
    int len = s->solution_lens[idx];
    printf("  Solution #%d (%d instructions):\n", idx + 1, len);
    char buf[64];
    for (int i = 0; i < len; i++) {
        pca_disasm(s->solutions[idx][i], buf, sizeof(buf));
        printf("    %02X: %04X  %s\n", i, s->solutions[idx][i], buf);
    }

    /* verify: run on first test case and show state */
    pca_vm_t vm;
    pca_init(&vm);
    pca_load(&vm, s->solutions[idx], len);
    for (int i = 0; i < s->task.num_inputs; i++)
        vm.ports[s->task.input_ports[i]] = s->task.tests[0].in[i];
    pca_run(&vm, s->task.max_cycles);
    printf("    verify test[0]: in=(");
    for (int i = 0; i < s->task.num_inputs; i++)
        printf("%s%u", i ? "," : "", s->task.tests[0].in[i]);
    printf(") → port%d=%u (expected %u) %s\n",
           s->task.output_ports[0],
           vm.ports[s->task.output_ports[0]],
           s->task.tests[0].out[0],
           vm.ports[s->task.output_ports[0]] == s->task.tests[0].out[0] ? "OK" : "FAIL");
}

/* --- main --- */

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: enumerate <task.json> [-d max_depth] [--no-branch] [--no-oep] [-v]\n");
        return 1;
    }

    solver_t solver;
    memset(&solver, 0, sizeof(solver));
    solver.use_branches = 1;
    solver.oep.enabled = 1;
    int max_depth = 4;

    /* parse args */
    const char *task_path = argv[1];
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc)
            max_depth = atoi(argv[++i]);
        else if (strcmp(argv[i], "--no-branch") == 0)
            solver.use_branches = 0;
        else if (strcmp(argv[i], "--no-oep") == 0)
            solver.oep.enabled = 0;
        else if (strcmp(argv[i], "-v") == 0)
            solver.verbose = 1;
    }

    /* load task */
    if (task_load(task_path, &solver.task) < 0) return 1;
    task_print(&solver.task);
    printf("\n");

    /* set up IN scaffold */
    solver.num_inputs = solver.task.num_inputs;
    uint8_t initial_live = 0;
    for (int i = 0; i < solver.num_inputs; i++) {
        int reg = i + 1;  /* r1, r2, ... */
        solver.program[i] = ENCODE_I(OP_IN, reg, solver.task.input_ports[i]);
        initial_live |= (1 << reg);
    }

    printf("Enumerating programs (max kernel depth %d, branches %s, OEP %s)...\n\n",
           max_depth, solver.use_branches ? "on" : "off",
           solver.oep.enabled ? "on" : "off");

    /* iterative deepening */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int depth = 0; depth <= max_depth; depth++) {
        int total_len = solver.num_inputs + depth + 2;  /* +2 for OUT+HLT */
        printf("--- kernel depth %d (total program length %d) ---\n", depth, total_len);

        uint64_t tried_before = solver.tried;
        int sols_before = solver.num_solutions;

        /* clear OEP tables for this depth level */
        for (int d = 0; d < 16; d++)
            oep_clear_depth(&solver.oep, d);

        enumerate_kernel(&solver, 0, depth, initial_live);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        uint64_t tried_this = solver.tried - tried_before;
        int sols_this = solver.num_solutions - sols_before;

        printf("  candidates: %llu  (%.1f M/sec)\n",
               (unsigned long long)tried_this,
               tried_this / elapsed / 1e6);
        if (solver.oep.enabled)
            printf("  OEP pruned: %llu\n", (unsigned long long)solver.oep.pruned);
        printf("  solutions found: %d (total: %d)\n", sols_this, solver.num_solutions);
        printf("  elapsed: %.2f sec\n\n", elapsed);

        /* print new solutions */
        for (int i = sols_before; i < solver.num_solutions; i++)
            print_solution(&solver, i);

        if (solver.num_solutions > 0) {
            printf("\nOptimal solution length: %d instructions (kernel depth %d)\n",
                   total_len, depth);
            printf("Found %d distinct program(s) at this length.\n", solver.num_solutions);
            break;  /* first depth with solutions = optimal */
        }
    }

    if (solver.num_solutions == 0) {
        printf("\nNo solutions found up to kernel depth %d.\n", max_depth);
        printf("Try increasing -d or enabling branches.\n");
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double total = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("\nTotal: %llu candidates in %.2f sec",
           (unsigned long long)solver.tried, total);
    if (solver.oep.enabled)
        printf(" (OEP pruned: %llu)", (unsigned long long)solver.oep.pruned);
    printf("\n");

    oep_free(&solver.oep);
    return solver.num_solutions > 0 ? 0 : 1;
}
