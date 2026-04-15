#define _POSIX_C_SOURCE 199309L
/*
 * enumerate.c — Exhaustive program synthesis for PCA-16
 *
 * Iterative-deepening DFS over instruction sequences.
 * Finds provably shortest correct programs for a given task spec.
 *
 * With --record: generates Phase 2 training data by recording
 * search states with reachability labels.
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
 *   enumerate <task.json> [-d max_depth] --record <output_dir> [--task-id N] [--neg-stride N]
 */

#include "../src/pca.h"
#include "../src/search_state.h"
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
 * After executing the partial program on ALL test inputs,
 * hash the combined register file + flags. If the same state was reached
 * by a different instruction sequence at the same depth, the
 * remaining instructions will produce the same output — prune.
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

    uint32_t key = hash | 1;
    uint32_t idx = hash & OEP_TABLE_MASK;

    for (int probe = 0; probe < 32; probe++) {
        uint32_t slot = oep->tables[depth][(idx + probe) & OEP_TABLE_MASK];
        if (slot == 0) {
            oep->tables[depth][(idx + probe) & OEP_TABLE_MASK] = key;
            return 0;
        }
        if (slot == key) {
            oep->pruned++;
            return 1;
        }
    }
    return 0;  /* table full at this slot, conservative */
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

/* --- exact hash set for optimal state tracking --- */

#define HASHSET_BITS 18
#define HASHSET_SIZE (1 << HASHSET_BITS)
#define HASHSET_MASK (HASHSET_SIZE - 1)

typedef struct {
    uint64_t *slots;
    int count;
} hashset64_t;

static void hashset_init(hashset64_t *hs)
{
    hs->slots = calloc(HASHSET_SIZE, sizeof(uint64_t));
    hs->count = 0;
}

static void hashset_free(hashset64_t *hs)
{
    free(hs->slots);
    hs->slots = NULL;
    hs->count = 0;
}

static void hashset_clear(hashset64_t *hs)
{
    memset(hs->slots, 0, HASHSET_SIZE * sizeof(uint64_t));
    hs->count = 0;
}

static int hashset_insert(hashset64_t *hs, uint64_t key)
{
    if (!hs->slots) return 0;
    uint64_t k = key | 1;  /* avoid 0 sentinel */
    uint32_t idx = (uint32_t)(key & HASHSET_MASK);
    for (int probe = 0; probe < 64; probe++) {
        uint64_t slot = hs->slots[(idx + probe) & HASHSET_MASK];
        if (slot == 0) {
            hs->slots[(idx + probe) & HASHSET_MASK] = k;
            hs->count++;
            return 1;  /* inserted */
        }
        if (slot == k) return 0;  /* already present */
    }
    return 0;  /* table full */
}

static int hashset_contains(const hashset64_t *hs, uint64_t key)
{
    if (!hs->slots) return 0;
    uint64_t k = key | 1;
    uint32_t idx = (uint32_t)(key & HASHSET_MASK);
    for (int probe = 0; probe < 64; probe++) {
        uint64_t slot = hs->slots[(idx + probe) & HASHSET_MASK];
        if (slot == 0) return 0;
        if (slot == k) return 1;
    }
    return 0;
}

/* --- recorder state --- */

typedef struct {
    /* Per-depth sets of semantic state hashes on optimal trajectories */
    hashset64_t optimal_states[SEARCH_MAX_DEPTH + 1];

    /* Per-depth dedup for recorded states */
    hashset64_t recorded_states[SEARCH_MAX_DEPTH + 1];

    FILE *states_fp;
    FILE *edges_fp;

    uint32_t task_id;
    uint8_t  target_depth;
    int      enabled;
    int      neg_stride;      /* keep 1 of every N negative states */
    uint64_t neg_seen;
    uint64_t states_written;
    uint64_t edges_written;
    uint64_t positives;
    uint64_t negatives;
} recorder_t;

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

    /* Phase 2 search state */
    search_ctx_t ctx;
    int use_search_state;
} solver_t;

/* --- immediates (for branching mode candidate gen) --- */

static const uint8_t ldi_vals[] = {0, 1, 2, 3, 4, 5, 8, 16, 32, 64, 128, 255};
#define N_LDI ((int)(sizeof(ldi_vals)/sizeof(ldi_vals[0])))

static const int8_t addi_vals[] = {-1, 1, -2, 2, -3, 3};
#define N_ADDI ((int)(sizeof(addi_vals)/sizeof(addi_vals[0])))

/* --- candidate generation (with branches, for non-recording mode) --- */

static int gen_candidates(uint16_t *out, int max, uint8_t live,
                          int pos, int kernel_depth, int use_branches)
{
    int n = 0;

    static const uint8_t alu_ops[] = {
        OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD,
        OP_AND, OP_OR,  OP_XOR, OP_SHL, OP_SHR, OP_ASR
    };
    static const int commut[] = {1,0,1,0,0, 1,1,1, 0,0,0};
    int n_alu = (int)(sizeof(alu_ops)/sizeof(alu_ops[0]));

    for (int oi = 0; oi < n_alu; oi++) {
        uint8_t op = alu_ops[oi];
        for (int rd = 1; rd < 8 && n < max; rd++)
            for (int rs = 0; rs < 8 && n < max; rs++) {
                if (rs && !(live & (1 << rs))) continue;
                int rt0 = commut[oi] ? rs : 0;
                for (int rt = rt0; rt < 8 && n < max; rt++) {
                    if (rt && !(live & (1 << rt))) continue;
                    if (rs == 0 && rt == 0) continue;
                    if (op == OP_ADD && rs == (int)rd && rt == 0) continue;
                    if (op == OP_ADD && rt == (int)rd && rs == 0) continue;
                    if (op == OP_SUB && rs == (int)rd && rt == 0) continue;
                    if (op == OP_MUL && (rs == 0 || rt == 0)) continue;
                    if ((op == OP_DIV || op == OP_MOD) && rt == 0) continue;
                    if ((op == OP_SHL || op == OP_SHR || op == OP_ASR) && rt == 0) continue;
                    out[n++] = ENCODE_R(op, rd, rs, rt);
                }
            }
    }

    for (int rd = 0; rd < 8 && n < max; rd++) {
        if (rd && !(live & (1 << rd))) continue;
        for (int rs = 0; rs < 8 && n < max; rs++) {
            if (rs && !(live & (1 << rs))) continue;
            if (rd == 0 && rs == 0) continue;
            if (rd == rs) continue;
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
            int idx = s->num_solutions++;
            memcpy(s->solutions[idx], s->program, plen * sizeof(uint16_t));
            s->solution_lens[idx] = plen;
        }
    }
}

static void try_output_ss(solver_t *s, const search_state_t *state, int kernel_end)
{
    if (s->num_solutions >= MAX_SOLUTIONS) return;

    uint8_t solving = search_state_leaf_outputs(&s->ctx, state);
    for (int r = 1; r < 8; r++) {
        if (!(solving & (1u << r))) continue;

        s->program[kernel_end] = ENCODE_I(OP_OUT, r, s->task.output_ports[0]);
        s->program[kernel_end + 1] = ENCODE_R(OP_HLT, 0, 0, 0);
        int plen = kernel_end + 2;

        s->tried++;

        /* Safety verify against VM */
        if (task_evaluate(&s->task, s->program, plen) == s->task.num_tests) {
            int idx = s->num_solutions++;
            memcpy(s->solutions[idx], s->program, plen * sizeof(uint16_t));
            s->solution_lens[idx] = plen;
        }
    }
}

/* DFS with search_state tracking (branchless mode) */
static void enumerate_kernel_ss(solver_t *s, recorder_t *rec,
                                const search_state_t *state,
                                int pos, int kernel_depth,
                                uint16_t parent_action)
{
    /* During recording, don't stop at MAX_SOLUTIONS — traverse full tree */
    int sol_cap = (rec && rec->enabled) ? 1000000 : MAX_SOLUTIONS;
    if (s->num_solutions >= sol_cap) return;

    int kernel_start = s->num_inputs;

    /* Record state if recording is enabled */
    if (rec && rec->enabled) {
        uint64_t key = search_state_key_hash64(&s->ctx, state);
        if (hashset_insert(&rec->recorded_states[state->depth], key)) {
            int positive = hashset_contains(
                &rec->optimal_states[state->depth], key);

            int do_record = positive;
            if (!positive) {
                rec->neg_seen++;
                do_record = (rec->neg_stride > 0 &&
                             (rec->neg_seen % (uint64_t)rec->neg_stride) == 0);
            }

            if (do_record) {
                dataset_state_record_t drec;
                search_state_serialize(
                    &s->ctx, state, rec->task_id, parent_action,
                    rec->target_depth,
                    positive ? 1 : 0,
                    positive ? (int8_t)(rec->target_depth - state->depth) : -1,
                    &drec);
                fwrite(&drec, sizeof(drec), 1, rec->states_fp);
                rec->states_written++;
                if (positive) rec->positives++;
                else rec->negatives++;
            }
        }
    }

    /* Leaf: try outputs */
    if (pos == kernel_depth) {
        try_output_ss(s, state, kernel_start + kernel_depth);
        return;
    }

    /* Generate branchless candidates */
    uint16_t cands[MAX_CANDIDATES];
    int nc = search_gen_candidates_branchless(cands, MAX_CANDIDATES, state);

    int prog_pos = kernel_start + pos;
    for (int i = 0; i < nc; i++) {
        if (s->num_solutions >= sol_cap) return;

        /* Transition */
        search_state_t child;
        if (!search_state_step(&s->ctx, state, cands[i], &child))
            continue;

        s->program[prog_pos] = cands[i];
        s->tried++;

        /* OEP: use lower 32 bits of 64-bit OEP hash */
        if (s->oep.enabled) {
            uint64_t h64 = search_state_oep_hash64(&s->ctx, &child);
            if (oep_check_and_insert(&s->oep, pos, (uint32_t)h64))
                continue;
        }

        /* Record edge if enabled */
        if (rec && rec->enabled && rec->edges_fp) {
            uint64_t parent_key = search_state_key_hash64(&s->ctx, state);
            uint64_t child_key = search_state_key_hash64(&s->ctx, &child);
            dataset_edge_record_t edge;
            memset(&edge, 0, sizeof(edge));
            edge.parent_hash = parent_key;
            edge.child_hash = child_key;
            edge.action = cands[i];
            edge.child_depth = child.depth;
            fwrite(&edge, sizeof(edge), 1, rec->edges_fp);
            rec->edges_written++;
        }

        enumerate_kernel_ss(s, rec, &child, pos + 1, kernel_depth, cands[i]);
    }
}

/* DFS with VM re-execution (original behavior, for branch mode) */
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

        uint8_t new_live = live;
        uint8_t op = INSN_OP(cands[i]);
        uint8_t rd = INSN_RD(cands[i]);
        if (op != OP_CMP && op != OP_BR && op != OP_NOP && op != OP_HLT &&
            op != OP_ST && rd != 0) {
            new_live |= (1 << rd);
        }

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
                for (int r = 0; r < PCA_NUM_REGS; r++) {
                    h ^= vm.r[r] & 0xFF;  h *= 16777619u;
                    h ^= vm.r[r] >> 8;    h *= 16777619u;
                }
                h ^= vm.flags; h *= 16777619u;
            }

            if (oep_check_and_insert(&s->oep, pos, h))
                continue;
        }

        s->tried++;
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

    pca_vm_t vm;
    pca_init(&vm);
    pca_load(&vm, s->solutions[idx], len);
    for (int i = 0; i < s->task.num_inputs; i++)
        vm.ports[s->task.input_ports[i]] = s->task.tests[0].in[i];
    pca_run(&vm, s->task.max_cycles);
    printf("    verify test[0]: in=(");
    for (int i = 0; i < s->task.num_inputs; i++)
        printf("%s%u", i ? "," : "", s->task.tests[0].in[i]);
    printf(") -> port%d=%u (expected %u) %s\n",
           s->task.output_ports[0],
           vm.ports[s->task.output_ports[0]],
           s->task.tests[0].out[0],
           vm.ports[s->task.output_ports[0]] == s->task.tests[0].out[0] ? "OK" : "FAIL");
}

/* --- Pass 1b: build optimal state sets from found solutions --- */

static void build_optimal_state_sets(solver_t *s, recorder_t *rec)
{
    search_state_t state;
    search_state_init_root(&s->ctx, &state);

    /* Insert root state at depth 0 */
    uint64_t root_key = search_state_key_hash64(&s->ctx, &state);
    hashset_insert(&rec->optimal_states[0], root_key);

    for (int si = 0; si < s->num_solutions; si++) {
        search_state_t cur;
        search_state_init_root(&s->ctx, &cur);

        /* Walk kernel instructions only (skip IN scaffold, OUT, HLT) */
        int kernel_start = s->num_inputs;
        int kernel_end = s->solution_lens[si] - 2;  /* before OUT+HLT */
        int kernel_len = kernel_end - kernel_start;

        for (int k = 0; k < kernel_len; k++) {
            uint16_t insn = s->solutions[si][kernel_start + k];
            search_state_t next;

            if (!search_state_step(&s->ctx, &cur, insn, &next)) {
                fprintf(stderr, "WARNING: optimal solution %d step %d failed\n",
                        si, k);
                break;
            }

            uint64_t key = search_state_key_hash64(&s->ctx, &next);
            hashset_insert(&rec->optimal_states[next.depth], key);
            cur = next;
        }
    }

    int total = 0;
    for (int d = 0; d <= rec->target_depth; d++) {
        total += rec->optimal_states[d].count;
    }
    printf("  optimal state set: %d states across %d depths\n",
           total, rec->target_depth + 1);
}

/* --- main --- */

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
            "Usage: enumerate <task.json> [-d max_depth] [--no-branch] [--no-oep] [-v]\n"
            "       enumerate <task.json> [-d max_depth] --record <output_dir> "
            "[--task-id N] [--neg-stride N]\n");
        return 1;
    }

    solver_t solver;
    memset(&solver, 0, sizeof(solver));
    solver.use_branches = 1;
    solver.oep.enabled = 1;
    int max_depth = 4;

    /* recording options */
    const char *record_dir = NULL;
    uint32_t task_id = 0;
    int neg_stride = 8;

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
        else if (strcmp(argv[i], "--record") == 0 && i + 1 < argc)
            record_dir = argv[++i];
        else if (strcmp(argv[i], "--task-id") == 0 && i + 1 < argc)
            task_id = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--neg-stride") == 0 && i + 1 < argc)
            neg_stride = atoi(argv[++i]);
    }

    /* load task */
    if (task_load(task_path, &solver.task) < 0) return 1;
    task_print(&solver.task);
    printf("\n");

    /* init search context */
    if (search_ctx_init(&solver.ctx, &solver.task) == 0) {
        solver.use_search_state = 1;
    }

    /* recording forces branchless search_state mode */
    if (record_dir) {
        if (!solver.use_search_state) {
            fprintf(stderr, "Cannot record: task not compatible with search_state "
                    "(multi-output?)\n");
            return 1;
        }
        /* recording is branchless */
        solver.use_branches = 0;
    }

    /* set up IN scaffold */
    solver.num_inputs = solver.task.num_inputs;
    uint8_t initial_live = 0;
    for (int i = 0; i < solver.num_inputs; i++) {
        int reg = i + 1;
        solver.program[i] = ENCODE_I(OP_IN, reg, solver.task.input_ports[i]);
        initial_live |= (1 << reg);
    }

    search_state_t root;
    if (solver.use_search_state)
        search_state_init_root(&solver.ctx, &root);

    int branchless_mode = solver.use_search_state && !solver.use_branches;

    printf("Enumerating programs (max kernel depth %d, branches %s, OEP %s%s)...\n\n",
           max_depth, solver.use_branches ? "on" : "off",
           solver.oep.enabled ? "on" : "off",
           branchless_mode ? ", search_state" : "");

    /* ===== PASS 1: Solve (find optimal depth D* and solutions) ===== */

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int optimal_depth = -1;

    for (int depth = 0; depth <= max_depth; depth++) {
        int total_len = solver.num_inputs + depth + 2;
        printf("--- kernel depth %d (total program length %d) ---\n", depth, total_len);

        uint64_t tried_before = solver.tried;
        int sols_before = solver.num_solutions;

        for (int d = 0; d < 16; d++)
            oep_clear_depth(&solver.oep, d);

        if (branchless_mode)
            enumerate_kernel_ss(&solver, NULL, &root, 0, depth, 0xFFFF);
        else
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

        for (int i = sols_before; i < solver.num_solutions; i++)
            print_solution(&solver, i);

        if (solver.num_solutions > 0) {
            optimal_depth = depth;
            printf("\nOptimal solution length: %d instructions (kernel depth %d)\n",
                   total_len, depth);
            printf("Found %d distinct program(s) at this length.\n", solver.num_solutions);
            break;
        }
    }

    if (solver.num_solutions == 0) {
        printf("\nNo solutions found up to kernel depth %d.\n", max_depth);
        oep_free(&solver.oep);
        return 1;
    }

    /* ===== PASS 2: Record dataset (if --record) ===== */

    if (record_dir && optimal_depth >= 0) {
        printf("\n===== Recording dataset (Pass 2) =====\n");

        recorder_t rec;
        memset(&rec, 0, sizeof(rec));
        rec.task_id = task_id;
        rec.target_depth = (uint8_t)optimal_depth;
        rec.neg_stride = neg_stride;
        rec.enabled = 1;

        /* Init hash sets */
        for (int d = 0; d <= optimal_depth; d++) {
            hashset_init(&rec.optimal_states[d]);
            hashset_init(&rec.recorded_states[d]);
        }

        /* Pass 1b: build optimal state sets from found solutions */
        printf("  building optimal state sets from %d solutions...\n",
               solver.num_solutions);
        build_optimal_state_sets(&solver, &rec);

        /* Open output files */
        char path[512];
        snprintf(path, sizeof(path), "%s/states_%06u.bin", record_dir, task_id);
        rec.states_fp = fopen(path, "wb");
        if (!rec.states_fp) {
            fprintf(stderr, "Cannot open %s for writing\n", path);
            goto cleanup_rec;
        }

        snprintf(path, sizeof(path), "%s/edges_%06u.bin", record_dir, task_id);
        rec.edges_fp = fopen(path, "wb");
        if (!rec.edges_fp) {
            fprintf(stderr, "Cannot open %s for writing\n", path);
            goto cleanup_rec;
        }

        /* Reset solver state for Pass 2 — uncap solutions so DFS
         * traverses the full search tree for complete recording */
        solver.tried = 0;
        solver.num_solutions = 0;
        solver.oep.pruned = 0;
        #define RECORD_MAX_SOLUTIONS 1000000  /* effectively unlimited */
        for (int d = 0; d < 16; d++)
            oep_clear_depth(&solver.oep, d);

        struct timespec r0, r1;
        clock_gettime(CLOCK_MONOTONIC, &r0);

        /* Re-run DFS at optimal depth with recording */
        enumerate_kernel_ss(&solver, &rec, &root, 0, optimal_depth, 0xFFFF);

        clock_gettime(CLOCK_MONOTONIC, &r1);
        double rec_elapsed = (r1.tv_sec - r0.tv_sec) +
                             (r1.tv_nsec - r0.tv_nsec) * 1e-9;

        printf("  recording done in %.2f sec\n", rec_elapsed);
        printf("  states: %llu written (%llu positive, %llu negative)\n",
               (unsigned long long)rec.states_written,
               (unsigned long long)rec.positives,
               (unsigned long long)rec.negatives);
        printf("  edges: %llu written\n",
               (unsigned long long)rec.edges_written);

cleanup_rec:
        if (rec.states_fp) fclose(rec.states_fp);
        if (rec.edges_fp) fclose(rec.edges_fp);
        for (int d = 0; d <= optimal_depth; d++) {
            hashset_free(&rec.optimal_states[d]);
            hashset_free(&rec.recorded_states[d]);
        }
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
