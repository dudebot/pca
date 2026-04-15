/*
 * search_state.h — Branchless search state for Phase 2 dataset generation
 *
 * CPU analogue of wf_entry_t from gpu/wavefront.cu.
 * Tracks multi-test register states, flags, and liveness through the
 * search tree without re-executing the VM at each node.
 */

#ifndef SEARCH_STATE_H
#define SEARCH_STATE_H

#include "pca.h"
#include "../tasks/spec.h"
#include <stdint.h>
#include <stdio.h>

#define SEARCH_MAX_DEPTH 16

/* Branchless search state: semantic state across all tests + liveness. */
typedef struct {
    uint16_t regs[MAX_TESTS][PCA_NUM_REGS];
    uint8_t  flags[MAX_TESTS];
    uint8_t  live;      /* bit i => r_i has been written / is an input */
    uint8_t  depth;     /* kernel instructions executed so far */
} search_state_t;

/* Per-task immutable context. */
typedef struct {
    task_spec_t task;
    uint8_t num_tests;
    uint8_t num_inputs;
    uint8_t num_outputs;
    uint8_t input_regs[MAX_IO_PORTS];   /* input i loaded into r(i+1) */
    uint8_t output_port;
} search_ctx_t;

/* Dataset record for serialization. */
typedef struct {
    uint64_t state_hash;            /* exact key: regs + flags + live + depth */
    uint64_t oep_hash;              /* regs + flags only (pruning key) */
    uint32_t task_id;
    uint16_t parent_action;         /* 0xFFFF for root */
    uint8_t  depth;
    uint8_t  budget_left;
    uint8_t  live;
    uint8_t  can_finish_optimally;  /* 0 or 1 */
    int8_t   remaining_steps;       /* -1 if not on optimal path */
    uint8_t  _pad[2];

    uint16_t regs[MAX_TESTS][PCA_NUM_REGS];
    uint8_t  flags[MAX_TESTS];
} dataset_state_record_t;

/* Dataset edge record for policy labels. */
typedef struct {
    uint64_t parent_hash;
    uint64_t child_hash;
    uint16_t action;
    uint8_t  child_depth;
    uint8_t  _pad[5];
} dataset_edge_record_t;

/* --- API --- */

/* Initialize context from a task spec. Returns 0 on success, -1 on error. */
int  search_ctx_init(search_ctx_t *ctx, const task_spec_t *task);

/* Set root state: inputs loaded into r1, r2, ..., liveness set. */
void search_state_init_root(const search_ctx_t *ctx, search_state_t *s);

/* Apply a branchless instruction to state.
 * Returns 1 on success, 0 if illegal (fault, wrong opcode class).
 * Allowed opcodes: ADD SUB MUL AND OR XOR SHL SHR ASR LDI ADDI.
 * DIV/MOD allowed but fault on /0 → return 0. */
int  search_state_step(const search_ctx_t *ctx,
                       const search_state_t *src,
                       uint16_t insn,
                       search_state_t *dst);

/* OEP-style hash: regs + flags only (matches enumerate.c pruning semantics). */
uint64_t search_state_oep_hash64(const search_ctx_t *ctx,
                                 const search_state_t *s);

/* Exact key hash: regs + flags + live + depth (for dedup/labeling). */
uint64_t search_state_key_hash64(const search_ctx_t *ctx,
                                 const search_state_t *s);

/* Returns bitmask of registers 1..7 that solve the task at this state.
 * Bit i set means regs[t][i] == expected_out[t] for all tests. */
uint8_t search_state_leaf_outputs(const search_ctx_t *ctx,
                                  const search_state_t *s);

/* Fill a dataset record from state + labels. */
void search_state_serialize(const search_ctx_t *ctx,
                            const search_state_t *s,
                            uint32_t task_id,
                            uint16_t parent_action,
                            uint8_t target_depth,
                            uint8_t can_finish_optimally,
                            int8_t remaining_steps,
                            dataset_state_record_t *out);

/* Generate branchless candidates (no CMP, no BR).
 * Uses liveness from state. Returns count written to out[]. */
int search_gen_candidates_branchless(uint16_t *out, int max,
                                     const search_state_t *s);

/* Debug: print state to file. */
void search_state_dump(const search_ctx_t *ctx,
                       const search_state_t *s, FILE *fp);

#endif /* SEARCH_STATE_H */
