/*
 * spec.h — Task specification format for PCA-16 program synthesis
 *
 * A task is a set of I/O test cases: for each test, specific values are
 * placed on input ports, the program runs, and output ports are checked.
 */

#ifndef SPEC_H
#define SPEC_H

#include "../src/pca.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_TESTS    32
#define MAX_IO_PORTS 8

typedef struct {
    uint16_t in[MAX_IO_PORTS];
    uint16_t out[MAX_IO_PORTS];
} test_case_t;

typedef struct {
    char     name[64];
    uint64_t max_cycles;
    uint8_t  input_ports[MAX_IO_PORTS];
    int      num_inputs;
    uint8_t  output_ports[MAX_IO_PORTS];
    int      num_outputs;
    test_case_t tests[MAX_TESTS];
    int      num_tests;
} task_spec_t;

/* Load a task from a JSON file. Returns 0 on success, -1 on error. */
int task_load(const char *path, task_spec_t *spec);

/* Evaluate a program against a task spec.
 * Returns number of test cases passed (0..num_tests).
 * A program is correct iff result == spec->num_tests. */
int task_evaluate(const task_spec_t *spec, const uint16_t *prog, int prog_len);

/* Print a task summary. */
void task_print(const task_spec_t *spec);

#ifdef __cplusplus
}
#endif

#endif /* SPEC_H */
