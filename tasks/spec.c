/*
 * spec.c — Task spec JSON loader and evaluator
 *
 * Minimal JSON parser for our exact schema. Not a general parser.
 */

#include "spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* --- tiny JSON helpers --- */

static const char *skip_ws(const char *p)
{
    while (*p && isspace((unsigned char)*p)) p++;
    return p;
}

/* Find "key": in JSON and return pointer after the colon */
static const char *find_key(const char *json, const char *key)
{
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return NULL;
    p += strlen(pattern);
    p = skip_ws(p);
    if (*p == ':') p++;
    return skip_ws(p);
}

/* Parse a JSON string value, return pointer after closing quote */
static const char *parse_str(const char *p, char *out, int maxlen)
{
    p = skip_ws(p);
    if (*p != '"') return NULL;
    p++;
    int i = 0;
    while (*p && *p != '"' && i < maxlen - 1)
        out[i++] = *p++;
    out[i] = '\0';
    if (*p == '"') p++;
    return p;
}

/* Parse a JSON integer */
static const char *parse_int(const char *p, long *out)
{
    p = skip_ws(p);
    char *end;
    *out = strtol(p, &end, 10);
    return end;
}

/* Parse a JSON array of integers: [n, n, ...] */
static const char *parse_int_array(const char *p, long *arr, int maxlen, int *count)
{
    p = skip_ws(p);
    if (*p != '[') return NULL;
    p++;
    *count = 0;
    while (*count < maxlen) {
        p = skip_ws(p);
        if (*p == ']') { p++; return p; }
        if (*p == ',') p++;
        p = parse_int(p, &arr[*count]);
        if (!p) return NULL;
        (*count)++;
    }
    /* skip to ] */
    while (*p && *p != ']') p++;
    if (*p == ']') p++;
    return p;
}

/* --- loader --- */

int task_load(const char *path, task_spec_t *spec)
{
    memset(spec, 0, sizeof(*spec));

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open task file: %s\n", path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *json = malloc(sz + 1);
    if (!json) { fclose(f); return -1; }
    size_t n = fread(json, 1, sz, f);
    json[n] = '\0';
    fclose(f);

    /* name */
    const char *p = find_key(json, "name");
    if (p) parse_str(p, spec->name, sizeof(spec->name));

    /* max_cycles */
    p = find_key(json, "max_cycles");
    if (p) {
        long v;
        parse_int(p, &v);
        spec->max_cycles = (uint64_t)v;
    } else {
        spec->max_cycles = 1000;
    }

    /* input_ports */
    p = find_key(json, "input_ports");
    if (p) {
        long ports[MAX_IO_PORTS];
        int cnt = 0;
        parse_int_array(p, ports, MAX_IO_PORTS, &cnt);
        spec->num_inputs = cnt;
        for (int i = 0; i < cnt; i++)
            spec->input_ports[i] = (uint8_t)ports[i];
    }

    /* output_ports */
    p = find_key(json, "output_ports");
    if (p) {
        long ports[MAX_IO_PORTS];
        int cnt = 0;
        parse_int_array(p, ports, MAX_IO_PORTS, &cnt);
        spec->num_outputs = cnt;
        for (int i = 0; i < cnt; i++)
            spec->output_ports[i] = (uint8_t)ports[i];
    }

    /* tests */
    p = find_key(json, "tests");
    if (p) {
        p = skip_ws(p);
        if (*p == '[') p++;
        spec->num_tests = 0;
        while (spec->num_tests < MAX_TESTS) {
            p = skip_ws(p);
            if (*p == ']' || *p == '\0') break;
            if (*p == ',') p++;
            p = skip_ws(p);
            if (*p != '{') break;

            /* find "in" and "out" arrays within this object */
            const char *obj_start = p;
            /* find matching } */
            int depth = 1;
            p++;
            while (*p && depth > 0) {
                if (*p == '{') depth++;
                else if (*p == '}') depth--;
                p++;
            }
            /* now obj_start..p is the test object */
            int tidx = spec->num_tests;

            const char *inp = find_key(obj_start, "in");
            if (inp) {
                long vals[MAX_IO_PORTS];
                int cnt = 0;
                parse_int_array(inp, vals, MAX_IO_PORTS, &cnt);
                for (int i = 0; i < cnt; i++)
                    spec->tests[tidx].in[i] = (uint16_t)(vals[i] & 0xFFFF);
            }

            const char *outp = find_key(obj_start, "out");
            if (outp) {
                long vals[MAX_IO_PORTS];
                int cnt = 0;
                parse_int_array(outp, vals, MAX_IO_PORTS, &cnt);
                for (int i = 0; i < cnt; i++)
                    spec->tests[tidx].out[i] = (uint16_t)(vals[i] & 0xFFFF);
            }

            spec->num_tests++;
        }
    }

    free(json);
    return 0;
}

/* --- evaluator --- */

int task_evaluate(const task_spec_t *spec, const uint16_t *prog, int prog_len)
{
    int passed = 0;
    for (int t = 0; t < spec->num_tests; t++) {
        pca_vm_t vm;
        pca_init(&vm);
        pca_load(&vm, prog, prog_len);

        /* set input ports */
        for (int i = 0; i < spec->num_inputs; i++)
            vm.ports[spec->input_ports[i]] = spec->tests[t].in[i];

        /* run */
        pca_run(&vm, spec->max_cycles);

        /* check: must halt and outputs must match */
        if (vm.status != PCA_HALTED) continue;
        int ok = 1;
        for (int i = 0; i < spec->num_outputs; i++) {
            if (vm.ports[spec->output_ports[i]] != spec->tests[t].out[i]) {
                ok = 0;
                break;
            }
        }
        if (ok) passed++;
    }
    return passed;
}

/* --- debug --- */

void task_print(const task_spec_t *spec)
{
    printf("Task: %s\n", spec->name);
    printf("  max_cycles: %llu\n", (unsigned long long)spec->max_cycles);
    printf("  inputs: ");
    for (int i = 0; i < spec->num_inputs; i++) printf("port%d ", spec->input_ports[i]);
    printf("\n  outputs: ");
    for (int i = 0; i < spec->num_outputs; i++) printf("port%d ", spec->output_ports[i]);
    printf("\n  tests: %d\n", spec->num_tests);
    for (int t = 0; t < spec->num_tests; t++) {
        printf("    [%d] in=(", t);
        for (int i = 0; i < spec->num_inputs; i++)
            printf("%s%u", i ? "," : "", spec->tests[t].in[i]);
        printf(") out=(");
        for (int i = 0; i < spec->num_outputs; i++)
            printf("%s%u", i ? "," : "", spec->tests[t].out[i]);
        printf(")\n");
    }
}
