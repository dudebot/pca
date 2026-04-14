/*
 * main.c — PCA-16 CLI: assemble, run, inspect
 *
 * Usage:
 *   pca <program.s>              Assemble and run, dump state on exit
 *   pca <program.s> -c 50000     Set max cycles (default: 100000)
 *   pca <program.s> -t           Trace: print each instruction
 *   pca <program.s> -d           Disassemble only (don't run)
 *   pca <program.s> -q           Quiet: only print r1 on exit
 */

#include "pca.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t n = fread(buf, 1, sz, f);
    buf[n] = '\0';
    fclose(f);
    return buf;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
            "PCA-16 — Predictive Coding Architecture emulator\n"
            "\n"
            "Usage: pca <program.s> [options]\n"
            "  -c N   max cycles (default 100000)\n"
            "  -t     trace execution\n"
            "  -d     disassemble only\n"
            "  -q     quiet: print only r1 value\n"
        );
        return 1;
    }

    const char *path = argv[1];
    uint64_t max_cycles = 100000;
    int trace = 0, disasm_only = 0, quiet = 0;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-c") == 0 && i + 1 < argc)
            max_cycles = strtoull(argv[++i], NULL, 0);
        else if (strcmp(argv[i], "-t") == 0)
            trace = 1;
        else if (strcmp(argv[i], "-d") == 0)
            disasm_only = 1;
        else if (strcmp(argv[i], "-q") == 0)
            quiet = 1;
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    /* read source */
    char *src = read_file(path);
    if (!src) {
        fprintf(stderr, "Error: cannot open '%s'\n", path);
        return 1;
    }

    /* assemble */
    uint16_t prog[PCA_PROG_SIZE];
    char err[256] = {0};
    int ninsns = pca_assemble(src, prog, PCA_PROG_SIZE, err, sizeof(err));
    free(src);

    if (ninsns < 0) {
        fprintf(stderr, "Assembly error: %s\n", err);
        return 1;
    }

    if (!quiet)
        fprintf(stderr, "Assembled %d instructions from %s\n", ninsns, path);

    /* disassemble */
    if (disasm_only || (!quiet && !trace)) {
        if (!quiet) pca_disasm_prog(prog, ninsns);
        if (disasm_only) return 0;
        if (!quiet) printf("\n");
    }

    /* create and run VM */
    pca_vm_t vm;
    pca_init(&vm);
    pca_load(&vm, prog, ninsns);

    if (trace) {
        char buf[64];
        while (vm.status == PCA_RUNNING && vm.cycles < max_cycles) {
            pca_disasm(vm.prog[vm.pc], buf, sizeof(buf));
            printf("[%04llu] PC=0x%02X  %s\n",
                   (unsigned long long)vm.cycles, vm.pc, buf);
            pca_step(&vm);
        }
    } else {
        pca_run(&vm, max_cycles);
    }

    /* output */
    if (quiet) {
        printf("%d\n", (int16_t)vm.r[1]);
    } else {
        printf("--- result ---\n");
        pca_dump(&vm);
    }

    return (vm.status == PCA_HALTED) ? 0 : 1;
}
