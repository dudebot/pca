/*
 * vm.c — PCA-16 virtual machine execution engine
 *
 * The hot loop is a flat switch dispatch. Each instruction decodes
 * in a few shifts, executes in a handful of operations. No microcode,
 * no pipeline, no surprises. Designed to run millions of programs/sec
 * for training loops.
 */

#include "pca.h"
#include <string.h>
#include <stdio.h>

void pca_init(pca_vm_t *vm)
{
    memset(vm, 0, sizeof(*vm));
    /* status starts as RUNNING, everything else zero */
}

void pca_load(pca_vm_t *vm, const uint16_t *prog, size_t len)
{
    if (len > PCA_PROG_SIZE) len = PCA_PROG_SIZE;
    memcpy(vm->prog, prog, len * sizeof(uint16_t));
}

/* Flag helpers ---------------------------------------------------- */

static inline void flags_zn(pca_vm_t *vm, uint16_t r)
{
    vm->flags = 0;
    if (r == 0)      vm->flags |= FLAG_Z;
    if (r & 0x8000)  vm->flags |= FLAG_N;
}

static inline void flags_add(pca_vm_t *vm, uint16_t a, uint16_t b)
{
    uint32_t full = (uint32_t)a + (uint32_t)b;
    uint16_t r = (uint16_t)full;
    vm->flags = 0;
    if (r == 0)                          vm->flags |= FLAG_Z;
    if (r & 0x8000)                      vm->flags |= FLAG_N;
    if (full > 0xFFFF)                   vm->flags |= FLAG_C;
    if (~(a ^ b) & (a ^ r) & 0x8000)    vm->flags |= FLAG_V;
}

static inline void flags_sub(pca_vm_t *vm, uint16_t a, uint16_t b)
{
    uint16_t r = a - b;
    vm->flags = 0;
    if (r == 0)                          vm->flags |= FLAG_Z;
    if (r & 0x8000)                      vm->flags |= FLAG_N;
    if (a >= b)                          vm->flags |= FLAG_C;
    if ((a ^ b) & (a ^ r) & 0x8000)     vm->flags |= FLAG_V;
}

/* Step ------------------------------------------------------------ */

void pca_step(pca_vm_t *vm)
{
    if (vm->status != PCA_RUNNING) return;

    uint16_t insn = vm->prog[vm->pc];
    uint8_t op   = INSN_OP(insn);
    uint8_t rd   = INSN_RD(insn);
    uint8_t rs   = INSN_RS(insn);
    uint8_t rt   = INSN_RT(insn);
    uint8_t imm  = INSN_IMM8(insn);
    int8_t simm  = (int8_t)imm;

    vm->pc++;   /* advance before execution (branch offsets are relative to next insn) */

    switch (op) {

    /* --- system --- */
    case OP_NOP: break;
    case OP_HLT: vm->status = PCA_HALTED; break;

    /* --- arithmetic (Type R, sets all flags) --- */
    case OP_ADD:
        flags_add(vm, vm->r[rs], vm->r[rt]);
        vm->r[rd] = vm->r[rs] + vm->r[rt];
        break;
    case OP_SUB:
        flags_sub(vm, vm->r[rs], vm->r[rt]);
        vm->r[rd] = vm->r[rs] - vm->r[rt];
        break;
    case OP_MUL: {
        uint32_t prod = (uint32_t)vm->r[rs] * (uint32_t)vm->r[rt];
        vm->r[rd] = (uint16_t)prod;
        flags_zn(vm, vm->r[rd]);
        if (prod > 0xFFFF) vm->flags |= FLAG_C;
        break;
    }
    case OP_DIV:
        if (vm->r[rt] == 0) { vm->status = PCA_FAULT; break; }
        vm->r[rd] = (uint16_t)((int16_t)vm->r[rs] / (int16_t)vm->r[rt]);
        flags_zn(vm, vm->r[rd]);
        break;
    case OP_MOD:
        if (vm->r[rt] == 0) { vm->status = PCA_FAULT; break; }
        vm->r[rd] = (uint16_t)((int16_t)vm->r[rs] % (int16_t)vm->r[rt]);
        flags_zn(vm, vm->r[rd]);
        break;

    /* --- bitwise (Type R, sets Z/N only) --- */
    case OP_AND:
        vm->r[rd] = vm->r[rs] & vm->r[rt];
        flags_zn(vm, vm->r[rd]);
        break;
    case OP_OR:
        vm->r[rd] = vm->r[rs] | vm->r[rt];
        flags_zn(vm, vm->r[rd]);
        break;
    case OP_XOR:
        vm->r[rd] = vm->r[rs] ^ vm->r[rt];
        flags_zn(vm, vm->r[rd]);
        break;

    /* --- shifts (Type R, sets Z/N only) --- */
    case OP_SHL:
        vm->r[rd] = vm->r[rs] << (vm->r[rt] & 0xF);
        flags_zn(vm, vm->r[rd]);
        break;
    case OP_SHR:
        vm->r[rd] = vm->r[rs] >> (vm->r[rt] & 0xF);
        flags_zn(vm, vm->r[rd]);
        break;
    case OP_ASR:
        vm->r[rd] = (uint16_t)((int16_t)vm->r[rs] >> (vm->r[rt] & 0xF));
        flags_zn(vm, vm->r[rd]);
        break;

    /* --- compare (Type R, rd and rs used, result discarded) --- */
    case OP_CMP:
        flags_sub(vm, vm->r[rd], vm->r[rs]);
        break;

    /* --- memory (Type R, 2-register) --- */
    case OP_LD:
        vm->r[rd] = vm->mem[vm->r[rs] & 0xFF];
        break;
    case OP_ST:
        vm->mem[vm->r[rd] & 0xFF] = vm->r[rs];
        break;

    /* --- immediate (Type I) --- */
    case OP_LDI:
        vm->r[rd] = imm;
        break;
    case OP_LUI:
        vm->r[rd] = (vm->r[rd] & 0x00FF) | ((uint16_t)imm << 8);
        break;
    case OP_ADDI:
        flags_add(vm, vm->r[rd], (uint16_t)(int16_t)simm);
        vm->r[rd] = vm->r[rd] + (uint16_t)(int16_t)simm;
        break;

    /* --- branch (Type I, condition in rd field) --- */
    case OP_BR: {
        int take = 0;
        switch (rd) {
        case COND_AL:  take = 1; break;
        case COND_EQ:  take = (vm->flags & FLAG_Z) != 0; break;
        case COND_NE:  take = (vm->flags & FLAG_Z) == 0; break;
        case COND_LT:  take = ((vm->flags & FLAG_N) != 0) != ((vm->flags & FLAG_V) != 0); break;
        case COND_GE:  take = ((vm->flags & FLAG_N) != 0) == ((vm->flags & FLAG_V) != 0); break;
        case COND_LTU: take = (vm->flags & FLAG_C) == 0; break;
        case COND_GEU: take = (vm->flags & FLAG_C) != 0; break;
        }
        if (take) vm->pc = (uint8_t)((int)vm->pc + simm);
        break;
    }

    /* --- jumps (Type R, target in rd) --- */
    case OP_JMP:
        vm->pc = (uint8_t)vm->r[rd];
        break;
    case OP_CALL:
        if (vm->sp >= PCA_STACK_DEPTH) { vm->status = PCA_FAULT; break; }
        vm->stack[vm->sp++] = vm->pc;
        vm->pc = (uint8_t)vm->r[rd];
        break;
    case OP_RET:
        if (vm->sp == 0) { vm->status = PCA_FAULT; break; }
        vm->pc = (uint8_t)vm->stack[--vm->sp];
        break;

    /* --- I/O (Type I) --- */
    case OP_IN:
        if (vm->on_read) vm->on_read(vm, imm & 0x07);
        vm->r[rd] = vm->ports[imm & 0x07];
        break;
    case OP_OUT:
        vm->ports[imm & 0x07] = vm->r[rd];
        if (vm->on_write) vm->on_write(vm, imm & 0x07, vm->r[rd]);
        break;

    default:
        vm->status = PCA_FAULT;
        break;
    }

    vm->r[0] = 0;  /* r0 is always zero */
    vm->cycles++;
}

/* Run ------------------------------------------------------------- */

int pca_run(pca_vm_t *vm, uint64_t max_cycles)
{
    while (vm->status == PCA_RUNNING && vm->cycles < max_cycles) {
        pca_step(vm);
    }
    if (vm->status == PCA_RUNNING && vm->cycles >= max_cycles)
        vm->status = PCA_TIMEOUT;
    return vm->status;
}

/* Debug dump ------------------------------------------------------ */

void pca_dump(const pca_vm_t *vm)
{
    const char *status_str[] = {"RUNNING", "HALTED", "TIMEOUT", "FAULT"};
    printf("PC=0x%02X  SP=%d  FLAGS=[%c%c%c%c]  CYCLES=%llu  STATUS=%s\n",
           vm->pc, vm->sp,
           (vm->flags & FLAG_Z) ? 'Z' : '-',
           (vm->flags & FLAG_N) ? 'N' : '-',
           (vm->flags & FLAG_C) ? 'C' : '-',
           (vm->flags & FLAG_V) ? 'V' : '-',
           (unsigned long long)vm->cycles,
           (vm->status < 4) ? status_str[vm->status] : "???");

    for (int i = 0; i < PCA_NUM_REGS; i++)
        printf("  r%d = 0x%04X  (%d)\n", i, vm->r[i], (int16_t)vm->r[i]);

    /* show non-zero memory */
    int has_mem = 0;
    for (int i = 0; i < PCA_MEM_SIZE; i++) {
        if (vm->mem[i] != 0) {
            if (!has_mem) { printf("  mem:\n"); has_mem = 1; }
            printf("    [0x%02X] = 0x%04X  (%d)\n", i, vm->mem[i], (int16_t)vm->mem[i]);
        }
    }

    /* show non-zero ports */
    for (int i = 0; i < PCA_NUM_PORTS; i++) {
        if (vm->ports[i] != 0)
            printf("  port[%d] = 0x%04X  (%d)\n", i, vm->ports[i], (int16_t)vm->ports[i]);
    }
}
