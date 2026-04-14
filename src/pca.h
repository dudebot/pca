/*
 * pca.h — PCA-16 ISA definitions, VM state, and API
 *
 * PCA-16: Predictive Coding Architecture, 16-bit
 * A minimal ISA designed for AI-driven program synthesis.
 *
 * - 8 registers (r0 hardwired to 0), 16-bit
 * - 256 words of data memory, 256 instructions of program memory
 * - 16-bit fixed instruction encoding, Harvard architecture
 * - ZNVC flags, 16-entry hardware call stack
 * - 25 opcodes: arithmetic, logic, memory, branch, I/O
 */

#ifndef PCA_H
#define PCA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Architecture constants */
#define PCA_NUM_REGS    8
#define PCA_MEM_SIZE    256
#define PCA_PROG_SIZE   256
#define PCA_STACK_DEPTH 16
#define PCA_NUM_PORTS   8

/* --- Opcodes (5-bit, 0x00–0x18) --- */
enum {
    OP_NOP  = 0x00,  /* no operation                                      */
    OP_HLT  = 0x01,  /* halt                                              */
    OP_ADD  = 0x02,  /* rd = rs + rt                                      */
    OP_SUB  = 0x03,  /* rd = rs - rt                                      */
    OP_MUL  = 0x04,  /* rd = rs * rt  (low 16 bits)                       */
    OP_DIV  = 0x05,  /* rd = (s16)rs / (s16)rt  (fault on /0)             */
    OP_MOD  = 0x06,  /* rd = (s16)rs % (s16)rt  (fault on /0)             */
    OP_AND  = 0x07,  /* rd = rs & rt                                      */
    OP_OR   = 0x08,  /* rd = rs | rt                                      */
    OP_XOR  = 0x09,  /* rd = rs ^ rt                                      */
    OP_SHL  = 0x0A,  /* rd = rs << (rt & 0xF)                             */
    OP_SHR  = 0x0B,  /* rd = rs >> (rt & 0xF)  logical                    */
    OP_ASR  = 0x0C,  /* rd = (s16)rs >> (rt & 0xF)  arithmetic            */
    OP_CMP  = 0x0D,  /* flags = rd - rs  (result discarded)               */
    OP_LD   = 0x0E,  /* rd = mem[rs & 0xFF]                               */
    OP_ST   = 0x0F,  /* mem[rd & 0xFF] = rs                               */
    OP_LDI  = 0x10,  /* rd = zero_extend(imm8)                            */
    OP_LUI  = 0x11,  /* rd = (rd & 0x00FF) | (imm8 << 8)                 */
    OP_ADDI = 0x12,  /* rd += sign_extend(imm8)                           */
    OP_BR   = 0x13,  /* if cond[rd], PC += sign_extend(imm8)              */
    OP_JMP  = 0x14,  /* PC = rd                                           */
    OP_CALL = 0x15,  /* push PC, PC = rd                                  */
    OP_RET  = 0x16,  /* PC = pop                                          */
    OP_IN   = 0x17,  /* rd = ports[imm8 & 7]                              */
    OP_OUT  = 0x18,  /* ports[imm8 & 7] = rd                              */
    OP_COUNT = 0x19
};

/* --- Branch condition codes (stored in rd field of BR) --- */
enum {
    COND_AL  = 0,   /* always                                             */
    COND_EQ  = 1,   /* Z=1  (equal / zero)                                */
    COND_NE  = 2,   /* Z=0  (not equal / not zero)                        */
    COND_LT  = 3,   /* N!=V (signed less than)                            */
    COND_GE  = 4,   /* N==V (signed greater or equal)                     */
    COND_LTU = 5,   /* C=0  (unsigned less than / no carry)               */
    COND_GEU = 6,   /* C=1  (unsigned greater or equal / carry)           */
};

/* --- Flags --- */
#define FLAG_Z  0x01  /* zero          */
#define FLAG_N  0x02  /* negative      */
#define FLAG_C  0x04  /* carry         */
#define FLAG_V  0x08  /* overflow      */

/* --- VM status --- */
#define PCA_RUNNING  0
#define PCA_HALTED   1
#define PCA_TIMEOUT  2
#define PCA_FAULT    3

/* --- Instruction encoding ---
 *
 * Type R (register):  [op:5][rd:3][rs:3][rt:3][pad:2]  = 16 bits
 * Type I (immediate): [op:5][rd:3][imm:8]              = 16 bits
 *
 * The opcode determines which format is used.
 */
#define INSN_OP(i)    (((i) >> 11) & 0x1F)
#define INSN_RD(i)    (((i) >> 8)  & 0x07)
#define INSN_RS(i)    (((i) >> 5)  & 0x07)
#define INSN_RT(i)    (((i) >> 2)  & 0x07)
#define INSN_IMM8(i)  ((i) & 0xFF)
#define INSN_SIMM8(i) ((int8_t)((i) & 0xFF))

#define ENCODE_R(op, rd, rs, rt) \
    ((uint16_t)(((op)&0x1F)<<11 | ((rd)&7)<<8 | ((rs)&7)<<5 | ((rt)&7)<<2))

#define ENCODE_I(op, rd, imm8) \
    ((uint16_t)(((op)&0x1F)<<11 | ((rd)&7)<<8 | ((imm8)&0xFF)))

/* --- VM state (fully exposed for "proprioception") --- */
typedef struct pca_vm {
    uint16_t r[PCA_NUM_REGS];         /* registers, r[0] always 0         */
    uint16_t mem[PCA_MEM_SIZE];       /* data memory                      */
    uint16_t prog[PCA_PROG_SIZE];     /* program memory (Harvard)         */
    uint8_t  pc;                      /* program counter                  */
    uint8_t  sp;                      /* call stack pointer               */
    uint16_t stack[PCA_STACK_DEPTH];  /* hardware call stack              */
    uint8_t  flags;                   /* ZNVC                             */
    uint8_t  status;                  /* PCA_RUNNING, HALTED, etc.        */
    uint64_t cycles;                  /* instruction counter              */

    /* I/O */
    uint16_t ports[PCA_NUM_PORTS];
    void (*on_write)(struct pca_vm *vm, uint8_t port, uint16_t val);
    void (*on_read)(struct pca_vm *vm, uint8_t port);
} pca_vm_t;

/* --- VM API --- */
void pca_init(pca_vm_t *vm);
void pca_load(pca_vm_t *vm, const uint16_t *prog, size_t len);
void pca_step(pca_vm_t *vm);
int  pca_run(pca_vm_t *vm, uint64_t max_cycles);
void pca_dump(const pca_vm_t *vm);

/* --- Assembler --- */
/* Returns instruction count on success, -1 on error.
 * Writes up to max_insns words into prog[].
 * On error, writes message into err (up to errlen bytes). */
int pca_assemble(const char *src, uint16_t *prog, size_t max_insns,
                 char *err, size_t errlen);

/* --- Disassembler --- */
void pca_disasm(uint16_t insn, char *buf, size_t len);
/* Disassemble an entire program */
void pca_disasm_prog(const uint16_t *prog, int count);

#ifdef __cplusplus
}
#endif

#endif /* PCA_H */
