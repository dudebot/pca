/*
 * asm.c — PCA-16 two-pass assembler and disassembler
 *
 * Syntax:
 *   label:              ; label definition
 *   ADD r1, r2, r3      ; 3-register
 *   LDI r1, 42          ; register + immediate (decimal or 0xHH)
 *   BZ  label           ; branch with label reference
 *   MOV r1, r2          ; pseudo: ADD r1, r2, r0
 *   NEG r1, r2          ; pseudo: SUB r1, r0, r2
 *   ; comment
 */

#include "pca.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <strings.h>

/* --- limits --- */
#define MAX_LABELS    128
#define MAX_LABEL_LEN 32
#define MAX_LINE      256
#define MAX_TOKENS    8

/* --- instruction formats --- */
enum {
    FMT_NONE,   /* no operands: NOP, HLT, RET                            */
    FMT_R3,     /* rd, rs, rt: arithmetic, logic, shifts                  */
    FMT_R2,     /* rd, rs: CMP, LD, ST, MOV                              */
    FMT_R1,     /* rd: JMP, CALL                                         */
    FMT_RI,     /* rd, imm8: LDI, LUI, ADDI, IN, OUT                    */
    FMT_BR,     /* label/offset: branch pseudo-instructions               */
    FMT_NEG,    /* rd, rs → SUB rd, r0, rs                               */
};

/* --- mnemonic table --- */
typedef struct {
    const char *name;
    uint8_t opcode;
    uint8_t format;
    uint8_t cond;   /* condition code for BR variants */
} mnemonic_t;

static const mnemonic_t mnemonics[] = {
    /* system */
    {"NOP",  OP_NOP,  FMT_NONE, 0},
    {"HLT",  OP_HLT,  FMT_NONE, 0},
    {"RET",  OP_RET,  FMT_NONE, 0},

    /* arithmetic */
    {"ADD",  OP_ADD,  FMT_R3,   0},
    {"SUB",  OP_SUB,  FMT_R3,   0},
    {"MUL",  OP_MUL,  FMT_R3,   0},
    {"DIV",  OP_DIV,  FMT_R3,   0},
    {"MOD",  OP_MOD,  FMT_R3,   0},

    /* bitwise */
    {"AND",  OP_AND,  FMT_R3,   0},
    {"OR",   OP_OR,   FMT_R3,   0},
    {"XOR",  OP_XOR,  FMT_R3,   0},

    /* shifts */
    {"SHL",  OP_SHL,  FMT_R3,   0},
    {"SHR",  OP_SHR,  FMT_R3,   0},
    {"ASR",  OP_ASR,  FMT_R3,   0},

    /* compare */
    {"CMP",  OP_CMP,  FMT_R2,   0},

    /* memory */
    {"LD",   OP_LD,   FMT_R2,   0},
    {"ST",   OP_ST,   FMT_R2,   0},

    /* immediate */
    {"LDI",  OP_LDI,  FMT_RI,   0},
    {"LUI",  OP_LUI,  FMT_RI,   0},
    {"ADDI", OP_ADDI, FMT_RI,   0},

    /* jump/call */
    {"JMP",  OP_JMP,  FMT_R1,   0},
    {"CALL", OP_CALL, FMT_R1,   0},

    /* I/O */
    {"IN",   OP_IN,   FMT_RI,   0},
    {"OUT",  OP_OUT,  FMT_RI,   0},

    /* branch pseudo-instructions (all map to OP_BR) */
    {"BRA",  OP_BR,   FMT_BR,   COND_AL},
    {"BZ",   OP_BR,   FMT_BR,   COND_EQ},
    {"BEQ",  OP_BR,   FMT_BR,   COND_EQ},
    {"BNZ",  OP_BR,   FMT_BR,   COND_NE},
    {"BNE",  OP_BR,   FMT_BR,   COND_NE},
    {"BN",   OP_BR,   FMT_BR,   COND_LT},
    {"BLT",  OP_BR,   FMT_BR,   COND_LT},
    {"BNN",  OP_BR,   FMT_BR,   COND_GE},
    {"BGE",  OP_BR,   FMT_BR,   COND_GE},
    {"BLTU", OP_BR,   FMT_BR,   COND_LTU},
    {"BGEU", OP_BR,   FMT_BR,   COND_GEU},

    /* pseudo-instructions */
    {"MOV",  OP_ADD,  FMT_R2,   0},   /* MOV rd, rs → ADD rd, rs, r0 */
    {"NEG",  OP_SUB,  FMT_NEG,  0},   /* NEG rd, rs → SUB rd, r0, rs */

    {NULL, 0, 0, 0}
};

/* --- label storage --- */
typedef struct {
    char name[MAX_LABEL_LEN];
    int  addr;
} label_t;

/* --- helpers --- */

static int parse_reg(const char *s)
{
    if ((s[0] == 'r' || s[0] == 'R') && s[1] >= '0' && s[1] <= '7' &&
        (s[2] == '\0' || s[2] == ',' || isspace((unsigned char)s[2])))
        return s[1] - '0';
    return -1;
}

static int parse_imm(const char *s, int *val)
{
    char *end;
    long v = strtol(s, &end, 0);
    if (end == s || (*end != '\0' && !isspace((unsigned char)*end)))
        return -1;
    *val = (int)v;
    return 0;
}

static int find_label(const label_t *labels, int nlabels, const char *name)
{
    for (int i = 0; i < nlabels; i++) {
        if (strcasecmp(labels[i].name, name) == 0)
            return labels[i].addr;
    }
    return -1;
}

static const mnemonic_t *find_mnemonic(const char *name)
{
    for (int i = 0; mnemonics[i].name; i++) {
        if (strcasecmp(mnemonics[i].name, name) == 0)
            return &mnemonics[i];
    }
    return NULL;
}

/* strip leading/trailing whitespace in place, return pointer into buf */
static char *strip(char *s)
{
    while (isspace((unsigned char)*s)) s++;
    char *end = s + strlen(s) - 1;
    while (end > s && isspace((unsigned char)*end)) *end-- = '\0';
    return s;
}

/* tokenize a line into mnemonic + operands, splitting on commas/spaces */
static int tokenize(char *line, char *tokens[], int max_tokens)
{
    int n = 0;

    /* first token: mnemonic (split on whitespace) */
    char *p = line;
    while (isspace((unsigned char)*p)) p++;
    if (*p == '\0') return 0;
    tokens[n++] = p;
    while (*p && !isspace((unsigned char)*p)) p++;
    if (*p) { *p++ = '\0'; }

    /* remaining tokens: split on comma, strip whitespace */
    while (*p && n < max_tokens) {
        while (isspace((unsigned char)*p) || *p == ',') p++;
        if (*p == '\0' || *p == ';') break;
        tokens[n++] = p;
        while (*p && *p != ',' && *p != ';') p++;
        if (*p == ',') { *p++ = '\0'; }
        else if (*p == ';') { *p = '\0'; break; }
        /* trim trailing whitespace from token */
        char *te = p - 1;
        while (te > tokens[n-1] && isspace((unsigned char)*te)) *te-- = '\0';
    }
    return n;
}

/* --- assembler --- */

int pca_assemble(const char *src, uint16_t *prog, size_t max_insns,
                 char *err, size_t errlen)
{
    label_t labels[MAX_LABELS];
    int nlabels = 0;
    int line_num = 0;
    const char *p;
    char line[MAX_LINE];

    /* ---- Pass 1: collect labels ---- */
    p = src;
    int addr = 0;
    while (*p) {
        /* read one line */
        int i = 0;
        while (*p && *p != '\n' && i < MAX_LINE - 1)
            line[i++] = *p++;
        line[i] = '\0';
        if (*p == '\n') p++;
        line_num++;

        /* strip comment */
        char *semi = strchr(line, ';');
        if (semi) *semi = '\0';

        char *s = strip(line);
        if (*s == '\0') continue;

        /* check for label */
        char *colon = strchr(s, ':');
        if (colon) {
            *colon = '\0';
            char *lbl = strip(s);
            if (nlabels >= MAX_LABELS) {
                snprintf(err, errlen, "line %d: too many labels", line_num);
                return -1;
            }
            strncpy(labels[nlabels].name, lbl, MAX_LABEL_LEN - 1);
            labels[nlabels].addr = addr;
            nlabels++;

            /* check if there's an instruction after the label */
            s = strip(colon + 1);
            if (*s == '\0') continue;
        }

        /* it's an instruction — just count it */
        addr++;
        if (addr > (int)max_insns) {
            snprintf(err, errlen, "line %d: program too large (max %zu)", line_num, max_insns);
            return -1;
        }
    }

    /* ---- Pass 2: emit instructions ---- */
    p = src;
    addr = 0;
    line_num = 0;
    while (*p) {
        int i = 0;
        while (*p && *p != '\n' && i < MAX_LINE - 1)
            line[i++] = *p++;
        line[i] = '\0';
        if (*p == '\n') p++;
        line_num++;

        char *semi = strchr(line, ';');
        if (semi) *semi = '\0';

        char *s = strip(line);
        if (*s == '\0') continue;

        /* skip label definition */
        char *colon = strchr(s, ':');
        if (colon) {
            s = strip(colon + 1);
            if (*s == '\0') continue;
        }

        /* tokenize */
        char *tokens[MAX_TOKENS];
        int ntok = tokenize(s, tokens, MAX_TOKENS);
        if (ntok == 0) continue;

        /* look up mnemonic */
        const mnemonic_t *m = find_mnemonic(tokens[0]);
        if (!m) {
            snprintf(err, errlen, "line %d: unknown mnemonic '%s'", line_num, tokens[0]);
            return -1;
        }

        uint16_t insn = 0;
        int rd = 0, rs = 0, rt = 0, imm_val = 0;

        switch (m->format) {
        case FMT_NONE:
            insn = ENCODE_R(m->opcode, 0, 0, 0);
            break;

        case FMT_R3:
            if (ntok < 4) {
                snprintf(err, errlen, "line %d: %s requires 3 registers", line_num, m->name);
                return -1;
            }
            rd = parse_reg(tokens[1]);
            rs = parse_reg(tokens[2]);
            rt = parse_reg(tokens[3]);
            if (rd < 0 || rs < 0 || rt < 0) {
                snprintf(err, errlen, "line %d: bad register in '%s'", line_num, s);
                return -1;
            }
            insn = ENCODE_R(m->opcode, rd, rs, rt);
            break;

        case FMT_R2:
            if (ntok < 3) {
                snprintf(err, errlen, "line %d: %s requires 2 registers", line_num, m->name);
                return -1;
            }
            rd = parse_reg(tokens[1]);
            rs = parse_reg(tokens[2]);
            if (rd < 0 || rs < 0) {
                snprintf(err, errlen, "line %d: bad register in '%s'", line_num, s);
                return -1;
            }
            /* rt defaults to 0 (r0) — correct for MOV→ADD rd,rs,r0 */
            insn = ENCODE_R(m->opcode, rd, rs, 0);
            break;

        case FMT_R1:
            if (ntok < 2) {
                snprintf(err, errlen, "line %d: %s requires 1 register", line_num, m->name);
                return -1;
            }
            rd = parse_reg(tokens[1]);
            if (rd < 0) {
                snprintf(err, errlen, "line %d: bad register in '%s'", line_num, s);
                return -1;
            }
            insn = ENCODE_R(m->opcode, rd, 0, 0);
            break;

        case FMT_RI:
            if (ntok < 3) {
                snprintf(err, errlen, "line %d: %s requires register and immediate", line_num, m->name);
                return -1;
            }
            rd = parse_reg(tokens[1]);
            if (rd < 0) {
                snprintf(err, errlen, "line %d: bad register in '%s'", line_num, s);
                return -1;
            }
            if (parse_imm(tokens[2], &imm_val) < 0) {
                snprintf(err, errlen, "line %d: bad immediate '%s'", line_num, tokens[2]);
                return -1;
            }
            insn = ENCODE_I(m->opcode, rd, imm_val);
            break;

        case FMT_BR:
            if (ntok < 2) {
                snprintf(err, errlen, "line %d: %s requires label or offset", line_num, m->name);
                return -1;
            }
            /* try as label first, then as immediate */
            {
                int target = find_label(labels, nlabels, tokens[1]);
                if (target >= 0) {
                    /* offset relative to next instruction */
                    imm_val = target - (addr + 1);
                } else if (parse_imm(tokens[1], &imm_val) < 0) {
                    snprintf(err, errlen, "line %d: unknown label '%s'", line_num, tokens[1]);
                    return -1;
                }
                if (imm_val < -128 || imm_val > 127) {
                    snprintf(err, errlen, "line %d: branch offset %d out of range (-128..127)",
                             line_num, imm_val);
                    return -1;
                }
                insn = ENCODE_I(m->opcode, m->cond, imm_val);
            }
            break;

        case FMT_NEG:
            /* NEG rd, rs → SUB rd, r0, rs */
            if (ntok < 3) {
                snprintf(err, errlen, "line %d: %s requires 2 registers", line_num, m->name);
                return -1;
            }
            rd = parse_reg(tokens[1]);
            rs = parse_reg(tokens[2]);
            if (rd < 0 || rs < 0) {
                snprintf(err, errlen, "line %d: bad register in '%s'", line_num, s);
                return -1;
            }
            /* SUB rd, r0, rs */
            insn = ENCODE_R(m->opcode, rd, 0, rs);
            break;

        default:
            snprintf(err, errlen, "line %d: internal error (bad format %d)", line_num, m->format);
            return -1;
        }

        prog[addr++] = insn;
    }

    return addr;
}

/* --- Disassembler --- */

static const char *op_names[] = {
    "NOP", "HLT",
    "ADD", "SUB", "MUL", "DIV", "MOD",
    "AND", "OR",  "XOR",
    "SHL", "SHR", "ASR",
    "CMP", "LD",  "ST",
    "LDI", "LUI", "ADDI",
    "BR",
    "JMP", "CALL", "RET",
    "IN",  "OUT"
};

static const char *cond_names[] = {
    "BRA", "BZ", "BNZ", "BLT", "BGE", "BLTU", "BGEU"
};

void pca_disasm(uint16_t insn, char *buf, size_t len)
{
    uint8_t op   = INSN_OP(insn);
    uint8_t rd   = INSN_RD(insn);
    uint8_t rs   = INSN_RS(insn);
    uint8_t rt   = INSN_RT(insn);
    uint8_t imm  = INSN_IMM8(insn);
    int8_t simm  = (int8_t)imm;

    if (op >= OP_COUNT) {
        snprintf(buf, len, "??? 0x%04X", insn);
        return;
    }

    switch (op) {
    case OP_NOP: snprintf(buf, len, "NOP"); break;
    case OP_HLT: snprintf(buf, len, "HLT"); break;
    case OP_RET: snprintf(buf, len, "RET"); break;

    /* 3-register */
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_MOD:
    case OP_AND: case OP_OR:  case OP_XOR:
    case OP_SHL: case OP_SHR: case OP_ASR:
        /* detect MOV pseudo: ADD rd, rs, r0 */
        if (op == OP_ADD && rt == 0)
            snprintf(buf, len, "MOV  r%d, r%d", rd, rs);
        /* detect NEG pseudo: SUB rd, r0, rt */
        else if (op == OP_SUB && rs == 0)
            snprintf(buf, len, "NEG  r%d, r%d", rd, rt);
        else
            snprintf(buf, len, "%-4s r%d, r%d, r%d", op_names[op], rd, rs, rt);
        break;

    /* 2-register */
    case OP_CMP:
        snprintf(buf, len, "CMP  r%d, r%d", rd, rs);
        break;
    case OP_LD:
        snprintf(buf, len, "LD   r%d, r%d", rd, rs);
        break;
    case OP_ST:
        snprintf(buf, len, "ST   r%d, r%d", rd, rs);
        break;

    /* 1-register */
    case OP_JMP:
        snprintf(buf, len, "JMP  r%d", rd);
        break;
    case OP_CALL:
        snprintf(buf, len, "CALL r%d", rd);
        break;

    /* immediate */
    case OP_LDI:
        snprintf(buf, len, "LDI  r%d, %d", rd, imm);
        break;
    case OP_LUI:
        snprintf(buf, len, "LUI  r%d, 0x%02X", rd, imm);
        break;
    case OP_ADDI:
        snprintf(buf, len, "ADDI r%d, %d", rd, simm);
        break;

    /* branch */
    case OP_BR:
        if (rd < 7)
            snprintf(buf, len, "%-4s %+d", cond_names[rd], simm);
        else
            snprintf(buf, len, "BR   %d, %+d", rd, simm);
        break;

    /* I/O */
    case OP_IN:
        snprintf(buf, len, "IN   r%d, %d", rd, imm & 7);
        break;
    case OP_OUT:
        snprintf(buf, len, "OUT  r%d, %d", rd, imm & 7);
        break;

    default:
        snprintf(buf, len, "??? 0x%04X", insn);
        break;
    }
}

void pca_disasm_prog(const uint16_t *prog, int count)
{
    char buf[64];
    for (int i = 0; i < count; i++) {
        pca_disasm(prog[i], buf, sizeof(buf));
        printf("  0x%02X: %04X  %s\n", i, prog[i], buf);
    }
}
