; pid.s — PID controller in PCA-16 assembly
;
; Same algorithm as pid.asm (6502), but on a sane ISA with
; native multiply. No shift-and-add ceremony needed.
;
; Memory map (data memory):
;   0x00 = setpoint     (8.8 fixed point)
;   0x01 = prev_error
;   0x02 = integral
;   0x03 = Kp           (8.8, e.g. 0x0200 = 2.0)
;   0x04 = Ki           (8.8, e.g. 0x0040 = 0.25)
;   0x05 = Kd           (8.8, e.g. 0x0100 = 1.0)
;   0x06 = int_max      (windup clamp)
;   0x07 = int_min      (windup clamp, negative)
;
; I/O:
;   port 0 = sensor input  (read)
;   port 1 = actuator output (write)
;
; Registers during main loop:
;   r1 = scratch / current value
;   r2 = error
;   r3 = integral
;   r4 = derivative
;   r5 = output accumulator
;   r6 = gain / temp
;   r7 = address pointer

; ---- init: load constants into data memory ----
init:
    ; setpoint = 0x8000 (128.0 in 8.8)
    LDI r1, 0x00
    LUI r1, 0x80
    LDI r7, 0x00
    ST  r7, r1          ; mem[0] = setpoint

    ; prev_error = 0
    LDI r7, 0x01
    ST  r7, r0          ; mem[1] = 0

    ; integral = 0
    LDI r7, 0x02
    ST  r7, r0          ; mem[2] = 0

    ; Kp = 0x0200 (2.0)
    LDI r1, 0x00
    LUI r1, 0x02
    LDI r7, 0x03
    ST  r7, r1          ; mem[3] = Kp

    ; Ki = 0x0040 (0.25)
    LDI r1, 0x40
    LDI r7, 0x04
    ST  r7, r1          ; mem[4] = Ki

    ; Kd = 0x0100 (1.0)
    LDI r1, 0x00
    LUI r1, 0x01
    LDI r7, 0x05
    ST  r7, r1          ; mem[5] = Kd

    ; int_max = 0x7F00
    LDI r1, 0x00
    LUI r1, 0x7F
    LDI r7, 0x06
    ST  r7, r1          ; mem[6] = int_max

    ; int_min = 0x8100 (-32512 in signed)
    LDI r1, 0x00
    LUI r1, 0x81
    LDI r7, 0x07
    ST  r7, r1          ; mem[7] = int_min

; ---- main PID loop ----
pid_loop:
    ; read sensor → r1 (treat as 8.8: sensor << 8)
    IN  r1, 0
    LDI r2, 8
    SHL r1, r1, r2      ; r1 = sensor << 8  (8.8 format)

    ; error = setpoint - sensor
    LDI r7, 0x00
    LD  r2, r7          ; r2 = setpoint
    SUB r2, r2, r1      ; r2 = error = setpoint - sensor

    ; integral += error
    LDI r7, 0x02
    LD  r3, r7          ; r3 = integral (from memory)
    ADD r3, r3, r2      ; r3 += error

    ; clamp integral
    LDI r7, 0x06
    LD  r6, r7          ; r6 = int_max
    CMP r3, r6
    BLT no_clamp_hi
    MOV r3, r6          ; integral = int_max
no_clamp_hi:
    LDI r7, 0x07
    LD  r6, r7          ; r6 = int_min
    CMP r6, r3
    BLT no_clamp_lo
    MOV r3, r6          ; integral = int_min
no_clamp_lo:
    ; store integral back
    LDI r7, 0x02
    ST  r7, r3

    ; derivative = error - prev_error
    LDI r7, 0x01
    LD  r4, r7          ; r4 = prev_error
    SUB r4, r2, r4      ; r4 = derivative = error - prev_error

    ; save current error as prev_error
    ST  r7, r2          ; mem[1] = error

    ; output = Kp*error + Ki*integral + Kd*derivative
    ; all in 8.8 fixed point: multiply gives 16.16, shift right 8 to get 8.8

    ; P term: (Kp * error) >> 8
    LDI r7, 0x03
    LD  r6, r7          ; r6 = Kp
    MUL r5, r6, r2      ; r5 = Kp * error (low 16 bits)
    LDI r1, 8
    ASR r5, r5, r1      ; r5 >>= 8 (keep 8.8 result)

    ; I term: (Ki * integral) >> 8
    LDI r7, 0x04
    LD  r6, r7          ; r6 = Ki
    MUL r6, r6, r3      ; r6 = Ki * integral
    LDI r1, 8
    ASR r6, r6, r1      ; r6 >>= 8
    ADD r5, r5, r6      ; output += I term

    ; D term: (Kd * derivative) >> 8
    LDI r7, 0x05
    LD  r6, r7          ; r6 = Kd
    MUL r6, r6, r4      ; r6 = Kd * derivative
    LDI r1, 8
    ASR r6, r6, r1      ; r6 >>= 8
    ADD r5, r5, r6      ; output += D term

    ; clamp output to 0..255 for actuator
    ; if negative, clamp to 0
    CMP r5, r0
    BGE not_neg
    MOV r5, r0          ; output = 0
not_neg:
    ; if > 255, clamp to 255
    LDI r6, 255
    CMP r6, r5
    BGE not_over
    MOV r5, r6          ; output = 255
not_over:
    ; write actuator
    OUT r5, 1

    BRA pid_loop
