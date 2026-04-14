; pid.asm — PID controller for the 6502
; predictive coding in assembly: the brain minimizes prediction error,
; this controller minimizes process error. same loop, different substrate.
;
; Fixed-point 8.8 arithmetic (high byte = integer, low byte = fraction)
; Expects a VIA or similar I/O chip mapped at $6000
;
; Hardware assumptions:
;   $6000 = sensor input port  (read)
;   $6001 = actuator output port (write)

; ============================================================
; Zero page variables (fast access, first 256 bytes of RAM)
; ============================================================
ZP_START = $00

setpoint       = ZP_START + $00  ; 16-bit: desired value (8.8)
sensor         = ZP_START + $02  ; 16-bit: current reading (8.8)
error          = ZP_START + $04  ; 16-bit: signed error (8.8)
prev_error     = ZP_START + $06  ; 16-bit: last cycle's error
integral       = ZP_START + $08  ; 16-bit: accumulated error
derivative     = ZP_START + $0A  ; 16-bit: error rate of change
output         = ZP_START + $0C  ; 16-bit: computed output

; PID gains (8.8 fixed point — tune these for your system)
Kp             = ZP_START + $0E  ; 16-bit: proportional gain
Ki             = ZP_START + $10  ; 16-bit: integral gain
Kd             = ZP_START + $12  ; 16-bit: derivative gain

; Scratch space for multiplication
mul_a          = ZP_START + $14  ; 16-bit: multiplicand
mul_b          = ZP_START + $16  ; 16-bit: multiplier
mul_result     = ZP_START + $18  ; 32-bit: product (we use middle 16 bits)

; Integral windup clamp limits (8.8)
int_max        = ZP_START + $1C  ; 16-bit: max integral value
int_min        = ZP_START + $1E  ; 16-bit: min integral value (negative)

; I/O addresses
IO_SENSOR      = $6000
IO_ACTUATOR    = $6001

; ============================================================
; Code origin
; ============================================================
    .org $8000

; ------------------------------------------------------------
; reset: entry point after power-on
; ------------------------------------------------------------
reset:
    CLD                     ; clear decimal mode (binary math only)
    LDX #$FF
    TXS                     ; initialize stack pointer to $01FF

    ; --- set default gains (tune to your plant) ---
    ; Kp = 2.0  ($0200)
    LDA #$02
    STA Kp+1
    LDA #$00
    STA Kp

    ; Ki = 0.25 ($0040)
    LDA #$00
    STA Ki+1
    LDA #$40
    STA Ki

    ; Kd = 1.0  ($0100)
    LDA #$01
    STA Kd+1
    LDA #$00
    STA Kd

    ; --- integral windup limits: +/- 127 ($7F00 / $8100) ---
    LDA #$00
    STA int_max
    LDA #$7F
    STA int_max+1

    LDA #$00
    STA int_min
    LDA #$81
    STA int_min+1

    ; --- setpoint = 128 ($8000 in 8.8) ---
    LDA #$00
    STA setpoint
    LDA #$80
    STA setpoint+1

    ; --- zero out state ---
    LDA #$00
    STA prev_error
    STA prev_error+1
    STA integral
    STA integral+1

; ------------------------------------------------------------
; pid_loop: main control loop — runs forever
; ------------------------------------------------------------
pid_loop:
    ; ---- read sensor ----
    LDA IO_SENSOR
    STA sensor+1            ; sensor high byte = raw reading
    LDA #$00
    STA sensor              ; sensor low byte = 0 (no fractional part)

    ; ---- error = setpoint - sensor ----
    SEC
    LDA setpoint
    SBC sensor
    STA error
    LDA setpoint+1
    SBC sensor+1
    STA error+1

    ; ---- integral += error (with windup clamping) ----
    CLC
    LDA integral
    ADC error
    STA integral
    LDA integral+1
    ADC error+1
    STA integral+1

    ; clamp integral: if integral > int_max, integral = int_max
    JSR clamp_integral

    ; ---- derivative = error - prev_error ----
    SEC
    LDA error
    SBC prev_error
    STA derivative
    LDA error+1
    SBC prev_error+1
    STA derivative+1

    ; ---- save current error for next cycle ----
    LDA error
    STA prev_error
    LDA error+1
    STA prev_error+1

    ; ---- output = Kp*error + Ki*integral + Kd*derivative ----

    ; -- P term: Kp * error --
    LDA error
    STA mul_a
    LDA error+1
    STA mul_a+1
    LDA Kp
    STA mul_b
    LDA Kp+1
    STA mul_b+1
    JSR multiply_16
    ; result in mul_result (32-bit), we want the middle 16 bits (8.8)
    LDA mul_result+1
    STA output
    LDA mul_result+2
    STA output+1

    ; -- I term: Ki * integral --
    LDA integral
    STA mul_a
    LDA integral+1
    STA mul_a+1
    LDA Ki
    STA mul_b
    LDA Ki+1
    STA mul_b+1
    JSR multiply_16

    ; output += I term
    CLC
    LDA output
    ADC mul_result+1
    STA output
    LDA output+1
    ADC mul_result+2
    STA output+1

    ; -- D term: Kd * derivative --
    LDA derivative
    STA mul_a
    LDA derivative+1
    STA mul_a+1
    LDA Kd
    STA mul_b
    LDA Kd+1
    STA mul_b+1
    JSR multiply_16

    ; output += D term
    CLC
    LDA output
    ADC mul_result+1
    STA output
    LDA output+1
    ADC mul_result+2
    STA output+1

    ; ---- clamp output to 0-255 for actuator ----
    ; if output+1 (high byte, integer part) is negative, clamp to 0
    LDA output+1
    BMI clamp_low
    ; if output+1 > 0, it's already > 255 concept... but we use it directly
    ; (output+1 is the integer part of our 8.8 result)
    BNE clamp_high          ; if high byte > 0 and has bits above $00...
    ; wait: output+1 IS the integer part. valid range is $00-$FF.
    ; if bit 7 is set, it's negative (signed), clamp to 0.
    ; otherwise use it directly.
    JMP write_output

clamp_low:
    LDA #$00
    STA output+1
    JMP write_output

clamp_high:
    ; if we got here, output+1 is positive and nonzero
    ; check if it's > $7F (would mean > 127, still valid for unsigned output)
    ; actually for an 8-bit actuator, any value $00-$FF is valid
    ; the BMI already caught negatives, so just use it
    JMP write_output

write_output:
    LDA output+1
    STA IO_ACTUATOR

    JMP pid_loop            ; loop forever

; ============================================================
; clamp_integral: keep integral within [int_min, int_max]
; ============================================================
clamp_integral:
    ; compare integral to int_max (signed 16-bit)
    LDA integral+1
    CMP int_max+1
    BMI @not_over           ; if integral high < max high, it's fine
    BNE @clamp_to_max       ; if integral high > max high, clamp
    ; high bytes equal, compare low
    LDA integral
    CMP int_max
    BCC @not_over           ; if integral low < max low, fine
@clamp_to_max:
    LDA int_max
    STA integral
    LDA int_max+1
    STA integral+1
    RTS

@not_over:
    ; compare integral to int_min
    LDA integral+1
    CMP int_min+1
    BPL @done               ; if integral high >= min high (signed), fine
    ; integral < int_min, clamp
    LDA int_min
    STA integral
    LDA int_min+1
    STA integral+1
@done:
    RTS

; ============================================================
; multiply_16: signed 16-bit multiply (8.8 × 8.8 → 32-bit result)
;
; Input:  mul_a (16-bit), mul_b (16-bit)
; Output: mul_result (32-bit, little-endian)
;
; Uses shift-and-add. Handles sign separately:
; negate inputs if needed, multiply unsigned, restore sign.
; ============================================================
multiply_16:
    ; clear result
    LDA #$00
    STA mul_result
    STA mul_result+1
    STA mul_result+2
    STA mul_result+3

    ; determine sign of result (XOR of input signs)
    LDA mul_a+1
    EOR mul_b+1
    PHP                     ; save sign flag on stack for later

    ; make mul_a positive if negative
    LDA mul_a+1
    BPL @a_positive
    JSR negate_a
@a_positive:

    ; make mul_b positive if negative
    LDA mul_b+1
    BPL @b_positive
    JSR negate_b
@b_positive:

    ; now multiply unsigned: shift mul_a left, if bit set in mul_b add
    LDY #16                 ; 16 bits to process
@mul_loop:
    ; shift mul_b right, check carry (low bit)
    LSR mul_b+1
    ROR mul_b
    BCC @no_add

    ; add mul_a to result (32-bit add, mul_a in low 16 bits)
    CLC
    LDA mul_result
    ADC mul_a
    STA mul_result
    LDA mul_result+1
    ADC mul_a+1
    STA mul_result+1
    LDA mul_result+2
    ADC #$00
    STA mul_result+2
    LDA mul_result+3
    ADC #$00
    STA mul_result+3

@no_add:
    ; shift mul_a left 1 bit (into 32-bit space)
    ASL mul_a
    ROL mul_a+1
    ; we also need to track overflow into higher bytes
    ; but since we're adding into a 32-bit accumulator, this is handled

    DEY
    BNE @mul_loop

    ; restore sign: if result should be negative, negate it
    PLP                     ; pull saved sign flags
    BPL @mul_done           ; if N flag clear, result is positive
    ; negate 32-bit result
    SEC
    LDA #$00
    SBC mul_result
    STA mul_result
    LDA #$00
    SBC mul_result+1
    STA mul_result+1
    LDA #$00
    SBC mul_result+2
    STA mul_result+2
    LDA #$00
    SBC mul_result+3
    STA mul_result+3

@mul_done:
    RTS

; --- helpers for sign handling ---
negate_a:
    SEC
    LDA #$00
    SBC mul_a
    STA mul_a
    LDA #$00
    SBC mul_a+1
    STA mul_a+1
    RTS

negate_b:
    SEC
    LDA #$00
    SBC mul_b
    STA mul_b
    LDA #$00
    SBC mul_b+1
    STA mul_b+1
    RTS

; ============================================================
; Vectors: tell the 6502 where to go on reset/interrupt
; ============================================================
    .org $FFFA
    .word $0000             ; NMI vector (unused)
    .word reset             ; RESET vector — entry point
    .word $0000             ; IRQ vector (unused)
