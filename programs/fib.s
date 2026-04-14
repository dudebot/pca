; fib.s — compute fibonacci(10) = 55
;
; registers:
;   r1 = loop counter (counts down)
;   r2 = fib(n-1)  (previous)
;   r3 = fib(n)    (current)
;   r4 = scratch

    LDI r2, 0       ; a = 0
    LDI r3, 1       ; b = 1
    LDI r1, 10      ; n = 10

loop:
    CMP r1, r0      ; if n == 0, done
    BZ  done
    ADD r4, r2, r3  ; temp = a + b
    MOV r2, r3      ; a = b
    MOV r3, r4      ; b = temp
    ADDI r1, -1     ; n--
    BRA loop

done:
    MOV r1, r2      ; result in r1 = fib(10) = 55
    HLT
