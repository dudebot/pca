; echo.s — read port 0, write port 1, repeat
; simplest possible I/O test

loop:
    IN  r1, 0       ; read sensor
    OUT r1, 1       ; echo to actuator
    BRA loop
