# pca

**Predictive coding in assembly.**

The brain doesn't passively receive input — it runs a prediction engine. Perception is the residual: the error between what you expected and what arrived. Minimize the prediction error and you've understood the signal. That's [predictive coding](https://en.wikipedia.org/wiki/Predictive_coding).

A PID controller does the same thing. It holds a setpoint (prediction), reads a sensor (observation), computes the error (surprise), and drives an actuator to minimize it. Same loop, different substrate. One runs on neurons, the other on a 6502.

This repo explores that intersection — control theory, neuroscience, and bare-metal programming on hardware simple enough to hold in your head.

## What's here

- `pid.asm` — A complete PID controller in 6502 assembly. Fixed-point 8.8 arithmetic, integral windup clamping, signed 16-bit multiplication via shift-and-add. ~250 bytes of ROM, ~30 bytes of zero-page RAM.

## The premise

If the reason we abandoned assembly was that humans can't manage the complexity — and machines can manage complexity differently than humans — what happens when you give a system *computational proprioception*? Not reasoning about machine state through text, but a learned sensory modality for the reachable state space itself.

A PID controller is the simplest version of this loop: sense, predict, correct. Start here.

## Toolchain

Targets [ca65](https://cc65.github.io/doc/ca65.html) (part of the cc65 suite). To assemble:

```
ca65 pid.asm -o pid.o
ld65 pid.o -t none -o pid.bin
```

No standard library. No operating system. Just bytes and addresses.
