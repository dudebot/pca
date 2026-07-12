# PCA-16: How One Non-ML Guy Reran the AlphaDev Playbook on a Desk GPU (and Where the Wall Is)

---

```
O spirit of Donald Knuth, who warned that premature
optimization is the root of all evil — behold: we optimized
nothing prematurely. We enumerated every branchless program
in the universe shorter than nine instructions and kept the
ones that worked. Let the branching factor be merciful.
Let the value function converge. HLT.
```

---

## The Setup

The brain doesn't passively receive input — it runs a prediction engine. Perception is the residual: the error between what you expected and what arrived. That's [predictive coding](https://en.wikipedia.org/wiki/Predictive_coding). A PID controller does the same thing with three opcodes' worth of math: hold a setpoint, read a sensor, minimize the surprise. Same loop, different substrate.

So here's the bet this repo made: if the reason humanity abandoned assembly is that *humans* can't manage the complexity, what happens when you give a machine **computational proprioception** — not reasoning about machine state through text, but a learned sensory modality for the reachable state space itself? The VM state is a 266-element vector. That's small enough for a neural network to swallow whole. The action space is the instruction set. The reward is correctness. The dream at the end of the road: prompt → fully performant assembly. An AlphaGo for programs.

To test it you need hardware simple enough to hold in your head, so I built one: **PCA-16**, a 16-bit ISA with 25 opcodes, 8 registers (`r0` hardwired to zero), 256 words of memory, and an emulator that runs ~338M instructions/sec — a fibonacci-length program evaluates in ~20μs, which means millions of candidate programs per second for search.

(Yes, the repo is called `pca` and contains no principal component analysis. Predictive Coding in Assembly. Mostly.)

The plan had the shape of a ratchet:

1. **Exhaustively solve tiny tasks** → perfect training data, no reward shaping, no approximations
2. **Train a value network** on that data → learned search guidance
3. **Guided search solves deeper tasks** than brute force can → new training data
4. **Bootstrap**: the depth-D model helps generate depth-D+1 data. Repeat until you're writing programs no human would consider.

Steps 1 and 2 worked. Step 3 is where this story gets honest.

---

## Finding 1: Exhaustive Search Works, and It Keeps You Honest

Given only input-output examples, the solver enumerates every possible program and keeps the shortest correct one. Nine hand-crafted tasks solved with *provably* optimal programs — "provably" meaning every shorter program was exhaustively tested and none are correct:

| Task | Optimal length | Description |
|------|---------------|-------------|
| negate | 4 insns | Two's complement negation |
| double | 4 insns | Multiply by 2 |
| square | 4 insns | Square the input |
| add | 5 insns | Add two inputs |
| abs | 6 insns | Absolute value |
| relu | 6 insns | max(0, x) |
| max | 7 insns | Maximum of two inputs |
| saturating_add | 8 insns | Add with overflow clamp |
| is_power2 | 7 insns | Bit trick: x & (-x) |

The solver independently rediscovered the `x & (-x)` lowest-set-bit trick — the same one humans pass around in bit-twiddling folklore. Nobody told it. It enumerated its way there. (is_power2 says 7, not the 8 the CPU solver originally "proved" — hold that thought two paragraphs.)

The raw branching factor per instruction is ~32K. Register liveness analysis, commutativity canonicalization, observational-equivalence pruning, and dead-instruction elimination cut that to ~500–1000 effective. At depth 8 the raw space is ~4×10²¹ candidates; the pruned tree is merely enormous.

**The hybrid CPU/GPU solver** is the artifact I'd defend in front of anyone: the CPU runs iterative-deepening DFS with the full pruning stack, then at a split depth stops recursing and emits a *prefix* — the partial program plus the minimal state a GPU thread needs to validate suffixes without redoing any pruning. Prefixes batch up and flush to the device in one memcpy; each GPU thread brute-forces one (prefix, suffix) pair. CPUs do the branchy thinking, GPUs do the dumb enumeration, and the entire design question is "what's the smallest state you must hand across the boundary?" Read order for the CUDA-curious: `gpu/hybrid.cu` lines 1–13, 54–60, 470–541, 677–680, 769–774.

The number-go-up history deserves its own table, because it happened twice. The hybrid solver, measured on an RTX 2060: abs **277×**, max **972×**, saturating_add 85×, is_power2 **76 minutes → 8.8 seconds (520×)**. Then the CPU solver got ~65× faster on its own (better pruning; is_power2: 76 min → 70 s), so by the time the RTX 5070 Ti table was recorded the *ratio* had collapsed to a modest-looking 5–21× — against a vastly better baseline. End to end, first CPU solver → hybrid on the 5070 Ti: is_power2 went **76 minutes → 3.3 seconds, ~1,400×**. At the original solver's 0.7M candidates/sec-per-core, that's one consumer GPU doing the work of several hundred contemporaneous CPU cores on this workload.

And the best part: **the brute-forcer audited the pruner.** The hybrid GPU found a 7-instruction is_power2 that the CPU solver had "proven" impossible — an OEP hash collision in the CPU's equivalence pruning had silently discarded the true optimum at depth 3. Exhaustive search keeps you honest, including about your exhaustive search.

Scaled up: 1,000 synthetic tasks generated by sampling random programs and deriving I/O specs from their behavior. 949 solved within depth budget → **25.4 million labeled search states**. Perfect supervision. Every label ground truth, because every label was *proven*.

---

## Finding 2: A 280K-Parameter MLP Genuinely Learns to Search

The value network is deliberately tiny: each test case encoded independently by a shared MLP over **bit-level features** (input bits, target bits, register bits, flags — the ISA is full of bitwise ops, and giving the network raw bits lets it reason about bit structure directly), pooled, then two heads: *reachability* (does this partial program extend to a solution?) and *distance* (how many instructions remain?).

Trained on 17.8M states from 663 tasks. Cross-task F1 on 143 held-out tasks: **0.522** (same-task F1 is 0.920 — that gap *is* the generalization challenge, quantified).

The pre-registered milestone question: **can a learned value function beat random search on held-out tasks?**

| Expansions (K) | Model | Random | Lift |
|---|---|---|---|
| 8 | 51.0% | 30.9% | +20.1% |
| 16 | 58.0% | 37.2% | +20.8% |
| 32 | 62.9% | 48.4% | +14.5% |
| 64 | 74.8% | 62.2% | +12.6% |

**Yes.** At tight budgets the model finds solutions 65% more often than random, and it solved 4 tasks random search couldn't reach in any of 5 trials. The advantage narrows as budgets grow — random search eventually stumbles into anything — but guidance learned purely from exhaustive data on *other* tasks transfers. Steps 1 and 2 of the ratchet: real.

---

## Finding 3: The Wall at Depth 4

Step 3 — bootstrap upward forever — is where the ratchet stopped ratcheting. The loop ran; the frontier stuck at depth 4 and stayed there through every escalation. The arithmetic is unkind: positives get exponentially sparser with depth, search runs single-threaded Python/ctypes at ~3.3ms per expansion, and a value function with cross-task F1 of 0.52 buys you a constant-factor improvement against an exponentially growing tree. Constant factor vs exponent is not a fair fight.

One idea remained untested: **exhaustive leaf completion**. The model is weakest deep in the tree, where signal is thinnest — so don't make it navigate the endgame. When guided search gets within radius R=3 of the target depth, hand off to the C brute-forcer to finish the last 3 instructions *exactly* (at ~11M states/sec, a radius-3 completion costs milliseconds). The model plays openings; brute force plays endgames.

That idea deserved a real test — and because "one more idea" is how projects consume years, it got a **pre-registered kill experiment**: the kill rule written down *before* the run, so the answer can't be motivated afterward.

**The rule** (registered 2026-07-07): run the bootstrap loop with R=3 to convergence (escalation exhaustion or 10 rounds). **KILL** if the solved frontier contains zero depth≥6 tasks that a 120-second branchless brute force can't also solve. Depth 6, not 5 — because R=3 completion hands the model depth 5–6 nearly for free, so depth 6 is the honest bar. With an A/B control (R=0 vs R=3, same checkpoint, same budget) so brute force can't quietly do all the work and call it learning.

Training data for the attempt: 778/1000 shallow tasks solved at depth ≤6 (6.6GB of states) plus a harder set — 396/500 solved at depth ≤8 with a 120s timeout, leaving **104 tasks brute force could not crack**. Those 104 are the wall, in census form.

---

## Finding 4: The Verdict

The run went 2026-07-10 → 2026-07-11 and was truncated mid-round-3 for
wall-clock (search phases cost 2h → 4h → 23h as completion calls turned out to
run 20–30 *seconds* each, not the estimated milliseconds — the loop's economics
broke even as its capability didn't). Truncation only understates the result:
more rounds can add beyond-brute-force solves, never subtract them.

**The A/B control passed.** Same checkpoint, same 500-expansion budget:
R=0 solved 392 tasks with nothing past depth 6; R=3 solved 397 **including
2×depth-7 and 4×depth-8**. The completion mechanism pays for itself — the deep
solves are not brute force wearing a trench coat.

**The verdict number: N = 35.** Of 48 depth≥6 tasks the loop solved, **35
cannot be solved by the 120-second brute force at all** — including 15
loop-*generated* frontier tasks (length-10 kernels) where the loop's d7–d8
solutions are *shorter than the programs that generated them*. The
pre-registered rule (KILL if zero) returns: **KEEP, decisively.**

The plot twist worth recording: mid-run, the val-F1 metric was *sinking*
(0.069 → 0.036) and the round times were exploding, and it looked exactly like
bootstrap collapse. It wasn't — each round's val split contains newly generated,
harder frontier tasks, so the metric measures a moving target. While the
scoreboard said "dying," the loop was quietly solving the frontier. Full
verdict artifacts: `kill_report/`, decision tree and caveats: `STATUS.md`.

> **The contingent block, resolved:** strong-KEEP (≥2-depth advance, monotone
> new solutions) was *not* cleanly met — the registered tree lands on **marginal
> KEEP**, whose prescription is one more loop run at MAX_ROUNDS=20 before any
> new code. Practical tension: at measured completion costs that run is weeks
> of wall-clock; the honest next step is fixing the 20–30s completion call
> (sublanguage restriction, threaded C completion) — which is "new code."
> Investment decision deferred to the owner; nothing pre-authorized. The
> machinery works; whether the target deserves it is a strategy question the
> next section already frames.

---

## The Elephant: DeepMind Already Ran This Play, and It's in Your C++ Compiler

Midway through this project's life, the honest context arrived (or rather, was finally looked at squarely): **AlphaDev** (DeepMind, *Nature*, June 2023) is this exact thesis executed with a datacenter. They wrapped AlphaZero around x86-64 assembly — each move appends an instruction, reward is correctness on test cases plus latency — and pointed it at the small fixed-size sorting routines inside `std::sort`: code so hot it executes trillions of times a day, hand-optimized by humans and essentially untouched for a decade.

It beat the humans. The crown jewel: on the 3-element sorting network it found a way to **drop a single instruction** — noticing that after a particular min/max sequence, one comparison was already implied by machine state. One instruction, unseen by a decade of experts staring at a ~17-instruction function. The discovered routines were verified, translated to C++, and **merged into LLVM's libc++** — they shipped in 2022, before the paper even published. If you've compiled C++ with Clang recently, an RL agent's assembly has run on your machine. (Reported wins: ~70% latency improvement on short sequences, ~1.7% on large sorts; they pulled the same trick on protobuf's VarInt deserialization.)

Two lessons in that story, and they bracket this repo perfectly:

1. **Even with a datacenter, the wins live at tiny depth.** DeepMind's headline result was *one instruction* removed from a ~17-instruction function. The combinatorial wall this repo hit at depth 4 is the same wall — they just rammed it with TPU pods, and even then the extractable value was single instructions.
2. **Target selection was the actual genius.** The ML was AlphaZero with a custom representation. The *impact* came from aiming it at libc++'s hottest inner loop, where one instruction × trillions of daily executions = real money. A toy ISA has no such target, and that isn't fixable with better engineering — there is no program on PCA-16 the world is waiting for.

So the thesis didn't fail. It was *right* — proven by a team of top-tier researchers with four-to-five orders of magnitude more compute, on a target a solo project can't access. There are worse epitaphs.

---

## What Does This Mean?

**What worked, and is worth stealing:**

- **Exhaustive small-scale search as a data factory.** Provably-perfect labels, zero reward shaping, zero approximation error. Every ML project should be so lucky with its supervision.
- **Learned search guidance transfers across tasks.** A 280K-param MLP reading raw bits beat random search by 20 points on tasks it had never seen. Small models learn real structure from clean data.
- **The CPU-prefix/GPU-suffix pattern.** Branchy pruning on the CPU, dumb parallel enumeration on the GPU, minimal state across the boundary. Generalizes to any search problem with a cheap-to-verify tail.
- **Pre-registered kill rules.** The single best process decision in the repo: the threshold was written down before the run, so whatever the verdict section above says, it can't have been negotiated with.

**What didn't:**

- **The loop's economics, not its capability.** The bootstrap *did* clear the wall once leaf completion arrived (N=35 beyond brute force) — but each completion call costs 20–30 seconds against a milliseconds estimate, so round times went 2h → 4h → 23h. Model-plays-openings/brute-force-plays-endgames works; making it *affordable* is unfinished engineering.
- **Trusting a val metric over a moving target.** Rounds looked like collapse (F1 sinking) while the frontier was actually being solved — because each round's validation set silently got harder. Metrics need frozen denominators.
- **Assuming novelty without a literature check.** The most expensive lesson here cost the least compute: a half-day of reading about AlphaDev, done at the start, would have reframed the entire project from "build the future" to "replicate the known ceiling at 1/10,000th scale." Vibe-coding is a hell of a drug. Check whether DeepMind already did it — *first*, not at the retrospective.

**What survives regardless of the verdict:** the PCA-16 emulator (a genuinely pleasant tiny ISA), the hybrid solver design, 25.4M provably-labeled states, and the discipline.

---

## Technical Details

- **ISA**: PCA-16 — 25 opcodes, fixed 16-bit encoding (`[op:5][rd:3][rs:3][rt:3][pad:2]` / `[op:5][rd:3][imm:8]`), 8 registers, Harvard architecture, ZNVC flags, 16-entry call stack. Full spec: `src/pca.h`.
- **Emulator**: ~338M insns/sec, flat switch dispatch. `make && ./pca programs/fib.s -q` → `55`.
- **Programs**: `fib.s` (fib(10) in 12 insns / 77 cycles), `echo.s`, `pid.s` (full PID controller, 8.8 fixed-point, integral windup clamping — the predictive-coding namesake).
- **Solvers**: `tools/enumerate.c` (CPU, full pruning stack), `gpu/hybrid.cu` (CPU-prefix/GPU-suffix; 85–972× vs same-era CPU on an RTX 2060, 5–21× vs the later much-faster CPU on an RTX 5070 Ti — see Finding 1), `src/search_state.c` (branchless search-state abstraction + `search_exhaustive_complete()` — the R=3 leaf completion, IDDFS with per-depth OEP tables, ~11M states/s single-thread).
- **Value network**: 280K params — shared per-test-case MLP over bit-level features, mean+max pooling, trunk, reachability + distance heads. `python/model.py`, trained by `python/train.py`.
- **Data**: 1,000 shallow + 500 deep synthetic tasks (in git); state records regenerable via `tools/gen_dataset.sh` (`PRUNE_EDGES=1` keeps peak disk states-only, ~12GB).
- **The kill experiment**: one command — `./run_kill_experiment.sh` (kill rule in its header; decision tree in `STATUS.md`; CPU-only: prefix `CUDA_VISIBLE_DEVICES=`). Reports land in `kill_report/`, logs in `logs/`.

---

*HLT.*
