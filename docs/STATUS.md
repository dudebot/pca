# PCA Status

## Current Phase: 2 — Learning from Exhaustive Data

### What's Built
- Task manifest (`data/manifest.json`) — 949 solved tasks, train/val/test split (663/143/143)
- Split-aware data loader with pre-extracted numpy arrays (~4GB RAM)
- Value network: 280K params, per-test encoder + mean/max pool + trunk + reachability/distance heads
- Training pipeline: weighted BCE, cosine LR, TensorBoard logging
- Search evaluator (`python/search_eval.py`) — ctypes bindings to libpca.so, best-first search vs random baseline

### Latest Results
- **Cross-task val F1 = 0.522** (Adam, 10 epochs, 17.8M records from 663 tasks)
- **Success@K on 143 held-out test tasks:**

| K | Model | Random | Lift |
|---|-------|--------|------|
| 8 | 51.0% | 30.9% | +20.1% |
| 16 | 58.0% | 37.2% | +20.8% |
| 32 | 62.9% | 48.4% | +14.5% |
| 64 | 74.8% | 62.2% | +12.6% |

- Model solved 4 depth-3 tasks that random search could not (0/5 trials)
- Muon optimizer tested, Adam wins decisively on this model scale (see TRAINING_RESULTS.md)

### Current Limitation
Training data is depth 1-3 only (synthetic tasks from 2-instruction random programs). The model has never seen positive examples at depth 4+. Deeper tasks (max, saturating_add, collatz) require curriculum learning.

### Next Steps
1. **Generate deeper synthetic tasks** — gen_tasks with program lengths 4-6, producing optimal depths 4-6
2. **Record deeper training data** — gen_dataset.sh with higher max_depth, larger neg_stride, longer timeouts
3. **Curriculum retrain** — combine depth 1-3 and depth 4-6 data, retrain value network
4. **Evaluate on harder tasks** — can model-guided search solve max(d*=4), abs(d*=3), saturating_add(d*=5)?
5. **Bootstrap loop** — use trained model to help solve depth 7+ tasks, record solutions, retrain

### What NOT to Build Yet
- Branch support in search_state_t (needs branchless milestone solidified first)
- Policy head (needs stronger value network justification)
- CEGIS verification (Phase 3)

### Feasibility Notes
- Collatz needs ~9-11 total instructions (kernel depth ~6-8) AND branch support — Phase 3 territory
- Depth 6+ brute force is hours-to-days even with GPU hybrid; value-guided search is the path
- The bootstrap hypothesis: learn at depth D, use model to generate data at depth D+1, repeat
