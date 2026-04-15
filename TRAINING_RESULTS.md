# Training Results — Value Network v1

## Key

| Column | Meaning |
|--------|---------|
| **Ep** | Epoch number |
| **Loss** | Training loss: BCE(reachability) + 0.5 * CE(remaining_steps) |
| **tP** | Train Precision — of states predicted positive, % that actually are |
| **tR** | Train Recall — of actually positive states, % the model found |
| **tF1** | Train F1 — harmonic mean of precision and recall |
| **vLoss** | Validation loss (same formula, on held-out tasks) |
| **vP** | Val Precision — cross-task generalization of precision |
| **vR** | Val Recall — cross-task generalization of recall |
| **vF1** | Val F1 — **the key metric** for cross-task generalization |

A "positive" state is one that lies on an optimal path to a correct program.
Validation is on 143 **held-out tasks** the model never saw during training.

## Configuration

- **Model**: ValueNetwork, 279,946 parameters
- **Data**: 17.8M train records (663 tasks), 3.8M val records (143 tasks)
- **Positive rate**: ~0.11% (heavy class imbalance)
- **Loss**: Weighted BCE (pos_weight=50) + 0.5 * CE on remaining steps
- **budget_left removed**: prevents oracle D* leak

---

## Adam (lr=3e-4, batch=2048, cosine LR schedule)

| Ep | Loss | tP | tR | tF1 | vLoss | vP | vR | vF1 |
|----|------|------|------|------|-------|------|------|------|
| 1 | 0.2048 | 0.118 | 0.618 | 0.198 | 0.0601 | 0.084 | 0.870 | 0.154 |
| 2 | 0.0566 | 0.196 | 0.927 | 0.324 | 0.0391 | 0.188 | 0.804 | 0.305 |
| 3 | 0.0390 | 0.253 | 0.955 | 0.400 | 0.0570 | 0.181 | 0.824 | 0.297 |
| 4 | 0.0279 | 0.311 | 0.974 | 0.471 | 0.0564 | 0.186 | 0.825 | 0.303 |
| 5 | 0.0207 | 0.372 | 0.982 | 0.540 | 0.0557 | 0.349 | 0.678 | 0.460 |
| 6 | 0.0152 | 0.439 | 0.989 | 0.608 | 0.0703 | 0.284 | 0.719 | 0.407 |
| 7 | 0.0093 | 0.501 | 0.993 | 0.665 | 0.0781 | 0.322 | 0.723 | 0.445 |
| 8 | 0.0050 | 0.575 | 0.996 | 0.729 | 0.0752 | 0.399 | 0.656 | 0.496 |
| 9 | 0.0034 | 0.644 | 0.999 | 0.783 | 0.0893 | 0.416 | 0.644 | 0.505 |
| **10** | **0.0022** | **0.692** | **1.000** | **0.818** | **0.0887** | **0.450** | **0.621** | **0.522** |

**Best**: Epoch 10, Val F1 = **0.522** (P=45.0%, R=62.1%)

---

## Muon (lr=3e-4, batch=2048, cosine LR, Muon for 2D + Adam for 1D)

| Ep | Loss | tP | tR | tF1 | vLoss | vP | vR | vF1 |
|----|------|------|------|------|-------|------|------|------|
| 1 | 0.4354 | 0.139 | 0.195 | 0.162 | 0.0591 | 0.154 | 0.671 | 0.250 |
| **2** | **0.0975** | **0.176** | **0.865** | **0.292** | **0.0353** | **0.252** | **0.779** | **0.381** |
| 3 | 0.0535 | 0.285 | 0.938 | 0.438 | 0.0388 | 0.171 | 0.845 | 0.285 |
| 4 | 0.0397 | 0.349 | 0.964 | 0.512 | 0.0333 | 0.239 | 0.772 | 0.365 |
| 5 | 0.0335 | 0.384 | 0.975 | 0.551 | 0.0368 | 0.175 | 0.830 | 0.289 |
| 6 | 0.0291 | 0.410 | 0.981 | 0.579 | 0.0388 | 0.176 | 0.825 | 0.290 |
| 7 | 0.0275 | 0.430 | 0.985 | 0.599 | 0.0412 | 0.175 | 0.810 | 0.287 |
| 8 | 0.0250 | 0.443 | 0.987 | 0.612 | 0.0410 | 0.171 | 0.818 | 0.282 |
| 9 | 0.0239 | 0.453 | 0.989 | 0.622 | 0.0427 | 0.167 | 0.811 | 0.277 |
| 10 | 0.0230 | 0.456 | 0.990 | 0.624 | 0.0413 | 0.183 | 0.801 | 0.297 |

**Best**: Epoch 2, Val F1 = **0.381** (P=25.2%, R=77.9%)

---

## Comparison

| | Adam | Muon |
|---|---|---|
| **Best Val F1** | **0.522** | 0.381 |
| **Best Epoch** | 10 | 2 |
| **Peak Val Precision** | 45.0% | 25.2% |
| **Peak Val Recall** | 87.0% (ep1) | 84.5% (ep3) |
| **Overfitting** | Moderate (train 0.818 vs val 0.522) | Severe (train 0.624 vs val 0.297) |
| **Late-epoch gains** | Strong (cosine LR drives 0.30→0.52) | None (val stuck at ~0.29 after ep2) |

**Verdict**: Adam wins decisively. Muon's gradient orthogonalization doesn't help on small MLPs (280K params, weight matrices 180-521 wide). Adam's per-parameter adaptive rates are more valuable at this scale. Muon is designed for large transformer weight matrices where the orthogonalization constraint improves conditioning — our matrices are too small to benefit.
