"""
model.py — Value network for PCA Phase 2.

Architecture:
- Per-test encoder: shared MLP over [inputs, targets, regs, flags] → 256
- Pool: concat(mean, max) across tests → 512
- Global features: live(8) + depth(1) → 9
- Trunk: 521 → 256 → 128
- Heads: reachability logit, remaining-steps classifier
"""

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    def __init__(self, per_test_dim=180, num_tests=10, max_depth=8):
        super().__init__()
        self.num_tests = num_tests
        self.max_depth = max_depth

        # Shared per-test encoder
        self.test_encoder = nn.Sequential(
            nn.Linear(per_test_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Trunk: pooled(512) + global(9) → 256 → 128
        # Global: live(8) + depth(1) = 9 (budget_left removed — it leaks oracle D*)
        self.trunk = nn.Sequential(
            nn.Linear(512 + 9, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Reachability head: binary logit
        self.reach_head = nn.Linear(128, 1)

        # Remaining steps head: classification over 0..max_depth
        self.dist_head = nn.Linear(128, max_depth + 1)

    def forward(self, per_test, live, depth):
        """
        Args:
            per_test: (B, T, 180) per-test features
            live: (B, 8) live register mask bits
            depth: (B,) current depth
        Returns:
            reach_logit: (B, 1)
            dist_logits: (B, max_depth+1)
            trunk_emb: (B, 128) for analysis
        """
        B, T, D = per_test.shape

        # Encode each test independently with shared weights
        flat = per_test.reshape(B * T, D)
        encoded = self.test_encoder(flat)  # (B*T, 256)
        encoded = encoded.reshape(B, T, 256)

        # Pool across tests: concat(mean, max)
        mean_pool = encoded.mean(dim=1)   # (B, 256)
        max_pool = encoded.max(dim=1)[0]  # (B, 256)
        pooled = torch.cat([mean_pool, max_pool], dim=1)  # (B, 512)

        # Global features (no budget — would leak oracle D*)
        global_feat = torch.cat([
            live,
            depth.unsqueeze(1),
        ], dim=1)  # (B, 9)

        # Trunk
        trunk_in = torch.cat([pooled, global_feat], dim=1)  # (B, 521)
        trunk_emb = self.trunk(trunk_in)  # (B, 128)

        # Heads
        reach_logit = self.reach_head(trunk_emb)  # (B, 1)
        dist_logits = self.dist_head(trunk_emb)   # (B, max_depth+1)

        return reach_logit, dist_logits, trunk_emb
