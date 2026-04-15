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
    def __init__(self, per_test_dim=180, num_tests=10, max_depth=8,
                 hidden_dim=256, trunk_dim=128):
        super().__init__()
        self.num_tests = num_tests
        self.max_depth = max_depth

        # Shared per-test encoder
        self.test_encoder = nn.Sequential(
            nn.Linear(per_test_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Trunk: pooled(2*hidden_dim) + global(9) → trunk_dim*2 → trunk_dim
        pool_dim = hidden_dim * 2  # mean + max pooling
        self.trunk = nn.Sequential(
            nn.Linear(pool_dim + 9, trunk_dim * 2),
            nn.ReLU(),
            nn.Linear(trunk_dim * 2, trunk_dim),
            nn.ReLU(),
        )

        # Reachability head: binary logit
        self.reach_head = nn.Linear(trunk_dim, 1)

        # Remaining steps head: classification over 0..max_depth
        self.dist_head = nn.Linear(trunk_dim, max_depth + 1)

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
        encoded = self.test_encoder(flat)
        H = encoded.shape[1]  # hidden_dim
        encoded = encoded.reshape(B, T, H)

        # Pool across tests: concat(mean, max)
        mean_pool = encoded.mean(dim=1)
        max_pool = encoded.max(dim=1)[0]
        pooled = torch.cat([mean_pool, max_pool], dim=1)

        # Global features (no budget — would leak oracle D*)
        global_feat = torch.cat([
            live,
            depth.unsqueeze(1),
        ], dim=1)

        # Trunk
        trunk_in = torch.cat([pooled, global_feat], dim=1)
        trunk_emb = self.trunk(trunk_in)

        # Heads
        reach_logit = self.reach_head(trunk_emb)
        dist_logits = self.dist_head(trunk_emb)

        return reach_logit, dist_logits, trunk_emb
