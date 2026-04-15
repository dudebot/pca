#!/usr/bin/env python3
"""
train.py — Train the PCA Phase 2 value network.

Usage:
  python train.py --epochs 10 --batch-size 2048 --lr 3e-4 --optimizer muon
  python train.py --epochs 10 --batch-size 2048 --lr 3e-4 --optimizer adam

Uses manifest-driven task splits (train/val by task, not random state split).
Supports TensorBoard logging and Muon optimizer.
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import PCADataset
from model import ValueNetwork


def train_epoch(model, loader, optimizer, device, max_depth, pos_weight=None):
    model.train()
    total_loss = 0
    total_reach_loss = 0
    total_dist_loss = 0
    n_batches = 0
    tp = fp = tn = fn = 0

    bce_kwargs = {}
    if pos_weight is not None:
        bce_kwargs['pos_weight'] = torch.tensor([pos_weight], device=device)

    for batch in loader:
        per_test = batch['per_test'].to(device)
        live = batch['live'].to(device)
        depth = batch['depth'].to(device)
        can_finish = batch['can_finish'].to(device)
        remaining = batch['remaining'].to(device)

        reach_logit, dist_logits, _ = model(per_test, live, depth)

        # Reachability loss (BCE with optional pos_weight for class imbalance)
        loss_reach = nn.functional.binary_cross_entropy_with_logits(
            reach_logit.squeeze(1), can_finish, **bce_kwargs)

        # Remaining steps loss (CE, masked to positives)
        positive_mask = can_finish > 0.5
        if positive_mask.any():
            remaining_clamped = remaining[positive_mask].clamp(0, max_depth)
            loss_dist = nn.functional.cross_entropy(
                dist_logits[positive_mask], remaining_clamped)
        else:
            loss_dist = torch.tensor(0.0, device=device)

        loss = loss_reach + 0.5 * loss_dist

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_reach_loss += loss_reach.item()
        total_dist_loss += loss_dist.item()
        n_batches += 1

        pred = (reach_logit.squeeze(1) > 0).float()
        tp += ((pred == 1) & (can_finish == 1)).sum().item()
        fp += ((pred == 1) & (can_finish == 0)).sum().item()
        tn += ((pred == 0) & (can_finish == 0)).sum().item()
        fn += ((pred == 0) & (can_finish == 1)).sum().item()

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    return {
        'loss': total_loss / max(1, n_batches),
        'reach_loss': total_reach_loss / max(1, n_batches),
        'dist_loss': total_dist_loss / max(1, n_batches),
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


@torch.no_grad()
def eval_epoch(model, loader, device, max_depth):
    model.eval()
    total_loss = 0
    n_batches = 0
    tp = fp = tn = fn = 0

    for batch in loader:
        per_test = batch['per_test'].to(device)
        live = batch['live'].to(device)
        depth = batch['depth'].to(device)
        can_finish = batch['can_finish'].to(device)
        remaining = batch['remaining'].to(device)

        reach_logit, dist_logits, _ = model(per_test, live, depth)

        loss_reach = nn.functional.binary_cross_entropy_with_logits(
            reach_logit.squeeze(1), can_finish)

        positive_mask = can_finish > 0.5
        if positive_mask.any():
            remaining_clamped = remaining[positive_mask].clamp(0, max_depth)
            loss_dist = nn.functional.cross_entropy(
                dist_logits[positive_mask], remaining_clamped)
        else:
            loss_dist = torch.tensor(0.0, device=device)

        loss = loss_reach + 0.5 * loss_dist
        total_loss += loss.item()
        n_batches += 1

        pred = (reach_logit.squeeze(1) > 0).float()
        tp += ((pred == 1) & (can_finish == 1)).sum().item()
        fp += ((pred == 1) & (can_finish == 0)).sum().item()
        tn += ((pred == 0) & (can_finish == 0)).sum().item()
        fn += ((pred == 0) & (can_finish == 1)).sum().item()

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    return {
        'loss': total_loss / max(1, n_batches),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', default=None,
                        help='Project root (auto-detected if not set)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max-depth', type=int, default=8)
    parser.add_argument('--num-tests', type=int, default=10)
    parser.add_argument('--pos-weight', type=float, default=None,
                        help='BCE pos_weight for class imbalance (auto if not set)')
    parser.add_argument('--save', default='checkpoints/value_v1.pt')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Limit number of task files per split (for debugging)')
    parser.add_argument('--workers', type=int, default=4,
                        help='DataLoader num_workers for parallel data loading')
    parser.add_argument('--run-name', default=None,
                        help='TensorBoard run name (default: auto from optimizer+lr)')
    args = parser.parse_args()

    if args.base_dir is None:
        args.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # TensorBoard
    if args.run_name is None:
        args.run_name = f"adam_lr{args.lr}_pw{args.pos_weight}_bs{args.batch_size}"
    log_dir = os.path.join(args.base_dir, 'runs', args.run_name)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard: {log_dir}")

    # Load datasets (manifest-driven task splits)
    print(f"\nLoading train split...")
    train_set = PCADataset(args.base_dir, split='train',
                           num_tests=args.num_tests, max_files=args.max_files)

    print(f"\nLoading val split...")
    val_set = PCADataset(args.base_dir, split='val',
                         num_tests=args.num_tests, max_files=args.max_files)

    if len(train_set) == 0:
        print("No training data. Run gen_dataset.sh and build_manifest.py first.")
        return

    pin = device.type == 'cuda'
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers,
                              pin_memory=pin, drop_last=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers,
                            pin_memory=pin,
                            persistent_workers=args.workers > 0)

    # Auto-compute pos_weight if not specified
    if args.pos_weight is None:
        args.pos_weight = 50.0
        print(f"\nUsing default pos_weight: {args.pos_weight}")

    # Model
    model = ValueNetwork(max_depth=args.max_depth,
                         num_tests=args.num_tests).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} parameters")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Training loop
    best_val_f1 = 0
    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"{'Ep':>3s}  {'Loss':>7s}  {'tP':>5s}  {'tR':>5s}  {'tF1':>5s}  "
          f"{'vLoss':>7s}  {'vP':>5s}  {'vR':>5s}  {'vF1':>5s}  {'Time':>5s}")
    print("-" * 72)

    for epoch in range(args.epochs):
        t0 = time.time()
        train_m = train_epoch(model, train_loader, optimizer, device,
                              args.max_depth, pos_weight=args.pos_weight)
        val_m = eval_epoch(model, val_loader, device, args.max_depth)
        scheduler.step()
        dt = time.time() - t0

        # Console output
        print(f"{epoch+1:3d}  {train_m['loss']:7.4f}  "
              f"{train_m['precision']:5.3f}  {train_m['recall']:5.3f}  "
              f"{train_m['f1']:5.3f}  "
              f"{val_m['loss']:7.4f}  {val_m['precision']:5.3f}  "
              f"{val_m['recall']:5.3f}  {val_m['f1']:5.3f}  "
              f"{dt:5.1f}s")

        # TensorBoard logging
        writer.add_scalars('loss', {
            'train': train_m['loss'],
            'val': val_m['loss'],
        }, epoch + 1)
        writer.add_scalars('f1', {
            'train': train_m['f1'],
            'val': val_m['f1'],
        }, epoch + 1)
        writer.add_scalars('precision', {
            'train': train_m['precision'],
            'val': val_m['precision'],
        }, epoch + 1)
        writer.add_scalars('recall', {
            'train': train_m['recall'],
            'val': val_m['recall'],
        }, epoch + 1)
        writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch + 1)

        if val_m['f1'] > best_val_f1:
            best_val_f1 = val_m['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_m,
                'args': vars(args),
            }, args.save)
            print(f"  -> saved (best val F1={best_val_f1:.3f})")

    writer.close()
    print(f"\nBest val F1: {best_val_f1:.3f}")
    print(f"Model saved to {args.save}")


if __name__ == '__main__':
    main()
