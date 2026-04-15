"""
dataset.py — Load PCA Phase 2 binary dataset records into PyTorch tensors.

Split-aware: reads data/manifest.json for train/val/test task splits.
Pre-extracts fields from memmapped binary into flat numpy arrays for fast
random access. Bit encoding is vectorized (no Python loops).
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset

MAX_TESTS = 32
NUM_REGS = 8
RECORD_SIZE = 576

# Field offsets within a record
OFF_TASK_ID = 16
OFF_DEPTH = 22
OFF_LIVE = 24
OFF_CAN_FINISH = 25
OFF_REMAINING = 26
OFF_REGS = 30
OFF_FLAGS = 542

# Pre-computed bit indices for vectorized encoding
_BITS16 = np.arange(16, dtype=np.uint16)
_BITS8 = np.arange(8, dtype=np.uint8)
_BITS4 = np.arange(4, dtype=np.uint8)


def load_task_io(task_json_path):
    """Load task I/O from JSON for the per-test encoder."""
    with open(task_json_path) as f:
        spec = json.load(f)

    tests = spec['tests']
    num_inputs = len(spec['input_ports'])
    num_tests = len(tests)

    inputs = np.zeros((num_tests, 2), dtype=np.uint16)
    outputs = np.zeros((num_tests, 1), dtype=np.uint16)
    for t, tc in enumerate(tests):
        for i, v in enumerate(tc['in']):
            if i < 2:
                inputs[t, i] = v
        outputs[t, 0] = tc['out'][0]

    return {
        'name': spec['name'],
        'num_inputs': num_inputs,
        'num_tests': num_tests,
        'inputs': inputs,
        'outputs': outputs,
    }


def bits_encode_array(arr, nbits=16):
    """Vectorized bit encoding using broadcasting. Shape: (*arr.shape, nbits)."""
    bits = _BITS16[:nbits]
    return ((arr[..., np.newaxis] >> bits) & 1).astype(np.float32)


def load_manifest(base_dir):
    """Load data/manifest.json."""
    path = os.path.join(base_dir, 'data', 'manifest.json')
    with open(path) as f:
        return json.load(f)


class PCADataset(Dataset):
    """
    Fast PyTorch dataset for PCA Phase 2 value network training.

    Pre-extracts all fields from binary records into flat numpy arrays
    during __init__. __getitem__ is pure array indexing + vectorized
    bit encoding (no Python loops, no per-record memmap access).
    """

    def __init__(self, base_dir, split='train', num_tests=10, max_files=None):
        self.num_tests = num_tests
        self.base_dir = base_dir

        manifest = load_manifest(base_dir)
        split_tasks = [t for t in manifest['tasks'] if t['split'] == split]
        if max_files:
            split_tasks = split_tasks[:max_files]

        # First pass: count total records
        task_files = []
        total = 0
        for t in split_tasks:
            task_id = t['task_id']
            n_records = t['n_records']
            if n_records == 0 or t.get('states_path') is None:
                continue
            states_path = os.path.join(base_dir, t['states_path'])
            if not os.path.exists(states_path):
                continue
            task_files.append((states_path, n_records, task_id, t['path']))
            total += n_records

        # Pre-allocate flat arrays
        nt = num_tests
        self.task_ids = np.zeros(total, dtype=np.uint32)
        self.depths = np.zeros(total, dtype=np.uint8)
        self.lives = np.zeros(total, dtype=np.uint8)
        self.can_finish_arr = np.zeros(total, dtype=np.float32)
        self.remaining_arr = np.zeros(total, dtype=np.int64)
        self.regs = np.zeros((total, nt, NUM_REGS), dtype=np.uint16)
        self.flags_raw = np.zeros((total, nt), dtype=np.uint8)

        # Extract fields from memmaps in bulk
        idx = 0
        task_io_map = {}
        for states_path, n_rec, task_id, task_rel_path in task_files:
            mmap = np.memmap(states_path, dtype=np.uint8, mode='r',
                             shape=(n_rec, RECORD_SIZE))

            end = idx + n_rec

            # Scalar fields — direct byte indexing across all records
            self.task_ids[idx:end] = task_id
            self.depths[idx:end] = mmap[:, OFF_DEPTH]
            self.lives[idx:end] = mmap[:, OFF_LIVE]
            self.can_finish_arr[idx:end] = mmap[:, OFF_CAN_FINISH].astype(np.float32)
            remaining_raw = mmap[:, OFF_REMAINING].view(np.int8)
            self.remaining_arr[idx:end] = np.maximum(remaining_raw, 0).astype(np.int64)

            # Regs: extract first num_tests × NUM_REGS uint16 values
            regs_bytes = nt * NUM_REGS * 2
            regs_raw = mmap[:, OFF_REGS:OFF_REGS + regs_bytes].copy()
            self.regs[idx:end] = regs_raw.view(np.uint16).reshape(n_rec, nt, NUM_REGS)

            # Flags: first num_tests bytes
            self.flags_raw[idx:end] = mmap[:, OFF_FLAGS:OFF_FLAGS + nt]

            idx += n_rec
            del mmap

            # Load task I/O
            if task_id not in task_io_map:
                task_path = os.path.join(base_dir, task_rel_path)
                task_io_map[task_id] = load_task_io(task_path)

        self.n = total

        # Precompute bit-encoded task I/O (constant per task, shared across records)
        self._task_io_cache = {}
        for task_id, tio in task_io_map.items():
            actual = min(tio['num_tests'], nt)
            inp_bits = bits_encode_array(tio['inputs'][:actual])
            out_bits = bits_encode_array(tio['outputs'][:actual])
            test_inputs = np.zeros((nt, 32), dtype=np.float32)
            test_targets = np.zeros((nt, 16), dtype=np.float32)
            test_inputs[:actual] = inp_bits.reshape(actual, -1)
            test_targets[:actual] = out_bits.reshape(actual, -1)
            self._task_io_cache[task_id] = np.concatenate(
                [test_inputs, test_targets], axis=1)  # (nt, 48)

        print(f"[{split}] {self.n:,} records from {len(task_files)} tasks")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        nt = self.num_tests
        task_id = self.task_ids[idx]

        # Regs bit encoding — vectorized, no loops
        regs = self.regs[idx]  # (nt, 8) uint16
        regs_bits = ((regs[:, :, np.newaxis] >> _BITS16) & 1).astype(np.float32)
        # regs_bits shape: (nt, 8, 16) → flatten to (nt, 128)
        regs_flat = regs_bits.reshape(nt, -1)

        # Flags bit encoding — vectorized
        flags = self.flags_raw[idx]  # (nt,) uint8
        flags_bits = ((flags[:, np.newaxis] >> _BITS4) & 1).astype(np.float32)
        # flags_bits shape: (nt, 4)

        # Per-test features: io_cache(48) + regs(128) + flags(4) = 180
        io_feats = self._task_io_cache.get(int(task_id))
        if io_feats is None:
            io_feats = np.zeros((nt, 48), dtype=np.float32)

        per_test = np.concatenate([io_feats, regs_flat, flags_bits], axis=1)

        # Live bits — vectorized
        live_bits = ((self.lives[idx] >> _BITS8) & 1).astype(np.float32)

        return {
            'per_test': torch.from_numpy(per_test),
            'live': torch.from_numpy(live_bits),
            'depth': torch.tensor(np.float32(self.depths[idx])),
            'can_finish': torch.tensor(self.can_finish_arr[idx]),
            'remaining': torch.tensor(self.remaining_arr[idx]),
        }
