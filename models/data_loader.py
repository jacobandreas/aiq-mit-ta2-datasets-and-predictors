"""Load dataset splits from a data directory.

Each split directory contains train.jsonl, (val.jsonl), and test.jsonl.
The causal_inference split has no val.jsonl; 10% of train is held out as val.

Items have the darpa3 standard format:
    {"id": ..., "x": ..., "y": ..., "split": ..., "features": {...}}
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional


def _read_jsonl(path: Path) -> list:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_splits(
    dataset_dir: Path,
    seed: int = 0,
    val_fraction: float = 0.1,
) -> Dict[str, List[dict]]:
    """
    Load train/val/test splits from a dataset directory.

    If val.jsonl is absent, hold out `val_fraction` of train items as val
    (deterministic based on `seed`).

    Returns dict with keys 'train', 'val', 'test'.
    """
    dataset_dir = Path(dataset_dir)
    splits: Dict[str, List[dict]] = {}

    for fname in ("train.jsonl", "test.jsonl"):
        path = dataset_dir / fname
        if path.exists():
            splits[fname.replace(".jsonl", "")] = _read_jsonl(path)

    val_path = dataset_dir / "val.jsonl"
    if val_path.exists():
        splits["val"] = _read_jsonl(val_path)
    elif "train" in splits:
        # Hold out val_fraction of train deterministically
        rng = random.Random(seed)
        items = list(splits["train"])
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_fraction))
        splits["val"] = items[:n_val]
        splits["train"] = items[n_val:]

    return splits


def discover_split_dirs(data_root: Path) -> list:
    """
    Walk data_root and return all leaf directories that contain test.jsonl.
    This covers both training splits (which also have train.jsonl) and
    eval-only splits (which may have only val.jsonl + test.jsonl).

    data_root/
      arithmetic_parametric/arithmetic_by_format/test.jsonl  ← leaf
      arithmetic_fixed/cross_lingual/test.jsonl              ← leaf (no train.jsonl)
    """
    data_root = Path(data_root)
    return sorted(set(path.parent for path in data_root.rglob("test.jsonl")))
