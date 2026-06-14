"""Greedy-decode evaluation of a HuggingFace causal LM on darpa3 items.

Kept torch-free at import time; torch and transformers are imported lazily
inside the functions that need them.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import List


# ── Scoring ───────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    """Lowercase, strip punctuation (keep word chars, spaces, hyphens).

    Matches the normalize_verbal() used in arithmetic-inconsistencies/src/data.py.
    """
    import re
    s = s.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    return " ".join(s.split())


def score_prediction(prediction: str, target: str) -> bool:
    """Normalised prefix match.

    1. Exact match after stripping whitespace.
    2. Normalised exact match (lowercase + strip punctuation).
    3. Normalised prefix match — prediction starts with the normalised target
       (handles models that emit trailing tokens after a correct answer).

    Normalization matches arithmetic-inconsistencies/attribution/src/data.py
    `normalize_verbal()` so that, e.g., "treinta y dos." matches "treinta y dos".
    """
    pred = prediction.strip()
    tgt = target.strip()
    if pred == tgt:
        return True
    pred_n = _normalize(pred)
    tgt_n = _normalize(tgt)
    if pred_n == tgt_n:
        return True
    if pred_n.startswith(tgt_n):
        rest = pred_n[len(tgt_n):]
        if not rest or not rest[0].isalnum():
            return True
    return False


# ── Model evaluation ──────────────────────────────────────────────────────────

def evaluate(
    model,
    tokenizer,
    items: List[dict],
    max_new_tokens: int = 32,
    batch_size: int = 16,
    device: str = "cuda",
) -> List[dict]:
    """
    Run greedy decode on a list of items and return predictions.

    Each returned dict:
        {id, x, y, prediction, correct, features}
    """
    import torch

    model.eval()
    predictions = []

    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        prompts = [item["x"] for item in batch]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        prompt_len = enc["input_ids"].shape[1]
        new_tokens = out[:, prompt_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for item, pred_text in zip(batch, decoded):
            correct = score_prediction(pred_text, item["y"])
            predictions.append({
                "id": item["id"],
                "x": item["x"],
                "y": item["y"],
                "prediction": pred_text.strip(),
                "correct": correct,
                "features": item.get("features", {}),
            })

    return predictions


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(predictions: List[dict]) -> dict:
    """
    Compute accuracy overall and broken down by each feature dimension.

    Returns:
        {
            "n": int,
            "accuracy": float,
            "by_feature": {
                "<feature_name>": {"<value>": {"n": int, "accuracy": float}, ...},
                ...
            }
        }
    """
    if not predictions:
        return {"n": 0, "accuracy": 0.0, "by_feature": {}}

    n = len(predictions)
    n_correct = sum(1 for p in predictions if p["correct"])
    accuracy = n_correct / n

    # Per-feature breakdown
    by_feature: dict = defaultdict(lambda: defaultdict(lambda: {"n": 0, "correct": 0}))
    for pred in predictions:
        for key, val in pred.get("features", {}).items():
            if isinstance(val, (str, int, float, bool)):
                by_feature[key][str(val)]["n"] += 1
                if pred["correct"]:
                    by_feature[key][str(val)]["correct"] += 1

    # Convert to accuracy
    by_feature_out: dict = {}
    for feat_name, values in by_feature.items():
        by_feature_out[feat_name] = {
            v: {"n": d["n"], "accuracy": d["correct"] / d["n"]}
            for v, d in values.items()
        }

    return {
        "n": n,
        "accuracy": accuracy,
        "by_feature": by_feature_out,
    }


# ── I/O helpers ───────────────────────────────────────────────────────────────

def write_predictions(predictions: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")


def write_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def read_predictions(path: Path) -> List[dict]:
    preds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))
    return preds
