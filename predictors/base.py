"""Base predictor interface and shared analysis utilities.

A Predictor takes validation predictions (with ground-truth labels) and test
predictions (without labels), and returns a scalar confidence score per test
item.  Scores are in (-inf, inf) and should positively correlate with
per-item accuracy.

Analysis pipeline
-----------------
1. Fit on val: isotonic regression to calibrate raw scores → probabilities.
2. Evaluate on test:
   - AUC  (uses raw scores via sklearn.metrics.roc_auc_score)
   - F1 at optimal threshold (sweep on val, apply to test)
   - ECE  (Expected Calibration Error on calibrated probabilities)
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List


class BasePredictor(ABC):
    """Abstract predictor.

    Subclasses override fit() and predict().  Both receive lists of prediction
    dicts — the same format written by models/evaluate.py:
        {"id": ..., "x": ..., "y": ..., "prediction": ...,
         "correct": bool, "features": {...}}
    """

    @abstractmethod
    def fit(self, val_preds: List[dict]) -> None:
        """Fit on labelled validation predictions."""

    @abstractmethod
    def predict(self, test_preds: List[dict]) -> List[float]:
        """Return one score per test prediction (higher → more likely correct)."""


# ── Analysis utilities ────────────────────────────────────────────────────────

def fit_isotonic(scores, labels):
    """Fit isotonic regression on (scores, labels) and return the calibrator.

    scores: array-like of floats
    labels: array-like of 0/1

    Returns an IsotonicRegression fitted object.
    """
    from sklearn.isotonic import IsotonicRegression
    import numpy as np

    scores = np.asarray(scores, dtype=float)
    scores = np.clip(scores, -1e9, 1e9)   # guard against ±inf
    labels = np.asarray(labels, dtype=float)
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(scores, labels)
    return ir


def compute_auc(scores, labels) -> float:
    """ROC AUC."""
    from sklearn.metrics import roc_auc_score
    import numpy as np

    scores = np.asarray(scores, dtype=float)
    scores = np.clip(scores, -1e9, 1e9)   # guard against ±inf from log-prob scores
    labels = np.asarray(labels, dtype=float)
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def compute_f_at_threshold(scores, labels, threshold: float) -> float:
    """F1 score at a given decision threshold."""
    from sklearn.metrics import f1_score
    import numpy as np

    preds = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    labels = np.asarray(labels, dtype=int)
    return float(f1_score(labels, preds, zero_division=0.0))


def find_optimal_threshold(scores, labels, n_thresholds: int = 100) -> tuple:
    """Sweep thresholds on (scores, labels) and return (best_f1, best_threshold)."""
    import numpy as np

    scores = np.asarray(scores, dtype=float)
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    best_f, best_t = 0.0, float(thresholds[0])
    for t in thresholds:
        f = compute_f_at_threshold(scores, labels, t)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_f, best_t


def compute_ece(probs, labels, n_bins: int = 10) -> float:
    """Expected Calibration Error using equal-frequency bins."""
    import numpy as np

    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    n = len(probs)
    if n == 0:
        return float("nan")

    # Sort by predicted probability
    order = np.argsort(probs)
    probs_sorted = probs[order]
    labels_sorted = labels[order]

    ece = 0.0
    bin_size = max(1, n // n_bins)
    for start in range(0, n, bin_size):
        end = min(start + bin_size, n)
        bp = probs_sorted[start:end].mean()
        bl = labels_sorted[start:end].mean()
        ece += (end - start) / n * abs(bp - bl)
    return float(ece)


# ── Full analysis pipeline ────────────────────────────────────────────────────

def analyze(
    val_scores: List[float],
    val_labels: List[int],
    test_scores: List[float],
    test_labels: List[int],
) -> dict:
    """
    Fit on val, evaluate on test.  Returns:
        {
            "val":  {"n", "auc", "f1_optimal", "threshold", "ece"},
            "test": {"n", "auc", "f1_at_val_threshold", "ece"},
        }
    """
    import numpy as np

    # Clip ±inf before any sklearn calls (e.g. token_prob can produce -inf scores)
    val_scores  = np.clip(np.asarray(val_scores,  dtype=float), -1e9, 1e9).tolist()
    test_scores = np.clip(np.asarray(test_scores, dtype=float), -1e9, 1e9).tolist()
    val_labels_int = [int(l) for l in val_labels]
    test_labels_int = [int(l) for l in test_labels]

    # Fit calibrator on val
    calibrator = fit_isotonic(val_scores, val_labels_int)

    # Val metrics
    val_auc = compute_auc(val_scores, val_labels_int)
    val_f1, val_threshold = find_optimal_threshold(val_scores, val_labels_int)
    val_probs = calibrator.predict(np.asarray(val_scores, dtype=float)).tolist()
    val_ece = compute_ece(val_probs, val_labels_int)

    # Test metrics (using threshold found on val)
    test_auc = compute_auc(test_scores, test_labels_int)
    test_probs = calibrator.predict(np.asarray(test_scores, dtype=float)).tolist()
    test_ece = compute_ece(test_probs, test_labels_int)
    test_f1 = compute_f_at_threshold(test_scores, test_labels_int, val_threshold)

    return {
        "val": {
            "n": len(val_scores),
            "auc": val_auc,
            "f1_optimal": val_f1,
            "threshold": val_threshold,
            "ece": val_ece,
        },
        "test": {
            "n": len(test_scores),
            "auc": test_auc,
            "f1_at_val_threshold": test_f1,
            "ece": test_ece,
        },
    }


def write_analysis(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
