"""CircuitOverlapPredictor: predicts per-item accuracy via MLP circuit overlap.

Workflow
--------
1. fit(val_preds, model, tokenizer):
   a. Compute gradient × activation attribution for every val item.
   b. Build the reference circuit = weighted-mean attribution over val items,
      weighted by correctness (1 for correct, 0 for wrong).  This concentrates
      the reference on the neurons that fire when the model gets things right.
   c. Compute sum_on_S overlap scores for every val item.
   d. Fit an IsotonicRegression calibrator on (val scores → val labels).

2. predict(test_preds, model, tokenizer):
   a. Compute attribution for every test item.
   b. Compute sum_on_S overlap with the fitted reference circuit.
   c. Return raw scores (higher = model more likely to be correct).

3. save / load  — persist reference circuit + calibrator to disk.

The `sum_on_S` score is returned as the primary confidence signal because it
was the strongest predictor in the original arithmetic-inconsistencies work
(rpb ≈ 0.38 vs. rpb ≈ 0.04 for cosine and ≈ −0.09 for Jaccard).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

from predictors.base import BasePredictor
from predictors.circuit_overlap.attribution import compute_gradient_attribution
from predictors.circuit_overlap.overlap import batch_overlap_scores, build_reference


class CircuitOverlapPredictor(BasePredictor):
    """Instance-level accuracy predictor based on MLP circuit overlap.

    Parameters
    ----------
    k_fraction : float
        Top-K fraction of neurons used to define the reference circuit "S".
        Default 0.01 (top 1%) matches the original paper's main result.
    batch_size : int
        Items per attribution forward pass.
    device : str
        PyTorch device string.
    weight_by_correct : bool
        If True, build the reference as a weighted mean (correct items get
        weight 1, incorrect items get weight 0).  This focuses the reference
        on the neurons involved in correct computations.
    """

    def __init__(
        self,
        k_fraction: float = 0.01,
        batch_size: int = 4,
        device: str = "cuda",
        weight_by_correct: bool = True,
    ):
        self.k_fraction = k_fraction
        self.batch_size = batch_size
        self.device = device
        self.weight_by_correct = weight_by_correct

        # Set after fit()
        self.reference_: Optional[np.ndarray] = None   # [n_layers, hidden]
        self.calibrator_ = None                         # IsotonicRegression

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, val_preds: List[dict], model=None, tokenizer=None) -> None:
        """
        Fit on labelled validation predictions.

        Args:
            val_preds: List of prediction dicts (must include 'correct' field).
            model:     HuggingFace CausalLM (needed to compute attributions).
            tokenizer: Matching tokenizer.
        """
        if model is None or tokenizer is None:
            raise ValueError("model and tokenizer are required for fit()")

        print(f"  [circuit_overlap] Computing val attributions (n={len(val_preds)}) …")
        val_attr = compute_gradient_attribution(
            model, tokenizer, val_preds,
            batch_size=self.batch_size,
            device=self.device,
        )  # [N_val, n_layers, hidden]

        # Build reference circuit
        if self.weight_by_correct:
            weights = np.array([float(p["correct"]) for p in val_preds])
        else:
            weights = None
        self.reference_ = build_reference(val_attr, weights=weights)

        # Compute val scores
        val_scores_dicts = batch_overlap_scores(val_attr, self.reference_, self.k_fraction)
        val_scores = [d["sum_on_S"] for d in val_scores_dicts]
        val_labels = [int(p["correct"]) for p in val_preds]

        # Fit calibrator
        from predictors.base import fit_isotonic
        self.calibrator_ = fit_isotonic(val_scores, val_labels)

        val_auc = _quick_auc(val_scores, val_labels)
        print(f"  [circuit_overlap] Val AUC (sum_on_S): {val_auc:.3f}")

    # ── predict ──────────────────────────────────────────────────────────────

    def predict(self, test_preds: List[dict], model=None, tokenizer=None) -> List[float]:
        """
        Return one confidence score per test prediction.

        Returns raw sum_on_S scores (not calibrated probabilities).
        Call predict_proba() for calibrated probabilities.
        """
        if self.reference_ is None:
            raise RuntimeError("Call fit() before predict()")
        if model is None or tokenizer is None:
            raise ValueError("model and tokenizer are required for predict()")

        print(f"  [circuit_overlap] Computing test attributions (n={len(test_preds)}) …")
        test_attr = compute_gradient_attribution(
            model, tokenizer, test_preds,
            batch_size=self.batch_size,
            device=self.device,
        )
        scores_dicts = batch_overlap_scores(test_attr, self.reference_, self.k_fraction)
        return [d["sum_on_S"] for d in scores_dicts]

    def predict_proba(self, scores: List[float]) -> List[float]:
        """Calibrate raw scores to probabilities using the fitted IsotonicRegression."""
        if self.calibrator_ is None:
            raise RuntimeError("Call fit() before predict_proba()")
        probs = self.calibrator_.predict(np.asarray(scores, dtype=float))
        return probs.tolist()

    def predict_all_metrics(
        self, test_preds: List[dict], model=None, tokenizer=None
    ) -> List[dict]:
        """Return all three overlap metrics per item (sum_on_S, cosine, jaccard)."""
        if self.reference_ is None:
            raise RuntimeError("Call fit() before predict_all_metrics()")
        test_attr = compute_gradient_attribution(
            model, tokenizer, test_preds,
            batch_size=self.batch_size,
            device=self.device,
        )
        return batch_overlap_scores(test_attr, self.reference_, self.k_fraction)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, directory: Path) -> None:
        """Save reference circuit and calibrator to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "reference_circuit.npy", self.reference_)
        with open(directory / "calibrator.pkl", "wb") as f:
            pickle.dump(self.calibrator_, f)
        meta = {"k_fraction": self.k_fraction, "weight_by_correct": self.weight_by_correct}
        with open(directory / "meta.json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, directory: Path) -> "CircuitOverlapPredictor":
        directory = Path(directory)
        with open(directory / "meta.json") as f:
            meta = json.load(f)
        predictor = cls(**meta)
        predictor.reference_ = np.load(directory / "reference_circuit.npy")
        with open(directory / "calibrator.pkl", "rb") as f:
            predictor.calibrator_ = pickle.load(f)
        return predictor


# ── Utility ───────────────────────────────────────────────────────────────────

def _quick_auc(scores, labels) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        labels = [int(l) for l in labels]
        if len(set(labels)) < 2:
            return float("nan")
        return float(roc_auc_score(labels, scores))
    except ImportError:
        return float("nan")
