"""EnsemblePredictor: combines circuit_overlap and linear_probe
via a logistic regression meta-learner.

Workflow
--------
1. fit(val_preds, model, tokenizer):
   a. Run both sub-predictors on val.
   b. Rank-normalize each sub-predictor's val scores to [0, 1] using a
      QuantileTransformer fitted on val (so test scores are mapped through
      the same val CDF at predict time).
   c. Fit a logistic regression meta-learner on the [N_val, 2] scaled matrix.

2. predict(test_preds, model, tokenizer):
   a. Get test scores from both sub-predictors.
   b. Apply the saved QuantileTransformer.
   c. Apply meta_lr_.predict_proba()[:, 1].
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

from predictors.base import BasePredictor, compute_auc, fit_isotonic
from predictors.circuit_overlap.predictor import CircuitOverlapPredictor
from predictors.linear_probe.predictor import LinearProbePredictor


class EnsemblePredictor(BasePredictor):
    """Instance-level accuracy predictor that ensembles circuit overlap and
    linear probe via a logistic regression meta-learner.

    Scores from each sub-predictor are rank-normalized to [0, 1] using a
    QuantileTransformer fitted on the val set before being passed to the
    meta-learner, so scale differences between predictors don't affect the LR.

    Parameters
    ----------
    k_fraction : float
        Top-K fraction for the circuit overlap sub-predictor.
    batch_size : int
        Items per forward pass (shared by all sub-predictors).
    device : str
        PyTorch device string.
    """

    def __init__(
        self,
        k_fraction: float = 0.01,
        batch_size: int = 4,
        device: str = "cuda",
    ):
        self.k_fraction = k_fraction
        self.batch_size = batch_size
        self.device = device

        self.co_predictor_ = CircuitOverlapPredictor(
            k_fraction=k_fraction, batch_size=batch_size, device=device,
        )
        self.lp_predictor_ = LinearProbePredictor(
            batch_size=batch_size, device=device,
        )

        self.scaler_     = None   # sklearn QuantileTransformer (fitted on val)
        self.meta_lr_    = None   # sklearn LogisticRegression
        self.calibrator_ = None   # IsotonicRegression

    # ── internals ────────────────────────────────────────────────────────────

    def _raw_feature_matrix(
        self,
        co_scores: List[float],
        lp_scores: List[float],
    ) -> np.ndarray:
        X = np.column_stack([
            np.asarray(co_scores, dtype=float),
            np.asarray(lp_scores, dtype=float),
        ])
        return np.clip(X, -1e9, 1e9)

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, val_preds: List[dict], model=None, tokenizer=None) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import QuantileTransformer

        if model is None or tokenizer is None:
            raise ValueError("model and tokenizer are required for fit()")

        print("  [ensemble] Fitting circuit_overlap sub-predictor …")
        self.co_predictor_.fit(val_preds, model=model, tokenizer=tokenizer)

        print("  [ensemble] Fitting linear_probe sub-predictor …")
        self.lp_predictor_.fit(val_preds, model=model, tokenizer=tokenizer)

        print("  [ensemble] Collecting val scores for meta-learner …")
        val_co = self.co_predictor_.predict(val_preds, model=model, tokenizer=tokenizer)
        val_lp = self.lp_predictor_.predict(val_preds, model=model, tokenizer=tokenizer)

        val_labels = np.array([int(p["correct"]) for p in val_preds])
        if len(np.unique(val_labels)) < 2:
            raise RuntimeError("Val set has only one class — cannot fit meta-learner.")

        X_val_raw = self._raw_feature_matrix(val_co, val_lp)

        # Rank-normalize each feature to [0, 1] using val distribution
        n_quantiles = min(len(val_preds), 1000)
        self.scaler_ = QuantileTransformer(
            n_quantiles=n_quantiles, output_distribution="uniform", random_state=0
        )
        X_val = self.scaler_.fit_transform(X_val_raw)

        self.meta_lr_ = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        self.meta_lr_.fit(X_val, val_labels)

        val_scores = self.meta_lr_.predict_proba(X_val)[:, 1].tolist()
        self.calibrator_ = fit_isotonic(val_scores, val_labels.tolist())

        val_auc = compute_auc(val_scores, val_labels.tolist())
        coef = self.meta_lr_.coef_[0]
        print(
            f"  [ensemble] Val AUC: {val_auc:.3f}  "
            f"meta-LR coef: CO={coef[0]:.3f}, LP={coef[1]:.3f}"
        )

    # ── predict ──────────────────────────────────────────────────────────────

    def predict(self, test_preds: List[dict], model=None, tokenizer=None) -> List[float]:
        if self.meta_lr_ is None:
            raise RuntimeError("Call fit() before predict()")
        if model is None or tokenizer is None:
            raise ValueError("model and tokenizer are required for predict()")

        print(f"  [ensemble] Getting test scores from sub-predictors (n={len(test_preds)}) …")
        test_co = self.co_predictor_.predict(test_preds, model=model, tokenizer=tokenizer)
        test_lp = self.lp_predictor_.predict(test_preds, model=model, tokenizer=tokenizer)

        X_test_raw = self._raw_feature_matrix(test_co, test_lp)
        X_test = self.scaler_.transform(X_test_raw)
        return self.meta_lr_.predict_proba(X_test)[:, 1].tolist()

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.co_predictor_.save(directory / "circuit_overlap")
        self.lp_predictor_.save(directory / "linear_probe")

        with open(directory / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler_, f)
        with open(directory / "meta_lr.pkl", "wb") as f:
            pickle.dump(self.meta_lr_, f)
        with open(directory / "calibrator.pkl", "wb") as f:
            pickle.dump(self.calibrator_, f)

        meta = {"k_fraction": self.k_fraction}
        with open(directory / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: Path) -> "EnsemblePredictor":
        directory = Path(directory)
        with open(directory / "meta.json") as f:
            meta = json.load(f)

        predictor = cls(**meta)
        predictor.co_predictor_ = CircuitOverlapPredictor.load(directory / "circuit_overlap")
        predictor.lp_predictor_ = LinearProbePredictor.load(directory / "linear_probe")

        with open(directory / "scaler.pkl", "rb") as f:
            predictor.scaler_ = pickle.load(f)
        with open(directory / "meta_lr.pkl", "rb") as f:
            predictor.meta_lr_ = pickle.load(f)
        with open(directory / "calibrator.pkl", "rb") as f:
            predictor.calibrator_ = pickle.load(f)

        return predictor
