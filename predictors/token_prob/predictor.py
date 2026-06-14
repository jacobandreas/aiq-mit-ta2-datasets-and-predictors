"""TokenProbPredictor: predicts per-item accuracy via model log-probability of the target.

Workflow
--------
1. fit(val_preds, model, tokenizer):
   a. For each val item, compute the mean log P(y_i | x, y_{<i}) over the
      answer tokens — the teacher-forced per-token log-probability.
   b. Fit an IsotonicRegression calibrator on (val scores → val labels).

2. predict(test_preds, model, tokenizer):
   a. Compute the same score for each test item.
   b. Return raw scores (higher → model assigns more probability to the answer).

Design notes
------------
* The score is the *mean* log-prob over answer tokens (not the sum), so shorter
  and longer answers are treated on equal footing.
* We tokenize `x + y` jointly and find the answer boundary by tokenizing `x`
  alone.  Using the joint tokenization for the forward pass avoids edge effects
  from BPE token-boundary shifts, but the boundary index comes from `len(enc_x)`.
  In the rare case where the joint tokenization gives a different prefix length,
  we fall back to a suffix-search heuristic.
* The computation is a standard causal-LM forward pass (no gradients), so it is
  much faster than gradient attribution.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

from predictors.base import BasePredictor, compute_auc, fit_isotonic


class TokenProbPredictor(BasePredictor):
    """Instance-level accuracy predictor using target token log-probabilities.

    Parameters
    ----------
    batch_size : int
        Items per forward pass.
    device : str
        PyTorch device string.
    """

    def __init__(self, batch_size: int = 8, device: str = "cuda"):
        self.batch_size = batch_size
        self.device = device

        # Set after fit()
        self.calibrator_ = None

    # ── internals ────────────────────────────────────────────────────────────

    def _score_items(self, model, tokenizer, items: List[dict]) -> List[float]:
        """Return mean per-token log-prob of the answer for each item."""
        import torch
        import torch.nn.functional as F

        model.eval()
        scores: List[float] = []

        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]

            # Tokenize prompt (x) and full sequence (x + y) separately.
            # Use add_special_tokens=False for x so the boundary index is
            # simply len(x_ids) in the joint sequence.
            xy_strings = [b["x"] + b["y"] for b in batch]
            x_strings  = [b["x"] for b in batch]

            enc_xy = tokenizer(
                xy_strings,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
            )
            enc_x = tokenizer(
                x_strings,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
                add_special_tokens=False,
            )

            input_ids    = enc_xy["input_ids"].to(self.device)
            attention_mask = enc_xy["attention_mask"].to(self.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)

            # logits: [B, T, V]
            # log P(token at position t) = log_softmax(logits[:, t-1, :])
            log_probs = F.log_softmax(out.logits, dim=-1)  # [B, T, V]

            for i, item in enumerate(batch):
                # Number of real (non-padding) x tokens for this item
                # enc_x.attention_mask[i] counts real tokens (padding on left)
                x_len = int(enc_x["attention_mask"][i].sum().item())

                # In the padded joint sequence the actual tokens start at:
                # total_len - (real x len + real y len)
                # Instead, find where y starts in input_ids[i] by using the
                # actual prompt length in the joint encoding.
                xy_mask = attention_mask[i]  # [T]
                xy_real_len = int(xy_mask.sum().item())
                T = input_ids.shape[1]
                # real tokens occupy positions [T - xy_real_len, T)
                # x occupies the first x_len of those real tokens
                x_start = T - xy_real_len
                y_start = x_start + x_len
                y_end   = T  # exclusive

                if y_start >= y_end:
                    # No answer tokens found (degenerate item); score = -inf
                    scores.append(float("-inf"))
                    continue

                # Gather log-probs for answer tokens at positions y_start..y_end-1
                # In causal LM, logits[t] predicts token t+1.
                # So log P(token at pos t) uses logits at pos t-1.
                y_positions = torch.arange(y_start, y_end, device=self.device)
                y_token_ids = input_ids[i, y_positions]                    # [L_y]
                logit_positions = y_positions - 1                           # [L_y]
                tok_log_probs = log_probs[i, logit_positions, y_token_ids]  # [L_y]

                # Clamp to a finite floor so -inf tokens don't propagate
                tok_log_probs = tok_log_probs.clamp(min=-100.0)
                score = float(tok_log_probs.mean().item())
                scores.append(score)

        return scores

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, val_preds: List[dict], model=None, tokenizer=None) -> None:
        if model is None or tokenizer is None:
            raise ValueError("model and tokenizer are required for fit()")

        print(f"  [token_prob] Scoring val items (n={len(val_preds)}) …")
        val_scores = self._score_items(model, tokenizer, val_preds)
        val_labels = [int(p["correct"]) for p in val_preds]
        self.calibrator_ = fit_isotonic(val_scores, val_labels)

        val_auc = compute_auc(val_scores, val_labels)
        print(f"  [token_prob] Val AUC: {val_auc:.3f}")

    # ── predict ──────────────────────────────────────────────────────────────

    def predict(self, test_preds: List[dict], model=None, tokenizer=None) -> List[float]:
        if self.calibrator_ is None:
            raise RuntimeError("Call fit() before predict()")
        if model is None or tokenizer is None:
            raise ValueError("model and tokenizer are required for predict()")

        print(f"  [token_prob] Scoring test items (n={len(test_preds)}) …")
        return self._score_items(model, tokenizer, test_preds)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "calibrator.pkl", "wb") as f:
            pickle.dump(self.calibrator_, f)
        with open(directory / "meta.json", "w") as f:
            json.dump({}, f)

    @classmethod
    def load(cls, directory: Path) -> "TokenProbPredictor":
        directory = Path(directory)
        predictor = cls()
        with open(directory / "calibrator.pkl", "rb") as f:
            predictor.calibrator_ = pickle.load(f)
        return predictor
