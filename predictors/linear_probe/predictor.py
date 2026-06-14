"""LinearProbePredictor: predicts per-item accuracy via logistic regression on hidden states.

Workflow
--------
1. fit(val_preds, model, tokenizer):
   a. Forward pass over val items (no gradients) with output_hidden_states=True.
   b. Extract the hidden state at the last non-padding prompt token from every
      layer.  Shape: [N_val, n_layers, hidden_dim].
   c. For each layer, fit a logistic regression (probe) on (hidden_state →
      correct/incorrect).
   d. Select the layer with the highest val AUC as the "best layer".

2. predict(test_preds, model, tokenizer):
   a. Extract hidden states for test items using the same procedure.
   b. Apply the probe for the best layer.
   c. Return predict_proba scores (P(correct)).

Design notes
------------
* Only the prompt (item["x"]) is fed to the model — we want the representation
  at the decision boundary (last prompt token), before generation begins.
* One logistic regression per layer is cheap (2000 × 3072 ≈ 6M floats) and lets
  us identify which layer is most predictive without concatenating all layers
  (which can be underdetermined at typical val set sizes).
* L-BFGS solver with C=0.1 regularisation works well in practice for this kind
  of linear readout.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

from predictors.base import BasePredictor, compute_auc, fit_isotonic


class LinearProbePredictor(BasePredictor):
    """Instance-level accuracy predictor using a logistic-regression probe on
    last-token hidden states.

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
        self.best_layer_: Optional[int] = None
        self.probe_ = None          # sklearn LogisticRegression
        self.calibrator_ = None     # IsotonicRegression
        self.val_aucs_: Optional[np.ndarray] = None  # per-layer AUCs

    # ── internals ────────────────────────────────────────────────────────────

    def _extract_hidden_states(
        self, model, tokenizer, items: List[dict]
    ) -> np.ndarray:
        """Return last-token hidden states for all layers.

        Returns
        -------
        np.ndarray of shape [N, n_layers, hidden_dim]
            Layer 0 is the embedding; layers 1..n_layers are transformer blocks.
        """
        import torch

        model.eval()
        all_hiddens = []

        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            prompts = [b["x"] for b in batch]

            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            with torch.no_grad():
                try:
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                except torch.cuda.OutOfMemoryError:
                    # Retry one item at a time if the batch doesn't fit
                    torch.cuda.empty_cache()
                    single_hs = []
                    for j in range(input_ids.shape[0]):
                        out_j = model(
                            input_ids=input_ids[j:j+1],
                            attention_mask=attention_mask[j:j+1],
                            output_hidden_states=True,
                        )
                        hs_j = torch.stack(out_j.hidden_states, dim=1)
                        sl = attention_mask[j:j+1].sum(dim=1) - 1
                        single_hs.append(hs_j[0, :, sl[0], :].float().cpu())
                        del out_j
                        torch.cuda.empty_cache()
                    all_hiddens.append(torch.stack(single_hs, dim=0).numpy())
                    continue

            # hidden_states: tuple of (n_layers+1) tensors, each [B, T, D]
            # Stack → [n_layers+1, B, T, D], then take last non-padding position
            hs = torch.stack(out.hidden_states, dim=1)  # [B, n_layers+1, T, D]

            # Index of the last real (non-padding) token per item
            seq_lens = attention_mask.sum(dim=1) - 1  # [B], 0-indexed last position
            B = hs.shape[0]
            last_token = hs[
                torch.arange(B, device=self.device),
                :,
                seq_lens,
                :,
            ]  # [B, n_layers+1, D]

            all_hiddens.append(last_token.float().cpu().numpy())

        return np.concatenate(all_hiddens, axis=0)  # [N, n_layers+1, D]

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, val_preds: List[dict], model=None, tokenizer=None) -> None:
        from sklearn.linear_model import LogisticRegression

        if model is None or tokenizer is None:
            raise ValueError("model and tokenizer are required for fit()")

        print(f"  [linear_probe] Extracting val hidden states (n={len(val_preds)}) …")
        val_hs = self._extract_hidden_states(model, tokenizer, val_preds)
        # val_hs: [N, n_layers+1, D]

        val_labels = np.array([int(p["correct"]) for p in val_preds])

        if len(np.unique(val_labels)) < 2:
            raise RuntimeError("Val set has only one class — cannot fit probe.")

        n_layers = val_hs.shape[1]
        aucs = np.zeros(n_layers)
        probes = []

        for layer_idx in range(n_layers):
            X = val_hs[:, layer_idx, :]
            probe = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
            probe.fit(X, val_labels)
            scores = probe.predict_proba(X)[:, 1]
            aucs[layer_idx] = compute_auc(scores.tolist(), val_labels.tolist())
            probes.append(probe)

        self.val_aucs_ = aucs
        self.best_layer_ = int(np.argmax(aucs))
        self.probe_ = probes[self.best_layer_]

        best_auc = aucs[self.best_layer_]
        val_scores = self.probe_.predict_proba(val_hs[:, self.best_layer_, :])[:, 1]
        self.calibrator_ = fit_isotonic(val_scores.tolist(), val_labels.tolist())

        print(
            f"  [linear_probe] Best layer: {self.best_layer_}/{n_layers - 1}, "
            f"val AUC: {best_auc:.3f}"
        )

    # ── predict ──────────────────────────────────────────────────────────────

    def predict(self, test_preds: List[dict], model=None, tokenizer=None) -> List[float]:
        if self.probe_ is None:
            raise RuntimeError("Call fit() before predict()")
        if model is None or tokenizer is None:
            raise ValueError("model and tokenizer are required for predict()")

        print(f"  [linear_probe] Extracting test hidden states (n={len(test_preds)}) …")
        test_hs = self._extract_hidden_states(model, tokenizer, test_preds)
        X = test_hs[:, self.best_layer_, :]
        scores = self.probe_.predict_proba(X)[:, 1]
        return scores.tolist()

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "probe.pkl", "wb") as f:
            pickle.dump(self.probe_, f)
        with open(directory / "calibrator.pkl", "wb") as f:
            pickle.dump(self.calibrator_, f)
        meta = {
            "best_layer": self.best_layer_,
            "val_aucs": self.val_aucs_.tolist() if self.val_aucs_ is not None else None,
        }
        with open(directory / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: Path) -> "LinearProbePredictor":
        directory = Path(directory)
        predictor = cls()
        with open(directory / "probe.pkl", "rb") as f:
            predictor.probe_ = pickle.load(f)
        with open(directory / "calibrator.pkl", "rb") as f:
            predictor.calibrator_ = pickle.load(f)
        with open(directory / "meta.json") as f:
            meta = json.load(f)
        predictor.best_layer_ = meta["best_layer"]
        if meta.get("val_aucs") is not None:
            predictor.val_aucs_ = np.array(meta["val_aucs"])
        return predictor
