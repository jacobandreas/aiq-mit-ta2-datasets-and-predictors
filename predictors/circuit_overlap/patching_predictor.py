"""PatchingCircuitOverlapPredictor: circuit overlap as in arithmetic-inconsistencies.

fit(train_items):  run counterfactual attribution patching on the train items
    and define the reference circuit S as the top ``k_fraction`` units of the
    *unweighted* mean attribution, by absolute value (the reference builds S
    from all kept items of the numeric format, without looking at labels).
predict(test_items):  run attribution patching on the test items and score
    each kept item by its loading on S, ``sum_on_S`` (the signed sum of its
    attribution at the units in S).  Items left without any contrast (see
    the fallback in patching.py) have no attribution and get NaN.

Items are dicts with ``x``, ``x_prime`` and ``k`` (see
predictors/circuit_overlap/patching.py).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from predictors.circuit_overlap.overlap import topk_indices_abs
from predictors.circuit_overlap.patching import run_patching


class PatchingCircuitOverlapPredictor:

    def __init__(self, k_fraction: float = 0.01, batch_size: int = 4, device: str = "cuda",
                 position: str = "last", fallback_max_draws: int = 20):
        self.k_fraction = k_fraction
        self.position = position
        self.fallback_max_draws = fallback_max_draws
        self.batch_size = batch_size
        self.device = device

        # Set after fit()
        self.reference_: Optional[np.ndarray] = None   # [n_layers, intermediate_size]
        self.ref_topk_: Optional[np.ndarray] = None    # flat indices of S
        self.fit_info_: Optional[dict] = None
        # Set after predict()
        self.predict_info_: Optional[dict] = None

    def fit(self, train_items: List[dict], model, tokenizer) -> None:
        total = {"sum": None, "n": 0}

        def consume(_idx, attr):
            total["sum"] = attr.astype(np.float64) if total["sum"] is None else total["sum"] + attr
            total["n"] += 1

        print(f"  [patching] train items: {len(train_items)}", flush=True)
        self.fit_info_ = run_patching(model, tokenizer, train_items, consume,
                                      batch_size=self.batch_size, device=self.device,
                                      position=self.position,
                                      fallback_max_draws=self.fallback_max_draws)
        if total["n"] == 0:
            raise ValueError("no train items survived the patching filters; cannot build S")
        self.reference_ = (total["sum"] / total["n"]).astype(np.float32)
        k = max(1, int(round(self.k_fraction * self.reference_.size)))
        self.ref_topk_ = topk_indices_abs(self.reference_, k)
        print(f"  [patching] reference circuit: {total['n']} items, K={k} units", flush=True)

    def predict(self, test_items: List[dict], model, tokenizer) -> np.ndarray:
        if self.ref_topk_ is None:
            raise RuntimeError("call fit() before predict()")
        scores = np.full(len(test_items), np.nan)

        def consume(idx, attr):
            scores[idx] = float(attr.ravel()[self.ref_topk_].sum())

        print(f"  [patching] test items: {len(test_items)}", flush=True)
        self.predict_info_ = run_patching(model, tokenizer, test_items, consume,
                                          batch_size=self.batch_size, device=self.device,
                                          position=self.position,
                                          fallback_max_draws=self.fallback_max_draws)
        return scores
