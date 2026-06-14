"""Circuit overlap metrics.

Ported from arithmetic-inconsistencies/attribution/scripts/run_overlap.py.

Given a reference circuit (aggregate attribution vector over reference items)
and a per-item attribution vector, computes three overlap scores:

  sum_on_S       -- sum of item's attributions at the reference's top-K positions
  cosine_with_S  -- cosine similarity between item vector and reference vector
  jaccard_with_S -- Jaccard overlap between item's top-K and reference's top-K

'sum_on_S' was the strongest predictor in the original work (rpb ≈ 0.38).
"""
from __future__ import annotations

import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def topk_indices_abs(vec: np.ndarray, k: int) -> np.ndarray:
    """Flat indices of the top-k entries by absolute value."""
    flat = np.abs(vec).ravel()
    if k >= flat.size:
        return np.arange(flat.size)
    return np.argpartition(flat, -k)[-k:]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity; upcasts to float32 to avoid fp16 underflow."""
    a = np.asarray(a).ravel().astype(np.float32, copy=False)
    b = np.asarray(b).ravel().astype(np.float32, copy=False)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def jaccard_sets(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard index between two sets of flat indices."""
    sa, sb = set(a.tolist()), set(b.tolist())
    if not sa and not sb:
        return float("nan")
    return len(sa & sb) / len(sa | sb)


# ── Reference circuit construction ───────────────────────────────────────────

def build_reference(
    attr: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build a reference circuit from a batch of attribution vectors.

    Args:
        attr:    [N, n_layers, intermediate_size]  float32
        weights: [N] optional per-item weights (e.g. 1 for correct, 0 for wrong).
                 If None, uses uniform mean.

    Returns:
        reference: [n_layers, intermediate_size]  float32
    """
    if weights is not None:
        w = np.asarray(weights, dtype=np.float32)
        total = w.sum()
        if total == 0:
            return attr.mean(axis=0).astype(np.float32)
        return (attr * w[:, None, None]).sum(axis=0) / total
    return attr.mean(axis=0).astype(np.float32)


# ── Per-item overlap scores ───────────────────────────────────────────────────

def overlap_scores(
    item_attr: np.ndarray,       # [n_layers, intermediate_size]
    reference: np.ndarray,       # [n_layers, intermediate_size]
    k_fraction: float = 0.01,
) -> dict:
    """
    Compute the three overlap metrics for one item.

    Returns dict with keys: sum_on_S, cosine_with_S, jaccard_with_S, K.
    """
    ref_flat = reference.ravel().astype(np.float32)
    item_flat = item_attr.ravel().astype(np.float32)

    K = max(1, int(round(k_fraction * ref_flat.size)))
    ref_topk = topk_indices_abs(ref_flat, K)

    # sum_on_S: signed sum of item attributions at reference top-K positions
    sum_S = float(item_flat[ref_topk].sum())

    # cosine_with_S: cosine between item vector and full reference vector
    cos_S = cosine(item_flat, ref_flat)

    # jaccard_with_S: overlap between item's top-K and reference's top-K
    item_topk = topk_indices_abs(item_flat, K)
    jac_S = jaccard_sets(item_topk, ref_topk)

    return {
        "sum_on_S": sum_S,
        "cosine_with_S": cos_S,
        "jaccard_with_S": jac_S,
        "K": K,
    }


def batch_overlap_scores(
    attr: np.ndarray,            # [N, n_layers, intermediate_size]
    reference: np.ndarray,       # [n_layers, intermediate_size]
    k_fraction: float = 0.01,
) -> list:
    """Compute overlap_scores for each item; return list of dicts."""
    ref_flat = reference.ravel().astype(np.float32)
    K = max(1, int(round(k_fraction * ref_flat.size)))
    ref_topk = topk_indices_abs(ref_flat, K)

    results = []
    for i in range(len(attr)):
        item_flat = attr[i].ravel().astype(np.float32)
        sum_S = float(item_flat[ref_topk].sum())
        cos_S = cosine(item_flat, ref_flat)
        item_topk = topk_indices_abs(item_flat, K)
        jac_S = jaccard_sets(item_topk, ref_topk)
        results.append({
            "sum_on_S": sum_S,
            "cosine_with_S": cos_S,
            "jaccard_with_S": jac_S,
            "K": K,
        })
    return results
