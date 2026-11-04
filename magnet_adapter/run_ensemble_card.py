#!/usr/bin/env python3
"""KWDagger node: circuit overlap + token prob rank ensemble (fixed or cross-domain weight).

Reads the per-item ``scores.csv`` files that ``run_predictor_card.py`` writes
(one per split, next to its ``results.json``) and combines the two
predictors as

    combined = w · rank(circuit overlap) + (1 − w) · rank(token prob)

where ranks are within-test-set percentiles (circuit overlap over the items it
scored; unscored items take the token prob percentile alone).  With
``--combine_contrasts own``, only items scored with their own counterfactual
use circuit overlap; fallback-contrast items take token prob alone.  The weight w
is fixed (``--weight``, default 0.5) or, with ``--weight tuned``, chosen on
the *other* domain's splits: arithmetic
targets use the EWOK splits and EWOK targets the arithmetic splits.  w is the
grid value (0, 0.05, …, 1) maximising the mean combined AUC over those splits,
so nothing about the target split's labels enters the ensemble.

Source runs are found under ``--scores_root``: every ``results.json`` with
``method == "patching"``, the given model, ``k_source``, and
``fallback_max_draws`` (which only runs with the redrawn fallback have), the
newest per split.

Usage:
    python magnet_adapter/run_ensemble_card.py \\
        --scores_root ./evaluation_runs_kwdagger \\
        --split_config ewok_physical_to_social --model huggingface/Qwen3-4B \\
        --results_fpath results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DOMAINS = {
    "arithmetic": ["cross_lingual", "arithmetic_by_format", "arithmetic_by_skill"],
    "ewok": ["ewok_physical_to_social", "ewok_social_to_physical"],
}
GRID = np.round(np.linspace(0.0, 1.0, 21), 2)


def _domain(split: str) -> str:
    for name, splits in DOMAINS.items():
        if split in splits:
            return name
    raise ValueError(f"split {split!r} is in no domain: {DOMAINS}")


def _auc(score, y) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(y, dtype=float), np.asarray(score, dtype=float)))


def _rank(x):
    from scipy.stats import rankdata
    return (rankdata(np.clip(x, -1e9, 1e9)) - 1) / max(len(x) - 1, 1)


def combine(df, w: float, contrasts: str = "all") -> np.ndarray:
    """``contrasts="own"`` uses circuit overlap only on items scored with their
    own counterfactual; fallback-contrast items take the token prob rank."""
    kept = df["circuit_overlap_kept"].to_numpy(dtype=bool)
    if contrasts == "own":
        kept = kept & (df["circuit_overlap_contrast"] == "own").to_numpy()
    tp_rank = _rank(df["token_prob"].to_numpy(dtype=float))
    out = tp_rank.copy()
    co_rank = _rank(df["circuit_overlap"].to_numpy(dtype=float)[kept])
    out[kept] = w * co_rank + (1 - w) * tp_rank[kept]
    return out


def find_sources(scores_root: Path, model: str, k_source: str, fallback_max_draws: int) -> dict:
    """{split: scores.csv path} for the newest matching patching run per split."""
    best = {}
    for rpath in scores_root.rglob("results.json"):
        try:
            r = json.loads(rpath.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (r.get("method") != "patching" or r.get("model") != model
                or r.get("k_source") != k_source
                or r.get("fallback_max_draws") != fallback_max_draws):
            continue
        spath = rpath.with_name("scores.csv")
        if not spath.exists():
            continue
        mtime = spath.stat().st_mtime
        if r["split_config"] not in best or mtime > best[r["split_config"]][0]:
            best[r["split_config"]] = (mtime, spath)
    return {split: p for split, (_, p) in best.items()}


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores_root", required=True, type=Path,
                        help="Directory searched for the patching card's results.json/scores.csv")
    parser.add_argument("--split_config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--k_source", default="completion")
    parser.add_argument("--fallback_max_draws", type=int, default=20)
    parser.add_argument("--combine_contrasts", choices=("all", "own"), default="all",
                        help="items whose circuit overlap score enters the combination: "
                             "all scored items, or only those with their own "
                             "counterfactual (default: all)")
    parser.add_argument("--weight", default="0.5",
                        help="circuit overlap weight w: a number in [0, 1], or 'tuned' to "
                             "choose it on the other domain's splits (default: 0.5)")
    parser.add_argument("--results_fpath", required=True)
    args = parser.parse_args()

    sources = find_sources(args.scores_root, args.model, args.k_source, args.fallback_max_draws)
    target_domain = _domain(args.split_config)
    tuned = args.weight == "tuned"
    tune_splits = [s for name, splits in DOMAINS.items() if name != target_domain
                   for s in splits] if tuned else []
    missing = [s for s in [args.split_config] + tune_splits if s not in sources]
    if missing:
        print(f"ERROR: no patching scores for {missing} under {args.scores_root} "
              f"(model={args.model!r}, k_source={args.k_source!r}, "
              f"fallback_max_draws={args.fallback_max_draws})", file=sys.stderr)
        sys.exit(1)
    for s in [args.split_config] + tune_splits:
        print(f"  {s}: {sources[s]}", flush=True)

    if tuned:
        tune = {s: pd.read_csv(sources[s]) for s in tune_splits}
        tune_auc = {float(w): float(np.mean([_auc(combine(df, w, args.combine_contrasts),
                                                  df["actual_mean"])
                                             for df in tune.values()])) for w in GRID}
        w_best = max(tune_auc, key=tune_auc.get)  # ties: the first (smaller) weight
        print("  tuning curve (mean other-domain combined AUC by w): "
              + ", ".join(f"{w:.2f}:{a:.3f}" for w, a in tune_auc.items()), flush=True)
    else:
        w_best = float(args.weight)
        if not 0.0 <= w_best <= 1.0:
            raise ValueError(f"--weight must be in [0, 1] or 'tuned', not {args.weight!r}")
        tune_auc = {w_best: None}

    df = pd.read_csv(sources[args.split_config])
    y = df["actual_mean"]
    co_auc = _auc(df["circuit_overlap"], y)
    tp_auc = _auc(df["token_prob"], y)
    combined_auc = _auc(combine(df, w_best, args.combine_contrasts), y)
    results = {
        "model": args.model,
        "split_config": args.split_config,
        "method": "patching_ensemble",
        "weight": w_best,
        "combine_contrasts": args.combine_contrasts,
        "tuned_on": tune_splits,
        "tuning_auc": tune_auc[w_best],
        "auc": co_auc,
        "token_prob_auc": tp_auc,
        "combined_auc": combined_auc,
        "combined_gain": combined_auc - tp_auc,
        "combined_equal_weight_auc": _auc(combine(df, 0.5, args.combine_contrasts), y),
        # For reference only: the weight that would be best on the target itself.
        "oracle_weight": float(max(GRID, key=lambda w: _auc(combine(df, w, args.combine_contrasts), y))),
        "n_test": len(df),
        "n_fallback": int((df["circuit_overlap_contrast"] == "fallback").sum()),
        "n_kept": int(df["circuit_overlap_kept"].sum()),
    }
    out_path = Path(args.results_fpath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
