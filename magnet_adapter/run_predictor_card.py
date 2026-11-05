#!/usr/bin/env python3
"""KWDagger node for the CircuitOverlap predictor evaluation card.

Loads one (model, split) pair of HELM runs, runs
CircuitOverlapInstancePredictor (attribution-patching circuit overlap: train
run → reference circuit, eval run → per-instance scores), computes AUC of the
scores against ground-truth exact_match, and writes a flat results JSON plus
per-item scores (scores.csv).

For ``--split_config X`` the eval run is ``...:split_config=X,model=...`` and
the train run is ``...:split_config=X_train,model=...`` (X's train items,
relabelled as test so HELM generates labelled completions for them).

Besides circuit overlap (``auc``), it reports the token prob baseline
(``token_prob_auc``) and the equal-weight rank average of the two
(``combined_auc``; circuit overlap enters only for items with their own
counterfactual), all from one model load, so one KWDagger node serves both the
"nontrivial AUC" card (circuit_overlap_patching.yaml) and the "improves the
baseline" card (circuit_overlap_patching_ensemble.yaml).

Output format (read into KWDagger's ``metrics.run_circuit_overlap.*``
namespace by :func:`load_kwdagger_result`):
    {"model": "huggingface/Qwen3-4B", "split_config": "cross_lingual",
     "auc": 0.709, "token_prob_auc": 0.60, "combined_auc": 0.69,
     "combined_gain": 0.09, "n_test": 1016, "test_accuracy": 0.10}

Usage:
    python -u magnet_adapter/run_predictor_card.py \\
        --helm_suite_path ./benchmark_output/runs/suite_2 \\
        --split_config cross_lingual \\
        --model huggingface/Qwen3-4B \\
        --model_path Qwen/Qwen3-4B \\
        --counterfactuals_dpath ./magnet_adapter/counterfactuals \\
        --results_fpath results.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _compute_auc(predicted_mean, actual_mean) -> float:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y_score = np.clip(np.asarray(predicted_mean, dtype=float), -1e9, 1e9)
    y_true  = np.asarray(actual_mean, dtype=float)
    if len(set(y_true.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _auc_or_none(merged, mask):
    import numpy as np
    sub = merged.loc[mask]
    auc = _compute_auc(sub["circuit_overlap"].tolist(), sub["actual_mean"].tolist()) if len(sub) else float("nan")
    return None if np.isnan(auc) else auc


def load_kwdagger_result(node, node_dpath):
    """Load the flat results JSON into KWDagger's result namespace."""
    from kwdagger.utils import util_dotdict

    output_fpath = Path(node_dpath) / node.out_paths[node.primary_out_key]
    metrics = json.loads(output_fpath.read_text())
    flat = util_dotdict.DotDict.from_nested({"metrics": metrics})
    return flat.insert_prefix(node.name, index=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--helm_suite_path",   required=True,
                        help="HELM suite directory, e.g. benchmark_output/runs/suite_2")
    parser.add_argument("--split_config",      required=True,
                        help="Split to evaluate; needs <split> and <split>_train runs")
    parser.add_argument("--model",             required=True,
                        help="HELM model id to evaluate (run_spec.adapter_spec.model)")
    parser.add_argument("--model_path",        default="",
                        help="HF Hub id or local directory for the weights "
                             "(default: the HELM model id)")
    parser.add_argument("--k_fraction",        type=float, default=0.01,
                        help="Top-K fraction for circuit overlap (default: 0.01)")
    parser.add_argument("--batch_size",        type=int,   default=4,
                        help="Items per attribution forward pass (default: 4)")
    parser.add_argument("--counterfactuals_dpath", required=True,
                        help="Directory of make_counterfactuals.py output; "
                             "uses <scenario>.jsonl, e.g. arithmetic_fixed.jsonl")
    parser.add_argument("--reference_id_regex", default="",
                        help="Build the reference circuit from train "
                             "items whose instance id matches (default: all)")
    parser.add_argument("--position",          choices=("last", "last_unaligned"),
                        default="last",
                        help="Read attribution at each prompt's last token, requiring "
                             "the two prompts to match in token length (last, as in the "
                             "reference) or not (last_unaligned) (default: last)")
    parser.add_argument("--fallback_max_draws", type=int, default=20,
                        help="Random length-matched prompts tried per item without a "
                             "usable counterfactual, until the model's answer differs "
                             "(default: 20)")
    parser.add_argument("--dtype",             choices=("bfloat16", "float16", "float32"),
                        default="bfloat16", help="Model weight dtype (default: bfloat16)")
    parser.add_argument("--results_fpath",     required=True,
                        help="Output JSON path")
    args = parser.parse_args()

    import numpy as np
    from magnet.backends.helm.helm_outputs import HelmRuns
    from magnet_adapter.circuit_overlap_predictor import CircuitOverlapInstancePredictor

    # ── Collect this (model, split)'s eval and train runs ─────────────────────
    split = re.escape(args.split_config)
    run_name_re = re.compile(rf":split_config={split}(_train)?,")
    eval_run_selector = rf":split_config={split},"
    candidate_paths = sorted(p for p in Path(args.helm_suite_path).iterdir()
                             if p.is_dir() and run_name_re.search(p.name))
    run_spec_df = HelmRuns(candidate_paths).run_spec() if candidate_paths else None
    model_col = "run_spec.adapter_spec.model"
    model_run_names = set() if run_spec_df is None else set(
        run_spec_df.loc[run_spec_df[model_col] == args.model, "run_spec.name"]
    )
    model_paths = [p for p in candidate_paths if p.name in model_run_names]
    if len(model_paths) != 2:
        print(f"ERROR: expected an eval run and a _train run for "
              f"{args.model!r} / {args.split_config!r} under {args.helm_suite_path}; "
              f"found {[p.name for p in model_paths]}", file=sys.stderr)
        sys.exit(1)

    print(f"[{args.model} / {args.split_config}] evaluating on:", flush=True)
    for p in model_paths:
        print(f"  {p.name}", flush=True)

    scenario = model_paths[0].name.split(":", 1)[0]
    counterfactuals_fpath = str(Path(args.counterfactuals_dpath) / f"{scenario}.jsonl")

    # ── Run predictor ─────────────────────────────────────────────────────────
    predictor = CircuitOverlapInstancePredictor(
        counterfactuals_fpath=counterfactuals_fpath,
        model_path=args.model_path,
        k_fraction=args.k_fraction,
        batch_size=args.batch_size,
        reference_id_regex=args.reference_id_regex,
        position=args.position,
        fallback_max_draws=args.fallback_max_draws,
        dtype=args.dtype,
        eval_run_selector=eval_run_selector,
    )
    comparison_df = predictor(helm_runs=HelmRuns(model_paths))

    # comparison_df holds circuit overlap scores; join the baseline and combined
    # scores on prediction_id so all three AUCs use the same labels.
    merged = comparison_df.merge(predictor.component_scores_, on="prediction_id",
                                 validate="one_to_one")
    actual = merged["actual_mean"].tolist()
    aucs = {name: _compute_auc(merged[name].tolist(), actual)
            for name in ("circuit_overlap", "token_prob", "combined")}
    kept = merged["circuit_overlap_kept"].to_numpy(dtype=bool)
    auc_kept = _compute_auc(merged.loc[kept, "circuit_overlap"].tolist(),
                            merged.loc[kept, "actual_mean"].tolist())
    token_prob_auc_kept = _compute_auc(merged.loc[kept, "token_prob"].tolist(),
                                       merged.loc[kept, "actual_mean"].tolist())
    combined_auc_kept = _compute_auc(merged.loc[kept, "combined"].tolist(),
                                     merged.loc[kept, "actual_mean"].tolist())
    print(f"[{args.model} / {args.split_config}] circuit_overlap AUC on the "
          f"{int(kept.sum())} scored items = {auc_kept:.4f}", flush=True)
    for name, value in aucs.items():
        print(f"[{args.model} / {args.split_config}] {name} AUC = {value:.4f}", flush=True)
    # MAGNET drops missing values from a result row and then fails to resolve
    # the claim's symbols, so an undefined AUC (single-class labels) must not
    # produce a results file; with none, the cell is reported as missing.
    if any(np.isnan(value) for value in aucs.values()):
        print(f"ERROR: AUC undefined for {args.model!r} / {args.split_config!r}: "
              f"test accuracy is {sum(actual) / len(actual):.4f} over {len(actual)} items",
              file=sys.stderr)
        sys.exit(1)

    # ── Write results ─────────────────────────────────────────────────────────
    results = {
        "model": args.model,
        "split_config": args.split_config,
        "auc": aucs["circuit_overlap"],
        # AUC over the items circuit overlap actually scored.  May be NaN.
        "auc_kept": None if np.isnan(auc_kept) else auc_kept,
        "n_kept": int(kept.sum()),
        "position": args.position,
        "fallback_max_draws": args.fallback_max_draws,
        "n_fallback": int((merged["circuit_overlap_contrast"] == "fallback").sum()),
        # Kept-item AUC split by contrast type; None where undefined.
        "auc_own": _auc_or_none(merged, merged["circuit_overlap_contrast"] == "own"),
        "auc_fallback": _auc_or_none(merged, merged["circuit_overlap_contrast"] == "fallback"),
        "token_prob_auc_kept": None if np.isnan(token_prob_auc_kept) else token_prob_auc_kept,
        "combined_auc_kept": None if np.isnan(combined_auc_kept) else combined_auc_kept,
        "token_prob_auc": aucs["token_prob"],
        "combined_auc": aucs["combined"],
        "combined_gain": aucs["combined"] - aucs["token_prob"],
        "n_test": len(merged),
        "test_accuracy": float(sum(actual) / len(actual)),
    }
    out_path = Path(args.results_fpath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Per-item scores, so ensembling can be revisited without recomputing.
    cols = ["prediction_id", "actual_mean", "circuit_overlap", "circuit_overlap_kept",
            "circuit_overlap_contrast", "token_prob", "combined"]
    merged[cols].to_csv(out_path.with_name("scores.csv"), index=False)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}", flush=True)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
