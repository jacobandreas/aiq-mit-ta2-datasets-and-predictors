#!/usr/bin/env python3
"""Runner for the CircuitOverlap predictor evaluation card.

Loads HELM benchmark outputs, runs CircuitOverlapInstancePredictor for each
model matching the run_spec_pattern, computes AUC of per-instance predictions
against ground truth, and writes a results JSON file.

The "result" wrapper is required by MAGNET's GenericPipelineProcessor, which
lifts each key inside "result" as a card symbol accessible in the claim block.

Output format:
    {
        "result": {
            "auc_by_model":  {"Qwen3-4B": 0.687},
            "summary":       {"n_models": 1, "mean_auc": 0.687, "min_auc": 0.687}
        }
    }

Usage:
    python -u magnet_adapter/run_predictor_card.py \\
        --helm_runs_path ./benchmark_output \\
        --run_spec_pattern 'arithmetic_fixed*model=Qwen*' \\
        --models_root /raid/lingo/models \\
        --results_fpath results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Imported after the sys.path insert, for the same reason everything else in
# this file is: the runner is executed as a script, not as a package member.
import magnet.theory as theory  # noqa: E402

# ---------------------------------------------------------------------------
# Theory bindings: theory/indexes/hygiene.yaml.
#
# The circuit-overlap card has no bespoke theorem behind its "AUC >= 0.65",
# and saying so precisely is worth more than leaving it unsaid: a threshold
# card still leans on the same premises every threshold card leans on.
#
# Every predicate is a no-op at run time -- decorators return their target
# unchanged -- and MAGNET reads them out of this file with `ast` rather than
# importing it, so auditing the card never needs torch or a GPU.
#
# The unaccounted premise this exists to surface is `hn_sufficient`: nothing
# anywhere computes whether the test split resolves an AUC gap of 0.15 above
# chance, and `max_test_items` in the card matrix can shrink it further with
# no effect on the reported number.
# ---------------------------------------------------------------------------


@theory.assumes(
    'Hygiene.Inference.threshold_exceeds_sampling_error::hn_sufficient',
    note='the load-bearing gap. This function returns a point AUC '
         'and nothing else. No confidence interval, no permutation null, no '
         'count of positives and negatives -- and the sampling error of an AUC '
         'is governed by the smaller of those two counts, not by the item '
         'total. The card asserts 0.65, i.e. 0.15 above chance; whether the '
         'cross_lingual test split resolves 0.15 at any stated confidence is '
         'never computed. `max_test_items` in the card matrix can shrink the '
         'split further and the reported number would not change shape. The '
         'guard here is `len(set(y_true)) < 2 -> nan`, which is a check that '
         'AUC is DEFINED, not a check that it is resolvable.',
)
@theory.assumes(
    'Hygiene.Measurement.measured_score_tracks_construct::hstable',
    note='the score being ranked is a gradient-times-activation '
         'attribution summed over the reference top-K. Both `dtype` '
         '(bfloat16 on GPU, float32 on CPU) and `batch_size` are card matrix '
         'parameters, and both change those attributions numerically. Nothing '
         'measures whether the AUC is stable across them, so the card cannot '
         'presently distinguish "the predictor works" from "the predictor '
         'works at bf16, batch 4".',
)
def _compute_auc(predicted_mean, actual_mean) -> float:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y_score = np.clip(np.asarray(predicted_mean, dtype=float), -1e9, 1e9)
    y_true  = np.asarray(actual_mean, dtype=float)
    if len(set(y_true.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


@theory.tests(
    'Hygiene.Inference.threshold_exceeds_sampling_error',
    note='this is where "AUC >= 0.65" is either discharged or left standing',
)
@theory.approximates(
    'Hygiene.Measurement.measured_score_tracks_construct',
    note='construct: gets the item right; measured: a rank correlation on a split',
)
@theory.satisfies(
    'Hygiene.Inference.threshold_exceeds_sampling_error::hprespecified',
    note='0.65 is fixed before results are seen, but hard-coded in the claim '
         'block rather than declared as an overridable card symbol',
)
@theory.assumes(
    'Hygiene.Inference.threshold_exceeds_sampling_error::hmultiple',
    note='`mean_auc` is a mean over whatever models matched '
         '`run_spec_pattern`, and the card matrix is explicitly designed to '
         'sweep `k_fraction` and model patterns. Every sweep point is a '
         'comparison against the same 0.65, and no correction is applied. With '
         'the default one-model cell `mean_auc` is that single model AUC, '
         'which makes the family size invisible rather than absent.',
)
@theory.satisfies(
    'Hygiene.Measurement.measured_score_tracks_construct::hscorer',
    note='HELM exact match, where the construct IS string equality',
)
@theory.assumes(
    'Hygiene.Measurement.measured_score_tracks_construct::hcontam',
    note='nothing establishes that the arithmetic_fixed cross_lingual '
         'items are outside Qwen3 pretraining. The direction is not obvious '
         'either: contamination would raise accuracy on the contaminated '
         'items, and the card predicts per-item accuracy, so it could either '
         'help or hurt the AUC depending on whether the circuit-overlap score '
         'tracks memorized items. That is a reason to record it, not to '
         'dismiss it.',
)
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--helm_runs_path",   required=True,
                        help="Path to HELM benchmark_output directory")
    parser.add_argument("--run_spec_pattern", default="arithmetic_fixed*",
                        help="Glob pattern for run directories (default: arithmetic_fixed*)")
    parser.add_argument("--models_root",      default="",
                        help="Local root for model weights (default: load from HF Hub)")
    parser.add_argument("--k_fraction",       type=float, default=0.01,
                        help="Top-K fraction for circuit overlap (default: 0.01)")
    parser.add_argument("--batch_size",       type=int,   default=4,
                        help="Items per attribution forward pass (default: 4)")
    parser.add_argument("--dtype",            default="bfloat16",
                        help="Model weight dtype (default: bfloat16; float32 for a CPU run)")
    # kwdagger renders a declared null default as the literal `None`, so a
    # plain type=int rejects the card's own default (SMELL-MAG-08).
    def _optional_int(text):
        return None if str(text).strip() in ("", "None", "null") else int(text)
    parser.add_argument("--max_train_items",  type=_optional_int, default=None,
                        help="Cap on train items for the reference circuit (mock-runs; default: all)")
    parser.add_argument("--max_test_items",   type=_optional_int, default=None,
                        help="Cap on scored test items (mock-runs; default: all)")
    parser.add_argument("--results_fpath",    required=True,
                        help="Output JSON path")
    args = parser.parse_args()

    import ubelt as ub
    from magnet.backends.helm.helm_outputs import HelmOutputs, HelmRuns
    from magnet_adapter.circuit_overlap_predictor import CircuitOverlapInstancePredictor

    # ── Collect matching runs ─────────────────────────────────────────────────
    helm_data = HelmOutputs(ub.Path(args.helm_runs_path))
    all_paths = []
    for suite in helm_data.suites():
        all_paths.extend(suite.runs(args.run_spec_pattern).paths)

    if not all_paths:
        print(f"ERROR: no runs found matching {args.run_spec_pattern!r} "
              f"under {args.helm_runs_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(all_paths)} run(s) matching {args.run_spec_pattern!r}", flush=True)

    # ── Group run paths by model ──────────────────────────────────────────────
    all_runs = HelmRuns(all_paths)
    run_spec_df = all_runs.run_spec()
    model_col = "run_spec.adapter_spec.model"
    if model_col not in run_spec_df.columns:
        print("ERROR: could not find 'run_spec.adapter_spec.model' column", file=sys.stderr)
        sys.exit(1)

    models = run_spec_df[model_col].unique()
    print(f"Models: {list(models)}", flush=True)

    # ── Evaluate predictor for each model ─────────────────────────────────────
    auc_by_model: dict[str, float] = {}

    for model_id in sorted(models):
        model_run_names = set(
            run_spec_df.loc[run_spec_df[model_col] == model_id, "run_spec.name"]
        )
        model_paths = [
            p for p in all_paths
            if p.name in model_run_names
        ]
        if not model_paths:
            print(f"[{model_id}] no run paths found, skipping", flush=True)
            continue

        model_runs = HelmRuns(model_paths)
        print(f"\n[{model_id}] evaluating on {len(model_paths)} run(s) …", flush=True)

        predictor = CircuitOverlapInstancePredictor(
            models_root=args.models_root,
            k_fraction=args.k_fraction,
            batch_size=args.batch_size,
            dtype=args.dtype,
            max_train_items=args.max_train_items,
            max_test_items=args.max_test_items,
        )
        comparison_df = predictor(helm_runs=model_runs)

        auc = _compute_auc(
            comparison_df["predicted_mean"].tolist(),
            comparison_df["actual_mean"].tolist(),
        )
        model_key = Path(model_id).name
        auc_by_model[model_key] = auc
        print(f"[{model_key}] AUC = {auc:.4f}", flush=True)

    # ── Write results ─────────────────────────────────────────────────────────
    aucs = [v for v in auc_by_model.values() if v == v]  # drop NaN
    results = {
        "result": {
            "auc_by_model": auc_by_model,
            "summary": {
                "n_models": len(auc_by_model),
                "mean_auc": sum(aucs) / len(aucs) if aucs else float("nan"),
                "min_auc":  min(aucs) if aucs else float("nan"),
            },
        }
    }

    out_path = Path(args.results_fpath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}", flush=True)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
