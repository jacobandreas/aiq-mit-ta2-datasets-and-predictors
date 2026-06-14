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


def _compute_auc(predicted_mean, actual_mean) -> float:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y_score = np.clip(np.asarray(predicted_mean, dtype=float), -1e9, 1e9)
    y_true  = np.asarray(actual_mean, dtype=float)
    if len(set(y_true.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


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
