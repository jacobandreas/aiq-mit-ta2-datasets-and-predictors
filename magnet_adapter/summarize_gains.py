#!/usr/bin/env python3
"""KWDagger node: summarize one model's combined-vs-token-prob gains across splits.

Receives a KWDagger gather manifest listing the ``results.json`` files of the
``run_circuit_overlap`` nodes for one model (one per split; see
run_predictor_card.py) and writes a flat summary for the ensemble card's
claim:

    {"model": "huggingface/Qwen3-4B", "n_splits": 5,
     "min_gain": 0.0, "mean_gain": 0.017, "worst_split": "cross_lingual",
     "mean_combined_auc": ..., "mean_token_prob_auc": ...,
     "gain_cross_lingual": 0.0, "weight_cross_lingual": 0.0, ...}

Usage:
    python -u magnet_adapter/summarize_gains.py --model huggingface/Qwen3-4B \\
        --results_fpaths manifest.txt --summary_fpath summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_kwdagger_result(node, node_dpath):
    """Load the flat summary JSON into KWDagger's result namespace."""
    from kwdagger.utils import util_dotdict

    output_fpath = Path(node_dpath) / node.out_paths[node.primary_out_key]
    metrics = json.loads(output_fpath.read_text())
    flat = util_dotdict.DotDict.from_nested({"metrics": metrics})
    return flat.insert_prefix(node.name, index=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="HELM model id the gathered results belong to")
    parser.add_argument("--results_fpaths", required=True,
                        help="KWDagger gather manifest: one run_circuit_overlap "
                             "results.json path per line")
    parser.add_argument("--summary_fpath", required=True,
                        help="Where to write the summary JSON")
    args = parser.parse_args()

    paths = [line.strip() for line in Path(args.results_fpaths).read_text().splitlines()
             if line.strip()]
    results = [json.loads(Path(p).read_text()) for p in paths]
    if not results:
        raise SystemExit(f"no results listed in {args.results_fpaths}")
    models = {r["model"] for r in results}
    if models != {args.model}:
        raise SystemExit(f"gathered results are for {sorted(models)}, expected {args.model!r}")
    splits = [r["split_config"] for r in results]
    if len(set(splits)) != len(splits):
        raise SystemExit(f"duplicate splits in gathered results: {splits}")

    gains = {r["split_config"]: r["combined_gain"] for r in results}
    worst = min(gains, key=gains.get)
    n = len(results)
    summary = {
        "model": args.model,
        "n_splits": n,
        "min_gain": gains[worst],
        "mean_gain": sum(gains.values()) / n,
        "worst_split": worst,
        "n_splits_gain_positive": sum(g > 0 for g in gains.values()),
        "mean_combined_auc": sum(r["combined_auc"] for r in results) / n,
        "mean_token_prob_auc": sum(r["token_prob_auc"] for r in results) / n,
        "mean_circuit_overlap_auc": sum(r["auc"] for r in results) / n,
    }
    for r in sorted(results, key=lambda r: r["split_config"]):
        summary[f"gain_{r['split_config']}"] = r["combined_gain"]
        summary[f"weight_{r['split_config']}"] = r["combined_weight"]

    out = Path(args.summary_fpath)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
