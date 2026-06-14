#!/usr/bin/env python3
"""Evaluate the darpa3 CircuitOverlapPredictor and report per-model AUC.

Discovers darpa3 run directories, loads precomputed val/test predictions,
fits the circuit overlap predictor on val, evaluates on test, then writes
a results JSON file.

MAGNET's GenericPipelineProcessor lifts each top-level key of the results
JSON into the card claim namespace.

Output format:
    {
        "auc_by_model":  {"meta-llama/Llama-3.2-3B": 0.43, ...},
        "summary":       {"n_models": 2, "mean_auc": 0.43, "min_auc": 0.43}
    }

Usage:
    python -u magnet_adapter/run_predictor_card.py \\
        --runs_dir /raid/lingo/jda/code/darpa3/runs \\
        --domain arithmetic_fixed \\
        --split_config cross_lingual \\
        --model_family llama \\
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _resolve_model_path(model_id: str, models_root: str) -> str:
    """Return local path if available under models_root, else the HF model ID."""
    if models_root:
        # darpa3 convention: <models_root>/<basename>
        local = Path(models_root) / Path(model_id).name
        if local.exists():
            return str(local)
    return model_id


def _load_model_and_tokenizer(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def _discover_run_dirs(runs_dir: Path, domain: str, split_config: str,
                       model_family: str) -> list[tuple[Path, str]]:
    """Find run dirs matching domain/split_config with the given model family."""
    base = runs_dir / domain / split_config
    if not base.exists():
        return []
    results = []
    for seed_dir in sorted(base.glob("*/seed0")):
        config_path = seed_dir / "config.json"
        val_path    = seed_dir / "predictions_val.jsonl"
        test_path   = seed_dir / "predictions_test.jsonl"
        if not (config_path.exists() and val_path.exists() and test_path.exists()):
            continue
        config = json.loads(config_path.read_text())
        model_id = config.get("model_config", {}).get("model_name_or_path", "")
        if not model_id:
            continue
        model_basename = Path(model_id).name.lower()
        if model_family.lower() not in model_basename:
            continue
        results.append((seed_dir, model_id))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs_dir",     required=True,
                        help="Root of darpa3 run directories")
    parser.add_argument("--domain",       default="arithmetic_fixed",
                        help="Dataset domain name (default: arithmetic_fixed)")
    parser.add_argument("--split_config", default="cross_lingual",
                        help="Split configuration (default: cross_lingual)")
    parser.add_argument("--model_family", required=True,
                        help="Model family filter, e.g. 'llama' or 'qwen'")
    parser.add_argument("--models_root",  default="",
                        help="Local root for model weights (default: load from HF Hub)")
    parser.add_argument("--k_fraction",   type=float, default=0.01,
                        help="Top-K fraction for circuit overlap (default: 0.01)")
    parser.add_argument("--batch_size",   type=int, default=4,
                        help="Items per attribution forward pass (default: 4)")
    parser.add_argument("--results_fpath", required=True,
                        help="Output JSON path")
    args = parser.parse_args()

    import torch
    from predictors.circuit_overlap.predictor import CircuitOverlapPredictor
    from predictors.base import compute_auc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs_dir = Path(args.runs_dir)

    run_dirs = _discover_run_dirs(runs_dir, args.domain, args.split_config,
                                  args.model_family)
    if not run_dirs:
        print(f"ERROR: no matching runs found under {runs_dir}/{args.domain}/{args.split_config} "
              f"for model_family={args.model_family!r}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(run_dirs)} run(s) to evaluate:", flush=True)
    for sd, mid in run_dirs:
        print(f"  {mid}  →  {sd}", flush=True)

    auc_by_model: dict[str, float] = {}

    for seed_dir, model_id in run_dirs:
        model_path = _resolve_model_path(model_id, args.models_root)
        print(f"\n[{model_id}] loading model from {model_path} …", flush=True)
        model, tokenizer = _load_model_and_tokenizer(model_path)
        model = model.to(device)

        val_preds  = _read_jsonl(seed_dir / "predictions_val.jsonl")
        test_preds = _read_jsonl(seed_dir / "predictions_test.jsonl")
        print(f"[{model_id}] val={len(val_preds)} test={len(test_preds)}", flush=True)

        predictor = CircuitOverlapPredictor(
            k_fraction=args.k_fraction,
            batch_size=args.batch_size,
            device=device,
        )
        predictor.fit(val_preds, model=model, tokenizer=tokenizer)
        test_scores = predictor.predict(test_preds, model=model, tokenizer=tokenizer)

        test_labels = [int(p.get("correct", False)) for p in test_preds]
        auc = compute_auc(test_scores, test_labels)
        # Key by basename so the card can reference e.g. "Llama-3.2-3B"
        # regardless of whether config.json stores a HF ID or a local path.
        model_key = Path(model_id).name
        auc_by_model[model_key] = auc
        print(f"[{model_key}] test AUC = {auc:.4f}", flush=True)

        del model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    aucs = list(auc_by_model.values())
    results = {
        "auc_by_model": auc_by_model,
        "summary": {
            "n_models":  len(aucs),
            "mean_auc":  sum(aucs) / len(aucs) if aucs else float("nan"),
            "min_auc":   min(aucs) if aucs else float("nan"),
        },
    }

    out_path = Path(args.results_fpath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}", flush=True)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
