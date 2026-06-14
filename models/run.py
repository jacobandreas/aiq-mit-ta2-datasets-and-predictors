"""Orchestrator for model evaluation and training.

Subcommands:
    eval    -- zero-shot evaluation of off-the-shelf models
    train   -- fine-tune / train-from-scratch (not yet active; use when ready)
    aggregate -- compute cross-model agreement statistics

Usage:
    # Off-the-shelf evaluation on all discovered split configs
    python -m models.run eval \\
        --data-dir data \\
        --families gpt2_small llama32_1b qwen3_0_6b \\
        --seeds 0 \\
        --output-dir runs \\
        --models-root /raid/lingo/models

    # Aggregate results
    python -m models.run aggregate --runs-dir runs

    # Force re-run even if output already exists
    python -m models.run eval ... --force
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_dir(output_dir: Path, dataset_dir: Path, data_root: Path, family: str, seed: int) -> Path:
    rel = dataset_dir.relative_to(data_root)
    return output_dir / str(rel) / family / f"seed{seed}"


def _is_done(run_dir: Path) -> bool:
    return (run_dir / "metrics.json").exists()


def _load_model_and_tokenizer(model_path: str, tokenizer_path: str, dtype_str: str):
    """Lazy-load to keep import time short."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"float32": torch.float32, "float16": torch.float16,
                   "bfloat16": torch.bfloat16}.get(dtype_str, torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"    # required for batch generation

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
    return model, tokenizer


def _device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# ── eval subcommand ───────────────────────────────────────────────────────────

def cmd_eval(args: argparse.Namespace) -> None:
    from models.config import load_model_config
    from models.data_loader import discover_split_dirs, load_splits
    from models.evaluate import (
        compute_metrics, evaluate, write_metrics, write_predictions,
    )
    from models.train import _run_eval_and_save

    data_root = Path(args.data_dir)
    out_root = Path(args.output_dir)
    dataset_dirs = discover_split_dirs(data_root)

    if not dataset_dirs:
        print(f"No split directories found under {data_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(dataset_dirs)} split config(s):")
    for d in dataset_dirs:
        print(f"  {d.relative_to(data_root)}")

    device = _device()
    print(f"Device: {device}")

    for family in args.families:
        cfg = load_model_config(family)

        if not cfg.zero_shot:
            print(f"[{family}] zero_shot=False — skipping (use 'train' subcommand)")
            continue

        model_path = cfg.resolve_model_path(args.models_root or None)
        # Use model_path for the tokenizer too: tokenizer files live alongside
        # model weights, so if we resolved a local path for the model, use it
        # for the tokenizer as well (avoids HF Hub auth for gated repos).
        tokenizer_path = model_path
        print(f"\n[{family}] Loading model from {model_path} …")
        try:
            model, tokenizer = _load_model_and_tokenizer(
                model_path, tokenizer_path, cfg.dtype
            )
            model = model.to(device)
        except Exception as e:
            print(f"[{family}] ERROR loading model: {e}", file=sys.stderr)
            continue

        for dataset_dir in dataset_dirs:
            for seed in args.seeds:
                run_dir = _run_dir(out_root, dataset_dir, data_root, family, seed)

                if _is_done(run_dir) and not args.force:
                    print(f"  [skip] {run_dir.relative_to(out_root)} already done")
                    continue

                rel_name = str(dataset_dir.relative_to(data_root))
                print(f"  {rel_name} seed={seed} …", end=" ", flush=True)

                splits = load_splits(dataset_dir, seed=seed)
                test_items = splits.get("test", [])
                val_items = splits.get("val", [])

                if not test_items:
                    print("(no test items, skipping)")
                    continue

                run_dir.mkdir(parents=True, exist_ok=True)

                # Write config
                run_config_dict = {
                    "dataset_dir": str(dataset_dir),
                    "family": family,
                    "seed": seed,
                    "output_dir": str(run_dir),
                    "models_root": args.models_root or "",
                    "model_config": cfg.to_dict(),
                }
                with open(run_dir / "config.json", "w") as f:
                    json.dump(run_config_dict, f, indent=2)

                # Evaluate
                all_metrics: dict = {}
                for split_name, items in (("val", val_items), ("test", test_items)):
                    if not items:
                        continue
                    preds = evaluate(
                        model, tokenizer, items,
                        max_new_tokens=cfg.max_new_tokens,
                        batch_size=cfg.eval_batch_size,
                        device=device,
                    )
                    write_predictions(preds, run_dir / f"predictions_{split_name}.jsonl")
                    all_metrics[split_name] = compute_metrics(preds)

                write_metrics(all_metrics, run_dir / "metrics.json")

                test_acc = all_metrics.get("test", {}).get("accuracy", 0.0)
                test_n = all_metrics.get("test", {}).get("n", 0)
                print(f"acc={test_acc:.3f} (n={test_n})")

        # Free GPU memory between families
        try:
            import torch
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass


# ── train subcommand (future) ─────────────────────────────────────────────────

def cmd_train(args: argparse.Namespace) -> None:
    from models.config import load_model_config, RunConfig
    from models.data_loader import discover_split_dirs, load_splits
    from models.train import train_and_eval

    data_root = Path(args.data_dir)
    out_root = Path(args.output_dir)
    dataset_dirs = discover_split_dirs(data_root)
    device = _device()

    for family in args.families:
        cfg = load_model_config(family)
        if cfg.zero_shot:
            print(f"[{family}] zero_shot=True — use 'eval' subcommand")
            continue

        model_path = cfg.resolve_model_path(args.models_root or None)

        for dataset_dir in dataset_dirs:
            for seed in args.seeds:
                run_dir = _run_dir(out_root, dataset_dir, data_root, family, seed)
                if _is_done(run_dir) and not args.force:
                    print(f"[skip] {run_dir}")
                    continue

                rel_name = str(dataset_dir.relative_to(data_root))
                print(f"Training {family} on {rel_name} seed={seed}")

                splits = load_splits(dataset_dir, seed=seed)

                run_cfg = RunConfig(
                    dataset_dir=str(dataset_dir),
                    family=family,
                    seed=seed,
                    output_dir=str(run_dir),
                    models_root=args.models_root or "",
                    model_config=cfg,
                )
                run_cfg.write(run_dir / "config.json")

                train_and_eval(
                    run_cfg,
                    train_items=splits.get("train", []),
                    val_items=splits.get("val", []),
                    test_items=splits.get("test", []),
                    log_path=run_dir / "train_log.jsonl",
                    device=device,
                )


# ── aggregate subcommand ──────────────────────────────────────────────────────

def cmd_aggregate(args: argparse.Namespace) -> None:
    from models.aggregate import aggregate

    runs_dir = Path(args.runs_dir)
    summary = aggregate(runs_dir)

    out_path = Path(args.output) if args.output else runs_dir / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {out_path}")

    for dataset, stats in summary.items():
        print(f"\n{'='*60}\nDataset: {dataset}")
        for run_key, m in sorted(stats["per_run"].items()):
            print(f"  {run_key:45s}  acc={m['accuracy']:.3f}  n={m['n']}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default="data",
                        help="Root of the data directory")
    parser.add_argument("--families", nargs="+",
                        default=["gpt2_small", "llama32_1b", "qwen3_0_6b"],
                        help="Model families to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                        help="Random seeds")
    parser.add_argument("--output-dir", default="runs",
                        help="Root output directory for run artifacts")
    parser.add_argument("--models-root", default="",
                        help="Local root for model weights (server: /raid/lingo/models)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if output already exists")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    eval_p = sub.add_parser("eval", help="Zero-shot evaluation of off-the-shelf models")
    _add_common(eval_p)

    train_p = sub.add_parser("train", help="Fine-tune / train-from-scratch")
    _add_common(train_p)

    agg_p = sub.add_parser("aggregate", help="Compute cross-model agreement statistics")
    agg_p.add_argument("--runs-dir", default="runs")
    agg_p.add_argument("--output", default=None)

    args = parser.parse_args()
    {"eval": cmd_eval, "train": cmd_train, "aggregate": cmd_aggregate}[args.cmd](args)


if __name__ == "__main__":
    main()
