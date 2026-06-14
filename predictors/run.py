"""Predictor orchestrator.

Subcommands:
    fit-predict   Run a predictor on a single (run_dir, predictor) pair.
    sweep         Run across all discovered run directories.
    analyze       Re-run analysis on saved predictions without recomputing
                  attributions (reads scores from predictor_*.jsonl files).

Usage:
    # Single run
    python -m predictors.run fit-predict \\
        --run-dir runs/arithmetic_parametric/arithmetic_by_format/gpt2_small/seed0 \\
        --predictor circuit_overlap \\
        --models-root /raid/lingo/models

    # Sweep all completed runs
    python -m predictors.run sweep \\
        --runs-dir runs \\
        --predictor circuit_overlap \\
        --models-root /raid/lingo/models

    # Aggregate analysis results
    python -m predictors.run aggregate --runs-dir runs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Helpers ───────────────────────────────────────────────────────────────────

PREDICTORS = {
    "circuit_overlap": "predictors.circuit_overlap.predictor.CircuitOverlapPredictor",
    "linear_probe":    "predictors.linear_probe.predictor.LinearProbePredictor",
    "token_prob":      "predictors.token_prob.predictor.TokenProbPredictor",
    "ensemble":        "predictors.ensemble.predictor.EnsemblePredictor",
}


def _load_predictor_class(name: str):
    module_path, cls_name = PREDICTORS[name].rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def _read_jsonl(path: Path) -> list:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _write_jsonl(path: Path, items: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def _load_model_and_tokenizer(run_dir: Path, config: dict, models_root: str = ""):
    """Load model from run_dir/final/ if it exists, else from HF Hub."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = config.get("model_config", {})
    local_final = run_dir / "final"
    if local_final.exists():
        model_path = str(local_final)
    else:
        mpath = model_cfg.get("model_name_or_path", "")
        if models_root:
            candidate = Path(models_root) / Path(mpath).name
            model_path = str(candidate) if candidate.exists() else mpath
        else:
            model_path = mpath

    dtype_str = model_cfg.get("dtype", "float16")
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}.get(dtype_str, torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    # Freeze params — we only need activation grads during attribution
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def _device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ── fit-predict ───────────────────────────────────────────────────────────────

def _run_one(
    run_dir: Path,
    predictor_name: str,
    models_root: str,
    batch_size: int,
    k_fraction: float,
    force: bool,
) -> None:
    pred_dir = run_dir / f"predictor_{predictor_name}"
    analysis_path = pred_dir / "analysis.json"

    if analysis_path.exists() and not force:
        print(f"  [skip] {run_dir.name} — {predictor_name} already done")
        return

    val_path = run_dir / "predictions_val.jsonl"
    test_path = run_dir / "predictions_test.jsonl"
    if not val_path.exists() or not test_path.exists():
        print(f"  [skip] {run_dir} — missing predictions files")
        return

    config_path = run_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    val_preds = _read_jsonl(val_path)
    test_preds = _read_jsonl(test_path)

    if not val_preds or not test_preds:
        print(f"  [skip] {run_dir} — empty predictions")
        return

    device = _device()
    run_dir_abs = run_dir.resolve()
    _label = run_dir_abs.relative_to(run_dir_abs.parents[3]) if len(run_dir_abs.parents) > 3 else run_dir_abs.name
    print(f"  Loading model for {_label} ({device}) …")
    model, tokenizer = _load_model_and_tokenizer(run_dir, config, models_root)
    model = model.to(device)

    PredClass = _load_predictor_class(predictor_name)
    import inspect
    sig = inspect.signature(PredClass.__init__)
    all_kwargs = {"k_fraction": k_fraction, "batch_size": batch_size, "device": device}
    constructor_kwargs = {k: v for k, v in all_kwargs.items() if k in sig.parameters}
    predictor = PredClass(**constructor_kwargs)

    # Fit on val
    predictor.fit(val_preds, model=model, tokenizer=tokenizer)

    # Predict on test
    test_scores = predictor.predict(test_preds, model=model, tokenizer=tokenizer)
    val_scores = predictor.predict(val_preds, model=model, tokenizer=tokenizer)

    # Save predictor
    predictor.save(pred_dir)

    # Write score files
    val_out = [
        {"id": p["id"], "score": s, "correct": p.get("correct"), "features": p.get("features")}
        for p, s in zip(val_preds, val_scores)
    ]
    test_out = [
        {"id": p["id"], "score": s, "correct": p.get("correct"), "features": p.get("features")}
        for p, s in zip(test_preds, test_scores)
    ]
    _write_jsonl(pred_dir / "scores_val.jsonl", val_out)
    _write_jsonl(pred_dir / "scores_test.jsonl", test_out)

    # Analysis
    from predictors.base import analyze, write_analysis
    val_labels = [int(p["correct"]) for p in val_preds]
    test_labels = [int(p.get("correct", False)) for p in test_preds]
    results = analyze(val_scores, val_labels, test_scores, test_labels)
    write_analysis(results, analysis_path)

    print(f"    val AUC={results['val']['auc']:.3f}  "
          f"test AUC={results['test']['auc']:.3f}  "
          f"test F1={results['test']['f1_at_val_threshold']:.3f}")

    # Free GPU memory
    try:
        import torch
        del model
        torch.cuda.empty_cache()
    except Exception:
        pass


def cmd_fit_predict(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: {run_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    predictors = list(PREDICTORS) if args.predictor == "all" else [args.predictor]
    for pred in predictors:
        _run_one(run_dir, pred, args.models_root,
                 args.batch_size, args.k_fraction, args.force)


# ── sweep ─────────────────────────────────────────────────────────────────────

def cmd_sweep(args: argparse.Namespace) -> None:
    """Run fit-predict across every completed run in runs_dir."""
    runs_dir = Path(args.runs_dir)
    run_dirs = sorted(
        p.parent for p in runs_dir.rglob("metrics.json")
        if (p.parent / "predictions_test.jsonl").exists()
    )
    predictors = list(PREDICTORS) if args.predictor == "all" else [args.predictor]
    print(f"Found {len(run_dirs)} completed run(s), running {predictors}")
    for run_dir in run_dirs:
        print(f"\n→ {run_dir.relative_to(runs_dir)}")
        for pred in predictors:
            _run_one(run_dir, pred, args.models_root,
                     args.batch_size, args.k_fraction, args.force)


# ── aggregate ─────────────────────────────────────────────────────────────────

def cmd_aggregate(args: argparse.Namespace) -> None:
    """Collect all predictor analysis.json files into a single comparison table."""
    runs_dir = Path(args.runs_dir)
    pred_names = list(PREDICTORS) if args.predictor == "all" else [args.predictor]

    # Gather results per predictor
    all_results: dict[str, dict[str, dict]] = {}  # predictor → run_key → metrics
    for pred in pred_names:
        all_results[pred] = {}
        for analysis_path in sorted(runs_dir.rglob(f"predictor_{pred}/analysis.json")):
            parts = analysis_path.relative_to(runs_dir).parts
            key = "/".join(parts[:-2])
            with open(analysis_path) as f:
                all_results[pred][key] = json.load(f)

    # Collect all run keys across all predictors
    all_keys = sorted({k for res in all_results.values() for k in res})
    if not all_keys:
        print(f"No analysis files found under {runs_dir} for {pred_names}")
        return

    # Write combined summary
    combined = {key: {pred: all_results[pred].get(key) for pred in pred_names}
                for key in all_keys}
    out_path = runs_dir / "predictor_comparison.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Summary written to {out_path}\n")

    # Print comparison table
    col_w = 22
    header = f"{'Run':<50}" + "".join(f"  {'val/test AUC':>{col_w}}" for _ in pred_names)
    sub    = f"{'':50}" + "".join(f"  {p:>{col_w}}" for p in pred_names)
    print(sub)
    print("-" * (50 + len(pred_names) * (col_w + 2)))
    for key in all_keys:
        row = f"{key:<50}"
        for pred in pred_names:
            res = all_results[pred].get(key)
            if res is None:
                row += f"  {'—':>{col_w}}"
            else:
                vauc = res.get("val", {}).get("auc", float("nan"))
                tauc = res.get("test", {}).get("auc", float("nan"))
                row += f"  {vauc:.3f} / {tauc:.3f}".rjust(col_w + 2)
        print(row)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--predictor", default="circuit_overlap",
                   choices=list(PREDICTORS) + ["all"], help="Predictor to use (or 'all')")
    p.add_argument("--models-root", default="",
                   help="Local root for model weights (server: /raid/lingo/models)")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Items per attribution forward pass")
    p.add_argument("--k-fraction", type=float, default=0.01,
                   help="Top-K fraction for circuit overlap (default: 1%%)")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if output already exists")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    fp = sub.add_parser("fit-predict", help="Run predictor on a single run directory")
    fp.add_argument("--run-dir", required=True)
    _add_common(fp)

    sw = sub.add_parser("sweep", help="Run predictor across all completed run dirs")
    sw.add_argument("--runs-dir", default="runs")
    _add_common(sw)

    ag = sub.add_parser("aggregate", help="Collect all analysis results into a comparison table")
    ag.add_argument("--runs-dir", default="runs")
    ag.add_argument("--predictor", default="all",
                    choices=list(PREDICTORS) + ["all"])

    args = parser.parse_args()
    {"fit-predict": cmd_fit_predict, "sweep": cmd_sweep,
     "aggregate": cmd_aggregate}[args.cmd](args)


if __name__ == "__main__":
    main()
