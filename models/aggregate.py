"""Cross-model agreement statistics over a completed runs/ directory.

Usage:
    python -m models.aggregate --runs-dir runs --output runs/summary.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def _read_predictions(path: Path) -> List[dict]:
    preds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))
    return preds


def _disagreement(preds_a: Dict[str, bool], preds_b: Dict[str, bool]) -> float:
    """Fraction of shared item IDs on which two prediction sets disagree."""
    shared = set(preds_a) & set(preds_b)
    if not shared:
        return 0.0
    return sum(1 for k in shared if preds_a[k] != preds_b[k]) / len(shared)


def aggregate(runs_dir: Path) -> dict:
    """
    Walk runs_dir and compute per-dataset cross-model agreement statistics.

    runs_dir/
      <dataset>/
        <family>/
          seed0/predictions_test.jsonl
          seed1/predictions_test.jsonl
          ...

    Returns nested dict:
        {
            "<dataset>": {
                "per_run": {
                    "<family>/seed<N>": {"n": int, "accuracy": float}
                },
                "family_consensus": {
                    "<family>": {"n": int, "accuracy": float}
                },
                "within_family_disagreement": {
                    "<family>": float          # mean pairwise disagreement between seeds
                },
                "between_family_disagreement": {
                    "<family_a>_vs_<family_b>": float
                },
            },
            ...
        }
    """
    runs_dir = Path(runs_dir)
    result = {}

    # Collect all prediction sets grouped by (dataset, family, seed)
    # Structure: data[dataset][family][seed] = {item_id: correct}
    data: dict = defaultdict(lambda: defaultdict(dict))

    for pred_path in sorted(runs_dir.rglob("predictions_test.jsonl")):
        # Expected path: runs_dir/<dataset>/<family>/seed<N>/predictions_test.jsonl
        parts = pred_path.relative_to(runs_dir).parts
        if len(parts) < 4:
            continue
        dataset, family, seed_dir = parts[0], parts[1], parts[2]
        if not seed_dir.startswith("seed"):
            continue
        seed = seed_dir

        preds = _read_predictions(pred_path)
        data[dataset][family][seed] = {p["id"]: p["correct"] for p in preds}

    for dataset, families in data.items():
        result[dataset] = {
            "per_run": {},
            "family_consensus": {},
            "within_family_disagreement": {},
            "between_family_disagreement": {},
        }

        # Per-run accuracy
        for family, seeds in families.items():
            for seed, id_correct in seeds.items():
                key = f"{family}/{seed}"
                n = len(id_correct)
                acc = sum(id_correct.values()) / n if n else 0.0
                result[dataset]["per_run"][key] = {"n": n, "accuracy": acc}

        # Within-family disagreement and consensus
        for family, seeds in families.items():
            seed_list = sorted(seeds.keys())
            if len(seed_list) < 2:
                result[dataset]["within_family_disagreement"][family] = 0.0
            else:
                pairs = [(seeds[a], seeds[b])
                         for i, a in enumerate(seed_list)
                         for b in seed_list[i + 1:]]
                result[dataset]["within_family_disagreement"][family] = (
                    sum(_disagreement(a, b) for a, b in pairs) / len(pairs)
                )

            # Consensus: majority vote across seeds
            all_ids = set().union(*[set(s.keys()) for s in seeds.values()])
            consensus = {}
            for item_id in all_ids:
                votes = [seeds[s].get(item_id, False) for s in seed_list]
                consensus[item_id] = sum(votes) > len(votes) / 2
            n = len(consensus)
            acc = sum(consensus.values()) / n if n else 0.0
            result[dataset]["family_consensus"][family] = {"n": n, "accuracy": acc}

        # Between-family disagreement (consensus vs consensus)
        family_names = sorted(families.keys())
        consensus_preds = {}
        for family, seeds in families.items():
            seed_list = sorted(seeds.keys())
            all_ids = set().union(*[set(s.keys()) for s in seeds.values()])
            consensus_preds[family] = {
                item_id: sum(seeds[s].get(item_id, False) for s in seed_list) > len(seed_list) / 2
                for item_id in all_ids
            }
        for i, fa in enumerate(family_names):
            for fb in family_names[i + 1:]:
                key = f"{fa}_vs_{fb}"
                result[dataset]["between_family_disagreement"][key] = (
                    _disagreement(consensus_preds[fa], consensus_preds[fb])
                )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output", default=None,
                        help="Write JSON summary to this path (default: <runs-dir>/summary.json)")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    summary = aggregate(runs_dir)

    out_path = Path(args.output) if args.output else runs_dir / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {out_path}")

    # Print a brief table
    for dataset, stats in summary.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        for run_key, m in sorted(stats["per_run"].items()):
            print(f"  {run_key:40s}  acc={m['accuracy']:.3f}  n={m['n']}")


if __name__ == "__main__":
    main()
