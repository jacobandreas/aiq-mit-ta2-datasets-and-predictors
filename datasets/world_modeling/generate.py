"""Generate world_modeling dataset from the EWOK-CORE HuggingFace dataset.

Each EWOK domain becomes a split. Items are stored by split name in
data/world_modeling/<domain>/train.jsonl and test.jsonl.

Usage:
    python -m datasets.world_modeling.generate \\
        --output-dir data/world_modeling
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets.utils import make_item, write_jsonl


def load_ewok(hf_split: str):
    import pandas as pd
    url = f"hf://datasets/ewok-core/ewok-core-1.0/data/{hf_split}/ewok-core-1.0.parquet"
    return pd.read_parquet(url)


def df_to_items(df, split: str) -> list[dict]:
    items = []
    for idx, row in df.iterrows():
        for ctx_num in (1, 2):
            context = row[f"Context{ctx_num}"]
            correct = "A" if ctx_num == 1 else "B"
            x = (
                f"{context}\n\n"
                f"Which statement is more appropriate given the context above?\n\n"
                f"A. {row['Target1']}\n"
                f"B. {row['Target2']}"
            )
            y = correct
            items.append(make_item(
                id=f"ewok_{split}_{idx}_ctx{ctx_num}",
                x=x,
                y=y,
                split=split,
                features={
                    "domain": row["Domain"],
                    "concept_a": row["ConceptA"],
                    "concept_b": row["ConceptB"],
                    "context_type": row["ContextType"],
                    "context_diff": row["ContextDiff"],
                    "target_diff": row["TargetDiff"],
                },
            ))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/world_modeling")
    args = parser.parse_args()

    out = Path(args.output_dir)

    for hf_split in ("train", "test"):
        print(f"Loading EWOK {hf_split} split from HuggingFace …")
        df = load_ewok(hf_split)
        items = df_to_items(df, hf_split)

        domains = sorted(set(it["features"]["domain"] for it in items))
        print(f"  {len(items)} items across domains: {domains}")

        for domain in domains:
            domain_items = [it for it in items if it["features"]["domain"] == domain]
            dest = out / domain.replace(" ", "_") / f"{hf_split}.jsonl"
            write_jsonl(dest, domain_items)
            print(f"  Wrote {len(domain_items)} items → {dest}")

    print("Done.")


if __name__ == "__main__":
    main()
