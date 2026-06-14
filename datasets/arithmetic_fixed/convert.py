"""Convert arithmetic-inconsistencies dataset to darpa3 format.

Source: ~/code/arithmetic-inconsistencies/arithmetic-dataset/
Each JSON file contains items with fields: id, x, y, split, n_terms, n_digits, has_carry, ...

Items are combined across languages into named split configurations.  Each
configuration assigns languages to the train or test partition:

    cross_lingual  (default)
        train: english, symbolic
        test:  spanish, italian

New split configurations can be added to SPLIT_CONFIGS below without changing
anything else; just re-run the converter.

All items from each language are included regardless of the source file's
original train/test assignment — the cross-lingual split is defined by
*which language* an item is in, not by which source partition it came from.

Usage:
    python -m datasets.arithmetic_fixed.convert \\
        --source-dir /path/to/arithmetic-inconsistencies/arithmetic-dataset \\
        --output-dir data/arithmetic_fixed
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets.utils import make_item, write_jsonl

# Map source filename stem → canonical language name
LANG_MAP = {
    "arith_dataset_english_counterfactual": "english",
    "arith_dataset_spanish_counterfactual": "spanish",
    "arith_dataset_italian_counterfactual": "italian",
    "arith_dataset_numeric_counterfactual": "symbolic",
}

# One-shot exemplar prepended to x so the model knows the expected output format.
# Matches ONE_SHOT_EXEMPLARS in arithmetic-inconsistencies/attribution/src/data.py.
ONE_SHOT_EXEMPLARS = {
    "english":  "seven plus five equals twelve\n",
    "spanish":  "siete más cinco es igual a doce\n",
    "italian":  "sette più cinque fa dodici\n",
    "symbolic": "7 + 5 = 12\n",
}

# Split configurations: maps config_name → {"train": [langs], "test": [langs]}
# Languages listed here determine which partition each item goes into.
SPLIT_CONFIGS: dict[str, dict[str, list[str]]] = {
    "cross_lingual": {
        "train": ["english", "symbolic"],
        "test":  ["spanish", "italian"],
    },
}


def load_language(src: Path, lang: str) -> list[dict]:
    """Load and convert all items for one language from the source JSON file."""
    exemplar = ONE_SHOT_EXEMPLARS[lang]
    items = []
    with open(src) as f:
        raw = json.load(f)
    for entry in raw:
        x = exemplar + entry["x"].strip()
        y = str(entry["y"]).strip()
        items.append(make_item(
            id=entry["id"],
            x=x,
            y=y,
            split=None,  # will be set per split config below
            features={
                "language": lang,
                "n_terms": entry.get("n_terms"),
                "n_digits": entry.get("n_digits"),
                "has_carry": entry.get("has_carry"),
            },
        ))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source-dir",
        default=str(
            Path.home()
            / "Library/CloudStorage/Dropbox/code/arithmetic-inconsistencies/arithmetic-dataset"
        ),
    )
    parser.add_argument("--output-dir", default="data/arithmetic_fixed")
    args = parser.parse_args()

    src = Path(args.source_dir)
    out = Path(args.output_dir)

    # Load all languages
    by_lang: dict[str, list[dict]] = {}
    for fname, lang in LANG_MAP.items():
        path = src / f"{fname}.json"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping")
            continue
        print(f"Loading {fname} → {lang}")
        by_lang[lang] = load_language(path, lang)
        print(f"  {len(by_lang[lang])} items")

    # Write each split configuration
    for config_name, partitions in SPLIT_CONFIGS.items():
        print(f"\nWriting split config: {config_name}")
        for split_name, langs in partitions.items():
            items = []
            for lang in langs:
                if lang not in by_lang:
                    print(f"  WARNING: {lang} not loaded, skipping")
                    continue
                for item in by_lang[lang]:
                    items.append({**item, "split": split_name})
            dest = out / config_name / f"{split_name}.jsonl"
            write_jsonl(dest, items)
            print(f"  {config_name}/{split_name}.jsonl: {len(items)} items "
                  f"({', '.join(langs)})")

    print("\nDone.")


if __name__ == "__main__":
    main()
