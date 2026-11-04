#!/usr/bin/env python3
"""Build counterfactual files for the attribution-patching circuit overlap predictor.

Each output line is ``{"id", "x", "x_prime"}``, keyed by the HELM instance
id.  ``x`` is the item's text as stored in the aiq-mit-ta2 data files (the
tail of the HELM prompt); the adapter builds the counterfactual prompt by
replacing that tail with ``x_prime``.  Items with no line get a random
length-matched fallback contrast from the predictor.

- arithmetic_fixed: x_prime is the sign-flipped problem from
  arithmetic-inconsistencies/arithmetic-dataset (same ``prob_XXXXXX`` id,
  ``symbolic`` = the reference's ``numeric`` format), with this item's
  one-shot exemplar kept.
- world_modeling (EWOK): x_prime is the item's minimal-pair partner, the same
  targets under the other context (``_ctx1`` ↔ ``_ctx2``), whose answer is the
  other letter.
- arithmetic_parametric: x_prime swaps every operator (+ ↔ −, × ↔ ÷), the
  analogue of the reference's sign flip for a set that also has × and ÷.
  The operator sequence is recovered by re-rendering the item's operands with
  datasets/arithmetic_parametric/generate.py, and x_prime is rendered and
  evaluated the same way.  Items whose swapped problem divides by zero or has
  a non-integer value get no counterfactual.  Needs num2words.

Usage:
    python magnet_adapter/make_counterfactuals.py \\
        --aiq_data_root ~/Library/CloudStorage/Dropbox/code/aiq-mit-ta2/aiq_mit_ta2/backends/helm/run_specs \\
        --reference_data ~/Library/CloudStorage/Dropbox/code/arithmetic-inconsistencies/arithmetic-dataset \\
        --out_dir magnet_adapter/counterfactuals
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_REF_FORMAT = {"symbolic": "numeric", "english": "english",
               "spanish": "spanish", "italian": "italian"}


def _read_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _arithmetic_fixed(aiq_root: Path, ref_root: Path) -> list:
    ref = {}
    for fmt in set(_REF_FORMAT.values()):
        for it in json.load(open(ref_root / f"arith_dataset_{fmt}_counterfactual.json")):
            ref[(fmt, it["id"])] = it
    out = []
    seen = set()
    for path in sorted((aiq_root / "arithmetic_fixed" / "data").glob("*/*.jsonl")):
        for it in _read_jsonl(path):
            # <split>/*.jsonl ids are "<fmt>_<fmt>_prob_N"; the <split>_train
            # copies use "<fmt>_prob_N".
            if it["id"] in seen:
                continue
            seen.add(it["id"])
            m = re.fullmatch(r"([a-z]+)_(?:\1_)?(prob_\d+)", it["id"])
            if not m:
                raise ValueError(f"unexpected arithmetic_fixed id {it['id']!r}")
            r = ref[(_REF_FORMAT[m.group(1)], m.group(2))]
            if not it["x"].endswith(r["x"]):
                raise ValueError(f"{it['id']}: {it['x']!r} does not end with {r['x']!r}")
            prefix = it["x"][:len(it["x"]) - len(r["x"])]
            out.append({"id": it["id"], "x": it["x"], "x_prime": prefix + r["x_prime"]})
    return out


_SWAP = {"+": "-", "-": "+", "*": "/", "/": "*"}


def _arithmetic_parametric(aiq_root: Path) -> tuple:
    from datasets.arithmetic_parametric.generate import _evaluate, render_item

    seen, out, n_skipped = set(), [], 0
    for path in sorted((aiq_root / "arithmetic_parametric" / "data").glob("*/*.jsonl")):
        for it in _read_jsonl(path):
            if it["id"] in seen:
                continue
            seen.add(it["id"])
            feats = it["features"]
            operands, fmt = feats["operands"], feats["format"]
            # Recover the operator sequence: the one whose rendering is x.
            matches = [list(ops) for ops in itertools.product(feats["operators"],
                                                              repeat=len(operands) - 1)
                       if render_item(operands, list(ops), 0, fmt)[0] == it["x"]]
            if len(matches) != 1:
                raise ValueError(f"{it['id']}: {len(matches)} operator sequences render {it['x']!r}")
            ops = matches[0]
            ops_prime = [_SWAP[op] for op in ops]
            value_prime = _evaluate(operands, ops_prime)
            if value_prime is None:
                n_skipped += 1
                continue
            x_prime, _ = render_item(operands, ops_prime, int(value_prime), fmt)
            out.append({"id": it["id"], "x": it["x"], "x_prime": x_prime})
    return out, n_skipped


def _ewok(aiq_root: Path) -> list:
    items = {}
    for path in sorted((aiq_root / "world_modeling" / "data").glob("*/*.jsonl")):
        for it in _read_jsonl(path):
            items[it["id"]] = it
    out = []
    for item_id, it in sorted(items.items()):
        m = re.fullmatch(r"(.*)_ctx([12])", item_id)
        if not m:
            raise ValueError(f"unexpected EWOK id {item_id!r}")
        partner = items[f"{m.group(1)}_ctx{3 - int(m.group(2))}"]
        if partner["y"] == it["y"]:
            raise ValueError(f"{item_id} and its partner have the same answer")
        out.append({"id": item_id, "x": it["x"], "x_prime": partner["x"]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aiq_data_root", required=True, type=Path,
                        help="aiq-mit-ta2 .../backends/helm/run_specs directory")
    parser.add_argument("--reference_data", required=True, type=Path,
                        help="arithmetic-inconsistencies/arithmetic-dataset directory")
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    aiq_root = args.aiq_data_root.expanduser()
    parametric, n_skipped = _arithmetic_parametric(aiq_root)
    print(f"arithmetic_parametric: {n_skipped} items have no counterfactual "
          f"(swapped problem divides by zero or is non-integer)")
    for name, rows in (("arithmetic_fixed", _arithmetic_fixed(aiq_root,
                                                              args.reference_data.expanduser())),
                       ("arithmetic_parametric", parametric),
                       ("world_modeling", _ewok(aiq_root))):
        path = args.out_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows):6d} counterfactuals → {path}")


if __name__ == "__main__":
    main()
