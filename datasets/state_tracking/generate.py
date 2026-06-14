"""Generate state_tracking dataset.

Items involve computing a sequence of swap operations on an ordered list.
Initial state is always the identity (ABC or ABCDE).

Two split configurations:
  state_tracking_by_length  -- format varies freely; train/val/test by swap count
  state_tracking_by_format  -- length varies freely; train/val/test by format

Usage:
    python -m datasets.state_tracking.generate \\
        --output-dir data/state_tracking \\
        --n-train 2000 --n-val 500 --n-test 1000 --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets.utils import make_item, write_jsonl

ITEMS_3 = list("ABC")
ITEMS_5 = list("ABCDE")

FORMATS = ["symbolic", "code", "natural"]


# ── State computation ────────────────────────────────────────────────────────

def apply_swaps(items: list[str], swaps: list[tuple[int, int]]) -> list[str]:
    state = list(items)
    for i, j in swaps:  # 1-indexed
        state[i - 1], state[j - 1] = state[j - 1], state[i - 1]
    return state


# ── Rendering ────────────────────────────────────────────────────────────────

def render_symbolic(items: list[str], swaps: list[tuple[int, int]]) -> tuple[str, str]:
    initial = "".join(items)
    ops = " ".join(f"{i}{j}" for i, j in swaps)
    x = f"{initial}: {ops} ="
    final = apply_swaps(items, swaps)
    y = "".join(final)
    return x, y


def render_code(items: list[str], swaps: list[tuple[int, int]]) -> tuple[str, str]:
    init_repr = "[" + ", ".join(f"'{c}'" for c in items) + "]"
    stmts = [f"lst = {init_repr}"]
    for i, j in swaps:
        stmts.append(f"lst[{i}], lst[{j}] = lst[{j}], lst[{i}]")
    stmts.append("lst =")
    x = "; ".join(stmts)
    final = apply_swaps(items, swaps)
    y = "[" + ", ".join(f"'{c}'" for c in final) + "]"
    return x, y


def render_natural(items: list[str], swaps: list[tuple[int, int]]) -> tuple[str, str]:
    boxes = ", ".join(f"box {c}" for c in items)
    intro = f"Start with {boxes}."
    swap_stmts = " ".join(f"Swap box {i} and box {j}." for i, j in swaps)
    x = f"{intro} {swap_stmts} What is the final order?"
    final = apply_swaps(items, swaps)
    y = "".join(final)
    return x, y


def render(items: list[str], swaps: list[tuple[int, int]], fmt: str) -> tuple[str, str]:
    if fmt == "symbolic":
        return render_symbolic(items, swaps)
    if fmt == "code":
        return render_code(items, swaps)
    return render_natural(items, swaps)


# ── Item generation ──────────────────────────────────────────────────────────

def generate_items(
    n: int,
    item_counts: list[int],
    lengths: list[int],
    formats: list[str],
    split: str,
    id_prefix: str,
    rng: random.Random,
) -> list[dict]:
    items_out = []
    while len(items_out) < n:
        n_items = rng.choice(item_counts)
        base = ITEMS_3 if n_items == 3 else ITEMS_5
        length = rng.choice(lengths)
        fmt = rng.choice(formats)

        swaps = [(rng.randint(1, n_items - 1), rng.randint(2, n_items)) for _ in range(length)]
        # ensure i < j for each swap
        swaps = [(min(a, b), max(a, b)) for a, b in swaps]
        # reject degenerate swaps (i == j)
        swaps = [(a, b) for a, b in swaps if a != b]
        if len(swaps) != length:
            continue

        x, y = render(base, swaps, fmt)
        idx = len(items_out)
        items_out.append(make_item(
            id=f"{id_prefix}_{idx:06d}",
            x=x,
            y=y,
            split=split,
            features={
                "item_count": n_items,
                "problem_length": length,
                "format": fmt,
            },
        ))
    return items_out


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/state_tracking")
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    rng = random.Random(args.seed)

    # ── state_tracking_by_length ────────────────────────────────────────────
    len_dir = out / "state_tracking_by_length"
    print("Generating state_tracking_by_length …")

    train = generate_items(args.n_train, [3, 5], [10], FORMATS, "train",
                           "st_len_train", rng)
    val = generate_items(args.n_val, [3, 5], [11, 12], FORMATS, "val",
                         "st_len_val", rng)
    test = generate_items(args.n_test, [3, 5], [13, 14], FORMATS, "test",
                          "st_len_test", rng)

    write_jsonl(len_dir / "train.jsonl", train)
    write_jsonl(len_dir / "val.jsonl", val)
    write_jsonl(len_dir / "test.jsonl", test)
    print(f"  train={len(train)}, val={len(val)}, test={len(test)}")

    # ── state_tracking_by_format ────────────────────────────────────────────
    fmt_dir = out / "state_tracking_by_format"
    print("Generating state_tracking_by_format …")

    all_lengths = list(range(5, 21))
    train = generate_items(args.n_train, [3, 5], all_lengths, ["symbolic"], "train",
                           "st_fmt_train", rng)
    val = generate_items(args.n_val, [3, 5], all_lengths, ["code"], "val",
                         "st_fmt_val", rng)
    test = generate_items(args.n_test, [3, 5], all_lengths, ["natural"], "test",
                          "st_fmt_test", rng)

    write_jsonl(fmt_dir / "train.jsonl", train)
    write_jsonl(fmt_dir / "val.jsonl", val)
    write_jsonl(fmt_dir / "test.jsonl", test)
    print(f"  train={len(train)}, val={len(val)}, test={len(test)}")

    print("Done.")


if __name__ == "__main__":
    main()
