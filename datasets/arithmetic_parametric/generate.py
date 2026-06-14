"""Generate arithmetic_parametric dataset.
# ruff: noqa: E501

Two split configurations:
  arithmetic_by_format  -- full operator set, format varies by split
  arithmetic_by_skill   -- symbolic format, operator set varies by split

Usage:
    python -m datasets.arithmetic_parametric.generate \\
        --output-dir data/arithmetic_parametric \\
        --n-train 2000 --n-val 500 --n-test 1000 --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
from fractions import Fraction
from pathlib import Path
from typing import Optional

try:
    from num2words import num2words
except ImportError:
    num2words = None

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets.utils import make_item, write_jsonl

# ── Chinese number converter ────────────────────────────────────────────────

_CN_DIGITS = "零一二三四五六七八九"
_CN_UNITS = ["", "十", "百", "千", "万", "十万", "百万", "千万", "亿"]


def _to_chinese(n: int) -> str:
    if n < 0:
        return "负" + _to_chinese(-n)
    if n == 0:
        return "零"
    s = str(n)
    length = len(s)
    result = []
    for i, ch in enumerate(s):
        d = int(ch)
        unit_idx = length - 1 - i
        if d != 0:
            result.append(_CN_DIGITS[d] + _CN_UNITS[unit_idx])
        else:
            if result and result[-1] != "零":
                result.append("零")
    out = "".join(result).rstrip("零")
    # 一十X → 十X at the start
    if out.startswith("一十"):
        out = out[1:]
    return out


# ── Expression generation ────────────────────────────────────────────────────

ALL_OPS = ["+", "-", "*", "/"]


def _evaluate(operands: list[int], ops: list[str]) -> Optional[Fraction]:
    """Evaluate left-to-right with standard * / before + -. Returns None on div-by-zero or non-integer."""
    vals = [Fraction(x) for x in operands]
    op_list = list(ops)

    # First pass: * and /
    i = 0
    while i < len(op_list):
        if op_list[i] in ("*", "/"):
            if op_list[i] == "/" and vals[i + 1] == 0:
                return None
            res = vals[i] * vals[i + 1] if op_list[i] == "*" else vals[i] / vals[i + 1]
            vals = vals[:i] + [res] + vals[i + 2:]
            op_list = op_list[:i] + op_list[i + 1:]
        else:
            i += 1

    # Second pass: + and -
    result = vals[0]
    for op, v in zip(op_list, vals[1:]):
        result = result + v if op == "+" else result - v

    if result.denominator != 1:
        return None
    return result


def _sample_expression(operators: list[str], rng: random.Random) -> Optional[tuple]:
    """Sample operands/ops and evaluate; return (operands, ops, value) or None if rejected."""
    n_operands = rng.randint(2, 5)
    operands = [rng.randint(0, 9) for _ in range(n_operands)]
    ops = [rng.choice(operators) for _ in range(n_operands - 1)]
    result = _evaluate(operands, ops)
    if result is None:
        return None
    return operands, ops, int(result)


# ── Format renderers ─────────────────────────────────────────────────────────

_OP_WORDS: dict[str, dict[str, str]] = {
    "english":  {"+": "plus",         "-": "minus",   "*": "times",           "/": "divided by", "=": "equals"},
    "spanish":  {"+": "más",           "-": "menos",   "*": "por",             "/": "dividido por", "=": "es igual a"},
    "italian":  {"+": "più",           "-": "meno",    "*": "per",             "/": "diviso per", "=": "è uguale a"},
    "chinese":  {"+": "加",            "-": "减",       "*": "乘",              "/": "除以",       "=": "等于"},
}

_NUM2WORDS_LANG = {"english": "en", "spanish": "es", "italian": "it"}


def _num_to_words(n: int, lang: str) -> str:
    if lang == "chinese":
        return _to_chinese(n)
    if num2words is None:
        raise RuntimeError("num2words not installed")
    return num2words(n, lang=_NUM2WORDS_LANG[lang])


def render_item(operands: list[int], ops: list[str], value: int, fmt: str) -> tuple[str, str]:
    """Return (x, y) for the given format."""
    if fmt == "symbolic":
        x = " ".join(
            [str(operands[0])]
            + [f"{op} {operands[i+1]}" for i, op in enumerate(ops)]
            + ["="]
        )
        y = str(value)
        return x, y

    words = _OP_WORDS[fmt]
    eq = words["="]
    op_words = [words[op] for op in ops]
    x_parts = [_num_to_words(operands[0], fmt)]
    for op_w, operand in zip(op_words, operands[1:]):
        x_parts += [op_w, _num_to_words(operand, fmt)]
    x_parts.append(eq)
    x = " ".join(x_parts)
    y = _num_to_words(value, fmt)
    return x, y


# ── Split generators ─────────────────────────────────────────────────────────

def generate_items(
    n: int,
    operators: list[str],
    formats: list[str],
    split: str,
    id_prefix: str,
    id_offset: int = 0,
    operator_pools: Optional[list] = None,
    rng: Optional[random.Random] = None,
) -> list[dict]:
    """
    Generate n items with the given operators/formats.

    If operator_pools is provided, each item samples an operator pool at random
    and records it in features.operator_pool.
    """
    if rng is None:
        rng = random.Random()
    items = []
    attempts = 0
    while len(items) < n:
        attempts += 1
        if attempts > n * 100:
            raise RuntimeError(f"Too many rejections generating {n} items")

        if operator_pools:
            pool_idx = rng.randrange(len(operator_pools))
            ops_to_use = operator_pools[pool_idx]
        else:
            ops_to_use = operators
            pool_idx = None

        result = _sample_expression(ops_to_use, rng)
        if result is None:
            continue
        operands, expr_ops, value = result

        fmt = rng.choice(formats)
        x, y = render_item(operands, expr_ops, value, fmt)

        idx = id_offset + len(items)
        features: dict = {
            "operators": sorted(set(expr_ops)),
            "format": fmt,
            "n_operands": len(operands),
            "operands": operands,
        }
        if pool_idx is not None:
            features["operator_pool"] = ops_to_use

        items.append(make_item(
            id=f"{id_prefix}_{idx:06d}",
            x=x,
            y=y,
            split=split,
            features=features,
        ))
    return items


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/arithmetic_parametric")
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    rng = random.Random(args.seed)

    # ── arithmetic_by_format ────────────────────────────────────────────────
    fmt_dir = out / "arithmetic_by_format"

    print("Generating arithmetic_by_format …")
    train_items = generate_items(
        args.n_train, ALL_OPS, ["symbolic", "english"], "train",
        "arith_fmt_train", rng=rng,
    )
    val_items = generate_items(
        args.n_val, ALL_OPS, ["spanish"], "val",
        "arith_fmt_val", rng=rng,
    )
    test_items = generate_items(
        args.n_test, ALL_OPS, ["italian", "chinese"], "test",
        "arith_fmt_test", rng=rng,
    )
    write_jsonl(fmt_dir / "train.jsonl", train_items)
    write_jsonl(fmt_dir / "val.jsonl", val_items)
    write_jsonl(fmt_dir / "test.jsonl", test_items)
    print(f"  train={len(train_items)}, val={len(val_items)}, test={len(test_items)}")

    # ── arithmetic_by_skill ─────────────────────────────────────────────────
    skill_dir = out / "arithmetic_by_skill"

    print("Generating arithmetic_by_skill …")
    train_items = generate_items(
        args.n_train, [], ["symbolic"], "train",
        "arith_skill_train",
        operator_pools=[["+", "-"], ["+", "*"]],
        rng=rng,
    )
    val_items = generate_items(
        args.n_val, ["+", "-", "*"], ["symbolic"], "val",
        "arith_skill_val", rng=rng,
    )
    test_items = generate_items(
        args.n_test, ["-", "*"], ["symbolic"], "test",
        "arith_skill_test", rng=rng,
    )
    write_jsonl(skill_dir / "train.jsonl", train_items)
    write_jsonl(skill_dir / "val.jsonl", val_items)
    write_jsonl(skill_dir / "test.jsonl", test_items)
    print(f"  train={len(train_items)}, val={len(val_items)}, test={len(test_items)}")

    print("Done.")


if __name__ == "__main__":
    main()
