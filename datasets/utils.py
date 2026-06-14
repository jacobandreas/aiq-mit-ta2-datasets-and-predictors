from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Union


def write_jsonl(path: Union[Path, str], items: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def read_jsonl(path: Union[Path, str]) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def make_item(
    id: str,
    x: str,
    y: str,
    split: str,
    features: dict,
) -> dict:
    return {"id": id, "x": x, "y": y, "split": split, "features": features}
