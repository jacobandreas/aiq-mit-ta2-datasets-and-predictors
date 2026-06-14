"""HELM scenario for arithmetic_parametric dataset."""

import os
from pathlib import Path
from typing import List

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG, TEST_SPLIT, TRAIN_SPLIT, Input, Instance, Output, Reference, Scenario,
)
from helm.common.general import ensure_directory_exists

from datasets.utils import read_jsonl

_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "arithmetic_parametric"


class ArithmeticParametricScenario(Scenario):
    name = "arithmetic_parametric"
    description = "Arithmetic parametric dataset"
    tags = ["reasoning", "arithmetic"]

    def __init__(self, split_config: str = "arithmetic_by_format"):
        super().__init__()
        self.split_config = split_config

    def get_instances(self, output_path: str) -> List[Instance]:
        ensure_directory_exists(output_path)
        instances = []
        config_dir = _DATA_ROOT / self.split_config
        split_map = {"train": TRAIN_SPLIT, "test": TEST_SPLIT}
        for fname in ("train.jsonl", "test.jsonl"):
            path = config_dir / fname
            if not path.exists():
                continue
            for item in read_jsonl(path):
                helm_split = split_map.get(item["split"], TEST_SPLIT)
                instances.append(Instance(
                    input=Input(text=item["x"]),
                    references=[Reference(Output(text=item["y"]), tags=[CORRECT_TAG])],
                    split=helm_split,
                    id=item["id"],
                ))
        return instances
