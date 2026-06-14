"""HELM scenario for arithmetic_fixed dataset."""

from pathlib import Path
from typing import List

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG, TEST_SPLIT, TRAIN_SPLIT, Input, Instance, Output, Reference, Scenario,
)
from helm.common.general import ensure_directory_exists

from datasets.utils import read_jsonl

_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "arithmetic_fixed"
_SPLIT_MAP = {"train": TRAIN_SPLIT, "val": TRAIN_SPLIT, "test": TEST_SPLIT}


class ArithmeticFixedScenario(Scenario):
    name = "arithmetic_fixed"
    description = "Fixed arithmetic dataset (from arithmetic-inconsistencies)"
    tags = ["reasoning", "arithmetic", "multilingual"]

    def __init__(self, split_config: str = "cross_lingual"):
        super().__init__()
        self.split_config = split_config

    def get_instances(self, output_path: str) -> List[Instance]:
        ensure_directory_exists(output_path)
        instances = []
        config_dir = _DATA_ROOT / self.split_config
        for fname in ("train.jsonl", "val.jsonl", "test.jsonl"):
            path = config_dir / fname
            if not path.exists():
                continue
            for item in read_jsonl(path):
                helm_split = _SPLIT_MAP.get(item["split"], TEST_SPLIT)
                instances.append(Instance(
                    input=Input(text=item["x"]),
                    references=[Reference(Output(text=item["y"]), tags=[CORRECT_TAG])],
                    split=helm_split,
                    id=item["id"],
                ))
        return instances
