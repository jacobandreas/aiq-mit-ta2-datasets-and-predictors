"""HELM scenario for state_tracking dataset."""

from pathlib import Path
from typing import List

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG, TEST_SPLIT, TRAIN_SPLIT, Input, Instance, Output, Reference, Scenario,
)
from helm.common.general import ensure_directory_exists

from datasets.utils import read_jsonl

_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "state_tracking"


class StateTrackingScenario(Scenario):
    name = "state_tracking"
    description = "State tracking via swap sequences"
    tags = ["reasoning", "state_tracking"]

    def __init__(self, split_config: str = "state_tracking_by_length"):
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
