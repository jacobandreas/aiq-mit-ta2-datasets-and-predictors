"""HELM scenario for world_modeling (EWOK-CORE) dataset."""

from pathlib import Path
from typing import List

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG, TEST_SPLIT, TRAIN_SPLIT, Input, Instance, Output, Reference, Scenario,
)
from helm.common.general import ensure_directory_exists

from datasets.utils import read_jsonl

_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "world_modeling"


class WorldModelingScenario(Scenario):
    name = "world_modeling"
    description = "EWOK-CORE world modeling dataset"
    tags = ["knowledge", "reasoning"]

    def __init__(self, domain: str | None = None):
        super().__init__()
        self.domain = domain

    def get_instances(self, output_path: str) -> List[Instance]:
        ensure_directory_exists(output_path)
        instances = []
        if self.domain:
            dirs = [_DATA_ROOT / self.domain.replace(" ", "_")]
        else:
            dirs = [d for d in _DATA_ROOT.iterdir() if d.is_dir()]

        for d in dirs:
            for fname, helm_split in (("train.jsonl", TRAIN_SPLIT), ("test.jsonl", TEST_SPLIT)):
                path = d / fname
                if not path.exists():
                    continue
                for item in read_jsonl(path):
                    instances.append(Instance(
                        input=Input(text=item["x"]),
                        references=[Reference(Output(text=item["y"]), tags=[CORRECT_TAG])],
                        split=helm_split,
                        id=item["id"],
                    ))
        return instances
