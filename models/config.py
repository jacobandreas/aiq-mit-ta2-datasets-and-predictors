"""Model and run configuration dataclasses with YAML loading."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# ── Sub-configs ───────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    lr: float = 5e-5
    batch_size: int = 32
    max_epochs: int = 20
    eval_steps: int = 200
    val_eval_items: int = 200
    patience: int = 5
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01


@dataclass
class ModelConfig:
    """Per-family model configuration (loaded from models/configs/<family>.yaml)."""

    family: str = ""
    model_name_or_path: str = ""
    tokenizer_name_or_path: str = ""
    zero_shot: bool = False          # skip training entirely
    max_new_tokens: int = 32
    eval_batch_size: int = 16
    dtype: str = "float16"           # float32 | float16 | bfloat16
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "ModelConfig":
        if not _YAML_AVAILABLE:
            raise RuntimeError("PyYAML not installed; run: pip install pyyaml")
        with open(path) as f:
            raw = yaml.safe_load(f)
        training_raw = raw.pop("training", {})
        training = TrainingConfig(**{k: v for k, v in training_raw.items()
                                     if k in TrainingConfig.__dataclass_fields__})
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(training=training, **known)

    def resolve_model_path(self, models_root: Optional[str]) -> str:
        """Return local path if it exists under models_root, else the HF Hub ID."""
        if models_root:
            candidate = Path(models_root) / Path(self.model_name_or_path).name
            if candidate.exists():
                return str(candidate)
        return self.model_name_or_path

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class RunConfig:
    """Fully resolved configuration for a single (dataset, family, seed) run."""

    dataset_dir: str = ""
    family: str = ""
    seed: int = 0
    output_dir: str = ""             # runs/<dataset>/<family>/seed<N>/
    models_root: str = ""
    force: bool = False

    # Resolved fields (filled in by orchestrator)
    model_config: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ── Config discovery ──────────────────────────────────────────────────────────

_CONFIG_DIR = Path(__file__).parent / "configs"


def load_model_config(family: str) -> ModelConfig:
    path = _CONFIG_DIR / f"{family}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No config found for family '{family}' at {path}")
    return ModelConfig.from_yaml(path)


def list_families() -> list:
    return sorted(p.stem for p in _CONFIG_DIR.glob("*.yaml"))
