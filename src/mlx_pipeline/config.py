"""Pipeline configuration loaded from a YAML file in configs/."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

FLAVORS = ("llm", "vlm", "embedding")


@dataclass
class PipelineConfig:
    flavor: str
    model: str
    data: dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)
    eval: dict = field(default_factory=dict)
    gate: dict = field(default_factory=dict)
    publish: dict = field(default_factory=dict)
    sweep: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.flavor not in FLAVORS:
            raise ValueError(f"flavor must be one of {FLAVORS}, got {self.flavor!r}")


def load_config(path: str | Path) -> PipelineConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return PipelineConfig(**raw)
