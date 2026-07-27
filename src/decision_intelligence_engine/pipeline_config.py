"""Configuration utilities for the CIFAR-10 data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataPipelineConfig:
    """Centralized configuration for CIFAR-10 data loading and preprocessing."""

    data_root: str = "data/raw/cifar10"
    train_subdir: str = "train"
    test_subdir: str = "test"
    image_size: tuple[int, int] = (32, 32)
    batch_size: int = 128
    validation_split: float = 0.2
    random_seed: int = 42
    shuffle_buffer_size: int = 4096
    num_parallel_calls: int = -1
    cache_train: bool = False
    cache_eval: bool = True
    prefetch: bool = True
    augmentation_enabled: bool = True
    augmentation_padding: int = 4
    augmentation_horizontal_flip: bool = True

    def __post_init__(self) -> None:
        if len(self.image_size) != 2 or any(int(v) <= 0 for v in self.image_size):
            raise ValueError("image_size must contain two positive integers.")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be > 0.")
        if not 0.0 < float(self.validation_split) < 1.0:
            raise ValueError("validation_split must be between 0 and 1 (exclusive).")
        if int(self.shuffle_buffer_size) <= 0:
            raise ValueError("shuffle_buffer_size must be > 0.")
        if int(self.augmentation_padding) < 0:
            raise ValueError("augmentation_padding must be >= 0.")

    def train_dir(self, project_root: Path) -> Path:
        return project_root / self.data_root / self.train_subdir

    def test_dir(self, project_root: Path) -> Path:
        return project_root / self.data_root / self.test_subdir


def _default_config_path(project_root: Path) -> Path:
    return project_root / "configs" / "data_pipeline.json"


def _to_int_pair(value: Any, field_name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must be a list or tuple of length 2.")
    return int(value[0]), int(value[1])


def load_data_pipeline_config(project_root: Path, config_path: Path | None = None) -> DataPipelineConfig:
    """Load data pipeline configuration from JSON or use dataclass defaults.

    Args:
        project_root: Root of the repository.
        config_path: Optional explicit path to a JSON config file.

    Returns:
        DataPipelineConfig with overrides applied.
    """

    resolved_path = config_path or _default_config_path(project_root)
    if not resolved_path.exists():
        return DataPipelineConfig()

    raw = json.loads(resolved_path.read_text(encoding="utf-8"))
    data = dict(raw)

    if "image_size" in data:
        data["image_size"] = _to_int_pair(data["image_size"], "image_size")

    return DataPipelineConfig(**data)
