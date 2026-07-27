"""Configuration utilities for MLflow experiment tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MlflowConfig:
    """Centralized configuration for local MLflow tracking."""

    enabled: bool = True
    tracking_dir: str = "mlruns"
    experiment_name: str = "decision_intelligence_engine"
    run_name_prefix: str = "mobilenetv2_frozen_baseline"
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tracking_dir.strip():
            raise ValueError("tracking_dir must not be empty.")
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must not be empty.")
        if not self.run_name_prefix.strip():
            raise ValueError("run_name_prefix must not be empty.")

    def tracking_uri(self, project_root: Path) -> str:
        tracking_path = (project_root / self.tracking_dir).resolve()
        return tracking_path.as_uri()

    def tracking_dir_path(self, project_root: Path) -> Path:
        return project_root / self.tracking_dir


def _default_config_path(project_root: Path) -> Path:
    return project_root / "configs" / "mlflow.json"


def _to_str_dict(value: Any, field_name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    return {str(key): str(item) for key, item in value.items()}


def load_mlflow_config(project_root: Path, config_path: Path | None = None) -> MlflowConfig:
    """Load MLflow configuration from JSON or use dataclass defaults."""

    resolved_path = config_path or _default_config_path(project_root)
    if not resolved_path.exists():
        return MlflowConfig()

    raw = json.loads(resolved_path.read_text(encoding="utf-8"))
    data = dict(raw)

    if "tags" in data:
        data["tags"] = _to_str_dict(data["tags"], "tags")

    return MlflowConfig(**data)