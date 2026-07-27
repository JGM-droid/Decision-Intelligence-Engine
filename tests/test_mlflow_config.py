from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.decision_intelligence_engine.mlflow_config import MlflowConfig, load_mlflow_config


def test_load_default_mlflow_config_when_file_missing(tmp_path: Path) -> None:
    cfg = load_mlflow_config(tmp_path)
    assert cfg == MlflowConfig()


def test_load_mlflow_config_and_coerce_tags(tmp_path: Path) -> None:
    cfg_path = tmp_path / "mlflow.json"
    cfg_path.write_text(
        json.dumps(
            {
                "enabled": False,
                "tracking_dir": "tracking",
                "experiment_name": "demo",
                "run_name_prefix": "baseline",
                "tags": {"phase": 4, "active": True},
            }
        ),
        encoding="utf-8",
    )

    cfg = load_mlflow_config(tmp_path, cfg_path)
    assert cfg.enabled is False
    assert cfg.tracking_dir == "tracking"
    assert cfg.experiment_name == "demo"
    assert cfg.run_name_prefix == "baseline"
    assert cfg.tags == {"phase": "4", "active": "True"}


def test_mlflow_config_validation_rejects_empty_values() -> None:
    with pytest.raises(ValueError):
        MlflowConfig(tracking_dir="")
    with pytest.raises(ValueError):
        MlflowConfig(experiment_name="")
    with pytest.raises(ValueError):
        MlflowConfig(run_name_prefix="")