from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.decision_intelligence_engine.pipeline_config import DataPipelineConfig, load_data_pipeline_config


def test_load_default_config_when_file_missing(tmp_path: Path) -> None:
    cfg = load_data_pipeline_config(tmp_path)
    assert cfg == DataPipelineConfig()


def test_load_json_config_and_coerce_image_size(tmp_path: Path) -> None:
    cfg_path = tmp_path / "custom.json"
    cfg_path.write_text(
        json.dumps({
            "batch_size": 16,
            "validation_split": 0.25,
            "image_size": [32, 32],
            "random_seed": 99,
        }),
        encoding="utf-8",
    )
    cfg = load_data_pipeline_config(tmp_path, cfg_path)
    assert cfg.batch_size == 16
    assert cfg.validation_split == 0.25
    assert cfg.image_size == (32, 32)
    assert cfg.random_seed == 99


def test_config_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        DataPipelineConfig(validation_split=1.0)
    with pytest.raises(ValueError):
        DataPipelineConfig(batch_size=0)
    with pytest.raises(ValueError):
        DataPipelineConfig(image_size=(32, 0))
