from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.decision_intelligence_engine.experiment_config import (
    get_experiment_by_id,
    load_experiment_configs,
    validate_experiment_matrix,
)


def test_matrix_contains_required_experiment_ids(project_root: Path | None = None) -> None:
    root = Path.cwd() if project_root is None else project_root
    configs = load_experiment_configs(root)
    observed = {cfg.experiment_id for cfg in configs}
    expected = {
        "mnetv2_control_frozen",
        "mnetv2_longer_frozen_epochs",
        "mnetv2_lower_lr_frozen",
        "mnetv2_higher_dropout_frozen",
        "mnetv2_partial_finetune_tail",
    }
    assert observed == expected


def test_matrix_validation_passes_for_repo_configs() -> None:
    result = validate_experiment_matrix(Path.cwd())
    assert result.control_experiment_id == "mnetv2_control_frozen"
    assert len(result.experiment_ids) == 5


def test_single_variable_experiment_rejects_extra_change(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / "configs" / "experiments").mkdir(parents=True, exist_ok=True)

    baseline = {
        "model_name": "mobilenetv2_frozen_baseline",
        "num_classes": 10,
        "dropout_rate": 0.2,
        "learning_rate": 0.0003,
        "epochs": 1,
        "steps_per_epoch": 100,
        "validation_steps": None,
        "evaluate_train_steps": None,
        "evaluate_val_steps": None,
        "evaluate_test_steps": None,
        "report_test_steps": None,
        "model_output_dir": "models",
        "report_output_dir": "reports",
        "random_seed": 42,
    }
    data = {
        "data_root": "data/raw/cifar10",
        "train_subdir": "train",
        "test_subdir": "test",
        "image_size": [32, 32],
        "batch_size": 128,
        "validation_split": 0.2,
        "random_seed": 42,
        "shuffle_buffer_size": 4096,
        "num_parallel_calls": -1,
        "cache_train": False,
        "cache_eval": True,
        "prefetch": True,
        "augmentation_enabled": True,
        "augmentation_padding": 4,
        "augmentation_horizontal_flip": True,
    }
    (project_root / "configs" / "baseline_training.json").write_text(json.dumps(baseline), encoding="utf-8")
    (project_root / "configs" / "data_pipeline.json").write_text(json.dumps(data), encoding="utf-8")

    source_root = Path.cwd() / "configs" / "experiments"
    for src in sorted(source_root.glob("*.json")):
        cfg = json.loads(src.read_text(encoding="utf-8"))
        if cfg["experiment_id"] == "mnetv2_lower_lr_frozen":
            cfg["dropout_rate"] = 0.3
        (project_root / "configs" / "experiments" / src.name).write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_experiment_matrix(project_root)


def test_fine_tune_exception_is_explicit() -> None:
    cfg = get_experiment_by_id(load_experiment_configs(Path.cwd()), "mnetv2_partial_finetune_tail")
    assert cfg.changed_variable == "backbone_trainability+learning_rate"
    assert cfg.freeze_backbone is False
    assert cfg.train_batch_norm is False
