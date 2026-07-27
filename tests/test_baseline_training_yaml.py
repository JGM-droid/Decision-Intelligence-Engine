from __future__ import annotations

from pathlib import Path

import pytest

from src.decision_intelligence_engine.baseline_training import BaselineTrainingConfig, load_baseline_training_config


def _write_yaml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_valid_yaml_loads_authoritative_config(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "configs" / "baseline_training.yaml",
        """model_name: mobilenetv2_frozen_baseline
num_classes: 10
dropout_rate: 0.2
learning_rate: 0.0003
epochs: 1
steps_per_epoch: 100
validation_steps: null
evaluate_train_steps: null
evaluate_val_steps: null
evaluate_test_steps: null
report_test_steps: null
model_output_dir: models
report_output_dir: reports
random_seed: 42
""",
    )

    cfg = load_baseline_training_config(tmp_path)
    assert cfg == BaselineTrainingConfig()


def test_malformed_yaml_fails(tmp_path: Path) -> None:
    _write_yaml(tmp_path / "configs" / "baseline_training.yaml", "model_name: [unterminated")

    with pytest.raises(ValueError, match="Malformed YAML baseline config"):
        load_baseline_training_config(tmp_path)


def test_missing_required_fields_fail(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "configs" / "baseline_training.yaml",
        """model_name: mobilenetv2_frozen_baseline
dropout_rate: 0.2
learning_rate: 0.0003
epochs: 1
steps_per_epoch: 100
validation_steps: null
evaluate_train_steps: null
evaluate_val_steps: null
evaluate_test_steps: null
report_test_steps: null
model_output_dir: models
report_output_dir: reports
random_seed: 42
""",
    )

    with pytest.raises(ValueError, match="Missing required baseline training config fields"):
        load_baseline_training_config(tmp_path)


def test_invalid_types_fail(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "configs" / "baseline_training.yaml",
        """model_name: mobilenetv2_frozen_baseline
num_classes: 10
dropout_rate: 0.2
learning_rate: 0.0003
epochs: one
steps_per_epoch: 100
validation_steps: null
evaluate_train_steps: null
evaluate_val_steps: null
evaluate_test_steps: null
report_test_steps: null
model_output_dir: models
report_output_dir: reports
random_seed: 42
""",
    )

    with pytest.raises(ValueError, match="epochs must be an integer"):
        load_baseline_training_config(tmp_path)


def test_unknown_fields_fail(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "configs" / "baseline_training.yaml",
        """model_name: mobilenetv2_frozen_baseline
num_classes: 10
dropout_rate: 0.2
learning_rate: 0.0003
epochs: 1
steps_per_epoch: 100
validation_steps: null
evaluate_train_steps: null
evaluate_val_steps: null
evaluate_test_steps: null
report_test_steps: null
model_output_dir: models
report_output_dir: reports
random_seed: 42
unexpected_field: true
""",
    )

    with pytest.raises(ValueError, match="Unsupported baseline training config fields"):
        load_baseline_training_config(tmp_path)
