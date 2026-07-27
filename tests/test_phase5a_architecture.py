from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from src.decision_intelligence_engine.baseline_training import (
    ModelRunConfig,
    _resolve_model_resolution,
    _resolve_preprocessing,
    build_transfer_model,
)
from src.decision_intelligence_engine.experiment_config import load_experiment_config
from src.decision_intelligence_engine.run_experiments import main as run_experiments_main


def _load_run_cfg(path: Path) -> ModelRunConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("model_input_resolution") is not None:
        raw["model_input_resolution"] = tuple(raw["model_input_resolution"])
    if raw.get("architecture_required_changes") is not None:
        raw["architecture_required_changes"] = tuple(raw["architecture_required_changes"])
    return ModelRunConfig(**raw)


def _toy_backbone(tf, input_shape: tuple[int, int, int]):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(4, 3, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.ReLU()(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def test_backbone_factory_supports_mobilenet_and_efficientnet(monkeypatch: pytest.MonkeyPatch) -> None:
    tf = pytest.importorskip("tensorflow")

    observed: list[str] = []

    def _stub_builder(backbone: str, input_shape: tuple[int, int, int], pretrained_weights: str):
        observed.append(backbone)
        return _toy_backbone(tf, input_shape)

    monkeypatch.setattr("src.decision_intelligence_engine.baseline_training._build_backbone", _stub_builder)

    cfg_mnet = _load_run_cfg(Path("configs/experiments/02_longer_frozen_training.json"))
    cfg_eff = _load_run_cfg(Path("configs/experiments/06_efficientnetb0_frozen_baseline.json"))

    build_transfer_model((32, 32, 3), cfg_mnet)
    build_transfer_model((32, 32, 3), cfg_eff)

    assert observed == ["MobileNetV2", "EfficientNetB0"]


def test_architecture_specific_preprocessing_and_resolution_defaults() -> None:
    cfg_mnet = _load_run_cfg(Path("configs/experiments/02_longer_frozen_training.json"))
    cfg_eff = _load_run_cfg(Path("configs/experiments/06_efficientnetb0_frozen_baseline.json"))

    assert _resolve_preprocessing(cfg_mnet) == "mobilenetv2_rescale_neg1_to_1"
    assert _resolve_preprocessing(cfg_eff) == "efficientnetb0_builtin_rescaling_with_input_scale_255"
    assert _resolve_model_resolution(cfg_mnet, (32, 32, 3)) == (32, 32)
    assert _resolve_model_resolution(cfg_eff, (32, 32, 3)) == (96, 96)


def test_invalid_model_input_resolution_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(Path("configs/experiments/06_efficientnetb0_frozen_baseline.json").read_text(encoding="utf-8"))
    payload["model_input_resolution"] = [0, 224]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_experiment_config(bad)


def test_runner_can_list_and_select_efficientnet_dry_run(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["run_experiments.py", "--list"])
    run_experiments_main()
    listed = json.loads(capsys.readouterr().out)
    ids = {row["experiment_id"] for row in listed}
    assert "efficientnetb0_control_frozen" in ids
    assert "mobilenetv2_96x96_control" in ids

    monkeypatch.setattr(sys, "argv", ["run_experiments.py", "--experiment", "efficientnetb0_control_frozen", "--dry-run"])
    run_experiments_main()
    dry = json.loads(capsys.readouterr().out)
    assert dry["run_count"] == 1
    assert dry["experiment_ids"] == ["efficientnetb0_control_frozen"]

    monkeypatch.setattr(sys, "argv", ["run_experiments.py", "--experiment", "mobilenetv2_96x96_control", "--dry-run"])
    run_experiments_main()
    dry_mobile = json.loads(capsys.readouterr().out)
    assert dry_mobile["run_count"] == 1
    assert dry_mobile["experiment_ids"] == ["mobilenetv2_96x96_control"]


def test_phase5a_tagging_fields_present() -> None:
    cfg = _load_run_cfg(Path("configs/experiments/06_efficientnetb0_frozen_baseline.json"))
    assert cfg.mlflow_tags["phase"] == "5A"
    assert cfg.mlflow_tags["experiment_category"] == "architecture_screening"
    assert cfg.mlflow_tags["changed_variable"] == "architecture"
    assert cfg.mlflow_tags["comparison_target"] == "mnetv2_longer_frozen_epochs"

    mobile_cfg = _load_run_cfg(Path("configs/experiments/07_mobilenetv2_96x96_control.json"))
    assert mobile_cfg.mlflow_tags["phase"] == "5A"
    assert mobile_cfg.mlflow_tags["experiment_category"] == "resolution_control"
    assert mobile_cfg.mlflow_tags["changed_variable"] == "model_input_resolution"
    assert mobile_cfg.preprocessing_function == "mobilenetv2_rescale_neg1_to_1"
