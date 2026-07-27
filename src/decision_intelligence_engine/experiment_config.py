"""Configuration loading and validation for controlled experiment runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import yaml

from .pipeline_config import DataPipelineConfig, load_data_pipeline_config


EXPERIMENT_DIR = Path("configs") / "experiments"
FINE_TUNE_EXPERIMENT_ID = "mnetv2_partial_finetune_tail"
ARCH_SCREENING_EXPERIMENT_ID = "efficientnetb0_control_frozen"
RESOLUTION_CONTROL_EXPERIMENT_ID = "mobilenetv2_96x96_control"

PHASE4B_REQUIRED_IDS = {
    "mnetv2_control_frozen",
    "mnetv2_longer_frozen_epochs",
    "mnetv2_lower_lr_frozen",
    "mnetv2_higher_dropout_frozen",
    "mnetv2_partial_finetune_tail",
}

SUPPORTED_BACKBONES = {"MobileNetV2", "EfficientNetB0"}
SUPPORTED_PREPROCESSING = {
    "MobileNetV2": {"mobilenetv2_rescale_neg1_to_1"},
    "EfficientNetB0": {"efficientnetb0_builtin_rescaling_with_input_scale_255"},
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Controlled experiment configuration for approved matrix/screening runs."""

    order: int
    experiment_id: str
    experiment_name: str
    experiment_category: str
    is_control: bool
    changed_variable: str
    control_value: str
    new_value: str
    rationale: str
    model_family: str
    backbone: str
    pretrained_weights: str
    num_classes: int
    freeze_backbone: bool
    unfreeze_last_n_layers: int | None
    unfreeze_last_fraction: float | None
    train_batch_norm: bool
    batch_size: int
    epochs: int
    steps_per_epoch: int | None
    validation_steps: int | None
    evaluate_train_steps: int | None
    evaluate_val_steps: int | None
    evaluate_test_steps: int | None
    report_test_steps: int | None
    learning_rate: float
    dropout_rate: float
    optimizer: str
    loss: str
    random_seed: int
    augmentation_enabled: bool
    data_split_strategy: str
    mlflow_tags: dict[str, str] = field(default_factory=dict)
    model_input_resolution: tuple[int, int] | None = None
    preprocessing_function: str | None = None
    architecture_required_changes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.order <= 0:
            raise ValueError("order must be > 0")
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must not be empty")
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must not be empty")
        if self.model_family not in SUPPORTED_BACKBONES:
            raise ValueError(f"model_family must be one of {sorted(SUPPORTED_BACKBONES)}")
        if self.backbone not in SUPPORTED_BACKBONES:
            raise ValueError(f"backbone must be one of {sorted(SUPPORTED_BACKBONES)}")
        if self.model_family != self.backbone:
            raise ValueError("model_family and backbone must match")
        if self.pretrained_weights != "imagenet":
            raise ValueError("pretrained_weights must be imagenet")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be > 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1)")
        if self.optimizer != "Adam":
            raise ValueError("optimizer must be Adam")
        if self.loss != "SparseCategoricalCrossentropy":
            raise ValueError("loss must be SparseCategoricalCrossentropy")
        if self.data_split_strategy != "deterministic_stratified_file_manifest":
            raise ValueError("data_split_strategy must be deterministic_stratified_file_manifest")

        model_input = self.model_input_resolution or ((96, 96) if self.backbone == "EfficientNetB0" else (32, 32))
        if len(model_input) != 2 or int(model_input[0]) <= 0 or int(model_input[1]) <= 0:
            raise ValueError("model_input_resolution must contain two positive integers")
        object.__setattr__(self, "model_input_resolution", (int(model_input[0]), int(model_input[1])))

        preprocessing = self.preprocessing_function or (
            "efficientnetb0_builtin_rescaling_with_input_scale_255"
            if self.backbone == "EfficientNetB0"
            else "mobilenetv2_rescale_neg1_to_1"
        )
        if preprocessing not in SUPPORTED_PREPROCESSING[self.backbone]:
            raise ValueError(
                f"preprocessing_function={preprocessing} is not supported for backbone={self.backbone}"
            )
        object.__setattr__(self, "preprocessing_function", preprocessing)

        arch_changes = tuple(str(item) for item in (self.architecture_required_changes or ()))
        object.__setattr__(self, "architecture_required_changes", arch_changes)
        if self.freeze_backbone:
            if (self.unfreeze_last_n_layers or 0) != 0:
                raise ValueError("frozen backbone experiments must set unfreeze_last_n_layers to 0")
            if self.unfreeze_last_fraction not in (None, 0.0):
                raise ValueError("frozen backbone experiments must not set unfreeze_last_fraction")
        else:
            has_n = self.unfreeze_last_n_layers is not None and self.unfreeze_last_n_layers > 0
            has_fraction = self.unfreeze_last_fraction is not None and self.unfreeze_last_fraction > 0
            if has_n == has_fraction:
                raise ValueError("unfrozen experiments must set exactly one of unfreeze_last_n_layers or unfreeze_last_fraction")


@dataclass(frozen=True)
class MatrixValidationResult:
    """Validation output for the controlled matrix."""

    experiment_ids: tuple[str, ...]
    control_experiment_id: str
    validation_messages: tuple[str, ...]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_experiment_config(path: Path) -> ExperimentConfig:
    data = _load_json(path)
    tags = {str(k): str(v) for k, v in data.get("mlflow_tags", {}).items()}
    data["mlflow_tags"] = tags
    if "model_input_resolution" in data and data["model_input_resolution"] is not None:
        data["model_input_resolution"] = tuple(int(v) for v in data["model_input_resolution"])
    if "architecture_required_changes" in data and data["architecture_required_changes"] is not None:
        data["architecture_required_changes"] = tuple(str(v) for v in data["architecture_required_changes"])
    return ExperimentConfig(**data)


def load_experiment_configs(project_root: Path) -> list[ExperimentConfig]:
    exp_dir = project_root / EXPERIMENT_DIR
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {exp_dir}")
    configs = [load_experiment_config(path) for path in sorted(exp_dir.glob("*.json"))]
    if not configs:
        raise ValueError(f"No experiment configs found in {exp_dir}")
    return sorted(configs, key=lambda cfg: cfg.order)


def load_experiment_configs_with_paths(project_root: Path) -> list[tuple[ExperimentConfig, Path]]:
    """Load experiment configs with source file paths."""

    exp_dir = project_root / EXPERIMENT_DIR
    pairs = [(load_experiment_config(path), path) for path in sorted(exp_dir.glob("*.json"))]
    return sorted(pairs, key=lambda pair: pair[0].order)


def get_experiment_by_id(configs: list[ExperimentConfig], experiment_id: str) -> ExperimentConfig:
    for cfg in configs:
        if cfg.experiment_id == experiment_id:
            return cfg
    raise KeyError(f"Experiment ID not found: {experiment_id}")


def _load_baseline_training_raw(project_root: Path) -> dict[str, Any]:
    yaml_path = project_root / "configs" / "baseline_training.yaml"
    json_path = project_root / "configs" / "baseline_training.json"
    if yaml_path.exists():
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Baseline training YAML must contain a mapping: {yaml_path}")
        return dict(data)
    return json.loads(json_path.read_text(encoding="utf-8"))


def _baseline_control_keys(train_raw: dict[str, Any], data_cfg: DataPipelineConfig) -> dict[str, Any]:
    return {
        "num_classes": int(train_raw["num_classes"]),
        "batch_size": data_cfg.batch_size,
        "epochs": int(train_raw["epochs"]),
        "steps_per_epoch": train_raw["steps_per_epoch"],
        "validation_steps": train_raw["validation_steps"],
        "evaluate_train_steps": train_raw["evaluate_train_steps"],
        "evaluate_val_steps": train_raw["evaluate_val_steps"],
        "evaluate_test_steps": train_raw["evaluate_test_steps"],
        "report_test_steps": train_raw["report_test_steps"],
        "learning_rate": float(train_raw["learning_rate"]),
        "dropout_rate": float(train_raw["dropout_rate"]),
        "random_seed": int(train_raw["random_seed"]),
        "augmentation_enabled": data_cfg.augmentation_enabled,
    }


def _config_control_keys(exp_cfg: ExperimentConfig) -> dict[str, Any]:
    return {
        "num_classes": exp_cfg.num_classes,
        "batch_size": exp_cfg.batch_size,
        "epochs": exp_cfg.epochs,
        "steps_per_epoch": exp_cfg.steps_per_epoch,
        "validation_steps": exp_cfg.validation_steps,
        "evaluate_train_steps": exp_cfg.evaluate_train_steps,
        "evaluate_val_steps": exp_cfg.evaluate_val_steps,
        "evaluate_test_steps": exp_cfg.evaluate_test_steps,
        "report_test_steps": exp_cfg.report_test_steps,
        "learning_rate": exp_cfg.learning_rate,
        "dropout_rate": exp_cfg.dropout_rate,
        "random_seed": exp_cfg.random_seed,
        "augmentation_enabled": exp_cfg.augmentation_enabled,
    }


def _validate_phase4b_subset(project_root: Path, cfgs: list[ExperimentConfig], messages: list[str]) -> None:
    phase4_cfgs = [cfg for cfg in cfgs if cfg.experiment_id in PHASE4B_REQUIRED_IDS]
    if len(phase4_cfgs) != 5:
        raise ValueError(f"Expected exactly 5 Phase 4B configs, found {len(phase4_cfgs)}")

    controls = [cfg for cfg in phase4_cfgs if cfg.is_control]
    if len(controls) != 1:
        raise ValueError("Exactly one control config is required for Phase 4B")
    control = controls[0]

    baseline_raw = _load_baseline_training_raw(project_root)
    data_cfg = load_data_pipeline_config(project_root)

    baseline_values = _baseline_control_keys(baseline_raw, data_cfg)
    control_values = _config_control_keys(control)
    for key, expected_value in baseline_values.items():
        if control_values[key] != expected_value:
            raise ValueError(f"Control config mismatch for {key}: {control_values[key]} != {expected_value}")

    control_model_values = {
        "freeze_backbone": True,
        "unfreeze_last_n_layers": 0,
        "unfreeze_last_fraction": None,
        "train_batch_norm": False,
        "optimizer": "Adam",
        "loss": "SparseCategoricalCrossentropy",
        "backbone": "MobileNetV2",
        "pretrained_weights": "imagenet",
    }
    for key, expected_value in control_model_values.items():
        if getattr(control, key) != expected_value:
            raise ValueError(f"Control config mismatch for {key}: {getattr(control, key)} != {expected_value}")

    single_change_ids = {
        "mnetv2_longer_frozen_epochs": "epochs",
        "mnetv2_lower_lr_frozen": "learning_rate",
        "mnetv2_higher_dropout_frozen": "dropout_rate",
    }

    allowed_shared = {
        "order",
        "experiment_id",
        "experiment_name",
        "experiment_category",
        "is_control",
        "changed_variable",
        "control_value",
        "new_value",
        "rationale",
        "mlflow_tags",
    }

    control_dict = asdict(control)
    for exp_id, changed_field in single_change_ids.items():
        cfg = get_experiment_by_id(phase4_cfgs, exp_id)
        if cfg.changed_variable != changed_field:
            raise ValueError(f"{exp_id} must declare changed_variable={changed_field}")
        cfg_dict = asdict(cfg)
        changed = [k for k, v in cfg_dict.items() if k not in allowed_shared and control_dict.get(k) != v]
        if changed != [changed_field]:
            raise ValueError(f"{exp_id} must only change {changed_field}; observed changes: {changed}")

    fine_tune = get_experiment_by_id(phase4_cfgs, FINE_TUNE_EXPERIMENT_ID)
    if fine_tune.changed_variable != "backbone_trainability+learning_rate":
        raise ValueError("Fine-tuning experiment must explicitly declare the approved multi-variable exception")

    fine_tune_allowed = {
        "freeze_backbone",
        "unfreeze_last_n_layers",
        "unfreeze_last_fraction",
        "learning_rate",
        "changed_variable",
        "control_value",
        "new_value",
        "rationale",
        "experiment_id",
        "experiment_name",
        "experiment_category",
        "is_control",
        "order",
        "mlflow_tags",
    }
    fine_dict = asdict(fine_tune)
    fine_changes = [k for k, v in fine_dict.items() if control_dict.get(k) != v]
    illegal_fine_changes = [k for k in fine_changes if k not in fine_tune_allowed]
    if illegal_fine_changes:
        raise ValueError(f"Fine-tuning config contains non-approved changes: {illegal_fine_changes}")

    messages.append("Validated 5-config controlled MobileNetV2 matrix.")
    messages.append(f"Control experiment: {control.experiment_id}")


def _validate_architecture_screening_subset(cfgs: list[ExperimentConfig], messages: list[str]) -> None:
    if ARCH_SCREENING_EXPERIMENT_ID not in {cfg.experiment_id for cfg in cfgs}:
        raise ValueError(f"Missing required architecture screening config: {ARCH_SCREENING_EXPERIMENT_ID}")

    candidate = get_experiment_by_id(cfgs, ARCH_SCREENING_EXPERIMENT_ID)
    target = get_experiment_by_id(cfgs, "mnetv2_longer_frozen_epochs")

    if candidate.experiment_category != "architecture_screening":
        raise ValueError("Architecture screening config must use experiment_category=architecture_screening")
    if candidate.changed_variable != "architecture":
        raise ValueError("Architecture screening config must declare changed_variable=architecture")
    if candidate.is_control:
        raise ValueError("Architecture screening config must set is_control=false")
    if candidate.backbone != "EfficientNetB0":
        raise ValueError("Architecture screening config must set backbone=EfficientNetB0")
    if candidate.model_family != "EfficientNetB0":
        raise ValueError("Architecture screening config must set model_family=EfficientNetB0")

    expected_arch_changes = {"model_input_resolution", "preprocessing_function"}
    observed_changes = set(candidate.architecture_required_changes)
    if not expected_arch_changes.issubset(observed_changes):
        raise ValueError(
            "Architecture screening config must declare architecture_required_changes including "
            "model_input_resolution and preprocessing_function"
        )

    disallowed_deltas = {
        "batch_size",
        "epochs",
        "learning_rate",
        "dropout_rate",
        "optimizer",
        "loss",
        "random_seed",
        "augmentation_enabled",
        "data_split_strategy",
        "freeze_backbone",
        "train_batch_norm",
        "unfreeze_last_n_layers",
        "unfreeze_last_fraction",
    }
    for field in disallowed_deltas:
        if getattr(candidate, field) != getattr(target, field):
            raise ValueError(f"Architecture screening config must not change {field}")

    phase_tag = candidate.mlflow_tags.get("phase")
    if phase_tag != "5A":
        raise ValueError("Architecture screening config mlflow_tags.phase must be 5A")

    if candidate.mlflow_tags.get("comparison_target") != "mnetv2_longer_frozen_epochs":
        raise ValueError("Architecture screening config must declare comparison_target=mnetv2_longer_frozen_epochs")

    messages.append("Validated Phase 5A EfficientNetB0 architecture-screening config.")


def _validate_resolution_control_subset(cfgs: list[ExperimentConfig], messages: list[str]) -> None:
    if RESOLUTION_CONTROL_EXPERIMENT_ID not in {cfg.experiment_id for cfg in cfgs}:
        raise ValueError(f"Missing required resolution control config: {RESOLUTION_CONTROL_EXPERIMENT_ID}")

    control = get_experiment_by_id(cfgs, "mnetv2_longer_frozen_epochs")
    candidate = get_experiment_by_id(cfgs, RESOLUTION_CONTROL_EXPERIMENT_ID)

    if candidate.experiment_category != "resolution_control":
        raise ValueError("Resolution control config must use experiment_category=resolution_control")
    if candidate.changed_variable != "model_input_resolution":
        raise ValueError("Resolution control config must declare changed_variable=model_input_resolution")
    if candidate.backbone != "MobileNetV2" or candidate.model_family != "MobileNetV2":
        raise ValueError("Resolution control config must preserve MobileNetV2 backbone/model_family")
    if candidate.preprocessing_function != "mobilenetv2_rescale_neg1_to_1":
        raise ValueError("Resolution control config must preserve MobileNetV2 preprocessing")
    if candidate.model_input_resolution != (96, 96):
        raise ValueError("Resolution control config must set model_input_resolution to (96, 96)")
    if set(candidate.architecture_required_changes) != {"model_input_resolution"}:
        raise ValueError("Resolution control config must declare architecture_required_changes=['model_input_resolution']")

    disallowed_deltas = {
        "backbone",
        "model_family",
        "pretrained_weights",
        "num_classes",
        "freeze_backbone",
        "unfreeze_last_n_layers",
        "unfreeze_last_fraction",
        "train_batch_norm",
        "batch_size",
        "epochs",
        "steps_per_epoch",
        "validation_steps",
        "evaluate_train_steps",
        "evaluate_val_steps",
        "evaluate_test_steps",
        "report_test_steps",
        "learning_rate",
        "dropout_rate",
        "optimizer",
        "loss",
        "random_seed",
        "augmentation_enabled",
        "data_split_strategy",
        "preprocessing_function",
    }
    for field in disallowed_deltas:
        if getattr(candidate, field) != getattr(control, field):
            raise ValueError(f"Resolution control config must not change {field}")

    if candidate.mlflow_tags.get("phase") != "5A":
        raise ValueError("Resolution control config mlflow_tags.phase must be 5A")
    if candidate.mlflow_tags.get("comparison_target") != "mnetv2_longer_frozen_epochs":
        raise ValueError("Resolution control config must declare comparison_target=mnetv2_longer_frozen_epochs")

    messages.append("Validated Phase 5A MobileNetV2 96x96 resolution-control config.")


def validate_experiment_matrix(
    project_root: Path,
    configs: list[ExperimentConfig] | None = None,
) -> MatrixValidationResult:
    """Validate approved Phase 4B + Phase 5A experiment configurations."""

    cfgs = configs or load_experiment_configs(project_root)
    messages: list[str] = []

    ids = [cfg.experiment_id for cfg in cfgs]
    if len(set(ids)) != len(ids):
        raise ValueError("Experiment IDs must be unique")

    missing_phase4 = sorted(PHASE4B_REQUIRED_IDS - set(ids))
    if missing_phase4:
        raise ValueError(f"Missing required Phase 4B experiment IDs: {missing_phase4}")

    _validate_phase4b_subset(project_root, cfgs, messages)
    _validate_architecture_screening_subset(cfgs, messages)
    _validate_resolution_control_subset(cfgs, messages)

    return MatrixValidationResult(
        experiment_ids=tuple(ids),
        control_experiment_id="mnetv2_control_frozen",
        validation_messages=tuple(messages),
    )


# Backward-compatible export name used by existing imports/tests.
MobileNetExperimentConfig = ExperimentConfig
