"""Configuration loading and validation for controlled MobileNetV2 experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .pipeline_config import DataPipelineConfig, load_data_pipeline_config


EXPERIMENT_DIR = Path("configs") / "experiments"
FINE_TUNE_EXPERIMENT_ID = "mnetv2_partial_finetune_tail"


@dataclass(frozen=True)
class MobileNetExperimentConfig:
    """Controlled experiment configuration for MobileNetV2 matrix runs."""

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
    mlflow_tags: dict[str, str]

    def __post_init__(self) -> None:
        if self.order <= 0:
            raise ValueError("order must be > 0")
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must not be empty")
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must not be empty")
        if self.model_family != "MobileNetV2":
            raise ValueError("model_family must be MobileNetV2")
        if self.backbone != "MobileNetV2":
            raise ValueError("backbone must be MobileNetV2")
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


def load_experiment_config(path: Path) -> MobileNetExperimentConfig:
    data = _load_json(path)
    tags = {str(k): str(v) for k, v in data.get("mlflow_tags", {}).items()}
    data["mlflow_tags"] = tags
    return MobileNetExperimentConfig(**data)


def load_experiment_configs(project_root: Path) -> list[MobileNetExperimentConfig]:
    exp_dir = project_root / EXPERIMENT_DIR
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {exp_dir}")
    configs = [load_experiment_config(path) for path in sorted(exp_dir.glob("*.json"))]
    if not configs:
        raise ValueError(f"No experiment configs found in {exp_dir}")
    return sorted(configs, key=lambda cfg: cfg.order)


def load_experiment_configs_with_paths(project_root: Path) -> list[tuple[MobileNetExperimentConfig, Path]]:
    """Load experiment configs with source file paths."""

    exp_dir = project_root / EXPERIMENT_DIR
    pairs = [(load_experiment_config(path), path) for path in sorted(exp_dir.glob("*.json"))]
    return sorted(pairs, key=lambda pair: pair[0].order)


def get_experiment_by_id(configs: list[MobileNetExperimentConfig], experiment_id: str) -> MobileNetExperimentConfig:
    for cfg in configs:
        if cfg.experiment_id == experiment_id:
            return cfg
    raise KeyError(f"Experiment ID not found: {experiment_id}")


def _load_baseline_training_raw(project_root: Path) -> dict[str, Any]:
    path = project_root / "configs" / "baseline_training.json"
    return json.loads(path.read_text(encoding="utf-8"))


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


def _config_control_keys(exp_cfg: MobileNetExperimentConfig) -> dict[str, Any]:
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


def validate_experiment_matrix(
    project_root: Path,
    configs: list[MobileNetExperimentConfig] | None = None,
) -> MatrixValidationResult:
    """Validate the approved five-experiment MobileNetV2 matrix."""

    cfgs = configs or load_experiment_configs(project_root)
    messages: list[str] = []

    if len(cfgs) != 5:
        raise ValueError(f"Expected exactly 5 experiment configs, found {len(cfgs)}")

    ids = [cfg.experiment_id for cfg in cfgs]
    if len(set(ids)) != len(ids):
        raise ValueError("Experiment IDs must be unique")

    controls = [cfg for cfg in cfgs if cfg.is_control]
    if len(controls) != 1:
        raise ValueError("Exactly one control config is required")
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
        cfg = get_experiment_by_id(cfgs, exp_id)
        if cfg.changed_variable != changed_field:
            raise ValueError(f"{exp_id} must declare changed_variable={changed_field}")
        cfg_dict = asdict(cfg)
        changed = [k for k, v in cfg_dict.items() if k not in allowed_shared and control_dict.get(k) != v]
        if changed != [changed_field]:
            raise ValueError(f"{exp_id} must only change {changed_field}; observed changes: {changed}")

    fine_tune = get_experiment_by_id(cfgs, FINE_TUNE_EXPERIMENT_ID)
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

    return MatrixValidationResult(
        experiment_ids=tuple(ids),
        control_experiment_id=control.experiment_id,
        validation_messages=tuple(messages),
    )
