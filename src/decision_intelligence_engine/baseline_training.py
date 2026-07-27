"""Shared transfer-learning training workflow for controlled experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from .data_pipeline import build_cifar10_datasets
from .mlflow_config import load_mlflow_config
from .mlflow_tracking import flatten_metrics, flatten_params
from .pipeline_config import DataPipelineConfig, load_data_pipeline_config


@dataclass(frozen=True)
class BaselineTrainingConfig:
    """Legacy baseline config retained for backward compatibility."""

    model_name: str = "mobilenetv2_frozen_baseline"
    num_classes: int = 10
    dropout_rate: float = 0.2
    learning_rate: float = 3e-4
    epochs: int = 1
    steps_per_epoch: int | None = 100
    validation_steps: int | None = None
    evaluate_train_steps: int | None = None
    evaluate_val_steps: int | None = None
    evaluate_test_steps: int | None = None
    report_test_steps: int | None = None
    model_output_dir: str = "models"
    report_output_dir: str = "reports"
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.num_classes <= 1:
            raise ValueError("num_classes must be > 1.")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1).")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0.")


@dataclass(frozen=True)
class ModelRunConfig:
    """Effective run configuration for a single transfer-learning experiment."""

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


# Backward-compatible alias retained for existing tests/imports.
MobileNetRunConfig = ModelRunConfig


SUPPORTED_BACKBONES = {"MobileNetV2", "EfficientNetB0"}


def _default_model_input_resolution(backbone: str) -> tuple[int, int]:
    if backbone == "EfficientNetB0":
        # Practical screening resolution that remains valid for ImageNet-initialized EfficientNetB0.
        return (96, 96)
    return (32, 32)


def _default_preprocessing_function(backbone: str) -> str:
    if backbone == "EfficientNetB0":
        return "efficientnetb0_builtin_rescaling_with_input_scale_255"
    return "mobilenetv2_rescale_neg1_to_1"


def load_baseline_training_config(project_root: Path, path: Path | None = None) -> BaselineTrainingConfig:
    """Load legacy baseline training config from JSON."""

    cfg_path = path or (project_root / "configs" / "baseline_training.json")
    if not cfg_path.exists():
        return BaselineTrainingConfig()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    return BaselineTrainingConfig(**data)


def _set_reproducibility(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)


def _create_run_name(prefix: str, experiment_id: str, timestamp: str) -> str:
    return f"{prefix}_{experiment_id}_{timestamp}"


def _build_backbone(backbone: str, input_shape: tuple[int, int, int], pretrained_weights: str) -> tf.keras.Model:
    if backbone == "MobileNetV2":
        return tf.keras.applications.MobileNetV2(
            include_top=False,
            weights=pretrained_weights,
            input_shape=input_shape,
        )
    if backbone == "EfficientNetB0":
        return tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=pretrained_weights,
            input_shape=input_shape,
        )
    raise ValueError(f"Unsupported backbone: {backbone}")


def _apply_backbone_trainability(
    backbone: tf.keras.Model,
    freeze_backbone: bool,
    unfreeze_last_n_layers: int | None,
    unfreeze_last_fraction: float | None,
    train_batch_norm: bool,
) -> tuple[int, int]:
    for layer in backbone.layers:
        layer.trainable = False

    if not freeze_backbone:
        total_layers = len(backbone.layers)
        if unfreeze_last_n_layers is not None and unfreeze_last_n_layers > 0:
            n_layers = min(unfreeze_last_n_layers, total_layers)
        elif unfreeze_last_fraction is not None and unfreeze_last_fraction > 0:
            n_layers = max(1, int(round(total_layers * unfreeze_last_fraction)))
        else:
            raise ValueError("Unfrozen runs must define unfreeze_last_n_layers or unfreeze_last_fraction")

        for layer in backbone.layers[-n_layers:]:
            if isinstance(layer, tf.keras.layers.BatchNormalization) and not train_batch_norm:
                layer.trainable = False
            else:
                layer.trainable = True

    trainable_backbone_layers = sum(1 for layer in backbone.layers if layer.trainable)
    return trainable_backbone_layers, len(backbone.layers)


def _resolve_model_resolution(cfg: ModelRunConfig, dataset_input_shape: tuple[int, int, int]) -> tuple[int, int]:
    if cfg.model_input_resolution is None:
        return _default_model_input_resolution(cfg.backbone)
    return (int(cfg.model_input_resolution[0]), int(cfg.model_input_resolution[1]))


def _resolve_preprocessing(cfg: ModelRunConfig) -> str:
    if cfg.preprocessing_function:
        return cfg.preprocessing_function
    return _default_preprocessing_function(cfg.backbone)


def build_transfer_model(input_shape: tuple[int, int, int], cfg: ModelRunConfig) -> tuple[tf.keras.Model, int, int]:
    """Create configurable transfer-learning model for approved backbones."""

    if cfg.backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone: {cfg.backbone}")

    model_resolution = _resolve_model_resolution(cfg, input_shape)
    backbone_input_shape = (model_resolution[0], model_resolution[1], 3)
    backbone = _build_backbone(backbone=cfg.backbone, input_shape=backbone_input_shape, pretrained_weights=cfg.pretrained_weights)

    trainable_backbone_layers, total_backbone_layers = _apply_backbone_trainability(
        backbone=backbone,
        freeze_backbone=cfg.freeze_backbone,
        unfreeze_last_n_layers=cfg.unfreeze_last_n_layers,
        unfreeze_last_fraction=cfg.unfreeze_last_fraction,
        train_batch_norm=cfg.train_batch_norm,
    )

    inputs = tf.keras.Input(shape=input_shape, name="image_input")
    x = inputs

    if model_resolution != input_shape[:2]:
        x = tf.keras.layers.Resizing(
            model_resolution[0],
            model_resolution[1],
            interpolation="bilinear",
            name=f"{cfg.backbone.lower()}_resize",
        )(x)

    preprocessing = _resolve_preprocessing(cfg)
    if preprocessing == "mobilenetv2_rescale_neg1_to_1":
        x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0, name="mobilenetv2_rescale")(x)
    elif preprocessing == "efficientnetb0_builtin_rescaling_with_input_scale_255":
        # TF 2.16/Keras 3 EfficientNetB0 includes internal rescaling and expects 0..255 scale.
        x = tf.keras.layers.Rescaling(scale=255.0, name="efficientnet_input_scale_255")(x)
    else:
        raise ValueError(f"Unsupported preprocessing function: {preprocessing}")

    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(cfg.dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(cfg.num_classes, activation="softmax", name="classifier")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{cfg.backbone.lower()}_{cfg.experiment_id}")
    return model, trainable_backbone_layers, total_backbone_layers


def build_mobilenetv2_model(input_shape: tuple[int, int, int], cfg: MobileNetRunConfig) -> tuple[tf.keras.Model, int, int]:
    """Backward-compatible helper retained for MobileNetV2-only test usage."""

    if cfg.backbone != "MobileNetV2":
        raise ValueError("build_mobilenetv2_model only supports MobileNetV2 configs")
    return build_transfer_model(input_shape=input_shape, cfg=cfg)


def _param_counts(model: tf.keras.Model) -> tuple[int, int, int]:
    trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    frozen = int(np.sum([np.prod(v.shape) for v in model.non_trainable_weights]))
    return trainable, frozen, trainable + frozen


def _collect_predictions(model: tf.keras.Model, dataset: tf.data.Dataset, steps: int | None) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []

    eval_ds = dataset.take(steps) if steps is not None else dataset
    for images, labels in eval_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_pred.append(preds.astype(np.int32))
        y_true.append(labels.numpy().astype(np.int32))

    return np.concatenate(y_true), np.concatenate(y_pred)


def _plot_confusion_matrix(cm: np.ndarray, class_names: tuple[str, ...], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    tick_positions = np.arange(len(class_names))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_training_history(history: tf.keras.callbacks.History, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    epoch_count = range(1, len(history.history.get("loss", [])) + 1)

    if "loss" in history.history:
        ax.plot(epoch_count, history.history["loss"], label="train_loss", linewidth=2)
    if "val_loss" in history.history:
        ax.plot(epoch_count, history.history["val_loss"], label="val_loss", linewidth=2)
    if "accuracy" in history.history:
        ax.plot(epoch_count, history.history["accuracy"], label="train_accuracy", linewidth=2)
    if "val_accuracy" in history.history:
        ax.plot(epoch_count, history.history["val_accuracy"], label="val_accuracy", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title("Training history")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _from_legacy_baseline(project_root: Path, data_cfg: DataPipelineConfig, mlflow_tags: dict[str, str]) -> ModelRunConfig:
    train_cfg = load_baseline_training_config(project_root)
    return ModelRunConfig(
        order=0,
        experiment_id="legacy_baseline",
        experiment_name="Legacy Baseline Frozen",
        experiment_category="legacy",
        is_control=True,
        changed_variable="none",
        control_value="baseline",
        new_value="baseline",
        rationale="Backward-compatible single baseline run path.",
        model_family="MobileNetV2",
        backbone="MobileNetV2",
        pretrained_weights="imagenet",
        num_classes=train_cfg.num_classes,
        freeze_backbone=True,
        unfreeze_last_n_layers=0,
        unfreeze_last_fraction=None,
        train_batch_norm=False,
        batch_size=data_cfg.batch_size,
        epochs=train_cfg.epochs,
        steps_per_epoch=train_cfg.steps_per_epoch,
        validation_steps=train_cfg.validation_steps,
        evaluate_train_steps=train_cfg.evaluate_train_steps,
        evaluate_val_steps=train_cfg.evaluate_val_steps,
        evaluate_test_steps=train_cfg.evaluate_test_steps,
        report_test_steps=train_cfg.report_test_steps,
        learning_rate=train_cfg.learning_rate,
        dropout_rate=train_cfg.dropout_rate,
        optimizer="Adam",
        loss="SparseCategoricalCrossentropy",
        random_seed=train_cfg.random_seed,
        augmentation_enabled=data_cfg.augmentation_enabled,
        data_split_strategy="deterministic_stratified_file_manifest",
        model_input_resolution=data_cfg.image_size,
        preprocessing_function="mobilenetv2_rescale_neg1_to_1",
        architecture_required_changes=(),
        mlflow_tags=mlflow_tags,
    )


def _from_experiment_dict(raw: dict[str, Any]) -> ModelRunConfig:
    data = dict(raw)
    tags = {str(k): str(v) for k, v in data["mlflow_tags"].items()}
    data["mlflow_tags"] = tags
    if "model_input_resolution" in data and data["model_input_resolution"] is not None:
        data["model_input_resolution"] = tuple(int(v) for v in data["model_input_resolution"])
    if "architecture_required_changes" in data and data["architecture_required_changes"] is not None:
        data["architecture_required_changes"] = tuple(str(v) for v in data["architecture_required_changes"])
    return ModelRunConfig(**data)


def run_transfer_experiment(
    project_root: Path,
    run_cfg: ModelRunConfig,
    experiment_config_path: Path | None = None,
) -> dict[str, Any]:
    """Run one transfer-learning experiment using shared data/eval/MLflow paths."""

    data_cfg = load_data_pipeline_config(project_root)
    mlflow_cfg = load_mlflow_config(project_root)

    if run_cfg.batch_size != data_cfg.batch_size:
        raise ValueError(f"Experiment batch_size {run_cfg.batch_size} must match pipeline batch_size {data_cfg.batch_size}")
    if run_cfg.augmentation_enabled != data_cfg.augmentation_enabled:
        raise ValueError("Experiment augmentation setting must match the approved pipeline setting")

    _set_reproducibility(run_cfg.random_seed)

    bundle = build_cifar10_datasets(project_root, data_cfg)
    input_shape = (data_cfg.image_size[0], data_cfg.image_size[1], 3)

    model_input_resolution = _resolve_model_resolution(run_cfg, input_shape)
    preprocessing_function = _resolve_preprocessing(run_cfg)

    model, trainable_backbone_layers, total_backbone_layers = build_transfer_model(
        input_shape=input_shape,
        cfg=run_cfg,
    )

    if run_cfg.optimizer != "Adam":
        raise ValueError("Only Adam optimizer is supported in the controlled matrix")
    if run_cfg.loss != "SparseCategoricalCrossentropy":
        raise ValueError("Only SparseCategoricalCrossentropy is supported in the controlled matrix")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=run_cfg.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    trainable_params, frozen_params, total_params = _param_counts(model)

    mlflow.set_tracking_uri(mlflow_cfg.tracking_uri(project_root))
    mlflow.set_experiment(mlflow_cfg.experiment_name)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = _create_run_name(mlflow_cfg.run_name_prefix, run_cfg.experiment_id, ts)

    with mlflow.start_run(run_name=run_name) as run:
        start = time.perf_counter()
        history = model.fit(
            bundle.train_ds,
            validation_data=bundle.val_ds,
            epochs=run_cfg.epochs,
            steps_per_epoch=run_cfg.steps_per_epoch,
            validation_steps=run_cfg.validation_steps,
            verbose=2,
        )
        training_time_sec = time.perf_counter() - start

        train_eval = model.evaluate(bundle.train_ds, steps=run_cfg.evaluate_train_steps, return_dict=True, verbose=0)
        val_eval = model.evaluate(bundle.val_ds, steps=run_cfg.evaluate_val_steps, return_dict=True, verbose=0)
        test_eval = model.evaluate(bundle.test_ds, steps=run_cfg.evaluate_test_steps, return_dict=True, verbose=0)

        y_true, y_pred = _collect_predictions(model, bundle.test_ds, run_cfg.report_test_steps)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(run_cfg.num_classes)))
        report_text = classification_report(
            y_true,
            y_pred,
            labels=list(range(run_cfg.num_classes)),
            target_names=list(bundle.class_names),
            zero_division=0,
        )

        model_dir = project_root / "models"
        report_dir = project_root / "reports"
        model_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"{run_cfg.experiment_id}_{ts}"
        model_path = model_dir / f"{base_name}.keras"
        metrics_path = report_dir / f"{base_name}_metrics.json"
        cm_path = report_dir / f"{base_name}_confusion_matrix.csv"
        cls_report_path = report_dir / f"{base_name}_classification_report.txt"
        cm_image_path = report_dir / f"{base_name}_confusion_matrix.png"
        history_plot_path = report_dir / f"{base_name}_training_history.png"

        model.save(model_path)
        loaded_model = tf.keras.models.load_model(model_path)
        sample_images, _ = next(iter(bundle.test_ds.take(1)))
        loaded_preds = loaded_model.predict(sample_images[:8], verbose=0)
        inference_ok = loaded_preds.shape == (8, run_cfg.num_classes)

        np.savetxt(cm_path, cm.astype(np.int32), fmt="%d", delimiter=",")
        cls_report_path.write_text(report_text, encoding="utf-8")
        _plot_confusion_matrix(cm, tuple(bundle.class_names), cm_image_path)
        _plot_training_history(history, history_plot_path)

        env = {
            "python": f"{tf.sysconfig.get_build_info().get('python_version', 'unknown')}",
            "tensorflow": tf.__version__,
            "keras": tf.keras.__version__,
            "mlflow": mlflow.__version__,
            "matplotlib": matplotlib.__version__,
        }

        verification = {
            "model_build_success": True,
            "model_train_success": True,
            "model_eval_success": True,
            "model_save_success": model_path.exists(),
            "model_load_success": loaded_model is not None,
            "inference_batch_success": bool(inference_ok),
            "mlflow_run_success": True,
        }

        result = {
            "timestamp": ts,
            "run_id": run.info.run_id,
            "run_name": run_name,
            "experiment": asdict(run_cfg),
            "data_config": asdict(data_cfg),
            "mlflow_config": asdict(mlflow_cfg),
            "class_names": list(bundle.class_names),
            "sample_counts": {
                "train": bundle.train_images,
                "val": bundle.val_images,
                "test": bundle.test_images,
            },
            "trainable_backbone_layers": trainable_backbone_layers,
            "total_backbone_layers": total_backbone_layers,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "total_params": total_params,
            "training_time_sec": training_time_sec,
            "history_final": {
                "loss": float(history.history["loss"][-1]),
                "accuracy": float(history.history["accuracy"][-1]),
                "val_loss": float(history.history["val_loss"][-1]),
                "val_accuracy": float(history.history["val_accuracy"][-1]),
            },
            "eval": {
                "train": {"loss": float(train_eval["loss"]), "accuracy": float(train_eval["accuracy"])},
                "val": {"loss": float(val_eval["loss"]), "accuracy": float(val_eval["accuracy"])},
                "test": {"loss": float(test_eval["loss"]), "accuracy": float(test_eval["accuracy"])},
            },
            "environment": env,
            "architecture": {
                "dataset_input_resolution": list(input_shape[:2]),
                "model_input_resolution": list(model_input_resolution),
                "preprocessing_function": preprocessing_function,
                "architecture_required_changes": list(run_cfg.architecture_required_changes),
            },
            "artifacts": {
                "model": str(model_path),
                "metrics_json": str(metrics_path),
                "confusion_matrix_csv": str(cm_path),
                "classification_report_txt": str(cls_report_path),
                "confusion_matrix_png": str(cm_image_path),
                "training_history_png": str(history_plot_path),
            },
            "verification": verification,
        }

        metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        params_payload = {
            "experiment": asdict(run_cfg),
            "data": {
                "root": data_cfg.data_root,
                "train_subdir": data_cfg.train_subdir,
                "test_subdir": data_cfg.test_subdir,
                "image_size": list(data_cfg.image_size),
                "batch_size": data_cfg.batch_size,
                "validation_split": data_cfg.validation_split,
                "random_seed": data_cfg.random_seed,
                "augmentation_enabled": data_cfg.augmentation_enabled,
                "split_strategy": "deterministic_stratified_file_manifest",
                "original_image_size": list(data_cfg.image_size),
                "model_input_resolution": list(model_input_resolution),
                "preprocessing_function": preprocessing_function,
                "architecture_required_changes": list(run_cfg.architecture_required_changes),
                "train_images": bundle.train_images,
                "val_images": bundle.val_images,
                "test_images": bundle.test_images,
            },
            "verification": verification,
            "environment": env,
        }

        metrics_payload = {
            "training": {
                "time_sec": training_time_sec,
                "trainable_params": trainable_params,
                "frozen_params": frozen_params,
                "total_params": total_params,
                "trainable_backbone_layers": trainable_backbone_layers,
                "total_backbone_layers": total_backbone_layers,
            },
            "history_final": result["history_final"],
            "eval": {
                "train_loss": float(train_eval["loss"]),
                "train_accuracy": float(train_eval["accuracy"]),
                "val_loss": float(val_eval["loss"]),
                "val_accuracy": float(val_eval["accuracy"]),
                "test_loss": float(test_eval["loss"]),
                "test_accuracy": float(test_eval["accuracy"]),
            },
        }

        mlflow.log_params(flatten_params(params_payload))
        mlflow.log_metrics(flatten_metrics(metrics_payload))

        for epoch_index, loss_value in enumerate(history.history.get("loss", [])):
            mlflow.log_metric("history.loss", float(loss_value), step=epoch_index)
        for epoch_index, accuracy_value in enumerate(history.history.get("accuracy", [])):
            mlflow.log_metric("history.accuracy", float(accuracy_value), step=epoch_index)
        for epoch_index, loss_value in enumerate(history.history.get("val_loss", [])):
            mlflow.log_metric("history.val_loss", float(loss_value), step=epoch_index)
        for epoch_index, accuracy_value in enumerate(history.history.get("val_accuracy", [])):
            mlflow.log_metric("history.val_accuracy", float(accuracy_value), step=epoch_index)

        mlflow_tags = {
            **mlflow_cfg.tags,
            **run_cfg.mlflow_tags,
            "run_id": run.info.run_id,
            "experiment_id": run_cfg.experiment_id,
            "experiment_category": run_cfg.experiment_category,
            "changed_variable": run_cfg.changed_variable,
            "control_flag": str(run_cfg.is_control).lower(),
            "model_family": run_cfg.model_family,
            "phase": run_cfg.mlflow_tags.get("phase", "4B"),
            "split_strategy": run_cfg.data_split_strategy,
            "repro_seed": str(run_cfg.random_seed),
            "backbone": run_cfg.backbone,
            "preprocessing_function": preprocessing_function,
        }
        mlflow.set_tags(mlflow_tags)

        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(metrics_path), artifact_path="reports")
        mlflow.log_artifact(str(cm_path), artifact_path="reports")
        mlflow.log_artifact(str(cls_report_path), artifact_path="reports")
        mlflow.log_artifact(str(cm_image_path), artifact_path="reports")
        mlflow.log_artifact(str(history_plot_path), artifact_path="reports")
        mlflow.log_artifact(str(project_root / "configs" / "baseline_training.json"), artifact_path="configs")
        mlflow.log_artifact(str(project_root / "configs" / "data_pipeline.json"), artifact_path="configs")
        if experiment_config_path is not None and experiment_config_path.exists():
            mlflow.log_artifact(str(experiment_config_path), artifact_path="configs")

        return result


def run_mobilenetv2_experiment(
    project_root: Path,
    run_cfg: MobileNetRunConfig,
    experiment_config_path: Path | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper for existing MobileNetV2 workflow calls."""

    return run_transfer_experiment(project_root=project_root, run_cfg=run_cfg, experiment_config_path=experiment_config_path)


def run_mobilenetv2_experiment_from_config(
    project_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Run one experiment directly from a JSON experiment config path."""

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    run_cfg = _from_experiment_dict(raw)
    return run_transfer_experiment(project_root=project_root, run_cfg=run_cfg, experiment_config_path=config_path)


def run_baseline_training(project_root: Path) -> dict[str, Any]:
    """Backward-compatible baseline entry point used by existing tests and docs."""

    data_cfg = load_data_pipeline_config(project_root)
    mlflow_cfg = load_mlflow_config(project_root)
    run_cfg = _from_legacy_baseline(project_root, data_cfg, mlflow_tags=mlflow_cfg.tags)
    return run_transfer_experiment(project_root=project_root, run_cfg=run_cfg, experiment_config_path=None)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    result = run_baseline_training(project_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
