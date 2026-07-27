"""Phase 3B baseline transfer-learning workflow (MobileNetV2 frozen backbone)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from .data_pipeline import build_cifar10_datasets
from .pipeline_config import load_data_pipeline_config


@dataclass(frozen=True)
class BaselineTrainingConfig:
    """Hyperparameters and artifact locations for baseline training."""

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


def load_baseline_training_config(project_root: Path, path: Path | None = None) -> BaselineTrainingConfig:
    """Load baseline training config from JSON."""

    cfg_path = path or (project_root / "configs" / "baseline_training.json")
    if not cfg_path.exists():
        return BaselineTrainingConfig()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    return BaselineTrainingConfig(**data)


def _set_reproducibility(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)


def build_mobilenetv2_frozen_model(input_shape: tuple[int, int, int], num_classes: int, dropout_rate: float) -> tf.keras.Model:
    """Create MobileNetV2 baseline with frozen ImageNet backbone."""

    backbone = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="image_input")
    # Pipeline outputs are normalized to [0, 1]. Convert to MobileNetV2 expected
    # range [-1, 1] without Lambda layers so model serialization stays safe.
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0, name="mobilenetv2_rescale")(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_frozen_baseline")


def _param_counts(model: tf.keras.Model) -> tuple[int, int]:
    trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    frozen = int(np.sum([np.prod(v.shape) for v in model.non_trainable_weights]))
    return trainable, frozen


def _collect_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    steps: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []

    eval_ds = dataset.take(steps) if steps is not None else dataset
    for images, labels in eval_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_pred.append(preds.astype(np.int32))
        y_true.append(labels.numpy().astype(np.int32))

    return np.concatenate(y_true), np.concatenate(y_pred)


def run_baseline_training(project_root: Path) -> dict[str, Any]:
    """Run the baseline training/evaluation/save/load workflow for Phase 3B."""

    data_cfg = load_data_pipeline_config(project_root)
    train_cfg = load_baseline_training_config(project_root)
    _set_reproducibility(train_cfg.random_seed)

    bundle = build_cifar10_datasets(project_root, data_cfg)
    input_shape = (data_cfg.image_size[0], data_cfg.image_size[1], 3)

    model = build_mobilenetv2_frozen_model(
        input_shape=input_shape,
        num_classes=train_cfg.num_classes,
        dropout_rate=train_cfg.dropout_rate,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=train_cfg.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    trainable_params, frozen_params = _param_counts(model)

    start = time.perf_counter()
    history = model.fit(
        bundle.train_ds,
        validation_data=bundle.val_ds,
        epochs=train_cfg.epochs,
        steps_per_epoch=train_cfg.steps_per_epoch,
        validation_steps=train_cfg.validation_steps,
        verbose=2,
    )
    training_time_sec = time.perf_counter() - start

    train_eval = model.evaluate(
        bundle.train_ds,
        steps=train_cfg.evaluate_train_steps,
        return_dict=True,
        verbose=0,
    )
    val_eval = model.evaluate(
        bundle.val_ds,
        steps=train_cfg.evaluate_val_steps,
        return_dict=True,
        verbose=0,
    )
    test_eval = model.evaluate(
        bundle.test_ds,
        steps=train_cfg.evaluate_test_steps,
        return_dict=True,
        verbose=0,
    )

    y_true, y_pred = _collect_predictions(model, bundle.test_ds, train_cfg.report_test_steps)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(train_cfg.num_classes)))
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(train_cfg.num_classes)),
        target_names=list(bundle.class_names),
        zero_division=0,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = project_root / train_cfg.model_output_dir
    report_dir = project_root / train_cfg.report_output_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{train_cfg.model_name}_{ts}.keras"
    metrics_path = report_dir / f"{train_cfg.model_name}_{ts}_metrics.json"
    cm_path = report_dir / f"{train_cfg.model_name}_{ts}_confusion_matrix.csv"
    cls_report_path = report_dir / f"{train_cfg.model_name}_{ts}_classification_report.txt"

    model.save(model_path)
    loaded_model = tf.keras.models.load_model(model_path)

    sample_images, _ = next(iter(bundle.test_ds.take(1)))
    loaded_preds = loaded_model.predict(sample_images[:8], verbose=0)
    inference_ok = loaded_preds.shape == (8, train_cfg.num_classes)

    np.savetxt(cm_path, cm.astype(np.int32), fmt="%d", delimiter=",")
    cls_report_path.write_text(report_text, encoding="utf-8")

    result = {
        "timestamp": ts,
        "data_config": asdict(data_cfg),
        "training_config": asdict(train_cfg),
        "class_names": list(bundle.class_names),
        "sample_counts": {
            "train": bundle.train_images,
            "val": bundle.val_images,
            "test": bundle.test_images,
        },
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
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
        "artifacts": {
            "model": str(model_path),
            "metrics_json": str(metrics_path),
            "confusion_matrix_csv": str(cm_path),
            "classification_report_txt": str(cls_report_path),
        },
        "verification": {
            "model_build_success": True,
            "model_train_success": True,
            "model_eval_success": True,
            "model_save_success": model_path.exists(),
            "model_load_success": loaded_model is not None,
            "inference_batch_success": bool(inference_ok),
        },
    }

    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    result = run_baseline_training(project_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
