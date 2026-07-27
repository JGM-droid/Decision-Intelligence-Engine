from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.decision_intelligence_engine import baseline_training


def _write_png(path: Path, tf, value: int) -> None:
    image = tf.ones((32, 32, 3), dtype=tf.uint8) * tf.cast(value % 255, tf.uint8)
    encoded = tf.io.encode_png(image).numpy()
    path.write_bytes(encoded)


@pytest.fixture
def tiny_baseline_project(tmp_path: Path):
    tf = pytest.importorskip("tensorflow")

    project_root = tmp_path
    data_root = project_root / "data" / "raw" / "cifar10"
    train = data_root / "train"
    test = data_root / "test"
    class_names = ["airplane", "automobile"]

    idx = 1
    for split_root, per_class in ((train, 4), (test, 2)):
        for cls in class_names:
            cls_dir = split_root / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            for _ in range(per_class):
                _write_png(cls_dir / f"{idx}.png", tf, value=idx)
                idx += 1

    (project_root / "configs").mkdir(parents=True, exist_ok=True)
    (project_root / "configs" / "baseline_training.json").write_text(
        json.dumps(
            {
                "epochs": 1,
                "steps_per_epoch": 1,
                "validation_steps": 1,
                "evaluate_train_steps": 1,
                "evaluate_val_steps": 1,
                "evaluate_test_steps": 1,
                "report_test_steps": 1,
                "num_classes": 2,
                "model_output_dir": "models",
                "report_output_dir": "reports",
                "random_seed": 123,
            }
        ),
        encoding="utf-8",
    )
    (project_root / "configs" / "mlflow.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "tracking_dir": "mlruns",
                "experiment_name": "test_experiment",
                "run_name_prefix": "smoke",
                "tags": {"suite": "pytest"},
            }
        ),
        encoding="utf-8",
    )
    (project_root / "configs" / "data_pipeline.json").write_text(
        json.dumps(
            {
                "data_root": "data/raw/cifar10",
                "image_size": [32, 32],
                "batch_size": 4,
                "validation_split": 0.25,
                "random_seed": 123,
                "shuffle_buffer_size": 32,
                "num_parallel_calls": 1,
                "cache_train": False,
                "cache_eval": False,
                "prefetch": False,
                "augmentation_enabled": False,
                "augmentation_padding": 4,
                "augmentation_horizontal_flip": True,
            }
        ),
        encoding="utf-8",
    )

    return project_root


def test_run_baseline_training_logs_mlflow_artifacts(tiny_baseline_project: Path) -> None:
    result = baseline_training.run_baseline_training(tiny_baseline_project)

    assert result["verification"]["mlflow_run_success"] is True
    assert result["artifacts"]["model"].endswith(".keras")
    assert Path(result["artifacts"]["model"]).exists()
    assert Path(result["artifacts"]["metrics_json"]).exists()
    assert Path(result["artifacts"]["confusion_matrix_png"]).exists()
    assert Path(result["artifacts"]["training_history_png"]).exists()

    mlruns = tiny_baseline_project / "mlruns"
    assert mlruns.exists()
    assert any(mlruns.rglob("meta.yaml"))
