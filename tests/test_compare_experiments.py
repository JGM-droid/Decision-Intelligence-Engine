from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.decision_intelligence_engine.compare_experiments import compare_phase4b_runs


def _write_text(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _create_run(client, experiment_id: str, payload: dict, artifact_root: Path) -> None:
    run = client.create_run(experiment_id=experiment_id, tags=payload["tags"])
    run_id = run.info.run_id
    for key, value in payload["params"].items():
        client.log_param(run_id, key, value)
    for key, value in payload["metrics"].items():
        client.log_metric(run_id, key, float(value))

    local = artifact_root / run_id
    _write_text(local / "configs" / "baseline_training.json", "{}")
    _write_text(local / "configs" / "data_pipeline.json", "{}")
    _write_text(local / "model" / f"{payload['tags']['experiment_id']}.keras", "model")
    _write_text(local / "reports" / "a_confusion_matrix.csv")
    _write_text(local / "reports" / "a_confusion_matrix.png")
    _write_text(local / "reports" / "a_classification_report.txt")
    _write_text(local / "reports" / "a_metrics.json")
    _write_text(local / "reports" / "a_training_history.png")

    for artifact in [
        local / "configs" / "baseline_training.json",
        local / "configs" / "data_pipeline.json",
        local / "model" / f"{payload['tags']['experiment_id']}.keras",
        local / "reports" / "a_confusion_matrix.csv",
        local / "reports" / "a_confusion_matrix.png",
        local / "reports" / "a_classification_report.txt",
        local / "reports" / "a_metrics.json",
        local / "reports" / "a_training_history.png",
    ]:
        artifact_path = "configs" if "configs" in artifact.parts else "model" if "model" in artifact.parts else "reports"
        client.log_artifact(run_id, str(artifact), artifact_path=artifact_path)

    client.set_terminated(run_id, status=payload.get("status", "FINISHED"))


def test_compare_phase4b_detects_missing_and_duplicates(tmp_path: Path) -> None:
    mlruns = tmp_path / "mlruns"
    mlruns.mkdir(parents=True, exist_ok=True)

    configs_dir = tmp_path / "configs"
    experiments_src = Path.cwd() / "configs" / "experiments"
    (configs_dir / "experiments").mkdir(parents=True, exist_ok=True)
    for src in experiments_src.glob("*.json"):
        (configs_dir / "experiments" / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    (configs_dir / "mlflow.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "tracking_dir": "mlruns",
                "experiment_name": "decision_intelligence_engine",
                "run_name_prefix": "mobilenetv2_frozen_baseline",
                "tags": {"project": "test", "phase": "4B"},
            }
        ),
        encoding="utf-8",
    )

    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = mlruns.resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.create_experiment("decision_intelligence_engine")
    client = MlflowClient(tracking_uri=tracking_uri)

    common_params = {
        "data.split_strategy": "deterministic_stratified_file_manifest",
        "data.train_images": "40000",
        "data.val_images": "10000",
        "data.test_images": "10000",
        "verification.model_load_success": "True",
        "verification.inference_batch_success": "True",
        "experiment.freeze_backbone": "True",
        "experiment.epochs": "1",
        "experiment.learning_rate": "0.0003",
        "experiment.dropout_rate": "0.2",
    }
    common_metrics = {
        "eval.train_accuracy": 0.13,
        "eval.val_accuracy": 0.14,
        "eval.test_accuracy": 0.14,
        "eval.train_loss": 2.3,
        "eval.val_loss": 2.31,
        "eval.test_loss": 2.31,
        "training.time_sec": 10.0,
        "training.trainable_params": 12810,
        "training.frozen_params": 2257984,
        "training.total_params": 2270794,
        "training.trainable_backbone_layers": 0,
    }

    base_tags = {
        "phase": "4B",
        "model_family": "MobileNetV2",
        "experiment_category": "control",
        "changed_variable": "none",
        "mlflow.runName": "synthetic_run",
    }

    _create_run(
        client,
        experiment_id,
        {
            "tags": {**base_tags, "experiment_id": "mnetv2_control_frozen"},
            "params": common_params,
            "metrics": common_metrics,
            "status": "FINISHED",
        },
        tmp_path / "artifacts",
    )

    _create_run(
        client,
        experiment_id,
        {
            "tags": {**base_tags, "experiment_id": "mnetv2_control_frozen"},
            "params": common_params,
            "metrics": {**common_metrics, "eval.val_accuracy": 0.12},
            "status": "FINISHED",
        },
        tmp_path / "artifacts",
    )

    payload = compare_phase4b_runs(tmp_path)

    assert payload["experiment_name"] == "decision_intelligence_engine"
    assert "mnetv2_control_frozen" in payload["duplicate_experiment_ids"]
    assert "mnetv2_partial_finetune_tail" in payload["missing_experiment_ids"]
    assert payload["rows"]
