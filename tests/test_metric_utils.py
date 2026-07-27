from __future__ import annotations

from pathlib import Path

import pytest

from src.decision_intelligence_engine.compare_experiments import _write_architecture_reports
from src.decision_intelligence_engine.metric_utils import compute_multiclass_metrics, compute_multiclass_metrics_from_confusion_matrix


def test_metric_computation_respects_class_order_and_zero_division() -> None:
    class_names = ("airplane", "automobile", "bird")
    y_true = [0, 0, 1, 2, 2]
    y_pred = [0, 0, 0, 0, 0]

    metrics, confusion = compute_multiclass_metrics(y_true, y_pred, class_names)

    assert confusion.tolist() == [[2, 0, 0], [1, 0, 0], [2, 0, 0]]
    assert metrics.accuracy == pytest.approx(0.4)
    assert metrics.macro_precision == pytest.approx((0.4 + 0.0 + 0.0) / 3)
    assert metrics.macro_recall == pytest.approx((1.0 + 0.0 + 0.0) / 3)
    assert metrics.macro_f1 == pytest.approx(((2 * 0.4 * 1.0) / (0.4 + 1.0) + 0.0 + 0.0) / 3)


def test_confusion_matrix_metrics_are_deterministic() -> None:
    matrix = [[2, 0, 0], [1, 0, 0], [2, 0, 0]]
    first = compute_multiclass_metrics_from_confusion_matrix(matrix)
    second = compute_multiclass_metrics_from_confusion_matrix(matrix)
    assert first == second


def test_architecture_report_output_is_deterministic(tmp_path: Path) -> None:
    payload = {
        "experiment_name": "demo",
        "experiment_id": "exp",
        "tracking_uri": "file:///tmp/mlruns",
        "decision": {"decision": "APPROVE", "reason": "demo reason"},
        "rows": [
            {
                "architecture": "MobileNetV2",
                "experiment_id": "mnetv2_longer_frozen_epochs",
                "run_id": "run-1",
                "run_name": "demo-run",
                "input_resolution": "[32, 32]",
                "preprocessing": "mobilenetv2_rescale_neg1_to_1",
                "epochs": 3,
                "learning_rate": 0.0003,
                "dropout": 0.2,
                "batch_size": 128,
                "train_accuracy": 0.1,
                "validation_accuracy": 0.2,
                "test_accuracy": 0.3,
                "test_macro_precision": 0.4,
                "test_macro_recall": 0.5,
                "test_macro_f1": 0.6,
                "train_loss": 2.0,
                "validation_loss": 1.9,
                "test_loss": 1.8,
                "generalization_gap": 0.1,
                "duration_sec": 10.0,
                "trainable_params": 10,
                "frozen_params": 20,
                "total_params": 30,
                "quality_gate_result": "PASS",
                "model_reload_result": "PASS",
                "inference_result": "PASS",
                "run_status": "FINISHED",
                "comparison_limitations": "none",
                "selection_eligibility": "ELIGIBLE",
                "notes": "ok",
            }
        ],
    }

    first = _write_architecture_reports(tmp_path, payload)
    second = _write_architecture_reports(tmp_path, payload)

    assert Path(first["md"]).read_text(encoding="utf-8") == Path(second["md"]).read_text(encoding="utf-8")
    assert Path(first["csv"]).read_text(encoding="utf-8") == Path(second["csv"]).read_text(encoding="utf-8")
