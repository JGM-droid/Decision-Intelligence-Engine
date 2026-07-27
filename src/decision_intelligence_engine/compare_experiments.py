"""Programmatic MLflow comparisons for Phase 4B and Phase 5A runs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from .mlflow_config import load_mlflow_config
from .metric_utils import compute_multiclass_metrics_from_confusion_matrix, load_confusion_matrix_csv


REQUIRED_METRICS = {
    "eval.train_accuracy",
    "eval.val_accuracy",
    "eval.test_accuracy",
    "eval.train_loss",
    "eval.val_loss",
    "eval.test_loss",
    "training.time_sec",
}

EXPECTED_SPLIT_STRATEGY = "deterministic_stratified_file_manifest"
EXPECTED_COUNTS = {"train": "40000", "val": "10000", "test": "10000"}
WARN_GAP_THRESHOLD = 0.10
WARN_DURATION_SECONDS = 45.0
WARN_PARAM_DELTA_RATIO = 0.15
MATERIAL_VAL_ACCURACY_THRESHOLD = 0.01
MATERIAL_VAL_LOSS_THRESHOLD = 0.10

PHASE4B_APPROVED_IDS = [
    "mnetv2_control_frozen",
    "mnetv2_longer_frozen_epochs",
    "mnetv2_lower_lr_frozen",
    "mnetv2_higher_dropout_frozen",
    "mnetv2_partial_finetune_tail",
]


@dataclass(frozen=True)
class ComparisonRow:
    experiment_id: str
    run_id: str
    run_name: str
    experiment_category: str
    changed_variable: str
    backbone_frozen_state: str
    trainable_backbone_layers: int
    epochs: int
    learning_rate: float
    dropout: float
    train_accuracy: float | None
    validation_accuracy: float | None
    test_accuracy: float | None
    train_loss: float | None
    validation_loss: float | None
    test_loss: float | None
    generalization_gap: float | None
    duration_sec: float | None
    trainable_params: int | None
    frozen_params: int | None
    total_params: int | None
    run_status: str
    quality_gate_result: str
    selection_eligibility: str
    notes: str


@dataclass(frozen=True)
class ArchitectureComparisonRow:
    architecture: str
    experiment_id: str
    run_id: str
    run_name: str
    input_resolution: str
    preprocessing: str
    epochs: int
    learning_rate: float
    dropout: float
    batch_size: int
    train_accuracy: float | None
    validation_accuracy: float | None
    test_accuracy: float | None
    test_macro_precision: float | None
    test_macro_recall: float | None
    test_macro_f1: float | None
    train_loss: float | None
    validation_loss: float | None
    test_loss: float | None
    generalization_gap: float | None
    duration_sec: float | None
    trainable_params: int | None
    frozen_params: int | None
    total_params: int | None
    quality_gate_result: str
    model_reload_result: str
    inference_result: str
    run_status: str
    comparison_limitations: str
    selection_eligibility: str
    notes: str


def _parse_bool(value: str | None) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _list_artifact_paths(client: MlflowClient, run_id: str, prefix: str = "") -> list[str]:
    rows: list[str] = []
    for item in client.list_artifacts(run_id, prefix):
        if item.is_dir:
            rows.extend(_list_artifact_paths(client, run_id, item.path))
        else:
            rows.append(item.path)
    return sorted(rows)


def _download_artifact_file(run_id: str, artifact_path: str) -> Path:
    return Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path))


def _load_test_macro_metrics(run: Run, artifact_paths: list[str]) -> tuple[float, float, float, float]:
    confusion_paths = [path for path in artifact_paths if path.startswith("reports/") and path.endswith("_confusion_matrix.csv")]
    if not confusion_paths:
        raise ValueError(f"Missing confusion matrix artifact for run {run.info.run_id}")
    confusion_path = _download_artifact_file(run.info.run_id, confusion_paths[0])
    try:
        matrix = load_confusion_matrix_csv(confusion_path)
        metrics = compute_multiclass_metrics_from_confusion_matrix(matrix)
        return metrics.accuracy, metrics.macro_precision, metrics.macro_recall, metrics.macro_f1
    except ValueError:
        return 0.0, 0.0, 0.0, 0.0


def _fetch_runs_by_experiment_id(client: MlflowClient, mlflow_experiment_id: str, experiment_id: str) -> list[Run]:
    return client.search_runs(
        [mlflow_experiment_id],
        filter_string=f"tags.experiment_id = '{experiment_id}'",
        order_by=["attributes.start_time DESC"],
        max_results=50,
    )


def _evaluate_quality_gates(run: Run, approved_ids: set[str], artifact_paths: list[str]) -> tuple[bool, list[str], list[str]]:
    notes: list[str] = []
    warnings: list[str] = []
    metrics = run.data.metrics
    params = run.data.params
    tags = run.data.tags

    exp_id = tags.get("experiment_id")
    status_ok = run.info.status == "FINISHED"
    if not status_ok:
        notes.append("status_not_finished")

    if exp_id not in approved_ids:
        notes.append("unapproved_experiment_id")

    missing_metrics = [name for name in REQUIRED_METRICS if name not in metrics]
    if missing_metrics:
        notes.append("missing_metrics:" + ",".join(sorted(missing_metrics)))

    model_load_ok = _parse_bool(params.get("verification.model_load_success"))
    inference_ok = _parse_bool(params.get("verification.inference_batch_success"))
    if not model_load_ok:
        notes.append("model_reload_failed")
    if not inference_ok:
        notes.append("inference_failed")

    val_acc = _safe_float(metrics.get("eval.val_accuracy"))
    test_acc = _safe_float(metrics.get("eval.test_accuracy"))
    if val_acc is None or not 0.0 <= val_acc <= 1.0:
        notes.append("invalid_val_accuracy")
    if test_acc is None or not 0.0 <= test_acc <= 1.0:
        notes.append("invalid_test_accuracy")

    for split_name, expected_count in EXPECTED_COUNTS.items():
        key = f"data.{split_name}_images"
        if params.get(key) != expected_count:
            notes.append(f"unexpected_count_{split_name}")

    if params.get("data.split_strategy") != EXPECTED_SPLIT_STRATEGY:
        notes.append("unexpected_split_strategy")

    required_artifacts = {
        "configs/data_pipeline.json",
    }
    has_model = any(path.startswith("model/") and path.endswith(".keras") for path in artifact_paths)
    has_baseline_config = any(
        path in {"configs/baseline_training.yaml", "configs/baseline_training.json"} for path in artifact_paths
    )
    has_reports = {
        "confusion_csv": any(path.startswith("reports/") and path.endswith("_confusion_matrix.csv") for path in artifact_paths),
        "confusion_png": any(path.startswith("reports/") and path.endswith("_confusion_matrix.png") for path in artifact_paths),
        "classification_report": any(path.startswith("reports/") and path.endswith("_classification_report.txt") for path in artifact_paths),
        "metrics_json": any(path.startswith("reports/") and path.endswith("_metrics.json") for path in artifact_paths),
        "history_png": any(path.startswith("reports/") and path.endswith("_training_history.png") for path in artifact_paths),
    }

    if not has_model:
        notes.append("missing_model_artifact")
    if not has_baseline_config:
        notes.append("missing_baseline_training_config")
    for label, present in has_reports.items():
        if not present:
            notes.append(f"missing_{label}")
    missing_fixed_artifacts = [path for path in required_artifacts if path not in artifact_paths]
    if missing_fixed_artifacts:
        notes.append("missing_config_artifacts")

    if val_acc is not None:
        control_val = 0.1347
        if val_acc < control_val:
            warnings.append("val_accuracy_below_control")

    train_acc = _safe_float(metrics.get("eval.train_accuracy"))
    if train_acc is not None and val_acc is not None and abs(train_acc - val_acc) > WARN_GAP_THRESHOLD:
        warnings.append("severe_generalization_gap")

    duration_sec = _safe_float(metrics.get("training.time_sec"))
    if duration_sec is not None and duration_sec > WARN_DURATION_SECONDS:
        warnings.append("high_duration")

    trainable_params = _safe_float(metrics.get("training.trainable_params"))
    frozen_params = _safe_float(metrics.get("training.frozen_params"))
    if trainable_params is not None and frozen_params is not None:
        total = trainable_params + frozen_params
        if total > 0:
            ratio = trainable_params / total
            if ratio > (0.5 + WARN_PARAM_DELTA_RATIO):
                warnings.append("high_trainable_ratio")

    eligible = len(notes) == 0
    return eligible, notes, warnings


def _row_from_run(client: MlflowClient, run: Run, approved_ids: set[str]) -> ComparisonRow:
    metrics = run.data.metrics
    params = run.data.params
    tags = run.data.tags

    artifact_paths = _list_artifact_paths(client, run.info.run_id)
    eligible, failures, warnings = _evaluate_quality_gates(run, approved_ids, artifact_paths)

    train_acc = _safe_float(metrics.get("eval.train_accuracy"))
    val_acc = _safe_float(metrics.get("eval.val_accuracy"))

    notes = failures + warnings
    quality = "PASS" if not failures else "FAIL"

    trainable = _safe_int(metrics.get("training.trainable_params"))
    frozen = _safe_int(metrics.get("training.frozen_params"))
    total = _safe_int(metrics.get("training.total_params"))
    if total is None and trainable is not None and frozen is not None:
        total = trainable + frozen

    return ComparisonRow(
        experiment_id=tags.get("experiment_id", "unknown"),
        run_id=run.info.run_id,
        run_name=tags.get("mlflow.runName", "unknown"),
        experiment_category=tags.get("experiment_category", "unknown"),
        changed_variable=tags.get("changed_variable", "unknown"),
        backbone_frozen_state=params.get("experiment.freeze_backbone", "unknown"),
        trainable_backbone_layers=_safe_int(metrics.get("training.trainable_backbone_layers")) or 0,
        epochs=_safe_int(params.get("experiment.epochs")) or 0,
        learning_rate=_safe_float(params.get("experiment.learning_rate")) or 0.0,
        dropout=_safe_float(params.get("experiment.dropout_rate")) or 0.0,
        train_accuracy=train_acc,
        validation_accuracy=val_acc,
        test_accuracy=_safe_float(metrics.get("eval.test_accuracy")),
        train_loss=_safe_float(metrics.get("eval.train_loss")),
        validation_loss=_safe_float(metrics.get("eval.val_loss")),
        test_loss=_safe_float(metrics.get("eval.test_loss")),
        generalization_gap=None if train_acc is None or val_acc is None else abs(train_acc - val_acc),
        duration_sec=_safe_float(metrics.get("training.time_sec")),
        trainable_params=trainable,
        frozen_params=frozen,
        total_params=total,
        run_status=run.info.status,
        quality_gate_result=quality,
        selection_eligibility="ELIGIBLE" if eligible else "INELIGIBLE",
        notes=";".join(notes) if notes else "ok",
    )


def _deterministic_sort(rows: list[ComparisonRow]) -> list[ComparisonRow]:
    return sorted(rows, key=lambda row: (row.experiment_id, row.run_id))


def _rank_rows(rows: list[ComparisonRow]) -> list[ComparisonRow]:
    eligible = [row for row in rows if row.selection_eligibility == "ELIGIBLE"]

    def key(row: ComparisonRow) -> tuple[float, float, float, float, str]:
        return (
            -(row.validation_accuracy or -1.0),
            row.validation_loss if row.validation_loss is not None else float("inf"),
            row.generalization_gap if row.generalization_gap is not None else float("inf"),
            (row.total_params or 10**12) + (row.duration_sec or 10**9),
            row.experiment_id,
        )

    return sorted(eligible, key=key)


def compare_phase4b_runs(project_root: Path) -> dict[str, Any]:
    mlflow_cfg = load_mlflow_config(project_root)
    tracking_uri = mlflow_cfg.tracking_uri(project_root)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    approved_ids = set(PHASE4B_APPROVED_IDS)
    order_by_id = {exp_id: idx + 1 for idx, exp_id in enumerate(PHASE4B_APPROVED_IDS)}

    experiment = client.get_experiment_by_name(mlflow_cfg.experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {mlflow_cfg.experiment_name}")

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="tags.phase = '4B' and tags.model_family = 'MobileNetV2'",
        order_by=["attributes.start_time DESC"],
        max_results=500,
    )

    latest_by_exp: dict[str, Run] = {}
    duplicates: dict[str, list[str]] = {}
    for run in runs:
        exp_id = run.data.tags.get("experiment_id", "")
        if exp_id not in latest_by_exp:
            latest_by_exp[exp_id] = run
        else:
            duplicates.setdefault(exp_id, []).append(run.info.run_id)

    missing = sorted([exp_id for exp_id in approved_ids if exp_id not in latest_by_exp])

    rows = [_row_from_run(client, run, approved_ids) for run in latest_by_exp.values() if run.data.tags.get("experiment_id") in approved_ids]
    rows = sorted(rows, key=lambda row: order_by_id.get(row.experiment_id, 999))
    ranked = _rank_rows(rows)

    winners = {
        "best_frozen": next((row.experiment_id for row in ranked if row.experiment_category == "frozen_variant" or row.experiment_category == "control"), None),
        "best_finetuned": next((row.experiment_id for row in ranked if row.experiment_category == "fine_tune"), None),
        "best_overall": ranked[0].experiment_id if ranked else None,
    }

    return {
        "tracking_uri": tracking_uri,
        "experiment_name": mlflow_cfg.experiment_name,
        "experiment_id": experiment.experiment_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "required_experiment_ids": sorted(approved_ids, key=lambda exp_id: order_by_id[exp_id]),
        "missing_experiment_ids": missing,
        "duplicate_experiment_ids": duplicates,
        "rows": [row.__dict__ for row in rows],
        "ranked_eligible_experiment_ids": [row.experiment_id for row in ranked],
        "winners": winners,
    }


def _evaluate_architecture_quality_gates(run: Run, artifact_paths: list[str]) -> tuple[str, str, str, bool, list[str], list[str]]:
    metrics = run.data.metrics
    params = run.data.params
    notes: list[str] = []
    warnings: list[str] = []

    missing_metrics = [name for name in REQUIRED_METRICS if name not in metrics]
    if missing_metrics:
        notes.append("missing_metrics:" + ",".join(sorted(missing_metrics)))

    if params.get("data.split_strategy") != EXPECTED_SPLIT_STRATEGY:
        notes.append("unexpected_split_strategy")

    for split_name, expected_count in EXPECTED_COUNTS.items():
        key = f"data.{split_name}_images"
        if params.get(key) != expected_count:
            notes.append(f"unexpected_count_{split_name}")

    model_reload_ok = _parse_bool(params.get("verification.model_load_success"))
    inference_ok = _parse_bool(params.get("verification.inference_batch_success"))
    if not model_reload_ok:
        notes.append("model_reload_failed")
    if not inference_ok:
        notes.append("inference_failed")

    has_model = any(path.startswith("model/") and path.endswith(".keras") for path in artifact_paths)
    has_metrics_json = any(path.startswith("reports/") and path.endswith("_metrics.json") for path in artifact_paths)
    has_confusion_csv = any(path.startswith("reports/") and path.endswith("_confusion_matrix.csv") for path in artifact_paths)
    has_confusion_png = any(path.startswith("reports/") and path.endswith("_confusion_matrix.png") for path in artifact_paths)
    has_classification = any(path.startswith("reports/") and path.endswith("_classification_report.txt") for path in artifact_paths)
    has_history = any(path.startswith("reports/") and path.endswith("_training_history.png") for path in artifact_paths)
    has_data_cfg = "configs/data_pipeline.json" in artifact_paths

    if not has_model:
        notes.append("missing_model_artifact")
    if not has_metrics_json:
        notes.append("missing_metrics_json")
    if not has_confusion_csv:
        notes.append("missing_confusion_csv")
    if not has_confusion_png:
        notes.append("missing_confusion_png")
    if not has_classification:
        notes.append("missing_classification_report")
    if not has_history:
        notes.append("missing_history_plot")
    if not has_data_cfg:
        notes.append("missing_data_pipeline_config")

    train_acc = _safe_float(metrics.get("eval.train_accuracy"))
    val_acc = _safe_float(metrics.get("eval.val_accuracy"))
    if train_acc is not None and val_acc is not None and abs(train_acc - val_acc) > WARN_GAP_THRESHOLD:
        warnings.append("severe_generalization_gap")

    duration = _safe_float(metrics.get("training.time_sec"))
    if duration is not None and duration > WARN_DURATION_SECONDS:
        warnings.append("high_duration")

    quality = "PASS" if not notes else "FAIL"
    eligibility = len(notes) == 0 and run.info.status == "FINISHED"
    return (
        quality,
        "PASS" if model_reload_ok else "FAIL",
        "PASS" if inference_ok else "FAIL",
        eligibility,
        notes,
        warnings,
    )


def _architecture_row_from_run(client: MlflowClient, run: Run, comparison_limitations: str) -> ArchitectureComparisonRow:
    metrics = run.data.metrics
    params = run.data.params
    tags = run.data.tags

    architecture = params.get("experiment.backbone", tags.get("backbone", "unknown"))
    input_resolution = params.get("data.model_input_resolution") or params.get("data.image_size", "unknown")
    preprocessing = params.get("data.preprocessing_function") or params.get("experiment.preprocessing_function")
    if not preprocessing:
        if architecture == "MobileNetV2":
            preprocessing = "mobilenetv2_rescale_neg1_to_1"
        elif architecture == "EfficientNetB0":
            preprocessing = "efficientnetb0_builtin_rescaling_with_input_scale_255"
        else:
            preprocessing = "unknown"

    train_acc = _safe_float(metrics.get("eval.train_accuracy"))
    val_acc = _safe_float(metrics.get("eval.val_accuracy"))

    artifact_paths = _list_artifact_paths(client, run.info.run_id)
    # The runtime MLflow client above is only used for artifact discovery; the metrics come from the
    # saved confusion matrix so the published summary reflects post-hoc test-set evaluation.
    test_metrics = _load_test_macro_metrics(run, artifact_paths)

    return ArchitectureComparisonRow(
        architecture=architecture,
        experiment_id=tags.get("experiment_id", "unknown"),
        run_id=run.info.run_id,
        run_name=tags.get("mlflow.runName", "unknown"),
        input_resolution=str(input_resolution),
        preprocessing=str(preprocessing),
        epochs=_safe_int(params.get("experiment.epochs")) or 0,
        learning_rate=_safe_float(params.get("experiment.learning_rate")) or 0.0,
        dropout=_safe_float(params.get("experiment.dropout_rate")) or 0.0,
        batch_size=_safe_int(params.get("experiment.batch_size")) or 0,
        train_accuracy=train_acc,
        validation_accuracy=val_acc,
        test_accuracy=_safe_float(metrics.get("eval.test_accuracy")),
        test_macro_precision=test_metrics[1],
        test_macro_recall=test_metrics[2],
        test_macro_f1=test_metrics[3],
        train_loss=_safe_float(metrics.get("eval.train_loss")),
        validation_loss=_safe_float(metrics.get("eval.val_loss")),
        test_loss=_safe_float(metrics.get("eval.test_loss")),
        generalization_gap=None if train_acc is None or val_acc is None else abs(train_acc - val_acc),
        duration_sec=_safe_float(metrics.get("training.time_sec")),
        trainable_params=_safe_int(metrics.get("training.trainable_params")),
        frozen_params=_safe_int(metrics.get("training.frozen_params")),
        total_params=_safe_int(metrics.get("training.total_params")),
        quality_gate_result="UNKNOWN",
        model_reload_result="UNKNOWN",
        inference_result="UNKNOWN",
        run_status=run.info.status,
        comparison_limitations=comparison_limitations,
        selection_eligibility="INELIGIBLE",
        notes="ok",
    )


def _inject_quality_results(row: ArchitectureComparisonRow, quality: str, reload_result: str, inference_result: str, eligible: bool, notes: list[str], warnings: list[str]) -> ArchitectureComparisonRow:
    merged_notes = notes + warnings
    return ArchitectureComparisonRow(
        architecture=row.architecture,
        experiment_id=row.experiment_id,
        run_id=row.run_id,
        run_name=row.run_name,
        input_resolution=row.input_resolution,
        preprocessing=row.preprocessing,
        epochs=row.epochs,
        learning_rate=row.learning_rate,
        dropout=row.dropout,
        batch_size=row.batch_size,
        train_accuracy=row.train_accuracy,
        validation_accuracy=row.validation_accuracy,
        test_accuracy=row.test_accuracy,
        test_macro_precision=row.test_macro_precision,
        test_macro_recall=row.test_macro_recall,
        test_macro_f1=row.test_macro_f1,
        train_loss=row.train_loss,
        validation_loss=row.validation_loss,
        test_loss=row.test_loss,
        generalization_gap=row.generalization_gap,
        duration_sec=row.duration_sec,
        trainable_params=row.trainable_params,
        frozen_params=row.frozen_params,
        total_params=row.total_params,
        quality_gate_result=quality,
        model_reload_result=reload_result,
        inference_result=inference_result,
        run_status=row.run_status,
        comparison_limitations=row.comparison_limitations,
        selection_eligibility="ELIGIBLE" if eligible else "INELIGIBLE",
        notes=";".join(merged_notes) if merged_notes else "ok",
    )


def _resolve_architecture_decision(rows: list[ArchitectureComparisonRow]) -> dict[str, Any]:
    by_id = {row.experiment_id: row for row in rows}
    baseline_32 = by_id.get("mnetv2_longer_frozen_epochs")
    baseline_96 = by_id.get("mobilenetv2_96x96_control")
    candidate = by_id.get("efficientnetb0_control_frozen")

    if baseline_32 is None or baseline_96 is None or candidate is None:
        return {
            "material_improvement_threshold": MATERIAL_VAL_ACCURACY_THRESHOLD,
            "decision": "Recommend one additional experiment if a significant confounding variable still exists.",
            "tuning_recommendation": "Recommend one additional experiment if a significant confounding variable still exists.",
            "reason": "Missing required architecture run(s).",
        }

    if (
        baseline_32.selection_eligibility != "ELIGIBLE"
        or baseline_96.selection_eligibility != "ELIGIBLE"
        or candidate.selection_eligibility != "ELIGIBLE"
    ):
        return {
            "material_improvement_threshold": MATERIAL_VAL_ACCURACY_THRESHOLD,
            "decision": "Recommend one additional experiment if a significant confounding variable still exists.",
            "tuning_recommendation": "Recommend one additional experiment if a significant confounding variable still exists.",
            "reason": "One or more architecture runs failed quality gates.",
        }

    target_val_acc = baseline_96.validation_accuracy or -1.0
    cand_val_acc = candidate.validation_accuracy or -1.0
    target_val_loss = baseline_96.validation_loss if baseline_96.validation_loss is not None else float("inf")
    cand_val_loss = candidate.validation_loss if candidate.validation_loss is not None else float("inf")
    resolution_delta = (baseline_96.validation_accuracy or -1.0) - (baseline_32.validation_accuracy or -1.0)

    acc_delta = cand_val_acc - target_val_acc
    loss_delta = target_val_loss - cand_val_loss

    if acc_delta >= MATERIAL_VAL_ACCURACY_THRESHOLD:
        return {
            "material_improvement_threshold": MATERIAL_VAL_ACCURACY_THRESHOLD,
            "decision": "APPROVE EfficientNetB0 as the final selected architecture.",
            "tuning_recommendation": "APPROVE EfficientNetB0 as the final selected architecture.",
            "reason": (
                f"After controlling MobileNetV2 at 96x96, EfficientNetB0 still improves validation accuracy by {acc_delta:.4f} "
                f"(threshold {MATERIAL_VAL_ACCURACY_THRESHOLD:.2f}); MobileNetV2 resolution-only gain was {resolution_delta:.4f}."
            ),
        }

    if abs(acc_delta) < MATERIAL_VAL_ACCURACY_THRESHOLD and loss_delta >= MATERIAL_VAL_LOSS_THRESHOLD:
        return {
            "material_improvement_threshold": MATERIAL_VAL_ACCURACY_THRESHOLD,
            "decision": "APPROVE EfficientNetB0 as the final selected architecture.",
            "tuning_recommendation": "APPROVE EfficientNetB0 as the final selected architecture.",
            "reason": (
                f"Against MobileNetV2 at the same 96x96 input resolution, validation accuracy is comparable (delta {acc_delta:.4f}) "
                f"while EfficientNetB0 validation loss is materially better by {loss_delta:.4f}."
            ),
        }

    return {
        "material_improvement_threshold": MATERIAL_VAL_ACCURACY_THRESHOLD,
        "decision": "Recommend one additional experiment if a significant confounding variable still exists.",
        "tuning_recommendation": "Recommend one additional experiment if a significant confounding variable still exists.",
        "reason": (
            f"After adding the 96x96 MobileNetV2 control, EfficientNetB0 did not achieve material validation improvement over the matched-resolution MobileNetV2 run (delta {acc_delta:.4f}, threshold {MATERIAL_VAL_ACCURACY_THRESHOLD:.2f})."
        ),
    }


def compare_architecture_runs(
    project_root: Path,
    mobile_experiment_id: str = "mnetv2_longer_frozen_epochs",
    mobile_96_experiment_id: str = "mobilenetv2_96x96_control",
    efficientnet_experiment_id: str = "efficientnetb0_control_frozen",
) -> dict[str, Any]:
    mlflow_cfg = load_mlflow_config(project_root)
    tracking_uri = mlflow_cfg.tracking_uri(project_root)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(mlflow_cfg.experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {mlflow_cfg.experiment_name}")

    expected_ids = [mobile_experiment_id, mobile_96_experiment_id, efficientnet_experiment_id]
    missing_ids: list[str] = []
    failed_or_unfinished_ids: list[str] = []
    selected_runs: dict[str, Run] = {}
    latest_status_by_id: dict[str, str] = {}

    for exp_id in expected_ids:
        runs = _fetch_runs_by_experiment_id(client, experiment.experiment_id, exp_id)
        if not runs:
            missing_ids.append(exp_id)
            continue
        latest_status_by_id[exp_id] = runs[0].info.status
        finished_run = next((run for run in runs if run.info.status == "FINISHED"), None)
        if finished_run is None:
            failed_or_unfinished_ids.append(exp_id)
            continue
        selected_runs[exp_id] = finished_run

    rows: list[ArchitectureComparisonRow] = []
    limitations: list[str] = []

    if len(selected_runs) == len(expected_ids):
        mobile_row = _architecture_row_from_run(client, selected_runs[mobile_experiment_id], comparison_limitations="")
        mobile_96_row = _architecture_row_from_run(client, selected_runs[mobile_96_experiment_id], comparison_limitations="")
        efficient_row = _architecture_row_from_run(client, selected_runs[efficientnet_experiment_id], comparison_limitations="")

        if len({mobile_row.input_resolution, mobile_96_row.input_resolution, efficient_row.input_resolution}) > 1:
            limitations.append(
                "At least one comparison pair uses different effective input resolutions."
            )
        if len({mobile_row.preprocessing, mobile_96_row.preprocessing, efficient_row.preprocessing}) > 1:
            limitations.append(
                "At least one comparison pair uses different preprocessing due to backbone requirements."
            )

        comparison_limitations = "; ".join(limitations) if limitations else "none"
        base_rows = {
            mobile_experiment_id: ArchitectureComparisonRow(**{**mobile_row.__dict__, "comparison_limitations": comparison_limitations}),
            mobile_96_experiment_id: ArchitectureComparisonRow(**{**mobile_96_row.__dict__, "comparison_limitations": comparison_limitations}),
            efficientnet_experiment_id: ArchitectureComparisonRow(**{**efficient_row.__dict__, "comparison_limitations": comparison_limitations}),
        }

        for exp_id in expected_ids:
            run = selected_runs[exp_id]
            artifact_paths = _list_artifact_paths(client, run.info.run_id)
            quality, reload_result, inference_result, eligible, notes, warnings = _evaluate_architecture_quality_gates(run, artifact_paths)
            rows.append(
                _inject_quality_results(
                    base_rows[exp_id],
                    quality=quality,
                    reload_result=reload_result,
                    inference_result=inference_result,
                    eligible=eligible,
                    notes=notes,
                    warnings=warnings,
                )
            )

    rows = sorted(rows, key=lambda row: expected_ids.index(row.experiment_id) if row.experiment_id in expected_ids else 99)
    decision = _resolve_architecture_decision(rows)

    return {
        "tracking_uri": tracking_uri,
        "experiment_name": mlflow_cfg.experiment_name,
        "experiment_id": experiment.experiment_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_experiment_ids": expected_ids,
        "missing_experiment_ids": missing_ids,
        "failed_or_unfinished_experiment_ids": failed_or_unfinished_ids,
        "latest_status_by_experiment_id": latest_status_by_id,
        "rows": [row.__dict__ for row in rows],
        "decision": decision,
        "comparison_limitations": limitations,
    }


def _write_reports(project_root: Path, payload: dict[str, Any]) -> dict[str, str]:
    report_dir = project_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / "model_comparison.csv"
    json_path = report_dir / "model_comparison.json"
    md_path = report_dir / "model_comparison.md"

    rows = payload["rows"]
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# MobileNetV2 Experiment Comparison",
        "",
        f"- Experiment: {payload['experiment_name']} ({payload['experiment_id']})",
        f"- Tracking URI: {payload['tracking_uri']}",
        f"- Best overall: {payload['winners']['best_overall']}",
        "",
        "| Experiment ID | Category | Changed Variable | Val Acc | Val Loss | Test Acc | Duration Sec | Eligibility | Notes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        md_lines.append(
            "| {experiment_id} | {experiment_category} | {changed_variable} | {validation_accuracy} | {validation_loss} | {test_accuracy} | {duration_sec} | {selection_eligibility} | {notes} |".format(
                **row
            )
        )

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "md": str(md_path),
    }


def _write_architecture_reports(project_root: Path, payload: dict[str, Any]) -> dict[str, str]:
    report_dir = project_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / "architecture_comparison.csv"
    json_path = report_dir / "architecture_comparison.json"
    md_path = report_dir / "architecture_comparison.md"

    rows = payload["rows"]
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# Architecture Comparison (Phase 5A)",
        "",
        f"- Experiment: {payload['experiment_name']} ({payload['experiment_id']})",
        f"- Tracking URI: {payload['tracking_uri']}",
        f"- Decision: {payload['decision']['decision']}",
        f"- Reason: {payload['decision']['reason']}",
        "- Macro precision/recall/F1 are post-hoc calculations from the saved test confusion matrices, not original MLflow logged metrics.",
        "",
        "| Architecture | Experiment ID | Run ID | Input Resolution | Preprocessing | Val Acc | Test Acc | Macro Precision | Macro Recall | Macro F1 | Val Loss | Test Loss | Duration Sec | Trainable Params | Frozen Params | Eligibility | Notes |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        md_lines.append(
            "| {architecture} | {experiment_id} | {run_id} | {input_resolution} | {preprocessing} | {validation_accuracy} | {test_accuracy} | {test_macro_precision} | {test_macro_recall} | {test_macro_f1} | {validation_loss} | {test_loss} | {duration_sec} | {trainable_params} | {frozen_params} | {selection_eligibility} | {notes} |".format(
                **row
            )
        )

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "md": str(md_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Phase 4B/5A MLflow runs")
    parser.add_argument("--write-reports", action="store_true", help="Write CSV/JSON/Markdown comparison reports")
    parser.add_argument(
        "--architecture",
        action="store_true",
        help="Run Phase 5A architecture comparison (MobileNetV2 selected vs EfficientNetB0 baseline)",
    )
    parser.add_argument(
        "--write-architecture-reports",
        action="store_true",
        help="Write architecture comparison CSV/JSON/Markdown reports",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    if args.architecture:
        payload = compare_architecture_runs(project_root)
        report_paths = _write_architecture_reports(project_root, payload) if args.write_architecture_reports else {}
        print(json.dumps({"architecture_comparison": payload, "report_paths": report_paths}, indent=2))
        return

    payload = compare_phase4b_runs(project_root)
    report_paths = _write_reports(project_root, payload) if args.write_reports else {}
    print(json.dumps({"comparison": payload, "report_paths": report_paths}, indent=2))


if __name__ == "__main__":
    main()
