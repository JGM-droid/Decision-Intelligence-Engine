"""Programmatic MLflow comparison for Phase 4B MobileNetV2 matrix runs."""

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

from .experiment_config import load_experiment_configs
from .mlflow_config import load_mlflow_config


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
        "configs/baseline_training.json",
        "configs/data_pipeline.json",
    }
    has_model = any(path.startswith("model/") and path.endswith(".keras") for path in artifact_paths)
    has_reports = {
        "confusion_csv": any(path.startswith("reports/") and path.endswith("_confusion_matrix.csv") for path in artifact_paths),
        "confusion_png": any(path.startswith("reports/") and path.endswith("_confusion_matrix.png") for path in artifact_paths),
        "classification_report": any(path.startswith("reports/") and path.endswith("_classification_report.txt") for path in artifact_paths),
        "metrics_json": any(path.startswith("reports/") and path.endswith("_metrics.json") for path in artifact_paths),
        "history_png": any(path.startswith("reports/") and path.endswith("_training_history.png") for path in artifact_paths),
    }

    if not has_model:
        notes.append("missing_model_artifact")
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

    configs = load_experiment_configs(project_root)
    approved_ids = {cfg.experiment_id for cfg in configs}
    order_by_id = {cfg.experiment_id: cfg.order for cfg in configs}

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Phase 4B MobileNetV2 runs from MLflow")
    parser.add_argument("--write-reports", action="store_true", help="Write CSV/JSON/Markdown comparison reports")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    payload = compare_phase4b_runs(project_root)
    report_paths = _write_reports(project_root, payload) if args.write_reports else {}

    print(json.dumps({"comparison": payload, "report_paths": report_paths}, indent=2))


if __name__ == "__main__":
    main()
