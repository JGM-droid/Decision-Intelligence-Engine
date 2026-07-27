"""Reusable multiclass evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import csv
import numpy as np
from sklearn.metrics import confusion_matrix


@dataclass(frozen=True)
class MulticlassMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float


def load_confusion_matrix_csv(path: Path) -> np.ndarray:
    raw_rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if row:
                raw_rows.append([value.strip() for value in row])

    def parse_numeric_rows(rows: list[list[str]]) -> np.ndarray:
        parsed_rows: list[list[int]] = []
        for row in rows:
            parsed_rows.append([int(float(value)) for value in row])
        matrix = np.asarray(parsed_rows, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Confusion matrix must be square: {path}")
        return matrix

    try:
        return parse_numeric_rows(raw_rows)
    except ValueError as raw_error:
        if len(raw_rows) < 2 or len(raw_rows[0]) < 2:
            raise raw_error

        try:
            return parse_numeric_rows([row[1:] for row in raw_rows[1:]])
        except ValueError:
            raise raw_error


def compute_multiclass_metrics_from_confusion_matrix(confusion: Sequence[Sequence[int]]) -> MulticlassMetrics:
    matrix = np.asarray(confusion, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Confusion matrix must be square.")
    total = float(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0

    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []

    for class_index in range(matrix.shape[0]):
        true_positive = float(matrix[class_index, class_index])
        false_positive = float(matrix[:, class_index].sum() - true_positive)
        false_negative = float(matrix[class_index, :].sum() - true_positive)

        precision_denom = true_positive + false_positive
        recall_denom = true_positive + false_negative

        precision = true_positive / precision_denom if precision_denom else 0.0
        recall = true_positive / recall_denom if recall_denom else 0.0
        f1_denom = precision + recall
        f1 = (2.0 * precision * recall / f1_denom) if f1_denom else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return MulticlassMetrics(
        accuracy=accuracy,
        macro_precision=float(np.mean(precisions)) if precisions else 0.0,
        macro_recall=float(np.mean(recalls)) if recalls else 0.0,
        macro_f1=float(np.mean(f1_scores)) if f1_scores else 0.0,
    )


def compute_multiclass_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> tuple[MulticlassMetrics, np.ndarray]:
    if not class_names:
        raise ValueError("class_names must not be empty.")
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return compute_multiclass_metrics_from_confusion_matrix(matrix), matrix
