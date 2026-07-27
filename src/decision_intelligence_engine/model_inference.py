"""Inference utilities for the selected CIFAR-10 classification model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

from .mlflow_config import load_mlflow_config


CANONICAL_CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
SELECTED_EFFICIENTNET_EXPERIMENT_ID = "efficientnetb0_control_frozen"
ARCHITECTURE_REPORT_PATH = Path("reports") / "architecture_comparison.json"


class ModelResolutionError(RuntimeError):
    """Raised when the selected model artifact cannot be resolved."""


class InvalidImageError(RuntimeError):
    """Raised when an input image cannot be validated or decoded."""


@dataclass(frozen=True)
class TopPrediction:
    class_name: str
    class_index: int
    confidence: float


@dataclass(frozen=True)
class ModelArtifactReference:
    model_path: Path
    source: str
    run_id: str | None
    run_name: str | None
    experiment_id: str | None
    architecture: str | None
    dataset_input_resolution: tuple[int, int] | None
    model_input_resolution: tuple[int, int] | None
    preprocessing_function: str | None
    class_names: tuple[str, ...]
    metadata_path: Path | None


@dataclass(frozen=True)
class PredictionResult:
    predicted_class: str
    class_index: int
    confidence: float
    top_predictions: tuple[TopPrediction, ...]
    run_id: str | None
    run_name: str | None
    experiment_id: str | None
    architecture: str | None
    model_path: str
    preprocessing_function: str | None
    dataset_input_resolution: tuple[int, int] | None
    model_input_resolution: tuple[int, int] | None


def _list_artifact_paths(client: MlflowClient, run_id: str, prefix: str = "") -> list[str]:
    paths: list[str] = []
    for item in client.list_artifacts(run_id, prefix):
        if item.is_dir:
            paths.extend(_list_artifact_paths(client, run_id, item.path))
        else:
            paths.append(item.path)
    return sorted(paths)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelResolutionError(f"Malformed JSON evidence file: {path}") from exc


def _to_resolution(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return int(value[0]), int(value[1])


def _extract_reference_from_report(report_payload: dict[str, Any]) -> dict[str, Any]:
    rows = report_payload.get("rows", [])
    if not isinstance(rows, list):
        raise ModelResolutionError(
            f"Malformed architecture comparison evidence in {ARCHITECTURE_REPORT_PATH}: rows must be a list."
        )
    for row in rows:
        if row.get("experiment_id") == SELECTED_EFFICIENTNET_EXPERIMENT_ID:
            if not row.get("run_id"):
                raise ModelResolutionError(
                    f"Malformed selected EfficientNetB0 row in {ARCHITECTURE_REPORT_PATH}: missing run_id."
                )
            return row
    raise ModelResolutionError(
        f"Selected EfficientNetB0 row not found in {ARCHITECTURE_REPORT_PATH}."
    )


def _metadata_from_metrics(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    architecture = payload.get("architecture", {})
    class_names = tuple(payload.get("class_names", ()))
    if not class_names:
        class_names = CANONICAL_CIFAR10_CLASSES
    return {
        "architecture": payload.get("experiment", {}).get("backbone"),
        "dataset_input_resolution": _to_resolution(architecture.get("dataset_input_resolution")),
        "model_input_resolution": _to_resolution(architecture.get("model_input_resolution")),
        "preprocessing_function": architecture.get("preprocessing_function"),
        "class_names": class_names,
    }


def resolve_selected_model_artifact(
    project_root: Path,
    model_path_override: str | Path | None = None,
) -> ModelArtifactReference:
    """Resolve the final selected EfficientNetB0 model artifact.

    Resolution order:
    1. Explicit model path override.
    2. Selected architecture-comparison evidence.
    3. MLflow artifact download for the selected run.
    """

    if model_path_override is not None:
        override = Path(model_path_override)
        if not override.is_absolute():
            override = (project_root / override).resolve()
        if not override.exists():
            raise ModelResolutionError(f"Model override path does not exist: {override}")
        if override.suffix != ".keras":
            raise ModelResolutionError("Model override must point to a .keras file")
        return ModelArtifactReference(
            model_path=override,
            source="override",
            run_id=None,
            run_name=None,
            experiment_id=None,
            architecture="EfficientNetB0",
            dataset_input_resolution=None,
            model_input_resolution=None,
            preprocessing_function=None,
            class_names=CANONICAL_CIFAR10_CLASSES,
            metadata_path=None,
        )

    report_path = project_root / ARCHITECTURE_REPORT_PATH
    if not report_path.exists():
        raise ModelResolutionError(f"Architecture comparison report not found: {report_path}")

    report_payload = _read_json(report_path)
    reference = _extract_reference_from_report(report_payload)
    if reference.get("run_status") != "FINISHED":
        raise ModelResolutionError("Selected EfficientNetB0 run is not FINISHED.")

    run_id = str(reference["run_id"])
    mlflow_cfg = load_mlflow_config(project_root)
    tracking_uri = mlflow_cfg.tracking_uri(project_root)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    artifact_paths = _list_artifact_paths(client, run_id)

    model_artifacts = [path for path in artifact_paths if path.startswith("model/") and path.endswith(".keras")]
    metrics_artifacts = [path for path in artifact_paths if path.startswith("reports/") and path.endswith("_metrics.json")]
    if not model_artifacts:
        raise ModelResolutionError(f"No .keras model artifact found for run {run_id}")
    if not metrics_artifacts:
        raise ModelResolutionError(f"No metrics artifact found for run {run_id}")
    model_artifacts = sorted(model_artifacts)
    metrics_artifacts = sorted(metrics_artifacts)

    cache_dir = project_root / "data" / "processed" / "inference_cache" / run_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_model_path = Path(
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=model_artifacts[0],
            dst_path=str(cache_dir),
        )
    )
    local_metrics_path = Path(
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=metrics_artifacts[0],
            dst_path=str(cache_dir),
        )
    )

    metadata = _metadata_from_metrics(local_metrics_path)
    return ModelArtifactReference(
        model_path=local_model_path,
        source="mlflow-selected-run",
        run_id=run_id,
        run_name=reference.get("run_name"),
        experiment_id=reference.get("experiment_id"),
        architecture=metadata["architecture"],
        dataset_input_resolution=metadata["dataset_input_resolution"],
        model_input_resolution=metadata["model_input_resolution"],
        preprocessing_function=metadata["preprocessing_function"],
        class_names=metadata["class_names"],
        metadata_path=local_metrics_path,
    )


class ModelInferenceService:
    """Load the selected classifier and run deterministic image inference."""

    def __init__(self, project_root: Path, model_path_override: str | Path | None = None) -> None:
        self.project_root = project_root
        self.reference = resolve_selected_model_artifact(project_root, model_path_override=model_path_override)
        self._model: tf.keras.Model | None = None

    def load_model(self) -> tf.keras.Model:
        if self._model is None:
            if not self.reference.model_path.exists():
                raise ModelResolutionError(f"Resolved model path does not exist: {self.reference.model_path}")
            self._model = tf.keras.models.load_model(self.reference.model_path)
            if self.reference.dataset_input_resolution is None:
                shape = self._model.input_shape
                if not isinstance(shape, tuple) or len(shape) != 4 or shape[1] is None or shape[2] is None:
                    raise ModelResolutionError("Loaded model has unexpected input shape")
                self.reference = ModelArtifactReference(
                    model_path=self.reference.model_path,
                    source=self.reference.source,
                    run_id=self.reference.run_id,
                    run_name=self.reference.run_name,
                    experiment_id=self.reference.experiment_id,
                    architecture=self.reference.architecture,
                    dataset_input_resolution=(int(shape[1]), int(shape[2])),
                    model_input_resolution=self.reference.model_input_resolution,
                    preprocessing_function=self.reference.preprocessing_function,
                    class_names=self.reference.class_names,
                    metadata_path=self.reference.metadata_path,
                )
        return self._model

    def _prepare_image(self, image_path: str | Path) -> np.ndarray:
        path = Path(image_path)
        if not path.exists():
            raise InvalidImageError(f"Image file does not exist: {path}")
        if path.is_dir():
            raise InvalidImageError(f"Image path points to a directory: {path}")

        resolution = self.reference.dataset_input_resolution or (32, 32)
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                resized = rgb.resize(resolution, Image.Resampling.BILINEAR)
                array = np.asarray(resized, dtype=np.float32) / 255.0
        except UnidentifiedImageError as exc:
            raise InvalidImageError(f"Unsupported or corrupted image file: {path}") from exc
        except OSError as exc:
            raise InvalidImageError(f"Unable to read image file: {path}") from exc

        if array.shape != (resolution[0], resolution[1], 3):
            raise InvalidImageError(f"Unexpected image tensor shape: {array.shape}")
        return np.expand_dims(array, axis=0)

    def predict(self, image_path: str | Path, top_k: int = 3) -> PredictionResult:
        if top_k <= 0:
            raise ValueError("top_k must be between 1 and 10")
        if top_k > len(self.reference.class_names):
            raise ValueError(f"top_k must be between 1 and {len(self.reference.class_names)}")

        model = self.load_model()
        batch = self._prepare_image(image_path)
        probabilities = model.predict(batch, verbose=0)
        if probabilities.shape[0] != 1 or probabilities.ndim != 2:
            raise RuntimeError(f"Unexpected prediction shape: {probabilities.shape}")

        scores = probabilities[0]
        if len(scores) != len(self.reference.class_names):
            raise RuntimeError(
                f"Prediction class count {len(scores)} does not match class mapping {len(self.reference.class_names)}"
            )

        sorted_indices = list(np.argsort(scores)[::-1])
        top_predictions = tuple(
            TopPrediction(
                class_name=self.reference.class_names[index],
                class_index=int(index),
                confidence=float(scores[index]),
            )
            for index in sorted_indices[: min(top_k, len(sorted_indices))]
        )
        best = top_predictions[0]
        return PredictionResult(
            predicted_class=best.class_name,
            class_index=best.class_index,
            confidence=best.confidence,
            top_predictions=top_predictions,
            run_id=self.reference.run_id,
            run_name=self.reference.run_name,
            experiment_id=self.reference.experiment_id,
            architecture=self.reference.architecture,
            model_path=str(self.reference.model_path),
            preprocessing_function=self.reference.preprocessing_function,
            dataset_input_resolution=self.reference.dataset_input_resolution,
            model_input_resolution=self.reference.model_input_resolution,
        )