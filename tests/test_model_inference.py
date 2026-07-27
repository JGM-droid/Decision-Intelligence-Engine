from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.decision_intelligence_engine.model_inference import (
    CANONICAL_CIFAR10_CLASSES,
    InvalidImageError,
    ModelArtifactReference,
    ModelInferenceService,
    ModelResolutionError,
    resolve_selected_model_artifact,
)


def _write_fake_image(path: Path) -> None:
    from PIL import Image

    array = np.full((40, 40, 3), 128, dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def _write_fake_model(path: Path) -> None:
    tf = pytest.importorskip("tensorflow")
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Flatten()(inputs)
    outputs = tf.keras.layers.Dense(10, activation="softmax", bias_initializer="zeros")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.save(path)


def test_explicit_model_path_override(tmp_path: Path) -> None:
    model_path = tmp_path / "fake.keras"
    _write_fake_model(model_path)

    reference = resolve_selected_model_artifact(tmp_path, model_path_override=model_path)
    assert reference.model_path == model_path
    assert reference.source == "override"
    assert reference.class_names == CANONICAL_CIFAR10_CLASSES


def test_missing_model_override_raises(tmp_path: Path) -> None:
    with pytest.raises(ModelResolutionError):
        resolve_selected_model_artifact(tmp_path, model_path_override=tmp_path / "missing.keras")


def test_inference_valid_image_with_fake_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path = tmp_path / "fake.keras"
    image_path = tmp_path / "image.png"
    _write_fake_model(model_path)
    _write_fake_image(image_path)

    reference = ModelArtifactReference(
        model_path=model_path,
        source="override",
        run_id="run123",
        run_name="demo",
        experiment_id="efficientnetb0_control_frozen",
        architecture="EfficientNetB0",
        dataset_input_resolution=(32, 32),
        model_input_resolution=(96, 96),
        preprocessing_function="efficientnetb0_builtin_rescaling_with_input_scale_255",
        class_names=CANONICAL_CIFAR10_CLASSES,
        metadata_path=None,
    )
    monkeypatch.setattr(
        "src.decision_intelligence_engine.model_inference.resolve_selected_model_artifact",
        lambda project_root, model_path_override=None: reference,
    )

    service = ModelInferenceService(tmp_path)
    result = service.predict(image_path, top_k=3)
    assert result.predicted_class in CANONICAL_CIFAR10_CLASSES
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.top_predictions) == 3
    assert result.preprocessing_function == "efficientnetb0_builtin_rescaling_with_input_scale_255"


def test_invalid_image_path_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path = tmp_path / "fake.keras"
    _write_fake_model(model_path)
    reference = ModelArtifactReference(
        model_path=model_path,
        source="override",
        run_id=None,
        run_name=None,
        experiment_id=None,
        architecture="EfficientNetB0",
        dataset_input_resolution=(32, 32),
        model_input_resolution=(96, 96),
        preprocessing_function="efficientnetb0_builtin_rescaling_with_input_scale_255",
        class_names=CANONICAL_CIFAR10_CLASSES,
        metadata_path=None,
    )
    monkeypatch.setattr(
        "src.decision_intelligence_engine.model_inference.resolve_selected_model_artifact",
        lambda project_root, model_path_override=None: reference,
    )
    service = ModelInferenceService(tmp_path)
    with pytest.raises(InvalidImageError):
        service.predict(tmp_path / "missing.png")


def test_corrupted_image_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path = tmp_path / "fake.keras"
    image_path = tmp_path / "broken.png"
    _write_fake_model(model_path)
    image_path.write_text("not an image", encoding="utf-8")
    reference = ModelArtifactReference(
        model_path=model_path,
        source="override",
        run_id=None,
        run_name=None,
        experiment_id=None,
        architecture="EfficientNetB0",
        dataset_input_resolution=(32, 32),
        model_input_resolution=(96, 96),
        preprocessing_function="efficientnetb0_builtin_rescaling_with_input_scale_255",
        class_names=CANONICAL_CIFAR10_CLASSES,
        metadata_path=None,
    )
    monkeypatch.setattr(
        "src.decision_intelligence_engine.model_inference.resolve_selected_model_artifact",
        lambda project_root, model_path_override=None: reference,
    )
    service = ModelInferenceService(tmp_path)
    with pytest.raises(InvalidImageError):
        service.predict(image_path)


def test_selected_model_resolution_uses_architecture_report_and_mlflow_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "rows": [
            {
                "experiment_id": "efficientnetb0_control_frozen",
                "run_id": "run-1",
                "run_name": "selected",
                "run_status": "FINISHED",
            }
        ]
    }
    (report_dir / "architecture_comparison.json").write_text(json.dumps(report_payload), encoding="utf-8")

    download_root = tmp_path / "downloads"
    download_root.mkdir(parents=True, exist_ok=True)
    model_file = download_root / "model" / "selected.keras"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    _write_fake_model(model_file)
    metrics_file = download_root / "reports" / "selected_metrics.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(
        json.dumps(
            {
                "experiment": {"backbone": "EfficientNetB0"},
                "architecture": {
                    "dataset_input_resolution": [32, 32],
                    "model_input_resolution": [96, 96],
                    "preprocessing_function": "efficientnetb0_builtin_rescaling_with_input_scale_255",
                },
                "class_names": list(CANONICAL_CIFAR10_CLASSES),
            }
        ),
        encoding="utf-8",
    )

    class FakeClient:
        def list_artifacts(self, run_id: str, prefix: str = ""):
            class Item:
                def __init__(self, path: str, is_dir: bool):
                    self.path = path
                    self.is_dir = is_dir

            if prefix == "":
                return [Item("model", True), Item("reports", True)]
            if prefix == "model":
                return [Item("model/selected.keras", False)]
            if prefix == "reports":
                return [Item("reports/selected_metrics.json", False)]
            return []

    monkeypatch.setattr("src.decision_intelligence_engine.model_inference.MlflowClient", lambda tracking_uri=None: FakeClient())
    monkeypatch.setattr(
        "src.decision_intelligence_engine.model_inference.mlflow.artifacts.download_artifacts",
        lambda run_id, artifact_path, dst_path: str(model_file if artifact_path.endswith(".keras") else metrics_file),
    )

    reference = resolve_selected_model_artifact(tmp_path)
    assert reference.run_id == "run-1"
    assert reference.model_path.name == "selected.keras"
    assert reference.model_input_resolution == (96, 96)
    assert reference.class_names == CANONICAL_CIFAR10_CLASSES


def test_malformed_architecture_report_raises_clear_error(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "architecture_comparison.json").write_text("{broken json", encoding="utf-8")

    with pytest.raises(ModelResolutionError, match="Malformed JSON evidence file"):
        resolve_selected_model_artifact(tmp_path)


def test_missing_model_artifact_raises_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "architecture_comparison.json").write_text(
        json.dumps({"rows": [{"experiment_id": "efficientnetb0_control_frozen", "run_id": "run-1", "run_status": "FINISHED"}]}),
        encoding="utf-8",
    )

    class FakeClient:
        def list_artifacts(self, run_id: str, prefix: str = ""):
            class Item:
                def __init__(self, path: str, is_dir: bool):
                    self.path = path
                    self.is_dir = is_dir

            if prefix == "":
                return [Item("reports", True)]
            if prefix == "reports":
                return [Item("reports/selected_metrics.json", False)]
            return []

    monkeypatch.setattr("src.decision_intelligence_engine.model_inference.MlflowClient", lambda tracking_uri=None: FakeClient())

    with pytest.raises(ModelResolutionError, match=r"No \.keras model artifact found"):
        resolve_selected_model_artifact(tmp_path)


def test_class_order_prefers_training_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "architecture_comparison.json").write_text(
        json.dumps({"rows": [{"experiment_id": "efficientnetb0_control_frozen", "run_id": "run-1", "run_status": "FINISHED"}]}),
        encoding="utf-8",
    )

    download_root = tmp_path / "downloads"
    download_root.mkdir(parents=True, exist_ok=True)
    model_file = download_root / "model" / "selected.keras"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    _write_fake_model(model_file)
    reversed_classes = list(reversed(CANONICAL_CIFAR10_CLASSES))
    metrics_file = download_root / "reports" / "selected_metrics.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(
        json.dumps(
            {
                "experiment": {"backbone": "EfficientNetB0"},
                "architecture": {"dataset_input_resolution": [32, 32], "model_input_resolution": [96, 96]},
                "class_names": reversed_classes,
            }
        ),
        encoding="utf-8",
    )

    class FakeClient:
        def list_artifacts(self, run_id: str, prefix: str = ""):
            class Item:
                def __init__(self, path: str, is_dir: bool):
                    self.path = path
                    self.is_dir = is_dir

            if prefix == "":
                return [Item("model", True), Item("reports", True)]
            if prefix == "model":
                return [Item("model/selected.keras", False)]
            if prefix == "reports":
                return [Item("reports/selected_metrics.json", False)]
            return []

    monkeypatch.setattr("src.decision_intelligence_engine.model_inference.MlflowClient", lambda tracking_uri=None: FakeClient())
    monkeypatch.setattr(
        "src.decision_intelligence_engine.model_inference.mlflow.artifacts.download_artifacts",
        lambda run_id, artifact_path, dst_path: str(model_file if artifact_path.endswith(".keras") else metrics_file),
    )

    reference = resolve_selected_model_artifact(tmp_path)
    assert reference.class_names == tuple(reversed_classes)


def test_top_k_range_is_validated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path = tmp_path / "fake.keras"
    image_path = tmp_path / "image.png"
    _write_fake_model(model_path)
    _write_fake_image(image_path)
    reference = ModelArtifactReference(
        model_path=model_path,
        source="override",
        run_id=None,
        run_name=None,
        experiment_id=None,
        architecture="EfficientNetB0",
        dataset_input_resolution=(32, 32),
        model_input_resolution=(96, 96),
        preprocessing_function="efficientnetb0_builtin_rescaling_with_input_scale_255",
        class_names=CANONICAL_CIFAR10_CLASSES,
        metadata_path=None,
    )
    monkeypatch.setattr(
        "src.decision_intelligence_engine.model_inference.resolve_selected_model_artifact",
        lambda project_root, model_path_override=None: reference,
    )
    service = ModelInferenceService(tmp_path)
    with pytest.raises(ValueError, match="top_k"):
        service.predict(image_path, top_k=11)