from __future__ import annotations

from pathlib import Path
import sys

import pytest

from src.decision_intelligence_engine import ask
from src.decision_intelligence_engine.model_inference import PredictionResult, TopPrediction
from src.decision_intelligence_engine.request_parser import Intent, parse_request


def _prediction(predicted_class: str = "frog") -> PredictionResult:
    classes = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    top = (
        TopPrediction(predicted_class, classes.index(predicted_class), 0.4063),
        TopPrediction("cat", 3, 0.201),
        TopPrediction("dog", 5, 0.101),
    )
    return PredictionResult(
        predicted_class=predicted_class,
        class_index=classes.index(predicted_class),
        confidence=0.4063,
        top_predictions=top,
        run_id="run1",
        run_name="selected",
        experiment_id="efficientnetb0_control_frozen",
        architecture="EfficientNetB0",
        model_path="model.keras",
        preprocessing_function="efficientnetb0_builtin_rescaling_with_input_scale_255",
        dataset_input_resolution=(32, 32),
        model_input_resolution=(96, 96),
    )


class FakeInferenceService:
    calls: list[tuple[str, int]] = []

    def __init__(self, project_root: Path, model_path_override: str | None = None) -> None:
        self.project_root = project_root
        self.model_path_override = model_path_override

    def predict(self, image_path: str, top_k: int = 3) -> PredictionResult:
        FakeInferenceService.calls.append((image_path, top_k))
        return _prediction()


class FakeExplainer:
    calls: list[str | None] = []

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name

    def explain(self, prediction: PredictionResult, question: str | None, top_k: int = 3) -> str:
        FakeExplainer.calls.append(question)
        return "Fake explanation."


@pytest.fixture(autouse=True)
def reset_fakes() -> None:
    FakeInferenceService.calls = []
    FakeExplainer.calls = []


@pytest.fixture
def patched_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ask, "ModelInferenceService", FakeInferenceService)
    monkeypatch.setattr(ask, "OpenAIExplainer", FakeExplainer)


def _run_cli(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["ask.py", *argv])
    return ask.main()


def test_normal_classification_request_runs_classifier_and_llm(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], patched_boundaries: None
) -> None:
    exit_code = _run_cli(
        monkeypatch,
        ["--question", "What is shown in this image?", "--image", "demo.png"],
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert FakeInferenceService.calls == [("demo.png", 3)]
    assert len(FakeExplainer.calls) == 1
    assert "Predicted class: frog" in out
    assert "The classifier predicts 'frog'" in out
    assert "Explanation:" in out


def test_target_class_match_prediction(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], patched_boundaries: None
) -> None:
    exit_code = _run_cli(
        monkeypatch,
        ["--question", "Does this image contain a frog?", "--image", "demo.png"],
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "The classifier predicts 'frog' with 40.63% confidence." in out
    assert "No, there is no frog" not in out


def test_target_class_differs_from_prediction(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], patched_boundaries: None
) -> None:
    exit_code = _run_cli(
        monkeypatch,
        ["--question", "Does this image contain a dog?", "--image", "demo.png"],
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "The classifier predicts 'frog' with 40.63% confidence rather than 'dog'." in out
    assert "no dog" not in out.lower()


def test_no_llm_classifier_only_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], patched_boundaries: None
) -> None:
    exit_code = _run_cli(
        monkeypatch,
        ["--question", "What is shown in this image?", "--image", "demo.png", "--no-llm"],
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert FakeInferenceService.calls == [("demo.png", 3)]
    assert FakeExplainer.calls == []
    assert "Explanation:" not in out


def test_missing_image_fails_before_classifier_and_openai(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], patched_boundaries: None
) -> None:
    exit_code = _run_cli(monkeypatch, ["--question", "What is shown in this image?"])
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "image is required" in err
    assert FakeInferenceService.calls == []
    assert FakeExplainer.calls == []


def test_unsupported_request_skips_classifier_and_openai(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], patched_boundaries: None
) -> None:
    exit_code = _run_cli(monkeypatch, ["--question", "What will the weather be tomorrow?"])
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "Unsupported request" in err
    assert FakeInferenceService.calls == []
    assert FakeExplainer.calls == []


def test_ambiguous_request_skips_classifier_and_openai(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], patched_boundaries: None
) -> None:
    exit_code = _run_cli(monkeypatch, ["--question", "Tell me about this."])
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "Ambiguous request" in err
    assert FakeInferenceService.calls == []
    assert FakeExplainer.calls == []


def test_empty_question_skips_classifier_and_openai(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], patched_boundaries: None
) -> None:
    exit_code = _run_cli(monkeypatch, ["--question", "   "])
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "Ambiguous request" in err
    assert FakeInferenceService.calls == []
    assert FakeExplainer.calls == []


def test_inference_error_is_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], patched_boundaries: None
) -> None:
    class BrokenInference:
        def __init__(self, project_root: Path, model_path_override: str | None = None) -> None:
            pass

        def predict(self, image_path: str, top_k: int = 3) -> PredictionResult:
            raise ValueError("bad image")

    monkeypatch.setattr(ask, "ModelInferenceService", BrokenInference)
    exit_code = _run_cli(
        monkeypatch,
        ["--question", "What is shown in this image?", "--image", "demo.png"],
    )
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "Error: bad image" in err


def test_routing_uses_real_parser(monkeypatch: pytest.MonkeyPatch, patched_boundaries: None) -> None:
    """The CLI must route on the real parse_request output, not the raw string."""
    captured: list[str] = []

    real_parse = ask.parse_request

    def recording_parse(text: str | None):
        request = real_parse(text)
        captured.append(request.intent.value)
        return request

    monkeypatch.setattr(ask, "parse_request", recording_parse)
    _run_cli(monkeypatch, ["--question", "What will the weather be tomorrow?"])
    assert captured == [Intent.UNSUPPORTED.value]

    _run_cli(monkeypatch, ["--question", "Does this image contain a frog?", "--image", "demo.png"])
    assert captured[-1] == Intent.CLASSIFY_IMAGE.value
