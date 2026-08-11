from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

from src.decision_intelligence_engine.explain_image import _load_local_environment, main
from src.decision_intelligence_engine.model_inference import PredictionResult, TopPrediction


def _prediction() -> PredictionResult:
    top = (
        TopPrediction("cat", 3, 0.544),
        TopPrediction("dog", 5, 0.201),
        TopPrediction("deer", 4, 0.101),
    )
    return PredictionResult(
        predicted_class="cat",
        class_index=3,
        confidence=0.544,
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
    def __init__(self, project_root: Path, model_path_override: str | None = None) -> None:
        self.project_root = project_root
        self.model_path_override = model_path_override
        FakeInferenceService.last_model_path_override = model_path_override

    def predict(self, image_path: str, top_k: int = 3) -> PredictionResult:
        return _prediction()


class FakeExplainer:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name
        FakeExplainer.last_model_name = model_name

    def explain(self, prediction: PredictionResult, question: str | None, top_k: int = 3) -> str:
        return "This looks like a cat, but the model may still be wrong."


def test_load_local_environment_reads_dotenv_when_variables_are_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("OPENAI_API_KEY=from-dotenv\nOPENAI_MODEL=dotenv-model\n", encoding="utf-8")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    _load_local_environment(tmp_path)

    assert os.getenv("OPENAI_API_KEY") == "from-dotenv"
    assert os.getenv("OPENAI_MODEL") == "dotenv-model"


def test_load_local_environment_does_not_overwrite_existing_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("OPENAI_API_KEY=from-dotenv\nOPENAI_MODEL=dotenv-model\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_API_KEY", "from-existing")
    monkeypatch.setenv("OPENAI_MODEL", "existing-model")

    _load_local_environment(tmp_path)

    assert os.getenv("OPENAI_API_KEY") == "from-existing"
    assert os.getenv("OPENAI_MODEL") == "existing-model"


def test_cli_classifier_only_mode(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("src.decision_intelligence_engine.explain_image.ModelInferenceService", FakeInferenceService)
    monkeypatch.setattr(sys, "argv", ["explain_image.py", "--image", "demo.png", "--no-llm"])
    exit_code = main()
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Predicted class: cat" in output
    assert "Top predictions:" in output
    assert "Explanation:" not in output


def test_cli_explanation_mode_with_mocks(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("src.decision_intelligence_engine.explain_image.ModelInferenceService", FakeInferenceService)
    monkeypatch.setattr("src.decision_intelligence_engine.explain_image.OpenAIExplainer", FakeExplainer)
    monkeypatch.setattr(sys, "argv", ["explain_image.py", "--image", "demo.png", "--question", "What is shown?"])
    exit_code = main()
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Explanation:" in output
    assert "This looks like a cat" in output


def test_cli_passes_model_path_and_openai_model(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("src.decision_intelligence_engine.explain_image.ModelInferenceService", FakeInferenceService)
    monkeypatch.setattr("src.decision_intelligence_engine.explain_image.OpenAIExplainer", FakeExplainer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "explain_image.py",
            "--image",
            "demo.png",
            "--model-path",
            "override.keras",
            "--openai-model",
            "gpt-test",
        ],
    )
    exit_code = main()
    _ = capsys.readouterr()
    assert exit_code == 0
    assert FakeInferenceService.last_model_path_override == "override.keras"
    assert FakeExplainer.last_model_name == "gpt-test"


def test_cli_rejects_invalid_top_k(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("src.decision_intelligence_engine.explain_image.ModelInferenceService", FakeInferenceService)
    monkeypatch.setattr(sys, "argv", ["explain_image.py", "--image", "demo.png", "--no-llm", "--top-k", "0"])
    exit_code = main()
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "top_k" in err


def test_cli_nonzero_failure(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    class BrokenInference:
        def __init__(self, project_root: Path, model_path_override: str | None = None) -> None:
            pass

        def predict(self, image_path: str, top_k: int = 3) -> PredictionResult:
            raise ValueError("bad image")

    monkeypatch.setattr("src.decision_intelligence_engine.explain_image.ModelInferenceService", BrokenInference)
    monkeypatch.setattr(sys, "argv", ["explain_image.py", "--image", "demo.png", "--no-llm"])
    exit_code = main()
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "Error: bad image" in err