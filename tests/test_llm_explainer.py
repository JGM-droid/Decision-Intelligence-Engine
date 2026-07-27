from __future__ import annotations

import pytest

from src.decision_intelligence_engine.llm_explainer import (
    DEFAULT_QUESTION,
    OpenAIExplainer,
    OpenAIExplanationError,
    build_explanation_prompt,
    extract_response_text,
    normalize_user_question,
)
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


def test_default_question_behavior() -> None:
    handled = normalize_user_question("   ")
    assert handled.question == DEFAULT_QUESTION
    assert handled.used_default is True


def test_vague_and_unsupported_questions_are_handled() -> None:
    vague = normalize_user_question("Explain this")
    unsupported = normalize_user_question("Who is in this image?")
    assert vague.used_default is True and vague.vague_input is True
    assert unsupported.used_default is True and unsupported.unsupported_request is True


def test_prompt_construction_contains_prediction_and_limitations() -> None:
    prompt = build_explanation_prompt(_prediction(), "What is shown here?", top_k=2)
    assert "Predicted class: cat" in prompt
    assert "Confidence: 54.40%" in prompt
    assert "supports only these CIFAR-10 classes" in prompt
    assert "must not override" in prompt


def test_extract_response_text_from_output_text() -> None:
    class Response:
        output_text = "Concise explanation."

    assert extract_response_text(Response()) == "Concise explanation."


def test_extract_response_text_rejects_empty_response() -> None:
    class Response:
        output_text = ""
        output = []

    with pytest.raises(OpenAIExplanationError):
        extract_response_text(Response())


def test_extract_response_text_from_nested_output() -> None:
    class Part:
        def __init__(self, text: str):
            self.text = text

    class Item:
        def __init__(self):
            self.content = [Part("Nested explanation text.")]

    class Response:
        output_text = None
        output = [Item()]

    assert extract_response_text(Response()) == "Nested explanation text."


def test_successful_mocked_openai_responses_call() -> None:
    captured: dict[str, str] = {}

    class FakeResponses:
        def create(self, model: str, input: str):
            captured["model"] = model
            captured["input"] = input

            class Response:
                output_text = "The model predicts cat with moderate confidence."

            return Response()

    class FakeClient:
        responses = FakeResponses()

    explainer = OpenAIExplainer(client=FakeClient(), model_name="demo-model")
    text = explainer.explain(_prediction(), "What is shown in this image?", top_k=2)
    assert text.startswith("The model predicts cat")
    assert captured["model"] == "demo-model"
    assert "Predicted class: cat" in captured["input"]


def test_missing_api_key_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    explainer = OpenAIExplainer()
    with pytest.raises(OpenAIExplanationError):
        explainer.explain(_prediction(), None)


def test_api_failure_is_wrapped() -> None:
    class FakeResponses:
        def create(self, model: str, input: str):
            raise TimeoutError("timed out")

    class FakeClient:
        responses = FakeResponses()

    explainer = OpenAIExplainer(client=FakeClient())
    with pytest.raises(OpenAIExplanationError):
        explainer.explain(_prediction(), "Explain this")