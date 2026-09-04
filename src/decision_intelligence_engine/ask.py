"""Natural-language-first CLI for the CIFAR-10 assistant.

Routing order:

    natural-language question
    -> request_parser.parse_request()
    -> validate request (intent, required image)
    -> run ModelInferenceService only for valid classification requests
    -> optionally invoke OpenAIExplainer
    -> user-facing response

Unsupported, ambiguous, and missing-image requests terminate before any model
loading or OpenAI call. The original explain_image CLI is unchanged and remains
the image-first entry point.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .explain_image import _load_local_environment
from .llm_explainer import OpenAIExplainer, OpenAIExplanationError
from .model_inference import (
    InvalidImageError,
    ModelInferenceService,
    ModelResolutionError,
    PredictionResult,
)
from .request_parser import Intent, ParsedRequest, parse_request


def _format_predictions(prediction: PredictionResult, top_k: int) -> str:
    lines = [f"Predicted class: {prediction.predicted_class}", f"Confidence: {prediction.confidence:.2%}"]
    if top_k > 0:
        lines.append("")
        lines.append("Top predictions:")
        for item in prediction.top_predictions[:top_k]:
            lines.append(f"- {item.class_name}: {item.confidence:.2%}")
    return "\n".join(lines)


def build_answer(prediction: PredictionResult, request: ParsedRequest) -> str:
    """Deterministic classifier-semantics answer for a parsed request.

    Target-class questions are answered from the single-label classifier
    prediction only; the classifier is not an object detector, so absence
    claims are never made.
    """

    if request.target_class is None:
        return (
            f"The classifier predicts '{prediction.predicted_class}' "
            f"with {prediction.confidence:.2%} confidence."
        )
    if prediction.predicted_class == request.target_class:
        return (
            f"The classifier predicts '{prediction.predicted_class}' "
            f"with {prediction.confidence:.2%} confidence."
        )
    return (
        f"The classifier predicts '{prediction.predicted_class}' "
        f"with {prediction.confidence:.2%} confidence rather than '{request.target_class}'."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Answer natural-language questions about CIFAR-10 images."
    )
    parser.add_argument("--question", required=True, help="Natural-language request")
    parser.add_argument("--image", help="Path to the input image (required for classification requests)")
    parser.add_argument("--model-path", help="Optional explicit .keras model path override")
    parser.add_argument("--openai-model", help="Optional OpenAI model override")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to display")
    parser.add_argument("--no-llm", action="store_true", help="Skip OpenAI and show classifier output only")
    args = parser.parse_args()

    request = parse_request(args.question)

    if request.intent is Intent.UNSUPPORTED:
        print(f"Unsupported request: {request.validation_error}", file=sys.stderr)
        return 1

    if request.intent is Intent.AMBIGUOUS:
        print(f"Ambiguous request: {request.validation_error}", file=sys.stderr)
        return 1

    if request.intent is Intent.CLASSIFY_IMAGE and request.requires_image and not args.image:
        print(
            "Error: an image is required for this request. "
            "Provide one with --image path/to/image.png.",
            file=sys.stderr,
        )
        return 1

    project_root = Path(__file__).resolve().parents[2]
    _load_local_environment(project_root)
    try:
        if args.top_k <= 0:
            raise ValueError("top_k must be between 1 and 10")
        inference = ModelInferenceService(project_root=project_root, model_path_override=args.model_path)
        prediction = inference.predict(args.image, top_k=args.top_k)

        print("Answer:")
        print(build_answer(prediction, request))
        print("")
        print(_format_predictions(prediction, top_k=args.top_k))

        if args.no_llm:
            return 0

        explainer = OpenAIExplainer(model_name=args.openai_model)
        explanation = explainer.explain(prediction=prediction, question=args.question, top_k=args.top_k)
        print("")
        print("Explanation:")
        print(explanation)
        return 0
    except (ModelResolutionError, InvalidImageError, OpenAIExplanationError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
