"""CLI entry point for classifier inference with optional OpenAI explanation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .llm_explainer import OpenAIExplainer, OpenAIExplanationError
from .model_inference import InvalidImageError, ModelInferenceService, ModelResolutionError


def _format_predictions(service_output, top_k: int) -> str:
    lines = [f"Predicted class: {service_output.predicted_class}", f"Confidence: {service_output.confidence:.2%}"]
    if top_k > 0:
        lines.append("")
        lines.append("Top predictions:")
        for item in service_output.top_predictions[:top_k]:
            lines.append(f"- {item.class_name}: {item.confidence:.2%}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CIFAR-10 classification and optional OpenAI explanation")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--question", help="Natural-language question about the model prediction")
    parser.add_argument("--model-path", help="Optional explicit .keras model path override")
    parser.add_argument("--openai-model", help="Optional OpenAI model override")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to display")
    parser.add_argument("--no-llm", action="store_true", help="Skip OpenAI and show classifier output only")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    try:
        if args.top_k <= 0:
            raise ValueError("top_k must be between 1 and 10")
        inference = ModelInferenceService(project_root=project_root, model_path_override=args.model_path)
        prediction = inference.predict(args.image, top_k=args.top_k)
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