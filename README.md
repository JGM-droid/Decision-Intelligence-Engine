# Decision Intelligence Engine

Decision Intelligence Engine is a CIFAR-10 computer-vision project that pairs a reproducible CNN classifier with a lightweight OpenAI explanation layer. The final selected predictive model is EfficientNetB0, chosen after controlled comparison against matched MobileNetV2 baselines.

## Project Overview

This repository is designed to read like a release candidate: experiments are reproducible, inference uses the actual tracked model artifact, and the user-facing CLI stays separate from model training.

What the project does:

- accepts natural-language questions and parses them deterministically into structured requests
- classifies CIFAR-10 images with a frozen EfficientNetB0 model
- explains classifier output with the OpenAI Responses API
- records experiments and artifacts in MLflow
- preserves a deterministic train/validation/test split
- supports classifier-only verification with no API key

## 🎥 Project Demo

[Watch the Decision Intelligence Engine Capstone Demo](https://youtu.be/9S8HbCtBMCs)

This video demonstrates the trained ML model, natural-language/LLM interaction, MLflow experiment tracking, edge-case handling, and automated tests.

## Status

Phase 5B is functionally complete. The repository is now focused on release readiness, portfolio polish, and documentation clarity rather than new ML features.

## Architecture

The current architecture is documented in [docs/architecture.md](docs/architecture.md). In short:

- data ingestion and preprocessing are deterministic
- model training and experiment tracking happen through the shared TensorFlow/MLflow pipeline
- natural-language requests are parsed deterministically before any model or LLM call
- inference resolves the selected EfficientNetB0 artifact from MLflow evidence
- OpenAI is only used to explain classifier output, not to make the prediction and not to parse requests

## Natural-Language Interface

The natural-language-first CLI is [src/decision_intelligence_engine/ask.py](src/decision_intelligence_engine/ask.py). The flow is:

```text
Natural-language request
        ↓
Deterministic request parser (request_parser.py, no LLM)
        ↓
Structured ParsedRequest (intent, target_class, requires_image)
        ↓
Validation / routing
        ↓
Image classifier, only when a valid classification request supplies an image
        ↓
PredictionResult (predicted class + confidence)
        ↓
Optional LLM explanation (skipped with --no-llm)
        ↓
User-facing response
```

The parser is deterministic and conservative. It never invents an intent: clearly out-of-scope requests are rejected, vague requests are reported as ambiguous, and only requests that match this application's CIFAR-10 classification capability reach the classifier. Unsupported, ambiguous, and missing-image requests terminate before any model loading and before any OpenAI call.

Demo commands:

```bash
# A. Normal natural-language classification request
python -m src.decision_intelligence_engine.ask --question "What is shown in this image?" --image docs/assets/cifar10_samples/frog_sample_1.png

# B. CIFAR-10 target-class question (answered from classifier semantics, not object detection)
python -m src.decision_intelligence_engine.ask --question "Does this image contain a frog?" --image docs/assets/cifar10_samples/frog_sample_1.png

# C. Missing-image request (clear error, no model or OpenAI call)
python -m src.decision_intelligence_engine.ask --question "What is shown in this image?"

# D. Out-of-scope request (rejected before any model or OpenAI call)
python -m src.decision_intelligence_engine.ask --question "What will the weather be tomorrow?"

# E. Classifier-only mode without OpenAI
python -m src.decision_intelligence_engine.ask --question "What is shown in this image?" --image docs/assets/cifar10_samples/frog_sample_1.png --no-llm

# F. Automated test suite
python -m pytest tests -v

# G. Regenerate MLflow experiment evidence
python -m src.decision_intelligence_engine.compare_experiments --write-reports
```

The original image-first CLI ([src/decision_intelligence_engine/explain_image.py](src/decision_intelligence_engine/explain_image.py)) is unchanged and remains fully supported.

## Key Files

- [src/decision_intelligence_engine/ask.py](src/decision_intelligence_engine/ask.py) - natural-language-first CLI entry point
- [src/decision_intelligence_engine/request_parser.py](src/decision_intelligence_engine/request_parser.py) - deterministic request parser
- [src/decision_intelligence_engine/explain_image.py](src/decision_intelligence_engine/explain_image.py) - original image-first CLI entry point
- [src/decision_intelligence_engine/model_inference.py](src/decision_intelligence_engine/model_inference.py) - model resolution and prediction
- [src/decision_intelligence_engine/llm_explainer.py](src/decision_intelligence_engine/llm_explainer.py) - OpenAI prompt and response handling
- [src/decision_intelligence_engine/baseline_training.py](src/decision_intelligence_engine/baseline_training.py) - shared training pipeline
- [docs/architecture_comparison.md](docs/architecture_comparison.md) - final EfficientNetB0 selection evidence
- [docs/requirement_traceability.md](docs/requirement_traceability.md) - evidence ledger

## Dataset

The canonical dataset layout is:

```text
data/raw/cifar10/train/<class>/*.png
data/raw/cifar10/test/<class>/*.png
```

Training and validation splits are derived deterministically from the training tree. The official CIFAR-10 test tree remains untouched.

## Screenshots and Demo Assets

This repository includes representative CIFAR-10 sample images under [docs/assets/cifar10_samples](docs/assets/cifar10_samples). Suggested portfolio screenshots to add later, if desired:

- the MLflow experiment comparison page
- a terminal capture of `python -m src.decision_intelligence_engine.explain_image --image ... --no-llm`
- a terminal capture of the OpenAI explanation mode

## Installation

1. Create and activate a Python 3.11 environment.
2. Install pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Ensure the CIFAR-10 folder layout exists at `data/raw/cifar10`.
4. Confirm MLflow can write to the local `mlruns/` directory.

## Environment Setup

Copy `.env.example` to `.env` if you want a local template, then set:

- `OPENAI_API_KEY` for explanation mode
- `OPENAI_MODEL` if you want to override the default OpenAI model

Do not commit secrets. `.env` is ignored by Git.

## Usage

Common commands:

```bash
python -m src.decision_intelligence_engine.run_experiments --all
python -m src.decision_intelligence_engine.compare_experiments --write-reports
python -m src.decision_intelligence_engine.select_experiment
python -m src.decision_intelligence_engine.compare_experiments --architecture --write-architecture-reports
python -m src.decision_intelligence_engine.select_architecture
python -m src.decision_intelligence_engine.ask --question "What is shown in this image?" --image path/to/image.png
python -m src.decision_intelligence_engine.ask --question "Does this image contain a frog?" --image path/to/image.png --no-llm
python -m src.decision_intelligence_engine.explain_image --image path/to/image.png --no-llm
python -m src.decision_intelligence_engine.explain_image --image path/to/image.png --question "What is shown here?"
```

The classifier remains the source of truth for the prediction. The LLM only explains the result and its caveats.

## Example Output

Classifier-only smoke test:

```text
Predicted class: frog
Confidence: 40.63%
Top predictions: frog, cat, dog
```

Explanation mode adds a short natural-language summary from OpenAI after the classifier result.

## Testing

Run the test suite locally with:

```bash
python -m pytest tests -v
python -m compileall src tests
```

The repository also includes focused tests for data loading, inference, CLI behavior, and OpenAI response handling.

## Project Structure

```text
configs/
data/
docs/
models/
reports/
src/decision_intelligence_engine/
tests/
```

## Future Roadmap

Potential follow-up items that would materially improve the portfolio later:

- a Streamlit front end for a richer demo experience
- Docker packaging for one-command setup
- a lightweight model registry or release tag workflow
- additional observability around inference confidence and drift

## Model Results

The final architecture choice is EfficientNetB0 at 96x96 resolution. The supporting comparison and MLflow evidence are summarized in [docs/architecture_comparison.md](docs/architecture_comparison.md).

## MLflow Experiment Evidence

The tracked experiment `decision_intelligence_engine` (id `320350008725726199`) under `mlruns/` contains 9 FINISHED runs covering 7 meaningfully different configurations, exceeding the requirement of at least 5 meaningful experiment runs:

- 5 controlled Phase 4B MobileNetV2 variants, listed with run IDs and test macro precision/recall/F1 in [reports/model_comparison.md](reports/model_comparison.md)
- 1 MobileNetV2 96x96 resolution-control run and 1 EfficientNetB0 96x96 run, listed with run IDs and macro metrics in [reports/architecture_comparison.md](reports/architecture_comparison.md) (the selected EfficientNetB0 run is `c6170c6a38b74c869ffef74892644f42`)
- 2 early Phase 4A baseline runs retained for audit history

3 interrupted KILLED runs also remain in the local store; they are not counted as experiment evidence.

All evidence is regenerated from the authoritative MLflow store with:

```bash
python -m src.decision_intelligence_engine.compare_experiments --write-reports
python -m src.decision_intelligence_engine.compare_experiments --architecture --write-architecture-reports
```

## Limitations

The explanation flow only describes CIFAR-10 classifier output. It does not verify correctness, and it should not be treated as an independent visual judgment. If OpenAI is unavailable, classifier-only mode still works.

## Reflection

The main engineering lesson from the project is that reproducibility and evidence quality matter more than chasing a quick baseline score. Input resolution turned out to be a real confounding variable, which is why the MobileNetV2 96x96 control exists and why the final architecture decision is tied to matched-resolution evidence rather than the first pair of runs.

Another key challenge was keeping deterministic classifier behavior separate from the nondeterministic explanation layer: the model inference path is testable on its own, MLflow artifact selection is explicit, and the OpenAI call is treated as a post-prediction explanation step rather than part of the prediction itself.
