# System Architecture

## Project Direction

Decision Intelligence Engine is a CLI-based computer-vision assistant. It uses a CNN trained on CIFAR-10 for image classification and an LLM-powered explanation layer to translate predictions into clear user-facing guidance.

## Dataset Architecture Decision

- The canonical raw dataset layout is:
  - data/raw/cifar10/train/<class>/*.png
  - data/raw/cifar10/test/<class>/*.png
- The official CIFAR-10 test split remains untouched.
- Validation is derived deterministically from the training tree.
- Supporting multiple CIFAR formats is intentionally avoided to keep the pipeline simple and reproducible.

## High-Level System Overview

The system has two tightly coupled layers:

- Predictive layer: image preprocessing plus CNN inference for class probabilities.
- Interaction layer: CLI input handling plus LLM explanation generation using model outputs and confidence context.

Phase 5B keeps these layers separate in code:

- model_inference.py: artifact resolution, image validation, preprocessing, and EfficientNetB0 inference
- llm_explainer.py: prompt construction and OpenAI Responses API interaction
- explain_image.py: CLI orchestration only

Current selected predictive model:

- EfficientNetB0 with ImageNet weights
- frozen backbone
- effective model input resolution 96x96
- selected after comparison against MobileNetV2 32x32 and MobileNetV2 96x96

## Mermaid Diagrams

### Training and Selection

```mermaid
flowchart LR
    A[Canonical CIFAR-10 folders] --> B[Deterministic stratified split]
    B --> C[TensorFlow preprocessing]
    C --> D[Transfer-learning training]
    D --> E[MLflow metrics + artifacts]
    E --> F[Comparison reports]
    F --> G[Architecture selection]
    G --> H[Selected EfficientNetB0 artifact]
```

### Inference and Explanation

```mermaid
flowchart LR
    A[User image path] --> B[Image validation]
    B --> C[ModelInferenceService]
    C --> D[EfficientNetB0 prediction]
    D --> E[PredictionResult]
    E --> F[OpenAIExplainer]
    F --> G[Readable explanation]
    E --> H[CLI output]
    G --> H
```

### MLflow Artifact Resolution

```mermaid
flowchart LR
    A[architecture_comparison.json] --> B[Selected run id]
    B --> C[MLflow artifact listing]
    C --> D[Download .keras model]
    C --> E[Download metrics JSON]
    D --> F[Cached inference artifact]
    E --> F
```

## User Workflow

1. User runs the CLI with an image path and optional natural-language question.
2. System validates image format and applies inference preprocessing.
3. CNN model returns class probabilities and a top prediction.
4. Explanation context is assembled from the prediction, confidence, and top-k alternatives.
5. LLM generates a concise natural-language explanation.
6. CLI prints prediction, confidence, and explanation.
7. If input is invalid or out of scope, the CLI returns a clear error or a safe fallback.

## Data Flow

### Training Data Flow

1. Acquire CIFAR-10 data and document the source.
2. Split the training tree into deterministic train and validation manifests.
3. Apply training transforms and architecture-compatible preprocessing.
4. Train CNN variants and log all run metadata to MLflow.
5. Evaluate on held-out test data and persist artifacts.
6. Compare runs and freeze the best artifact for inference.

### Inference Data Flow

1. Accept a user-supplied image path.
2. Validate file type, dimensions, and decode success.
3. Apply deterministic inference transforms, including architecture-compatible resizing and preprocessing.
4. Run the selected EfficientNetB0 inference path.
5. Build explanation payload from predicted class, confidence, and top-k alternatives.
6. Generate LLM explanation and return a structured response.

Supported CIFAR-10 classes for inference and explanation:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## Model Training Workflow

1. Baseline CNN setup with fixed seed and reproducible split strategy.
2. Train at least 3 meaningfully different configurations.
3. Expand to at least 5 total MLflow runs with varied architecture and/or hyperparameters.
4. Track per-epoch and final metrics, including accuracy and loss plus classification reports where applicable.
5. Evaluate final candidates on held-out test data.
6. Programmatically select the best run using MLflow search/query workflow.

## Inference Workflow

1. Load the selected EfficientNetB0 model artifact from the finalized local model artifact path.
2. Preprocess the incoming image with training-compatible resizing and EfficientNetB0-compatible scaling.
3. Compute class probabilities and top-k predictions.
4. Apply confidence guardrails:
   - low confidence -> uncertainty-aware response
   - ambiguous top classes -> present alternatives
5. Send compact prediction context to the LLM.
6. Return a user-facing explanation with caveats.

Classifier-only mode:

- The CLI can skip OpenAI with --no-llm.
- This mode is used for local verification, offline testing, and environments without OPENAI_API_KEY.

## MLflow Integration Points

- Experiment setup: experiment name, run tags, dataset version metadata.
- Training runs: hyperparameters, architecture notes, transform strategy.
- Metric logging: training, validation, and test metrics plus confidence diagnostics.
- Artifact logging: trained model, label mapping, and key evaluation outputs.
- Selection: use MLflow search results to rank runs and determine the best model.

## LLM Integration Points

- Prompt input:
  - predicted class
  - confidence score
  - top-k alternatives
  - model limitations template
- Prompt goals:
  - explain the prediction in plain language
  - avoid fabricated certainty
  - include caveats for low-confidence outputs
- Output contract:
  - concise explanation
  - confidence-aware interpretation
  - optional follow-up suggestion, such as uploading a clearer image

## Error Handling Strategy

- Input validation errors:
  - unsupported file types
  - unreadable or corrupt images
  - missing upload path
- Inference errors:
  - model artifact not found
  - shape or transform mismatch
  - runtime inference exceptions
- LLM errors:
  - API timeout or rate limit
  - malformed response
  - provider unavailability
- Fallback behavior:
  - always return a deterministic prediction summary even if LLM fails
  - provide a plain, non-LLM template explanation on LLM failure
  - log errors for debugging and postmortem analysis

Phase 5B CLI behavior:

- classifier-only mode succeeds without an API key
- explanation mode returns clear user-facing errors for missing API key, API failure, timeout, or malformed response
- missing or vague questions fall back to a deterministic default prompt

## Future Scalability Considerations

- Model evolution:
  - swap CNN backbones without changing the interface contract
  - support transfer learning and model versioning
- Deployment scaling:
  - separate the inference service from the UI process
  - cache the model in memory for low-latency requests
- Observability:
  - add structured logging and inference telemetry
  - monitor confidence drift and class distribution drift
- Product growth:
  - batch inference mode
  - user feedback loop for mislabeled or uncertain cases
  - optional multimodal prompt enhancements for richer explanations

## Phase 3 Entry Criteria

- Dataset architecture standardized
- Data audit completed
- Preprocessing strategy approved
- Transfer-learning architecture selected
- TensorFlow/Keras established as the project framework
