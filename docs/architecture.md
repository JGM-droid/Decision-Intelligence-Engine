# System Architecture

## Project Direction

Decision Intelligence Engine is an AI-powered computer vision assistant. It uses a CNN trained on CIFAR-10 for image classification and an LLM-powered explanation layer to translate predictions into clear user-facing guidance.

## Dataset Architecture Decision

- During project setup, the official CIFAR source was unavailable from the current development network.
- A verified mirror containing the identical CIFAR-10 dataset in folder-based image layout was successfully obtained.
- The folder layout is adopted as the canonical raw dataset representation:
  - data/raw/cifar10/train/<class>/*.png
  - data/raw/cifar10/test/<class>/*.png
- All future preprocessing, training, validation, inference, testing, and deployment will use this layout.
- Supporting multiple CIFAR formats is intentionally avoided to prevent unnecessary architecture and maintenance complexity.

## High-Level System Overview

The system has two tightly coupled layers:

- Predictive layer: image preprocessing + CNN inference for class probabilities.
- Interaction layer: user input handling + LLM explanation generation using model outputs and confidence context.

The architecture is intentionally modular so model training, experiment tracking, inference, and user interface can evolve independently.

## User Workflow

1. User opens the interface and uploads an image.
2. System validates image format/size and applies inference preprocessing.
3. CNN model returns class probabilities and top prediction.
4. Explanation context is assembled (top-k classes, confidence, caveats).
5. LLM generates a concise natural-language explanation.
6. UI shows prediction, confidence, and explanation.
7. If input is invalid or out of scope, system returns guided corrective prompts.

## Component Diagram

```text
+-------------------+        +----------------------------+
|      User         |        |       MLflow Tracking      |
| (uploads image)   |        | params/metrics/models/runs |
+---------+---------+        +-------------+--------------+
          |                                ^
          v                                |
+---------+---------+                      |
|  Streamlit UI     |                      |
| upload + display  |                      |
+---------+---------+                      |
          |                                |
          v                                |
+---------+---------+      log events      |
| Inference Service +----------------------+
| preprocess + predict |
+----+------------+----+
     |            |
     |            v
     |     +------+----------------+
     |     |   LLM Explanation     |
     |     | prompt + response gen |
     |     +------+----------------+
     |            |
     v            v
+----+------------+----+
| Response Assembler    |
| class/confidence/text |
+----+------------+----+
     |
     v
+----+------------+----+
| Result to User        |
+-----------------------+

Offline Training Path:
Raw CIFAR-10 -> Preprocessing Pipeline -> CNN Training -> Evaluation -> MLflow -> Model Registry/Best Model Artifact
```

## Data Flow

## Phase 2A Data Audit Outcomes

- Confirmed on-disk dataset layout for this environment: data/raw/cifar10/train/<class> and data/raw/cifar10/test/<class> (PNG files).
- Confirmed split sizes: 50,000 train and 10,000 test, balanced across 10 classes.
- Confirmed image shape: 32x32 RGB.
- Architecture impact:
  - No resizing required for the baseline CNN path.
  - Keep official test split immutable; derive validation only from training split.
  - Preserve class-name mapping in artifacts for consistent LLM explanations.

### Training Data Flow

1. Acquire CIFAR-10 data and document data version.
2. Split into train/validation/test sets with reproducible seeds.
3. Apply training transforms (normalization, optional augmentation).
4. Train CNN variants and log all run metadata to MLflow.
5. Evaluate on held-out test set and persist artifacts.
6. Select best run and freeze inference-ready model artifact.

### Inference Data Flow

1. Accept user-uploaded image.
2. Validate file type, dimensions, and decode success.
3. Apply deterministic inference transforms.
4. Run best-model inference.
5. Build explanation payload (class name, confidence, alternatives).
6. Generate LLM explanation and return structured response.

## Model Training Workflow

1. Baseline CNN setup with fixed seed and reproducible split strategy.
2. Train at least 3 meaningfully different configurations.
3. Expand to at least 5 total MLflow runs with varied architecture and/or hyperparameters.
4. Track per-epoch and final metrics (for example accuracy, precision/recall/F1, calibration-related confidence summaries).
5. Evaluate final candidates on held-out test set.
6. Programmatically select best run using MLflow search/query workflow.

## Inference Workflow

1. Load selected best model artifact from local model artifact path.
2. Preprocess incoming image with training-compatible normalization.
3. Compute class probabilities and top-k predictions.
4. Apply confidence guardrails:
   - low confidence -> uncertainty-aware response
   - ambiguous top classes -> present alternatives
5. Send compact prediction context to LLM.
6. Return user-facing explanation with caveats.

## MLflow Integration Points

- Experiment setup: experiment name, run tags, dataset version metadata.
- Training runs: hyperparameters, architecture notes, transform strategy.
- Metric logging: training/validation/test metrics and confidence diagnostics.
- Artifact logging: trained model, label mapping, and key evaluation outputs.
- Selection: use MLflow run search to rank runs and determine best model.

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
  - optional follow-up suggestion (for example, upload a clearer image)

## Error Handling Strategy

- Input validation errors:
  - unsupported file types
  - unreadable/corrupt images
  - missing upload
- Inference errors:
  - model artifact not found
  - shape/transform mismatch
  - runtime inference exceptions
- LLM errors:
  - API timeout/rate limit
  - malformed response
  - provider unavailability
- Fallback behavior:
  - always return deterministic prediction summary even if LLM fails
  - provide plain, non-LLM template explanation on LLM failure
  - log errors for debugging and postmortem analysis

## Future Scalability Considerations

- Model evolution:
  - swap CNN backbones without changing interface contract
  - support transfer learning and model versioning
- Deployment scaling:
  - separate inference service from UI process
  - cache model in memory for low-latency requests
- Observability:
  - add structured logging and inference telemetry
  - monitor confidence drift and class distribution drift
- Product growth:
  - batch inference mode
  - user feedback loop for mislabeled/uncertain cases
  - optional multimodal prompt enhancements for richer explanations

## Phase 3 Entry Criteria

- ✓ Dataset architecture standardized
- ✓ Data audit completed
- ✓ Preprocessing strategy approved
- ✓ Transfer learning architecture selected
- ✓ TensorFlow/Keras established as the project framework
