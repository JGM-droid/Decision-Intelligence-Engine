# Project Roadmap

## Phase 1 - Architecture

- Confirm project direction: CIFAR-10 image classification + LLM explanation layer.
- Finalize system architecture and component boundaries.
- Define training/inference contracts and error-handling strategy.
- Confirm success criteria and evidence mapping to requirement traceability.

## Phase 2 - Dataset Acquisition, EDA, and Image Preprocessing

- Acquire CIFAR-10 and document source, version, and structure.
- Run image EDA: class balance, sample visualization, data quality checks.
- Define preprocessing and augmentation strategy for train vs inference paths.
- Document preprocessing decisions and leakage-prevention controls.

## Phase 3 - CNN Training

- Implement baseline CNN training pipeline.
- Train at least 3 meaningfully different CNN configurations.
- Capture reproducible training setup (seeds, splits, transforms).
- Evaluate baseline and candidate models on validation/test data.

## Phase 4 - MLflow Experiments

- Integrate MLflow into all training runs.
- Log parameters, metrics, artifacts, and dataset metadata for every run.
- Phase 4A complete: corrected frozen MobileNetV2 baseline now writes one local MLflow run per execution.
- Baseline runs log hyperparameters, data description, evaluation metrics, model artifacts, and training plots.
- Phase 4B complete: controlled five-run MobileNetV2 matrix executed with deterministic config validation.
- Programmatic run comparison and deterministic selection implemented via MLflow search-based tooling.

## Phase 5 - Model Selection

- Programmatically identify best run using MLflow search/query.
- Validate chosen model on held-out test set and summarize tradeoffs.
- Freeze best model artifact and label mapping for inference.
- Document model selection rationale and performance summary.
- Phase 5A complete: selected MobileNetV2 32x32, MobileNetV2 96x96 resolution control, and EfficientNetB0 96x96 were compared through one shared MLflow-backed workflow.
- Architecture comparison now has dedicated reports, a matched-resolution control, and deterministic final-selection logic.

## Phase 6 - LLM Integration

- Define prediction-to-prompt schema for explanation generation.
- Implement confidence-aware explanation rules and caveat policy.
- Add handling for uncertain/ambiguous/out-of-scope responses.
- Validate secure API-key usage via environment variables only.
- Phase 5B complete: classifier inference and OpenAI explanation are implemented as separate modules with CLI access.

## Phase 7 - Command-Line Interface

- Build image path and prediction display workflow.
- Surface class prediction, confidence, and explanation text.
- Add clear UX for invalid input and recovery guidance.
- Ensure reproducible local run instructions for reviewers.

## Phase 8 - Testing

- Add preprocessing tests (missing/corrupt inputs, transform consistency, immutability where applicable).
- Add model tests (prediction shape/type and minimum threshold checks).
- Add interface tests (input handling, parsing/response behavior, edge cases).
- Achieve passing test run with pytest tests/ -v.

## Phase 9 - Docker

- Create optional container build for reproducible execution.
- Verify docker build and docker run flow for reviewers.
- Document container assumptions and runtime configuration.

## Phase 10 - Final Documentation and Demo

- Complete README sections: setup, usage, architecture, results, reflection.
- Finalize requirement traceability with evidence links.
- Prepare short demo covering normal and edge-case user flows.
- Perform final repository quality pass and submission readiness check.
