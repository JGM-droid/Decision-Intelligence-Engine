# MLflow Tracking

## Purpose

MLflow is used to capture reproducible baseline training evidence for the corrected frozen MobileNetV2 workflow. Each execution creates one local run under `mlruns/`.

## Configuration

The central settings live in `configs/mlflow.json`:

- `enabled`: toggles MLflow tracking for the workflow
- `tracking_dir`: local MLflow store directory
- `experiment_name`: experiment used for all baseline runs
- `run_name_prefix`: shared prefix for run names
- `tags`: project metadata attached to each run

The current run-name prefix remains `mobilenetv2_frozen_baseline` for historical continuity with the recorded Phase 4A and Phase 5A runs already present in MLflow. Changing it now would only affect future runs and would not rename the existing tracked evidence, so it is left unchanged for this commit.

## Logged Content

Each baseline run records:

- model hyperparameters and training controls
- data pipeline configuration and split strategy
- final train, validation, and test metrics
- per-epoch history metrics
- model artifact, metrics JSON, confusion matrix CSV, classification report, confusion matrix plot, and training-history plot

Phase 4B and Phase 5A runs additionally log:

- experiment ID/category/change tags
- run-level reproducibility tags
- environment version metadata
- experiment config artifacts from `configs/experiments/`
- architecture-required input/preprocessing evidence when architecture differs from MobileNetV2

## Current Status

Phase 4A, Phase 4B, and Phase 5A are complete:

- baseline instrumentation verified
- five controlled MobileNetV2 runs executed
- one MobileNetV2 96x96 resolution-control run executed through the shared pipeline
- one EfficientNetB0 96x96 architecture-comparison run executed through the same shared pipeline
- MLflow search-based comparison and deterministic selection implemented

Comparison tooling:

- `python -m src.decision_intelligence_engine.compare_experiments --write-reports`
- `python -m src.decision_intelligence_engine.select_experiment`
- `python -m src.decision_intelligence_engine.compare_experiments --architecture --write-architecture-reports`
- `python -m src.decision_intelligence_engine.select_architecture`