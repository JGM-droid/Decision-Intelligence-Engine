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

## Logged Content

Each baseline run records:

- model hyperparameters and training controls
- data pipeline configuration and split strategy
- final train, validation, and test metrics
- per-epoch history metrics
- model artifact, metrics JSON, confusion matrix CSV, classification report, confusion matrix plot, and training-history plot

Phase 4B matrix runs additionally log:

- experiment ID/category/change tags
- run-level reproducibility tags
- environment version metadata
- experiment config artifacts from `configs/experiments/`

## Current Status

Phase 4A and Phase 4B are complete for MobileNetV2:

- baseline instrumentation verified
- five controlled MobileNetV2 runs executed
- MLflow search-based comparison and deterministic selection implemented

Comparison tooling:

- `python -m src.decision_intelligence_engine.compare_experiments --write-reports`
- `python -m src.decision_intelligence_engine.select_experiment`