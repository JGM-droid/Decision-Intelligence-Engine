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

## Current Status

Phase 4A is complete for the corrected frozen baseline. The workflow is now instrumented, but the project still needs at least five meaningfully different runs before model comparison and selection can begin.