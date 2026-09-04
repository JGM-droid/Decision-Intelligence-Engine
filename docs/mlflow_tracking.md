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
- the authoritative baseline training config from `configs/baseline_training.yaml` when present, with JSON retained only for legacy compatibility
- architecture-required input/preprocessing evidence when architecture differs from MobileNetV2

## Current Status

Phase 4A, Phase 4B, and Phase 5A are complete:

- baseline instrumentation verified
- five controlled MobileNetV2 runs executed
- one MobileNetV2 96x96 resolution-control run executed through the shared pipeline
- one EfficientNetB0 96x96 architecture-comparison run executed through the same shared pipeline
- MLflow search-based comparison and deterministic selection implemented

## Run Evidence

The tracked experiment `decision_intelligence_engine` (id `320350008725726199`) contains 9 legitimate FINISHED runs covering 7 distinct configurations, which exceeds the requirement of at least 5 meaningful experiment runs. Three interrupted KILLED runs remain in the local store and are excluded below.

### Phase 4B: Controlled MobileNetV2 experiment matrix (32x32 input)

| Run ID | Experiment | Changed Parameter | Epochs | LR | Dropout | Val Acc | Test Acc | Macro P | Macro R | Macro F1 | Status | Selected |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `015da38ba79d4c5aaedb6deba02b9b6d` | mnetv2_control_frozen | none (control) | 1 | 3e-4 | 0.20 | 0.1342 | 0.1398 | 0.1268 | 0.1398 | 0.0688 | FINISHED | no |
| `7e1e8eb9ae7a431191ee43e9aa115e48` | mnetv2_longer_frozen_epochs | epochs | 3 | 3e-4 | 0.20 | 0.1485 | 0.1563 | 0.1452 | 0.1563 | 0.1125 | FINISHED | yes (Phase 4B best) |
| `9e4284670cc0495ea596fa98618fd119` | mnetv2_lower_lr_frozen | learning_rate | 1 | 1e-4 | 0.20 | 0.1176 | 0.1241 | 0.0951 | 0.1241 | 0.0837 | FINISHED | no |
| `b8b1fc7c29ca41b8b35efd797939fb11` | mnetv2_higher_dropout_frozen | dropout_rate | 1 | 3e-4 | 0.35 | 0.1351 | 0.1400 | 0.1266 | 0.1400 | 0.0695 | FINISHED | no |
| `fd89ecec91214456a7d41517a93488c9` | mnetv2_partial_finetune_tail | trainability + lr | 1 | 3e-5 | 0.20 | 0.1484 | 0.1495 | 0.1077 | 0.1495 | 0.0741 | FINISHED | no |

### Phase 5A: Matched-resolution architecture comparison

| Run ID | Experiment | Architecture | Input Res | Epochs | Val Acc | Test Acc | Macro P | Macro R | Macro F1 | Status | Selected |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `8bd86fb64084413dacbebb9fec3c6561` | mobilenetv2_96x96_control | MobileNetV2 | [96, 96] | 3 | 0.3912 | 0.3909 | 0.5638 | 0.3909 | 0.2910 | FINISHED | no |
| `c6170c6a38b74c869ffef74892644f42` | efficientnetb0_control_frozen | EfficientNetB0 | [96, 96] | 3 | 0.5405 | 0.5440 | 0.7153 | 0.5440 | 0.4918 | FINISHED | yes (final model) |

### Phase 4A: Early baseline runs (audit history)

| Run ID | Run | Val Acc | Test Acc | Macro P/R/F1 | Status |
| --- | --- | ---: | ---: | --- | --- |
| `36b1c3955cac41469c6b272ecabbfb5a` | mobilenetv2_frozen_baseline_20260727_154101 | 0.1343 | 0.1398 | N/A | FINISHED |
| `3cba977d14cb41149943839974b69f96` | mobilenetv2_frozen_baseline_20260727_154201 | 0.1347 | 0.1407 | N/A | FINISHED |

Run IDs, statuses, and accuracy metrics above are read directly from the local MLflow store. Macro precision/recall/F1 are post-hoc calculations from each run's saved test confusion matrix; N/A means that run predates confusion-matrix artifact logging. Full per-run artifacts (metrics JSON, confusion matrices, training-history plots) are regenerated locally from the authoritative MLflow store with:

```bash
python -m src.decision_intelligence_engine.compare_experiments --write-reports
python -m src.decision_intelligence_engine.compare_experiments --architecture --write-architecture-reports
```

Narrative context for these runs is in [docs/mobilenetv2_experiments.md](mobilenetv2_experiments.md) and [docs/architecture_comparison.md](architecture_comparison.md).

Comparison tooling:

- `python -m src.decision_intelligence_engine.compare_experiments --write-reports`
- `python -m src.decision_intelligence_engine.select_experiment`
- `python -m src.decision_intelligence_engine.compare_experiments --architecture --write-architecture-reports`
- `python -m src.decision_intelligence_engine.select_architecture`