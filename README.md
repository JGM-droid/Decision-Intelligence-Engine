# Decision Intelligence Engine

Decision Intelligence Engine is a computer-vision decision-support project built around CIFAR-10 image classification. The current workstream uses a deterministic data pipeline, a corrected MobileNetV2 baseline, a controlled five-run MobileNetV2 experiment matrix, and local MLflow tracking for reproducible model selection.

## Project Description

The project classifies CIFAR-10 images and will later surface those predictions through a focused user-facing interface with explanation support. Phase 4A added MLflow around the corrected baseline; Phase 4B adds a controlled MobileNetV2 matrix with programmatic MLflow comparison and deterministic selection criteria.

## Intended Users and Problem

The project is intended for reviewers and future end users who need a simple image-classification assistant with traceable training evidence. The immediate problem is not a consumer chatbot; it is reproducible model training and experiment tracking for a narrow vision task.

## Dataset

The canonical dataset layout is `data/raw/cifar10/train/<class>/*.png` and `data/raw/cifar10/test/<class>/*.png`. Training, validation, and test splits are derived deterministically from the training tree, while the official test tree remains untouched.

## Setup

Install the pinned dependencies from `requirements.txt` into the project virtual environment and ensure TensorFlow can see the local CIFAR-10 files. MLflow writes to the local `mlruns/` directory, which is ignored by Git.

## API-Key Configuration

Not applicable for the current phase. No LLM provider is wired in yet, and no API keys are required for the baseline or MLflow workflow.

## Usage

Run the controlled experiment matrix with the project interpreter, for example:

```bash
python -m src.decision_intelligence_engine.run_experiments --all
python -m src.decision_intelligence_engine.compare_experiments --write-reports
python -m src.decision_intelligence_engine.select_experiment
```

These commands execute the approved experiments, generate MLflow-backed comparison reports, and produce a deterministic winner summary.

## Architecture

The system is organized around three layers: data pipeline, model training/evaluation, and experiment tracking. The current architecture keeps preprocessing in shared code, uses a frozen MobileNetV2 control baseline, and records each run through MLflow for later comparison.

## Model Results

The current selected MobileNetV2 configuration is the frozen longer-training variant based on explicit validation-first criteria. Fine-tuning was tested and tracked, but did not materially outperform the strongest frozen configuration.

## Limitations

The project does not yet include the LLM interface or Streamlit app, and architecture comparison is not started yet. Results are specific to MobileNetV2 under controlled local-runtime constraints.

## Reflection

The main lesson so far is that reproducible file-manifest splitting and serialization-safe preprocessing matter more than a quick baseline score. Local experiment tracking is now in place so later model iterations can be compared cleanly.

## Demo

The current demonstration is the controlled experiment matrix plus MLflow comparison outputs. Start the UI with `mlflow ui --backend-store-uri file:./mlruns` after the matrix run.
