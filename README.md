# Decision Intelligence Engine

Decision Intelligence Engine is a computer-vision decision-support project built around CIFAR-10 image classification. The current workstream uses a deterministic data pipeline, a corrected MobileNetV2 baseline, a controlled five-run MobileNetV2 experiment matrix, a MobileNetV2 96x96 resolution-control run, and a matched-resolution EfficientNetB0 architecture comparison, all tracked in local MLflow for reproducible model selection.

## Project Description

The project classifies CIFAR-10 images and will later surface those predictions through a focused user-facing interface with explanation support. Phase 4A added MLflow around the corrected baseline; Phase 4B added a controlled MobileNetV2 matrix; Phase 5A completed a three-way architecture comparison across MobileNetV2 32x32, MobileNetV2 96x96, and EfficientNetB0 96x96.

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
python -m src.decision_intelligence_engine.run_experiments --experiment mobilenetv2_96x96_control
python -m src.decision_intelligence_engine.run_experiments --experiment efficientnetb0_control_frozen
python -m src.decision_intelligence_engine.compare_experiments --architecture --write-architecture-reports
python -m src.decision_intelligence_engine.select_architecture
```

These commands execute approved experiments, generate MLflow-backed comparison reports, and produce deterministic selection summaries.

## Architecture

The system is organized around three layers: data pipeline, model training/evaluation, and experiment tracking. The current architecture keeps preprocessing in shared code, supports both MobileNetV2 and EfficientNetB0 through one training path, and records each run through MLflow for later comparison.

## Model Results

The current selected MobileNetV2 configuration is the frozen longer-training variant based on explicit validation-first criteria for Phase 4B. After the additional Phase 5A resolution-control run, EfficientNetB0 is selected as the final architecture because it still outperformed MobileNetV2 when both used 96x96 effective inputs.

## Limitations

The project does not yet include the LLM interface or Streamlit app. The final Phase 5A comparison is limited to one matched-resolution EfficientNetB0 baseline and one MobileNetV2 resolution-control run, so it should not be interpreted as an exhaustive architecture search.

## Reflection

The main lesson so far is that reproducible file-manifest splitting and serialization-safe preprocessing matter more than a quick baseline score. Local experiment tracking is now in place so later model iterations can be compared cleanly.

## Demo

The current demonstration is the controlled experiment matrix plus the final architecture comparison outputs. Start the UI with `mlflow ui --backend-store-uri file:./mlruns` after the experiment runs.
