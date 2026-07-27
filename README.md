# Decision Intelligence Engine

Decision Intelligence Engine is a computer-vision decision-support project built around CIFAR-10 image classification. The current workstream uses a deterministic data pipeline, a corrected MobileNetV2 baseline, a controlled five-run MobileNetV2 experiment matrix, a MobileNetV2 96x96 resolution-control run, a matched-resolution EfficientNetB0 architecture comparison, and a Phase 5B OpenAI explanation CLI built on top of the final selected EfficientNetB0 model.

## Project Description

The project classifies CIFAR-10 images and surfaces those predictions through a focused explanation layer. Phase 4A added MLflow around the corrected baseline; Phase 4B added a controlled MobileNetV2 matrix; Phase 5A completed a three-way architecture comparison across MobileNetV2 32x32, MobileNetV2 96x96, and EfficientNetB0 96x96; Phase 5B adds a CLI that combines classifier inference with an OpenAI-generated explanation.

## Intended Users and Problem

The project is intended for reviewers and future end users who need a simple image-classification assistant with traceable training evidence. The immediate problem is not a consumer chatbot; it is reproducible model training and experiment tracking for a narrow vision task.

## Dataset

The canonical dataset layout is `data/raw/cifar10/train/<class>/*.png` and `data/raw/cifar10/test/<class>/*.png`. Training, validation, and test splits are derived deterministically from the training tree, while the official test tree remains untouched.

## Setup

Install the pinned dependencies from `requirements.txt` into the project virtual environment and ensure TensorFlow can see the local CIFAR-10 files. MLflow writes to the local `mlruns/` directory, which is ignored by Git.

For Phase 5B explanation mode, set `OPENAI_API_KEY` in the environment. You may also set `OPENAI_MODEL` to override the default OpenAI model name used by the CLI.

## API-Key Configuration

Set `OPENAI_API_KEY` in the environment before running explanation mode. Do not store secrets in source files, JSON configs, tests, or committed `.env` files. `.env` is ignored by Git, and `.env.example` contains placeholders only.

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
python -m src.decision_intelligence_engine.explain_image --image path/to/image.png --no-llm
python -m src.decision_intelligence_engine.explain_image --image path/to/image.png --question "What is shown in this image and how confident is the model?"
```

The explain-image CLI supports classifier-only verification with `--no-llm` and explanation mode through the OpenAI Responses API. The classifier remains the source of the prediction; the LLM only explains that prediction.

## Architecture

The system is organized around four layers: data pipeline, model training/evaluation, experiment tracking, and an inference-to-LLM explanation layer. Model inference and OpenAI interaction are intentionally separate so classifier behavior remains testable and deterministic even when the LLM is disabled or unavailable.

## Model Results

The current selected MobileNetV2 configuration is the frozen longer-training variant based on explicit validation-first criteria for Phase 4B. After the additional Phase 5A resolution-control run, EfficientNetB0 is selected as the final architecture because it still outperformed MobileNetV2 when both used 96x96 effective inputs. Phase 5B uses that finalized EfficientNetB0 artifact for inference and explanation.

## Limitations

The current explanation flow only explains classifier outputs within the CIFAR-10 label space. It does not verify whether the classifier is correct, and it must not be treated as an independent vision judgment. If OpenAI is unavailable, classifier-only mode still works.

## Reflection

The main lesson so far is that reproducible file-manifest splitting and serialization-safe preprocessing matter more than a quick baseline score. Local experiment tracking is now in place so later model iterations can be compared cleanly.

## Demo

The current demonstration is the controlled experiment matrix plus the final architecture comparison outputs. Start the UI with `mlflow ui --backend-store-uri file:./mlruns` after the experiment runs.
