# Project Charter

## Project Goal

Build Decision Intelligence Engine: an AI-powered computer vision assistant that combines a CNN trained on CIFAR-10 with an LLM-powered explanation layer, supported by reproducible experimentation, robust testing, and clear documentation.

## Project Scope

- Build an end-to-end intelligent application with two connected components:
	- a trained CNN image classifier for CIFAR-10
	- an LLM-powered explanation interface that uses the trained model for inference
- Work from CIFAR-10 image data (60,000 labeled images, 10 classes).
- Deliver production-minded engineering artifacts: experiment tracking, tests, documentation, and a working demo.

## Non-Goals

- Not a general-purpose chatbot.
- Not a mock inference workflow.
- Not a repository that commits raw data or model artifact binaries.

## Required System Components

- Trained CNN image classification model (CIFAR-10)
- Natural-language LLM interface
- MLflow experiment tracking
- Automated tests
- Documented GitHub repository
- Short end-to-end demo

## Team Roles

- ChatGPT = lead architect and reviewer
- GitHub Copilot = implementation engineer
- User = project owner and command executor

## Engineering Principles

- Architecture first
- Small verified phases
- No unnecessary refactors
- Grading traceability
- Secure API-key handling
- Reproducible experiments
- No data leakage
- No mock model in the final interface
- Actual best trained model must be used for inference

## Technical Requirements Snapshot

### Data and Model Training

- Preprocessing must include image normalization, shape/channel consistency checks, and documented augmentation strategy.
- Image preprocessing and augmentation decisions must be documented.
- Use a held-out test set and prevent data leakage.
- Train at least 3 meaningfully different CNN model configurations.
- Report at least 3 task-appropriate metrics per model (for example accuracy, precision, recall, F1, top-k accuracy).
- Select and justify a best model based on performance and tradeoffs.

### MLflow Experiment Tracking

- Integrate MLflow into training workflow.
- Each run must log hyperparameters, data version/description, metrics, and trained model artifact.
- Log at least 5 meaningfully different experiment runs.
- Use mlflow.search_runs() to compare runs and identify best run programmatically.

### LLM Interface

- Interface must accept user-uploaded images and user questions about predictions.
- Interface must load and call the actual selected trained CNN model.
- Response must include predicted class, confidence context, explanation, and caveats.
- Handle invalid uploads, ambiguous/low-confidence predictions, and out-of-scope prompts gracefully.
- API keys must be stored via environment variables; no hardcoded secrets.

### Testing

- At least 4 preprocessing tests:
	- invalid or unreadable image handling
	- transform shape/channel consistency
	- normalization correctness
	- preprocessing immutability where applicable
- At least 2 model tests:
	- prediction type/shape
	- minimum performance threshold
- At least 2 interface tests:
	- explanation response contract behavior
	- invalid/incomplete input handling
- pytest tests/ -v must pass with zero failures.

### Repository and Documentation

- Maintain a clean, logical repository structure.
- README must cover description, intended users/problem, setup, API-key config, data acquisition, usage, architecture, results, and reflection.
- Training must be configuration-driven (YAML), not hardcoded hyperparameters.
- Exclude data files and model artifacts from Git.
- Include pinned dependency versions in requirements.txt.
- Provide final natural-language demo covering normal and edge-case behavior.
- Optional stretch: working Dockerfile and docker build/run instructions.

## Delivery Formats Allowed

- Jupyter notebook with interactive loop
- Streamlit app (selected target interface)
- Command-line application
- Flask/FastAPI endpoint
- Optional Dockerized packaging

## Evaluation Priorities (100 Points + Bonus)

- Data and model quality: 20 points
- Experiment tracking: 15 points
- LLM interface quality: 30 points
- Testing: 15 points
- Documentation and structure: 20 points
- Bonus: Docker packaging up to +3 points

## Definition of Done

- End-to-end user flow works: image upload -> model inference -> confidence/context extraction -> LLM explanation response.
- Best model is selected from tracked experiments and used by the interface.
- All required tests pass.
- Repository documentation is complete and reproducible.
- Final demo shows:
	- successful normal query path
	- prediction pipeline visibility
	- actual model inference
	- generated response
	- at least one invalid upload or out-of-scope query handled safely

## Execution Phases

1. Architecture finalization for CIFAR-10 CNN + LLM explanation design.
2. Dataset acquisition, EDA, and image preprocessing design.
3. CNN training baseline and model variation experiments.
4. MLflow integration and experiment run expansion.
5. Best-model selection and inference artifact freeze.
6. LLM explanation integration and edge-case behavior.
7. Streamlit interface implementation.
8. Test suite completion and stabilization.
9. Optional Docker packaging and verification.
10. Final documentation and demo preparation.

## Traceability Source of Truth

- Requirement-level progress and evidence tracking is maintained in docs/requirement_traceability.md.
- Milestone execution plan is maintained in docs/roadmap.md.
