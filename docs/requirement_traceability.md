# Requirement Traceability

| Category | Requirement | Syllabus Minimum | Status | Evidence | Notes |
| --- | --- | --- | --- | --- | --- |
| Data and model | dataset has at least 500 rows | Yes | Not Started | TBD | |
| Data and model | dataset has a clear target variable | Yes | Not Started | TBD | |
| Data and model | missing values handled | Yes | Not Started | TBD | |
| Data and model | categorical variables encoded where applicable | Yes | Not Started | TBD | |
| Data and model | numeric variables scaled or normalized where appropriate | Yes | Not Started | TBD | |
| Data and model | preprocessing decisions documented | Yes | Not Started | TBD | |
| Data and model | no data leakage | Yes | Not Started | TBD | |
| Data and model | held-out test set used | Yes | Not Started | TBD | |
| Data and model | at least 3 meaningfully different model configurations | Yes | Completed | [docs/mobilenetv2_experiments.md](docs/mobilenetv2_experiments.md) | Five controlled MobileNetV2 configurations executed. |
| Data and model | at least 3 appropriate evaluation metrics per model | Yes | Completed | [reports/model_comparison.md](reports/model_comparison.md) | Train/validation/test accuracy and loss logged per run. |
| Data and model | reasonable model performance | Yes | Completed | [docs/architecture_comparison.md](docs/architecture_comparison.md) | EfficientNetB0 reached 0.5405 validation accuracy and 0.5440 test accuracy in the final matched-resolution comparison. |
| Data and model | best model selected and justified | Yes | Completed | [docs/architecture_comparison.md](docs/architecture_comparison.md) | EfficientNetB0 was selected after a three-way comparison against MobileNetV2 32x32 and a MobileNetV2 96x96 resolution-control run. |
| Data and model | controlled cross-architecture screening before architecture tuning | Yes | Completed | [docs/architecture_comparison.md](docs/architecture_comparison.md) | Phase 5A adds a matched-resolution MobileNetV2 control, a final EfficientNetB0 comparison run, and deterministic architecture-selection logic. |
| MLflow | MLflow integrated into training workflow | Yes | Completed | [src/decision_intelligence_engine/baseline_training.py](src/decision_intelligence_engine/baseline_training.py) | Local MLflow tracking added for the corrected baseline. |
| MLflow | every run logs all hyperparameters | Yes | Completed | [src/decision_intelligence_engine/baseline_training.py](src/decision_intelligence_engine/baseline_training.py) | Hyperparameters, data config, and baseline settings are logged. |
| MLflow | every run logs data version or data description | Yes | Completed | [configs/mlflow.json](configs/mlflow.json) | Run metadata includes the canonical data layout and split strategy. |
| MLflow | every run logs all evaluation metrics | Yes | Completed | [src/decision_intelligence_engine/baseline_training.py](src/decision_intelligence_engine/baseline_training.py) | Train, validation, test, and history metrics are logged. |
| MLflow | every run logs the trained model as an artifact | Yes | Completed | [src/decision_intelligence_engine/baseline_training.py](src/decision_intelligence_engine/baseline_training.py) | Keras model, reports, and diagnostic plots are logged. |
| MLflow | at least 5 meaningfully different experiment runs | Yes | Completed | [docs/mobilenetv2_experiments.md](docs/mobilenetv2_experiments.md) | Controlled five-run Phase 4B matrix completed. |
| MLflow | mlflow.search_runs() used | Yes | Completed | [src/decision_intelligence_engine/compare_experiments.py](src/decision_intelligence_engine/compare_experiments.py) | Run retrieval and filtering are driven by MLflow search APIs. |
| MLflow | best run identified programmatically | Yes | Completed | [src/decision_intelligence_engine/select_experiment.py](src/decision_intelligence_engine/select_experiment.py) | Deterministic selection outputs strongest frozen, fine-tuned, and overall run. |
| MLflow | architecture comparison logged and reported from MLflow data | Yes | Completed | [src/decision_intelligence_engine/compare_experiments.py](src/decision_intelligence_engine/compare_experiments.py) | Phase 5A architecture comparison reports are generated from MLflow search results without manual metric entry. |
| LLM interface | focused interface for the trained model, not a general chatbot | Yes | Completed | [src/decision_intelligence_engine/explain_image.py](src/decision_intelligence_engine/explain_image.py) | CLI accepts an image path and optional natural-language question for classifier explanation only. |
| LLM interface | natural-language input parsed into required model features | Yes | Completed | [src/decision_intelligence_engine/llm_explainer.py](src/decision_intelligence_engine/llm_explainer.py) | User question is combined with predicted class, confidence, and top predictions before the OpenAI call. |
| LLM interface | actual selected trained model loaded | Yes | Completed | [src/decision_intelligence_engine/model_inference.py](src/decision_intelligence_engine/model_inference.py) | Selected EfficientNetB0 artifact is resolved from Phase 5A architecture-selection evidence and downloaded through MLflow. |
| LLM interface | parsed features passed into the model | Yes | Completed | [src/decision_intelligence_engine/model_inference.py](src/decision_intelligence_engine/model_inference.py) | Incoming images are validated, resized to the model's expected input tensor shape, and passed into the saved classifier. |
| LLM interface | prediction returned | Yes | Completed | [src/decision_intelligence_engine/explain_image.py](src/decision_intelligence_engine/explain_image.py) | CLI returns predicted class, confidence, and top predictions. |
| LLM interface | contextual explanation generated | Yes | Completed | [src/decision_intelligence_engine/llm_explainer.py](src/decision_intelligence_engine/llm_explainer.py) | OpenAI Responses API returns a concise explanation of the classifier output. |
| LLM interface | caveats and limitations included | Yes | Completed | [src/decision_intelligence_engine/llm_explainer.py](src/decision_intelligence_engine/llm_explainer.py) | Prompt requires stating CIFAR-10 scope and that confidence is not proof. |
| LLM interface | missing input handled | Yes | Completed | [src/decision_intelligence_engine/llm_explainer.py](src/decision_intelligence_engine/llm_explainer.py) | Missing or blank questions fall back to a safe default explanation request. |
| LLM interface | ambiguous input handled | Yes | Completed | [src/decision_intelligence_engine/llm_explainer.py](src/decision_intelligence_engine/llm_explainer.py) | Vague or out-of-scope questions are normalized to a deterministic classifier-focused explanation. |
| LLM interface | out-of-scope input handled | Yes | Completed | [src/decision_intelligence_engine/llm_explainer.py](src/decision_intelligence_engine/llm_explainer.py) | Unsupported requests are answered with explicit CIFAR-10 scope limitations. |
| LLM interface | functional, understandable user interface | Yes | Completed | [src/decision_intelligence_engine/explain_image.py](src/decision_intelligence_engine/explain_image.py) | CLI provides classifier-only and explanation modes with clear errors and non-zero exit codes on failure. |
| LLM interface | API keys stored in environment variables | Yes | Completed | [README.md](README.md) | OpenAI access uses `OPENAI_API_KEY` and optional `OPENAI_MODEL` from the environment. |
| LLM interface | no hardcoded secrets | Yes | Completed | [.env.example](.env.example) | Placeholder-only env example added; `.env` remains ignored by Git. |
| Testing | at least 4 preprocessing tests | Yes | Not Started | TBD | |
| Testing | preprocessing test for missing values | Yes | Not Started | TBD | |
| Testing | preprocessing test for categorical encoding | Yes | Not Started | TBD | |
| Testing | preprocessing test for numeric scaling | Yes | Not Started | TBD | |
| Testing | preprocessing test confirming original dataframe is not modified | Yes | Not Started | TBD | |
| Testing | at least 2 model tests | Yes | Completed | [tests/test_model_inference.py](tests/test_model_inference.py) | Model loading, inference, artifact resolution, class mapping, and invalid image behavior are covered. |
| Testing | model prediction type and shape test | Yes | Completed | [tests/test_model_inference.py](tests/test_model_inference.py) | Inference tests validate confidence bounds, class mapping, top-k output, and prediction shape assumptions. |
| Testing | minimum performance threshold test | Yes | Not Started | TBD | |
| Testing | at least 2 interface tests | Yes | Completed | [tests/test_explain_image_cli.py](tests/test_explain_image_cli.py) | CLI classifier-only mode, explanation mode, and non-zero failure behavior are covered. |
| Testing | natural-language parsing test | Yes | Completed | [tests/test_llm_explainer.py](tests/test_llm_explainer.py) | Default, vague, and out-of-scope question handling is covered deterministically. |
| Testing | incomplete or invalid input handling test | Yes | Completed | [tests/test_model_inference.py](tests/test_model_inference.py) | Missing image path, corrupted image, and API-key failure paths are covered. |
| Testing | pytest tests/ -v passes with zero failures | Yes | Completed | [tests/test_llm_explainer.py](tests/test_llm_explainer.py) | Phase 5B verification includes a passing full test suite. |
| Repository and documentation | clean logical repository structure | Yes | Not Started | TBD | |
| Repository and documentation | README project description | Yes | Not Started | TBD | |
| Repository and documentation | README intended users and problem solved | Yes | Not Started | TBD | |
| Repository and documentation | README setup instructions | Yes | Not Started | TBD | |
| Repository and documentation | README API-key configuration instructions | Yes | Not Started | TBD | |
| Repository and documentation | README data acquisition instructions | Yes | Not Started | TBD | |
| Repository and documentation | README usage instructions | Yes | Not Started | TBD | |
| Repository and documentation | README architecture overview | Yes | Not Started | TBD | |
| Repository and documentation | README model results summary | Yes | Not Started | TBD | |
| Repository and documentation | README reflection | Yes | Not Started | TBD | |
| Repository and documentation | YAML training configuration | Yes | Not Started | TBD | Deferred until scope decisions are finalized. |
| Repository and documentation | no hardcoded training hyperparameters | Yes | Not Started | TBD | |
| Repository and documentation | data files excluded from Git | Yes | Not Started | TBD | |
| Repository and documentation | model artifacts excluded from Git | Yes | Not Started | TBD | |
| Repository and documentation | requirements.txt with pinned versions | Yes | Not Started | TBD | Deferred until stack decisions are finalized. |
| Repository and documentation | public GitHub repository | Yes | Not Started | TBD | |
| Repository and documentation | final natural-language demo | Yes | Not Started | TBD | |
| Repository and documentation | final demo shows parsing | Yes | Not Started | TBD | |
| Repository and documentation | final demo shows actual model inference | Yes | Not Started | TBD | |
| Repository and documentation | final demo shows generated response | Yes | Not Started | TBD | |
| Repository and documentation | final demo shows at least one incomplete or out-of-scope query | Yes | Not Started | TBD | |
| Repository and documentation | optional working Dockerfile | Optional | Not Started | TBD | Optional deliverable. |
| Repository and documentation | optional docker build and docker run instructions | Optional | Not Started | TBD | Optional deliverable. |
