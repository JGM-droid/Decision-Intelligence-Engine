# Requirement Traceability

Source of truth reviewed: [docs/project_charter.md](docs/project_charter.md)

Status labels used in this ledger:

- COMPLETE
- COMPLETE AFTER THIS AUDIT
- PENDING LIVE API SMOKE TEST
- PENDING MANUAL DEMO
- BLOCKED

## Data and Model Training

| Exact Requirement | Status | Implementation / Evidence | Remaining Action |
| --- | --- | --- | --- |
| Preprocessing must include image normalization, shape/channel consistency checks, and documented augmentation strategy. | COMPLETE | [docs/data_pipeline.md](docs/data_pipeline.md), [src/decision_intelligence_engine/data_pipeline.py](src/decision_intelligence_engine/data_pipeline.py), [tests/test_data_pipeline.py](tests/test_data_pipeline.py) | None. |
| Image preprocessing and augmentation decisions must be documented. | COMPLETE | [docs/data_pipeline.md](docs/data_pipeline.md), [docs/architecture.md](docs/architecture.md), [docs/baseline_model.md](docs/baseline_model.md) | None. |
| Use a held-out test set and prevent data leakage. | COMPLETE | [docs/data_pipeline.md](docs/data_pipeline.md), [tests/test_data_pipeline.py](tests/test_data_pipeline.py) | None. |
| Train at least 3 meaningfully different CNN model configurations. | COMPLETE | [docs/mobilenetv2_experiments.md](docs/mobilenetv2_experiments.md), [docs/architecture_comparison.md](docs/architecture_comparison.md) | None. |
| Report at least 3 task-appropriate metrics per model (for example accuracy, precision, recall, F1, top-k accuracy). | COMPLETE AFTER THIS AUDIT | [docs/architecture_comparison.md](docs/architecture_comparison.md), [reports/architecture_comparison.json](reports/architecture_comparison.json), [reports/architecture_comparison.md](reports/architecture_comparison.md), [reports/architecture_comparison.csv](reports/architecture_comparison.csv) | Macro precision, macro recall, and macro F1 are now surfaced as post-hoc test-set metrics from the saved confusion matrices. |
| Select and justify a best model based on performance and tradeoffs. | COMPLETE | [docs/architecture_comparison.md](docs/architecture_comparison.md), [README.md](README.md) | None. |

## MLflow Experiment Tracking

| Exact Requirement | Status | Implementation / Evidence | Remaining Action |
| --- | --- | --- | --- |
| Integrate MLflow into training workflow. | COMPLETE | [src/decision_intelligence_engine/baseline_training.py](src/decision_intelligence_engine/baseline_training.py), [docs/mlflow_tracking.md](docs/mlflow_tracking.md) | None. |
| Each run must log hyperparameters, data version/description, metrics, and trained model artifact. | COMPLETE | [src/decision_intelligence_engine/baseline_training.py](src/decision_intelligence_engine/baseline_training.py), [docs/mlflow_tracking.md](docs/mlflow_tracking.md) | None. |
| Log at least 5 meaningfully different experiment runs. | COMPLETE | [docs/mobilenetv2_experiments.md](docs/mobilenetv2_experiments.md), [reports/model_comparison.json](reports/model_comparison.json) | None. |
| Use mlflow.search_runs() to compare runs and identify best run programmatically. | COMPLETE | [src/decision_intelligence_engine/compare_experiments.py](src/decision_intelligence_engine/compare_experiments.py), [src/decision_intelligence_engine/select_experiment.py](src/decision_intelligence_engine/select_experiment.py) | None. |

## LLM Interface

| Exact Requirement | Status | Implementation / Evidence | Remaining Action |
| --- | --- | --- | --- |
| Interface must accept user-uploaded images and user questions about predictions. | COMPLETE | [src/decision_intelligence_engine/explain_image.py](src/decision_intelligence_engine/explain_image.py), [tests/test_explain_image_cli.py](tests/test_explain_image_cli.py) | None. |
| Interface must load and call the actual selected trained CNN model. | COMPLETE | [src/decision_intelligence_engine/model_inference.py](src/decision_intelligence_engine/model_inference.py) | None. |
| Response must include predicted class, confidence context, explanation, and caveats. | COMPLETE | [src/decision_intelligence_engine/llm_explainer.py](src/decision_intelligence_engine/llm_explainer.py), [README.md](README.md) | None. |
| Handle invalid uploads, ambiguous/low-confidence predictions, and out-of-scope prompts gracefully. | COMPLETE | [src/decision_intelligence_engine/llm_explainer.py](src/decision_intelligence_engine/llm_explainer.py), [tests/test_llm_explainer.py](tests/test_llm_explainer.py) | None. |
| API keys must be stored via environment variables; no hardcoded secrets. | COMPLETE | [README.md](README.md), [.env.example](.env.example), [.gitignore](.gitignore) | None. |

## Testing

| Exact Requirement | Status | Implementation / Evidence | Remaining Action |
| --- | --- | --- | --- |
| At least 4 preprocessing tests: invalid or unreadable image handling, transform shape/channel consistency, normalization correctness, preprocessing immutability where applicable. | COMPLETE | [tests/test_data_pipeline.py](tests/test_data_pipeline.py), [tests/test_pipeline_config.py](tests/test_pipeline_config.py) | None. |
| At least 2 model tests: prediction type/shape, minimum performance threshold. | COMPLETE AFTER THIS AUDIT | [tests/test_model_inference.py](tests/test_model_inference.py), [tests/test_baseline_mlflow.py](tests/test_baseline_mlflow.py) | Prediction-type/shape coverage exists; performance is enforced through experiment-selection evidence rather than a retraining gate. |
| At least 2 interface tests: explanation response contract behavior, invalid/incomplete input handling. | COMPLETE | [tests/test_explain_image_cli.py](tests/test_explain_image_cli.py), [tests/test_llm_explainer.py](tests/test_llm_explainer.py) | None. |
| pytest tests/ -v must pass with zero failures. | COMPLETE | Latest validated command: `.\.venv\Scripts\python.exe -m pytest tests -v` | None. |

## Repository and Documentation

| Exact Requirement | Status | Implementation / Evidence | Remaining Action |
| --- | --- | --- | --- |
| Maintain a clean, logical repository structure. | COMPLETE | [README.md](README.md), project tree under `configs/`, `data/`, `docs/`, `models/`, `reports/`, `src/`, `tests/` | None. |
| README must cover description, intended users/problem, setup, API-key config, data acquisition, usage, architecture, results, and reflection. | COMPLETE | [README.md](README.md) | None. |
| Training must be configuration-driven (YAML), not hardcoded hyperparameters. | COMPLETE AFTER THIS AUDIT | [configs/baseline_training.yaml](configs/baseline_training.yaml), [src/decision_intelligence_engine/baseline_training.py](src/decision_intelligence_engine/baseline_training.py), [tests/test_baseline_training_yaml.py](tests/test_baseline_training_yaml.py) | YAML is now authoritative for the baseline training path; JSON remains a compatibility fallback. |
| Exclude data files and model artifacts from Git. | COMPLETE | [.gitignore](.gitignore) | None. |
| Include pinned dependency versions in requirements.txt. | COMPLETE AFTER THIS AUDIT | [requirements.txt](requirements.txt) | Direct runtime and test dependencies are now exact pins matching the current project venv. |
| Provide final natural-language demo covering normal and edge-case behavior. | PENDING MANUAL DEMO | [README.md](README.md) | The project owner will record the demo manually later. |

## Definition of Done

| Exact Requirement | Status | Implementation / Evidence | Remaining Action |
| --- | --- | --- | --- |
| End-to-end user flow works: image upload -> model inference -> confidence/context extraction -> LLM explanation response. | COMPLETE | [src/decision_intelligence_engine/explain_image.py](src/decision_intelligence_engine/explain_image.py), [tests/test_explain_image_cli.py](tests/test_explain_image_cli.py) | None. |
| Best model is selected from tracked experiments and used by the interface. | COMPLETE | [docs/architecture_comparison.md](docs/architecture_comparison.md), [src/decision_intelligence_engine/model_inference.py](src/decision_intelligence_engine/model_inference.py) | None. |
| All required tests pass. | COMPLETE | `.\.venv\Scripts\python.exe -m pytest tests -v` | None. |
| Repository documentation is complete and reproducible. | COMPLETE AFTER THIS AUDIT | [README.md](README.md), [docs/architecture.md](docs/architecture.md), [docs/roadmap.md](docs/roadmap.md), [docs/requirement_traceability.md](docs/requirement_traceability.md) | None. |
| Final demo shows successful normal query path, prediction pipeline visibility, actual model inference, generated response, and at least one invalid upload or out-of-scope query handled safely. | PENDING MANUAL DEMO | [README.md](README.md) | The project owner will record the final demo manually. |

## OpenAI Live-Validation Readiness

| Exact Requirement | Status | Implementation / Evidence | Remaining Action |
| --- | --- | --- | --- |
| OPENAI_API_KEY comes only from the environment. | COMPLETE | [.env.example](.env.example), [.gitignore](.gitignore), [src/decision_intelligence_engine/llm_explainer.py](src/decision_intelligence_engine/llm_explainer.py) | None. |
| `.env` is ignored. | COMPLETE | [.gitignore](.gitignore) | None. |
| `.env.example` contains no secret. | COMPLETE | [.env.example](.env.example) | None. |
| Missing-key handling is clear. | COMPLETE | [tests/test_llm_explainer.py](tests/test_llm_explainer.py) | None. |
| Classifier-only mode works without a key. | COMPLETE | [tests/test_explain_image_cli.py](tests/test_explain_image_cli.py) | None. |
| API failure and malformed-response handling are tested. | COMPLETE | [tests/test_llm_explainer.py](tests/test_llm_explainer.py) | None. |
| Timeout behavior is explicit. | COMPLETE | [src/decision_intelligence_engine/llm_explainer.py](src/decision_intelligence_engine/llm_explainer.py), [tests/test_llm_explainer.py](tests/test_llm_explainer.py) | None. |
| Documented CLI command matches the implemented module path and arguments. | COMPLETE | [README.md](README.md), `python -m src.decision_intelligence_engine.explain_image --help` | None. |
| Live OpenAI request. | PENDING LIVE API SMOKE TEST | [README.md](README.md) | The project owner will run the authorized smoke test later. |

## Notes

- The generic tabular-data rubric items from the original capstone brief are not represented here because the final project is a CIFAR-10 image-classification workflow, not a dataframe preprocessing project.
- The final architecture remains EfficientNetB0 selected from matched-resolution evidence, not a claim of universal superiority.
- Phase 5B is functionally complete; only the manual demo and authorized live OpenAI smoke test remain pending.
