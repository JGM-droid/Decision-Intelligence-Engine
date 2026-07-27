# Data Pipeline Architecture (Phase 3A)

## Overview

This pipeline is the reusable TensorFlow/Keras input layer for all future phases.

- Canonical dataset layout: data/raw/cifar10/train/<class>/*.png and data/raw/cifar10/test/<class>/*.png
- Official test set is preserved and never used for validation splitting.
- Validation split is created only from the training directory, with fixed random seed for reproducibility.

## Public Functions

- load_data_pipeline_config(project_root, config_path=None)
  - Loads centralized pipeline config from configs/data_pipeline.json.
  - Falls back to DataPipelineConfig defaults if config file is missing.

- build_cifar10_datasets(project_root, config)
  - Creates train/validation/test tf.data datasets.
  - Applies decoding, batching, normalization, optional augmentation, caching, and prefetching.
  - Returns dataset bundle with split sizes and class mapping metadata.

- inspect_dataset_batch(dataset)
  - Returns shape/dtype/min/max diagnostics for one batch.

## Preprocessing Flow

1. Decode images from canonical class folders via image_dataset_from_directory.
2. Batch images using configured batch_size.
3. Normalize pixel values from [0, 255] to [0, 1].
4. Apply tf.data performance options (parallel mapping, optional caching, prefetching).

## Augmentation Strategy

Training-only augmentation is modular and currently includes:

- zero-padding
- random crop back to target image size
- horizontal flip

Validation, test, and inference datasets do not receive augmentation.

## Assumptions

- CIFAR-10 is already available in canonical folder layout.
- Class folder names are consistent across train and test directories.
- Image size remains 32x32 for baseline pipeline; resizing behavior is still configurable.

## Configuration

Pipeline settings are centralized in configs/data_pipeline.json, including:

- image_size
- batch_size
- validation_split
- random_seed
- augmentation settings
- tf.data optimization toggles

This prevents hardcoded values from being scattered across modules.

## Runtime Verification (Completed)

Verified environment:

- Python executable: .venv/Scripts/python.exe
- TensorFlow: 2.16.2
- Keras: 3.15.0

Verified dataset split counts (canonical CIFAR-10 folder layout):

- train: 40000
- validation: 10000
- test: 10000

Verified batch shapes with configured batch_size=128:

- train images: (128, 32, 32, 3)
- validation images: (128, 32, 32, 3)
- test images: (128, 32, 32, 3)
- train labels: (128,)
- validation labels: (128,)
- test labels: (128,)

Normalization checks:

- train min/max: 0.0 / 1.0
- validation min/max: 0.0 / 1.0
- test min/max: 0.0 / 1.0

Label ID checks:

- train range: 0..9
- validation range: 0..9
- test range: 0..9

Determinism checks:

- validation data deterministic across repeated dataset construction with fixed seed
- test data deterministic across repeated dataset construction with fixed seed

Augmentation scope checks:

- training batches differ when augmentation is enabled vs disabled
- validation batches are identical with augmentation enabled vs disabled
- test batches are identical with augmentation enabled vs disabled

Verification commands used:

- python -m py_compile src/decision_intelligence_engine/pipeline_config.py src/decision_intelligence_engine/data_pipeline.py tests/test_pipeline_config.py tests/test_data_pipeline.py notebooks/verify_data_pipeline_runtime.py
- python -m pytest tests/test_pipeline_config.py tests/test_data_pipeline.py -v
- python notebooks/verify_data_pipeline_runtime.py

## Split Strategy Correction (Phase 3B Corrective Review)

### Defect Found

The previous validation split path relied on Keras directory splitting with non-representative ordering behavior in this environment. Validation data became class-skewed (ship/truck dominated), which made validation metrics unreliable for model selection.

### Canonical Fix

The pipeline now uses one canonical deterministic stratified file-manifest split:

1. Start from the official 50,000-image training pool only.
2. Split each class independently using configured validation_split and random_seed.
3. Build train/validation manifests from per-class shuffled indices.
4. Keep official 10,000-image test set untouched.
5. Assert no train/validation overlap.

This strategy is deterministic, class-balanced, and independent of folder/file alphabetical ordering.

### Per-Class Split Counts

| Class | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| airplane | 4000 | 1000 | 1000 |
| automobile | 4000 | 1000 | 1000 |
| bird | 4000 | 1000 | 1000 |
| cat | 4000 | 1000 | 1000 |
| deer | 4000 | 1000 | 1000 |
| dog | 4000 | 1000 | 1000 |
| frog | 4000 | 1000 | 1000 |
| horse | 4000 | 1000 | 1000 |
| ship | 4000 | 1000 | 1000 |
| truck | 4000 | 1000 | 1000 |

Totals:

- train: 40000
- validation: 10000
- test: 10000

