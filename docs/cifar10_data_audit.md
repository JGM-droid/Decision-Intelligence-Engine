# CIFAR-10 Dataset Audit

## Summary

- Source used in this environment: https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz
- Archive location: data/raw/cifar-10-python.tar.gz
- Extracted location: data/raw/cifar10
- Canonical dataset format: folder-based image layout
- Archive size (bytes): 135107811

## Dataset Structure

- Canonical raw layout:
    - train/<class>/*.png
    - test/<class>/*.png
- Label set (10 classes): airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Counts and Splits

- Total images: 60000
- Training images: 50000
- Test images: 10000
- Split ratio: 50000:10000

## Image Properties

- Width x Height: 32 x 32
- Channels: 3 (RGB)
- Flattened vector length per sample: 3072

## Class Distribution (Training)

| Class | Count |
| --- | ---: |
| airplane | 5000 |
| automobile | 5000 |
| bird | 5000 |
| cat | 5000 |
| deer | 5000 |
| dog | 5000 |
| frog | 5000 |
| horse | 5000 |
| ship | 5000 |
| truck | 5000 |

## Class Distribution (Test)

| Class | Count |
| --- | ---: |
| airplane | 1000 |
| automobile | 1000 |
| bird | 1000 |
| cat | 1000 |
| deer | 1000 |
| dog | 1000 |
| frog | 1000 |
| horse | 1000 |
| ship | 1000 |
| truck | 1000 |

## Label Mapping

| Label ID | Label Name |
| ---: | --- |
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

## Sample Visualizations

The script exported two sample images per class to docs/assets/cifar10_samples.

- airplane: docs/assets/cifar10_samples/airplane_sample_1.png, docs/assets/cifar10_samples/airplane_sample_2.png
- automobile: docs/assets/cifar10_samples/automobile_sample_1.png, docs/assets/cifar10_samples/automobile_sample_2.png
- bird: docs/assets/cifar10_samples/bird_sample_1.png, docs/assets/cifar10_samples/bird_sample_2.png
- cat: docs/assets/cifar10_samples/cat_sample_1.png, docs/assets/cifar10_samples/cat_sample_2.png
- deer: docs/assets/cifar10_samples/deer_sample_1.png, docs/assets/cifar10_samples/deer_sample_2.png
- dog: docs/assets/cifar10_samples/dog_sample_1.png, docs/assets/cifar10_samples/dog_sample_2.png
- frog: docs/assets/cifar10_samples/frog_sample_1.png, docs/assets/cifar10_samples/frog_sample_2.png
- horse: docs/assets/cifar10_samples/horse_sample_1.png, docs/assets/cifar10_samples/horse_sample_2.png
- ship: docs/assets/cifar10_samples/ship_sample_1.png, docs/assets/cifar10_samples/ship_sample_2.png
- truck: docs/assets/cifar10_samples/truck_sample_1.png, docs/assets/cifar10_samples/truck_sample_2.png

## Preprocessing Needs (Phase 2A Recommendation)

- Normalization: Required. Scale pixel values from [0, 255] to [0, 1], then apply channel-wise normalization for stable CNN training.
- Resizing: Not required for baseline, because CIFAR-10 is consistently 32x32 RGB. Optional resizing can be used only if a later backbone requires larger inputs.
- Augmentation opportunities: horizontal flip, random crop with padding, mild color jitter, and cutout-like masking can improve generalization.
- Label encoding: use deterministic class-to-index mapping from folder names and persist it with model artifacts.
- Train/validation strategy: keep official test set untouched; derive validation split from training data with stratification and fixed random seed.

## Pipeline Design Recommendation

1. Keep raw dataset immutable in data/raw.
2. Build deterministic split metadata (train/val/test indices) and version it in docs or configs.
3. Implement separate transform stacks:
   - train: normalization + augmentation
   - val/test/inference: normalization only
4. Persist class-name mapping alongside model artifacts for inference/LLM explanation consistency.
5. Add data-quality checks before training (shape, channel count, class count, per-class cardinality).

## Risks Identified

- Low image resolution (32x32) limits fine-grained interpretability of visual features.
- LLM may over-explain uncertain predictions unless confidence guardrails are enforced.
- Augmentation overuse can hurt rather than help if not validated in MLflow experiments.
