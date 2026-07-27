# Baseline Model (Phase 3B)

## Why MobileNetV2

MobileNetV2 was selected as the first transfer-learning baseline because it is:

- lightweight and fast for verification-focused training
- widely used and stable in TensorFlow/Keras
- compatible with ImageNet pretrained initialization

This phase prioritizes pipeline correctness and reproducibility, not final accuracy.

## Architecture Summary

- Backbone: MobileNetV2 (ImageNet weights, include_top=False)
- Backbone state: frozen (not trainable)
- Head:
  - Rescaling to MobileNetV2 expected input range [-1, 1]
  - GlobalAveragePooling2D
  - Dropout(0.2)
  - Dense(10, softmax)

## Trainable vs Frozen Parameters

- Trainable parameters: 12,810
- Frozen parameters: 2,257,984

## Preprocessing Used

Reused the existing production data pipeline (no duplicated dataset preprocessing code):

- canonical folder dataset loading from data/raw/cifar10
- image decoding and batching
- normalization to [0, 1]
- training-only augmentation:
  - horizontal flip
  - random crop with padding
- deterministic validation and test datasets

Model-side input adaptation for pretrained MobileNetV2 is applied with a serialization-safe Rescaling layer to convert [0, 1] to [-1, 1].

## Hyperparameters

Source: configs/baseline_training.json

- epochs: 1
- steps_per_epoch: 100
- learning_rate: 0.0003
- dropout_rate: 0.2
- validation_steps: full validation split
- evaluation steps: full train/validation/test splits
- random_seed: 42

## Evaluation Metrics

Corrected baseline run (stratified split):

- Train loss: 2.3139
- Train accuracy: 0.1289
- Validation loss: 2.3140
- Validation accuracy: 0.1345
- Test loss: 2.3105
- Test accuracy: 0.1401

## Confusion Matrix and Classification Report

Artifacts:

- reports/mobilenetv2_frozen_baseline_20260727_141855_confusion_matrix.csv
- reports/mobilenetv2_frozen_baseline_20260727_141855_classification_report.txt

Observed summary:

- Overall test accuracy: 0.14
- Macro F1: 0.08
- Model remains under-trained by design (single short epoch) and strongly biased toward a few classes.

## Corrective Review Note

The prior baseline run at timestamp 20260727_141254 used a defective validation split distribution produced by the previous splitting strategy.

- Evidence: validation split was heavily class-skewed, and validation accuracy (0.2443) was not aligned with test accuracy (0.1313).
- Impact: that run is invalid for model selection.
- Resolution: reran the unchanged frozen MobileNetV2 baseline after enforcing deterministic stratified split in the shared data pipeline.

Old vs corrected accuracy comparison:

- Old train/val/test: 0.1435 / 0.2443 / 0.1313
- Corrected train/val/test: 0.1289 / 0.1345 / 0.1401

The corrected validation and test metrics are now reasonably aligned for a short baseline run.

## Verification Checklist

- Model builds successfully
- Model trains successfully
- Model evaluates on train/validation/test successfully
- Model saves successfully
- Saved model loads successfully
- Inference on a small batch succeeds

## Observations

- End-to-end transfer-learning training pipeline is now functional and reproducible.
- Frozen-backbone baseline can be used as the control point for future MLflow experiments.

## Phase 4B Follow-Up

A controlled MobileNetV2 matrix was executed in Phase 4B with one-variable frozen variants and one constrained fine-tuning variant.

- strongest frozen configuration: `mnetv2_longer_frozen_epochs`
- strongest fine-tuned configuration: `mnetv2_partial_finetune_tail`
- overall selected MobileNetV2: `mnetv2_longer_frozen_epochs`

Selection used validation accuracy as the primary metric, with tie-breakers on validation loss, generalization gap, complexity, and duration.

## Limitations

- This run is intentionally short and under-trained.
- MobileNetV2 receives 32x32 inputs, so pretrained features are not used at their native scale.
- No fine-tuning was performed in this phase by design.
