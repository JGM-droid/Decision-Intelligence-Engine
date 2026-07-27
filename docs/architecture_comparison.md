# Architecture Comparison (Phase 5A)

## Objective

Phase 5A completes a controlled architecture comparison across three approved runs:

- MobileNetV2 32x32 selected frozen baseline from Phase 4B
- MobileNetV2 96x96 frozen resolution-control run
- EfficientNetB0 96x96 frozen architecture comparison run

Selected MobileNetV2 reference:

- experiment_id: mnetv2_longer_frozen_epochs
- backbone: MobileNetV2 (ImageNet)
- frozen backbone
- epochs: 3
- learning_rate: 0.0003
- dropout_rate: 0.20
- batch_size: 128

## Why EfficientNetB0

EfficientNetB0 was selected as the next architecture candidate because it is a canonical ImageNet-pretrained backbone with a stronger parameter-efficiency design than legacy mobile baselines, while still being practical for local controlled experimentation.

## Controlled Experiments

Experiment files:

- configs/experiments/06_efficientnetb0_frozen_baseline.json
- configs/experiments/07_mobilenetv2_96x96_control.json

EfficientNetB0 architecture comparison run:

- experiment_id: efficientnetb0_control_frozen
- experiment_name: EfficientNetB0 Frozen Architecture Baseline
- experiment_category: architecture_screening
- changed_variable: architecture

MobileNetV2 resolution-control run:

- experiment_id: mobilenetv2_96x96_control
- experiment_name: MobileNetV2 96x96 Resolution Control
- experiment_category: resolution_control
- changed_variable: model_input_resolution

Configuration controls held constant where scientifically appropriate:

- pretrained_weights: imagenet
- freeze_backbone: true
- optimizer: Adam
- loss: SparseCategoricalCrossentropy
- epochs: 3
- learning_rate: 0.0003
- dropout_rate: 0.2
- batch_size: 128
- random_seed: 42
- augmentation_enabled: true
- split_strategy: deterministic_stratified_file_manifest

## Required Differences

MobileNetV2 96x96 control changes only:

- model_input_resolution: [96, 96]
- preprocessing_function remains mobilenetv2_rescale_neg1_to_1

EfficientNetB0 changes explicit architecture-required fields:

- model_input_resolution: [96, 96]
- preprocessing_function: efficientnetb0_builtin_rescaling_with_input_scale_255

Original dataset properties are unchanged:

- source examples and split membership unchanged
- original CIFAR-10 storage resolution remains 32x32
- labels and class counts unchanged

## Results Summary

| Model | Experiment ID | Input Resolution | Preprocessing | Validation Accuracy | Test Accuracy | Validation Loss | Test Loss | Duration Sec |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| MobileNetV2 | mnetv2_longer_frozen_epochs | [32, 32] | mobilenetv2_rescale_neg1_to_1 | 0.1485 | 0.1563 | 2.2785 | 2.2751 | 26.6103 |
| MobileNetV2 | mobilenetv2_96x96_control | [96, 96] | mobilenetv2_rescale_neg1_to_1 | 0.3912 | 0.3909 | 2.4730 | 2.4620 | 124.1321 |
| EfficientNetB0 | efficientnetb0_control_frozen | [96, 96] | efficientnetb0_builtin_rescaling_with_input_scale_255 | 0.5405 | 0.5440 | 1.2762 | 1.2862 | 170.4359 |

## Comparison Rule

Primary selection metric:

- validation accuracy

Tie-breakers:

1. lower validation loss
2. smaller train-validation gap
3. lower complexity
4. shorter duration

Material-improvement threshold used in the decision logic:

- absolute validation-accuracy improvement >= 0.01

## Final Conclusion

- MobileNetV2 improved materially when moved from 32x32 to 96x96, confirming input resolution was a real confound in the original two-run comparison.
- EfficientNetB0 still outperformed MobileNetV2 at the matched 96x96 input resolution on validation accuracy, test accuracy, validation loss, and test loss.
- The final Phase 5A evidence supports selecting EfficientNetB0 as the final architecture for this project scope.

## Phase 5A Limitation

The original two-run comparison was not architecture-only because input resolution and architecture-specific preprocessing changed together. The added MobileNetV2 96x96 control removes the largest confound, but the final matched-resolution comparison still preserves backbone-specific preprocessing differences required by each pretrained model family.

## Outputs

The architecture comparison is generated from MLflow data and written to:

- reports/architecture_comparison.csv
- reports/architecture_comparison.json
- reports/architecture_comparison.md

Decision output CLI:

- python -m src.decision_intelligence_engine.select_architecture
