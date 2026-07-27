# MobileNetV2 Experiments (Phase 4B)

## Objective

Phase 4B runs a controlled MobileNetV2 experiment matrix on the corrected deterministic CIFAR-10 split. The goal is to extract the strongest MobileNetV2 configuration before introducing any new architecture.

## Controlled Matrix

The approved matrix contains exactly five experiments:

1. `mnetv2_control_frozen`: control frozen baseline.
2. `mnetv2_longer_frozen_epochs`: epochs-only change.
3. `mnetv2_lower_lr_frozen`: learning-rate-only change.
4. `mnetv2_higher_dropout_frozen`: dropout-only change.
5. `mnetv2_partial_finetune_tail`: controlled fine-tuning exception (trainability and learning rate).

Configuration files live in `configs/experiments/`.

## One-Variable Policy

Experiments 2 through 4 are validated to change exactly one variable relative to control. The validation logic rejects accidental multi-variable drift.

Fine-tuning is the only approved exception. It changes:

- backbone trainability (unfreeze final controlled tail)
- learning rate (reduced for stability)

BatchNormalization layers remain frozen during fine-tuning.

## Constants Held Across Runs

All runs keep these constants unless explicitly approved otherwise:

- dataset and corrected deterministic stratified split
- preprocessing and augmentation policy
- input size and class count
- optimizer family (`Adam`) and loss (`SparseCategoricalCrossentropy`)
- random seed
- shared training/evaluation code path
- shared MLflow logging path and metric names

## MLflow Naming and Tags

- experiment name: `decision_intelligence_engine`
- run name pattern: `mobilenetv2_frozen_baseline_<experiment_id>_<timestamp>`
- required tags include: `experiment_id`, `experiment_category`, `changed_variable`, `control_flag`, `model_family`, `phase`

## Selection Criteria

Primary metric:

- highest validation accuracy

Tie-breakers, in order:

1. lower validation loss
2. smaller train-validation generalization gap
3. lower complexity (`total_params`) 
4. shorter training duration

Test accuracy is held for reporting/sanity only.

## Quality Gates

A run is ineligible if any of the following fail:

- MLflow status is not `FINISHED`
- required metrics are missing
- model reload verification is false
- inference smoke verification is false
- validation or test accuracy is outside [0, 1]
- train/val/test counts differ from expected corrected split counts
- split strategy differs from `deterministic_stratified_file_manifest`
- run cannot be mapped to an approved experiment ID

Warnings (non-fatal) include:

- validation accuracy below control
- generalization gap above 0.10
- duration above 45 seconds
- unexpectedly high trainable-parameter ratio

## Results

| Experiment ID | Category | Changed Variable | Val Acc | Val Loss | Test Acc | Duration (s) | Trainable Params |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| mnetv2_control_frozen | control | none | 0.1342 | 2.3146 | 0.1398 | 9.7609 | 12810 |
| mnetv2_longer_frozen_epochs | frozen_variant | epochs | 0.1485 | 2.2785 | 0.1563 | 26.6103 | 12810 |
| mnetv2_lower_lr_frozen | frozen_variant | learning_rate | 0.1176 | 2.3132 | 0.1241 | 13.2536 | 12810 |
| mnetv2_higher_dropout_frozen | frozen_variant | dropout_rate | 0.1351 | 2.3142 | 0.1400 | 12.4861 | 12810 |
| mnetv2_partial_finetune_tail | fine_tune | backbone_trainability+learning_rate | 0.1484 | 4.0919 | 0.1495 | 18.8480 | 1207690 |

## Selected Configurations

- strongest frozen: `mnetv2_longer_frozen_epochs`
- strongest fine-tuned: `mnetv2_partial_finetune_tail`
- overall selected MobileNetV2: `mnetv2_longer_frozen_epochs`

The frozen longer-epochs run wins by primary metric (validation accuracy), with a slight edge over partial fine-tuning. The improvement is modest and should be treated as incremental rather than transformative.

## Limitations

- MobileNetV2 still receives 32x32 images, below native pretraining scale.
- The matrix is intentionally conservative for local runtime.
- Findings are architecture-local and not yet compared against EfficientNet.

## Next Step Before Architecture Comparison

- lock this selected MobileNetV2 configuration as the architecture-comparison control
- proceed to cross-architecture experiments only after preserving this evidence baseline