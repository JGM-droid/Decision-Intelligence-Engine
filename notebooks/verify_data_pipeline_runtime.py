"""Runtime verification for the Phase 3A CIFAR-10 data pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.decision_intelligence_engine import (
    build_cifar10_datasets,
    inspect_dataset_batch,
    load_data_pipeline_config,
)


def first_batch_arrays(dataset):
    images, labels = next(iter(dataset.take(1)))
    return images.numpy(), labels.numpy()


def full_label_range(dataset) -> tuple[int, int]:
    min_seen = None
    max_seen = None
    for _, labels in dataset:
        labels_np = labels.numpy()
        local_min = int(labels_np.min())
        local_max = int(labels_np.max())
        min_seen = local_min if min_seen is None else min(min_seen, local_min)
        max_seen = local_max if max_seen is None else max(max_seen, local_max)
    return int(min_seen), int(max_seen)


def main() -> None:
    root = Path(".")
    cfg = load_data_pipeline_config(root)

    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")
    print(f"Using config: {cfg}")

    bundle = build_cifar10_datasets(root, cfg)
    print("datasets_constructed:", {
        "train": bundle.train_ds is not None,
        "val": bundle.val_ds is not None,
        "test": bundle.test_ds is not None,
    })

    train_info = inspect_dataset_batch(bundle.train_ds)
    val_info = inspect_dataset_batch(bundle.val_ds)
    test_info = inspect_dataset_batch(bundle.test_ds)

    print("class_count:", len(bundle.class_names))
    print("class_names:", bundle.class_names)
    print("sample_counts:", {
        "train": bundle.train_images,
        "val": bundle.val_images,
        "test": bundle.test_images,
    })
    print("batch_shapes:", {
        "train": train_info["image_shape"],
        "val": val_info["image_shape"],
        "test": test_info["image_shape"],
        "train_labels": train_info["label_shape"],
        "val_labels": val_info["label_shape"],
        "test_labels": test_info["label_shape"],
    })
    print("normalization:", {
        "train_min": train_info["image_min"],
        "train_max": train_info["image_max"],
        "val_min": val_info["image_min"],
        "val_max": val_info["image_max"],
        "test_min": test_info["image_min"],
        "test_max": test_info["image_max"],
    })

    train_label_range = full_label_range(bundle.train_ds)
    val_label_range = full_label_range(bundle.val_ds)
    test_label_range = full_label_range(bundle.test_ds)
    print("label_ranges:", {
        "train": train_label_range,
        "val": val_label_range,
        "test": test_label_range,
    })

    bundle_repeat = build_cifar10_datasets(root, cfg)
    val_a_x, val_a_y = first_batch_arrays(bundle.val_ds)
    val_b_x, val_b_y = first_batch_arrays(bundle_repeat.val_ds)
    test_a_x, test_a_y = first_batch_arrays(bundle.test_ds)
    test_b_x, test_b_y = first_batch_arrays(bundle_repeat.test_ds)

    print("deterministic_eval:", {
        "val_images_equal": bool((val_a_x == val_b_x).all()),
        "val_labels_equal": bool((val_a_y == val_b_y).all()),
        "test_images_equal": bool((test_a_x == test_b_x).all()),
        "test_labels_equal": bool((test_a_y == test_b_y).all()),
    })

    cfg_no_aug = cfg.__class__(**{**cfg.__dict__, "augmentation_enabled": False})
    bundle_no_aug = build_cifar10_datasets(root, cfg_no_aug)

    tr_aug_x, _ = first_batch_arrays(bundle.train_ds)
    tr_no_aug_x, _ = first_batch_arrays(bundle_no_aug.train_ds)
    v_aug_x, _ = first_batch_arrays(bundle.val_ds)
    v_no_aug_x, _ = first_batch_arrays(bundle_no_aug.val_ds)
    t_aug_x, _ = first_batch_arrays(bundle.test_ds)
    t_no_aug_x, _ = first_batch_arrays(bundle_no_aug.test_ds)

    print("augmentation_scope:", {
        "train_differs_with_aug": bool(not (tr_aug_x == tr_no_aug_x).all()),
        "val_same_with_or_without_aug": bool((v_aug_x == v_no_aug_x).all()),
        "test_same_with_or_without_aug": bool((t_aug_x == t_no_aug_x).all()),
    })


if __name__ == "__main__":
    main()
