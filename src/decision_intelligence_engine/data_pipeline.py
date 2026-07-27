"""Reusable TensorFlow/Keras data pipeline for canonical CIFAR-10 layout."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .pipeline_config import DataPipelineConfig


@dataclass(frozen=True)
class DatasetBundle:
    """Container for train/validation/test datasets and metadata."""

    train_ds: Any
    val_ds: Any
    test_ds: Any
    class_names: tuple[str, ...]
    class_to_index: dict[str, int]
    train_images: int
    val_images: int
    test_images: int
    train_file_paths: tuple[str, ...]
    val_file_paths: tuple[str, ...]
    test_file_paths: tuple[str, ...]
    train_class_counts: dict[str, int]
    val_class_counts: dict[str, int]
    test_class_counts: dict[str, int]


@dataclass(frozen=True)
class SplitManifest:
    """Canonical deterministic split manifest for CIFAR-10."""

    class_names: tuple[str, ...]
    class_to_index: dict[str, int]
    train_paths: tuple[str, ...]
    train_labels: tuple[int, ...]
    val_paths: tuple[str, ...]
    val_labels: tuple[int, ...]
    test_paths: tuple[str, ...]
    test_labels: tuple[int, ...]

    def class_counts(self, labels: tuple[int, ...]) -> dict[str, int]:
        counts = Counter(labels)
        return {name: int(counts[idx]) for idx, name in enumerate(self.class_names)}


def _require_tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorFlow is required to run the data pipeline. "
            "Install it in the active environment before executing pipeline code."
        ) from exc
    return tf


def _autotune_or_value(tf: Any, num_parallel_calls: int) -> Any:
    return tf.data.AUTOTUNE if num_parallel_calls == -1 else int(num_parallel_calls)


def _count_images(path: Path) -> int:
    extensions = {".png", ".jpg", ".jpeg"}
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in extensions)


def _list_class_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def _list_image_files(root: Path) -> list[Path]:
    allowed = {".png", ".jpg", ".jpeg"}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in allowed])


def _decode_and_resize(tf: Any, path: Any, label: Any, image_size: tuple[int, int]) -> tuple[Any, Any]:
    bytes_ = tf.io.read_file(path)
    image = tf.io.decode_image(bytes_, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, image_size, method="bilinear")
    image = tf.cast(image, tf.float32)
    return image, label


def _build_dataset_from_manifest(
    tf: Any,
    paths: tuple[str, ...],
    labels: tuple[int, ...],
    config: DataPipelineConfig,
    shuffle: bool,
) -> Any:
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    if shuffle:
        ds = ds.shuffle(config.shuffle_buffer_size, seed=config.random_seed, reshuffle_each_iteration=True)

    num_calls = _autotune_or_value(tf, config.num_parallel_calls)
    ds = ds.map(
        lambda p, y: _decode_and_resize(tf, p, y, config.image_size),
        num_parallel_calls=num_calls,
    )
    ds = ds.map(lambda images, labels: _normalize_batch(tf, images, labels), num_parallel_calls=num_calls)
    ds = ds.batch(config.batch_size)
    return ds


def build_cifar10_split_manifest(project_root: Path, config: DataPipelineConfig) -> SplitManifest:
    """Create deterministic, stratified train/validation split and fixed test manifest.

    The split is performed per class from the official training pool, ensuring:
    - all classes appear in train and validation
    - class-balanced allocation according to validation_split
    - no train/validation overlap
    - reproducibility with configured seed
    """

    train_root = config.train_dir(project_root)
    test_root = config.test_dir(project_root)
    if not train_root.exists() or not test_root.exists():
        raise FileNotFoundError(
            "Canonical CIFAR-10 directories not found. Expected: "
            f"{train_root} and {test_root}"
        )

    class_dirs = _list_class_dirs(train_root)
    class_names = tuple([p.name for p in class_dirs])
    class_to_index = {name: idx for idx, name in enumerate(class_names)}

    rng = np.random.default_rng(config.random_seed)

    train_paths: list[str] = []
    train_labels: list[int] = []
    val_paths: list[str] = []
    val_labels: list[int] = []

    for class_name in class_names:
        class_idx = class_to_index[class_name]
        files = _list_image_files(train_root / class_name)
        if not files:
            raise ValueError(f"No image files found for class: {class_name}")

        perm = rng.permutation(len(files))
        val_count = int(round(len(files) * config.validation_split))
        if val_count <= 0 or val_count >= len(files):
            raise ValueError(
                f"Invalid validation split for class {class_name}: {val_count}/{len(files)}"
            )

        val_idx = perm[:val_count]
        train_idx = perm[val_count:]

        val_files = [str(files[i]) for i in val_idx]
        tr_files = [str(files[i]) for i in train_idx]

        val_paths.extend(val_files)
        val_labels.extend([class_idx] * len(val_files))
        train_paths.extend(tr_files)
        train_labels.extend([class_idx] * len(tr_files))

    # Stable ordering for deterministic eval traversal.
    val_pairs = sorted(zip(val_paths, val_labels), key=lambda x: x[0])
    test_pairs_unsorted: list[tuple[str, int]] = []
    for class_name in class_names:
        class_idx = class_to_index[class_name]
        files = _list_image_files(test_root / class_name)
        if not files:
            raise ValueError(f"No test files found for class: {class_name}")
        test_pairs_unsorted.extend([(str(p), class_idx) for p in files])
    test_pairs = sorted(test_pairs_unsorted, key=lambda x: x[0])

    train_path_set = set(train_paths)
    val_path_set = set([p for p, _ in val_pairs])
    if train_path_set & val_path_set:
        raise ValueError("Train/validation split overlap detected.")

    return SplitManifest(
        class_names=class_names,
        class_to_index=class_to_index,
        train_paths=tuple(train_paths),
        train_labels=tuple(train_labels),
        val_paths=tuple([p for p, _ in val_pairs]),
        val_labels=tuple([y for _, y in val_pairs]),
        test_paths=tuple([p for p, _ in test_pairs]),
        test_labels=tuple([y for _, y in test_pairs]),
    )


def _build_augmentation_model(tf: Any, config: DataPipelineConfig) -> Any:
    layers = []
    if config.augmentation_padding > 0:
        layers.append(tf.keras.layers.ZeroPadding2D(padding=config.augmentation_padding))
        layers.append(
            tf.keras.layers.RandomCrop(
                height=config.image_size[0],
                width=config.image_size[1],
                seed=config.random_seed,
            )
        )
    if config.augmentation_horizontal_flip:
        layers.append(tf.keras.layers.RandomFlip(mode="horizontal", seed=config.random_seed))
    return tf.keras.Sequential(layers, name="cifar10_train_augmentation")


def _normalize_batch(tf: Any, images: Any, labels: Any) -> tuple[Any, Any]:
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels


def _dataset_options(tf: Any, deterministic: bool) -> Any:
    options = tf.data.Options()
    options.experimental_deterministic = deterministic
    return options


def build_cifar10_datasets(project_root: Path, config: DataPipelineConfig) -> DatasetBundle:
    """Build train/validation/test datasets from canonical CIFAR-10 folder layout.

    Dataset assumptions:
    - data/raw/cifar10/train/<class>/*.png
    - data/raw/cifar10/test/<class>/*.png
    """

    tf = _require_tensorflow()
    manifest = build_cifar10_split_manifest(project_root, config)

    train_images = len(manifest.train_paths)
    val_images = len(manifest.val_paths)
    test_images = len(manifest.test_paths)

    parallel_calls = _autotune_or_value(tf, config.num_parallel_calls)

    train_ds = _build_dataset_from_manifest(
        tf,
        manifest.train_paths,
        manifest.train_labels,
        config,
        shuffle=True,
    )
    val_ds = _build_dataset_from_manifest(
        tf,
        manifest.val_paths,
        manifest.val_labels,
        config,
        shuffle=False,
    )
    test_ds = _build_dataset_from_manifest(
        tf,
        manifest.test_paths,
        manifest.test_labels,
        config,
        shuffle=False,
    )

    if config.augmentation_enabled:
        augmenter = _build_augmentation_model(tf, config)

        def _augment(images: Any, labels: Any) -> tuple[Any, Any]:
            return augmenter(images, training=True), labels

        train_ds = train_ds.map(_augment, num_parallel_calls=parallel_calls)

    if config.cache_train:
        train_ds = train_ds.cache()
    if config.cache_eval:
        val_ds = val_ds.cache()
        test_ds = test_ds.cache()

    train_ds = train_ds.with_options(_dataset_options(tf, deterministic=False))
    val_ds = val_ds.with_options(_dataset_options(tf, deterministic=True))
    test_ds = test_ds.with_options(_dataset_options(tf, deterministic=True))

    if config.prefetch:
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return DatasetBundle(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        class_names=manifest.class_names,
        class_to_index=manifest.class_to_index,
        train_images=train_images,
        val_images=val_images,
        test_images=test_images,
        train_file_paths=manifest.train_paths,
        val_file_paths=manifest.val_paths,
        test_file_paths=manifest.test_paths,
        train_class_counts=manifest.class_counts(manifest.train_labels),
        val_class_counts=manifest.class_counts(manifest.val_labels),
        test_class_counts=manifest.class_counts(manifest.test_labels),
    )


def inspect_dataset_batch(dataset: Any) -> dict[str, Any]:
    """Return shape and normalization diagnostics from one dataset batch."""

    tf = _require_tensorflow()
    images, labels = next(iter(dataset.take(1)))
    return {
        "image_shape": tuple(images.shape),
        "label_shape": tuple(labels.shape),
        "image_dtype": str(images.dtype),
        "label_dtype": str(labels.dtype),
        "image_min": float(tf.reduce_min(images).numpy()),
        "image_max": float(tf.reduce_max(images).numpy()),
    }


def inspect_label_range(dataset: Any) -> tuple[int, int]:
    """Return min/max label values from one dataset batch."""

    tf = _require_tensorflow()
    _, labels = next(iter(dataset.take(1)))
    min_label = int(tf.reduce_min(labels).numpy())
    max_label = int(tf.reduce_max(labels).numpy())
    return min_label, max_label
