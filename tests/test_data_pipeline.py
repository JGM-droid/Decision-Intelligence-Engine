from __future__ import annotations

from pathlib import Path
from collections import Counter

import pytest

from src.decision_intelligence_engine.data_pipeline import (
    build_cifar10_datasets,
    build_cifar10_split_manifest,
    inspect_dataset_batch,
)
from src.decision_intelligence_engine.pipeline_config import DataPipelineConfig


def _write_png(path: Path, tf, value: int) -> None:
    image = tf.ones((32, 32, 3), dtype=tf.uint8) * tf.cast(value % 255, tf.uint8)
    encoded = tf.io.encode_png(image).numpy()
    path.write_bytes(encoded)


@pytest.fixture
def tiny_cifar_fixture(tmp_path: Path):
    tf = pytest.importorskip("tensorflow")

    root = tmp_path / "data" / "raw" / "cifar10"
    train = root / "train"
    test = root / "test"
    class_names = ["airplane", "automobile"]

    idx = 1
    for split_root, per_class in ((train, 12), (test, 4)):
        for cls in class_names:
            cls_dir = split_root / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            for _ in range(per_class):
                _write_png(cls_dir / f"{idx}.png", tf, value=idx)
                idx += 1

    config = DataPipelineConfig(
        data_root="data/raw/cifar10",
        image_size=(32, 32),
        batch_size=4,
        validation_split=0.25,
        random_seed=123,
        shuffle_buffer_size=32,
        num_parallel_calls=1,
        cache_train=False,
        cache_eval=False,
        prefetch=False,
        augmentation_enabled=True,
        augmentation_padding=4,
        augmentation_horizontal_flip=True,
    )

    return tmp_path, config


def _first_batch_arrays(dataset):
    images, labels = next(iter(dataset.take(1)))
    return images.numpy(), labels.numpy()


def test_missing_dataset_path_behavior(tmp_path: Path) -> None:
    pytest.importorskip("tensorflow")
    cfg = DataPipelineConfig(data_root="does/not/exist")
    with pytest.raises(FileNotFoundError):
        build_cifar10_datasets(tmp_path, cfg)


def test_expected_class_mapping_and_counts(tiny_cifar_fixture) -> None:
    project_root, cfg = tiny_cifar_fixture
    bundle = build_cifar10_datasets(project_root, cfg)

    assert bundle.class_names == ("airplane", "automobile")
    assert bundle.class_to_index == {"airplane": 0, "automobile": 1}
    assert bundle.train_images + bundle.val_images == 24
    assert bundle.test_images == 8


def test_batch_shapes_and_normalization_range(tiny_cifar_fixture) -> None:
    project_root, cfg = tiny_cifar_fixture
    bundle = build_cifar10_datasets(project_root, cfg)

    train_info = inspect_dataset_batch(bundle.train_ds)
    val_info = inspect_dataset_batch(bundle.val_ds)
    test_info = inspect_dataset_batch(bundle.test_ds)

    assert train_info["image_shape"] == (4, 32, 32, 3)
    assert val_info["image_shape"] == (4, 32, 32, 3)
    assert test_info["image_shape"] == (4, 32, 32, 3)
    assert train_info["label_shape"] == (4,)
    assert val_info["label_shape"] == (4,)
    assert test_info["label_shape"] == (4,)

    assert 0.0 <= val_info["image_min"] <= 1.0
    assert 0.0 <= val_info["image_max"] <= 1.0
    assert 0.0 <= test_info["image_min"] <= 1.0
    assert 0.0 <= test_info["image_max"] <= 1.0

    observed_labels = set()
    for _, labels in bundle.train_ds.take(3):
        observed_labels.update(labels.numpy().tolist())
    assert observed_labels.issubset({0, 1})
    assert observed_labels


def test_reproducible_validation_splitting(tiny_cifar_fixture) -> None:
    project_root, cfg = tiny_cifar_fixture

    bundle_a = build_cifar10_datasets(project_root, cfg)
    bundle_b = build_cifar10_datasets(project_root, cfg)

    images_a, labels_a = _first_batch_arrays(bundle_a.val_ds)
    images_b, labels_b = _first_batch_arrays(bundle_b.val_ds)

    assert (images_a == images_b).all()
    assert (labels_a == labels_b).all()


def test_augmentation_training_only(tiny_cifar_fixture) -> None:
    project_root, cfg_aug = tiny_cifar_fixture
    cfg_no_aug = DataPipelineConfig(
        **{**cfg_aug.__dict__, "augmentation_enabled": False}
    )

    bundle_aug = build_cifar10_datasets(project_root, cfg_aug)
    bundle_no_aug = build_cifar10_datasets(project_root, cfg_no_aug)

    train_aug_img, _ = _first_batch_arrays(bundle_aug.train_ds)
    train_no_aug_img, _ = _first_batch_arrays(bundle_no_aug.train_ds)
    val_aug_img, _ = _first_batch_arrays(bundle_aug.val_ds)
    val_no_aug_img, _ = _first_batch_arrays(bundle_no_aug.val_ds)
    test_aug_img, _ = _first_batch_arrays(bundle_aug.test_ds)
    test_no_aug_img, _ = _first_batch_arrays(bundle_no_aug.test_ds)

    assert (val_aug_img == val_no_aug_img).all()
    assert (test_aug_img == test_no_aug_img).all()
    assert not (train_aug_img == train_no_aug_img).all()


def test_split_manifest_balanced_no_overlap_reproducible(tiny_cifar_fixture) -> None:
    project_root, cfg = tiny_cifar_fixture

    manifest_a = build_cifar10_split_manifest(project_root, cfg)
    manifest_b = build_cifar10_split_manifest(project_root, cfg)

    assert manifest_a.train_paths == manifest_b.train_paths
    assert manifest_a.val_paths == manifest_b.val_paths
    assert manifest_a.test_paths == manifest_b.test_paths

    train_set = set(manifest_a.train_paths)
    val_set = set(manifest_a.val_paths)
    assert not (train_set & val_set)

    assert len(manifest_a.train_paths) == 18
    assert len(manifest_a.val_paths) == 6
    assert len(manifest_a.test_paths) == 8

    tr_counts = Counter(manifest_a.train_labels)
    va_counts = Counter(manifest_a.val_labels)
    te_counts = Counter(manifest_a.test_labels)
    assert tr_counts == {0: 9, 1: 9}
    assert va_counts == {0: 3, 1: 3}
    assert te_counts == {0: 4, 1: 4}


def test_real_cifar_split_integrity() -> None:
    """Integration check on canonical CIFAR-10 split when dataset is available."""

    project_root = Path.cwd()
    train_root = project_root / "data" / "raw" / "cifar10" / "train"
    test_root = project_root / "data" / "raw" / "cifar10" / "test"
    if not train_root.exists() or not test_root.exists():
        pytest.skip("Canonical CIFAR-10 dataset is not present.")

    cfg = DataPipelineConfig(
        data_root="data/raw/cifar10",
        image_size=(32, 32),
        batch_size=128,
        validation_split=0.2,
        random_seed=42,
        augmentation_enabled=True,
    )

    manifest = build_cifar10_split_manifest(project_root, cfg)
    assert len(manifest.class_names) == 10

    train_set = set(manifest.train_paths)
    val_set = set(manifest.val_paths)
    assert not (train_set & val_set)

    tr_counts = Counter(manifest.train_labels)
    va_counts = Counter(manifest.val_labels)
    te_counts = Counter(manifest.test_labels)

    assert len(manifest.train_paths) == 40000
    assert len(manifest.val_paths) == 10000
    assert len(manifest.test_paths) == 10000

    # Expected class-balanced allocation from CIFAR-10 canonical counts.
    assert all(tr_counts[i] == 4000 for i in range(10))
    assert all(va_counts[i] == 1000 for i in range(10))
    assert all(te_counts[i] == 1000 for i in range(10))

    # Reproducibility for same seed.
    manifest_repeat = build_cifar10_split_manifest(project_root, cfg)
    assert manifest_repeat.train_paths == manifest.train_paths
    assert manifest_repeat.val_paths == manifest.val_paths
