from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.decision_intelligence_engine.pipeline_config import load_data_pipeline_config
from src.decision_intelligence_engine.data_pipeline import build_cifar10_split_manifest


def count_from_paths(paths: list[str], class_names: tuple[str, ...]) -> dict[str, int]:
    d = Counter()
    for p in paths:
        d[Path(p).parent.name] += 1
    return {n: int(d[n]) for n in class_names}


def main() -> None:
    root = PROJECT_ROOT
    cfg = load_data_pipeline_config(root)

    train_dir = cfg.train_dir(root)
    test_dir = cfg.test_dir(root)
    common = dict(
        labels="inferred",
        label_mode="int",
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        seed=cfg.random_seed,
    )

    old_train = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=cfg.validation_split,
        subset="training",
        shuffle=True,
        **common,
    )
    old_val = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=cfg.validation_split,
        subset="validation",
        shuffle=False,
        **common,
    )
    old_test = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        shuffle=False,
        **common,
    )

    class_names = tuple(old_train.class_names)
    old_counts = {
        "train": count_from_paths(old_train.file_paths, class_names),
        "val": count_from_paths(old_val.file_paths, class_names),
        "test": count_from_paths(old_test.file_paths, class_names),
    }

    manifest = build_cifar10_split_manifest(root, cfg)
    new_counts = {
        "train": manifest.class_counts(manifest.train_labels),
        "val": manifest.class_counts(manifest.val_labels),
        "test": manifest.class_counts(manifest.test_labels),
    }

    print("CLASS,OLD_TRAIN,OLD_VAL,OLD_TEST,NEW_TRAIN,NEW_VAL,NEW_TEST")
    for name in class_names:
        print(
            f"{name},{old_counts['train'][name]},{old_counts['val'][name]},{old_counts['test'][name]},"
            f"{new_counts['train'][name]},{new_counts['val'][name]},{new_counts['test'][name]}"
        )


if __name__ == "__main__":
    main()
