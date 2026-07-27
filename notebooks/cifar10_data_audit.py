"""Phase 2A data audit for canonical CIFAR-10 folder layout.

Canonical raw dataset layout:
- data/raw/cifar10/train/<class>/*.png
- data/raw/cifar10/test/<class>/*.png

This script:
1. Downloads CIFAR-10 mirror archive into data/raw if missing.
2. Extracts dataset into canonical folder layout if missing.
3. Audits structure, counts, class balance, and image properties.
4. Exports two sample images per class to docs/assets/cifar10_samples.
5. Writes audit report to docs/cifar10_data_audit.md.
"""

from __future__ import annotations

import os
import shutil
import struct
import tarfile
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
ARCHIVE_PATH = DATA_RAW_DIR / "cifar-10-python.tar.gz"
IMAGEFOLDER_DIR = DATA_RAW_DIR / "cifar10"
TRAIN_DIR = IMAGEFOLDER_DIR / "train"
TEST_DIR = IMAGEFOLDER_DIR / "test"
AUDIT_REPORT_PATH = PROJECT_ROOT / "docs" / "cifar10_data_audit.md"
SAMPLES_DIR = PROJECT_ROOT / "docs" / "assets" / "cifar10_samples"

# Canonical source for this project environment.
CIFAR10_MIRROR_URL = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"


def ensure_dirs() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def download_cifar10() -> None:
    if ARCHIVE_PATH.exists():
        print(f"Archive already exists: {ARCHIVE_PATH}")
        return

    print(f"Downloading CIFAR-10 from {CIFAR10_MIRROR_URL}")
    with urllib.request.urlopen(CIFAR10_MIRROR_URL, timeout=180) as response:
        with ARCHIVE_PATH.open("wb") as out_file:
            shutil.copyfileobj(response, out_file)
    print(f"Downloaded to: {ARCHIVE_PATH}")


def extract_cifar10() -> None:
    if TRAIN_DIR.exists() and TEST_DIR.exists():
        print(f"Canonical extracted dataset already present: {IMAGEFOLDER_DIR}")
        return

    print(f"Extracting archive: {ARCHIVE_PATH}")
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(path=DATA_RAW_DIR)

    if not (TRAIN_DIR.exists() and TEST_DIR.exists()):
        raise FileNotFoundError(
            "Expected canonical folder layout not found after extraction: "
            "data/raw/cifar10/train and data/raw/cifar10/test"
        )

    print(f"Extracted to canonical layout: {IMAGEFOLDER_DIR}")


def count_images_in_dir(path: Path) -> int:
    return sum(
        1
        for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def png_dimensions_and_channels(path: Path) -> tuple[int, int, int]:
    with path.open("rb") as f:
        sig = f.read(8)
        if sig != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Not a PNG file: {path}")
        length = struct.unpack(">I", f.read(4))[0]
        chunk_type = f.read(4)
        if chunk_type != b"IHDR" or length != 13:
            raise ValueError(f"Missing IHDR in PNG: {path}")
        ihdr = f.read(13)
        width = struct.unpack(">I", ihdr[0:4])[0]
        height = struct.unpack(">I", ihdr[4:8])[0]
        color_type = ihdr[9]

        channel_map = {
            0: 1,  # grayscale
            2: 3,  # RGB
            3: 1,  # indexed
            4: 2,  # grayscale + alpha
            6: 4,  # RGBA
        }
        channels = channel_map.get(color_type, -1)
        return width, height, channels


def audit_dataset() -> tuple[list[str], int, int, list[tuple[str, int]], list[tuple[str, int]], dict[str, list[str]]]:
    label_names = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])

    train_dist: list[tuple[str, int]] = []
    test_dist: list[tuple[str, int]] = []
    for name in label_names:
        train_dist.append((name, count_images_in_dir(TRAIN_DIR / name)))
        test_dist.append((name, count_images_in_dir(TEST_DIR / name)))

    train_count = sum(c for _, c in train_dist)
    test_count = sum(c for _, c in test_dist)

    sample_paths: dict[str, list[str]] = {name: [] for name in label_names}
    for name in label_names:
        class_files = sorted(
            [p for p in (TRAIN_DIR / name).iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )
        for idx, src in enumerate(class_files[:2], start=1):
            dst = SAMPLES_DIR / f"{name}_sample_{idx}{src.suffix.lower()}"
            shutil.copy2(src, dst)
            sample_paths[name].append(dst.relative_to(PROJECT_ROOT).as_posix())

    first_class = label_names[0]
    first_image = sorted((TRAIN_DIR / first_class).glob("*.png"))[0]
    width, height, channels = png_dimensions_and_channels(first_image)
    if (width, height) != (32, 32):
        raise ValueError(f"Unexpected image dimensions: {(width, height)}")
    if channels not in (3, 4):
        raise ValueError(f"Unexpected channel count: {channels}")

    return label_names, train_count, test_count, train_dist, test_dist, sample_paths


def make_markdown_report(
    archive_size_bytes: int,
    label_names: list[str],
    train_count: int,
    test_count: int,
    train_dist: list[tuple[str, int]],
    test_dist: list[tuple[str, int]],
    sample_paths: dict[str, list[str]],
) -> str:
    total = train_count + test_count
    width, height, channels = 32, 32, 3

    train_table = "\n".join(f"| {name} | {count} |" for name, count in train_dist)
    test_table = "\n".join(f"| {name} | {count} |" for name, count in test_dist)
    label_mapping_rows = "\n".join(f"| {idx} | {name} |" for idx, name in enumerate(label_names))

    sample_block = "\n".join(
        f"- {class_name}: {', '.join(sample_paths.get(class_name, []))}"
        for class_name in label_names
    )

    return f"""# CIFAR-10 Dataset Audit

## Summary

- Source used in this environment: {CIFAR10_MIRROR_URL}
- Archive location: data/raw/cifar-10-python.tar.gz
- Extracted location: data/raw/cifar10
- Canonical dataset format: folder-based image layout
- Archive size (bytes): {archive_size_bytes}

## Dataset Structure

- Canonical raw layout:
    - train/<class>/*.png
    - test/<class>/*.png
- Label set ({len(label_names)} classes): {", ".join(label_names)}

## Counts and Splits

- Total images: {total}
- Training images: {train_count}
- Test images: {test_count}
- Split ratio: {train_count}:{test_count}

## Image Properties

- Width x Height: {width} x {height}
- Channels: {channels} (RGB)
- Flattened vector length per sample: {width * height * channels}

## Class Distribution (Training)

| Class | Count |
| --- | ---: |
{train_table}

## Class Distribution (Test)

| Class | Count |
| --- | ---: |
{test_table}

## Label Mapping

| Label ID | Label Name |
| ---: | --- |
{label_mapping_rows}

## Sample Visualizations

The script exported two sample images per class to docs/assets/cifar10_samples.

{sample_block}

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
"""


def main() -> None:
    ensure_dirs()
    download_cifar10()
    extract_cifar10()

    label_names, train_count, test_count, train_dist, test_dist, sample_paths = audit_dataset()

    archive_size = os.path.getsize(ARCHIVE_PATH)
    report = make_markdown_report(
        archive_size_bytes=archive_size,
        label_names=label_names,
        train_count=train_count,
        test_count=test_count,
        train_dist=train_dist,
        test_dist=test_dist,
        sample_paths=sample_paths,
    )

    AUDIT_REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote audit report: {AUDIT_REPORT_PATH}")
    print(f"Exported sample images to: {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
