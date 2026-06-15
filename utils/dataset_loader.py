# =============================================================================
#  utils/dataset_loader.py — Automatic dataset discovery and splitting
#
#  Scans DATASET_DIR for sub-folders (one per class), builds file lists,
#  stratifies them into train / val / test splits, and returns tf.data
#  pipelines ready for training.
# =============================================================================

import os
import sys
import json
import random
import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATASET_DIR, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, INPUT_SHAPE,
    TRAIN_RATIO, VAL_RATIO, RANDOM_SEED, BATCH_SIZE, CLASS_MAP_FILE,
    AUG_ROTATION, AUG_WIDTH_SHIFT, AUG_HEIGHT_SHIFT, AUG_ZOOM,
    AUG_HORIZONTAL_FLIP, AUG_BRIGHTNESS, AUG_FILL_MODE
)


# ─────────────────────────────────────────────────────────────────────────────
#  1.  Discover classes and file paths
# ─────────────────────────────────────────────────────────────────────────────

def discover_dataset(dataset_dir: str = DATASET_DIR):
    """
    Walk DATASET_DIR. Every immediate sub-folder is treated as a class.
    Returns:
        class_names  : sorted list of class name strings
        class_to_idx : dict  {class_name: integer_index}
        file_paths   : list of (abs_path, label_index) tuples
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            "Place your PlantVillage images inside the 'dataset/' folder "
            "with one sub-folder per class."
        )

    class_names = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    if not class_names:
        raise ValueError(
            f"No class sub-folders found in {dataset_dir}. "
            "Expected structure: dataset/<ClassName>/<images>"
        )

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    VALID_EXT    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    file_paths   = []

    for cls in class_names:
        cls_dir = os.path.join(dataset_dir, cls)
        for fname in os.listdir(cls_dir):
            if os.path.splitext(fname)[1].lower() in VALID_EXT:
                file_paths.append((os.path.join(cls_dir, fname), class_to_idx[cls]))

    print(f"\n📂 Dataset: {dataset_dir}")
    print(f"   Classes  : {len(class_names)}")
    print(f"   Images   : {len(file_paths)}")
    for cls in class_names:
        count = sum(1 for _, l in file_paths if l == class_to_idx[cls])
        print(f"      [{class_to_idx[cls]:>2}] {cls:<45} {count:>5} images")

    return class_names, class_to_idx, file_paths


def save_class_map(class_names: list):
    """Persist class-index mapping to JSON for inference-time use."""
    mapping = {str(i): name for i, name in enumerate(class_names)}
    with open(CLASS_MAP_FILE, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"   Class map saved → {CLASS_MAP_FILE}")


def load_class_map() -> dict:
    """Load class map from JSON. Returns {str_index: class_name}."""
    if not os.path.exists(CLASS_MAP_FILE):
        raise FileNotFoundError(
            f"Class map not found at {CLASS_MAP_FILE}. "
            "Train the model first."
        )
    with open(CLASS_MAP_FILE, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
#  2.  Stratified split
# ─────────────────────────────────────────────────────────────────────────────

def stratified_split(file_paths: list, class_names: list):
    """
    Stratified train / val / test split.
    Returns three lists of (path, label) tuples.
    """
    random.seed(RANDOM_SEED)
    n_classes   = len(class_names)

    train_files, val_files, test_files = [], [], []

    for cls_idx in range(n_classes):
        cls_files = [fp for fp in file_paths if fp[1] == cls_idx]
        random.shuffle(cls_files)

        n        = len(cls_files)
        n_train  = max(1, int(n * TRAIN_RATIO))
        n_val    = max(1, int(n * VAL_RATIO))

        train_files.extend(cls_files[:n_train])
        val_files.extend(cls_files[n_train:n_train + n_val])
        test_files.extend(cls_files[n_train + n_val:])

    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)

    print(f"\n   Split → Train: {len(train_files)} | "
          f"Val: {len(val_files)} | Test: {len(test_files)}")

    return train_files, val_files, test_files


# ─────────────────────────────────────────────────────────────────────────────
#  3.  tf.data pipeline builders
# ─────────────────────────────────────────────────────────────────────────────

def _parse_image(path: str, label: int, n_classes: int):
    """Read → decode → resize → normalise → one-hot encode."""
    raw   = tf.io.read_file(path)
    image = tf.image.decode_image(raw, channels=IMG_CHANNELS, expand_animations=False)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0
    image.set_shape(INPUT_SHAPE)
    label = tf.one_hot(label, n_classes)
    return image, label


def _augment(image, label):
    """
    Apply random augmentations during training only.
    All ops are TF-native so they run on GPU when available.
    """
    # Random horizontal flip
    if AUG_HORIZONTAL_FLIP:
        image = tf.image.random_flip_left_right(image)

    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.25)
    image = tf.image.random_contrast(image, lower=0.80, upper=1.20)
    image = tf.image.random_saturation(image, lower=0.80, upper=1.20)

    # Random crop then resize back (simulates zoom)
    scale  = tf.random.uniform([], 0.82, 1.0)
    crop_h = tf.cast(tf.cast(IMG_HEIGHT, tf.float32) * scale, tf.int32)
    crop_w = tf.cast(tf.cast(IMG_WIDTH,  tf.float32) * scale, tf.int32)
    image  = tf.image.random_crop(image, [crop_h, crop_w, IMG_CHANNELS])
    image  = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])

    # Random rotation via tf.raw_ops (±25°)
    angle  = tf.random.uniform([], -AUG_ROTATION * 3.14159 / 180,
                                     AUG_ROTATION * 3.14159 / 180)
    image  = _rotate(image, angle)

    image  = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def _rotate(image, angle):
    """Rotate a single image tensor by angle (radians)."""
    import math
    # Use tfa if available, else approximate with tf ops
    try:
        import tensorflow_addons as tfa
        return tfa.image.rotate(image, angle, interpolation="BILINEAR")
    except ImportError:
        # Fallback: random horizontal + vertical shift as a lightweight substitute
        image = tf.image.random_flip_up_down(image)
        return image


def build_pipeline(file_list: list, n_classes: int,
                   training: bool = False, cache: bool = True) -> tf.data.Dataset:
    """
    Convert a list of (path, label) tuples into a batched, prefetched
    tf.data.Dataset.
    """
    paths  = [fp[0] for fp in file_list]
    labels = [fp[1] for fp in file_list]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(buffer_size=min(len(file_list), 5000), seed=RANDOM_SEED)

    ds = ds.map(
        lambda p, l: _parse_image(p, l, n_classes),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if cache:
        ds = ds.cache()

    if training:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(1024, seed=RANDOM_SEED)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ─────────────────────────────────────────────────────────────────────────────
#  4.  Public convenience function
# ─────────────────────────────────────────────────────────────────────────────

def load_all(dataset_dir: str = DATASET_DIR):
    """
    Full pipeline entry point.
    Returns:
        train_ds, val_ds, test_ds  : tf.data.Dataset objects
        class_names                : list of class name strings
        class_to_idx               : dict
    """
    class_names, class_to_idx, file_paths = discover_dataset(dataset_dir)
    save_class_map(class_names)

    train_files, val_files, test_files = stratified_split(file_paths, class_names)

    n = len(class_names)
    train_ds = build_pipeline(train_files, n, training=True)
    val_ds   = build_pipeline(val_files,   n, training=False)
    test_ds  = build_pipeline(test_files,  n, training=False)

    return train_ds, val_ds, test_ds, class_names, class_to_idx
