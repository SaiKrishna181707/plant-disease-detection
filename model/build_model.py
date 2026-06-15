# =============================================================================
#  model/build_model.py — Custom CNN architecture built entirely from scratch
#
#  ╔══════════════════════════════════════════════════════════════════════╗
#  ║  NO pretrained weights. NO transfer learning. NO imported backbone. ║
#  ║  Every layer is defined and trained from random initialisation.     ║
#  ╚══════════════════════════════════════════════════════════════════════╝
#
#  Architecture summary (128×128 input):
#
#  Input (128×128×3)
#   │
#   ├─ Block 1 → Conv(32)  → BN → ReLU → Conv(32)  → BN → ReLU → MaxPool → Drop(0.25)
#   ├─ Block 2 → Conv(64)  → BN → ReLU → Conv(64)  → BN → ReLU → MaxPool → Drop(0.25)
#   ├─ Block 3 → Conv(128) → BN → ReLU → Conv(128) → BN → ReLU → MaxPool → Drop(0.30)
#   ├─ Block 4 → Conv(256) → BN → ReLU → Conv(256) → BN → ReLU → MaxPool → Drop(0.30)
#   ├─ Block 5 → Conv(512) → BN → ReLU                           → MaxPool → Drop(0.35)
#   │
#   ├─ GlobalAveragePooling2D
#   │
#   ├─ Dense(1024) → BN → ReLU → Dropout(0.50)
#   ├─ Dense(512)  → BN → ReLU → Dropout(0.40)
#   │
#   └─ Dense(N_CLASSES, activation='softmax')
# =============================================================================

import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import INPUT_SHAPE, LEARNING_RATE


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: Conv block (Conv → BN → ReLU)
# ─────────────────────────────────────────────────────────────────────────────

def conv_bn_relu(filters: int, kernel_size: int = 3,
                 strides: int = 1, padding: str = "same",
                 l2_reg: float = 1e-4):
    """
    Returns a Sequential sub-block: Conv2D → BatchNormalization → ReLU.
    Using l2 weight regularisation to reduce overfitting.
    """
    return tf.keras.Sequential([
        layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,             # BN has its own bias term
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(l2_reg),
        ),
        layers.BatchNormalization(),
        layers.Activation("relu"),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  Main builder
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int,
                input_shape: tuple = INPUT_SHAPE,
                learning_rate: float = LEARNING_RATE) -> tf.keras.Model:
    """
    Build and compile a custom CNN from scratch.

    Args:
        num_classes   : Number of output classes (auto-detected from dataset).
        input_shape   : (H, W, C) tuple, from config.
        learning_rate : Initial learning rate.

    Returns:
        Compiled tf.keras.Model.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    # ── Block 1 — 32 filters ─────────────────────────────────────────────────
    x = conv_bn_relu(32)(inputs)
    x = conv_bn_relu(32)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)
    # spatial: 64×64

    # ── Block 2 — 64 filters ─────────────────────────────────────────────────
    x = conv_bn_relu(64)(x)
    x = conv_bn_relu(64)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)
    # spatial: 32×32

    # ── Block 3 — 128 filters ────────────────────────────────────────────────
    x = conv_bn_relu(128)(x)
    x = conv_bn_relu(128)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.Dropout(0.30, name="drop3")(x)
    # spatial: 16×16

    # ── Block 4 — 256 filters ────────────────────────────────────────────────
    x = conv_bn_relu(256)(x)
    x = conv_bn_relu(256)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)
    x = layers.Dropout(0.30, name="drop4")(x)
    # spatial: 8×8

    # ── Block 5 — 512 filters ────────────────────────────────────────────────
    x = conv_bn_relu(512)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool5")(x)
    x = layers.Dropout(0.35, name="drop5")(x)
    # spatial: 4×4

    # ── Global Average Pooling (replaces large Flatten) ──────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # ── Dense head ───────────────────────────────────────────────────────────
    x = layers.Dense(
        1024, use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_1024",
    )(x)
    x = layers.BatchNormalization(name="bn_dense1")(x)
    x = layers.Activation("relu", name="relu_dense1")(x)
    x = layers.Dropout(0.50, name="drop_dense1")(x)

    x = layers.Dense(
        512, use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_512",
    )(x)
    x = layers.BatchNormalization(name="bn_dense2")(x)
    x = layers.Activation("relu", name="relu_dense2")(x)
    x = layers.Dropout(0.40, name="drop_dense2")(x)

    # ── Output ───────────────────────────────────────────────────────────────
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_initializer="glorot_uniform",
        name="predictions",
    )(x)

    # ── Build & compile ──────────────────────────────────────────────────────
    model = tf.keras.Model(inputs, outputs, name="PlantDiseaseCNN_Scratch")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        ),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def print_model_summary(model: tf.keras.Model):
    """Print parameter count breakdown."""
    model.summary(line_length=90)
    total   = model.count_params()
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"\n   Total params      : {total:,}")
    print(f"   Trainable params  : {trainable:,}")
    print(f"   Non-trainable     : {total - trainable:,}\n")
