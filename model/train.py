# =============================================================================
#  model/train.py — Training entry point
#
#  Run:  python model/train.py
#
#  What it does:
#    1. Auto-discovers all classes from the dataset/ folder
#    2. Builds the custom CNN from scratch
#    3. Trains with early stopping, LR scheduling, checkpointing
#    4. Logs to TensorBoard
#    5. Saves model + history + training plots to models/ and results/
# =============================================================================

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import (
    MODEL_DIR, RESULTS_DIR, LOG_DIR, MODEL_FILENAME, HISTORY_FILE,
    EPOCHS, BATCH_SIZE, LEARNING_RATE,
    LR_DECAY_FACTOR, LR_PATIENCE, MIN_LR, ES_PATIENCE
)
from utils.dataset_loader import load_all
from utils.visualization  import plot_training_history, plot_sample_grid
from model.build_model    import build_model, print_model_summary


# ─────────────────────────────────────────────────────────────────────────────
#  GPU setup
# ─────────────────────────────────────────────────────────────────────────────

def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"   GPU(s) found: {[g.name for g in gpus]}")
    else:
        print("   No GPU found — training on CPU.")


# ─────────────────────────────────────────────────────────────────────────────
#  Callbacks
# ─────────────────────────────────────────────────────────────────────────────

def build_callbacks(model_path: str) -> list:
    return [
        # Save the best model by val_accuracy
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        # Stop training when val_accuracy stops improving
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=ES_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce LR when val_loss plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_DECAY_FACTOR,
            patience=LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=1,
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=LOG_DIR,
            histogram_freq=1,
            update_freq="epoch",
        ),
        # CSV log for every epoch
        tf.keras.callbacks.CSVLogger(
            os.path.join(RESULTS_DIR, "epoch_log.csv"),
            append=False,
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  Class weighting (handles imbalanced datasets)
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(train_ds: tf.data.Dataset, n_classes: int) -> dict:
    """
    Count samples per class across the training dataset and compute
    balanced class weights using the sklearn formula.
    """
    counts = np.zeros(n_classes, dtype=np.int64)

    for _, labels in train_ds.unbatch():
        idx = np.argmax(labels.numpy())
        counts[idx] += 1

    total = counts.sum()
    weights = {}
    for i in range(n_classes):
        if counts[i] > 0:
            weights[i] = (total / (n_classes * counts[i]))
        else:
            weights[i] = 1.0

    print(f"\n   Class weights computed (imbalance correction applied).")
    return weights


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("   🌿 Plant Disease Detection — Training from Scratch")
    print("=" * 65)

    # ── GPU ──
    configure_gpu()

    # ── Load dataset ──
    print("\n[1/5] Loading and splitting dataset …")
    train_ds, val_ds, test_ds, class_names, class_to_idx = load_all()
    n_classes = len(class_names)

    # Visualise sample images
    print("\n[2/5] Generating sample grid …")
    plot_sample_grid(train_ds, class_names)

    # ── Build model ──
    print("\n[3/5] Building custom CNN architecture …")
    model = build_model(n_classes)
    print_model_summary(model)

    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    callbacks  = build_callbacks(model_path)

    # ── Class weights ──
    class_weights = compute_class_weights(train_ds, n_classes)

    # ── Train ──
    print(f"\n[4/5] Training for up to {EPOCHS} epochs …")
    print(f"   Batch size   : {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Early stop   : patience={ES_PATIENCE}")
    print(f"   LR reduce    : factor={LR_DECAY_FACTOR}, patience={LR_PATIENCE}")
    print(f"   Model save   : {model_path}\n")

    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"\n   Training completed in {elapsed/60:.1f} min")

    # ── Save history ──
    print("\n[5/5] Saving results …")
    history_dict = {k: [float(v) for v in vals]
                    for k, vals in history.history.items()}
    with open(HISTORY_FILE, "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"   History saved → {HISTORY_FILE}")

    # ── Plot training curves ──
    plot_training_history(history_dict)

    # ── Final model evaluation on val set ──
    print("\n📊 Final evaluation on validation set:")
    val_results = model.evaluate(val_ds, verbose=1)
    for name, val in zip(model.metrics_names, val_results):
        print(f"   {name:20s}: {val:.4f}")

    print(f"\n✅ Training complete!")
    print(f"   Model saved to : {model_path}")
    print(f"   Run evaluation : python model/evaluate.py")
    print(f"   Run web app    : python app/app.py")
    print(f"   TensorBoard    : tensorboard --logdir {LOG_DIR}")


if __name__ == "__main__":
    main()
