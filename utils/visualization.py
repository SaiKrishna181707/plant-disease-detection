# =============================================================================
#  utils/visualization.py — All plotting utilities
#
#  Generates and saves:
#    • Sample image grid
#    • Training / validation accuracy + loss curves
#    • Confusion matrix (raw + normalised)
#    • Per-class accuracy bar chart
# =============================================================================

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import RESULTS_DIR, HISTORY_FILE


# ─────────────────────────────────────────────────────────────────────────────
#  Training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(history_dict: dict, save: bool = True) -> str:
    """
    Plot accuracy and loss curves from a Keras history dict.

    Args:
        history_dict: dict with keys 'accuracy', 'val_accuracy', 'loss', 'val_loss'
        save: whether to write the PNG to RESULTS_DIR

    Returns:
        path to the saved PNG (or empty string if save=False)
    """
    epochs  = range(1, len(history_dict["accuracy"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Training History", fontsize=16, fontweight="bold")

    # ── Accuracy ──
    ax1.plot(epochs, history_dict["accuracy"],     "b-o", markersize=3, label="Train Acc")
    ax1.plot(epochs, history_dict["val_accuracy"], "r-o", markersize=3, label="Val Acc")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.4)
    ax1.set_ylim([0, 1.05])

    best_val_acc = max(history_dict["val_accuracy"])
    best_epoch   = history_dict["val_accuracy"].index(best_val_acc) + 1
    ax1.axvline(best_epoch, color="green", linestyle="--", linewidth=1,
                label=f"Best val ({best_val_acc:.3f})")
    ax1.legend()

    # ── Loss ──
    ax2.plot(epochs, history_dict["loss"],     "b-o", markersize=3, label="Train Loss")
    ax2.plot(epochs, history_dict["val_loss"], "r-o", markersize=3, label="Val Loss")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Categorical Crossentropy")
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()

    path = ""
    if save:
        path = os.path.join(RESULTS_DIR, "training_history.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"   Training curves saved → {path}")
    plt.show()
    plt.close()
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: list, save: bool = True) -> str:
    """
    Plot raw-count and row-normalised confusion matrices side by side.
    """
    cm   = confusion_matrix(y_true, y_pred)
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    norm = np.nan_to_num(norm)

    n    = len(class_names)
    fs   = max(6, min(10, 200 // n))   # auto font size
    fig, axes = plt.subplots(1, 2, figsize=(max(20, n * 1.5), max(14, n * 1.1)))
    fig.suptitle("Confusion Matrix — Plant Disease CNN", fontsize=15, fontweight="bold")

    for ax, data, title, fmt in zip(
        axes,
        [cm,         norm],
        ["Raw counts", "Row-normalised (recall)"],
        ["d",          ".2f"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="YlOrRd",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, annot_kws={"size": fs}
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True",      fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fs)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=fs)

    plt.tight_layout()
    path = ""
    if save:
        path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        print(f"   Confusion matrix saved → {path}")
    plt.show()
    plt.close()
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  Per-class accuracy
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: list, save: bool = True) -> str:
    """Horizontal bar chart showing per-class recall/accuracy."""
    cm        = confusion_matrix(y_true, y_pred)
    per_class = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)

    colors = [
        "#2ecc71" if v >= 0.85 else
        "#f39c12" if v >= 0.70 else
        "#e74c3c"
        for v in per_class
    ]

    fig, ax = plt.subplots(figsize=(12, max(6, len(class_names) * 0.45)))
    bars = ax.barh(class_names, per_class * 100, color=colors)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 115)
    ax.axvline(80, color="navy", linestyle="--", linewidth=0.8, label="80% line")
    ax.axvline(90, color="green", linestyle="--", linewidth=0.8, label="90% line")
    ax.legend(fontsize=9)

    for bar, val in zip(bars, per_class):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    path = ""
    if save:
        path = os.path.join(RESULTS_DIR, "per_class_accuracy.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"   Per-class accuracy saved → {path}")
    plt.show()
    plt.close()
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  Sample image grid
# ─────────────────────────────────────────────────────────────────────────────

def plot_sample_grid(dataset, class_names: list, n: int = 16, save: bool = True) -> str:
    """Show a grid of sample images with class labels from a tf.data.Dataset."""
    import tensorflow as tf

    images, labels = next(iter(dataset.unbatch().batch(n)))
    images = images.numpy()
    labels = np.argmax(labels.numpy(), axis=1)

    cols = 4
    rows = max(1, n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    fig.suptitle("Sample Training Images", fontsize=15, fontweight="bold")

    for i, ax in enumerate(np.array(axes).flat):
        if i < len(images):
            ax.imshow(images[i])
            name = class_names[labels[i]].replace("_", "\n").replace("___", "\n")
            ax.set_title(name, fontsize=7)
        ax.axis("off")

    plt.tight_layout()
    path = ""
    if save:
        path = os.path.join(RESULTS_DIR, "sample_grid.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        print(f"   Sample grid saved → {path}")
    plt.show()
    plt.close()
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  Load history from JSON and re-plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_from_saved_history(history_path: str = HISTORY_FILE) -> str:
    """Reload a saved JSON history file and re-generate the training curves."""
    with open(history_path, "r") as f:
        history_dict = json.load(f)
    return plot_training_history(history_dict)
