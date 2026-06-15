# =============================================================================
#  model/evaluate.py — Comprehensive model evaluation
#
#  Run:  python model/evaluate.py
#
#  Produces:
#    • Overall accuracy, precision, recall, F1
#    • Top-3 accuracy
#    • Per-class classification report (sklearn)
#    • Confusion matrix PNG
#    • Per-class accuracy bar chart
#    • Saves all metrics to results/evaluation_report.json
# =============================================================================

import os
import sys
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    top_k_accuracy_score,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import MODEL_DIR, MODEL_FILENAME, RESULTS_DIR, TOP_K
from utils.dataset_loader import load_all, load_class_map
from utils.visualization  import (
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_from_saved_history,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Collect predictions
# ─────────────────────────────────────────────────────────────────────────────

def get_predictions(model: tf.keras.Model,
                    dataset: tf.data.Dataset):
    """
    Run the model over an entire dataset and return:
        y_true       : (N,) integer labels
        y_pred       : (N,) integer predicted labels
        y_pred_probs : (N, C) softmax probabilities
    """
    y_true_list, y_prob_list = [], []

    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        y_prob_list.append(probs)
        y_true_list.append(labels.numpy())

    y_prob  = np.concatenate(y_prob_list, axis=0)
    y_true  = np.concatenate(y_true_list, axis=0)
    y_true  = np.argmax(y_true, axis=1)
    y_pred  = np.argmax(y_prob, axis=1)

    return y_true, y_pred, y_prob


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("   🌿 Plant Disease Detection — Evaluation")
    print("=" * 65)

    # ── Load model ──
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}.\n"
            "Run 'python model/train.py' first."
        )

    print(f"\n📦 Loading model from {model_path} …")
    model = tf.keras.models.load_model(model_path)

    # ── Load dataset (we only need the test split) ──
    print("\n📂 Loading test split …")
    _, _, test_ds, class_names, _ = load_all()
    n_classes = len(class_names)

    # ── Predict ──
    print("\n🔍 Running predictions on test set …")
    y_true, y_pred, y_prob = get_predictions(model, test_ds)

    # ── Core metrics ──
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(   y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(       y_true, y_pred, average="weighted", zero_division=0)
    top3 = top_k_accuracy_score(y_true, y_prob, k=min(TOP_K, n_classes))

    print("\n" + "─" * 55)
    print(f"   Overall Accuracy     : {acc  * 100:.2f}%")
    print(f"   Top-{TOP_K} Accuracy      : {top3 * 100:.2f}%")
    print(f"   Weighted Precision   : {prec * 100:.2f}%")
    print(f"   Weighted Recall      : {rec  * 100:.2f}%")
    print(f"   Weighted F1 Score    : {f1   * 100:.2f}%")
    print("─" * 55)

    # ── Classification report ──
    print("\n📋 Per-Class Classification Report:\n")
    report_str = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )
    print(report_str)

    # ── Save JSON report ──
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    summary = {
        "overall_accuracy":    round(acc  * 100, 2),
        "top3_accuracy":       round(top3 * 100, 2),
        "weighted_precision":  round(prec * 100, 2),
        "weighted_recall":     round(rec  * 100, 2),
        "weighted_f1":         round(f1   * 100, 2),
        "n_classes":           n_classes,
        "n_test_samples":      len(y_true),
        "per_class":           report_dict,
    }
    report_path = os.path.join(RESULTS_DIR, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   Report saved → {report_path}")

    # ── Plots ──
    print("\n📊 Generating evaluation plots …")
    plot_confusion_matrix(y_true, y_pred, class_names)
    plot_per_class_accuracy(y_true, y_pred, class_names)

    # Regenerate training curves if history exists
    from config import HISTORY_FILE
    if os.path.exists(HISTORY_FILE):
        plot_from_saved_history(HISTORY_FILE)

    print("\n✅ Evaluation complete!")
    print(f"   All plots saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
