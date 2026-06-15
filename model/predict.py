# =============================================================================
#  model/predict.py — Single-image and batch prediction utilities
#
#  Used by:
#    • The Flask web app (app/app.py)
#    • Direct CLI usage:  python model/predict.py --image path/to/leaf.jpg
# =============================================================================

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import MODEL_DIR, MODEL_FILENAME, RESULTS_DIR, TOP_K
from utils.preprocess    import preprocess_image_path, preprocess_image_bytes, preprocess_batch
from utils.dataset_loader import load_class_map


# ─────────────────────────────────────────────────────────────────────────────
#  Model singleton — loaded once and cached
# ─────────────────────────────────────────────────────────────────────────────

_model      = None
_class_map  = None   # {str_index: class_name}


def get_model() -> tf.keras.Model:
    global _model
    if _model is None:
        model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Run 'python model/train.py' to train the model first."
            )
        _model = tf.keras.models.load_model(model_path)
    return _model


def get_class_map() -> dict:
    global _class_map
    if _class_map is None:
        _class_map = load_class_map()
    return _class_map


# ─────────────────────────────────────────────────────────────────────────────
#  Core inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_top_k_result(probs: np.ndarray, class_map: dict, k: int) -> list:
    """
    Given a 1-D probability array, return a list of top-k dicts:
        [{"rank": 1, "class_index": 5, "class_name": "...", "confidence": 92.3}, …]
    """
    top_indices = np.argsort(probs)[::-1][:k]
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        results.append({
            "rank":        rank,
            "class_index": int(idx),
            "class_name":  class_map[str(idx)],
            "confidence":  round(float(probs[idx]) * 100, 2),
        })
    return results


def _format_name(raw: str) -> str:
    """
    Convert 'Tomato___Early_blight' or 'Pepper__bell___Bacterial_spot'
    into a human-readable 'Tomato: Early Blight'.
    """
    cleaned = raw.replace("___", "|").replace("__", "|").replace("_", " ")
    parts   = [p.strip().title() for p in cleaned.split("|") if p.strip()]
    if len(parts) >= 2:
        return f"{parts[0]}: {' '.join(parts[1:])}"
    return cleaned.title()


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def predict_single_path(image_path: str, top_k: int = TOP_K) -> dict:
    """
    Predict disease from an image file path.

    Returns:
        {
          "image_path": "...",
          "predictions": [{"rank":1, "class_name":"...", "confidence":92.3}, …]
        }
    """
    model     = get_model()
    class_map = get_class_map()

    tensor    = preprocess_image_path(image_path)          # (1, H, W, C)
    probs     = model.predict(tensor, verbose=0)[0]        # (C,)

    top_preds = _build_top_k_result(probs, class_map, top_k)

    # Add human-readable names
    for p in top_preds:
        p["display_name"] = _format_name(p["class_name"])

    return {
        "image_path":  image_path,
        "predictions": top_preds,
    }


def predict_single_bytes(raw_bytes: bytes, filename: str = "upload",
                         top_k: int = TOP_K) -> dict:
    """
    Predict disease from raw image bytes (used by Flask).

    Returns same structure as predict_single_path.
    """
    model     = get_model()
    class_map = get_class_map()

    tensor    = preprocess_image_bytes(raw_bytes)
    probs     = model.predict(tensor, verbose=0)[0]

    top_preds = _build_top_k_result(probs, class_map, top_k)

    for p in top_preds:
        p["display_name"] = _format_name(p["class_name"])

    return {
        "image_path":  filename,
        "predictions": top_preds,
    }


def predict_batch(image_paths: list, top_k: int = TOP_K,
                  save_results: bool = True) -> list:
    """
    Predict diseases for a list of image paths.

    Args:
        image_paths  : list of file path strings
        top_k        : number of top predictions per image
        save_results : if True, write JSON to results/batch_predictions.json

    Returns:
        list of prediction dicts (same structure as predict_single_path)
    """
    if not image_paths:
        return []

    model     = get_model()
    class_map = get_class_map()

    batch     = preprocess_batch(image_paths)              # (N, H, W, C)
    all_probs = model.predict(batch, verbose=1)            # (N, C)

    results = []
    for path, probs in zip(image_paths, all_probs):
        top_preds = _build_top_k_result(probs, class_map, top_k)
        for p in top_preds:
            p["display_name"] = _format_name(p["class_name"])
        results.append({"image_path": path, "predictions": top_preds})

    if save_results:
        out_path = os.path.join(RESULTS_DIR, "batch_predictions.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n   Batch results saved → {out_path}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plant Disease Prediction")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, help="Path to a single image")
    group.add_argument("--batch",  nargs="+", help="Paths to multiple images")
    parser.add_argument("--top_k", type=int, default=TOP_K,
                        help=f"Number of top predictions (default: {TOP_K})")
    args = parser.parse_args()

    if args.image:
        print(f"\n🌿 Predicting: {args.image}\n")
        result = predict_single_path(args.image, top_k=args.top_k)
        for p in result["predictions"]:
            print(f"  #{p['rank']}  {p['display_name']:<45}  {p['confidence']:>6.2f}%")

    elif args.batch:
        print(f"\n🌿 Batch prediction for {len(args.batch)} images …\n")
        results = predict_batch(args.batch, top_k=args.top_k, save_results=True)
        for r in results:
            top = r["predictions"][0]
            print(f"  {os.path.basename(r['image_path']):<30}  "
                  f"→ {top['display_name']:<40}  {top['confidence']:>6.2f}%")


if __name__ == "__main__":
    main()
