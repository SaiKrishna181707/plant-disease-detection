# =============================================================================
#  app/app.py — Flask web application for Plant Disease Detection
#
#  Run:  python app/app.py
#  Open: http://localhost:5000
#
#  Routes:
#    GET  /              → index / upload page
#    POST /predict       → JSON inference endpoint
#    GET  /history       → upload history page
#    GET  /status        → system status dashboard
#    GET  /health        → health check
#    GET  /static/...    → static assets
# =============================================================================

import os
import sys
import json
import uuid
import base64
import datetime

import tensorflow as tf
from flask import (
    Flask, request, render_template, jsonify,
    redirect, url_for
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import (
    FLASK_HOST, FLASK_PORT, MAX_UPLOAD_MB, ALLOWED_EXTS,
    UPLOAD_DIR, RESULTS_DIR, MODEL_FILENAME
)
from model.predict import predict_single_bytes, get_model, get_class_map

# ─────────────────────────────────────────────────────────────────────────────
#  App factory
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.secret_key = os.urandom(24)
APP_START = datetime.datetime.now()

# In-memory upload history (cleared on restart)
_upload_history = []
HISTORY_FILE    = os.path.join(RESULTS_DIR, "upload_history.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Disease information database
# ─────────────────────────────────────────────────────────────────────────────

DISEASE_INFO = {
    "healthy": {
        "description": "This plant appears healthy with no signs of disease.",
        "symptoms":    "No symptoms detected.",
        "treatment":   "Continue regular care: proper watering, fertilisation, and monitoring.",
        "severity":    "None",
    },
    "default": {
        "description": "A plant disease has been detected. Consult an agricultural expert for accurate diagnosis.",
        "symptoms":    "Visible lesions, discolouration, wilting, or spots on leaves.",
        "treatment":   (
            "1. Remove and destroy infected plant parts immediately.\n"
            "2. Avoid overhead watering to reduce leaf moisture.\n"
            "3. Apply appropriate fungicide or bactericide as directed.\n"
            "4. Improve air circulation around plants.\n"
            "5. Consult your local agricultural extension office."
        ),
        "severity":    "Moderate — act promptly.",
    },
    "uncertain_prediction": {
        "description": "The uploaded image does not match any disease class with sufficient confidence.",
        "symptoms":    "Please upload a clear leaf image from a supported crop.",
        "treatment":   "Retake the image and ensure the leaf occupies most of the frame.",
        "severity":    "Unknown",
    },
    "blight": {
        "description": "Blight causes rapid browning and death of plant tissue.",
        "symptoms":    "Brown or black lesions, water-soaked spots, wilting shoots.",
        "treatment":   (
            "1. Apply copper-based fungicide every 7–10 days.\n"
            "2. Remove all infected tissue and dispose away from garden.\n"
            "3. Avoid wetting foliage; use drip irrigation.\n"
            "4. Rotate crops next season to break disease cycle."
        ),
        "severity":    "High — act immediately.",
    },
    "rust": {
        "description": "Rust is a fungal disease causing orange-brown pustules on leaves.",
        "symptoms":    "Orange, yellow, or brown powdery pustules on leaf undersides.",
        "treatment":   (
            "1. Apply sulfur-based or triazole fungicide.\n"
            "2. Remove heavily infected leaves.\n"
            "3. Ensure good air circulation.\n"
            "4. Water at the base, not on foliage."
        ),
        "severity":    "Moderate.",
    },
    "scab": {
        "description": "Scab causes rough, corky lesions on fruit and leaves.",
        "symptoms":    "Dark, scabby lesions on fruit surface and leaves.",
        "treatment":   (
            "1. Apply captan or myclobutanil fungicide at bud break.\n"
            "2. Rake and destroy fallen leaves.\n"
            "3. Prune for better air circulation.\n"
            "4. Choose resistant varieties for future planting."
        ),
        "severity":    "Moderate.",
    },
    "spot": {
        "description": "Leaf spot diseases cause circular lesions on foliage.",
        "symptoms":    "Round or irregular spots with yellow halos on leaves.",
        "treatment":   (
            "1. Apply chlorothalonil or copper fungicide.\n"
            "2. Remove and destroy infected leaves.\n"
            "3. Avoid overhead irrigation.\n"
            "4. Space plants adequately for air flow."
        ),
        "severity":    "Low to Moderate.",
    },
    "mold": {
        "description": "Mold diseases thrive in humid conditions and can spread rapidly.",
        "symptoms":    "White, grey, or brown fuzzy growth on plant surfaces.",
        "treatment":   (
            "1. Improve ventilation and reduce humidity.\n"
            "2. Apply potassium bicarbonate or neem oil spray.\n"
            "3. Remove affected tissue promptly.\n"
            "4. Water in the morning so foliage dries during the day."
        ),
        "severity":    "Moderate.",
    },
}


def get_disease_info(class_name: str) -> dict:
    """Match class name to disease info using keyword lookup."""
    if class_name == "uncertain_prediction":
        return DISEASE_INFO["uncertain_prediction"]

    lower = class_name.lower()
    if "healthy" in lower:
        return DISEASE_INFO["healthy"]
    for key in ["blight", "rust", "scab", "spot", "mold"]:
        if key in lower:
            return DISEASE_INFO[key]
    return DISEASE_INFO["default"]


# ─────────────────────────────────────────────────────────────────────────────
#  History helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_history():
    global _upload_history
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            _upload_history = json.load(f)


def save_history():
    with open(HISTORY_FILE, "w") as f:
        json.dump(_upload_history[-100:], f, indent=2)   # keep last 100


def add_to_history(filename: str, prediction: dict, image_b64: str):
    entry = {
        "id":         str(uuid.uuid4())[:8],
        "filename":   filename,
        "timestamp":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "top_class":  prediction["predictions"][0]["display_name"],
        "confidence": prediction["predictions"][0]["confidence"],
        "image_b64":  image_b64,
        "predictions": prediction["predictions"],
    }
    _upload_history.insert(0, entry)
    save_history()
    return entry


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    model_ready = True
    try:
        get_model()
        get_class_map()
    except FileNotFoundError:
        model_ready = False
    return render_template("index.html", model_ready=model_ready)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    ext = os.path.splitext(file.filename)[1].lstrip(".").lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"File type '.{ext}' not allowed. "
                                  f"Use: {', '.join(ALLOWED_EXTS)}"}), 400

    try:
        raw_bytes = file.read()
        result    = predict_single_bytes(raw_bytes, filename=file.filename)

        # Attach disease info for top prediction
        top_class = result["predictions"][0]["class_name"]
        result["disease_info"] = get_disease_info(top_class)

        # Base64 encode for browser display
        b64 = base64.b64encode(raw_bytes).decode("utf-8")
        result["image_data"] = f"data:image/{ext};base64,{b64}"

        # Add to history
        add_to_history(file.filename, result, b64)

        return jsonify(result)

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


@app.route("/status")
def status():
    model_ok = True
    n_classes = 0
    model_name = MODEL_FILENAME
    tf_version = tf.__version__

    try:
        get_model()
        n_classes = len(get_class_map())
    except Exception:
        model_ok = False

    uptime_delta = datetime.datetime.now() - APP_START
    uptime = str(uptime_delta).split(".")[0]

    status_cards = [
        {"label": "Model Loaded", "ok": model_ok},
        {"label": "Flask Running", "ok": True},
        {"label": "Prediction Engine Active", "ok": model_ok},
        {"label": "Upload System Active", "ok": os.path.isdir(UPLOAD_DIR)},
    ]

    return render_template(
        "status.html",
        status_cards=status_cards,
        model_name=model_name,
        n_classes=n_classes,
        tf_version=tf_version,
        uptime=uptime,
        history_count=len(_upload_history),
    )


@app.route("/history")
def history():
    return render_template("history.html", history=_upload_history)


@app.route("/history/clear", methods=["POST"])
def clear_history():
    global _upload_history
    _upload_history = []
    save_history()
    return redirect(url_for("history"))


@app.route("/health")
def health():
    model_ok = True
    try:
        get_model()
        c = get_class_map()
        n_classes = len(c)
    except Exception:
        model_ok  = False
        n_classes = 0

    uptime_delta = datetime.datetime.now() - APP_START
    uptime = str(uptime_delta).split(".")[0]

    return jsonify({
        "status":    "ok" if model_ok else "model_not_loaded",
        "model_file": MODEL_FILENAME,
        "n_classes": n_classes,
        "tensorflow": tf.__version__,
        "uptime": uptime,
        "upload_history": len(_upload_history),
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Startup
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_history()

    print("=" * 55)
    print("   🌿 Plant Disease Detection — Web App")
    print("=" * 55)

    # Pre-load model at startup
    try:
        get_model()
        cm = get_class_map()
        print(f"   Model loaded — {len(cm)} classes ready")
    except FileNotFoundError as e:
        print(f"   ⚠️  {e}")
        print("   App will start but predictions won't work until model is trained.")

    print(f"\n   Open: http://{FLASK_HOST}:{FLASK_PORT}")
    print("   Press Ctrl+C to stop\n")

    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=False,
        threaded=True,
    )
