# =============================================================================
#  config.py — Central configuration for Plant Disease Detection system
# =============================================================================

import os

# ── Root paths ────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
# Dataset only needed for training — not required on Render
DATASET_DIR  = os.environ.get("DATASET_DIR", os.path.join(BASE_DIR, "dataset"))
MODEL_DIR    = os.path.join(BASE_DIR, "models")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
UPLOAD_DIR   = os.path.join(BASE_DIR, "app", "static", "uploads")

for _d in (MODEL_DIR, RESULTS_DIR, UPLOAD_DIR):
    os.makedirs(_d, exist_ok=True)

# ── Image settings ────────────────────────────────────────────────────────────
IMG_HEIGHT   = 128
IMG_WIDTH    = 128
IMG_CHANNELS = 3
INPUT_SHAPE  = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# ── Dataset split ─────────────────────────────────────────────────────────────
TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
TEST_RATIO   = 0.10
RANDOM_SEED  = 42

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE      = 32
EPOCHS          = 50
LEARNING_RATE   = 0.001
LR_DECAY_FACTOR = 0.5
LR_PATIENCE     = 5
ES_PATIENCE     = 12
MIN_LR          = 1e-7

# ── Augmentation ──────────────────────────────────────────────────────────────
AUG_ROTATION        = 25
AUG_WIDTH_SHIFT     = 0.12
AUG_HEIGHT_SHIFT    = 0.12
AUG_ZOOM            = 0.20
AUG_HORIZONTAL_FLIP = True
AUG_BRIGHTNESS      = [0.75, 1.30]
AUG_FILL_MODE       = "nearest"

# ── Model persistence ─────────────────────────────────────────────────────────
MODEL_FILENAME  = "plant_disease_cnn.keras"
CLASS_MAP_FILE  = os.path.join(MODEL_DIR, "class_names.json")
HISTORY_FILE    = os.path.join(RESULTS_DIR, "training_history.json")

# ── Prediction ────────────────────────────────────────────────────────────────
TOP_K = 3

# ── TensorBoard ───────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(BASE_DIR, "logs")

# ── Flask ─────────────────────────────────────────────────────────────────────
FLASK_HOST    = "0.0.0.0"
FLASK_PORT    = int(os.environ.get("PORT", 5000))
MAX_UPLOAD_MB = 16
ALLOWED_EXTS  = {"png", "jpg", "jpeg", "webp", "bmp"}
