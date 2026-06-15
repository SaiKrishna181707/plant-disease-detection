# =============================================================================
#  config.py — Central configuration for Plant Disease Detection system
#  All hyperparameters, paths, and flags live here. Edit this file only.
# =============================================================================

import os

# ── Root paths ────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = "/kaggle/input/datasets/arjuntejaswi/plant-village/PlantVillage"
MODEL_DIR    = os.path.join(BASE_DIR, "models")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
UPLOAD_DIR   = os.path.join(BASE_DIR, "app", "static", "uploads")

for _d in (MODEL_DIR, RESULTS_DIR, UPLOAD_DIR):
    os.makedirs(_d, exist_ok=True)

# ── Image settings ────────────────────────────────────────────────────────────
IMG_HEIGHT   = 128          # resize target height
IMG_WIDTH    = 128          # resize target width
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
LR_DECAY_FACTOR = 0.5       # ReduceLROnPlateau factor
LR_PATIENCE     = 5         # epochs before LR reduction
ES_PATIENCE     = 12        # early-stopping patience
MIN_LR          = 1e-7

# ── Augmentation ──────────────────────────────────────────────────────────────
AUG_ROTATION    = 25        # degrees
AUG_WIDTH_SHIFT = 0.12
AUG_HEIGHT_SHIFT= 0.12
AUG_ZOOM        = 0.20
AUG_HORIZONTAL_FLIP = True
AUG_BRIGHTNESS  = [0.75, 1.30]
AUG_FILL_MODE   = "nearest"

# ── Model persistence ─────────────────────────────────────────────────────────
MODEL_FILENAME  = "plant_disease_cnn.keras"
CLASS_MAP_FILE  = os.path.join(MODEL_DIR, "class_names.json")
HISTORY_FILE    = os.path.join(RESULTS_DIR, "training_history.json")

# ── Prediction ────────────────────────────────────────────────────────────────
TOP_K           = 3         # top-k predictions to return

# ── TensorBoard ───────────────────────────────────────────────────────────────
LOG_DIR         = os.path.join(BASE_DIR, "logs")

# ── Flask ─────────────────────────────────────────────────────────────────────
FLASK_HOST      = "0.0.0.0"
FLASK_PORT      = 5000
MAX_UPLOAD_MB   = 16
ALLOWED_EXTS    = {"png", "jpg", "jpeg", "webp", "bmp"}
