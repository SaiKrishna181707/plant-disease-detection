# 🌿 Plant Disease Detection — Custom CNN from Scratch

> A production-quality deep learning system for automated plant disease
> classification. **No pretrained models. No transfer learning.**
> Every weight is trained from random initialisation on your PlantVillage data.

---

## 📁 Project Structure

```
plant-disease-detection/
│
├── config.py                        ← All hyperparameters & paths
├── requirements.txt
├── README.md
│
├── dataset/                         ← PUT YOUR PLANTVILLAGE FOLDER HERE
│   ├── Tomato___Early_blight/
│   ├── Tomato___Late_blight/
│   ├── Pepper__bell___healthy/
│   └── ...  (all class folders)
│
├── models/                          ← Auto-created; stores trained model
│   ├── plant_disease_cnn.keras
│   └── class_names.json
│
├── results/                         ← Auto-created; stores plots & reports
│   ├── training_history.png
│   ├── training_history.json
│   ├── confusion_matrix.png
│   ├── per_class_accuracy.png
│   ├── sample_grid.png
│   ├── evaluation_report.json
│   ├── batch_predictions.json
│   ├── epoch_log.csv
│   └── upload_history.json
│
├── utils/
│   ├── dataset_loader.py            ← Auto-discovers classes, builds tf.data pipelines
│   ├── preprocess.py                ← Single-image / batch preprocessing
│   └── visualization.py            ← All plotting utilities
│
├── model/
│   ├── build_model.py               ← Custom CNN architecture (scratch)
│   ├── train.py                     ← Training entry point
│   ├── evaluate.py                  ← Full evaluation + metrics
│   └── predict.py                   ← Single & batch inference CLI
│
├── app/
│   ├── app.py                       ← Flask web application
│   ├── static/
│   │   ├── css/style.css
│   │   ├── js/app.js
│   │   └── uploads/                 ← Uploaded images
│   └── templates/
│       ├── index.html               ← Upload + prediction UI
│       └── history.html             ← Upload history page
│
└── logs/                            ← TensorBoard logs (auto-created)
```

---

## ⚡ Quick Start

### 1 · Install dependencies
```bash
pip install -r requirements.txt
```

### 2 · Place your dataset
```
plant-disease-detection/
└── dataset/
    ├── Tomato___Early_blight/    ← one sub-folder per class
    ├── Tomato___Late_blight/
    └── ...
```
The system **auto-detects all classes** — no hardcoding needed.

### 3 · Train the model
```bash
python model/train.py
```
- Discovers all classes automatically
- Trains a 5-block custom CNN from scratch
- Saves best checkpoint to `models/`
- Generates training curves in `results/`

### 4 · Evaluate
```bash
python model/evaluate.py
```
Outputs accuracy, precision, recall, F1, confusion matrix, per-class bars.

### 5 · Predict from CLI
```bash
# Single image
python model/predict.py --image path/to/leaf.jpg

# Batch
python model/predict.py --batch img1.jpg img2.jpg img3.jpg --top_k 3
```

### 6 · Launch web app
```bash
python app/app.py
```
Open **http://localhost:5000** — drag-drop a leaf image and get instant results.

### 7 · TensorBoard
```bash
tensorboard --logdir logs/
```

---

## 🧠 CNN Architecture (100% from scratch)

```
Input (128 × 128 × 3)
│
├─ Block 1 → Conv2D(32)  × 2 → BN → ReLU → MaxPool(2×2) → Dropout(0.25)
├─ Block 2 → Conv2D(64)  × 2 → BN → ReLU → MaxPool(2×2) → Dropout(0.25)
├─ Block 3 → Conv2D(128) × 2 → BN → ReLU → MaxPool(2×2) → Dropout(0.30)
├─ Block 4 → Conv2D(256) × 2 → BN → ReLU → MaxPool(2×2) → Dropout(0.30)
├─ Block 5 → Conv2D(512) × 1 → BN → ReLU → MaxPool(2×2) → Dropout(0.35)
│
├─ GlobalAveragePooling2D
│
├─ Dense(1024) → BN → ReLU → Dropout(0.50)
├─ Dense(512)  → BN → ReLU → Dropout(0.40)
│
└─ Dense(N_CLASSES, Softmax)
```

**Key design decisions:**
- `He normal` initialisation for all conv/dense weights (optimal for ReLU)
- `L2 weight regularisation (1e-4)` on all conv and dense layers
- `GlobalAveragePooling2D` instead of Flatten — fewer parameters, less overfitting
- `BatchNormalization` after every conv layer — faster convergence, stable training
- Progressive dropout (0.25 → 0.35 in conv, 0.40–0.50 in dense) — prevents co-adaptation

---

## 📊 Training Features

| Feature | Detail |
|---------|--------|
| Optimiser | Adam (β₁=0.9, β₂=0.999) |
| Loss | Categorical Crossentropy |
| Metrics | Accuracy, Top-3 Acc, Precision, Recall |
| LR Schedule | ReduceLROnPlateau (factor=0.5, patience=5) |
| Early Stopping | patience=12, restore best weights |
| Checkpoint | Best val_accuracy saved automatically |
| Class Weights | Computed per-class for imbalanced datasets |
| Augmentation | Flip, brightness, contrast, saturation, crop+resize, rotation |
| TensorBoard | Full histogram + scalar logging |

---

## 🔧 Configuration

Edit `config.py` to change anything:

| Key | Default | Description |
|-----|---------|-------------|
| `IMG_HEIGHT / IMG_WIDTH` | 128 | Input resolution |
| `BATCH_SIZE` | 32 | Training batch size |
| `EPOCHS` | 50 | Maximum epochs |
| `LEARNING_RATE` | 0.001 | Initial Adam LR |
| `ES_PATIENCE` | 12 | Early stopping patience |
| `TOP_K` | 3 | Top-K predictions |

---

## 📋 Resume Bullet Points

```
• Built a custom 5-block CNN from scratch (no transfer learning) using
  TensorFlow/Keras, training on 20k+ PlantVillage leaf images across
  15+ disease classes with automatic class discovery.

• Applied data augmentation (flip, rotation, crop, brightness/contrast),
  BatchNormalization, and progressive Dropout regularisation, achieving
  high multi-class validation accuracy with L2 weight decay.

• Implemented a full ML pipeline including stratified splits, LR scheduling,
  early stopping, TensorBoard logging, and a Flask web app with drag-drop
  upload, top-3 confidence display, and disease treatment advice.
```
