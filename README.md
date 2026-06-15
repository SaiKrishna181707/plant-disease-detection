# 🌿 Plant Disease Detection System

A production-grade Flask + TensorFlow web app for crop disease identification using a custom Convolutional Neural Network trained on the PlantVillage dataset.

## Overview

This repository contains a complete plant disease detection pipeline:
- image preprocessing and augmentation
- custom CNN training from scratch
- prediction and confidence handling
- Flask web app with drag-and-drop image upload
- upload history tracking and status dashboard

The app now includes improved low-confidence handling for unrelated images, a branded UI, and a dashboard showing model readiness and runtime metadata.

## Architecture

- `model/` — model architecture, training, evaluation, and inference logic
- `utils/` — dataset loading, preprocessing, and visualization utilities
- `app/` — Flask frontend, templates, and static assets
- `config.py` — central application and training configuration
- `README.md` — documentation and usage instructions

## Tech Stack

- Python 3
- TensorFlow / Keras
- Flask
- HTML/CSS/JavaScript
- Render (deployment target)

## Deployment URL

The app is designed to deploy to Render or any Python web host.

Example deployment URL:

`https://your-render-app.onrender.com`

Replace this with your actual Render service URL after deployment.

## Screenshots

1. **Detect page** — drag-and-drop uploader, hero stats, and prediction panel.
2. **Status page** — runtime health dashboard, model state, TensorFlow version.
3. **History page** — past image uploads with confidence and prediction details.

> Add actual screenshot images here once available, e.g. `screenshots/detect.png`.

## Run locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the web app

```bash
python app/app.py
```

### 3. Open in browser

Visit `http://localhost:5000`.

### 4. Use the web UI

- drag and drop or click to upload a leaf image
- check prediction confidence and disease advice
- visit `/status` for system health info
- visit `/history` to review upload history

## CLI usage

### Single image prediction

```bash
python model/predict.py --image path/to/leaf.jpg
```

### Batch prediction

```bash
python model/predict.py --batch img1.jpg img2.jpg img3.jpg --top_k 3
```

## Notes

- The app now flags predictions with top confidence below 70% as **Uncertain Prediction**.
- Uncertain uploads show a warning and do not display misleading disease severity.
- The status dashboard reports model loading, server uptime, TensorFlow version, and upload system state.

## Recommended next steps

- Train the model locally with `python model/train.py`.
- Deploy to Render using the Render Python service.
- Replace the placeholder deployment URL in this README with the live site link.
