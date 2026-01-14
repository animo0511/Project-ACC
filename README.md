# Anime Character Classifier (ACC)

A Deep Learning project for identifying anime characters from images.

## 📌 Project Overview

Anime Character Classifier (ACC) is a Computer Vision project designed to classify anime characters across different styles, angles, and lighting conditions.
The goal is to build a working end-to-end pipeline: dataset → training → evaluation → deployment.

*Note: I’m still learning AI/ML and built this project through experimentation, curiosity, and heavy vibe-coding powered by antigravity. 🙂*

## 🚀 Roadmap

### Phase 1: Core Concept & Planning
- Define pilot character list (10 characters)
- Finalize dataset acquisition strategy

### Phase 2: Dataset Collection
- Scrape images from sources such as MyAnimeList, Safebooru, etc.
- Target: 300–800 images per character
- **Preprocessing**:
  - Resize → 224x224
  - Normalize
  - Deduplicate / filtering

### Phase 3: Labeling & Dataset Split
- **Dataset structure**:
  - `train/` (70%)
  - `val/`   (20%)
  - `test/`  (10%)
- **Label encoding example**:
  - `{"goku": 0, "naruto": 1, ...}`

### Phase 4: Model Selection
- **Architecture**: ResNet50 / EfficientNet-B0 (Transfer Learning)
- **Framework**: PyTorch

### Phase 5: Training
- **Data augmentation**:
  - Rotation
  - Flip
  - Color Jitter
- **Tracking / visualization**:
  - TensorBoard or Weights & Biases
- **Optimization**:
  - Adam optimizer
  - CrossEntropyLoss

### Phase 6: Evaluation
- **Metrics**:
  - Top-1 Accuracy
  - F1-Score
  - Confusion Matrix
- **Target MVP accuracy**: ≥ 80%

### Phase 7: Deployment
- **Backend**: FastAPI
- **Frontend**: React + Vite
- **Model serving**: ONNX Runtime

### Phase 8: Extras / Stretch Features
- Face detection pipeline (YOLO / MediaPipe)
- Grad-CAM visualizations for explainability

## 🛠 Tech Stack

- **Languages**: Python, JavaScript
- **ML/DL**: PyTorch, Torchvision, Albumentations, Scikit-learn
- **CV**: OpenCV, MediaPipe
- **Web**: FastAPI, React, Vite
- **Tools**: Git, Docker (optional)

## 📦 Installation

```bash
pip install -r requirements.txt
cd web/frontend && npm install
```

## 🧠 Learning Journey Note

- This project is part of my journey into Machine Learning & AI.
- I am not an expert (yet) — I’m learning concepts as I build.
- A lot of this code was written by experimenting, debugging, and vibe-coding with the help of antigravity. 🚀
