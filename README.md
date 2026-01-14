# Anime Character Classifier (ACC)

## 📌 Project Overview
A deep learning project to classify anime characters using state-of-the-art Computer Vision techniques.
**Goal**: Build a system capable of recognizing characters from varying angles, styles, and lighting conditions.

## 🚀 Roadmap

### Phase 1: Core Concept & Planning
- [ ] Define Pilot Character List (10 characters)
- [ ] Finalize Dataset Strategy

### Phase 2: Dataset Collection
- [ ] Scrape images from web sources (MyAnimeList, Safebooru, etc.)
- [ ] Target: 300-800 images per character
- [ ] Preprocessing: Resize (224x224), Normalize, Deduplicate

### Phase 3: Labeling & Split
- [ ] Structure: Train (70%) / Val (20%) / Test (10%)
- [ ] Label Encoding: `{"goku": 0, "naruto": 1, ...}`

### Phase 4: Model Selection
- [ ] Architecture: ResNet50 / EfficientNet-B0 (Transfer Learning)
- [ ] Framework: PyTorch

### Phase 5: Training
- [ ] Data Augmentation (Rotation, FLip, Color Jitter)
- [ ] Tracking: TensorBoard / Weights & Biases
- [ ] Optimization: Adam, CrossEntropyLoss

### Phase 6: Evaluation
- [ ] Metrics: Top-1 Accuracy, F1-Score, Confusion Matrix
- [ ] Target: >80% Accuracy for MVP

### Phase 7: Deployment
- [ ] **Backend**: FastAPI
- [ ] **Frontend**: React + Vite
- [ ] **Serving**: ONNX Runtime

### Phase 8: Features++
- [ ] Face Detection Pipeline (YOLO/MediaPipe)
- [ ] Grad-CAM Visualizations

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
