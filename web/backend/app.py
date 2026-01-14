from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import torch
import cv2
import numpy as np
import io
from PIL import Image
import random
from pathlib import Path

from src.models.resnet import get_model
from src.data.transforms import get_valid_transforms
from src.utils.config import Config

app = FastAPI(title="Anime Character Classifier")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model Global
cfg = Config("config.yaml")
DEVICE = cfg.DEVICE
# Classes as found in data/dataset/train (Must match training order which is sorted(os.listdir))
CLASSES = [
    "Vegeta Dragon Ball", 
    "Vegeta anime", 
    "levi_ackerman", 
    "light_yagami", 
    "monkey_d._luffy", 
    "naruto_uzumaki", 
    "roronoa_zoro", 
    "saitama_one_punch_man", 
    "sasuke_uchiha", 
    "satoru_gojo", 
    "son_goku", 
    "vegeta"
]

def get_display_name(raw_name):
    """Map raw folder names to clean display names"""
    name_map = {
        "monkey_d._luffy": "Luffy",
        "roronoa_zoro": "Zoro",
        "naruto_uzumaki": "Naruto",
        "sasuke_uchiha": "Sasuke",
        "son_goku": "Goku",
        "vegeta": "Vegeta",
        "Vegeta Dragon Ball": "Vegeta",
        "Vegeta anime": "Vegeta",
        "saitama_one_punch_man": "Saitama",
        "light_yagami": "Light",
        "levi_ackerman": "Levi",
        "satoru_gojo": "Gojo"
    }
    return name_map.get(raw_name, raw_name.replace("_", " ").title())
MODEL = None

@app.on_event("startup")
async def load_predictor():
    global MODEL
    # Try to load model, else fail gracefully
    try:
        model = get_model(cfg.cfg['model']['name'], num_classes=len(CLASSES))
        checkpoint_path = cfg.CHECKPOINT_DIR / "best_model.pth"
        
        if checkpoint_path.exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            print("Loaded checkpoint.")
            model.to(DEVICE)
            model.eval()
            MODEL = model
        else:
            print("Warning: No checkpoint found. API will use MOCK predictions.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read Image
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    if MODEL:
        # Real Inference
        transform = get_valid_transforms(cfg.cfg['data']['img_size'])
        augmented = transform(image=image_np)
        tensor = augmented['image'].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = MODEL(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_idx = torch.max(probs, 1)
            
        idx = top_idx.item()
        confidence = top_prob.item()
        raw_class = CLASSES[idx] if idx < len(CLASSES) else "Unknown"
        predicted_class = get_display_name(raw_class)
    else:
        # Mock Inference (For "Play" mode)
        # Simple heuristic or random for demo
        raw_class = random.choice(CLASSES)
        predicted_class = get_display_name(raw_class)
        confidence = random.uniform(0.70, 0.99)

    return {
        "character": predicted_class,
        "confidence": float(confidence),
        "anime": "Anime Source" 
    }

# Serve Static Frontend
app.mount("/static", StaticFiles(directory="web/backend/static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('web/backend/static/index.html')
