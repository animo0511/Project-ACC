import cv2
import os
import hashlib
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
IMG_SIZE = (224, 224)

# Map messy folder names to clean 10 classes
CLASS_MAPPING = {
    "monkey_d._luffy": "Luffy",
    "Monkey D. Luffy anime": "Luffy",
    
    "roronoa_zoro": "Zoro",
    "Roronoa Zoro anime": "Zoro",
    
    "naruto_uzumaki": "Naruto",
    "Naruto Uzumaki anime": "Naruto",
    
    "sasuke_uchiha": "Sasuke",
    "Sasuke Uchiha anime": "Sasuke",
    
    "son_goku": "Goku",
    "Son Goku anime": "Goku",
    
    "vegeta": "Vegeta",
    "Vegeta anime": "Vegeta",
    "Vegeta Dragon Ball": "Vegeta",
    
    "saitama_one_punch_man": "Saitama",
    "Saitama One Punch Man anime": "Saitama",
    
    "light_yagami": "Light_Yagami",
    "Light Yagami anime": "Light_Yagami",
    
    "levi_ackerman": "Levi",
    "Levi Ackerman anime": "Levi",
    
    "satoru_gojo": "Gojo",
    "Satoru Gojo anime": "Gojo"
}

def compute_hash(image_path):
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def process_images():
    if not RAW_DIR.exists():
        logging.error("Raw data directory not found!")
        return

    # Clear previous run
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    PROCESSED_DIR.mkdir(parents=True)

    stats = {"total": 0, "processed": 0, "duplicates": 0, "errors": 0}
    
    # Iterate over each character folder
    for character_dir in RAW_DIR.iterdir():
        if not character_dir.is_dir():
            continue
            
        raw_name = character_dir.name
        
        # Decide target class name
        if raw_name in CLASS_MAPPING:
            clean_name = CLASS_MAPPING[raw_name]
        else:
            # Fallback normalization
            clean_name = raw_name.split()[0].replace("_", "").capitalize()
            
        target_dir = PROCESSED_DIR / clean_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Processing {raw_name} -> {clean_name}...")
        
        # We need global dedup per class to handle merged folders
        # Since we might process multiple raw folders into one target folder,
        # we ideally should load existing hashes from target, but simplest is 
        # to just rely on unique filenames or handle collisions.
        
        for img_path in tqdm(list(character_dir.glob("*.[jJpP]*"))):
            stats["total"] += 1
            try:
                # 1. Read
                img = cv2.imread(str(img_path))
                if img is None:
                    stats["errors"] += 1
                    continue
                    
                # 2. Resize
                h, w = img.shape[:2]
                scale = min(IMG_SIZE[0]/w, IMG_SIZE[1]/h)
                new_w, new_h = int(w*scale), int(h*scale)
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                canvas = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)
                y_offset = (IMG_SIZE[1] - new_h) // 2
                x_offset = (IMG_SIZE[0] - new_w) // 2
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                
                # 3. Save with unique name
                # using hash as filename is a good way to auto-dedup
                img_hash = hashlib.md5(canvas.tobytes()).hexdigest()
                save_path = target_dir / f"{img_hash}.jpg"
                
                if not save_path.exists():
                    cv2.imwrite(str(save_path), canvas)
                    stats["processed"] += 1
                else:
                    stats["duplicates"] += 1
                
            except Exception as e:
                logging.error(f"Error processing {img_path}: {e}")
                stats["errors"] += 1

    logging.info(f"Processing complete. Stats: {stats}")

if __name__ == "__main__":
    process_images()
