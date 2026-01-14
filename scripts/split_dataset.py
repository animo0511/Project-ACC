import shutil
import random
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROCESSED_DIR = Path("data/processed")
DATASET_DIR = Path("data/dataset") # Final spot for split data

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

def split_dataset():
    if not PROCESSED_DIR.exists():
        logging.error(f"{PROCESSED_DIR} does not exist. Run process_data.py first.")
        return

    # Create split directories
    for split in ['train', 'val', 'test']:
        split_dir = DATASET_DIR / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)

    # Iterate over classes
    class_dirs = [d for d in PROCESSED_DIR.iterdir() if d.is_dir()]
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        logging.info(f"Splitting {class_name}...")
        
        # Create class subdirectories in each split
        for split in ['train', 'val', 'test']:
            (DATASET_DIR / split / class_name).mkdir(exist_ok=True)
            
        images = list(class_dir.glob("*.[jJpP]*"))
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        # n_test is the rest
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train+n_val]
        test_imgs = images[n_train+n_val:]
        
        # Copy files
        for img in train_imgs:
            shutil.copy(img, DATASET_DIR / 'train' / class_name / img.name)
            
        for img in val_imgs:
            shutil.copy(img, DATASET_DIR / 'val' / class_name / img.name)
            
        for img in test_imgs:
            shutil.copy(img, DATASET_DIR / 'test' / class_name / img.name)
            
        logging.info(f"  Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

    logging.info("Dataset splitting complete.")

if __name__ == "__main__":
    split_dataset()
