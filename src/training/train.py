import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from pathlib import Path

# Project Imports
from src.data.dataset import AnimeDataset
from src.data.transforms import get_train_transforms, get_valid_transforms
from src.models.resnet import get_model
from src.utils.config import Config
from src.utils.logger import setup_logger

def train():
    # 1. Setup
    cfg = Config("config.yaml")
    logger = setup_logger(log_file="training.log")
    
    device = cfg.DEVICE
    logger.info(f"Using device: {device}")
    
    # 2. Data
    logger.info("Loading Datasets...")
    train_dataset = AnimeDataset(
        data_dir=cfg.DATA_DIR / "train",
        transform=get_train_transforms(cfg.cfg['data']['img_size'])
    )
    val_dataset = AnimeDataset(
        data_dir=cfg.DATA_DIR / "val",
        transform=get_valid_transforms(cfg.cfg['data']['img_size'])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0) # 0 for windows safety
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    
    classes = train_dataset.classes
    num_classes = len(classes)
    logger.info(f"Classes ({num_classes}): {classes}")
    
    # 3. Model
    logger.info(f"Initializing Model: {cfg.cfg['model']['name']}")
    model = get_model(cfg.cfg['model']['name'], num_classes=num_classes)
    model.to(device)
    
    # 4. Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    
    # 5. Training Loop
    best_acc = 0.0
    
    for epoch in range(cfg.EPOCHS):
        logger.info(f"Epoch {epoch+1}/{cfg.EPOCHS}")
        
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': train_loss/len(train_loader)})
            
        train_acc = 100 * train_correct / total
        logger.info(f"Train Loss: {train_loss/len(train_loader):.4f} | Acc: {train_acc:.2f}%")
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        logger.info(f"Val Loss: {val_loss/len(val_loader):.4f} | Acc: {val_acc:.2f}%")
        
        # Save Best
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = cfg.CHECKPOINT_DIR / "best_model.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model to {save_path}")

    logger.info("Training Complete.")

if __name__ == "__main__":
    train()
