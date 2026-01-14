import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from src.data.dataset import AnimeDataset
from src.data.transforms import get_valid_transforms
from src.models.resnet import get_model
from src.utils.config import Config
from src.utils.logger import setup_logger

def evaluate():
    cfg = Config("config.yaml")
    logger = setup_logger(log_file="eval.log")
    device = cfg.DEVICE
    
    # Load Test Data
    logger.info("Loading Test Set...")
    test_dataset = AnimeDataset(
        data_dir=cfg.DATA_DIR / "test",
        transform=get_valid_transforms(cfg.cfg['data']['img_size'])
    )
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    classes = test_dataset.classes
    num_classes = len(classes)
    
    # Load Model
    logger.info(f"Loading Model: {cfg.cfg['model']['name']}")
    model = get_model(cfg.cfg['model']['name'], num_classes=num_classes)
    
    # Load Checkpoint
    checkpoint_path = cfg.CHECKPOINT_DIR / "best_model.pth"
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        return
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Predict
    all_preds = []
    all_labels = []
    
    logger.info("Running Inference...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Metrics
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    print(df_report)
    df_report.to_csv("docs/classification_report.csv")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig("docs/confusion_matrix.png")
    logger.info("Saved evaluation results to docs/")

if __name__ == "__main__":
    evaluate()
