import torch
import cv2
import albumentations
import numpy as np
import sys

def verify():
    print(f"Python: {sys.version}")
    
    # Check PyTorch
    try:
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch not found!")

    # Check OpenCV
    try:
        print(f"OpenCV: {cv2.__version__}")
    except ImportError:
        print("OpenCV not found!")

    # Check Albumentations
    try:
        print(f"Albumentations: {albumentations.__version__}")
    except ImportError:
        print("Albumentations not found!")

if __name__ == "__main__":
    verify()
