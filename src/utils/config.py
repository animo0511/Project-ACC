import torch
import yaml
from pathlib import Path

class Config:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
    def get(self, key, default=None):
        keys = key.split('.')
        val = self.cfg
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k, default)
            else:
                return default
        return val

    @property
    def DATA_DIR(self):
        return Path(self.cfg.get('data', {}).get('root', 'data/dataset'))
        
    @property
    def CHECKPOINT_DIR(self):
        return Path(self.cfg.get('model', {}).get('checkpoint_dir', 'models/checkpoints'))

    @property
    def DEVICE(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @property
    def EPOCHS(self):
        return self.cfg.get('training', {}).get('epochs', 10)
        
    @property
    def BATCH_SIZE(self):
        return self.cfg.get('training', {}).get('batch_size', 32)
        
    @property
    def LEARNING_RATE(self):
        return self.cfg.get('training', {}).get('learning_rate', 0.001)
