import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__). parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Data configuration
DATA_CONFIG = {
    'train_path': DATA_DIR / 'train',
    'val_path': DATA_DIR / 'val',
    'test_path': DATA_DIR / 'test',
    'image_size': 224,
    'batch_size': 32,
    'num_workers': 4,
}

# Training configuration
TRAIN_CONFIG = {
    'num_epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'dropout': 0.5,
    'num_classes': 2,
    'device': 'cuda',  # or 'cpu'
}

# Class names
CLASS_NAMES = ['intact', 'broken']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

# Model checkpoint
CHECKPOINT_PATH = MODEL_DIR / "best_model.pth"
LAST_CHECKPOINT_PATH = MODEL_DIR / "last_checkpoint.pth"