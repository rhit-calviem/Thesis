import torch
import os

# DEVICE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MODEL & TRAINING HYPERPARAMETERS
UPSCALE_FACTOR = 2
PATCH_SIZE = 96
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_WORKERS = 4
LOSS_FN = "L1" # Options: "L1", "MSE"

# DATA PATHS
HR_TRAIN_DIR = 'data/DIV2K/DIV2K_train_HR'

# Testing datasets
TEST_DATASETS = {
    "Set5": "data/Set5",
    "Set14": "data/Set14",
    "Urban100": "data/Urban100",
    "BSD100": "data/BSD100"
}

# SAVE PATHS
MODEL_SAVE_DIR = 'Models'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
MODEL_SAVE_PATH = f'{MODEL_SAVE_DIR}/hrmodel.pth'

# VISUALIZATION
VISUALIZATION_SAVE_PATH = 'Pictures/sr_visualization.png'
