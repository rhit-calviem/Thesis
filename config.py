import torch
import os

# DEVICE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MODEL & TRAINING HYPERPARAMETERS
UPSCALE_FACTOR = 4  # or 2/3 per experiment
LR_PATCH_SIZE = 64    # LR patch
PATCH_SIZE = LR_PATCH_SIZE * UPSCALE_FACTOR  # HR patch size
BATCH_SIZE = 64
EPOCHS = None         # prefer iter-based training, but if epoch-based adapt
LEARNING_RATE = 5e-4
OPTIMIZER = "AdamW"
WEIGHT_DECAY = 1e-4   # paper uses AdamW, choose small wd
NUM_ITERATIONS = 1000
LR_DROP_ITERS = [200000, 400000, 600000]
NUM_WORKERS = 4
LOSS_FN = "L1"  # or "MSE"

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
