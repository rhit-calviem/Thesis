import torch

# DEVICE CONFIGURATION
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MODEL & TRAINING HYPERPARAMETERS
UPSCALE_FACTOR = 2
PATCH_SIZE = 96 
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_WORKERS = 4 

# DATA PATHS
# Training data
HR_TRAIN_DIR = 'data/DIV2K/DIV2K_train_HR'

# Testing data
TEST_BASE_DIR = 'data/Set5' 
TEST_SUB_DIR = f'image_SRF_{UPSCALE_FACTOR}' # created for current use of Set5 data for eval

MODEL_SAVE_PATH = 'basic_sr_model.pth'

