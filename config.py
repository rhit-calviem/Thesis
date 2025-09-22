import torch

# --- Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model & Training Hyperparameters ---
UPSCALE_FACTOR = 2
# HR Patches are cropped to PATCH_SIZE x PATCH_SIZE
PATCH_SIZE = 96 
BATCH_SIZE = 16
# Number of epochs to train for. 5 is a good start for pipeline testing.
EPOCHS = 5
# Learning rate for the Adam optimizer
LEARNING_RATE = 1e-4
# Number of CPU workers for loading data
NUM_WORKERS = 4 

# --- Data Paths ---
# Training data: Assumes a flat directory of HR images
HR_TRAIN_DIR = 'data/DIV2K/DIV2K_train_HR'

# --- Test Data Paths (Handles sub-folders) ---
# Base directory for the test set
TEST_BASE_DIR = 'data/Set5' 
# The script will automatically look inside the folder corresponding to the upscale factor
# e.g., 'data/Set5/image_SRF_2' for UPSCALE_FACTOR = 2
TEST_SUB_DIR = f'image_SRF_{UPSCALE_FACTOR}'

# --- Model Saving ---
MODEL_SAVE_PATH = 'basic_sr_model.pth'

