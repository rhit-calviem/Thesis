import os
import math
import random
from glob import glob
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as F

import matplotlib.pyplot as plt
from tqdm import tqdm

# Import from our other files
from config import *
from models import PlaceholderSRModel

# --- 1. The Data Pipeline ---

class TrainDataset(Dataset):
    """
    Dataset for training. Creates LR patches on-the-fly from a directory of HR images.
    """
    def __init__(self, data_dir):
        super(TrainDataset, self).__init__()
        self.image_files = sorted(glob(f'{data_dir}/*'))
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        hr_image = Image.open(self.image_files[idx]).convert("RGB")
        lr_patch, hr_patch = self._get_train_patch(hr_image)
        return self._augment(lr_patch, hr_patch)
    
    def _get_train_patch(self, hr_image):
        hr_w, hr_h = hr_image.size
        if hr_w < PATCH_SIZE or hr_h < PATCH_SIZE:
             hr_image = F.resize(hr_image, (PATCH_SIZE, PATCH_SIZE), T.InterpolationMode.BICUBIC)
             hr_w, hr_h = hr_image.size
        
        top = random.randint(0, hr_h - PATCH_SIZE)
        left = random.randint(0, hr_w - PATCH_SIZE)
        hr_patch = hr_image.crop((left, top, left + PATCH_SIZE, top + PATCH_SIZE))
        
        lr_patch_size = PATCH_SIZE // UPSCALE_FACTOR
        lr_patch = hr_patch.resize((lr_patch_size, lr_patch_size), Image.BICUBIC)
        
        return self.to_tensor(lr_patch), self.to_tensor(hr_patch)

    def _augment(self, lr_img, hr_img):
        if random.random() > 0.5: lr_img, hr_img = F.hflip(lr_img), F.hflip(hr_img)
        if random.random() > 0.5: lr_img, hr_img = F.vflip(lr_img), F.vflip(hr_img)
        return lr_img, hr_img

class TestDataset(Dataset):
    """
    Dataset for testing. Loads pre-existing LR/HR image pairs from a structured directory.
    """
    def __init__(self, base_dir, sub_dir):
        super(TestDataset, self).__init__()
        self.data_path = os.path.join(base_dir, sub_dir)
        # Find all the High-Resolution images
        self.hr_files = sorted(glob(os.path.join(self.data_path, '*_HR.png')))
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        # Infer the Low-Resolution path from the HR path
        lr_path = hr_path.replace('_HR.png', '_LR.png')
        
        hr_image = Image.open(hr_path).convert("RGB")
        lr_image = Image.open(lr_path).convert("RGB")

        return self.to_tensor(lr_image), self.to_tensor(hr_image), hr_path

# --- 2. Helper Functions ---
def calculate_psnr(original, compressed):
    """Calculates Peak Signal-to-Noise Ratio."""
    original, compressed = original.cpu().numpy(), compressed.cpu().numpy()
    mse = np.mean((original - compressed) ** 2)
    if mse == 0: return 100
    psnr = 20 * math.log10(1.0 / math.sqrt(mse))
    return psnr

def visualize_and_save_result(model, dataset, device, save_path='sr_visualization.png'):
    """Visualizes a random sample and saves the plot to a file."""
    model.eval()
    idx = random.randint(0, len(dataset) - 1)
    lr_image, hr_image, filename = dataset[idx]
    lr_image_batch = lr_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        sr_image = model(lr_image_batch).clamp(0.0, 1.0).squeeze(0)
        
    to_pil = T.ToPILImage()
    images = [to_pil(img) for img in [lr_image, sr_image.cpu(), hr_image]]
    titles = [f'Low-Res Input\n{images[0].size}', f'Model Output\n{images[1].size}', f'Ground Truth\n{images[2].size}']
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(f'SR Result for: {os.path.basename(filename)}', fontsize=16)
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    
    # --- THIS IS THE KEY CHANGE ---
    # Instead of showing, save the figure to a file
    plt.savefig(save_path)
    plt.close() # Close the figure to free up memory

# --- 3. Main Execution Block ---
def main():
    # --- Setup ---
    print(f"Using device: {DEVICE}")
    train_dataset = TrainDataset(HR_TRAIN_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_dataset = TestDataset(TEST_BASE_DIR, TEST_SUB_DIR)

    model = PlaceholderSRModel(upscale_factor=UPSCALE_FACTOR).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for lr_patches, hr_patches in progress_bar:
            lr_patches, hr_patches = lr_patches.to(DEVICE), hr_patches.to(DEVICE)
            
            optimizer.zero_grad()
            sr_patches = model(lr_patches)
            loss = criterion(sr_patches, hr_patches)
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix(loss=f"{loss.item():.5f}")
    
    print('Finished Training.')
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # --- Evaluation ---
    print(f"\n--- Evaluating on {TEST_BASE_DIR}/{TEST_SUB_DIR} ---")
    model.eval()
    total_psnr = 0
    with torch.no_grad():
        for lr_image, hr_image, _ in tqdm(test_dataset, desc="Testing"):
            lr_image, hr_image = lr_image.to(DEVICE), hr_image.to(DEVICE)
            # Add batch dimension for the model
            sr_image = model(lr_image.unsqueeze(0)).clamp(0.0, 1.0).squeeze(0)
            total_psnr += calculate_psnr(hr_image, sr_image)
            
    avg_psnr = total_psnr / len(test_dataset)
    print(f"\nAverage PSNR: {avg_psnr:.2f} dB")

    # --- Visualization ---
    visualize_and_save_result(model, test_dataset, DEVICE)
    print(f"\nVisualization saved to 'sr_visualization.png'.")
    print("Download the file from the VS Code file explorer to view it.")


if __name__ == '__main__':
    main()

