#IMPORTS
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import *
from models import Model
from utils_data import *
from utils import *

def main():
    print(f"Using device: {DEVICE}")
    train_dataset = TrainDataset(HR_TRAIN_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_dataset = TestDataset(TEST_BASE_DIR, TEST_SUB_DIR)

    model = Model(upscale_factor=UPSCALE_FACTOR).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()

    # TRAINING LOOP
    print("\nStarting Training")
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

    # EVALUATION LOOP
    print(f"\nEvaluating on {TEST_BASE_DIR}/{TEST_SUB_DIR}")
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

    # VISUALIZATION
    visualize_and_save_result(model, test_dataset, DEVICE)
    print(f"\nVisualization saved to 'sr_visualization.png'.")


if __name__ == '__main__':
    main()

