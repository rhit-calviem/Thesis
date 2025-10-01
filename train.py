import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import *
from models import Model
from utils_data import TrainDataset
from torch.utils.data import DataLoader

# Training function
def main():
    print(f"Using device: {DEVICE}")
    # load training data
    train_dataset = TrainDataset(HR_TRAIN_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    # initialize model, loss function, optimizer
    model = Model(upscale_factor=UPSCALE_FACTOR).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss() if LOSS_FN == "L1" else nn.MSELoss()

    # training loop
    print("\nStarting Training")
    for epoch in range(EPOCHS):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for lr_patches, hr_patches in progress_bar:
            lr_patches, hr_patches = lr_patches.to(DEVICE), hr_patches.to(DEVICE)
            
            # forward pass, loss computation, backward pass, optimizer step
            optimizer.zero_grad()
            sr_patches = model(lr_patches)
            loss = criterion(sr_patches, hr_patches)
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix(loss=f"{loss.item():.5f}")
    
    print('Finished Training.')
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()
