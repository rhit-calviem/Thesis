import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import *
from models import Model
from utils_data import TrainDataset
from torch.utils.data import DataLoader

# Training function
def train_model():
    print(f"Using device: {DEVICE}")
    # load training data
    train_dataset = TrainDataset(HR_TRAIN_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    # initialize model, loss function, optimizer
    model = Model(upscale_factor=UPSCALE_FACTOR).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss() if LOSS_FN == "L1" else nn.MSELoss()

    loss_history = []

    print("\nStarting Training")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for lr_patches, hr_patches in progress_bar:
            lr_patches, hr_patches = lr_patches.to(DEVICE), hr_patches.to(DEVICE)
            optimizer.zero_grad()
            sr_patches = model(lr_patches)
            loss = criterion(sr_patches, hr_patches)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.5f}")
        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.5f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    return model, loss_history

if __name__ == '__main__':
    train_model()