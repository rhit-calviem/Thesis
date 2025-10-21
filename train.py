import torch
import torch.nn as nn
from tqdm import tqdm
from itertools import cycle
import os

from config import *
from models import Model
from utils_data import TrainDataset
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast


# Big change: moving to iteration-based training instead of epoch-based


def train_model(resume_path=None):
    print(f"Using device: {DEVICE}")

    # Load dataset
    train_dataset = TrainDataset(HR_TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Initialize model, loss, optimizer, scheduler
    model = Model(upscale_factor=UPSCALE_FACTOR).to(DEVICE)
    criterion = nn.L1Loss() if LOSS_FN == "L1" else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=LR_DROP_ITERS,
        gamma=0.5,
    )
    scaler = GradScaler("cuda")
    autocast_context = lambda: autocast("cuda")


    start_iter = 0
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_iter = checkpoint["iter"]
        print(f"✅ Resumed from {resume_path} at iteration {start_iter}")

    total_iters = NUM_ITERATIONS
    log_interval = 10
    save_interval = 50000

    print(f"Starting training for {total_iters:,} iterations")

    model.train()
    running_loss = 0.0

    data_iter = iter(cycle(train_loader))
    progress_bar = tqdm(range(start_iter + 1, total_iters + 1), desc="Training", ncols=100)

    loss_history = []
    for iter_idx in progress_bar:
        lr_patches, hr_patches = next(data_iter)
        lr_patches, hr_patches = lr_patches.to(DEVICE), hr_patches.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context():
            sr_patches = model(lr_patches)
            loss = criterion(sr_patches, hr_patches)


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item()

        if iter_idx % log_interval == 0:
            avg_loss = running_loss / log_interval
            progress_bar.set_postfix(
                loss=f"{avg_loss:.5f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )
            loss_history.append(avg_loss)
            running_loss = 0.0

        if iter_idx % save_interval == 0 or iter_idx == total_iters:
            ckpt_path = f"{MODEL_SAVE_DIR}/iter_{iter_idx}.pth"
            torch.save(
                {
                    "iter": iter_idx,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                ckpt_path,
            )
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"Training completed. Final model saved to {MODEL_SAVE_PATH}")
    return model, loss_history


if __name__ == "__main__":
    train_model()
