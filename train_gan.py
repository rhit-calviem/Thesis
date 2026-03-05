import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import cycle
from tqdm import tqdm
import os

from config import *
from models import OmniSR, Discriminator, VGGPerceptualLoss
from utils_data import TrainDataset


def train_gan(pretrained_generator_path: str):
    """
    Phase 2: GAN fine-tuning. Loads a Phase 1 (L1-trained) checkpoint and
    fine-tunes with L1 + Perceptual + Relativistic average GAN loss.
    """
    os.makedirs(GAN_SAVE_DIR, exist_ok=True)

    # --- Datasets ---
    train_dataset = TrainDataset(HR_TRAIN_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    data_iter = iter(cycle(train_loader))

    # --- Models ---
    generator = OmniSR(upscale_factor=UPSCALE_FACTOR, num_osag=NUM_BLOCKS).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    # Load the Phase 1 pre-trained weights — critical for stable GAN training
    ckpt = torch.load(pretrained_generator_path, map_location=DEVICE)
    generator.load_state_dict(ckpt["model"])
    print(f"Loaded Phase 1 generator from: {pretrained_generator_path}")

    # --- Loss functions ---
    pixel_loss_fn = nn.L1Loss()
    perceptual_loss_fn = VGGPerceptualLoss().to(DEVICE)
    adversarial_loss_fn = nn.BCEWithLogitsLoss()

    # --- Optimizers — generator and discriminator must be separate ---
    # Lower LR than Phase 1, betas follow ESRGAN convention
    opt_G = torch.optim.Adam(generator.parameters(),     lr=GAN_LR, betas=(0.9, 0.99))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=GAN_LR, betas=(0.9, 0.99))

    g_loss_history, d_loss_history = [], []
    log_interval = 500
    save_interval = 50000

    progress_bar = tqdm(range(1, GAN_ITERATIONS + 1), desc="GAN Training", ncols=110)

    for iter_idx in progress_bar:
        lr_patches, hr_patches = next(data_iter)
        lr_patches = lr_patches.to(DEVICE)
        hr_patches = hr_patches.to(DEVICE)
        generator.eval()          # freeze BN stats while training D
        discriminator.train()

        with torch.no_grad():
            sr_patches = generator(lr_patches)   # generate fakes (no grad needed for D step)

        real_pred = discriminator(hr_patches)    # D(HR)
        fake_pred = discriminator(sr_patches)    # D(SR)

        # Relativistic average GAN discriminator loss:
        # Real images should score ABOVE the mean fake score
        # Fake images should score BELOW the mean real score
        real_labels = torch.ones_like(real_pred)
        fake_labels = torch.zeros_like(fake_pred)

        d_loss_real = adversarial_loss_fn(real_pred - fake_pred.mean().detach(), real_labels)
        d_loss_fake = adversarial_loss_fn(fake_pred - real_pred.mean().detach(), fake_labels)
        d_loss = (d_loss_real + d_loss_fake) * 0.5

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()
        generator.train()
        discriminator.eval()      # freeze D stats while training G

        sr_patches = generator(lr_patches)       # re-generate with grad

        real_pred = discriminator(hr_patches).detach()   # no grad through D
        fake_pred = discriminator(sr_patches)

        # Pixel loss
        l_pixel = pixel_loss_fn(sr_patches, hr_patches)

        # Perceptual loss (VGG features)
        l_perceptual = perceptual_loss_fn(sr_patches, hr_patches)

        # Relativistic average GAN generator loss (flipped perspective vs D):
        # Generator wants fake to look more real than average real,
        # and real to look less real than average fake
        g_adv_real = adversarial_loss_fn(real_pred - fake_pred.mean(), fake_labels)
        g_adv_fake = adversarial_loss_fn(fake_pred - real_pred.mean(), real_labels)
        l_adv = (g_adv_real + g_adv_fake) * 0.5

        g_loss = l_pixel + LAMBDA_PERCEPTUAL * l_perceptual + LAMBDA_ADV * l_adv

        opt_G.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        opt_G.step()

        if iter_idx % log_interval == 0:
            g_loss_history.append(g_loss.item())
            d_loss_history.append(d_loss.item())
            progress_bar.set_postfix(
                G=f"{g_loss.item():.4f}",
                D=f"{d_loss.item():.4f}",
                pix=f"{l_pixel.item():.4f}",
                perc=f"{l_perceptual.item():.4f}",
                adv=f"{l_adv.item():.4f}",
            )

        if iter_idx % save_interval == 0 or iter_idx == GAN_ITERATIONS:
            ckpt_path = f"{GAN_SAVE_DIR}/gan_iter_{iter_idx}.pth"
            torch.save({
                "iter": iter_idx,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
                "g_loss_history": g_loss_history,
                "d_loss_history": d_loss_history,
            }, ckpt_path)
            print(f"\nSaved GAN checkpoint: {ckpt_path}")

    return generator, g_loss_history, d_loss_history


if __name__ == "__main__":
    import glob
    # Find the best Phase 1 checkpoint automatically
    checkpoints = sorted(
        glob.glob(os.path.join(MODEL_SAVE_DIR, "iter_*.pth")),
        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
    )
    if not checkpoints:
        raise FileNotFoundError("No Phase 1 checkpoint found. Run train.py first.")
    best_ckpt = checkpoints[-1]
    print(f"Starting GAN fine-tuning from: {best_ckpt}")
    train_gan(best_ckpt)