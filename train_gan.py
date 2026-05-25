import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import cycle
from tqdm import tqdm
import os
import glob # Needed to scan for checkpoints

from config import *
from Architecture.models_original import OmniSR
from Architecture.modification import Discriminator, VGGPerceptualLoss
from utils_data import TrainDataset
# Import evaluation and reporting tools from your other files
from eval import evaluate_model
from run_pipeline import generate_report


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
    generator = OmniSR(upscale_factor=UPSCALE_FACTOR).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    # --- Loss functions ---
    pixel_loss_fn = nn.L1Loss()
    perceptual_loss_fn = VGGPerceptualLoss().to(DEVICE)
    adversarial_loss_fn = nn.BCEWithLogitsLoss()

    # --- Optimizers ---
    opt_G = torch.optim.Adam(generator.parameters(),     lr=GAN_LR, betas=(0.9, 0.99))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=GAN_LR, betas=(0.9, 0.99))

    # --- AUTO-RESUME LOGIC ---
    start_iter = 0
    g_loss_history, d_loss_history = [], []
    
    # Check for existing GAN checkpoints in the GAN folder
    gan_checkpoints = glob.glob(os.path.join(GAN_SAVE_DIR, "gan_iter_*.pth"))
    
    if gan_checkpoints:
        # Sort by iteration number (assumes format gan_iter_XXXX.pth)
        gan_checkpoints.sort(key=lambda x: int(os.path.basename(x).split('_')[2].split('.')[0]))
        latest_gan_ckpt = gan_checkpoints[-1]
        
        checkpoint = torch.load(latest_gan_ckpt, map_location=DEVICE)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        opt_G.load_state_dict(checkpoint["opt_G"])
        opt_D.load_state_dict(checkpoint["opt_D"])
        g_loss_history = checkpoint.get("g_loss_history", [])
        d_loss_history = checkpoint.get("d_loss_history", [])
        start_iter = checkpoint["iter"]
        print(f"✅ Resumed GAN training from {latest_gan_ckpt} at iteration {start_iter}")
    else:
        # No GAN progress found, load the Phase 1 pre-trained weights
        if not os.path.exists(pretrained_generator_path):
            raise FileNotFoundError(f"Phase 1 checkpoint not found: {pretrained_generator_path}")
            
        ckpt = torch.load(pretrained_generator_path, map_location=DEVICE)
        generator.load_state_dict(ckpt["model"])
        print(f"🚀 Starting fresh GAN fine-tuning from Phase 1: {pretrained_generator_path}")

    log_interval = 500
    save_interval = 50000

    # Start loop from start_iter + 1
    progress_bar = tqdm(range(start_iter + 1, GAN_ITERATIONS + 1), desc="GAN Training", ncols=110)

    for iter_idx in progress_bar:
        lr_patches, hr_patches = next(data_iter)
        lr_patches = lr_patches.to(DEVICE)
        hr_patches = hr_patches.to(DEVICE)
        
        # --- Train Discriminator ---
        generator.eval()          
        discriminator.train()

        with torch.no_grad():
            sr_patches = generator(lr_patches)

        real_pred = discriminator(hr_patches)
        fake_pred = discriminator(sr_patches)

        real_labels = torch.ones_like(real_pred)
        fake_labels = torch.zeros_like(fake_pred)

        d_loss_real = adversarial_loss_fn(real_pred - fake_pred.mean().detach(), real_labels)
        d_loss_fake = adversarial_loss_fn(fake_pred - real_pred.mean().detach(), fake_labels)
        d_loss = (d_loss_real + d_loss_fake) * 0.5

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # --- Train Generator ---
        generator.train()
        discriminator.eval()

        sr_patches = generator(lr_patches)
        real_pred = discriminator(hr_patches).detach()
        fake_pred = discriminator(sr_patches)

        l_pixel = pixel_loss_fn(sr_patches, hr_patches)
        l_perceptual = perceptual_loss_fn(sr_patches, hr_patches)

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
    # Manually specify Phase 1 checkpoint
    best_ckpt = "Models/replication/models_5/x2/iter_800000.pth"

    if not os.path.exists(best_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {best_ckpt}")

    print(f"Starting GAN fine-tuning from: {best_ckpt}")
    gen_model, g_loss, d_loss = train_gan(best_ckpt)

    print("\n--- Starting Final Evaluation on Test Datasets ---")
    eval_results = []
    for name in TEST_DATASETS:
        res = evaluate_model(gen_model, name)
        eval_results.append(res)
        print(res)

    generate_report(gen_model, g_loss, eval_results, UPSCALE_FACTOR)