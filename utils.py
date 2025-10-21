# utils.py
# ---------------------------------------------------------------------
# Utility functions for metrics, visualization, and image conversions
# ---------------------------------------------------------------------

import os
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

import matplotlib
matplotlib.use('Agg')  # for headless environments (no GUI)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import UPSCALE_FACTOR


# ---------------------------------------------------------------------
# === Color space conversions ===
# ---------------------------------------------------------------------
def rgb_to_y_channel(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert an RGB image tensor (C,H,W) in [0,1] to a single-channel Y (luminance) tensor.
    Uses ITU-R BT.601 coefficients.
    """
    if tensor.ndim != 3 or tensor.size(0) != 3:
        raise ValueError("Input must be RGB tensor (3,H,W)")

    r = tensor[0:1, ...]
    g = tensor[1:2, ...]
    b = tensor[2:3, ...]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y


# ---------------------------------------------------------------------
# === Evaluation Metrics ===
# ---------------------------------------------------------------------
def calculate_psnr(original: torch.Tensor, compressed: torch.Tensor) -> float:
    """
    Compute PSNR (Peak Signal-to-Noise Ratio) between two [0,1] tensors.
    Only Y channel is used.
    """
    original_y = rgb_to_y_channel(original).cpu().numpy()
    compressed_y = rgb_to_y_channel(compressed).cpu().numpy()
    mse = np.mean((original_y - compressed_y) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse))


def calculate_mse(original: torch.Tensor, compressed: torch.Tensor) -> float:
    """
    Compute MSE (Mean Squared Error) on the Y channel between two [0,1] tensors.
    """
    original_y = rgb_to_y_channel(original).cpu().numpy()
    compressed_y = rgb_to_y_channel(compressed).cpu().numpy()
    return np.mean((original_y - compressed_y) ** 2)


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size=11, C1=0.01**2, C2=0.03**2) -> float:
    """
    Compute SSIM (Structural Similarity Index) on Y channel.
    Implementation uses a sliding window with mean and variance pooling.
    """
    img1_y = rgb_to_y_channel(img1).unsqueeze(0)  # (1,1,H,W)
    img2_y = rgb_to_y_channel(img2).unsqueeze(0)

    mu1 = F.avg_pool2d(img1_y, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2_y, window_size, stride=1, padding=window_size // 2)

    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = F.avg_pool2d(img1_y * img1_y, window_size, 1, window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2_y * img2_y, window_size, 1, window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1_y * img2_y, window_size, 1, window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()


# ---------------------------------------------------------------------
# === Visualization Helpers ===
# ---------------------------------------------------------------------
def visualize_and_save_result(model, dataset_or_path, device, save_path='sr_visualization.png'):
    """
    Visualize SR model output.
    - If dataset_or_path is a dataset: picks random sample.
    - If it's a string path: loads and processes that image.
    """
    model.eval()
    to_pil = T.ToPILImage()

    if isinstance(dataset_or_path, str):  # custom image path
        img = Image.open(dataset_or_path).convert("RGB")
        lr_img = img.resize((img.width // UPSCALE_FACTOR, img.height // UPSCALE_FACTOR), Image.BICUBIC)
        lr_tensor = T.ToTensor()(lr_img).unsqueeze(0).to(device)
        with torch.no_grad():
            sr_tensor = model(lr_tensor).clamp(-1.0, 1.0).squeeze(0)
        sr_tensor = (sr_tensor + 1.0) / 2.0  # [-1,1] → [0,1]

        lr_img = to_pil(lr_tensor.squeeze(0).cpu())
        sr_img = to_pil(sr_tensor.cpu())
        hr_img = img
        filename = os.path.basename(dataset_or_path)

    else:  # dataset case
        idx = random.randint(0, len(dataset_or_path) - 1)
        lr_image_tensor, hr_image_tensor, filename = dataset_or_path[idx]

        lr_tensor = lr_image_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            sr_tensor = model(lr_tensor).clamp(-1.0, 1.0).squeeze(0)
        sr_tensor = (sr_tensor + 1.0) / 2.0
        hr_image_tensor = (hr_image_tensor + 1.0) / 2.0
        lr_image_tensor = (lr_image_tensor + 1.0) / 2.0

        to_pil = T.ToPILImage()
        lr_img = to_pil(lr_image_tensor)
        sr_img = to_pil(sr_tensor.cpu())
        hr_img = to_pil(hr_image_tensor)

    # --- Plot ---
    images = [lr_img, sr_img, hr_img]
    titles = [
        f'Low-Res Input\n{lr_img.size}',
        f'Model Output\n{sr_img.size}',
        f'Ground Truth\n{hr_img.size}'
    ]

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[img.width for img in images])

    for i, (im, title) in enumerate(zip(images, titles)):
        ax = plt.subplot(gs[i])
        ax.imshow(im)
        ax.set_title(title)
        ax.axis('off')

    fig.suptitle(f'SR Result: {os.path.basename(filename)}', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)


def visualize_sample(model, dataset_name="Set5", device="cpu"):
    """
    Return a matplotlib Figure for a random visualization sample.
    """
    from utils_data import get_test_dataset

    dataset = get_test_dataset(dataset_name)
    idx = random.randint(0, len(dataset) - 1)
    lr_image_tensor, hr_image_tensor, filename = dataset[idx]

    lr_tensor = lr_image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        sr_tensor = model(lr_tensor).clamp(-1.0, 1.0).squeeze(0)
    sr_tensor = (sr_tensor + 1.0) / 2.0
    hr_image_tensor = (hr_image_tensor + 1.0) / 2.0
    lr_image_tensor = (lr_image_tensor + 1.0) / 2.0

    to_pil = T.ToPILImage()
    lr_img = to_pil(lr_image_tensor)
    sr_img = to_pil(sr_tensor.cpu())
    hr_img = to_pil(hr_image_tensor)

    images = [lr_img, sr_img, hr_img]
    titles = ["Low-Res Input", "Super-Resolved", "Ground Truth"]

    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    for i, (im, title) in enumerate(zip(images, titles)):
        ax = plt.subplot(gs[i])
        ax.imshow(im)
        ax.set_title(title)
        ax.axis('off')

    fig.suptitle(f"{dataset_name} Sample — {filename}", fontsize=12)
    plt.tight_layout()
    return fig
