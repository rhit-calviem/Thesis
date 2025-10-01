# IMPORTS
import os
import math
import random
from glob import glob
import numpy as np
import torch.nn.functional as F


import torch
import torchvision.transforms as T
from PIL import Image

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend so plots can be saved without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import *

def _to_y_channel(tensor):
    # Convert RGB tensor (C,H,W) in [0,1] to Y channel tensor - had to look at paper and code to understand this
    pil_img = T.ToPILImage()(tensor.cpu().clamp(0,1))
    y, _, _ = pil_img.convert("YCbCr").split()
    return T.ToTensor()(y)  # shape (1, H, W)

def calculate_psnr(original, compressed):
    # Calculate PSNR between two images (tensors) in [0,1] range only Y channel
    original_y = _to_y_channel(original)
    compressed_y = _to_y_channel(compressed)
    orig_np, comp_np = original_y.numpy(), compressed_y.numpy()
    mse = np.mean((orig_np - comp_np) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(1.0 / math.sqrt(mse))

def calculate_mse(original, compressed):
    # Calculate MSE between two images (tensors) in [0,1] range only Y channel
    original_y = _to_y_channel(original)
    compressed_y = _to_y_channel(compressed)
    return np.mean((original_y.numpy() - compressed_y.numpy()) ** 2)

def calculate_ssim(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    # Calculate SSIM between two images (tensors) in [0,1] range only Y channel
    img1_y = _to_y_channel(img1).unsqueeze(0)  # (1,1,H,W)
    img2_y = _to_y_channel(img2).unsqueeze(0)
    mu1 = F.avg_pool2d(img1_y, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2_y, window_size, stride=1, padding=window_size//2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
    sigma1_sq = F.avg_pool2d(img1_y*img1_y, window_size,1,window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2_y*img2_y, window_size,1,window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1_y*img2_y, window_size,1,window_size//2) - mu1_mu2
    ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    return ssim_map.mean().item()



def visualize_and_save_result(model, dataset_or_path, device, save_path='sr_visualization.png'):
    # Visualiz:
    # - If dataset_or_path is a dataset -> pick random sample.
    # - If dataset_or_path is a string path -> load that image directly.

    model.eval()
    to_pil = T.ToPILImage()

    if isinstance(dataset_or_path, str):  # image path
        # Load image, super-resolve, convert to PIL
        img = Image.open(dataset_or_path).convert("RGB")
        lr_img = img.resize((img.width // UPSCALE_FACTOR, img.height // UPSCALE_FACTOR), Image.BICUBIC)
        lr_tensor = T.ToTensor()(lr_img).unsqueeze(0).to(device)
        with torch.no_grad():
            sr_tensor = model(lr_tensor).clamp(-1.0, 1.0).squeeze(0)

        lr_img = to_pil(lr_tensor.squeeze(0).cpu())
        sr_img = to_pil(sr_tensor.cpu())
        hr_img = img

        filename = os.path.basename(dataset_or_path)

    else:  # dataset case
        # else, pick random sample from dataset, super-resolve, convert to PIL
        idx = random.randint(0, len(dataset_or_path) - 1)
        lr_image_tensor, hr_image_tensor, filename = dataset_or_path[idx]

        lr_tensor = lr_image_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            sr_tensor = model(lr_tensor).clamp(0.0, 1.0).squeeze(0)

        lr_img = to_pil(lr_image_tensor)
        sr_img = to_pil(sr_tensor.cpu())
        hr_img = to_pil(hr_image_tensor)

    # PLot
    images = [lr_img, sr_img, hr_img]
    titles = [
        f'Low-Res Input\n{lr_img.size}',
        f'Model Output\n{sr_img.size}',
        f'Ground Truth\n{hr_img.size}'
    ]

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[img.width for img in images])

    for i, (img, title) in enumerate(zip(images, titles)):
        ax = plt.subplot(gs[i])
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    fig.suptitle(f'SR Result for: {os.path.basename(filename)}', fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)

    

