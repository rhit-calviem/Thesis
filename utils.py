#IMPORTS
import os
import math
import random
from glob import glob
import numpy as np

import torch
import torchvision.transforms as T

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import from our other files
from config import *


def calculate_psnr(original, compressed):
    """Calculates Peak Signal-to-Noise Ratio."""
    original, compressed = original.cpu().numpy(), compressed.cpu().numpy()
    mse = np.mean((original - compressed) ** 2)
    if mse == 0: 
        return 100
    psnr = 20 * math.log10(1.0 / math.sqrt(mse))
    return psnr

def visualize_and_save_result(model, dataset, device, save_path='sr_visualization.png'):
    """
    Visualizes a random sample, displaying images with their relative actual sizes,
    and saves the plot to a file.
    """
    model.eval()
    idx = random.randint(0, len(dataset) - 1)
    lr_image_tensor, hr_image_tensor, filename = dataset[idx]
    lr_image_batch = lr_image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        sr_image_tensor = model(lr_image_batch).clamp(0.0, 1.0).squeeze(0)

    # Convert tensors to PIL Images to easily get their dimensions
    to_pil = T.ToPILImage()
    lr_img = to_pil(lr_image_tensor)
    sr_img = to_pil(sr_image_tensor.cpu())
    hr_img = to_pil(hr_image_tensor)

    images = [lr_img, sr_img, hr_img]
    titles = [f'Low-Res Input\n{images[0].size}', f'Model Output\n{images[1].size}', f'Ground Truth\n{images[2].size}']

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[lr_img.width, sr_img.width, hr_img.width])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    axes = [ax1, ax2, ax3]

    fig.suptitle(f'SR Result for: {os.path.basename(filename)}', fontsize=16)

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(save_path)
    plt.close(fig)