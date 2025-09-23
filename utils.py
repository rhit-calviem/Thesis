# IMPORTS
import os
import math
import random
from glob import glob
import numpy as np

import torch
import torchvision.transforms as T

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend so plots can be saved without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import *

# UTILITY FUNCTIONS
def calculate_psnr(original, compressed):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images, which measures
    the quality of the outputimage compared to the original.
    """
    # Convert tensors to numpy arrays
    original, compressed = original.cpu().numpy(), compressed.cpu().numpy()
    
    # Calculate MSE
    mse = np.mean((original - compressed) ** 2)
    
    if mse == 0: 
        return 100
    psnr = 20 * math.log10(1.0 / math.sqrt(mse))
    return psnr


def visualize_and_save_result(model, dataset, device, save_path='sr_visualization.png'):
    """
    Visualizes a random example from the dataset, comparing:
      - The low-resolution input
      - The model's super-resolved output
      - The ground truth high-resolution image
    """
    model.eval() 

    # Get random image
    idx = random.randint(0, len(dataset) - 1)
    lr_image_tensor, hr_image_tensor, filename = dataset[idx]

    # Run image through model
    lr_image_batch = lr_image_tensor.unsqueeze(0).to(device)  # add batch dimension
    with torch.no_grad():
        sr_image_tensor = model(lr_image_batch).clamp(0.0, 1.0).squeeze(0)  # clamp to [0,1] range

    # Convert tensors to PIL images
    to_pil = T.ToPILImage()
    lr_img = to_pil(lr_image_tensor)
    sr_img = to_pil(sr_image_tensor.cpu())
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
