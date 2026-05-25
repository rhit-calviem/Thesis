import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import os
import time

from config import DEVICE, UPSCALE_FACTOR
from Architecture.models_original import OmniSR
# --- NEW IMPORTS ---
from Architecture.modification import Discriminator, VGGPerceptualLoss


def measure_latency(model, input_tensor, runs=50, warmup=10):
    """Measures average inference latency of the model."""
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    avg_latency = sum(times) / len(times)
    print(f"\nLatency Results:")
    print(f"Average latency: {avg_latency*1000:.3f} ms")
    print(f"FPS: {1/avg_latency:.2f}")
    return avg_latency


def predict_and_compare(lr_path, hr_path, checkpoint_path):
    # 1. Load Generator (OmniSR)
    model = OmniSR(upscale_factor=UPSCALE_FACTOR).to(DEVICE)
    
    # 2. Load Discriminator
    discriminator = Discriminator().to(DEVICE)
    
    # 3. Load Perceptual Loss
    perceptual_loss_fn = VGGPerceptualLoss().to(DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # Handle GAN checkpoint structure
    if "generator" in checkpoint:
        model.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        print(f"Loaded GAN checkpoint from iteration {checkpoint['iter']}")
    else:
        model.load_state_dict(checkpoint)
        print("Warning: Loading Generator weights only. Discriminator remains uninitialized.")

    model.eval()
    discriminator.eval()

    # 4. Load Images
    lr_img = Image.open(lr_path).convert("RGB")
    hr_img = Image.open(hr_path).convert("RGB") # Ground Truth for Loss calculation

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    lr_tensor = transform(lr_img).unsqueeze(0).to(DEVICE)
    hr_tensor = transform(hr_img).unsqueeze(0).to(DEVICE)

    # Measure Latency
    measure_latency(model, lr_tensor)

    # 5. Prediction & Evaluation
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
        
        # Calculate Perceptual Loss
        perc_loss = perceptual_loss_fn(sr_tensor, hr_tensor)
        
        # Get Discriminator score
        # Higher values usually mean the Discriminator thinks it's more "real"
        disc_pred = discriminator(sr_tensor)
        avg_realness = torch.sigmoid(disc_pred).mean().item()

    # 6. Post-process for visualization
    sr_display = (sr_tensor.clamp(-1.0, 1.0).squeeze(0).cpu() * 0.5) + 0.5
    sr_img = T.ToPILImage()(sr_display)
    bicubic_img = lr_img.resize(sr_img.size, Image.BICUBIC)

    # --- OUTPUTS ---
    os.makedirs('Outputs', exist_ok=True)
    sr_img.save('Outputs/building_gan.png')
    
    print(f"\n--- Model Evaluation ---")
    print(f"VGG Perceptual Loss: {perc_loss.item():.6f}")
    print(f"Discriminator Realness Score: {avg_realness:.4f} (1.0 = Perfect)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    axes[0].imshow(lr_img)
    axes[0].set_title(f"Original LR")
    axes[1].imshow(bicubic_img)
    axes[1].set_title("Bicubic Upscale")
    axes[2].imshow(sr_img)
    axes[2].set_title(f"OmniSR (GAN)")

    for ax in axes: ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('Outputs/comparison_view_gan.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    # Point these to valid image pairs in your dataset
    LR_PATH = "data/Urban100/image_SRF_2/img_005_SRF_2_LR.png"
    HR_PATH = "data/Urban100/image_SRF_2/img_005_SRF_2_HR.png"
    CHECKPOINT_PATH = "Models/GAN/gan_iter_200000.pth"

    if os.path.exists(LR_PATH) and os.path.exists(HR_PATH):
        predict_and_compare(LR_PATH, HR_PATH, CHECKPOINT_PATH)
    else:
        print("Error: Missing image paths.")