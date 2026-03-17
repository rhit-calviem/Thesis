import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import os
import time

from config import DEVICE, UPSCALE_FACTOR
from models import OmniSR


def measure_latency(model, input_tensor, runs=50, warmup=10):
    """
    Measures average inference latency of the model.
    """

    # Warmup runs (important for GPU)
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


def predict_and_compare(image_path, model_path):

    # 1. Load Model
    model = OmniSR(upscale_factor=UPSCALE_FACTOR).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 2. Load Image
    img = Image.open(image_path).convert("RGB")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    lr_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # -------------------------
    # LATENCY MEASUREMENT
    # -------------------------
    measure_latency(model, lr_tensor)

    # 3. Model Prediction
    with torch.no_grad():
        sr_tensor = model(lr_tensor).clamp(-1.0, 1.0).squeeze(0).cpu()

    # 4. Post-process
    sr_tensor = (sr_tensor * 0.5) + 0.5
    sr_img = T.ToPILImage()(sr_tensor)

    # 5. Bicubic baseline
    bicubic_img = img.resize(sr_img.size, Image.BICUBIC)

    # --- SAVE OUTPUTS ---
    os.makedirs('Outputs', exist_ok=True)

    sr_img.save('Outputs/prediction_only.png')
    print("Saved pure prediction to Outputs/prediction_only.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 10))

    axes[0].imshow(img)
    axes[0].set_title("Original LR")
    axes[0].axis('off')

    axes[1].imshow(bicubic_img)
    axes[1].set_title("Bicubic Upscale (Base)")
    axes[1].axis('off')

    axes[2].imshow(sr_img)
    axes[2].set_title("OmniSR Prediction")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('Outputs/comparison_view.png', dpi=300)
    plt.close()

    print("Saved comparison plot to Outputs/comparison_view.png")


if __name__ == "__main__":

    INPUT_IMAGE_PATH = "data/Urban100/image_SRF_2/img_062_SRF_2_LR.png"
    CHECKPOINT_PATH = "Models/replication/models_5/x2/iter_800000.pth"

    if os.path.exists(INPUT_IMAGE_PATH):
        predict_and_compare(INPUT_IMAGE_PATH, CHECKPOINT_PATH)
    else:
        print(f"Error: Could not find image at {INPUT_IMAGE_PATH}")