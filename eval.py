import torch
import sys
from tqdm import tqdm

from config import *
from models import Model
from utils_data import get_test_dataset
from utils import calculate_psnr, calculate_mse, calculate_ssim

def main(dataset_name="Set5"):
    print(f"Evaluating on {dataset_name} with model {MODEL_SAVE_PATH}")
    dataset = get_test_dataset(dataset_name)

    model = Model(upscale_factor=UPSCALE_FACTOR).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    total_psnr, total_mse, total_ssim = 0, 0, 0
    with torch.no_grad():
        for lr_image, hr_image, _ in tqdm(dataset, desc="Evaluating"):
            lr_image, hr_image = lr_image.to(DEVICE), hr_image.to(DEVICE)
            sr_image = model(lr_image.unsqueeze(0)).clamp(-1.0, 1.0).squeeze(0)

            total_psnr += calculate_psnr(hr_image, sr_image)
            total_mse += calculate_mse(hr_image, sr_image)
           # total_ssim += calculate_ssim(hr_image.unsqueeze(0), sr_image.unsqueeze(0))
            total_ssim += calculate_ssim(hr_image, sr_image)

    n = len(dataset)
    print(f"\nResults on {dataset_name}:")
    print(f"Average PSNR: {total_psnr/n:.2f} dB")
    print(f"Average MSE : {total_mse/n:.6f}")
    print(f"Average SSIM: {total_ssim/n:.4f}")

if __name__ == "__main__":
    dataset_arg = sys.argv[1] if len(sys.argv) > 1 else "Set5"  # automatically eval on Set5 if not specified
    main(dataset_arg)
