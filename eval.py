import torch
from tqdm import tqdm

from config import *
from models import Model
from utils_data import get_test_dataset
from utils import calculate_psnr, calculate_mse, calculate_ssim


def evaluate_model(model, dataset_name="Set5"):
    print(f"Evaluating on {dataset_name}")
    dataset = get_test_dataset(dataset_name)

    model.eval()
    total_psnr, total_mse, total_ssim = 0, 0, 0
    with torch.no_grad():
        for lr_image, hr_image, _ in tqdm(dataset, desc=f"Eval {dataset_name}"):
            lr_image, hr_image = lr_image.to(DEVICE), hr_image.to(DEVICE)
            sr_image = model(lr_image.unsqueeze(0)).clamp(-1.0, 1.0).squeeze(0)

            # Map from [-1,1] to [0,1]
            sr_image = (sr_image + 1.0) / 2.0
            hr_image = (hr_image + 1.0) / 2.0

            total_psnr += calculate_psnr(hr_image, sr_image)
            total_mse += calculate_mse(hr_image, sr_image)
            total_ssim += calculate_ssim(hr_image, sr_image)

    n = len(dataset)
    return {
        "dataset": dataset_name,
        "psnr": total_psnr / n,
        "mse": total_mse / n,
        "ssim": total_ssim / n,
    }


if __name__ == "__main__":
    model = Model(upscale_factor=UPSCALE_FACTOR).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    for name in TEST_DATASETS:
        result = evaluate_model(model, name)
        print(result)
