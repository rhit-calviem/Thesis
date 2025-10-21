import os
import random
from glob import glob
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F
from config import *


class TrainDataset(Dataset):
    """Loads HR images, generates LR on the fly by bicubic downsampling, extracts patches."""

    def __init__(self, data_dir):
        super().__init__()
        self.image_files = sorted(glob(f"{data_dir}/*"))
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        hr_image = Image.open(self.image_files[idx]).convert("RGB")
        lr_patch, hr_patch = self._get_train_patch(hr_image)
        lr_patch, hr_patch = self._augment(lr_patch, hr_patch)

        lr_t = self.to_tensor(lr_patch)
        hr_t = self.to_tensor(hr_patch)

        # normalize to [-1,1]
        lr_t = (lr_t - 0.5) / 0.5
        hr_t = (hr_t - 0.5) / 0.5

        return lr_t, hr_t

    def _get_train_patch(self, hr_image):
        hr_w, hr_h = hr_image.size
        if hr_w < PATCH_SIZE or hr_h < PATCH_SIZE:
            new_w = max(hr_w, PATCH_SIZE)
            new_h = max(hr_h, PATCH_SIZE)
            hr_image = F.resize(hr_image, (new_h, new_w), T.InterpolationMode.BICUBIC)

        left = random.randint(0, hr_image.width - PATCH_SIZE)
        top = random.randint(0, hr_image.height - PATCH_SIZE)
        hr_patch = hr_image.crop((left, top, left + PATCH_SIZE, top + PATCH_SIZE))

        lr_size = PATCH_SIZE // UPSCALE_FACTOR
        lr_patch = hr_patch.resize((lr_size, lr_size), Image.Resampling.BICUBIC)
        return lr_patch, hr_patch

    def _augment(self, lr_img, hr_img):
        if random.random() > 0.5:
            lr_img, hr_img = F.hflip(lr_img), F.hflip(hr_img)
        if random.random() > 0.5:
            angle = random.choice([90, 270])
            lr_img, hr_img = F.rotate(lr_img, angle), F.rotate(hr_img, angle)
        return lr_img, hr_img



class TestDataset(Dataset):
    """Your datasets: each has image_SRF_{scale}/ with LR and HR pairs."""

    def __init__(self, base_dir, upscale_factor):
        super().__init__()
        self.base_dir = os.path.join(base_dir, f"image_SRF_{upscale_factor}")
        self.hr_files = sorted(glob(os.path.join(self.base_dir, "*_HR.png")))
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        lr_path = hr_path.replace("_HR.png", "_LR.png")
        hr_image = Image.open(hr_path).convert("RGB")
        lr_image = Image.open(lr_path).convert("RGB")

        lr_t = self.to_tensor(lr_image)
        hr_t = self.to_tensor(hr_image)

        # normalize to [-1,1]
        lr_t = (lr_t - 0.5) / 0.5
        hr_t = (hr_t - 0.5) / 0.5

        return lr_t, hr_t, os.path.basename(hr_path)


def get_test_dataset(name):
    if name not in TEST_DATASETS:
        raise ValueError(f"Unknown dataset {name}. Available: {list(TEST_DATASETS.keys())}")
    base_dir = TEST_DATASETS[name]
    return TestDataset(base_dir, UPSCALE_FACTOR)
