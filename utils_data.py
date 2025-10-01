#IMPORTS
import os
import random
from glob import glob
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F
from config import *

# Normalization transforms
to_tensor = T.ToTensor()
normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

# DATA PIPELINE
class TrainDataset(Dataset):
    # Dataset for training. Loads HR images, generates LR images by downsampling and extracts random patches for training.

    def __init__(self, data_dir):
        super(TrainDataset, self).__init__()
        # Collect all image file paths in the dataset directory
        self.image_files = sorted(glob(f'{data_dir}/*'))
        self.to_tensor = T.ToTensor()  # convert PIL images to PyTorch tensors

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        hr_image = Image.open(self.image_files[idx]).convert("RGB")

        # Extract aligned HR/LR patches
        lr_patch, hr_patch = self._get_train_patch(hr_image)

        # Data augmentation
        lr_patch, hr_patch = self._augment(lr_patch, hr_patch)

        return lr_patch, hr_patch

    def _get_train_patch(self, hr_image):
        hr_w, hr_h = hr_image.size

        # Ensure the HR image is large enough for the patch size
        if hr_w < PATCH_SIZE or hr_h < PATCH_SIZE:
            new_w = max(hr_w, PATCH_SIZE)
            new_h = max(hr_h, PATCH_SIZE)
            hr_image = F.resize(hr_image, (new_h, new_w), T.InterpolationMode.BICUBIC)
            hr_patch = F.center_crop(hr_image, (PATCH_SIZE, PATCH_SIZE))
        # else, randomly crop a patch
        else:
            top = random.randint(0, hr_h - PATCH_SIZE)
            left = random.randint(0, hr_w - PATCH_SIZE)
            hr_patch = hr_image.crop((left, top, left + PATCH_SIZE, top + PATCH_SIZE))

        # Downsample HR patch to create LR patch
        lr_patch_size = PATCH_SIZE // UPSCALE_FACTOR
        lr_patch = hr_patch.resize((lr_patch_size, lr_patch_size), Image.Resampling.BICUBIC)

        return hr_patch, lr_patch

    def _augment(self, hr_img, lr_img):
        # Random flips
        if random.random() > 0.5:
            hr_img, lr_img = F.hflip(hr_img), F.hflip(lr_img)
        if random.random() > 0.5:
            hr_img, lr_img = F.vflip(hr_img), F.vflip(lr_img)

        # Random 90° rotations
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            hr_img, lr_img = F.rotate(hr_img, angle), F.rotate(lr_img, angle)

        # Convert to tensor & normalize
        return self.to_tensor(lr_img), self.to_tensor(hr_img)


class TestDataset(Dataset):
    # Dataset for testing. Loads paired LR and HR images from specified data.
    
    def __init__(self, base_dir, upscale_factor):
        super(TestDataset, self).__init__()
        # Collect all HR image file paths in the dataset directory
        self.data_path = os.path.join(base_dir, f"image_SRF_{upscale_factor}")
        self.hr_files = sorted(glob(os.path.join(self.data_path, '*_HR.png')))
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        lr_path = hr_path.replace('_HR.png', '_LR.png')
        hr_image = Image.open(hr_path).convert("RGB")
        lr_image = Image.open(lr_path).convert("RGB")
        # Convert to tensor & normalize
        return self.to_tensor(lr_image), self.to_tensor(hr_image), hr_path


def get_test_dataset(name):
    # Loads one of the registered test datasets by name (Set5, Set14, Urban100, BSD100).
    if name not in TEST_DATASETS:
        raise ValueError(f"Unknown dataset {name}. Available: {list(TEST_DATASETS.keys())}")
    base_dir = TEST_DATASETS[name]
    return TestDataset(base_dir, UPSCALE_FACTOR)