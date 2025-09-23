#IMPORTS
import os
import random
from glob import glob
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as F
from config import *

# DATA PIPELINE
class TrainDataset(Dataset):
    """
    Dataset for training. Loads HR images, generates LR images by downsampling
    and extracts random patches for training.
    """

    def __init__(self, data_dir):
        super(TrainDataset, self).__init__()
        # Collect all image file paths in the dataset directory
        self.image_files = sorted(glob(f'{data_dir}/*'))
        self.to_tensor = T.ToTensor()  # convert PIL images to PyTorch tensors

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load an HR image and convert it to RGB
        hr_image = Image.open(self.image_files[idx]).convert("RGB")

        # Generate matching LR and HR training patches
        lr_patch, hr_patch = self._get_train_patch(hr_image)

        # Apply data augmentation (random flips) and return
        return self._augment(lr_patch, hr_patch)
    
    def _get_train_patch(self, hr_image):
        hr_w, hr_h = hr_image.size

        if hr_w < PATCH_SIZE or hr_h < PATCH_SIZE:
            # resize so that the image is at least PATCH_SIZE in both dimensions
            new_w = max(hr_w, PATCH_SIZE)
            new_h = max(hr_h, PATCH_SIZE)
            hr_image = F.resize(hr_image, (new_h, new_w), T.InterpolationMode.BICUBIC)

            # take a center crop of exactly PATCH_SIZE
            hr_patch = F.center_crop(hr_image, (PATCH_SIZE, PATCH_SIZE))
        else:
            # Otherwise do a random crop as usual
            top = random.randint(0, hr_h - PATCH_SIZE)
            left = random.randint(0, hr_w - PATCH_SIZE)
            hr_patch = hr_image.crop((left, top, left + PATCH_SIZE, top + PATCH_SIZE))

        # Create the LR patch by downsampling - bicubic downsampling for now
        lr_patch_size = PATCH_SIZE // UPSCALE_FACTOR
        lr_patch = hr_patch.resize((lr_patch_size, lr_patch_size), Image.BICUBIC)

        return self.to_tensor(lr_patch), self.to_tensor(hr_patch)

    def _augment(self, lr_img, hr_img): # data augmentation as done in the paper
        if random.random() > 0.5: lr_img, hr_img = F.hflip(lr_img), F.hflip(hr_img)
        if random.random() > 0.5: lr_img, hr_img = F.vflip(lr_img), F.vflip(hr_img)
        return lr_img, hr_img

class TestDataset(Dataset):
    """
    Dataset for testing. Loads paired LR and HR images from specified data.
    """
    def __init__(self, base_dir, sub_dir):
        super(TestDataset, self).__init__()
        self.data_path = os.path.join(base_dir, sub_dir)

        # Collect all HR file paths, since using set5 for testing right now, the LR names will be inferred
        self.hr_files = sorted(glob(os.path.join(self.data_path, '*_HR.png')))
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        # Paths for HR and corresponding LR images
        hr_path = self.hr_files[idx]
        lr_path = hr_path.replace('_HR.png', '_LR.png')
        
        # Load both HR and LR images as RGB
        hr_image = Image.open(hr_path).convert("RGB")
        lr_image = Image.open(lr_path).convert("RGB")

        # Return paired LR, HR tensors and HR path
        return self.to_tensor(lr_image), self.to_tensor(hr_image), hr_path
