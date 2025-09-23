#IMPORTS
import os
import random
from glob import glob
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as F

# Import from our other files
from config import *

# DATA PIPELINE
class TrainDataset(Dataset):
    """
    Dataset for training. Loads HR images, generates LR images on-the-fly by downsampling
    and extracts random patches for training.
    """

    def __init__(self, data_dir):
        super(TrainDataset, self).__init__()
        self.image_files = sorted(glob(f'{data_dir}/*'))
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        hr_image = Image.open(self.image_files[idx]).convert("RGB")
        lr_patch, hr_patch = self._get_train_patch(hr_image)
        return self._augment(lr_patch, hr_patch)
    
    def _get_train_patch(self, hr_image):
        hr_w, hr_h = hr_image.size
        if hr_w < PATCH_SIZE or hr_h < PATCH_SIZE:
             hr_image = F.resize(hr_image, (PATCH_SIZE, PATCH_SIZE), T.InterpolationMode.BICUBIC)
             hr_w, hr_h = hr_image.size
        
        top = random.randint(0, hr_h - PATCH_SIZE)
        left = random.randint(0, hr_w - PATCH_SIZE)
        hr_patch = hr_image.crop((left, top, left + PATCH_SIZE, top + PATCH_SIZE))
        
        lr_patch_size = PATCH_SIZE // UPSCALE_FACTOR
        lr_patch = hr_patch.resize((lr_patch_size, lr_patch_size), Image.BICUBIC)
        
        return self.to_tensor(lr_patch), self.to_tensor(hr_patch)

    def _augment(self, lr_img, hr_img):
        if random.random() > 0.5: lr_img, hr_img = F.hflip(lr_img), F.hflip(hr_img)
        if random.random() > 0.5: lr_img, hr_img = F.vflip(lr_img), F.vflip(hr_img)
        return lr_img, hr_img

class TestDataset(Dataset):
    """
    Dataset for testing. Loads paired LR and HR images from specified directories.
    """
    def __init__(self, base_dir, sub_dir):
        super(TestDataset, self).__init__()
        self.data_path = os.path.join(base_dir, sub_dir)
        # Find all the High-Resolution images
        self.hr_files = sorted(glob(os.path.join(self.data_path, '*_HR.png')))
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        lr_path = hr_path.replace('_HR.png', '_LR.png')
        
        hr_image = Image.open(hr_path).convert("RGB")
        lr_image = Image.open(lr_path).convert("RGB")

        return self.to_tensor(lr_image), self.to_tensor(hr_image), hr_path
