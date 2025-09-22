import torch.nn as nn

class PlaceholderSRModel(nn.Module):
    """
    A very small, lightweight placeholder model for Super-Resolution.
    It's designed to be simple and fast, perfect for verifying the data pipeline.
    """
    def __init__(self, upscale_factor=2):
        super(PlaceholderSRModel, self).__init__()
        
        # A single convolutional layer for feature extraction
        self.feature_extractor = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        
        # The upsampling block using the efficient PixelShuffle layer
        self.upsampler = nn.Sequential(
            nn.Conv2d(64, 64 * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.ReLU(inplace=True)
        )
        
        # The final layer to reconstruct the 3-channel RGB image
        self.reconstructor = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        features = self.feature_extractor(x)
        upscaled = self.upsampler(features)
        reconstructed = self.reconstructor(upscaled)
        return reconstructed