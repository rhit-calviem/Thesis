import torch.nn as nn

class Model(nn.Module):
    def __init__(self, upscale_factor=2):
        super(Model, self).__init__()
        
        # Extract features
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        
        # Upsample using PixelShuffle
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 3 * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor)
        )
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.upsample(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)