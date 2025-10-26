# import torch.nn as nn

# class Model(nn.Module):
#     def __init__(self, upscale_factor=2):
#         super(Model, self).__init__()
        
#         # Extract features
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        
#         # Upsample using PixelShuffle
#         self.upsample = nn.Sequential(
#             nn.Conv2d(64, 3 * (upscale_factor ** 2), kernel_size=3, padding=1),
#             nn.PixelShuffle(upscale_factor)
#         )
        
#         # Initialize weights
#         self._initialize_weights()

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.upsample(x)
#         return x

#     # will probably move to utils_model.py later if I need more helper functions for the model
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 # initilize conv weights with Kaiming normal
#                 nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#                 if m.bias is not None:
#                     # set biases to zero
#                     nn.init.zeros_(m.bias) 

import torch
import torch.nn as nn

# Squeeze-and-Excitation Layer
class SELayer(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# Depthwise convolution
class DWConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# Pointwise convolution
class PWConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# Local Convolution Block (LCB)
class LocalConvBlock(nn.Module):   # PWConv -> DWConv -> SE -> PWConv + residual
    def __init__(self, channels):
        super().__init__()
        self.pw1 = PWConv(channels, channels)
        self.dw = DWConv(channels)
        self.se = SELayer(channels)
        self.pw2 = PWConv(channels, channels)

    def forward(self, x):
        identity = x                     # save for skip connection
        out = self.pw1(x)
        out = self.dw(out)
        out = self.se(out)
        out = self.pw2(out)
        return out + identity            # skip connection
