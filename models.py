import torch
import torch.nn as nn
from config import NUM_BLOCKS as num_osag

# Squeeze-and-Excitation Layer
class SELayer(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    
    '''
    y = self.avg_pool(x).view(b, c) - This step "squeezes" all the spatial information from the entire feature map (H x W) into a single value for each channel.
    it takes the input tensor x (shape [b, c, h, w]) and calculates the global average for each of the c channels making the output tensor of shape [b, c, 1, 1].
    .view(b, c) then reshapes this into a 2D tensor of shape [b, c]. This vector y is now a "channel descriptor" that represents the global information for each channel.

    FC
    First nn.Linear (Reduction): The vector y (shape [b, c]) is passed through the first linear layer, which reduces its dimension from in_channels to in_channels // reduction. 
    This is a "bottleneck" that saves computation and helps in learning a more generalized relationship.
    nn.ReLU (Non-linearity): This activation function allows the model to learn a non-linear interaction between the channels.
    Second nn.Linear (Expansion): The vector is passed through the second linear layer, which expands the dimension back up to the original in_channels.
    nn.Sigmoid (Gating): The final sigmoid activation squashes the output values for each channel to be in the range of 0 to 1.
    This output y (now shape [b, c]) represents the learned "importance" for each channel. A value of 1.0 means "very important," and 0.0 means "not important."

    .view(b, c, 1, 1) - This final step applies the learned "importance" weights to the original feature map.
    The weights vector y (shape [b, c]) is first reshaped back to [b, c, 1, 1].
    This y tensor is then multiplied with the original input tensor x (shape [b, c, h, w]).
    Due to broadcasting, each entire channel in x is multiplied by its corresponding single importance weight from y.
    The result is a recalibrated feature map where the channels that the fc block deemed important are preserved or enhanced, and the channels deemed unimportant are diminished.
    '''


# Depthwise convolution
class DWConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# Pointwise convolution or 1x1 convolution
class PWConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# Local Convolution Block (LCB)
class LocalConvBlock(nn.Module): # PWConv -> DWConv -> SE -> PWConv + skip
    def __init__(self, channels):
        super().__init__()
        self.pw1 = PWConv(channels, channels)
        self.dw = DWConv(channels)
        self.se = SELayer(channels)
        self.pw2 = PWConv(channels, channels)

    def forward(self, x):
        identity = x # save for skip connection
        out = self.pw1(x)
        out = self.dw(out)
        out = self.se(out)
        out = self.pw2(out)
        return out + identity # skip connection

# OSAG
class OSAG(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.lcb = LocalConvBlock(channels)

    def forward(self, x):
        identity = x  # save for skip connection
        block = self.lcb(x)
        return block + identity


# Omni-SR model
class OmniSR(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, channels=64, upscale_factor=4, num_osag=5):
        super().__init__()

        self.shallow = nn.Conv2d(in_channels, channels, 3, 1, 1) # Shallow feature extraction

        self.osag_blocks = nn.Sequential(*[OSAG(channels) for _ in range(num_osag)]) # Deep feature extraction

        self.conv_agg = nn.Conv2d(channels, channels, 3, 1, 1) # Feature aggregation 3x3 conv in cascading manner after OSAG blocks

        self.reconstruction = nn.Sequential( # Reconstruction
            nn.Conv2d(channels, channels * (upscale_factor ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale_factor), # pixel shuffle for upscaling
            nn.Conv2d(channels, out_channels, 3, 1, 1)
        )

    def forward(self, x):
        x0 = self.shallow(x)
        x_deep = self.osag_blocks(x0)
        x_deep = self.conv_agg(x_deep)
        x_fused = x0 + x_deep
        out = self.reconstruction(x_fused)
        return out
    
    '''
    PixelShuffle(upscale_factor) - This layer rearranges elements in a tensor of shape (C * r^2, H, W) to a tensor of shape (C, H * r, W * r), where r is the upscale factor.
    For this model, it takes the output from the preceding convolutional layer, which has been expanded to have channels equal to channels * (upscale_factor ** 2), 
    and rearranges these channels into a higher resolution spatial grid to represent the higher resolution image.
    '''