import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_BLOCKS as num_osag
import torchvision.models as tvm

class CoordinateAttention(nn.Module):    
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        reduced_channels = max(8, in_channels // reduction)  # Ensure at least 8 channels
        
        # Adaptive pooling along height and width directions, These pool along one spatial dimension while preserving the other
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        # Shared transformation to encode coordinate information, input will be concatenated H and W pooled features
        self.conv1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(reduced_channels)
        self.act = nn.SiLU()
        
        # Separate convolutions to generate attention weights for height and width
        self.conv_h = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, stride=0)
        
    def forward(self, x):
        identity = x  # Save input for final multiplication
        
        batch_size, channels, height, width = x.size()
        
        # Coordinate Information Embedding
        x_h = self.pool_h(x) # Pool along width direction to get height-wise global context
        x_w = self.pool_w(x) # Pool along height direction to get width-wise global context
        
        # Transpose x_w to align with x_h for concatenation
        x_w = x_w.permute(0, 1, 3, 2)
        
        # Concatenate along the width dimension to combine height and width coordinate information
        y = torch.cat([x_h, x_w], dim=2)
        
        # Shared encoding transformation - Reduce channels and apply non-linearity to learn coordinate attention
        y = self.conv1(y) 
        y = self.bn1(y)
        y = self.act(y)
        
        # Split back into height and width components
        x_h, x_w = torch.split(y, [height, width], dim=2)
        
        # Transpose x_w back to original orientation
        x_w = x_w.permute(0, 1, 3, 2)
        
        # Generate attention weights for each direction and then apply sigmoid to get attention weights in range [0, 1]
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        # Apply coordinate attention to input and combine with original input
        out = identity * a_h * a_w
        return out
    

class VGGPerceptualLoss(nn.Module):
    """
    Computes perceptual loss using VGG19 features.
    Uses features before relu4_4 (conv4_4), consistent with ESRGAN's 'VGG54' convention.
    """
    def __init__(self):
        super().__init__()
        vgg = tvm.vgg19(weights=tvm.VGG19_Weights.DEFAULT).features

        # Layer 26 = relu4_4 in VGG19. Using [:27] captures up to and including it.
        # ESRGAN uses features BEFORE activation (layer 25, the raw conv output).
        # Using 25 here follows the ESRGAN paper exactly.
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:25]).eval()

        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # VGG stays frozen forever

        # VGG19 was trained on ImageNet with these normalization values
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        # Your pipeline uses [-1, 1]. VGG expects ImageNet-normalized [0, 1].
        sr_norm = (sr * 0.5 + 0.5 - self.mean) / self.std
        hr_norm = (hr * 0.5 + 0.5 - self.mean) / self.std
        return F.l1_loss(self.feature_extractor(sr_norm),
                         self.feature_extractor(hr_norm))


class Discriminator(nn.Module):
    """
    PatchGAN discriminator. Outputs a spatial grid of real/fake predictions
    rather than a single scalar — each output value covers a receptive field
    patch of the input. This is better for SR than a global discriminator
    because local texture quality matters more than global image judgement.
    """
    def __init__(self, in_channels=3, base_ch=64):
        super().__init__()

        def conv_block(in_c, out_c, stride, norm=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=not norm)]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # No BatchNorm on first layer — standard practice
            *conv_block(in_channels, base_ch,    stride=2, norm=False),   # -> H/2
            *conv_block(base_ch,     base_ch*2,  stride=2),               # -> H/4
            *conv_block(base_ch*2,   base_ch*4,  stride=2),               # -> H/8
            *conv_block(base_ch*4,   base_ch*8,  stride=1),               # -> H/8 (stride 1 here)
            nn.Conv2d(base_ch*8, 1, kernel_size=4, padding=1)            # -> patch output map
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)  # Standard GAN init
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)