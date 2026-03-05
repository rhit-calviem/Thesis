import torch
import torch.nn as nn
import torch.nn.functional as F
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
    
class OSA(nn.Module): # Omni Self-Attention (OSA) block.
# I followed very closesly section 3.2 and Eqs. (1)–(5) in the paper.
# I implemented the following steps as described in the paper:
#   1. Spatial self-attention  (Eq. 1–2)
#   2. Rotation to channel axis (R operator)
#   3. Channel self-attention  (Eq. 3–4)
#   4. Inverse rotation (R^{-1}) to restore shape (Eq. 5)

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for Q_s, K_s, V_s
        self.qkv = nn.Linear(dim, dim * 3)

        # Final projection after channel-attention output
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape  # x = (B * nW, HW, C)

        # Spatial Self-Attention (Left half of Fig. 3)
        # Produce Q_s, K_s, V_s   (HW × C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        Q_s = qkv[:, :, 0].permute(0, 2, 1, 3)  # (B, h, HW, d)
        K_s = qkv[:, :, 1].permute(0, 2, 1, 3)  # (B, h, HW, d)
        V_s = qkv[:, :, 2].permute(0, 2, 1, 3)  # (B, h, HW, d)

        # Spatial attention map: Softmax(Q_s K_s^T) <- (HW×HW)
        # Diagram: Q_s x K_s → S
        attn_s = (Q_s @ K_s.transpose(-1, -2)) * self.scale
        attn_s = attn_s.softmax(dim=-1)

        # Spatial aggregation: Y_s = (SpatialAttn) x V_s
        Y_s = attn_s @ V_s # do I ever use this?

        # Rotation, rotate Q', K', V' from HW×C to C×HW
        Q_c = Q_s.transpose(-1, -2)    # (B, h, d, HW)
        K_c = K_s.transpose(-1, -2)    # (B, h, d, HW)
        V_c = Y_s.transpose(-1, -2)    # (B, h, d, HW)

        # Channel Self-Attention  (Right half of Fig. 3)
        # Channel attn map: Softmax(K_c Q_c^T) <- (C×C)
        # Diagram: K_c x Q_c -> S
        attn_c = (K_c @ Q_c.transpose(-1, -2)) * self.scale
        attn_c = attn_c.softmax(dim=-1)

        # Channel aggregation: Y_c (C×HW × HW×C)
        y_c = attn_c @ V_c # (B, h, d, HW)

        # Rotation inverse, rotate Y_c back to HW×C
        y_c = y_c.transpose(-1, -2)

        # Merge heads and project
        y_osa = y_c.permute(0, 2, 1, 3).reshape(B, N, C)
        y_osa = self.proj_out(y_osa)

        # Only issue I had while implementing this is that it is not specified 
        # whther Yc is used directly as the output replacement of Vs or if it 
        # is passed through proj_out and summed to Yc at the final output.
        return y_osa

    
class GDFN(nn.Module):
    # Gated Depthwise Feed-Forward Network - looked at his paper "Restormer: Efficient transformer for high-resolution image restoration"
    # for information about it and its structure
    def __init__(self, dim, expansion_factor=2.66):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_dim * 2, 1, bias=False)
        self.dw = DWConv(hidden_dim*2)
        self.project_out = nn.Conv2d(hidden_dim, dim, 1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dw(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2   # gating mechanism - initially used relu, but the Restromer paper and the OSAG paper also use gelu
        x = self.project_out(x)
        return x
    
# Window partitions and reverses for both MESO and GLOBAL OSA blocks, they differ slightly in how they partition the feature maps and reconstruct them.
# For MESO-level OSA   
def window_partition(x, window_size): # Takes feature map (B, C, H, W) and splits it into non-overlapping windows. Each window becomes a flattened sequence of size (window_size^2, C).
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size) # reshape into 6d tensor with 2 extra dims for windowing
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous() # Move window dims next to each other, channels last
    windows = windows.view(-1, window_size * window_size, C) # Flatten spatial window (P,P) → token sequence P*P
    return windows

def window_reverse(windows, window_size, H, W, B): # Reconstruct feature map from flattened windows. This reverses the earlier method, complete mirror basically
    C = windows.shape[-1]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(B, C, H, W)
    return x

# For Global OSA
def grid_partition(x, grid_size): # Partitions feature map (B, C, H, W) into non-overlapping grids. Each grid becomes a flattened sequence of size (h_step * w_step, C).
    B, C, H, W = x.shape
    # assert H % grid_size == 0 and W % grid_size == 0 # Ensure divisibility
    h_step = H // grid_size
    w_step = W // grid_size
    x = x.view(B, C, grid_size, h_step, grid_size, w_step)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous() # Move grid indices next to batch
    tokens = x.view(-1, h_step * w_step, C)
    return tokens

def grid_reverse(tokens, grid_size, H, W, B): # Reconstruct feature map from flattened grids, this is a mirror of the earlier method.
    C = tokens.shape[-1]
    h_step = H // grid_size
    w_step = W // grid_size
    x = tokens.view(B, grid_size, grid_size, h_step, w_step, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(B, C, H, W)
    return x


class MesoOSABlock(nn.Module):
    # Meso-scale Omni Self-Attention block, it performs attention inside non-overlapping windows, matching the mid-range receptive field stage in Section 3.2.
    def __init__(self, dim, window_size=8, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.osa = OSA(dim, num_heads) # self attention

        self.norm2 = nn.LayerNorm(dim)
        self.gdfn = GDFN(dim) # gated depthwise FFN

    def forward(self, x):
        B, C, H, W = x.shape
        skip = x

        # Pad so that H and W are divisible by window_size (paper assumes this for meso-scale windows)
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        _, _, H_pad, W_pad = x.shape

        # Split feature map into windows for OSA
        x_windows = window_partition(x, self.window_size)

        # Normalize and apply Omni Self-Attention inside each window
        x_windows = self.norm1(x_windows)
        x_windows = self.osa(x_windows)

        # Rebuild the spatial feature map
        x = window_reverse(x_windows, self.window_size, H_pad, W_pad, B)

        # Remove the padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]

        # First residual connection
        x = skip + x

        # Feed-forward
        skip = x
        x_ln = x.permute(0, 2, 3, 1).contiguous()
        x_ln = self.norm2(x_ln)
        x_ln = x_ln.permute(0, 3, 1, 2).contiguous()

        x = self.gdfn(x_ln)

        # Second residual connection
        x = skip + x
        return x
    
class GlobalOSABlock(nn.Module):
    def __init__(self, dim, grid_size=8, num_heads=4):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size

        self.norm1 = nn.LayerNorm(dim)
        self.osa = OSA(dim, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.gdfn = GDFN(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        skip = x

        # Pad if needed
        pad_h = (self.grid_size - H % self.grid_size) % self.grid_size
        pad_w = (self.grid_size - W % self.grid_size) % self.grid_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        _, _, H_pad, W_pad = x.shape

        # Global grid partition
        x_tokens = grid_partition(x, self.grid_size)

        # Global OSA
        x_tokens = self.norm1(x_tokens)
        x_tokens = self.osa(x_tokens)

        # Reverse grid
        x = grid_reverse(x_tokens, self.grid_size, H_pad, W_pad, B)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]

        # First residual
        x = skip + x

        # FFN
        skip = x
        x_ln = x.permute(0, 2, 3, 1).contiguous()
        x_ln = self.norm2(x_ln)
        x_ln = x_ln.permute(0, 3, 1, 2).contiguous()

        x = self.gdfn(x_ln)

        # Second residual
        x = skip + x
        return x
    
class ESA(nn.Module): # ESA block taken from the "Residual Local Feature Network for Efficient Super-Resolution" paper, there is a really good image that i used to implement this block.
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced_channels = channels // reduction
        self.conv1 = nn.Conv2d(channels, reduced_channels, 1)
        self.strided_conv = nn.Conv2d(reduced_channels, reduced_channels, 3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=3, padding=3)
        self.conv_groups = nn.Conv2d(reduced_channels, reduced_channels, 3, padding=1, groups=reduced_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(reduced_channels, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        residual = out 
        out = self.strided_conv(out)
        out = self.maxpool(out)
        out = self.conv_groups(out)
        out = self.relu(out)
        out = F.interpolate(out, size=residual.shape[2:], mode='bilinear', align_corners=False)
        out = out + residual 
        out = self.conv2(out)
        out = self.sigmoid(out)
        return identity * out
 

# OSAG
class OSAG(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.lcb = LocalConvBlock(channels)
        self.meso = MesoOSABlock(channels)
        self.glob = GlobalOSABlock(channels)
        self.esa = ESA(channels)

    def forward(self, x):
        identity = x  # save for skip connection
        x = self.lcb(x)
        x = self.meso(x)
        x = self.glob(x)
        x = self.esa(x)
        return x + identity # add skip connection

# Omni-SR model
class OmniSR(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, channels=64, upscale_factor=2, num_osag=1):
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