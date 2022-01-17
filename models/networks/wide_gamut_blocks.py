import torch
import torch.nn as nn
import torch.nn.functional as F
from .deep_wb_blocks import DoubleConvBlock


def safe_depth_cat(x1, x2):
    """Safely concatenate given tensors along depth (i.e. channel) axis"""
    # https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    # input is (Batch, Channel, Height, Width)
    dy = x2.size()[2] - x1.size()[2]
    dx = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
    return torch.cat([x2, x1], dim=1)


class SafeUpBlock(nn.Module):
    """Upscale block: double conv block -> transpose conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConvBlock(in_channels * 2, in_channels)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        return torch.relu(self.up(self.conv(safe_depth_cat(x1, x2))))


class SafeOutputBlock(nn.Module):
    """Output block: double conv block -> output conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_conv = nn.Sequential(
            DoubleConvBlock(in_channels * 2, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x1, x2):
        return self.out_conv(safe_depth_cat(x1, x2))
