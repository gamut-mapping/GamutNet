from abc import ABC, abstractmethod

import torch
from torch import nn

from .deep_wb_blocks import (DoubleConvBlock, DownBlock, BridgeDown, BridgeUP)
from .wide_gamut_blocks import SafeUpBlock, SafeOutputBlock  # Concatenation-safe


class BaseWideGamutNet(nn.Module, ABC):
    def __init__(self, n_in_channels, using_residual=True, limiting_output_range=False):
        super(BaseWideGamutNet, self).__init__()
        self.n_in_channels = n_in_channels
        self.n_out_channels = 3  # (R, G, B)
        self.using_residual = using_residual
        self.limiting_output_range = limiting_output_range

    @abstractmethod
    def _forward(self, x):
        raise NotImplementedError

    def forward(self, x):
        x0 = x[:, :3, :, :]  # drop the mask channel
        if not self.using_residual:
            x0 = torch.zeros(x0.shape)
        out = self._forward(x) + x0  # residual learning
        if self.limiting_output_range:
            out = torch.sigmoid(out)
        return out


class WideGamutNet(BaseWideGamutNet):
    def __init__(self, n_in_channels, using_residual=True, limiting_output_range=False):
        super(WideGamutNet, self).__init__(n_in_channels, using_residual, limiting_output_range)
        # Contracting
        self.encoder_inc = DoubleConvBlock(self.n_in_channels, 24)
        self.encoder_down1 = DownBlock(24, 48)
        self.encoder_down2 = DownBlock(48, 96)
        self.encoder_down3 = DownBlock(96, 192)
        self.encoder_bridge_down = BridgeDown(192, 384)
        # Expanding
        self.decoder_bridge_up = BridgeUP(384, 192)
        self.decoder_up1 = SafeUpBlock(192, 96)  # Concatenation-safe
        self.decoder_up2 = SafeUpBlock(96, 48)  # Concatenation-safe
        self.decoder_up3 = SafeUpBlock(48, 24)  # Concatenation-safe
        self.decoder_out = SafeOutputBlock(24, self.n_out_channels)  # Concatenation-safe

    def _forward(self, x):
        # Contracting
        x1 = self.encoder_inc(x)
        x2 = self.encoder_down1(x1)
        x3 = self.encoder_down2(x2)
        x4 = self.encoder_down3(x3)
        x5 = self.encoder_bridge_down(x4)
        # Expanding
        x = self.decoder_bridge_up(x5)
        x = self.decoder_up1(x, x4)
        x = self.decoder_up2(x, x3)
        x = self.decoder_up3(x, x2)
        x = self.decoder_out(x, x1)
        return x


class SmallWideGamutNet(BaseWideGamutNet):
    def __init__(self, n_in_channels, using_residual=True, limiting_output_range=False):
        super(SmallWideGamutNet, self).__init__(n_in_channels, using_residual, limiting_output_range)
        # Contracting
        self.encoder_inc = DoubleConvBlock(self.n_in_channels, 24)
        self.encoder_down1 = DownBlock(24, 48)
        self.encoder_down2 = DownBlock(48, 96)
        self.encoder_bridge_down = BridgeDown(96, 192)
        # Expanding
        self.decoder_bridge_up = BridgeUP(192, 96)
        self.decoder_up1 = SafeUpBlock(96, 48)  # Concatenation-safe
        self.decoder_up2 = SafeUpBlock(48, 24)  # Concatenation-safe
        self.decoder_out = SafeOutputBlock(24, self.n_out_channels)  # Concatenation-safe

    def _forward(self, x):
        # Contracting
        x1 = self.encoder_inc(x)
        x2 = self.encoder_down1(x1)
        x3 = self.encoder_down2(x2)
        x4 = self.encoder_bridge_down(x3)
        # Expanding
        x = self.decoder_bridge_up(x4)
        x = self.decoder_up1(x, x3)
        x = self.decoder_up2(x, x2)
        x = self.decoder_out(x, x1)
        return x


class TinyWideGamutNet(BaseWideGamutNet):
    def __init__(self, n_in_channels, using_residual=True, limiting_output_range=False):
        super(TinyWideGamutNet, self).__init__(n_in_channels, using_residual, limiting_output_range)
        # Contracting
        self.encoder_inc = DoubleConvBlock(self.n_in_channels, 24)
        self.encoder_down1 = DownBlock(24, 48)
        self.encoder_bridge_down = BridgeDown(48, 96)
        # Expanding
        self.decoder_bridge_up = BridgeUP(96, 48)
        self.decoder_up1 = SafeUpBlock(48, 24)  # Concatenation-safe
        self.decoder_out = SafeOutputBlock(24, self.n_out_channels)  # Concatenation-safe

    def _forward(self, x):
        # Contracting
        x1 = self.encoder_inc(x)
        x2 = self.encoder_down1(x1)
        x3 = self.encoder_bridge_down(x2)
        # Expanding
        x = self.decoder_bridge_up(x3)
        x = self.decoder_up1(x, x2)
        x = self.decoder_out(x, x1)
        return x
