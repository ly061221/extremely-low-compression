import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Custom
from torch.nn import init

from GAN_hific.normalisation import channel, instance


class Residual_single(nn.Module):
    def __init__(self, inc, kernel_size=3, stride=1,
                 channel_norm=True, activation='relu'):
        """
        input_dims: Dimension of input tensor (B,C,H,W)
        """
        super(Residual_single, self).__init__()

        self.activation = getattr(F, activation)
        in_channels = inc
        norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=False)

        if channel_norm is True:
            self.interlayer_norm = channel.ChannelNorm2D_wrap
        else:
            self.interlayer_norm = instance.InstanceNorm2D_wrap

        pad_size = int((kernel_size - 1) / 2)
        self.pad = nn.ReflectionPad2d(pad_size)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride)
        self.norm1 = self.interlayer_norm(in_channels, **norm_kwargs)
        self.norm2 = self.interlayer_norm(in_channels, **norm_kwargs)

    def forward(self, x):
        identity_map = x
        res = self.pad(x)
        res = self.conv1(res)
        res = self.norm1(res)
        res = self.activation(res)

        res = self.pad(res)
        res = self.conv2(res)
        res = self.norm2(res)

        return torch.add(res, identity_map)


class ResidualBlock_decoder(nn.Module):
    def __init__(self, input_channel, num=5):
        super(ResidualBlock_decoder, self).__init__()
        self.singleblock = Residual_single(input_channel, kernel_size=3, stride=1, channel_norm=True, activation='relu')
        self.numblock = num
        self.trunk_conv = nn.Conv2d(input_channel, input_channel, 3, 1, 1, bias=True)

    def forward(self, x):
        y = x
        for m in range(self.numblock):
            y = self.singleblock.forward(y)
        trunk = self.trunk_conv(y)
        x = x + trunk

        return x


class Residual_single2(nn.Module):
    def __init__(self, inc, kernel_size=3, stride=1, activation='leaky_relu'):
        """
        input_dims: Dimension of input tensor (B,C,H,W)
        """
        super(Residual_single2, self).__init__()

        self.activation = getattr(F, activation)
        in_channels = inc
        norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=False)

        pad_size = int((kernel_size - 1) / 2)
        self.pad = nn.ReflectionPad2d(pad_size)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity_map = x
        res = self.pad(x)
        res = self.lrelu(self.conv1(res))

        res = self.pad(res)
        res = self.conv2(res)

        return torch.add(0.2 * res, identity_map)


class ResidualBlock_decoder2(nn.Module):
    def __init__(self, input_channel, num=9):
        super(ResidualBlock_decoder2, self).__init__()
        self.singleblock = Residual_single2(input_channel, kernel_size=3, stride=1, activation='leaky_relu')
        self.numblock = num
        self.trunk_conv = nn.Conv2d(input_channel, input_channel, 3, 1, 1, bias=True)

    def forward(self, x):
        y = x
        for m in range(self.numblock):
            y = self.singleblock.forward(y)
        trunk = self.trunk_conv(y)
        x = x + trunk
        # x = x + y

        return x
