
from .cheng2020_layers import *
from GAN_Models.RRDB_model import *
from GAN_hific.SRGAN import *


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class ResidualBottleneck_Enhance_relu(nn.Module):
    def __init__(self, N=256, channel=256) -> None:
        super().__init__()
        self.branch = nn.Sequential(
            conv1x1(N, channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            conv1x1(channel, N)
        )

    def forward(self, x):
        out = x + self.branch(x)
        return out


class BottleneckNetwork(nn.Module):
    def __init__(self, inchannel=192, outchannel=64):
        super(BottleneckNetwork, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(outchannel)
        )

    def forward(self, x):
        return self.bottleneck(x)


class CBAE(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        """
        CBAE
        :param in_channels: input channels
        :param reduction_ratio: channel reduction ratio
        """
        super(CBAE, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels // reduction_ratio, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, stride=1, padding=0)
        )
        self.sigmoid = nn.Sigmoid()
        self.resconv = ResidualBottleneck_Enhance_relu(in_channels, in_channels)

    def forward(self, x):
        attention_weights = self.bottleneck(x)
        attention_weights = self.sigmoid(attention_weights)  # (B, C, 1, 1)
        out = self.resconv(x) * attention_weights  # (B, C, H, W)
        return out


class FFM(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(FFM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, 3, 1, 1),
            nn.ReLU()
        )
        self.channel_attn = CBAE(inchannel)
        self.conv1 = conv1x1(inchannel, outchannel)

    def forward(self, f, enh_f):
        f_fusion = torch.cat((f, enh_f), dim=1)
        f_fusion_attn = self.channel_attn(f_fusion)
        out = f_fusion + f_fusion_attn
        out = self.conv1(out)
        return out


class RRDBBlock_up_enh(nn.Module):
    def __init__(self, pixsize):
        super(RRDBBlock_up_enh, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
        )
        self.RRDBX3_up = nn.Sequential(
            RRDBNet_decoder(64, 3, pixsize),
            self.upsample
        )
        self.upconv = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(upscale_factor=2),
        )

    def forward(self, x):
        enh_feat = self.RRDBX3_up(x)
        out = enh_feat + self.upconv(x)
        return out


class Decodercheng_featenh(nn.Module):
    def __init__(self):
        super(Decodercheng_featenh, self).__init__()
        N = 192
        self.conv1 = conv1x1(192, 64)
        self.RRDB_res2enh = RRDBBlock_up_enh(32)
        self.RRDB_res3enh = RRDBBlock_up_enh(64)
        self.decoder_res1 = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N))
        self.decoder_res2 = nn.Sequential(
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N))
        self.decoder_res3 = nn.Sequential(
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlock(N, N))
        self.decoder_res4 = nn.Sequential(
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N))
        self.feat_fusion = FFM(256, 192)
        self.decoder_outimg = subpel_conv3x3(N, 3, 2)

    def forward(self, x):
        y1 = self.decoder_res1(x)  # 192x16x16
        y2 = self.decoder_res2(y1)  # 192x32x32
        y2_enh = self.RRDB_res2enh(self.conv1(y2))  # y2_enh, 64*64*64
        y3 = self.decoder_res3(y2)  # 192x64x64
        y3_enh = self.RRDB_res3enh(self.conv1(y3) + y2_enh)  # y2_enh, 64*128*128
        y4 = self.decoder_res4(y3)  # 192x128x128
        out = self.feat_fusion(y4, y3_enh)
        out = self.decoder_outimg(out)
        return out

