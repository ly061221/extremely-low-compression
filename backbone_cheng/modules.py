import torch.nn as nn
import torch
from backbone.entropy_model_my import GaussianConditional, EntropyBottleneck
import math


# part of hypermodel
class HyperEncoder(nn.Module):
    def __init__(self, N, M):
        super(HyperEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=M, out_channels=N, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=N, out_channels=N, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=N, out_channels=N, kernel_size=5, stride=2, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU(inplace=True)(x)
        x = self.conv2(x)
        x = nn.LeakyReLU(inplace=True)(x)
        x = self.conv3(x)
        return x


# part of hypermodel
class HyperDecoder(nn.Module):
    def __init__(self, N, M):
        super(HyperDecoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels=N, out_channels=M, kernel_size=5, stride=2,
                                          output_padding=1, padding=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=M, out_channels=M * 3 // 2, kernel_size=5, stride=2,
                                          output_padding=1,
                                          padding=2)
        self.deconv3 = nn.ConvTranspose2d(in_channels=M * 3 // 2, out_channels=M * 2, kernel_size=3, stride=1,
                                          output_padding=0,
                                          padding=1)

    def forward(self, x):
        x = self.deconv1(x)
        x = nn.LeakyReLU(inplace=True)(x)
        x = self.deconv2(x)
        x = nn.LeakyReLU(inplace=True)(x)
        x = self.deconv3(x)
        return x


#  part of contextprediction
class MaskedConv2d(nn.Conv2d):
    '''
    Implementation of the Masked convolution from the paper
    Van den Oord, Aaron, et al. "Conditional image generation with pixelcnn decoders." Advances in neural information processing systems. 2016.
    https://arxiv.org/pdf/1606.05328.pdf
    '''

    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ('A', 'B')
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


# part of hypermodel
class ContextPrediction(nn.Module):
    def __init__(self, M):
        super(ContextPrediction, self).__init__()
        self.masked = MaskedConv2d("A", in_channels=M, out_channels=2 * M, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        return self.masked(x)


# part of hypermodel
class EntropyParameters(nn.Module):
    def __init__(self, M):
        super(EntropyParameters, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=M * 12 // 3, out_channels=M * 10 // 3, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=M * 10 // 3, out_channels=M * 8 // 3, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=M * 8 // 3, out_channels=M * 6 // 3, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU(inplace=True)(x)
        x = self.conv2(x)
        x = nn.LeakyReLU(inplace=True)(x)
        x = self.conv3(x)
        return x


class Hyper_Model(nn.Module):
    def __init__(self, N):
        super(Hyper_Model, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hyper_encoder = HyperEncoder(N, N).to(self.device)
        self.hyper_decoder = HyperDecoder(N, N).to(self.device)
        self.entropy = EntropyParameters(N).to(self.device)
        self.context = ContextPrediction(N).to(self.device)
        self.gaussian_conditional = GaussianConditional(None)
        self.entropy_bottleneck = EntropyBottleneck(N)

    def quantize_my(self, x):
        """
        Quantize function:  The use of round function during training will cause the gradient to be 0 and will stop encoder from training.
        Therefore to immitate quantisation we add a uniform noise between -1/2 and 1/2
        :param x: Tensor
        :return: Tensor
        """
        uniform = -1 * torch.rand(x.shape) + 1 / 2
        return x + uniform.to(self.device)

    def quantize(self, inputs, mode, means=None):
        if mode not in ("noise", "dequantize", "symbols"):
            raise ValueError(f'Invalid quantization mode: "{mode}"')

        if mode == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs

        outputs = inputs.clone()
        if means is not None:
            outputs -= means

        outputs = torch.round(outputs)

        if mode == "dequantize":
            if means is not None:
                outputs += means
            return outputs

        assert mode == "symbols", mode
        outputs = outputs.int()
        return outputs

    def quantize_compressai(self, x):
        half = float(0.5)
        noise = torch.empty_like(x).uniform_(-half, half)
        inputs = x + noise
        return inputs

    def forward(self, x):
        z = self.hyper_encoder(x)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.hyper_decoder(z_hat)
        y_hat = self.gaussian_conditional.quantize(
            x, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context(y_hat)
        gaussian_params = self.entropy(torch.cat((params, ctx_params), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(x, scales_hat, means=means_hat)

        return {
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


class RB(nn.Module):
    def __init__(self, channel, use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0)
        else:
            self.conv3 = None

    def forward(self, x):
        feat = self.conv1(x)
        feat = nn.LeakyReLU()(feat)
        feat = self.conv2(feat)
        feat = nn.LeakyReLU()(feat)
        if self.conv3:
            x = self.conv3(x)
        feat += x
        return feat


class RB_enh(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=channel[1], out_channels=channel[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=channel[1], out_channels=channel[0], kernel_size=1, padding=0)

    def forward(self, x):
        feat = self.conv1(x)
        feat = nn.LeakyReLU()(feat)
        feat = self.conv2(feat)
        feat = nn.LeakyReLU()(feat)
        feat = self.conv3(feat)
        feat += x
        return feat


class ResNet_Transforms_module_192(nn.Module):
    def __init__(self, res_channel):
        super(ResNet_Transforms_module_192, self).__init__()
        a = res_channel
        self.conv1 = nn.ConvTranspose2d(in_channels=192, out_channels=a, kernel_size=5, stride=2, output_padding=1,
                                        padding=2)  # 192*64*64
        # self.RB_upsample = RB_upsample([192, 256])
        self.RB1_1 = RB([a, a])  # 192 64 64
        self.RB1_2 = RB_enh([a, 384])  # 192 64 64
        self.RB1_3 = RB([a, a])  # 192 64 64
        self.conv2 = nn.Conv2d(in_channels=a, out_channels=256, kernel_size=3, padding=1)  # 512 64 64
        self.RB2_1 = RB([256, 256])  # 512 64 64
        self.RB2_2 = RB_enh([256, 512])  # 512 64 64
        self.RB2_3 = RB([256, 256])  # 512 64 64

        self.conv = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)  # 512 64 64

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.RB1_1(feat)
        feat = self.RB1_2(feat)
        feat = self.RB1_3(feat)
        feat = self.conv2(feat)
        feat = self.RB2_1(feat)
        feat = self.RB2_2(feat)
        feat = self.RB2_3(feat)
        feat = self.conv(feat)
        return feat


class ResNet50Modified(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Modified, self).__init__()
        self.base_layers = nn.Sequential(*list(original_model.children())[:5])
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
        self.avgpool = original_model.avgpool
        self.fc = original_model.fc

    def forward(self, x):
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def rate_loss(output, target):
    N, _, H, W = target.size()
    num_pixels = N * H * W

    out = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in output["likelihoods"].values()
    )
    return out


from .cheng2020_layers import *
# -------------------------------------------------------#
# cheng2020:
# encoder:
# decoder:
# -------------------------------------------------------#

#  cheng-encoder
class Encodercheng(nn.Module):
    def __init__(self):
        super(Encodercheng, self).__init__()
        N = 192
        self.encoder = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
            AttentionBlock(N),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


#  cheng-decoder
class Decodercheng(nn.Module):
    def __init__(self):
        super(Decodercheng, self).__init__()
        N = 192
        self.decoder = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

