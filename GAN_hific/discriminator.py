import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, image_dims, context_dims, C, spectral_norm=True):
        """ 
        Convolutional patchGAN discriminator used in [1].
        Accepts as input generator output G(z) or x ~ p*(x) where
        p*(x) is the true data distribution.
        Contextual information provided is encoder output y = E(x)
        ========
        Arguments:
        image_dims:     Dimensions of input image, (C_in,H,W)
        context_dims:   Dimensions of contextual information, (C_in', H', W')
        C:              Bottleneck depth, controls bits-per-pixel
                        C = 220 used in [1], C = C_in' if encoder output used
                        as context.

        [1] Mentzer et. al., "High-Fidelity Generative Image Compression", 
            arXiv:2006.09965 (2020).
        """
        super(Discriminator, self).__init__()

        self.image_dims = image_dims
        self.context_dims = context_dims
        im_channels = self.image_dims[0]
        kernel_dim = 4
        context_C_out = 12
        filters = (64, 128, 256, 512)

        # Upscale encoder output - (C, 16, 16) -> (12, 256, 256)
        self.context_conv = nn.Conv2d(C, context_C_out, kernel_size=3, padding=1, padding_mode='reflect')
        self.context_upsample = nn.Upsample(scale_factor=16, mode='nearest')

        # Images downscaled to 500 x 1000 + randomly cropped to 256 x 256
        # assert image_dims == (im_channels, 256, 256), 'Crop image to 256 x 256!'

        # Layer / normalization options
        # TODO: calculate padding properly
        cnn_kwargs = dict(stride=2, padding=1, padding_mode='reflect')
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        if spectral_norm is True:
            norm = nn.utils.spectral_norm
        else:
            norm = nn.utils.weight_norm

        # (C_in + C_in', 256,256) -> (64,128,128), with implicit padding
        # TODO: Check if removing spectral norm in first layer works
        self.conv1 = norm(nn.Conv2d(im_channels + context_C_out, filters[0], kernel_dim, **cnn_kwargs))

        # (128,128) -> (64,64)
        self.conv2 = norm(nn.Conv2d(filters[0], filters[1], kernel_dim, **cnn_kwargs))

        # (64,64) -> (32,32)
        self.conv3 = norm(nn.Conv2d(filters[1], filters[2], kernel_dim, **cnn_kwargs))

        # (32,32) -> (16,16)
        self.conv4 = norm(nn.Conv2d(filters[2], filters[3], kernel_dim, **cnn_kwargs))

        self.conv_out = nn.Conv2d(filters[3], 1, kernel_size=1, stride=1)

    def forward(self, x, y):
        """
        x: Concatenated real/gen images
        y: Quantized latents
        """
        batch_size = x.size()[0]

        # Concatenate upscaled encoder output y as contextual information
        y = self.activation(self.context_conv(y))
        y = self.context_upsample(y)

        x = torch.cat((x, y), dim=1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        out_logits = self.conv_out(x).view(-1, 1)
        out = torch.sigmoid(out_logits)

        return out, out_logits


def GAN_D_loss(D_real, D_gen):
    D_loss_real = torch.mean(torch.square(D_real - 1.0))
    D_loss_gen = torch.mean(torch.square(D_gen))
    D_loss = 0.5 * (D_loss_real + D_loss_gen)

    return D_loss


def GAN_G_loss(D_gen):
    G_loss = 0.5 * torch.mean(torch.square(D_gen - 1.0))
    return G_loss


from GAN_hific.loss import perceptual_loss as ps


class LPIPSLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, device):
        super().__init__()
        self.lpips = ps.PerceptualLoss(device, model='net-lin', net='vgg',
                                       use_gpu=torch.cuda.is_available())

    def forward(self, output, target):

        lpips_loss = self.lpips(output, target)
        # for i in range(len(lpips_loss)):
        #     out = 0
        #     out += lpips_loss[i]

        out = lpips_loss.mean()
        return out
