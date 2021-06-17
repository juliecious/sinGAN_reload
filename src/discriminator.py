import torch
import torch.nn as nn
from conv_block import ConvBlock

class Discriminator(nn.Module):
    """Single Discriminator for one scale."""

    def __init__(self, lr, betas=(0.5, 0.999)):
        """Constructor

        Args:
            lr (float): Learning rate
            betas (tuple): Betas default to (0.5, 0.999)
        """
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            ConvBlock(3, kernels),
            ConvBlock(kernels, kernels),
            ConvBlock(kernels, kernels),
            ConvBlock(kernels, kernels),
            nn.Conv2d(kernels, 3, kernel),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=betas)

    def forward(self, noise, img):
        """Forward pass

        Args:
            noise (tensor): Noise map
            img (tensor): Upsampled image

        Returns:
            tensor: generated image
        """
        input = noise + img
        residuals = self.net(input)
        result = img + residuals

        return result
