import torch
import torch.nn as nn
from conv_block import ConvBlock

class Generator(nn.Module):
    """Single Generator for one scale."""

    def __init__(self, kernels, lr):
        """Constructor

        Args:
            kernels (int): Number of kernels per block
            lr (float): Learning rate
        """
        super(Generator, self).__init__()
        betas = [0.5, 0.999]

        self.net = nn.Sequential(
            ConvBlock(3, kernels),
            ConvBlock(kernels, kernels),
            ConvBlock(kernels, kernels),
            ConvBlock(kernels, kernels),
            ConvBlock(kernels, 3, nn.Tanh()),
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
