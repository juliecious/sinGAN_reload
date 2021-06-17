import torch
import torch.nn as nn
from conv_block import ConvBlock

class Discriminator(nn.Module):
    """Single Discriminator for one scale."""

    def __init__(self, lr=5e-4, betas=(0.5, 0.999)):
        """Constructor

        Args:
            lr (float): Learning rate
            betas (tuple): Betas default to (0.5, 0.999)
        """
        super(Discriminator, self).__init__()

        output_dim = 32
        self.net = nn.Sequential(
            ConvBlock(3, output_dim, 3),
            ConvBlock(output_dim, output_dim, 3),
            ConvBlock(output_dim, output_dim, 3),
            ConvBlock(output_dim, output_dim, 3),
            ConvBlock(output_dim, 1, 3)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=betas)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1600], gamma=0.1)

    def forward(self, img):
        """Forward pass

        Return generated image
        """
        return self.net(img)
