import torch
import torch.nn as nn
from conv_block import ConvBlock

class Generator(nn.Module):
    """
    Define single generator for one scale. It consists of 5 conv layers
    whose output is a residual image that is added back to the input image
    """

    def __init__(self, lr, betas=(0.5, 0.999)):
        """Constructor

        Args:
            lr (float): Learning rate
            betas (tuple): Betas default to (0.5, 0.999)
        """
        super(Generator, self).__init__()

        """
        According to the paper, we start with 32 kernels per block at the coarest scale
        and increase this number by a factor of 2 every 4 scale
        """

        self.net = nn.Sequential(
            ConvBlock(input_dim=3, output_dim=32, kernel_size=3),
            ConvBlock(input_dim=32, output_dim=16, kernel_size=3),
            ConvBlock(input_dim=16, output_dim=8, kernel_size=3),
            ConvBlock(input_dim=8, output_dim=4, kernel_size=3),
            ConvBlock(input_dim=4, output_dim=3, kernel_size=3, act_fn=nn.Tanh())
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=betas)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1000, 2000], gamma=0.1)
        self.zero_pad = nn.ZeroPad2d(5)

    def forward(self, noise, img):
        """Forward pass

        Args:
            noise (tensor): Noise map
            img (tensor): Upsampled image

        Returns:
            tensor: generated image
        """
        input = self.zero_pad(noise) + self.zero_pad(img)
        residuals = self.net(input)
        result = img + residuals

        return result
