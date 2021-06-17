import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride=1, act_fn=nn.LeakyReLU(0.2, inplace=False)):
        """Constructor

        Args:
            input (int): Number of input channels
            output (int): Number of output channels of ConvBlock
            stride (int): Default to 1
            act_fn (function): Activation function. Default to LeakyReLU

        """
        super(ConvBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size, stride=stride),
            nn.BatchNorm2d(output_dim),
            act_fn,
        )

    def forward(self, x):
        return self.net(x)

class Generator(nn.Module):
    """
    Define single generator for one scale. It consists of 5 conv layers
    whose output is a residual image that is added back to the input image.
    The generator's last conv-block uses Tanh instead of ReLU.
    """
    def __init__(self, lr=5e-4, betas=(0.5, 0.999)):
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
        output_dim = 32
        self.net = nn.Sequential(
            ConvBlock(3, output_dim, 3),
            ConvBlock(output_dim, output_dim, 3),
            ConvBlock(output_dim, output_dim, 3),
            ConvBlock(output_dim, output_dim, 3),
            ConvBlock(output_dim, output_dim, 3, act_fn=nn.Tanh())
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=betas)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1600], gamma=0.1)
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
        return img + residuals

class Discriminator(nn.Module):
    """
    Define single Discriminator for one scale. The discriminator last conv-block has neither
    normalization nor activation.
    """

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

        Args:
            img (tensor): Upsampled image

        Returns:
            tensor: generated image
        """
        return self.net(img)