import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, act_fn=nn.LeakyReLU(0.2, inplace=False)):
        """Constructor

        Args:
            input (int): Number of input channels
            output (int): Number of output channels of ConvBlock
            stride (int): Default to 1
            act_fn (function): Activation function. Default to LeakyReLU

        """
        super(ConvBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size),
            nn.BatchNorm2d(output_dim),
            act_fn,
        )

        # Weight initalisation
        self.net[0].weight.data.normal_(0.0, 0.02)
        self.net[1].weight.data.normal_(1.0, 0.02)
        self.net[1].bias.data.fill_(0)

    def forward(self, x):
        return self.net(x)

class Generator(nn.Module):
    """
    Define single generator for one scale. It consists of 5 conv layers
    whose output is a residual image that is added back to the input image.
    The generator's last conv-block uses Tanh instead of ReLU.
    """
    def __init__(self, output_dim=32, lr=5e-4, betas=(0.5, 0.999), milestones=[1600], gamma=0.1):
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
            ConvBlock(3, output_dim, 3),
            ConvBlock(output_dim, output_dim, 3),
            ConvBlock(output_dim, output_dim, 3),
            ConvBlock(output_dim, output_dim, 3),
            nn.Conv2d(output_dim, 3, 3),
            nn.Tanh()
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=betas)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
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
        return (img + residuals).clamp(-1, 1)

class Discriminator(nn.Module):
    """
    Define single Discriminator for one scale. The discriminator last conv-block has neither
    normalization nor activation.
    """

    def __init__(self, output_dim=32, lr=5e-4, betas=(0.5, 0.999), milestones=[1600], gamma=0.1):
        """Constructor

        Args:
            lr (float): Learning rate
            betas (tuple): Betas default to (0.5, 0.999)
        """
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            ConvBlock(3, output_dim, 3),
            ConvBlock(output_dim, output_dim, 3),
            ConvBlock(output_dim, output_dim, 3),
            ConvBlock(output_dim, output_dim, 3),
            nn.Conv2d(output_dim, 1, 3),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=betas)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)

    def forward(self, img):
        """Forward pass

        Args:
            img (tensor): Upsampled image

        Returns:
            tensor: generated image
        """
        return self.net(img)