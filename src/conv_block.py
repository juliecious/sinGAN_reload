import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, input, output, act_fn=nn.LeakyReLU(0.2)):
        """Constructor

        Args:
            input (int): Number of input channels
            output (int): Number of output channels of ConvBlock
            act_fn (function): Activation to use
        """
        super(ConvBlock, self).__init__()
        kernel = (3, 3)
        padding = 0
        padding_mode = 'zeros'

        self.net = nn.Sequential(
            nn.Conv2d(input, output, kernel, padding=padding, padding_mode=padding_mode),
            nn.BatchNorm2d(output),
            act_fn,
        )

    def forward(self, x):
        return self.net(x)