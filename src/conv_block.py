import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride=1, act_fn=nn.LeakyReLU(0.2, inplace=False)):
        """Constructor

        Args:
            input (int): Number of input channels
            output (int): Number of output channels of ConvBlock
            stride (int): Default to 1
            act_fn (function): Activation function default to LeakyReLU

        """
        super(ConvBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size, stride=stride),
            nn.BatchNorm2d(output_dim),
            act_fn,
        )

    def forward(self, x):
        return self.net(x)