import torch
import torch.nn as nn
from conv_block import ConvBlock

class Discriminator(nn.Module):
    """Single Discriminator for one scale."""

    def __init__(self):
        super(Discriminator, self).__init__()
        pass

    def forward(self, noise, img):
        pass
