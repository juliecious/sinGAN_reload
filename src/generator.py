import torch
import torch.nn as nn
from conv_block import ConvBlock

class Generator(nn.Module):
    """Single Generator for one scale."""

    def __init__(self):
        super(Generator, self).__init__()
        #...

    def forward(self, noise, img):
        #...
