import torch
import torch.nn
import numpy as np
from skimage import io
from skimage.transform import pyramid_gaussian
import matplotlib.pyplot as plt

img = io.imread('test.jpg')
pyr = list(pyramid_gaussian(img, max_layer=3, downscale=2, multichannel=True))


#class Pyramid:
#    """Gaussian's pyramid of one image."""
