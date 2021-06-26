"""Functionality for working with images."""
import torch
import torch.nn
import numpy as np
from skimage import io
from skimage.filters import gaussian
import matplotlib.pyplot as plt


def normalize(img):
    """Normalizes the image between [-1, 1]."""
    return 2*img / 255 - 1

def load_img(path, device):
    """Load image as tensor with shape [C, H, W]."""
    img = io.imread(path)

    # Change to tensor
    img = torch.as_tensor(img).permute(2, 0, 1)
    img = img.to(device).float()

    # Resize to max dimension of 250px
    size = torch.as_tensor(img.shape[1:])
    if size.max().item() > 250:
        scale = 250/size.max().item()
        size = tuple((size*scale).int().tolist())
        img = torch.nn.functional.interpolate(img.unsqueeze(0), size=size, mode='nearest')
        img = img.squeeze(0)

    # Lastly normalize image between [-1, 1]
    img = normalize(img)

    # Add an extra dimension for batch size
    img = img.unsqueeze(0)

    return img

def plot_pyr(pyr, index=0):
    """Plots all images of an image pyramid."""
    N = len(pyr)

    for i, img in enumerate(pyr):
        # Transform images
        img = img[index].permute(1, 2, 0).cpu().detach().numpy()
        img = (img+1)/2

        plt.subplot(1, N, i+1)
        plt.imshow(img, vmin=0, vmax=1)
    plt.show()

def gaussian_smoothing(img, device):
    """Apply gaussian smoothing on the image."""
    np_img = img.squeeze(0).cpu().permute(1,2,0).numpy()
    smooth_img = gaussian(np_img, sigma=0.4, multichannel=True)
    smooth_img = torch.from_numpy(smooth_img).permute(2, 0, 1).unsqueeze(0).to(device)

    return smooth_img

def pyramid(start_img, shapes, device):
    """Creates an image pyramid of the given images and shapes."""
    img = start_img.clone()
    pyr = [img]

    for size in reversed(shapes[:-1]):
        img = gaussian_smoothing(img, device)
        img = torch.nn.functional.interpolate(img, size=size, mode='bicubic')
        img = img.clamp(-1, 1)
        pyr.append(img)

    return pyr[::-1]
