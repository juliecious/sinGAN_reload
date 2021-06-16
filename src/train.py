import torch
import torch.nn as nn
from skimage import io
import numpy as np
from generator import Generator
from discriminator import Discriminator
import matplotlib.pyplot as plt
import util

# Init variables
device = torch.device('cuda:0')
lr=5e-4
iters=2000
batch_size = 64

# Import picture
img = io.imread('../assets/test.jpg')
img = torch.as_tensor(img).to(device).permute(2, 0, 1)
H, W = img.shape[1:]
assert H == W, 'Image has to be quadratic!'

# Based on the paper, we choose the parameters that the
# coarsest scale is 25px and r is as near as possible to 4/3
N = int(np.round(np.log(H/25)/np.log(4/3) + 1))
r = (H/25)**(1/(N-1))
#upsample = nn.Upsample(scale_factor=r, mode='nearest')
zero_pad = nn.ZeroPad2d(5)

print(f'Number of scales: {N} and scale factor: {r}')

# Init generators and discriminators
G = [Generator(32*(int(i/4)+1), lr).to(device) for i in range(N)]
D = [Discriminator(32*(int(i/4)+1), lr).to(device) for i in range(N)]

def get_pyr_shapes(N, r, start_shape=(25, 25)):
    # Get the shape of the images/noise maps of each scale

    # Init variables
    shape = torch.tensor(start_shape)
    shapes = [shape]

    # For every scale n, create the noise map with right scale
    for n in range(N):
        shape = shape*r
        shapes.append(tuple(shape.int().tolist()))
    return shapes

def noise(N, r, batch_size=1, shape=(25, 25)):
    # Init variables
    z = []
    noise_shape = torch.tensor(shape)
    mean = 0
    sigma = 1

    # For every scale n, create the noise map with right scale
    for n in range(N):
        z_n = torch.normal(mean, sigma, size=(batch_size, 3,) + tuple(noise_shape.int().tolist())).to(device)
        z.append(z_n)
        noise_shape = noise_shape*r
    return z

def plot_pyr(pyr, index=0):
    # Plots an image pyramid, pyr = List of images
    N = len(pyr)

    for i, img in enumerate(pyr):
        # Transform images
        img = img[index].permute(1, 2, 0).cpu().detach().numpy()

        plt.subplot(1, N, i+1)
        plt.imshow(img)
    plt.show()

def sample_imgs(high, r, batch_size, shape=(25, 25)):
    # Sample noise maps
    z = noise(high, r, batch_size=batch_size, shape=shape)

    # First scale only receives noise
    x = []
    x_n = torch.zeros(z[0].shape).to(device)

    # Go through all scales and create image
    for n in range(high):
        upsample = torch.nn.Upsample(size=tuple(z[n].shape[2:]), mode='nearest')

        x_n = upsample(x_n)
        x_n = G[n](z[n], x_n)
        x.append(x_n)

    return x

def train(N, r, iters, batch_size):
    # Train each scale one after the other
    for n in range(N):
        for i in range(iters):
            # Sample only for the current scale
            fake_imgs = sample_imgs(n, r, batch_size)

            #



# Test noise
#pyr = noise(N, r, shape=(25, 25))
#for t in pyr:
#    print(t.shape)

#plot_pyr(pyr)

# Test generating imgs
#imgs = sample_imgs(N, r, 1)
#for t in imgs:
#    print(t)

#plot_pyr(imgs)

train(N, r, iters, batch_size)