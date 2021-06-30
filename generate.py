"""Generate new images using a trained SinGAN."""
import torch
from src.singan import SinGAN
import argparse
from datetime import datetime
from skimage import io
import numpy as np
from skimage.color import lab2rgb

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
parser.add_argument('--num_imgs', type=int, default=1, help='number of images to create')
parser.add_argument('--shape', type=int, nargs="+", default=(256, 256), help='output shape of images')
parser.add_argument('--save_path', type=str, default='./train', help='path to save images')

# Get arguments
args = parser.parse_args()

# Init variables
device = torch.device('cuda:0') if args.device=='cuda' else torch.device('cpu')
num_imgs = args.num_imgs
shape = tuple(args.shape)

# Create SinGAN model
singan = SinGAN(device, 0.1, 0.1, 10, 1, 1, 1, None)

# Load trained model (look at standard path)
singan.load()

# Check for training progress of SinGAN
if not singan.trained_scale == singan.N:
    print('SinGAN is not completely trained! You can use train.py --load to train it completely.')
    input('Press enter to continue')

# Generate new images
imgs = singan.generate(num_imgs, shape)

# Save images use as name the current date
now = datetime.now()
name = now.strftime('%Y_%m_%d-%H_%M_%S')

PATH = args.save_path + '/' + name
for i, img in enumerate(imgs):
    img = img[0].cpu().detach().permute(1, 2, 0)
    img = img.numpy()
    img[:,:,0] += 1
    img[:,:,0] *= 50
    img[:,:,1:] *= 127.5
    img[:,:,1:] -= 0.5
    img = (lab2rgb(img)*255).astype(np.uint8)
    io.imsave(PATH  + f'_{i}.png', img)