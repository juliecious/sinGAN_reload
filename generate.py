"""Generate new images using a trained SinGAN."""
from numpy import sin
import torch
from src.singan import SinGAN
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
parser.add_argument('--num_imgs', type=int, default=1, help='number of images to create')
parser.add_argument('--shape', type=int, default=1, help='output shape of images')

# Get arguments
args = parser.parse_args()

# Init variables
device = torch.device('cuda:0') if args.device=='cuda' else torch.device('cpu')
lr = args.lr

# Create SinGAN model
singan = SinGAN(device, 0.1, 0.1, 10, 1, 1, 1, None)

# Load trained model (look at standard path)
singan.load()

# Singan should be trained
assert singan.trained_scale == singan.N, 'SinGAN is not completely trained! Use train.py with --load to train it completely!'

# Generate new images
singan.generate()