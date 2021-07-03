"""Training script for SinGAN."""
import torch
from src.singan import SinGAN
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--iters', type=int, default=2000, help='iterations per scale')
parser.add_argument('--sample_interval', type=int, default=500, help='iteration intervals to sample test images')
parser.add_argument('--lam', type=float, default=0.1, help='lambda parameter for gradient penalty')
parser.add_argument('--network_iters', type=int, default=3, help='iterations per network update')
parser.add_argument('--alpha', type=int, default=10, help='reconstruction loss weight')
parser.add_argument('--path', type=str, default='./assets/new_york.jpg', help='image path')
parser.add_argument('--load', default=False, action='store_true', help='load current network')

# Get arguments
args = parser.parse_args()

# Init variables
device = torch.device('cuda:0') if args.device=='cuda' else torch.device('cpu')
lr = args.lr
iters = args.iters
sample_interval = args.sample_interval
lam = args.lam
network_iters = args.network_iters
alpha = args.alpha

# Create SinGAN model
singan = SinGAN(device, lr, lam, alpha, iters, sample_interval,  network_iters, args.path)

# Load current network if using load
if args.load:
    singan.load()

# Train SinGAN
singan.train()