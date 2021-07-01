"""Generate new images using a trained SinGAN."""
import torch
from src.singan import SinGAN
import argparse
from datetime import datetime
from skimage import io
import numpy as np
from src.image import load_img
from skimage.color import lab2rgb

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
parser.add_argument('--path', type=str, default='./assets/clip_art.png', help='path to clip art image')
parser.add_argument('--save_path', type=str, default='./train', help='path to save images')
parser.add_argument('--start', type=int, default=1, help='start scale, normally 1 or 2')

# Get arguments
args = parser.parse_args()

# Init variables
device = torch.device('cuda:0') if args.device=='cuda' else torch.device('cpu')
path = args.path
start = args.start

# Load clip art image
clip_art = load_img(path, device)

# Create SinGAN model
singan = SinGAN(device, 0.1, 0.1, 10, 1, 1, 1, None)

# Load trained model (look at standard path)
singan.load()

# Check for training progress of SinGAN
if not singan.trained_scale == singan.N:
    print('SinGAN is not completely trained! You can use train.py --load to train it completely.')
    input('Press enter to continue')

# Generate new images
img = singan.paint_to_img(clip_art, start=1)

# Save images use as name the current date
now = datetime.now()
date = now.strftime('%Y_%m_%d-%H_%M_%S')

# Save image
PATH = args.save_path + '/clipart_' + date + f'.png'
img = img[0].cpu().detach().permute(1, 2, 0)
img = img.numpy()
img[:,:,0] += 1
img[:,:,0] *= 50
img[:,:,1:] *= 127.5
img[:,:,1:] -= 0.5
img = (lab2rgb(img)*255).astype(np.uint8)
io.imsave(PATH, img)