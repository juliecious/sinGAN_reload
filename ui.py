import os
import io

from src.singan import SinGAN
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from src.image import load_img
import numpy as np
import PySimpleGUI as sg
import torch
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
parser.add_argument('--scale', type=int, default=2, help='injection_scale scale, from 0 to N')

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
shape = (250, 250)
injection_scale = args.scale

# Set window layout
selection = [
    sg.Text('Select an image for training: '),
    sg.Input(size=(35, 1), disabled=True, enable_events=True, key="-FILE-"),
    sg.FileBrowse(),
]

injection = [
    sg.Text('Select an image for injection: '),
    sg.Input(size=(35, 1), disabled=True, enable_events=True, key="-PATH_INJECTED-"),
    sg.FileBrowse(disabled=True, key="-FILE_INJECTED-"),
]

train_img = [
    [sg.Text('Training Image:'),],
    [sg.Image(background_color='white',size=(250, 250), key='-IMAGE-')],
]

generation = [
    [sg.Text('Generated Image:'),],
    [sg.Image(background_color='white',size=(250, 250), key='-GENERATED-')],
]

actions = [
    [sg.Text('SinGAN actions:'),],
    [sg.Button('Load', disabled=False, size=(10, 1), tooltip='Loads an already trained SinGAN model', key='-LOAD-')],
    [sg.Button('Train', disabled=True, size=(10, 1), tooltip='Train the SinGAN', key='-TRAIN-')],
    [sg.Button('Generate', disabled=True, size=(10, 1), tooltip='Generate new images', key='-GENERATE-')],
    [sg.Button('Inject', disabled=True, size=(10, 1), tooltip='Inject image into SinGAN (for Paint-to-Image and Harmonization)', key='-INJECT-')],
]

layout = [
    selection,
    injection,
    [
        sg.Column(train_img),
        sg.VSeperator(),
        sg.Column(generation),
        sg.VSeperator(),
        sg.Column(actions),
    ],
    [sg.Text('Currently nothing loaded!', size=(80, 1), key='-TRAIN_OUT-')]
]

# Create window
window = sg.Window('SinGAN reloaded', layout)

# Get loading image
img_loading = Image.open('./loading.png')
img_loading.thumbnail((250, 250))
bits_loading = io.BytesIO()
img_loading.save(bits_loading, format='PNG')

# Event loop
while True:
    event, values = window.read()

    if event == 'Exit' or event == sg.WIN_CLOSED:
        break
    if event == '-FILE-':
        # Import image and update windows layout
        # Using pil is the official solution
        path = values['-FILE-']
        if os.path.exists(path):
            img = Image.open(path)
            img.thumbnail((250, 250))
            bits = io.BytesIO()
            img.save(bits, format='PNG')

            # Show loading image
            window['-IMAGE-'].update(data=bits_loading.getvalue())
            window.refresh()

            # If image exists, create SinGAN model
            singan = SinGAN(device, lr, lam, alpha, iters, sample_interval,  network_iters, path)

            # Update button
            window['-TRAIN-'].update(disabled=False)
            window['-GENERATE-'].update(disabled=True)
            window['-INJECT-'].update(disabled=True)
            window['-FILE_INJECTED-'].update(disabled=True)

            # Update window
            window['-IMAGE-'].update(data=bits.getvalue())

            # Update text
            window['-TRAIN_OUT-'].update(value='Image loaded!')

    if event == '-LOAD-':
        # Disable Button
        window['-LOAD-'].update(disabled=True)

        # Show loading image
        window['-IMAGE-'].update(data=bits_loading.getvalue())
        window.refresh()

        # Create dummy singan
        singan = SinGAN(device, 0.1, 0.1, 10, 1, 1, 1, None)

        # Load SinGAN
        singan.load()

        # Convert image into PIL img
        img = singan.img[0].cpu().detach().permute(1, 2, 0)
        img = img.numpy()
        img[:,:,0] += 1
        img[:,:,0] *= 50
        img[:,:,1:] *= 127.5
        img[:,:,1:] -= 0.5
        img = (lab2rgb(img)*255).astype(np.uint8)
        img = Image.fromarray(np.uint8(img)).convert('RGB')

        # Display loaded image
        img.thumbnail((250, 250))
        bits = io.BytesIO()
        img.save(bits, format='PNG')
        window['-IMAGE-'].update(data=bits.getvalue())
        window['-GENERATE-'].update(disabled=False)
        window['-INJECT-'].update(disabled=False)
        window['-FILE_INJECTED-'].update(disabled=False)

        # Update text
        window['-TRAIN_OUT-'].update(value='Model loaded!')

    if event == '-TRAIN-':
        # Disable Buttons
        window['-TRAIN-'].update(disabled=True)
        window['-LOAD-'].update(disabled=True)
        window.refresh()

        # Start training
        singan.train(window=window)

        # Enable Buttons
        window['-TRAIN-'].update(disabled=False)
        window['-LOAD-'].update(disabled=False)
        window['-GENERATE-'].update(disabled=False)
        window['-INJECT-'].update(disabled=False)
        window['-FILE_INJECTED-'].update(disabled=False)

    if event == '-GENERATE-':
        # Disable Buttons
        window['-GENERATE-'].update(disabled=True)
        window.refresh()

        # Generate new images
        img = singan.generate(1, shape)[0]

        # Convert image to PIL
        img = img[0].cpu().detach().permute(1, 2, 0)
        img = img.numpy()
        img[:,:,0] += 1
        img[:,:,0] *= 50
        img[:,:,1:] *= 127.5
        img[:,:,1:] -= 0.5
        img = (lab2rgb(img)*255).astype(np.uint8)
        img = Image.fromarray(np.uint8(img)).convert('RGB')

        # Display generated image
        img.thumbnail((250, 250))
        bits = io.BytesIO()
        img.save(bits, format='PNG')
        window['-GENERATED-'].update(data=bits.getvalue())

        # Disable Buttons
        window['-GENERATE-'].update(disabled=False)

    if event == '-INJECT-':
        # Disable Buttons
        window['-INJECT-'].update(disabled=True)
        window['-FILE_INJECTED-'].update(disabled=True)
        window.refresh()

        path = values['-PATH_INJECTED-']
        if os.path.exists(path):
            # Load Injection file and inject it
            clip_art = load_img(path, device)
            img = singan.paint_to_img(clip_art, injection_scale=injection_scale)

            # Convert image to PIL
            img = img[0].cpu().detach().permute(1, 2, 0)
            img = img.numpy()
            img[:,:,0] += 1
            img[:,:,0] *= 50
            img[:,:,1:] *= 127.5
            img[:,:,1:] -= 0.5
            img = (lab2rgb(img)*255).astype(np.uint8)
            img = Image.fromarray(np.uint8(img)).convert('RGB')

            # Display generated image
            img.thumbnail((250, 250))
            bits = io.BytesIO()
            img.save(bits, format='PNG')
            window['-GENERATED-'].update(data=bits.getvalue())

        # Disable Buttons
        window['-INJECT-'].update(disabled=False)
        window['-FILE_INJECTED-'].update(disabled=False)
