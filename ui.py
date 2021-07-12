import os
import io
from src.singan import SinGAN
from PIL import Image
import PySimpleGUI as sg
import torch

# Set parameters
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
lr = 5e-4
iters = 2000
sample_interval = 500
lam = 0.1
network_iters = 3
alpha = 10

# Set window layout
selection = [
    sg.Text('Select an image for training: '),
    sg.Input(size=(35, 1), disabled=True, enable_events=True, key="-FILE-"),
    sg.FileBrowse(),
]

train_img = [
    [sg.Text('Training Image:'),],
    [sg.Image(background_color='white',size=(250, 250), key='-IMAGE-')],
]

generation = [
    [sg.Text('Generated Image:'),],
    [sg.Image(background_color='white',size=(250, 250), key='-IMAGE-')],
]

actions = [
    [sg.Text('SinGAN actions:'),],
    [sg.Button('Load', disabled=False, size=(10, 1), tooltip='Loads an already trained SinGAN model', key='-LOAD-')],
    [sg.Button('Train', disabled=True, size=(10, 1), tooltip='Train the SinGAN', key='-TRAIN-')],
    [sg.Button('Generate', disabled=True, size=(10, 1), tooltip='Generate new images')],
    [sg.Button('Inject', disabled=True, size=(10, 1), tooltip='Inject image into SinGAN (for Paint-to-Image and Harmonization)')],
]

layout = [
    selection,
    [
        sg.Column(train_img),
        sg.VSeperator(),
        sg.Column(actions),
    ]
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
            window['-IMAGE-'].update(data=bits_loading.getvalue())

            # If image exists, create SinGAN model
            singan = SinGAN(device, lr, lam, alpha, iters, sample_interval,  network_iters, path)

            # Update button
            window['-TRAIN-'].update(disabled=False)

            # Update window
            window['-IMAGE-'].update(data=bits.getvalue())

    if event == '-LOAD-':
        # Disable Button
        window['-LOAD-'].update(disabled=True)

        # Load SinGAN
        singan.load()

    if event == '-TRAIN-':
        # Disable Button
        window['-TRAIN-'].update(disabled=True)

        # Start training
        singan.train()
