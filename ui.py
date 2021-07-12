import os
import io
from PIL import Image
import PySimpleGUI as sg

# Set file types
#file_types = [()]

# Set window layout
selection = [
    sg.Text('Select an image for training: '),
    sg.Input(size=(35, 1), disabled=True, enable_events=True, key="-FILE-"),
    sg.FileBrowse(),
]

layout = [
    selection,
    [
        sg.Image(size=(250, 250), key='-IMAGE-'),
        sg.Button('Train'),
    ]
]

# Create window
window = sg.Window('SinGAN reloaded', layout)

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
            window['-IMAGE-'].update(data=bits.getvalue())
    if event == 'Train':
        print('TEST')
