# SinGAN reload

SinGAN reload is a PyTorch reimplementation of the paper "SinGAN: Learning a Generative Model from a Single Natural Image" (ICCV 2019, [Arxiv](https://arxiv.org/pdf/1905.01164.pdf)). It enables users to transfer a paint into a realistic image.

We intend to re-implement the paint-to-image task from SinGan, and extend the application to digital pathology use case. Furthermore, we aim to provide a user-friendly frontend platform for the above mentioned image manipulation tasks.

![img.png](assets/slice.png)

![img.png](assets/spongebob.png)


## Installation

Run the following command to install dependencies.
```bash
pip install -r requirements.txt
```

## Training

Now, put your training images under the assets folder and execute:
```bash
python train.py --path ./assets/[your_image_name]
```

Depending on the hardware the training takes about 1-2 hours. To train only on the cpu use `--device cpu`.
During training, a ./train folder with all network weights should be created. That's the default folder
and all other scripts will search for it.

# User Interface

To use the UI instead, execute:

```bash
python ui.py
```

You can set the hyperparameters of the algorithm like in training via extra arguments. Use `--help` to get information about other possible arguments.

## Generating new images

After training, you can create new images using:
```bash
python generate.py
```

It will load the SinGAN model from the ./train folder and save the generated images in it.
Use `--help` to get information about other possible arguments.

## Paint to image

After training, you will need an abstract clip art of the desired image. Then you can use:
```bash
python inject.py --path ./assets/[clip_art_name] --scale 1
```

Use scales [0-2]. The resulting image will also be saved under ./train.

## Harmonization

Train with the base image. After training, you will need the combined image. Then you can use:
```bash
python inject.py --path ./assets/[clip_art_name] --scale N
```

Use higher scales. The resulting image will also be saved under ./train.