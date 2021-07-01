"""SinGAN network"""
import torch
import torch.nn as nn
from skimage import io
import numpy as np
from src.architecture import Generator, Discriminator
from src.image import pyramid, load_img, plot_pyr, get_pyr_shapes, gaussian_smoothing
from skimage.color import rgb2lab, lab2rgb
import src.util as util
import matplotlib.pyplot as plt
import pickle

TRAIN_PATH = './train'

class SinGAN:
    """Generative model which uses only on image to train on."""

    def __init__(self, device, lr, lam, alpha, iters, sample_interval,  network_iters, path, rmse_factor=0.1):
        """Constructor

        Args:
            device (torch.device): device to train on, either cuda or cpu
            lr (float): learning rate
            lam (float): gradient penalty lambda
            alpha (float): weighting of reconstruction loss
            iters (int): iterations per scale
            sample_interval (int): iteration intervals to sample test images
            network_iters (int): iterations per network update
            path (str): image path
            rmse_factor (float): Scaling factor for rmse error
        """

        # Init variables
        self.device = device
        self.lr = lr
        self.lam = lam
        self.alpha = alpha
        self.iters = iters
        self.sample_interval = sample_interval
        self.network_iters = network_iters
        self.criterion = nn.MSELoss()
        self.zero_pad = nn.ZeroPad2d(5)
        self.trained_scale = 0
        self.rmse_factor = rmse_factor

        self.sigma = [1.0]
        self.z_recon = []

        # Load image
        if path is not None:
            self.img = load_img(path, device)
            max_dim = torch.as_tensor(self.img.shape[1:]).int().max().item()

            # Based on the paper, we choose the parameters that the
            # coarsest scale is 25px and r is as near as possible to 4/3
            self.N = int(np.round(np.log(max_dim/25)/np.log(4/3) + 1))
            self.r = (max_dim/25)**(1/(self.N-1))
            print(f'Number of scales: {self.N} and scale factor: {self.r}')
        else:
            self.img = None
            max_dim = 0
            self.N = 0
            self.r = 0

        # Init generators and discriminators
        self.G = [Generator(32*(int(i/4)+1), lr).to(self.device) for i in range(self.N)]
        self.D = [Discriminator(32*(int(i/4)+1), lr).to(self.device) for i in range(self.N)]

    def noise(self, N, shape=(25, 25)):
        """Samples latent space noise.

        Args:
            N (int): Samples noise from first to the N-th scale
            shape (tuple, optional): Shape of noise at first scale. Defaults to (25, 25).

        Returns:
            List: List of noise maps for each Scale with length N
        """
        # Init variables
        z = []
        noise_shape = torch.tensor(shape)

        # For every scale n, create the noise map with right scale
        for n in range(N):
            # For first scale use noise map for all channels
            if n == 0:
                z_n = torch.randn((1, 1,) + tuple(noise_shape.int().tolist())).to(self.device)
                z_n = z_n.expand(-1, 3, -1, -1)
            else:
                z_n = torch.randn((1, 3,) + tuple(noise_shape.int().tolist())).to(self.device)
            z.append(z_n*self.sigma[n])
            noise_shape = noise_shape*self.r
        return z

    def sample_img(self, N, shape=(25, 25), z=None, start=0, start_img=None):
        """Samples an image from the SinGAN.

        Args:
            N (int): scale to sample from
            shape (tuple, optional): Shape of image at first scale. Defaults to (25, 25).
            z (List, optional): Noise Maps. If nothing is given, the noise is sampled. Defaults to None.

        Returns:
            List<torch.tensor>: List of output images of different scales
        """
        # Sample noise maps if not present
        if z is None:
            z = self.noise(N, shape=shape)

        # First scale only receives noise
        x = []
        if start_img is None:
            x_n = torch.zeros(z[0].shape).to(self.device)
        else:
            # Or use the start image if given for paint to image task
            x_n = start_img

        # Go through all scales and create image
        for n in range(start, N):
            upsample = torch.nn.Upsample(size=tuple(z[n].shape[2:]), mode='bilinear', align_corners=True)

            x_n = upsample(x_n)
            x_n = self.G[n](z[n], x_n)
            x.append(x_n)

        return x

    def save(self):
        """Saves the models."""
        self.G[self.trained_scale-1].save(TRAIN_PATH + f'/G_{self.trained_scale-1}.pt')
        self.D[self.trained_scale-1].save(TRAIN_PATH + f'/D_{self.trained_scale-1}.pt')

        with open(TRAIN_PATH + '/SinGAN.pkl', 'wb') as f:
            pickle.dump({
            'trained_scale': self.trained_scale,
            'img': self.img,
            'N': self.N,
            'r': self.r,
            'sigma': self.sigma,
            'z_recon': self.z_recon,
            }, f)

    def load(self):
        """Loads the models."""
        with open(TRAIN_PATH + '/SinGAN.pkl', 'rb') as f:
            checkpoint = pickle.load(f)

        self.trained_scale = checkpoint['trained_scale']
        self.img = checkpoint['img']
        self.N = checkpoint['N']
        self.r = checkpoint['r']
        self.sigma = checkpoint['sigma']
        self.z_recon = checkpoint['z_recon']

        # Recreate Generators and Discriminators
        self.G = [Generator(32*(int(i/4)+1), self.lr).to(self.device) for i in range(self.N)]
        self.D = [Discriminator(32*(int(i/4)+1), self.lr).to(self.device) for i in range(self.N)]

        for i in range(self.trained_scale):
            self.G[i].load(TRAIN_PATH + f'/G_{i}.pt')
            self.D[i].load(TRAIN_PATH + f'/D_{i}.pt')
        print(f'Loaded SinGAN, model was trained up to scale {self.trained_scale-1} of {self.N-1}!')

    def train(self):
        """Trains the SinGAN architecture."""
        # Get image pyramid scales
        shapes = get_pyr_shapes(self.N, self.r)

        # Create image pyramid
        pyr = pyramid(self.img, shapes, self.device)

        # Train each scale one after the other
        start = self.trained_scale+1
        for n in range(start, self.N+1):
            # Get training image of the respective scale
            real_img = pyr[n-1]

            for i in range(self.iters+1):
                # For first scale create reconstruction noise
                if n == 1:
                    # In the paper they said only once, however, in their code
                    # they do it every iteration for the first scale??
                    z_start = torch.randn((1, 3, 25, 25)).to(self.device)
                    self.z_recon = [z_start]

                # Draw noise for this scale
                z = self.noise(n)

                # ---------------------
                #  Train Discriminator
                # ---------------------
                for _ in range(self.network_iters):
                    self.D[n-1].optimizer.zero_grad()

                    # Real images
                    real_validity = self.D[n-1](real_img)

                    # Sample only for the current scale
                    fake_img = self.sample_img(n, z=z)[-1]

                    # Fake images
                    fake_validity = self.D[n-1](fake_img.detach())

                    # Gradient penalty
                    gradient_penalty = util.calculate_gradient_penalty(real_img, fake_img, self.D[n-1], self.device, self.lam)

                    # Adversarial loss
                    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty
                    d_loss.backward()
                    self.D[n-1].optimizer.step()

                # -----------------
                #  Train Generator
                # -----------------
                for _ in range(self.network_iters):
                    self.G[n-1].optimizer.zero_grad()

                    # Generate a batch of images
                    fake_img = self.sample_img(n, z=z)[-1]

                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = self.D[n-1](fake_img)

                    # Train generator
                    g_loss = -torch.mean(fake_validity)
                    g_loss.backward(retain_graph=True)

                    # Reconstruction image
                    recon_img = self.sample_img(n, z=self.z_recon)[-1]

                    # Reconstruction loss
                    loss = self.alpha*self.criterion(recon_img, real_img)
                    loss.backward()
                    self.G[n-1].optimizer.step()

                if i % 10 == 0:
                    print(
                        "[Scale %d/%d] [Iter %d/%d] [D loss: %f] [G loss: %f]"
                        % (n-1, self.N-1, i, self.iters, d_loss.item(), g_loss.item())
                    )

                if i % self.sample_interval == 0:
                    # Save sampled image
                    img = fake_img[0].cpu().detach().permute(1, 2, 0)
                    img = img.numpy()
                    img[:,:,0] += 1
                    img[:,:,0] *= 50
                    img[:,:,1:] *= 127.5
                    img[:,:,1:] -= 0.5
                    img = (lab2rgb(img)*255).astype(np.uint8)
                    io.imsave(TRAIN_PATH + f'/{n-1}_sample.png', img)

                    # Save reconstructed image
                    img = recon_img[0].cpu().detach().permute(1, 2, 0)
                    img = img.numpy()
                    img[:,:,0] += 1
                    img[:,:,0] *= 50
                    img[:,:,1:] *= 127.5
                    img[:,:,1:] -= 0.5
                    img = (lab2rgb(img)*255).astype(np.uint8)
                    io.imsave(TRAIN_PATH + f'/{n-1}_recon.png', img)

                # Scheduler
                self.G[n-1].scheduler.step()
                self.D[n-1].scheduler.step()

            if n < self.N:
                # Calculate the RMSE to get the next sigma_n
                recon_img = self.sample_img(n, z=self.z_recon)[-1]
                upsample = torch.nn.Upsample(size=tuple(shapes[n]), mode='bilinear', align_corners=True)

                recon_img  = upsample(recon_img)
                rmse = torch.sqrt(self.criterion(recon_img, pyr[n]))
                self.sigma.append(rmse.item()*self.rmse_factor)

                # Add zero noise map to reconstructions noise maps
                self.z_recon.append(torch.zeros((1, 3,) + shapes[n], device=self.device))

            # Update trained scale
            self.trained_scale = n

            # Save models
            self.save()

    def generate(self, num_imgs, output_shape):
        """Generates new images.

        Args:
            num_imgs (int): Number of images
            output_shape (tuple or List): 2-dimensional shape (height, width) of output images

        Returns:
            List: Sampled output images
        """
        # Get shape of smallest scale
        shape = np.array(output_shape)
        shape = shape / (self.r ** (self.trained_scale-1))
        shape = np.round(shape).astype(np.int32)

        # Sample images
        imgs = [self.sample_img(self.trained_scale, shape=shape)[-1] for _ in range(num_imgs)]

        return imgs

    def paint_to_img(self, clip_art, injection_scale=1):
        """Transforms a clip art into a realistic image using SinGAN

        Args:
            clip_art (torch.tensor): (B, C, H, W) clip art image
            injection_scale (int, optional): Start scale to inject clip art into. Defaults to 1.

        Returns:
            torch.tensor: realistic image
        """

        # Get start shape, ratio of painted image should be preserved
        H, W = clip_art.shape[2:]
        img_ratio = min(H,W)/max(H,W)
        width = 25*self.r**(injection_scale-1)

        if H > W:
            shape = (int(width), int(width*img_ratio))
        else:
            shape = (int(width*img_ratio), int(width))

        # Downscale clip_art
        clip_art = gaussian_smoothing(clip_art, self.device)
        clip_art = torch.nn.functional.interpolate(clip_art, size=shape, mode='bicubic')

        # Sample image
        img = self.sample_img(self.trained_scale, shape=shape, start=injection_scale, start_img=clip_art)[-1]

        return img