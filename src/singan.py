"""SinGAN network"""
import torch
import torch.nn as nn
from skimage import io
import numpy as np
from src.architecture import Generator, Discriminator
from src.image import pyramid, load_img, plot_pyr, get_pyr_shapes
from skimage.color import rgb2lab, lab2rgb
import src.util as util
import matplotlib.pyplot as plt
import pickle

TRAIN_PATH = './train'

class SinGAN():
    def __init__(self, device, lr, lam, alpha, iters, sample_interval,  network_iters, path):
        """Constructor
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

        # Load image
        self.img = load_img(path, device)
        max_dim = torch.as_tensor(self.img.shape[1:]).int().max().item()

        # Based on the paper, we choose the parameters that the
        # coarsest scale is 25px and r is as near as possible to 4/3
        self.N = int(np.round(np.log(max_dim/25)/np.log(4/3) + 1))
        self.r = (max_dim/25)**(1/(self.N-1))
        self.zero_pad = nn.ZeroPad2d(5)
        self.trained_scale = 0

        print(f'Number of scales: {self.N} and scale factor: {self.r}')

        # Init generators and discriminators
        self.G = [Generator(32*(int(i/4)+1), lr).to(self.device) for i in range(self.N)]
        self.D = [Discriminator(32*(int(i/4)+1), lr).to(self.device) for i in range(self.N)]

    def noise(self, N, r, sigma, shape=(25, 25)):
        """Samples latent space noise."""
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
            z.append(z_n*sigma[n])
            noise_shape = noise_shape*r
        return z

    def sample_img(self, high, r, shape=(25, 25), z=None, sigma=None):
        """Samples image pyramid."""
        # Sample noise maps if not present
        if z is None:
            z = self.noise(high, r, sigma, shape=shape)

        # First scale only receives noise
        x = []
        x_n = torch.zeros(z[0].shape).to(self.device)

        # Go through all scales and create image
        for n in range(high):
            upsample = torch.nn.Upsample(size=tuple(z[n].shape[2:]), mode='bilinear', align_corners=True)

            x_n = upsample(x_n)
            x_n = self.G[n](z[n], x_n)
            x.append(x_n)

        return x

    def save(self):
        """Saves the models."""
        self.G[self.trained_scale-1].save(TRAIN_PATH + f'/G_{self.trained_scale-1}')
        self.D[self.trained_scale-1].save(TRAIN_PATH + f'/D_{self.trained_scale-1}')

        with open(TRAIN_PATH + '/SinGAN.pkl', 'wb') as f:
            pickle.dump({
            'trained_scale', self.trained_scale,
            'img', self.img,
            'N', self.N,
            'r', self.r
            }, f)

    def load(self):
        """Loads the models."""
        with open(TRAIN_PATH + '/SinGAN.pkl', 'rb') as f:
            checkpoint = pickle.load(f)

        self.G = checkpoint['generators']
        self.D = checkpoint['discriminators']
        self.trained_scale = checkpoint['trained_scale']

        print(f'Loaded SinGAN, model is trained to scale {self.trained_scale}!')

    def train(self):
        """Trains the SinGAN architecture."""
        # Get image pyramid scales
        shapes = get_pyr_shapes(self.N, self.r)

        # Create image pyramid
        pyr = pyramid(self.img, shapes, self.device)

        # Init sigmas
        sigma = [1.0]

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
                    z_recon = [z_start]

                # Draw noise for this scale
                z = self.noise(n, self.r, sigma)

                # ---------------------
                #  Train Discriminator
                # ---------------------
                for _ in range(self.network_iters):
                    self.D[n-1].optimizer.zero_grad()

                    # Real images
                    real_validity = self.D[n-1](real_img)

                    # Sample only for the current scale
                    fake_img = self.sample_img(n, self.r, z=z)[-1]

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
                    fake_img = self.sample_img(n, self.r, z=z)[-1]

                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = self.D[n-1](fake_img)

                    # Train generator
                    g_loss = -torch.mean(fake_validity)
                    g_loss.backward(retain_graph=True)

                    # Reconstruction image
                    recon_img = self.sample_img(n, self.r, z=z_recon)[-1]

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
                recon_img = self.sample_img(n, self.r, z=z_recon)[-1]
                upsample = torch.nn.Upsample(size=tuple(shapes[n]), mode='bilinear', align_corners=True)

                recon_img  = upsample(recon_img)
                rmse = torch.sqrt(self.criterion(recon_img, pyr[n]))
                sigma.append(rmse.item()*0.1)

                # Add zero noise map to reconstructions noise maps
                z_recon.append(torch.zeros((1, 3,) + shapes[n], device=self.device))

            # Update trained scale
            self.trained_scale = n

            # Save models
            self.save()
