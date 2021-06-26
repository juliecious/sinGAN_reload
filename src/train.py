import torch
import torch.nn as nn
from skimage import io
import numpy as np
from architecture import Generator, Discriminator
from image import pyramid, load_img, plot_pyr
import matplotlib.pyplot as plt
import util

# Init variables
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
lr = 5e-4
iters = 2000
batch_size = 1
interpolate_mode = 'nearest'
sample_interval = 500
lam = 0.1
network_iters=1
alpha = 10
criterion = nn.MSELoss()

# Load image
img = load_img('../assets/test.jpg', device)
max_dim = torch.as_tensor(img.shape[1:]).int().max().item()

# Based on the paper, we choose the parameters that the
# coarsest scale is 25px and r is as near as possible to 4/3
N = int(np.round(np.log(max_dim/25)/np.log(4/3) + 1))
r = (max_dim/25)**(1/(N-1))
zero_pad = nn.ZeroPad2d(5)

print(f'Number of scales: {N} and scale factor: {r}')

# Init generators and discriminators
G = [Generator(32*(int(i/4)+1), lr).to(device) for i in range(N)]
D = [Discriminator(32*(int(i/4)+1), lr).to(device) for i in range(N)]

def get_pyr_shapes(N, r, start_shape=(25, 25)):
    """Get the shape of the images/noise maps of each scale."""

    # Init variables
    shape = torch.tensor(start_shape)
    shapes = []

    # For every scale n, create the noise map with right scale
    for n in range(N):
        shapes.append(tuple(shape.int().tolist()))
        shape = shape*r
    return shapes

def noise(N, r, sigma, batch_size=1, shape=(25, 25)):
    # Init variables
    z = []
    noise_shape = torch.tensor(shape)

    # For every scale n, create the noise map with right scale
    for n in range(N):
        # For first scale use noise map for all channels
        if n == 0:
            z_n = torch.randn((batch_size, 1,) + tuple(noise_shape.int().tolist())).to(device)
            z_n = z_n.expand(-1, 3, -1, -1)
        else:
            z_n = torch.randn((batch_size, 3,) + tuple(noise_shape.int().tolist())).to(device)
        z.append(z_n*sigma[n])
        noise_shape = noise_shape*r
    return z

def sample_img(high, r, batch_size, shape=(25, 25), z=None, sigma=None):
    # Sample noise maps if not present
    if z is None:
        z = noise(high, r, sigma, batch_size=batch_size, shape=shape)

    # First scale only receives noise
    x = []
    x_n = torch.zeros(z[0].shape).to(device)

    # Go through all scales and create image
    for n in range(high):
        upsample = torch.nn.Upsample(size=tuple(z[n].shape[2:]))

        x_n = upsample(x_n)
        x_n = G[n](z[n], x_n)
        x.append(x_n)

    return x

def train(N, r, iters, batch_size, img):
    # Get image pyramid scales
    shapes = get_pyr_shapes(N, r)

    # Create image pyramid
    pyr = pyramid(img, shapes, device)

    # Init sigmas
    sigma = [1.0]

    # Train each scale one after the other
    for n in range(1, N+1):
        # Get training image of the respective scale
        real_img = pyr[n-1]

        for i in range(iters+1):
            # For first scale create reconstruction noise
            if n == 1:
                # In the paper they said only once, however, in their code
                # they do it every iteration for the first scale??
                z_start = torch.randn((batch_size, 3, 25, 25)).to(device)
                z_recon = [z_start]

            # Draw noise for this scale
            z = noise(n, r, sigma)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            for _ in range(network_iters):
                D[n-1].optimizer.zero_grad()

                # Real images
                real_validity = D[n-1](real_img)

                # Sample only for the current scale
                fake_img = sample_img(n, r, batch_size, z=z)[-1]

                # Fake images
                fake_validity = D[n-1](fake_img.detach())

                # Gradient penalty
                gradient_penalty = util.calculate_gradient_penalty(real_img, fake_img, D[n-1], device, lam)

                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty
                d_loss.backward()
                D[n-1].optimizer.step()

            # -----------------
            #  Train Generator
            # -----------------
            for _ in range(network_iters):
                G[n-1].optimizer.zero_grad()

                # Generate a batch of images
                fake_img = sample_img(n, r, batch_size, z=z)[-1]

                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = D[n-1](fake_img)

                # Train generator
                g_loss = -torch.mean(fake_validity)
                g_loss.backward(retain_graph=True)

                # Reconstruction image
                recon_img = sample_img(n, r, batch_size, z=z_recon)[-1]

                # Reconstruction loss
                loss = alpha*criterion(recon_img, real_img)
                loss.backward()
                G[n-1].optimizer.step()

            if i % 10 == 0:
                print(
                    "[Scale %d/%d] [Iter %d/%d] [D loss: %f] [G loss: %f]"
                    % (n-1, N-1, i, iters, d_loss.item(), g_loss.item())
                )

            if i % sample_interval == 0:
                # Save sampled image
                img = fake_img[0].cpu().detach().permute(1, 2, 0)
                img = (img + 1) / 2
                img = (img*255).type(torch.uint8).numpy()
                io.imsave(f'../train/{n-1}_{i}_sample.png', img)

                # Save reconstructed image
                img = recon_img[0].cpu().detach().permute(1, 2, 0)
                img = (img + 1) / 2
                img = (img*255).type(torch.uint8).numpy()
                io.imsave(f'../train/{n-1}_{i}_recon.png', img)

            # Scheduler
            G[n-1].scheduler.step()
            D[n-1].scheduler.step()

        if n < N:
            # Calculate the RMSE to get the next sigma_n
            recon_img = sample_img(n, r, batch_size, z=z_recon)[-1]
            upsample = torch.nn.Upsample(size=tuple(shapes[n]))

            recon_img  = upsample(recon_img)
            rmse = torch.sqrt(criterion(recon_img, pyr[n]))
            sigma.append(rmse.item()*0.1)

            # Add zero noise map to reconstructions noise maps
            z_recon.append(torch.zeros((batch_size, 3,) + shapes[n], device=device))

        # Save models
        G[n-1].save(f'../train/gen{n-1}.pt')
        D[n-1].save(f'../train/disc{n-1}.pt')

train(N, r, iters, batch_size, img)
