import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable


def calculate_reconstruction_loss(recon_image, real_image):
    """
    Calcuate root mean squared error bewteen reconstructed image and the real image in each layer
    """
    return torch.sqrt(nn.MSELoss(recon_image, real_image))

def calculate_gradient_penalty(real_imgs, fake_imgs, d, device, lam):
    """Calculates the Wasserstein gradient penalty loss."""

    # Get batch size
    batch_size = real_imgs.shape[0]

    # Generate a random number epsilon for every image in the batch from a uniform distribution U[0,1]
    eps = torch.rand(batch_size, device=device)

    # Get interpolated images
    int_imgs = eps * real_imgs + (1 - eps) * fake_imgs
    int_imgs = int_imgs.to(device)
    int_imgs = Variable(int_imgs, requires_grad=True)

    # Calculate probability of interpolated examples
    int_prob = d(int_imgs)

    grad_outputs = torch.ones(int_prob.size(), device=device)

    # Calculate gradients of probabilities w.r.t examples
    grad = autograd.grad(outputs=int_prob,
                         inputs=int_imgs,
                         grad_outputs=grad_outputs,
                         create_graph=True,
                         retain_graph=True)[0]

    # Calculate the gradient penalty
    grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean() * lam

    return grad_penalty
