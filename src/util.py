import torch
from torch import autograd

from torch.autograd import Variable


def move_to_gpu_if_available(t):
    if torch.cuda.is_available():
        return t.to(torch.device('cuda'))
    else: # use CPU
        return t.to(torch.device('cpu'))

def calculate_gradient_penalty(real_images, fake_images, discriminator, lambda_term=10):

    # generate a random number epsilon from a uniform distribution U[0,1]
    epsilon = torch.rand(1, 1).expand(real_images.size())
    epsilon = move_to_gpu_if_available(epsilon)

    interpolated = epsilon * real_images + (1 - epsilon) * fake_images
    interpolated = move_to_gpu_if_available(interpolated)
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = discriminator(interpolated)

    grad_outputs = torch.ones(prob_interpolated.size())

    # calculate gradients of probabilities w.r.t examples
    gradients = autograd.grad(outputs=prob_interpolated,
                              inputs=interpolated,
                              grad_outputs=move_to_gpu_if_available(grad_outputs),
                              create_graph=True,
                              retain_graph=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term

    return gradient_penalty