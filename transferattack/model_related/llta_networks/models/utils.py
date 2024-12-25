import math
import numpy as np
import scipy.stats as st
import torch
import torch.nn as nn
import torch.nn.functional as F


class Gaussian:
    def __init__(self, loc, scale=0.001):
        self.loc = loc
        self.scale = scale

    def sample_like(self, input_tensor):
        s = np.random.normal(loc=0., scale=self.scale, size=input_tensor.shape)
        s = torch.FloatTensor(s).to(input_tensor.device)
        return s
    
    def sample(self, size, device='cpu'):
        s = np.random.normal(loc=0., scale=self.scale, size=size)
        s = torch.FloatTensor(s).to(device)
        return s

    def prob(self, x):
        diff = x - self.loc
        diff = (diff / self.scale) ** 2
        coef = ((2 * math.pi) ** 0.5) * self.scale
        return (-0.5 * diff).exp() / coef


def get_Gaussian_kernel(nsig, kernlen):
    # define Gaussian kernel
    kern1d = st.norm.pdf(np.linspace(-nsig, nsig, kernlen))
    kernel = np.outer(kern1d, kern1d)
    kernel = kernel / kernel.sum()
    return kernel
