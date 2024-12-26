import torch
import torch.nn as nn
import torch.nn.functional as F

class DecayReLU(nn.Module):
    """
    forward:  out  = relu(x)
    backward: grad = gamma * relu(x)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, gamma):
        gamma = gamma.view(x.shape[0], 1, 1, 1)
        x = F.relu(x)
        return x * gamma - x.data * gamma + x.data
