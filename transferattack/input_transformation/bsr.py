import torch
import random

from ..utils import *
from ..gradient.mifgsm import MIFGSM

class BSR(MIFGSM):
    """
    BSR Attack
    'Boosting Adversarial Transferability by Block Shuffle and Rotation'(https://https://arxiv.org/abs/2308.10299)
    
    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_scale (int): the number of shuffled copies in each iteration.
        num_block (int): the number of block in the image.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., num_scale=10, num_block=3
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., num_scale=20, num_block=3, targeted=False, random_start=False,
                norm='linfty', loss='crossentropy', device=None, attack='SIM', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.num_scale = num_scale
        self.num_block = num_block

    def get_length(self, length):
        rand = np.random.uniform(size=self.num_block)
        rand_norm = np.round(rand/rand.sum()*length).astype(np.int32)
        rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
        return tuple(rand_norm)

    def shuffle_single_dim(self, x, dim):
        lengths = self.get_length(x.size(dim))
        x_strips = list(x.split(lengths, dim=dim))
        random.shuffle(x_strips)
        return x_strips


    def shuffle(self, x):
        dims = [2,3]
        random.shuffle(dims)
        x_strips = self.shuffle_single_dim(x, dims[0])
        return torch.cat([torch.cat(self.shuffle_single_dim(x_strip, dim=dims[1]), dim=dims[1]) for x_strip in x_strips], dim=dims[0])

    def transform(self, x, **kwargs):
        """
        Scale the input for BSR
        """
        return torch.cat([self.shuffle(x) for _ in range(self.num_scale)])

    def get_loss(self, logits, label):
        """
        Calculate the loss
        """
        return -self.loss(logits, label.repeat(self.num_scale)) if self.targeted else self.loss(logits, label.repeat(self.num_scale))
