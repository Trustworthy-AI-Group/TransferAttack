import torch

from ..utils import *
from ..gradient.mifgsm import MIFGSM

class SIM(MIFGSM):
    """
    SIM Attack
    'Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks (ICLR 2020)'(https://arxiv.org/abs/1908.06281)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_scale (int): the number of scaled copies in each iteration.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., num_scale=5
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., num_scale=5, targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None, attack='SIM', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.num_scale = num_scale

    def transform(self, x, **kwargs):
        """
        Scale the input for SIM
        """
        return torch.cat([x / (2**i) for i in range(self.num_scale)])

    def get_loss(self, logits, label):
        """
        Calculate the loss
        """
        return -self.loss(logits, label.repeat(self.num_scale)) if self.targeted else self.loss(logits, label.repeat(self.num_scale))