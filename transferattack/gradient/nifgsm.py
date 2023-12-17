import torch

from ..utils import *
from .mifgsm import MIFGSM

class NIFGSM(MIFGSM):
    """
    NI-FGSM Attack
    'Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks (ICLR 2020)'(https://arxiv.org/abs/1908.06281)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1.
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, random_start=False,
                norm='linfty', loss='crossentropy', device=None, attack='NI-FGSM', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)

    def transform(self, x, momentum, **kwargs):
        """
        look ahead for NI-FGSM
        """
        return x + self.alpha*self.decay*momentum