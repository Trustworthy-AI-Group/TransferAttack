import torch

from ..utils import *
from ..attack import Attack

class IFGSM(Attack):
    """
    I-FGSM Attack
    'Adversarial Examples in the Physical World (ICLR 2017)'(https://arxiv.org/abs/1607.02533)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        targeted (bool): targeted/untargeted attack
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, targeted=False, random_start=False,
                norm='linfty', loss='crossentropy',device=None, attack='I-FGSM', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = 0