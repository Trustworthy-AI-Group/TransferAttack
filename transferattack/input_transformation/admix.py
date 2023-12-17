import torch

from ..utils import *
from ..gradient.mifgsm import MIFGSM

class Admix(MIFGSM):
    """
    Admix Attack
    'Admix: Enhancing the Transferability of Adversarial Attacks (ICCV 2021)'(https://arxiv.org/abs/2102.00436)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_scale (int): the number of scaled copies in each iteration.
        num_admix (int): the number of admixed images in each iteration.
        admix_strength (float): the strength of admixed images.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., num_scale=5, num_admix=3, admix_strength=0.2
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., num_scale=5, num_admix=3, admix_strength=0.2, targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None, attack='Admix', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.num_scale = num_scale
        self.num_admix = num_admix
        self.admix_strength = admix_strength

    def transform(self, x, **kwargs):
        """
        Admix the input for Admix Attack
        """
        admix_images = torch.concat([(x + self.admix_strength * x[torch.randperm(x.size(0))].detach()) for _ in range(self.num_admix)], dim=0)
        return torch.concat([admix_images / (2 ** i) for i in range(self.num_scale)])

    def get_loss(self, logits, label):
        """
        Calculate the loss
        """
        return -self.loss(logits, label.repeat(self.num_scale*self.num_admix)) if self.targeted else self.loss(logits, label.repeat(self.num_scale*self.num_admix))