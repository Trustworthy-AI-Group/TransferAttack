import torch

from ..utils import *
from ..attack import Attack

class ENS(Attack):
    """
    ENS Attack
    'Delving into Transferable Adversarial Examples and Black-box Attacks (ICLR 2017)'(https://arxiv.org/abs/1611.02770)

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

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/ens/ensemble --attack ens --model='resnet18,resnet101,resnext50_32x4d,densenet121'
        python main.py --input_dir ./path/to/data --output_dir adv_data/ens/ensemble --eval
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, random_start=False,
                norm='linfty', loss='crossentropy', device=None, attack='ENS', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay

