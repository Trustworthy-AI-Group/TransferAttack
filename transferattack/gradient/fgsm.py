import torch

from ..utils import *
from ..attack import Attack

class FGSM(Attack):
    """
    FGSM Attack
    'Explaining and Harnessing Adversarial Examples (ICLR 2015)'(https://arxiv.org/abs/1412.6572)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255
        
    Example script:
        python main.py --output_dir adv_data/fgsm/resnet18 --attack fgsm --model=resnet18
    """

    def __init__(self, model_name, epsilon=16/255, targeted=False, random_start=False, norm='linfty', loss='crossentropy',
                device=None, **kwargs):
        super().__init__('FGSM', model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = epsilon
        self.epoch = 1
        self.decay = 0
