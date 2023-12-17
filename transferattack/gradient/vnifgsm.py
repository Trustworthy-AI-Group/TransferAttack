import torch

from ..utils import *
from .vmifgsm import VMIFGSM

class VNIFGSM(VMIFGSM):
    """
    VNI-FGSM Attack
    'Enhancing the transferability of adversarial attacks through variance tuning (CVPR 2021)'(https://arxiv.org/abs/2103.15571)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        beta (float): the relative value for the neighborhood.
        num_neighbor (int): the number of samples for estimating the gradient variance.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, beta=1.5, num_neighbor=20, epoch=10, decay=1.

    Example script:
        python main.py --attack vnifgsm --output_dir adv_data/vnifgsm/resnet18
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, beta=1.5, num_neighbor=20, epoch=10, decay=1., targeted=False, 
                random_start=False, norm='linfty', loss='crossentropy', device=None, attack='VNI-FGSM', **kwargs):
        super().__init__(model_name, epsilon, alpha, beta, num_neighbor, epoch, decay, targeted, random_start, norm, loss, device, attack)
    
    def transform(self, x, momentum):
        """
        look ahead for NI-FGSM
        """
        return x + self.alpha*self.decay*momentum