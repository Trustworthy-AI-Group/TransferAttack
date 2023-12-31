import torch

from ..utils import *
from ..gradient.mifgsm import MIFGSM


class MaskBlock(MIFGSM):
    """
    MaskBlock (Mask Block Attack)
    'MaskBlock: Transferable Adversarial Examples with Bayes Approach' (https://arxiv.org/abs/2208.06538)

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
        device (torch.device): the device for data. If it is None, the device would be same as model.
        patch_size: the patch size.
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=2/255, epoch=10, decay=1., patch_size=56

    Example script:
        python main.py --attack maskblock --output_dir adv_data/maskblock/resnet18
        python main.py --attack maskblock --output_dir adv_data/maskblock/resnet18 --eval
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=2/255, epoch=10, decay=1., patch_size=56, targeted=False, random_start=False, 
                norm='linfty', loss='crossentropy', device=None, attack='MaskBlock', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay=decay, targeted=targeted, random_start=random_start, norm=norm, loss=loss, device=device, attack=attack, **kwargs)
        self.patch_size = patch_size
        self.num = 0

    def transform(self, x, **kwargs):
        _, _, w, h = x.shape
        y_axis = [i for i in range(0, h+1, self.patch_size)]
        x_axis = [i for i in range(0, w+1, self.patch_size)]
        self.num = 0
        xs = []
        for i, idx_x in enumerate(x_axis[1:]):
            for j, idx_y in enumerate(y_axis[1:]):
                x_copy = x.clone()
                x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y] = 0
                xs.append(x_copy)
                self.num += 1
        return torch.cat(xs, dim=0)

    def get_loss(self, logits, label):
        """
        Calculate the loss
        """
        return -self.loss(logits, label.repeat(self.num)) if self.targeted else self.loss(logits, label.repeat(self.num))