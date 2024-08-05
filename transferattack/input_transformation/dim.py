import torch
import torch.nn.functional as F

from ..utils import *
from ..gradient.mifgsm import MIFGSM

class DIM(MIFGSM):
    """
    DIM Attack
    'Improving Transferability of Adversarial Examples with Input Diversity (CVPR 2019)'(https://arxiv.org/abs/1803.06978)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        resize_rate (float): the relative size of the resized image
        diversity_prob (float): the probability for transforming the input image
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
    
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1, resize_rate=1.1, diversity_prob=0.5

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/dim/resnet18 --attack dim --model=resnet18
        python main.py --input_dir ./path/to/data --output_dir adv_data/dim/resnet18 --eval
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., resize_rate=1.1, diversity_prob=0.5, targeted=False, 
                random_start=False, norm='linfty', loss='crossentropy', device=None, attack='DIM', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        if resize_rate < 1:
            raise Exception("Error! The resize rate should be larger than 1.")
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
    
    def transform(self, x, **kwargs):
        """
        Random transform the input images
        """
        # do not transform the input image
        if torch.rand(1) > self.diversity_prob:
            return x
        
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        # resize the input image to random size
        rnd = torch.randint(low=min(img_size, img_resize), high=max(img_size, img_resize), size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)

        # randomly add padding
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        # resize the image back to img_size
        return F.interpolate(padded, size=[img_size, img_size], mode='bilinear', align_corners=False)