import torch
import torch.nn.functional as F

from ..utils import *
from ..gradient.mifgsm import MIFGSM

import scipy.stats as st
import numpy as np

class TIM(MIFGSM):
    """
    TIM Attack
    'Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks (CVPR 2019)'(https://arxiv.org/abs/1904.02884)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        kernel_type (str): the type of kernel (gaussian/uniform/linear).
        kernel_size (int): the size of kernel.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., kernel_type='gaussian', kernel_size=15

    Example script:
        python main.py --attack tim --output_dir adv_data/tim/resnet18
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., kernel_type='gaussian', kernel_size=15, targeted=False, 
                random_start=False, norm='linfty', loss='crossentropy', device=None, attack='TIM', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.kernel = self.generate_kernel(kernel_type, kernel_size)

    def generate_kernel(self, kernel_type, kernel_size, nsig=3):
        """
        Generate the gaussian/uniform/linear kernel

        Arguments:
            kernel_type (str): the method for initilizing the kernel
            kernel_size (int): the size of kernel
        """
        if kernel_type.lower() == 'gaussian':
            x = np.linspace(-nsig, nsig, kernel_size)
            kern1d = st.norm.pdf(x)
            kernel_raw = np.outer(kern1d, kern1d)
            kernel = kernel_raw / kernel_raw.sum()
        elif kernel_type.lower() == 'uniform':
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        elif kernel_type.lower() == 'linear':
            kern1d = 1 - np.abs(np.linspace((-kernel_size+1)//2, (kernel_size-1)//2, kernel_size)/(kernel_size**2))
            kernel_raw = np.outer(kern1d, kern1d)
            kernel = kernel_raw / kernel_raw.sum()
        else:
            raise Exception("Unspported kernel type {}".format(kernel_type))
        
        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return torch.from_numpy(stack_kernel.astype(np.float32)).to(self.device)

    def get_grad(self, loss, delta, **kwargs):
        """
        Overridden for TIM attack.
        """
        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
        grad = F.conv2d(grad, self.kernel, stride=1, padding='same', groups=3)
        return grad