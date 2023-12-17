import numpy as np
import torch
import torch.nn as nn

from ..gradient.mifgsm import MIFGSM
from ..utils import *


class SGM(MIFGSM):
    """
    SGM (Skip Gradient Method)
    'Skip Connections Matter: On the Transferability of Adversarial Examples Generated with ResNets (ICLR 2020)'(https://openreview.net/forum?id=BJlRs34Fvr)

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
        gamma (float): the decay factor for gradient from residual modules

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., gamma=0.2 (0.5) and 0.5 (0.7) on ResNet and DenseNet in PGD (FGSM)
        (in sgm official paper, epsilon=16/255, alpha=2/255)

    Example script:
        python main.py --attack=sgm --output_dir adv_data/sgm/resnet18
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, gamma=0.2, random_start=False, norm='linfty', loss='crossentropy', device=None, attack='SGM', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        if gamma < 1.0:
            if model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
                register_hook_for_resnet(self.model, arch=model_name, gamma=gamma)
            elif model_name in ['densenet121', 'densenet169', 'densenet201']:
                register_hook_for_densenet(self.model, arch=model_name, gamma=gamma)
            else:
                raise ValueError('Current code only supports resnet/densenet. '
                                'You are using {}'.format(model_name))

def backward_hook(gamma):
    """
    implement SGM through grad through ReLU
    (This code is copied from https://github.com/csdongxian/skip-connections-matter)
    """
    def _backward_hook(module, grad_in, grad_out):
        if isinstance(module, nn.ReLU):
            return (gamma * grad_in[0],)
    return _backward_hook


def backward_hook_norm(module, grad_in, grad_out):
    """
    normalize the gradient to avoid gradient explosion or vanish
    (This code is copied from https://github.com/csdongxian/skip-connections-matter)
    """
    std = torch.std(grad_in[0])
    return (grad_in[0] / std,)


def register_hook_for_resnet(model, arch, gamma):
    """
    register hook for resnet models
    (This code is copied from https://github.com/csdongxian/skip-connections-matter)
    """
    # There is only 1 ReLU in Conv module of ResNet-18/34
    # and 2 ReLU in Conv module ResNet-50/101/152
    if arch in ['resnet50', 'resnet101', 'resnet152']:
        gamma = np.power(gamma, 0.5)
    backward_hook_sgm = backward_hook(gamma)

    for name, module in model.named_modules():
        if 'relu' in name and not '0.relu' in name:
            module.register_backward_hook(backward_hook_sgm)

        # e.g., 1.layer1.1, 1.layer4.2, ...
        if len(name.split('.')) >= 2 and 'layer' in name.split('.')[-2]:
            module.register_backward_hook(backward_hook_norm)


def register_hook_for_densenet(model, arch, gamma):
    """
    register hook for densenet models
    (This code is copied from https://github.com/csdongxian/skip-connections-matter)
    """
    # There are 2 ReLU in Conv module of DenseNet-121/169/201.
    gamma = np.power(gamma, 0.5)
    backward_hook_sgm = backward_hook(gamma)
    for name, module in model.named_modules():
        if 'relu' in name and not 'transition' in name:
            module.register_backward_hook(backward_hook_sgm)
