import torch

from ..utils import *
from ..attack import Attack

class IFGSSM(Attack):
    """
    I-FGSSM Attack
    'Staircase Sign Method for Boosting Adversarial Attacks'(https://arxiv.org/abs/2104.09722)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        targeted (bool): targeted/untargeted attack
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        k (float): percentile interval 
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, k=1.5625
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, targeted=False, random_start=False, 
                norm='linfty', loss='crossentropy', device=None, k=1.5625,**kwargs):
        super().__init__('I-FGSSM', model_name, epsilon, targeted, random_start, norm, loss, device, **kwargs)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = 0
        self.k = k
    def ssign(self, noise):
        noise_staircase = torch.zeros_like(noise)
        N, C, H, W = noise.size()
        medium = []
        sign = torch.sign(noise)
        temp_noise = noise
        abs_noise = abs(noise)
        base = self.k / 100
        for i in np.arange(self.k, 100.1, self.k):
            medium_now = torch.quantile(abs_noise.reshape(-1, H*W), q = float(i/100), dim = 1, keepdim = True, interpolation='lower').reshape(N, C, 1, 1)
            medium.append(medium_now)
        for j in range(len(medium)):
            update = sign * (abs(temp_noise) <= medium[j]).float() * (base + 2 * base * j)
            noise_staircase += update
            temp_noise += update * 1e5

        return noise_staircase
    def update_delta(self, delta, data, grad, alpha, **kwargs):
        if self.norm == 'linfty':
            delta = torch.clamp(delta + alpha * self.ssign(grad), -self.epsilon, self.epsilon)
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
            scaled_grad = grad / (grad_norm + 1e-20)
            delta = (delta + scaled_grad * alpha).view(delta.size(0), -1).renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(delta)
        delta = clamp(delta, img_min-data, img_max-data)
        return delta
