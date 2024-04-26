import torch
import torch.nn.functional as F

from ..utils import *
from ..gradient.mifgsm import MIFGSM

class USMM(MIFGSM):
    """
    Uniform Scale and Mix Mask (USMM) Attack
    'Boost Adversarial Transferability by Uniform Scale and Mix Mask Method'(https://arxiv.org/pdf/2311.12051.pdf)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        scale_low (float): the lower bound of the scale factor.
        scale_high (float): the upper bound of the scale factor.
        num_scale (int): the number of scaled copies in each iteration.
        num_mix (int): the number of mixed images in each iteration.
        mix_range (float): the mix range size.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., scale_low=0.1, scale_high=0.75, num_scale=5, num_mix=3, mix_range=0.5

    Example script:
        python main.py --attack usmm --output_dir adv_data/mig/resnet18 # attack
        python main.py --attack usmm --output_dir adv_data/mig/resnet18 --eval # evaluation
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., scale_low=0.1, scale_high=0.75, num_scale=5, num_mix=3, mix_range=0.5, targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None, attack='USMM', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.num_scale = num_scale
        self.num_mix = num_mix
        self.mix_range = mix_range

    def transform(self, x, **kwargs):
        # Uniform scale
        scales = [self.scale_low + (self.scale_high - self.scale_low) * i / (self.num_scale - 1) for i in range(self.num_scale)]
        x_scales = [x * scale for scale in scales]

        # Mix mask
        mixed_images = torch.cat([x_scale * ((1 - self.mix_range) * torch.ones_like(x) + 2 * self.mix_range * x[torch.randperm(x.size(0))].detach()) for _ in range(self.num_mix) for x_scale in x_scales], dim=0)

        # Return clamped images
        return torch.clamp(mixed_images, 0, 1)
        
    def get_loss(self, logits, label):
        """
        Calculate the loss
        """
        return -self.loss(logits, label.repeat(self.num_scale*self.num_mix)) if self.targeted else self.loss(logits, label.repeat(self.num_scale*self.num_mix))
    
    def forward(self, data, label, **kwargs):
        """
        The US-MM attack procedure

        Arguments:
            data (N, C, H, W): tensor for input images
            labels (N,): tensor for ground-truth labels if untargetd
            labels (2,N): tensor for [ground-truth, targeted labels] if targeted
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):
            # Obtain the output
            x_trans = self.transform(data+delta, momentum=momentum).clone().detach().to(self.device)
            x_trans.requires_grad = True
            logits = self.get_logits(x_trans)

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad = self.get_grad(loss, x_trans)
            grad = torch.sum(torch.stack(grad.split(data.shape[0])), dim=0)
            
            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()