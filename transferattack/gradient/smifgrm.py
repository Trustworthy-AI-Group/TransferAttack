import torch
import torch.nn.functional as F

from ..utils import *
from ..attack import Attack

class SMIFGRM(Attack):
    """
    SMI-FGRM Attack
    'Sampling-based Fast Gradient Rescaling Method for Highly Transferable Adversarial Attacks'(https://arxiv.org/abs/2307.02828)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        beta (float): the sampling range.
        num_neighbor (int): the number of samples for calculating average gradients.
        rescale_factor (int): the rescale factor
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, beta=1.5, num_neighbor=12, epoch=10, decay=1., rescale_factor=2

    Example script:
        python main.py --attack smifgsm --output_dir adv_data/smifgsm/resnet18
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, beta=1.5, num_neighbor=12, rescale_factor=2, epoch=10, decay=1., targeted=False, 
                random_start=False, norm='linfty', loss='crossentropy', device=None, attack='SMI-FGRM', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.radius = beta * epsilon
        self.epoch = epoch
        self.decay = decay
        self.num_neighbor = num_neighbor
        self.rescale_factor = rescale_factor

    def get_sampled_grad(self, data, delta, label, momentum, **kwargs):
        """
        Calculate the sampled gradients    
        """
        grad = 0
        samples = data + delta
        for _ in range(self.num_neighbor):
            # Obtain the output
            logits = self.get_logits(self.transform(samples))

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad += self.get_grad(loss, delta)

            samples += torch.zeros_like(delta).uniform_(-self.radius, self.radius).to(self.device)

        return grad / self.num_neighbor
    
    def rescale_grad(self, grad, **kwargs):
        """
        Rescale the gradient
        """
        log_abs_grad = grad.abs().log2()
        grad_mean = torch.mean(log_abs_grad, dim=(1,2,3), keepdim=True)
        grad_std = torch.std(log_abs_grad, dim=(1,2,3), keepdim=True)
        norm_grad = (log_abs_grad - grad_mean) / grad_std
        return self.rescale_factor * grad.sign() * torch.sigmoid(norm_grad)

    def forward(self, data, label, **kwargs):
        """
        The attack procedure for VMI-FGSM

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):
            # Obtain the output
            grad = self.get_sampled_grad(data, delta, label, momentum)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Rescale the momentum
            momentum = self.rescale_grad(momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()