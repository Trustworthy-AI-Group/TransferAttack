import torch

from ..utils import *
from ..attack import Attack

class VMIFGSM(Attack):
    """
    VMI-FGSM Attack
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
        python main.py --input_dir ./path/to/data --output_dir adv_data/vmifgsm/resnet18 --attack vmifgsm --model=resnet18
        python main.py --input_dir ./path/to/data --output_dir adv_data/vmifgsm/resnet18 --eval
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, beta=1.5, num_neighbor=20, epoch=10, decay=1., targeted=False, 
                random_start=False, norm='linfty', loss='crossentropy', device=None, attack='VMI-FGSM', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.radius = beta * epsilon
        self.epoch = epoch
        self.decay = decay
        self.num_neighbor = num_neighbor

    def get_variance(self, data, delta, label, cur_grad, momentum, **kwargs):
        """
        Calculate the gradient variance    
        """
        grad = 0
        for _ in range(self.num_neighbor):
            # Obtain the output
            # This is inconsistent for transform!
            logits = self.get_logits(self.transform(data+delta+torch.zeros_like(delta).uniform_(-self.radius, self.radius).to(self.device), momentum=momentum))

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad += self.get_grad(loss, delta)

        return grad / self.num_neighbor - cur_grad

    def forward(self, data, label, **kwargs):
        """
        The attack procedure for VMI-FGSM

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum, variance = 0, 0
        for _ in range(self.epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad+variance, momentum)

            # Calculate the variance
            variance = self.get_variance(data, delta, label, grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()