import torch

from ..utils import *
from ..attack import Attack

class DTA(Attack):
    """
    DTA Attack
    'Improving the Transferability of Adversarial Examples via Direction Tuning'(https://arxiv.org/abs/2303.15109)

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
        u (float): decay factor for gradient
        K (int): number of iterations
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, beta=1.5, num_neighbor=20, epoch=10, decay=1.
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, beta=1.5, K=10, u=0.8, epoch=10, decay=1., targeted=False, 
                random_start=False, norm='linfty', loss='crossentropy', device=None, attack='VMI-FGSM', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.radius = beta * epsilon
        self.epoch = epoch
        self.decay = decay
        self.K = K
        self.u = u

    
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

        momentum = 0
        for _ in range(self.epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))
    
            # Calculate the loss
            loss = self.get_loss(logits, label)
            # Calculate the gradients
            t_grad = self.get_grad(loss, delta)
            gt = t_grad.clone().detach()
            delta_tk = delta.clone().detach()
            delta_tk.requires_grad = True
            gtk = 0.
            momentum_tk = 0.
            for k in range(self.K):
                logits = self.get_logits(self.transform(data+delta_tk+gt, momentum=momentum))

                # Calculate the loss
                loss = self.get_loss(logits, label)

                # Calculate the gradients
                grad = self.get_grad(loss, delta_tk)
                gt = self.u*gt + grad/torch.norm(grad, p=1)
                # Calculate the momentum
                gtk = gtk + grad
                momentum_tk = self.get_momentum(grad, momentum_tk)

                # Update adversarial perturbation
                delta_tk = self.update_delta(delta_tk, data, momentum_tk, self.alpha)
            grad = self.decay*t_grad + gtk/self.K
            momentum = self.get_momentum(grad, momentum)
            delta = self.update_delta(delta, data, momentum, self.alpha)
        return delta.detach()