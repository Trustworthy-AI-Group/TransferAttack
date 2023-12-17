import torch
from ..utils import *
from ..attack import Attack

class PGN(Attack):
    """
    PGN (Penalizing Gradient Norm)
    'Boosting Adversarial Transferability by Achieving Flat Local Maxima (NeurIPS 2023)' (https://arxiv.org/abs/2306.05225)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        beta (float): the relative value for the neighborhood.
        num_neighbor (int): the number of samples for estimating the gradient variance.
        gamma (float): the balanced coefficient.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model.
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, beta=3.0, gamma=0.5, num_neighbor=20, epoch=10, decay=1.
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, beta=3.0, gamma=0.5, num_neighbor=20, epoch=10, decay=1., targeted=False, 
                random_start=False, norm='linfty', loss='crossentropy', device=None, attack='PGN', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = epsilon / epoch
        self.zeta = beta * epsilon
        self.gamma = gamma
        self.epoch = epoch
        self.decay = decay
        self.num_neighbor = num_neighbor

    def get_averaged_gradient(self, data, delta, label, **kwargs):
        """
        Calculate the averaged updated gradient    
        """
        averaged_gradient = 0
        for _ in range(self.num_neighbor):

            # Random sample an example
            x_near = self.transform(data + delta + torch.zeros_like(delta).uniform_(-self.zeta, self.zeta).to(self.device))

            # Calculate the output of the x_near
            logits = self.get_logits(x_near)

            # Calculate the loss of the x_near
            loss = self.get_loss(logits, label)

            # Calculate the gradient of the x_near
            g_1 = self.get_grad(loss, delta)

            # Compute the predicted point x_next
            x_next = self.transform(x_near + self.alpha*(-g_1 / (torch.abs(g_1).mean(dim=(1,2,3), keepdim=True))))

            # Calculate the output of the x_next
            logits = self.get_logits(x_next)

            # Calculate the loss of the x_next
            loss = self.get_loss(logits, label)

            # Calculate the gradient of the x_next
            g_2 = self.get_grad(loss, delta)

            # Calculate the gradients
            averaged_gradient += (1-self.gamma)*g_1 + self.gamma*g_2

        return averaged_gradient / self.num_neighbor

    def forward(self, data, label, **kwargs):
        """
        The attack procedure for PGN

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

        momentum, averaged_gradient = 0, 0
        for _ in range(self.epoch):

            # Calculate the averaged updated gradient
            averaged_gradient = self.get_averaged_gradient(data, delta, label)

            # Calculate the momentum
            momentum = self.get_momentum(averaged_gradient, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()
