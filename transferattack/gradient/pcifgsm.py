import torch

from ..utils import *
from ..attack import Attack

class PCIFGSM(Attack):
    """
    PCIFGSM Attack
    'Adversarial Attack Based on Prediction-Correction'(https://arxiv.org/abs/2306.01809)

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

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1.
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None, attack='PC-FGSM', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.K = 1

    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

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
            delta_pre = self.init_delta(data)
            g_pre = torch.zeros_like(delta)

            logits = self.get_logits(self.transform(data+delta, momentum=momentum))
            # Calculate the loss
            loss = self.get_loss(logits, label)
            # Calculate the gradients
            grad = self.get_grad(loss, delta)
            g_pre = self.decay * g_pre + grad / (torch.norm(grad, p=1))

            for k in range(self.K):
                # Obtain the output
                logits = self.get_logits(self.transform(data+delta+delta_pre, momentum=momentum))

                # Calculate the loss
                loss = self.get_loss(logits, label)

                # Calculate the gradients
                grad = self.get_grad(loss, delta_pre)
                g_pre = self.decay * g_pre + grad / (self.K*torch.norm(grad, p=1))
                # Calculate the momentum
                # momentum = self.get_momentum(grad, momentum)

                # Update adversarial perturbation
                delta_pre = self.update_delta(delta_pre, data, grad, self.epsilon)
            momentum = self.get_momentum(g_pre, momentum)
            delta = self.update_delta(delta, data, momentum, self.alpha)
        return delta.detach()