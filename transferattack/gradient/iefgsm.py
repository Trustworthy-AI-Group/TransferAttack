import torch

from ..utils import *
from ..attack import Attack

class IEFGSM(Attack):
    """
    IE-FGSM Attack
    'Boosting Transferability of Adversarial Example via an Enhanced Euler's Method (ICASSP 2023)'(https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10096558)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        targeted (bool): targeted/untargeted attack
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10

    Example script:
        python main.py --attack iefgsm --output_dir adv_data/iefgsm/resnet18
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, targeted=False, random_start=False,
                norm='linfty', loss='crossentropy', device=None, **kwargs):
        super().__init__('IE-FGSM', model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = 1.0

    def forward(self, data, label, **kwargs):
        """
        The IE-FGSM attack procedure

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
            logits = self.get_logits(self.transform(data+delta))

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the present gradient
            g_p = grad / (grad.abs().mean(dim=(1,2,3), keepdim=True))

            # self.model.zero_grad()

            # Obtain the anticipatory output
            logits = self.get_logits(self.transform(data+delta+self.alpha*g_p))

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the anticipatory gradient
            g_a = grad / (grad.abs().mean(dim=(1,2,3), keepdim=True))

            # Get average gradient
            momentum = self.decay * momentum + (g_p + g_a) / 2

            # momentum = (g_p + g_a) / 2

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()