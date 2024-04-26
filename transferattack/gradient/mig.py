import torch
import torch.nn.functional as F
# from captum.attr import IntegratedGradients

from ..utils import *
from .mifgsm import MIFGSM


class MIG(MIFGSM):
    """
    MIG Attack
    'Transferable Adversarial Attack for Both Vision Transformers and Convolutional Networks via Momentum Integrated Gradients (ICCV 2023)'(https://openaccess.thecvf.com/content/ICCV2023/papers/Ma_Transferable_Adversarial_Attack_for_Both_Vision_Transformers_and_Convolutional_Networks_ICCV_2023_paper.pdf)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        s_factor (int): the order of the approximation of the integral.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=0.64/255, epoch=25, decay=1., s_factor=20,

    Example script:
        python main.py --attack mig --output_dir adv_data/mig/resnet18 # attack
        python main.py --attack mig --output_dir adv_data/mig/resnet18 --eval # evaluation
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., s_factor=20,
                 targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None, attack='MIG', **kwargs):
        super().__init__(model_name, epsilon, epsilon/epoch, epoch, decay,
                         targeted, random_start, norm, loss, device, attack, **kwargs)
        self.s_factor = s_factor

    def transform(self, data, **kwargs):
        x_base = torch.zeros_like(data).to(self.device)
        return torch.cat([x_base + i/self.s_factor * (data - x_base) for i in range(1, self.s_factor+1)], dim=0)
    
    def get_loss(self, logits, label):
        loss = torch.mean(logits.gather(1, label.view(-1, 1)))
        return loss if self.targeted else -loss
    
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
        x_base = torch.zeros_like(data).to(self.device)
        # ig = IntegratedGradients(self.model)
        for _ in range(self.epoch):
            # Obtain the outputs
            logits = self.get_logits(self.transform(data+delta))

            # Softmax the output
            probs = F.softmax(logits, dim=1)

            # Calculate the loss
            loss = self.get_loss(probs, label.repeat(self.s_factor))

            # Calculate the gradient
            grad = self.get_grad(loss, delta)

            # Calculate the integrated gradient
            i_grad = (data + delta - x_base) * grad / self.s_factor
            
            # Update the momentum
            momentum = self.get_momentum(i_grad, momentum)

            # Update the adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()