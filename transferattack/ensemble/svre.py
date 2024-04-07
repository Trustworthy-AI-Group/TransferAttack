import torch
import torch.nn as nn

import numpy as np
from ..utils import *
from ..attack import Attack

class SVRE(Attack):
    """
    SVRE Attack
    'Stochastic variance reduced ensemble adversarial attack for boosting the adversarial transferability (CVPR 2022)'(https://openaccess.thecvf.com/content/CVPR2022/papers/Xiong_Stochastic_Variance_Reduced_Ensemble_Adversarial_Attack_for_Boosting_the_Adversarial_CVPR_2022_paper.pdf)

    Arguments:
        model (torch.nn.Module): the surrogate model for attack.
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
        python main.py --input_dir ./path/to/data --output_dir adv_data/svre/ensemble --attack svre --model='resnet18,resnet101,resnext50_32x4d,densenet121'
        python main.py --input_dir ./path/to/data --output_dir adv_data/svre/ensemble --eval
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1.0, targeted=False, random_start=True, 
                 norm='linfty', loss='crossentropy', device=None, attack='SVRE', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.K = len(model_name)
        self.M = 4*self.K
        self.beta = alpha

    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data (N, C, H, W): tensor for input images
            labels (N,): tensor for ground-truth labels if untargetd
            labels (2,N): tensor for [ground-truth, targeted labels] if targeted
        """
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        momentum_G = 0.
        delta = self.init_delta(data).to(self.device)

        for _ in range(self.epoch):
            """Calculate the gradient of the ensemble model"""
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta))
            # Calculate the loss
            loss = self.get_loss(logits, label)
            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            """Stochastic variance reduction via M updates"""
            # Update adversarial perturbation
            inner_G = 0.
            inner_delta = delta.clone().detach()
            inner_delta.requires_grad = True

            for _ in range(self.M):
                k = np.random.randint(self.K)
                inner_k_logits = self.get_logits_by_model_k(self.transform(data+inner_delta), k)
                inner_k_grad = self.get_grad(self.get_loss(inner_k_logits, label), inner_delta)

                adv_k_logits = self.get_logits_by_model_k(self.transform(data+delta), k)
                adv_k_grad = self.get_grad(self.get_loss(adv_k_logits, label), delta)
                gm = inner_k_grad - (adv_k_grad - grad)

                """Update the inner gradient by momentum"""
                inner_G = self.get_momentum(gm, inner_G)
                inner_delta = self.update_delta(inner_delta, data, inner_G, self.beta)
                
            """Update the outer gradient by momentum"""
            momentum_G = self.get_momentum(inner_G, momentum_G)
            delta = self.update_delta(delta, data, momentum_G, self.alpha)

        return delta.detach()
    
    def get_logits_by_model_k(self, x, k):
        """
        The inference stage, which should be overridden when the attack need to change the models (e.g., ensemble-model attack, ghost, etc.) or the input (e.g. DIM, SIM, etc.)
        """
        return self.model.models[k](x)
