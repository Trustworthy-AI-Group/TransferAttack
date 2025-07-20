import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from ..utils import *
from ..attack import Attack


class SMER(Attack):
    """
    SMER Attack
    'Ensemble Diversity Facilitates Adversarial Transferability (CVPR 2024)'
    (https://openaccess.thecvf.com/content/CVPR2024/papers/Tang_Ensemble_Diversity_Facilitates_Adversarial_Transferability_CVPR_2024_paper.pdf)

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
        python main.py --input_dir ./path/to/data --output_dir adv_data/smer/ensemble --attack smer --model='resnet50,vgg16,mobilenet_v2,inception_v3'
        python main.py --input_dir ./path/to/data --output_dir adv_data/smer/ensemble --eval
    """

    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1.0, targeted=False,
                 random_start=True, norm='linfty', loss='crossentropy', device=None, attack='SMER', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.num_model = len(model_name)
        self.m_smer = self.num_model*4
        self.weight_selection = Weight_Selection(self.num_model).cuda()
        self.optimizer = torch.optim.SGD(self.weight_selection.parameters(), lr=2e-2, weight_decay=2e-3)

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
        data = data.clone().detach().cuda()
        label = label.clone().detach().cuda()

        # Initialize adversarial perturbation
        delta = self.init_delta(data)
        momentum = 0.

        for i in range(self.epoch):
            if delta.grad is not None:
                delta.grad.zero_()
            delta.requires_grad_(True)
            x_inner_delta = delta.detach()

            noise_inner_all = torch.zeros([self.m_smer, *delta.shape]).cuda()
            grad_inner = torch.zeros_like(delta)
            options = []

            for i in range(int(self.m_smer/self.num_model)):
                options_single=[j for j in range(self.num_model)]
                np.random.shuffle(options_single)
                options.append(options_single)
            options = np.reshape(options, -1)

            for j in range(self.m_smer):
                option = options[j]
                grad_single = self.model.models[option]
                x_inner_delta.requires_grad = True
                out_logits = grad_single(data+x_inner_delta)
                if type(out_logits) is list:
                    out = self.weight_selection(out_logits[0], option)
                    aux_out = self.weight_selection(out_logits[1], option)
                else:
                    out = self.weight_selection(out_logits, option)             
                loss = F.cross_entropy(out, label)
                if type(out_logits) is list:
                    loss = loss + F.cross_entropy(aux_out, label)
                noise_im_inner = torch.autograd.grad(loss, x_inner_delta)[0]
                group_logits = 0
                group_aux_logits = 0
                for m_step, model_s in enumerate(self.model.models):
                    out_logits = model_s(data + x_inner_delta)
                    if type(out_logits) is list:
                        logits = self.weight_selection(out_logits[0], m_step)
                        aux_logits = self.weight_selection(out_logits[1], m_step)
                    else:
                        logits = self.weight_selection(out_logits, m_step)
                    group_logits = group_logits + logits / self.num_model

                    if type(out_logits) is list:
                        group_aux_logits = group_aux_logits + aux_logits / self.num_model
                loss = F.cross_entropy(group_logits, label)

                if type(out_logits) is list:
                    loss = loss + F.cross_entropy(group_aux_logits, label)
                outer_loss = -torch.log(loss)
                x_inner_delta.requires_grad = False
                outer_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                noise_inner = noise_im_inner

                grad_inner = self.get_momentum(noise_inner, grad_inner)
                
                x_inner_delta = self.update_delta(x_inner_delta, data, grad_inner, self.alpha)

                noise_inner_all[j] = grad_inner.clone()

            noise =noise_inner_all[-1].clone()

            momentum = self.get_momentum(noise, momentum)
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()
    
class Weight_Selection(nn.Module):
    def __init__(self, weight_len) -> None:
        super(Weight_Selection, self).__init__()
        self.weight = nn.parameter.Parameter(torch.ones([weight_len]))

    def forward(self, x, index):
        x = self.weight[index] * x
        return x