import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from ..utils import *
from ..attack import Attack


class AdaEA(Attack):
    """
    AdaEA Attack
    'An Adaptive Model Ensemble Adversarial Attack for Boosting Adversarial Transferability (ICCV 2023)'(https://arxiv.org/abs/2308.02897)

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
        python main.py --input_dir ./path/to/data --output_dir adv_data/adaea/ensemble --attack adaea --model='resnet18,resnet101,resnext50_32x4d,densenet121'
        python main.py --input_dir ./path/to/data --output_dir adv_data/adaea/ensemble --eval
    """

    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1.0, targeted=False,
                 random_start=True, beta=10, threshold=-0.3, norm='linfty', loss='crossentropy', device=None, attack='AdaEA', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.num_model = len(model_name)
        self.beta = beta
        self.threshold = threshold

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
        B, C, H, W = data.size()
        loss_func = nn.CrossEntropyLoss()

        momentum_G = 0.
        delta = torch.zeros_like(data).to(self.device)+0.001 * torch.randn(data.shape, device=self.device)
        delta.requires_grad =True
        for i in range(self.epoch):
            
            """Calculate the gradient of the ensemble model"""
            # Obtain the output
            outputs = [self.model.models[idx](delta+data) for idx in range(self.num_model)]
            losses = [loss_func(outputs[idx], label) for idx in range(self.num_model)]
            grads = [torch.autograd.grad(losses[idx], delta, retain_graph=True, create_graph=False)[0]
                     for idx in range(self.num_model)]

            # AGM
            alpha = self.agm(ori_data=data, cur_adv=data+delta, grad=grads, label=label)

            # DRF
            cos_res = self.drf(grads, data_size=(B, C, H, W))
            cos_res[cos_res >= self.threshold] = 1.
            cos_res[cos_res < self.threshold] = 0.

            output = torch.stack(outputs, dim=0) * alpha.view(self.num_model, 1, 1)
            output = output.sum(dim=0)
            loss = loss_func(output, label)
            grad = torch.autograd.grad(loss.sum(dim=0), delta)[0]
            grad = grad * cos_res

            momentum_G = self.get_momentum(grad, momentum_G)
            delta = self.update_delta(delta, data, momentum_G, self.alpha)

        return delta.detach()
    
    def agm(self, ori_data, cur_adv, grad, label):
        """
        Adaptive gradient modulation
        :param ori_data: natural images
        :param cur_adv: adv examples in last iteration
        :param grad: gradient in this iteration
        :param label: ground truth
        :return: coefficient of each model
        """
        loss_func = torch.nn.CrossEntropyLoss()

        # generate adversarial example
        adv_exp = [self.get_adv_example(ori_data=ori_data, adv_data=cur_adv, grad=grad[idx])
                   for idx in range(self.num_model)]
        loss_self = [loss_func(self.model.models[idx](adv_exp[idx]), label) for idx in range(self.num_model)]
        w = torch.zeros(size=(self.num_model,), device=self.device)

        for j in range(self.num_model):
            for i in range(self.num_model):
                if i == j:
                    continue
                w[j] += loss_func(self.model.models[i](adv_exp[j]), label) / loss_self[i] * self.beta
        w = torch.softmax(w, dim=0)

        return w

    def drf(self, grads, data_size):
        """
        disparity-reduced filter
        :param grads: gradients of each model
        :param data_size: size of input images
        :return: reduce map
        """
        reduce_map = torch.zeros(size=(self.num_model, self.num_model, data_size[0], data_size[-2], data_size[-1]),
                                 dtype=torch.float, device=self.device)
        sim_func = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        reduce_map_result = torch.zeros(size=(self.num_model, data_size[0], data_size[-2], data_size[-1]),
                                        dtype=torch.float, device=self.device)
        for i in range(self.num_model):
            for j in range(self.num_model):
                if i >= j:
                    continue
                reduce_map[i][j] = sim_func(F.normalize(grads[i], dim=1), F.normalize(grads[j], dim=1))
            if i < j:
                one_reduce_map = (reduce_map[i, :].sum(dim=0) + reduce_map[:, i].sum(dim=0)) / (self.num_model - 1)
                reduce_map_result[i] = one_reduce_map

        return reduce_map_result.mean(dim=0).view(data_size[0], 1, data_size[-2], data_size[-1])

    def get_adv_example(self, ori_data, adv_data, grad):
        """
        :param ori_data: original image
        :param adv_data: adversarial image in the last iteration
        :param grad: gradient in this iteration
        :return: adversarial example in this iteration
        """
        adv_example = adv_data.detach() + grad.sign() * self.alpha
        delta = torch.clamp(adv_example - ori_data.detach(), -self.epsilon, self.epsilon)
        return torch.clamp(ori_data.detach() + delta, max=1.0, min=0.0)
    
