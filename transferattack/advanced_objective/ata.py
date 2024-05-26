import torch

from ..utils import *
from ..attack import Attack
import torch.nn as nn

class ATA(Attack):
    """
    ATA Attack
    'Boosting the Transferability of Adversarial Samples via Attention (CVPR 2020) (https://openaccess.thecvf.com/content_CVPR_2020/papers/Wu_Boosting_the_Transferability_of_Adversarial_Samples_via_Attention_CVPR_2020_paper.pdf)'

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        layer_name (str): the feature layer name
        alpha (float): the step size.
        lamda (float): the regularization constant for calculating loss
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, lamda=1, epoch=10, decay=1.
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, random=False, epoch=10,
                  targeted=False, lamda=1, layer_name = 'layer4',
                 random_start=False, norm='linfty', loss='crossentropy', device=None, attack='ATA', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.lamda = lamda
        self.random = random
        self.relu = nn.ReLU()
        self.feature_layer = self.find_layer(layer_name)
        self.mid_output = 0
        self.mid_grad = 0

    def __forward_hook(self,m,i,o):
        self.mid_output = o

    def __backward_hook(self,m,i,o):
        self.mid_grad = o

    def find_layer(self,layer_name):
        parser = layer_name.split(' ')
        m = self.model[1]
        for layer in parser:
            if layer not in m._modules.keys():
                print("Selected layer is not in Model")
                exit() 
            else:
                m = m._modules.get(layer)
        return m
    
    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        h = self.feature_layer.register_forward_hook(self.__forward_hook)
        h2 = self.feature_layer.register_full_backward_hook(self.__backward_hook)

        ori_output = self.model(data)
        ori_loss = 0
        ori_output = torch.softmax(ori_output, 1)
        for batch_i in range(data.shape[0]):
            ori_loss += ori_output[batch_i][label[batch_i]]
        self.model.zero_grad()
        ori_loss.backward()
        mid_grad_ori = torch.zeros(self.mid_grad[0].size()).cuda()
        mid_fmap_ori = torch.zeros(self.mid_output.size()).cuda()

        mid_grad_ori.copy_(self.mid_grad[0])
        mid_fmap_ori.copy_(self.mid_output)

        ori_grad_weights = torch.mean(mid_grad_ori,axis=(2,3),keepdims=True)
        ori_weighted_activation = ori_grad_weights * mid_fmap_ori
        ori_weighted_activation = torch.sum(ori_weighted_activation, dim=1)
        ori_weighted_activation = self.relu(ori_weighted_activation)

        for _ in range(self.epoch):
            # Obtain the output
            logits = self.model(data+delta)
            logits = torch.softmax(logits, 1)
            adv_loss = 0
            for batch_i in range(data.shape[0]):
                adv_loss = logits[batch_i][label[batch_i]]
            self.model.zero_grad()
            adv_loss.backward(retain_graph=True)
            logits_l1 = self.model(data+delta)
            loss1 = self.loss(logits_l1, label) if self.targeted else self.loss(logits_l1, label)
            adv_grad_weights = torch.mean(self.mid_grad[0], axis=(2, 3), keepdims=True)
            adv_weighted_activation = adv_grad_weights * self.mid_output
            adv_weighted_activation = torch.sum(adv_weighted_activation, dim=1)
            adv_weighted_activation = self.relu(adv_weighted_activation)

            loss2 = self.lamda * torch.norm(adv_weighted_activation-ori_weighted_activation)**2

            # Calculate the loss
            loss = loss2 + loss1

            self.model.zero_grad()

            # Calculate the gradients
            grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, grad, self.alpha)

        h.remove()
        h2.remove()
        return delta.detach()