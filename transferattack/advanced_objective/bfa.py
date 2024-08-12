import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision.transforms as T

from ..utils import *
from ..gradient.mifgsm import MIFGSM


class BFA(MIFGSM):
    """
    BFA Attack
    Improving the transferability of adversarial examples through black-box feature attacks (Neurocomputing 2024) (https://www.sciencedirect.com/science/article/abs/pii/S0925231224006349)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        eta (float): the perturbation size for mask gradient.
        num_ens (int): the fitting iteration steps.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        feature_layer: feature layer to launch the attack
        drop_rate : probability to drop random pixel

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., eta=28, num_ens=30, layer_name='layer2.7' for ResNet152

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/bfa/resnet18 --attack bfa --model resnet18
        python main.py --input_dir ./path/to/data --output_dir adv_data/bfa/resnet18 --eval

    NOTE:
        1) ResNet18 is not mentioned in the original paper. Following the setting for ResNet152 in the paper, we select the last block of the second layer for ResNet18 as the feature layer.
        2) The implementation refers to the official code of BFA attack (https://github.com/tlemangen/BFA).
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., eta=28, num_ens=30,
                 targeted=False, random_start=False, layer_name='layer2.1', norm='linfty', loss='crossentropy', device=None, attack='BFA', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.eta = eta
        self.num_ens = num_ens
        self.layer_name = layer_name
        self.num_classes = 1000
        self.feature_maps = None
        self.register_hook()

    def hook(self, module, input, output):
        self.feature_maps = output
        return None

    def register_hook(self):
        for name, module in self.model[1].named_modules():
            if name == self.layer_name:
                module.register_forward_hook(hook=self.hook)

    def get_maskgrad(self, data, labels):
        data = data.clone().detach()
        data.requires_grad = True
        logits = self.get_logits(self.transform(data))
        loss = self.get_loss(logits, labels)
        maskgrad = self.get_grad(loss, data)
        maskgrad /= torch.sqrt(torch.sum(torch.square(maskgrad), dim=(1, 2, 3), keepdim=True))
        return maskgrad.detach()

    def get_aggregate_gradient(self, data, labels):
        _ = self.get_logits(self.transform(data))
        data_masked = data.clone().detach()
        aggregate_grad = torch.zeros_like(self.feature_maps)
        targets = F.one_hot(labels.type(torch.int64), self.num_classes).float().to(self.device)
        for _ in range(self.num_ens):
            g = self.get_maskgrad(data_masked, labels)
            # get fitted image
            data_masked = data + self.eta * g
            logits = self.get_logits(self.transform(data_masked))
            loss = torch.sum(logits * targets, dim=1).mean()
            aggregate_grad += torch.autograd.grad(loss, self.feature_maps)[0]
        aggregate_grad /= -torch.sqrt(torch.sum(torch.square(aggregate_grad), dim=(1, 2, 3), keepdim=True))
        return aggregate_grad
    
    def bfa_loss_function(self, aggregate_grad, x):
        _ = self.get_logits(self.transform(x))
        fia_loss = torch.mean(torch.sum(aggregate_grad * self.feature_maps, dim=(1, 2, 3)))
        return fia_loss
    
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

        # Obtain the aggregate gradient
        aggregate_grad = self.get_aggregate_gradient(data, label)

        momentum = 0
        for _ in range(self.epoch):
            # Calculate the loss
            loss = self.bfa_loss_function(aggregate_grad, data + delta)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)
            
        return delta.detach()