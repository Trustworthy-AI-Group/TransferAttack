import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

from ..utils import *
from ..gradient.mifgsm import MIFGSM


class BPA(MIFGSM):
    """
    BPA: Backward Propagation Attack
    'Rethinking the Backward Propagation for Adversarial Transferability (NeurIPS 2023)'(https://arxiv.org/abs/2306.12685)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        bpa_layer (str): start bpa from this layer.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=8/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., bpa_layer=3_1 for resnet50

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/bpa/resnet18 --attack bpa --model=resnet18
        python main.py --input_dir ./path/to/data --output_dir adv_data/bpa/resnet18 --eval

    Note:
        Currently, we only support Resnet model.
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., bpa_layer='3_1',
                 targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None, attack='BPA', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.bpa_layer = bpa_layer

    def load_model(self, model_name):
        if 'resnet' not in model_name:
            raise ValueError(
                'Model {} not supported. Currently we only support Resnet.'.format(model_name))
        
        if model_name in models.__dict__.keys():
            print('=> Loading model {} from torchvision.models'.format(model_name))
            model = models.__dict__[model_name](pretrained=True)
        else:
            raise ValueError('Model {} not supported.'.format(model_name))

        model.maxpool = MaxPool2dK3S2P1()

        for i in range(1, len(model.layer3)):
            model.layer3[i].relu = ReLU_SiLU()

        for i in range(len(model.layer4)):
            model.layer4[i].relu = ReLU_SiLU()

        return wrap_model(model.eval().cuda())


# Refer to the code https://github.com/Trustworthy-AI-Group/BPA
class MaxPool2dK3S2P1Function(Function):
    temperature = 10.

    @staticmethod
    def forward(ctx, input_):
        with torch.no_grad():
            output = F.max_pool2d(input_, 3, 2, 1)
        ctx.save_for_backward(input_, output)
        return output.to(input_.device)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            input_, output = ctx.saved_tensors
            input_unfold = F.unfold(input_, 3, padding=1, stride=2).reshape(
                (input_.shape[0], input_.shape[1], 3*3, grad_output.shape[2]*grad_output.shape[3]))

            output_unfold = torch.exp(
                MaxPool2dK3S2P1Function.temperature*input_unfold).sum(dim=2, keepdim=True)

            grad_output_unfold = grad_output.reshape(
                output.shape[0], output.shape[1], 1, -1).repeat(1, 1, 9, 1)
            grad_input_unfold = grad_output_unfold * \
                torch.exp(MaxPool2dK3S2P1Function.temperature *
                          input_unfold) / output_unfold
            grad_input_unfold = grad_input_unfold.reshape(
                input_.shape[0], -1, output.shape[2]*output.shape[3])
            grad_input = F.fold(grad_input_unfold,
                                input_.shape[2:], 3, padding=1, stride=2)
            return grad_input.to(input_.device)


# Refer to the code https://github.com/Trustworthy-AI-Group/BPA
class MaxPool2dK3S2P1(nn.Module):
    def __init__(self):
        super(MaxPool2dK3S2P1, self).__init__()

    def forward(self, input):
        return MaxPool2dK3S2P1Function.apply(input)


# Refer to the code https://github.com/Trustworthy-AI-Group/BPA
class ReLU_SiLU_Function(Function):
    temperature = 1.

    @staticmethod
    def forward(ctx, input_):
        with torch.no_grad():
            output = torch.relu(input_)
        ctx.save_for_backward(input_)
        return output.to(input_.device)

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        with torch.no_grad():
            grad_input = input_ * \
                torch.sigmoid(input_) * (1 - torch.sigmoid(input_)) + \
                torch.sigmoid(input_)
            grad_input = grad_input * grad_output * ReLU_SiLU_Function.temperature
        return grad_input.to(input_.device)


# Refer to the code https://github.com/Trustworthy-AI-Group/BPA
class ReLU_SiLU(nn.Module):
    def __init__(self):
        super(ReLU_SiLU, self).__init__()

    def forward(self, input):
        return ReLU_SiLU_Function.apply(input)
