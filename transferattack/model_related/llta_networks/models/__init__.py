import torch
import torch.nn as nn
import pretrainedmodels

from . import decayresnet
from . import decaydensenet
from .utils import Gaussian, get_Gaussian_kernel

class Wrap(nn.Module):
    def __init__(self, model):
        super(Wrap, self).__init__()
        self.model = model
        self.mean = self.model.mean
        self.std = self.model.std
        self.input_size = self.model.input_size

        self._mean = torch.tensor(self.mean).view(3,1,1)
        self._std = torch.tensor(self.std).view(3,1,1)

    def forward(self, x):
        device = x.device
        return self.model.forward((x - self._mean.to(device)) / self._std.to(device))

class Gamma_Wrap_for_ResNet(nn.Module):
    def __init__(self, arch, model):
        super(Gamma_Wrap_for_ResNet, self).__init__()
        self.arch = arch
        self.model = model
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.input_size = [3, 224, 224]

        self._mean = torch.tensor(self.mean).view(3,1,1)
        self._std = torch.tensor(self.std).view(3,1,1)
    
    def forward(self, x, gammas):
        device = x.device
        return self.model.forward((x - self._mean.to(device)) / self._std.to(device), gammas)

class Gamma_Wrap_for_DenseNet(nn.Module):
    def __init__(self, arch, model):
        super(Gamma_Wrap_for_DenseNet, self).__init__()
        self.arch = arch
        self.model = model
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.input_size = [3, 224, 224]

        self._mean = torch.tensor(self.mean).view(3,1,1)
        self._std = torch.tensor(self.std).view(3,1,1)

    def forward(self, x, gammas):
        device = x.device
        return self.model.forward((x - self._mean.to(device)) / self._std.to(device), gammas) 


def make_Gamma_Wrap(arch):
    if "resnet" in arch:
        model = getattr(decayresnet, arch)(pretrained=True)
        return Gamma_Wrap_for_ResNet(arch, model)

    elif "densenet" in arch:
        model = getattr(decaydensenet, arch)(pretrained=True)
        return Gamma_Wrap_for_DenseNet(arch, model)

    else:
        raise NotImplementedError(f"Gamma wrap only support ResNet-Family and DenseNet-Family")   


def make_model(arch):
    # for evaluation
    model = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained='imagenet')
    return Wrap(model)
