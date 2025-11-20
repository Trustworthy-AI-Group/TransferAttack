import torch
import torch.nn.functional as F

from ..utils import *
from ..gradient.mifgsm import MIFGSM
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torchvision
import pandas as pd
###########################
# Generator: Resnet
###########################
# To control feature map in generator
ngf = 64

class GeneratorResnet(nn.Module):
    def __init__(self, inception=False):
        '''
        :param inception: if True crop layer will be added to go from 3x300x300 t0 3x299x299.
        :param data_dim: for high dimentional dataset (imagenet) 6 resblocks will be add otherwise only 2.
        '''
        super(GeneratorResnet, self).__init__()
        self.inception = inception
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        # Input size = 3, n/4, n/4
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)
        
        self.resblock3 = ResidualBlock(ngf * 4)
        self.resblock4 = ResidualBlock(ngf * 4)
        self.resblock5 = ResidualBlock(ngf * 4)
        self.resblock6 = ResidualBlock(ngf * 4)
        
        # Input size = 3, n/4, n/4
        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
        )

        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)

    def forward(self, input):
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.upsampl1(x)
        x = self.upsampl2(x)
        x = self.blockf(x)
        if self.inception:
            x = self.crop(x)
        return (torch.tanh(x) + 1) / 2 # Output range [0 1]


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),

            nn.Dropout(0.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual
    

class DSVA(MIFGSM):
    """
    Dual Self-supervised ViT features Attack(DSVA) ICCV 2025
    Boosting Generative Adversarial Transferability with Self-supervised  Vision Transformer Features (DSVA) (https://arxiv.org/abs/2506.21046)
    
    Official Link: https://github.com/spencerwooo/dSVA
    TransferAttack framework provides an alternative download link: https://huggingface.co/NexusBohanLiu/dSVA/blob/main/model.pth 
    
    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/dsva/generation --attack dsva  
        python main.py --input_dir ./path/to/data --output_dir adv_data/dsva/generation --eval 
    
    Arguments:
        model_name (str): the name of the model.
        epsilon (float): the perturbation budget.
    """
    
    def __init__(self, model_name, *args, **kwargs):
        super().__init__(model_name)
        self.netG = self.load_Gmodel()
        
        
    def load_Gmodel(self):
        netG = GeneratorResnet()
        model_path = './path/to/checkpoints/model.pth'        
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            model_state = netG.state_dict()
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k in model_state and model_state[k].shape == v.shape:
                    filtered_state_dict[k] = v
            netG.load_state_dict(filtered_state_dict)  
        except Exception as e:
            print("No pre-trained generator model found, \
                please visit https://huggingface.co/NexusBohanLiu/dSVA/blob/main/model.pth \
                to download model, and put it under './path/to/checkpoints/'. ")
            return None        
        netG.to(self.device)
        netG.eval()
        return netG
    
    def forward(self, data: Tensor, label: Tensor, **kwargs):
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        with torch.no_grad():
            adv_imgs = self.netG(data).detach()
        perturbations = adv_imgs - data
        perturbations = torch.clamp(perturbations, -self.epsilon, self.epsilon)
        return perturbations
    
    
