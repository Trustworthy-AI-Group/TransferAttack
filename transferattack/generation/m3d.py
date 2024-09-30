import math

import torch
import torch.nn as nn
from torch import Tensor

from ..gradient.mifgsm import MIFGSM
from ..utils import *


class M3D(MIFGSM):
    """
    M3D Attack
    'Minimizing Maximum Model Discrepancy for Transferable Black-box Targeted Attacks (CVPR 2023)'(https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_Minimizing_Maximum_Model_Discrepancy_for_Transferable_Black-Box_Targeted_Attacks_CVPR_2023_paper.pdf)

    Arguments:
        model_name (str): the surrogate model name for attack.

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/m3d/resnet50 --attack m3d --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/m3d/resnet50 --attack m3d --model=resnet50 --eval
    """

    def __init__(self, model_name="resnet18", *args, **kwargs):
        super().__init__(model_name, *args, **kwargs)

        self.model_name = model_name
        self.netG_list = []
        for target_class in generation_target_classes:
            self.netG_list.append(self.load_Gmodel(target_class))

    def load_Gmodel(self, target_class):
        netG = GeneratorResnet()
        file_path = "/path/to/checkpoint/m3d/netG_{}_9_{}.pth".format(self.model_name, target_class)
        try:
            netG.load_state_dict(torch.load(file_path))
        except:
            raise FileExistsError(
                f"No pre-trained generator model found at {file_path}, please visit "
                "https://github.com/Asteriajojo/M3D or "
                "https://huggingface.co/Trustworthy-AI-Group/TransferAttack/blob/main/M3D.zip "
                "to download the model."
            )

        netG.to(self.device)
        netG.eval()
        return netG

    def forward(self, data: Tensor, label: Tensor, idx, **kwargs):
        netG = self.netG_list[idx]
        data = data.clone().detach().to(self.device)
        kernel_size = 3
        pad = 2
        sigma = 1
        kernel = get_gaussian_kernel(kernel_size=kernel_size, pad=pad, sigma=sigma).cuda()
        with torch.no_grad():
            adv_imgs = netG(data).detach()
            adv_imgs = kernel(adv_imgs)
        perturbations = adv_imgs - data
        perturbations = torch.clamp(perturbations, -self.epsilon, self.epsilon)
        return perturbations


###########################
# Generator: Resnet
###########################
ngf = 64


class GeneratorResnet(nn.Module):
    def __init__(self, inception=False):
        """
        :param inception: if True crop layer will be added to go from 3x300x300 t0 3x299x299.
        :param data_dim: for high dimentional dataset (imagenet) 6 resblocks will be add otherwise only 2.
        """
        super(GeneratorResnet, self).__init__()
        self.inception = inception
        # Input_size = 3, n, n
        self.block1 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True))

        # Input size = 3, n, n
        self.block2 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True))

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True))

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
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.blockf = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0))

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

        return (torch.tanh(x) + 1) / 2  # Output range [0 1]


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_filters),
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual


# From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
def get_gaussian_kernel(kernel_size=3, pad=2, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.0
    variance = sigma**2.0

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(-torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels, padding=kernel_size - pad, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter
