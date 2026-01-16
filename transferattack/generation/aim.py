import abc
from collections import OrderedDict
from pathlib import Path
from typing import Union
from torchvision import transforms
import torch
from torch import nn
import os
from PIL import Image
from ..utils import generation_target_classes
class BaseGenerativeAttack(abc.ABC):

    def __init__(self,
                 device: Union[str, torch.device],
                 epsilon: float = 16 / 255) -> None:
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.set_adv_gen()
        self.set_mode('eval')
        self.epsilon = epsilon

    @abc.abstractmethod
    def set_adv_gen(self):
        pass

    def load_ckpt(self, ckpt: Union[str, Path, OrderedDict]) -> None:
        if isinstance(ckpt, str):
            ckpt = Path(ckpt)
        if isinstance(ckpt, Path):
            if not ckpt.exists():
                raise FileNotFoundError(f'File not found: {ckpt}')
            ckpt = torch.load(ckpt, map_location=self.device)
        self.adv_gen.load_state_dict(ckpt)
        self.adv_gen.to(self.device)

    def save_ckpt(self, ckpt: Union[str, Path]) -> None:
        if isinstance(ckpt, str):
            ckpt = Path(ckpt)
        _adv_gen_cpu = self.adv_gen.to('cpu')
        torch.save(_adv_gen_cpu.state_dict(), ckpt)

    def get_params(self) -> torch.nn.Parameter:
        return self.adv_gen.parameters()

    def get_model(self) -> torch.nn.Module:
        return self.adv_gen

    def set_mode(self, mode: str) -> None:
        assert mode in ['train', 'eval']
        self.adv_gen.train() if mode == 'train' else self.adv_gen.eval()

    @abc.abstractmethod
    def forward(self, *args) -> torch.Tensor:
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class EnhancedBN(nn.Module):
    def __init__(self, nc: int, sty_nc: int = 3, sty_nhidden: int = 128):
        super(EnhancedBN, self).__init__()
        self.bn = nn.BatchNorm2d(nc)
        self.mapping = nn.Conv2d(
            in_channels=sty_nc,
            out_channels=sty_nhidden,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.gamma = nn.Conv2d(
            in_channels=sty_nhidden,
            out_channels=nc,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.beta = nn.Conv2d(
            in_channels=sty_nhidden,
            out_channels=nc,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.init_weight()

    def init_weight(self):
        nn.init.kaiming_normal_(self.mapping.weight)
        nn.init.kaiming_normal_(self.gamma.weight)
        nn.init.kaiming_normal_(self.beta.weight)

    def forward(self, base, sty):
        bn = self.bn(base)
        sty_resized = torch.nn.functional.interpolate(
            sty, size=bn.size()[2:], mode='bilinear'
        )
        actv = torch.nn.functional.relu(self.mapping(sty_resized))
        # style injection
        bn = bn * (1 + self.gamma(actv)) + self.beta(actv)
        return bn


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
        )
        self.bn1 = EnhancedBN(num_filters)
        self.block2 = nn.Sequential(
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
        )
        self.bn2 = EnhancedBN(num_filters)

    def forward(self, x, sty):
        residual = self.block1(x)
        residual = self.bn1(residual, sty)
        residual = self.block2(residual)
        residual = self.bn2(residual, sty)
        return x + residual



ngf = 64


class ResNetGenerator(nn.Module):
    def __init__(self):
        super(ResNetGenerator, self).__init__()
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False),
        )
        self.bn1 = EnhancedBN(ngf)
        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(
                ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False
            ),
        )
        self.bn2 = EnhancedBN(ngf * 2)
        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(
                ngf * 2,
                ngf * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
        )
        self.bn3 = EnhancedBN(ngf * 4)
        # Input size = 3, n/4, n/4
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)
        self.resblock3 = ResidualBlock(ngf * 4)
        self.resblock4 = ResidualBlock(ngf * 4)
        self.resblock5 = ResidualBlock(ngf * 4)
        self.resblock6 = ResidualBlock(ngf * 4)
        # Input size = 3, n/4, n/4
        self.upsampl1 = nn.ConvTranspose2d(
            ngf * 4,
            ngf * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.ubn1 = EnhancedBN(ngf * 2)
        # Input size = 3, n/2, n/2
        self.upsampl2 = nn.ConvTranspose2d(
            ngf * 2,
            ngf,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.ubn2 = EnhancedBN(ngf)
        # Input size = 3, n, n
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
        )

    def forward(self, input, sty):
        x = self.block1(input)
        x = self.bn1(x, sty)
        x = torch.nn.functional.relu(x)
        x = self.block2(x)
        x = self.bn2(x, sty)
        x = torch.nn.functional.relu(x)
        x = self.block3(x)
        x = self.bn3(x, sty)
        x = torch.nn.functional.relu(x)
        # =============================
        x = self.resblock1(x, sty)
        x = self.resblock2(x, sty)
        x = self.resblock3(x, sty)
        x = self.resblock4(x, sty)
        x = self.resblock5(x, sty)
        x = self.resblock6(x, sty)
        # =============================
        x = self.upsampl1(x)
        x = self.ubn1(x, sty)
        x = torch.nn.functional.relu(x)
        x = self.upsampl2(x)
        x = self.ubn2(x, sty)
        x = torch.nn.functional.relu(x)
        x = self.blockf(x)
        return (torch.tanh(x) + 1) / 2

class AIM(BaseGenerativeAttack):
    """
    AIM
    'AIM: Additional Image Guided Generation of Transferable Adversarial Attacks (AAAI 2025)'(https://arxiv.org/abs/2501.01106)

    Arguments:
        epsilon (float): the perturbation budget.
        related_path(str): the path of pretrained ResNetGenerator and guided images.
        img_size(int): the size of guided images.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, related_path='./transferattack/generation/aim_related', device='cuda',img_size=224

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/aim/ --attack aim --targeted 
        python main.py --input_dir ./path/to/data --output_dir adv_data/aim/ --eval --targeted
    """
    def __init__(self, epsilon=16/255, related_path='./transferattack/generation/aim_related', device='cuda',img_size=224, targeted=True, model_name=None):
        super().__init__(device=device, epsilon=epsilon)
        self.targeted = targeted
        self.path = related_path
        self.img_size = img_size

    def set_adv_gen(self):
        self.adv_gen = ResNetGenerator().to(self.device)
    
    def prepare(self,idx):
        idx = generation_target_classes[idx]
        self.load_ckpt(os.path.join(self.path, f"model_{idx}.pth"))
        img = Image.open(os.path.join(self.path, f"{idx}.JPEG")).convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        totensor = transforms.ToTensor()
        self.x_guid = totensor(img).unsqueeze(0)
        
        
    def forward(self, data, labels, idx):
        self.prepare(idx)
        assert len(labels) == 2
        label = labels[0].clone().detach().to(self.device).long()
        adv_label = labels[1].clone().detach().to(self.device).long()
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        adv_label = adv_label.clone().detach().to(self.device)
        x_guid = self.x_guid.clone().detach().to(self.device)

        x_adv =  self.adv_gen(data, x_guid)
        x_adv = torch.min(torch.max(x_adv, data - self.epsilon),
                          data + self.epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return (x_adv-data).detach()
    
