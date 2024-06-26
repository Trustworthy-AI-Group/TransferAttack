from torch import nn
import torch
from ..utils import *
from ..gradient.mifgsm import MIFGSM
import torch.nn.functional as F
from torch.autograd import Variable as V


# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out



class Generator(nn.Module):
    def __init__(self,
                 gen_input_nc,
                 image_nc,
                 ):
        super(Generator, self).__init__()

        encoder_lis = [
            # MNIST:1*28*28
            nn.Conv2d(gen_input_nc, 8, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # 8*26*26
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # 16*12*12
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # 32*5*5
        ]

        bottle_neck_lis = [ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),]

        decoder_lis = [
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # state size. 16 x 11 x 11
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # state size. 8 x 23 x 23
            nn.ConvTranspose2d(8, image_nc, kernel_size=6, stride=1, padding=0, bias=False),
            nn.Tanh()
            # state size. image_nc x 28 x 28
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x

    
class GE_ADVGAN(MIFGSM):
    """
    GE-ADVGAN
    'GE-AdvGAN: Improving the transferability of adversarial samples by gradient editing-based adversarial generative model (SDM 2024)'(https://epubs.siam.org/doi/abs/10.1137/1.9781611978032.81)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        gamma (float): the scalar weight to trade-off the contributions of each loss  function.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        c (int): The number of channels in the input images.

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., c=3

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/ge_advgan/resnet18 --attack ge_advgan --model=resnet18
        python main.py --input_dir ./path/to/data --output_dir adv_data/ge_advgan/resnet18 --eval
    """
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, random_start=False, 
                    norm='linfty', loss='crossentropy', device=None, attack='GE_ADVGAN', checkpoint_path='./path/to/checkpoints/', c=3, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.checkpoint_path = checkpoint_path
        self.c = c
        self.model_name = model_name
        self.netG = self.load_ge_advgan_model()
        print("=> loaded trained GEadvGAN model")
        
    def load_ge_advgan_model(self, **kwargs):
        netG = Generator(self.c,self.c)
        
        weight_name = os.path.join(self.checkpoint_path, f'{self.model_name}.pth')
        
        if not os.path.exists(weight_name):
            raise ValueError("Please download the checkpoint of the 'GE_ADVGAN' from https://drive.google.com/drive/folders/1eF-QF_NjYVQw_bCnBJGxsbsfF1E_Ay7K?usp=drive_link, and put it into the path '{}'.".format(self.checkpoint_path))
        
        netG.load_state_dict(torch.load(weight_name))
        
        return netG.eval().to(self.device)

    def crop(self, perturbation, img_width, img_height):
        if img_width % 2 == 0 and img_height % 2 == 0:
            return perturbation
        elif img_width % 2 == 1 and img_height % 2 == 1:
            return perturbation[:,:,:-1,:-1]
        elif img_width % 2 == 1 and img_height % 2 == 0:
            return perturbation[:,:,:-1,:]
        elif img_width % 2 == 0 and img_height % 2 == 1:
            return perturbation[:,:,:,:-1]

    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        data = data.clone().detach().to(self.device)
        delta = self.netG(data)
        delta = self.crop(delta, data.shape[2], data.shape[3])
        adv_images = torch.clamp(delta, -self.epsilon, self.epsilon) + data
        adv_images = torch.clamp(adv_images, img_min, img_max)
        delta = adv_images - data
        return delta.detach()

