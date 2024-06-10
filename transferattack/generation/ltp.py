import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import *
from ..attack import Attack

class LTP(Attack):
    """
    LTP Attack
    'Learning transferable adversarial perturbations (NeurIPS 2021)'(https://proceedings.neurips.cc/paper/2021/hash/7486cef2522ee03547cfb970a404a874-Abstract.html)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255.

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/ltp/generation --attack ltp --model=generation
        python main.py --input_dir ./path/to/data --output_dir adv_data/ltp/generation --eval
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, random_start=False,
                norm='linfty', loss='crossentropy', device=None, attack='LTP', checkpoint_path='./path/to/checkpoints/', **kwargs):
        self.checkpoint_path = checkpoint_path
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha

    def load_model(self, model_name):
        # download model: https://github.com/krishnakanthnakka/Transferable_Perturbations
        if model_name == 'generation':
            model_path = os.path.join(self.checkpoint_path, '1_net_G.pth')
        else:
            raise ValueError('model:{} not supported'.format(model_name))

        if not os.path.exists(model_path):
            raise ValueError("Please download checkpoints '1_net_G.pth' (Discriminator is ResNet152) from 'https://drive.google.com/drive/folders/1QkJh9EPGyq_LnzzU5mzpkBNhJFxIxGMu', and put them into the path './path/to/checkpoints'.")
        
        model = GeneratorResnet(gen_dropout=0.0, data_dim='high')
 
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        return model.eval().cuda()
    
    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        data = data.clone().detach().to(self.device)

        # Generating Adversarial Examples
        adv_data = self.model(data)

        # Computing delta
        delta = adv_data-data
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        delta = clamp(delta, img_min-data, img_max-data)
        return delta.detach()



###########################
# Generator: Resnet
###########################

# To control feature map in generator

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class GeneratorResnet(nn.Module):
    def __init__(self, gen_dropout, data_dim, inception=False, isTrain=False, ngf = 64):
        '''
        :param inception: if True crop layer will be added to go from 3x300x300 t0 3x299x299.
        :param data_dim: for high dimentional dataset (imagenet) 6 resblocks will be add otherwise only 2.
        '''
        super(GeneratorResnet, self).__init__()

        # logger = logging.getLogger("CDA.inference")
        # if isTrain:
        #     logger.info("Gen Dropout: {}, Depth: {}".format(gen_dropout, data_dim))

        self.inception = inception
        self.data_dim = data_dim
        # Input_size = 3, n, n
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            # nn.InstanceNorm2d(ngf),

            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),


            nn.BatchNorm2d(ngf * 2),
            # nn.InstanceNorm2d(ngf * 2),

            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            # nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        # Input size = 3, n/4, n/4
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(ngf * 4, gen_dropout)
        self.resblock2 = ResidualBlock(ngf * 4, gen_dropout)
        if self.data_dim == 'high':
            self.resblock3 = ResidualBlock(ngf * 4, gen_dropout)
            self.resblock4 = ResidualBlock(ngf * 4, gen_dropout)
            self.resblock5 = ResidualBlock(ngf * 4, gen_dropout)
            self.resblock6 = ResidualBlock(ngf * 4, gen_dropout)
            # self.resblock7 = ResidualBlock(ngf*4)
            # self.resblock8 = ResidualBlock(ngf*4)
            # self.resblock9 = ResidualBlock(ngf*4)

        # Input size = 3, n/4, n/4
        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),

            nn.BatchNorm2d(ngf * 2),
            # nn.InstanceNorm2d(ngf * 2),

            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            # nn.InstanceNorm2d(ngf),
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
        if self.data_dim == 'high':
            x = self.resblock3(x)
            x = self.resblock4(x)
            x = self.resblock5(x)
            x = self.resblock6(x)
            # x = self.resblock7(x)
            # x = self.resblock8(x)
            # x = self.resblock9(x)
        x = self.upsampl1(x)
        x = self.upsampl2(x)
        x = self.blockf(x)

        #print(x.shape)

        if self.inception:
            x = self.crop(x)

        # CHANGED
        return (torch.tanh(x) + 1) / 2.0  # Output range [0 1]

        # return torch.sigmoid(x)  # Output range [0 1]


class ResidualBlock(nn.Module):
    def __init__(self, num_filters, gen_dropout):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),

            nn.BatchNorm2d(num_filters),
            # nn.InstanceNorm2d(num_filters),


            nn.ReLU(True),
            # CHANGED

            # if gen_dropout > 0.01:
            nn.Dropout(gen_dropout),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),

            nn.BatchNorm2d(num_filters)
            # nn.InstanceNorm2d(num_filters),

        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual
