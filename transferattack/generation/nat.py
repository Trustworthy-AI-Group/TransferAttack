import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
from ..utils import *
from ..attack import Attack

class NAT(Attack):
    """
    NAT Attack
    'NAT: Learning to Attack Neurons for Enhanced Adversarial Transferability (WACV 2025)'
    (https://openaccess.thecvf.com/content/WACV2025/papers/Nakka_NAT_Learning_to_Attack_Neurons_for_Enhanced_Adversarial_Transferability_WACV_2025_paper.pdf)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.  
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        nat_attacked_neuron (int): the specific neuron index to attack.
        checkpoint_path (str): the path to load pre-trained generators.

    Official arguments:
        epsilon=16/255, nat_attacked_neuron=250.

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/nat/generation --attack nat --model=generation --nat_attacked_neuron 250
        python main.py --input_dir ./path/to/data --output_dir adv_data/nat/generation --eval --nat_attacked_neuron 250
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, targeted=False, random_start=False,
                norm='linfty', loss='crossentropy', device=None, attack='NAT', nat_attacked_neuron=250, 
                checkpoint_path='./checkpoints/', **kwargs):
        
        print("\n" + "="*80)
        print("🎯 INITIALIZING NAT ATTACK (WACV 2025)")
        print("   Neural Activation Targeting for Enhanced Adversarial Transferability")
        print(f"   Target Neuron: {nat_attacked_neuron} | Epsilon: {epsilon:.4f}")
        print("="*80)
        
        self.nat_attacked_neuron = nat_attacked_neuron
        self.checkpoint_path = checkpoint_path
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha

    def load_model(self, model_name):
        """
        Load the NAT generator model from HuggingFace or local checkpoint
        
        Arguments:
            model_name (str): the name of surrogate model in model_list in utils.py
        
        Returns:
            model (torch.nn.Module): the NAT generator model
        """
        if model_name == 'generation':
            try:
                # Try to download from HuggingFace
                from huggingface_hub import snapshot_download
                
                os.makedirs(self.checkpoint_path, exist_ok=True)
                
                snapshot_download(
                    repo_id="KKNakka/NAT",
                    local_dir=self.checkpoint_path,
                    local_dir_use_symlinks=False,
                )
                
                model_path = os.path.join(self.checkpoint_path, f'0_net_G_neuron={self.nat_attacked_neuron}.pth')
            except Exception as e:
                # Fallback to local path
                model_path = os.path.join(self.checkpoint_path, f'0_net_G_neuron={self.nat_attacked_neuron}.pth')
                
            if not os.path.exists(model_path):
                raise ValueError(f"NAT checkpoint not found at '{model_path}'. Please download from HuggingFace: https://huggingface.co/KKNakka/NAT")
                
            print("\n" + "="*40)
            print(f"📥 LOADING NAT GENERATOR MODEL")
            print(f"   Model Path: {model_path}")
            print(f"   Target Neuron: {self.nat_attacked_neuron}")
            print(f"   For other neuron, please specify --nat_attacked_neuron argument (default: 250)")
            print("="*40)

            # Load the generator directly (the .pth file contains the full model)
            model = torch.load(model_path, map_location='cpu', weights_only=True)
            
            if hasattr(model, 'eval'):
                model = model.eval()
            else:
                # If it's a state dict, create the model and load weights
                generator = StableGeneratorResnet(gen_dropout=0.0, data_dim='high')
                if isinstance(model, dict) and 'model_state_dict' in model:
                    generator.load_state_dict(model['model_state_dict'])
                else:
                    generator.load_state_dict(model)
                model = generator.eval()
                
        else:
            raise ValueError('model:{} not supported for NAT attack'.format(model_name))

        return model.cuda()
    
    def forward(self, data, label, **kwargs):
        """
        The NAT attack procedure - generates adversarial examples using the pre-trained generator
        targeting specific neurons instead of full embeddings like LTP.

        Arguments:
            data: (N, C, H, W) tensor for input images
            label: (N,) tensor for ground-truth labels if untargeted, otherwise targeted labels
        """
        
        data = data.clone().detach().to(self.device)

        # Generate adversarial examples using the pre-trained NAT generator
        with torch.no_grad():
            adv_data = self.model(data)

        # Clip the perturbation within epsilon bounds
        delta = adv_data - data
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        delta = clamp(delta, img_min-data, img_max-data)        
        return delta.detach()


###########################
# NAT Generator: StableGeneratorResnet  
###########################

# To control feature map in generator
ngf = 64

class StableGeneratorResnet(nn.Module):
    """
    Modified ResNet-based generator for NAT attack.
    Key differences from LTP generator:
    - Removed ReflectionPad2d for deterministic results
    - Modified padding strategy
    - Targets specific neurons instead of full embeddings
    """
    
    def __init__(self, gen_dropout, data_dim, inception=False, isTrain=False):
        """
        :param inception: if True crop layer will be added to go from 3x300x300 to 3x299x299.
        :param data_dim: for high dimensional dataset (imagenet) 6 resblocks will be add otherwise only 2.
        :param gen_dropout: dropout rate for residual blocks
        :param isTrain: training mode flag  
        """
        super(StableGeneratorResnet, self).__init__()
        
        assert data_dim == "high", "NAT generator requires data_dim='high' for ImageNet"
        
        self.inception = inception
        self.data_dim = data_dim
        
        # Input_size = 3, n, n
        self.block1 = nn.Sequential(
            # Removed nn.ReflectionPad2d(3) for deterministic results
            nn.Conv2d(
                in_channels=3, out_channels=ngf, kernel_size=7, padding=3, bias=False
            ),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        )

        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=ngf,
                out_channels=ngf * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
        )

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
        )

        # Input size = 3, n/4, n/4
        # Residual Blocks: 6 for high-dimensional data (ImageNet)
        self.resblock1 = ResidualBlock(ngf * 4, gen_dropout)
        self.resblock2 = ResidualBlock(ngf * 4, gen_dropout)
        self.resblock3 = ResidualBlock(ngf * 4, gen_dropout)
        self.resblock4 = ResidualBlock(ngf * 4, gen_dropout)
        self.resblock5 = ResidualBlock(ngf * 4, gen_dropout)
        self.resblock6 = ResidualBlock(ngf * 4, gen_dropout)

        # Input size = 3, n/4, n/4  
        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(
                ngf * 4,
                ngf * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
        )

        # Input size = 3, n/2, n/2
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(
                ngf * 2,
                ngf,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        )

        # Input size = 3, n, n
        self.blockf = nn.Sequential(
            # Removed nn.ReflectionPad2d(3) for deterministic results
            nn.Conv2d(ngf, 3, kernel_size=7, padding=3)
        )

        # Crop layer for InceptionV3 compatibility
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

        # Output range [0, 1] 
        return (torch.tanh(x) + 1) / 2.0


class ResidualBlock(nn.Module):
    """
    Residual block for StableGeneratorResnet
    Modified to remove ReflectionPad for deterministic behavior
    """
    def __init__(self, num_filters, gen_dropout):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            # Removed nn.ReflectionPad2d(1) for deterministic results
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),
            nn.Dropout(gen_dropout),
            # Removed nn.ReflectionPad2d(1) for deterministic results
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_filters),
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual