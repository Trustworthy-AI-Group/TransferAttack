import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import os

from ..utils import *
from ..gradient.mifgsm import MIFGSM


class STM(MIFGSM):
    """
    STM (Style Transfer attack Method)
    'Improving the Transferability of Adversarial Examples with Arbitrary Style Transfer. (ACM MM 2023)' (https://arxiv.org/abs/2308.10601) 
    (This code is copied from https://github.com/Zhijin-Ge/STM)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_style (int): the number of style transfer images.
        gamma (float): mixing up factor.
        beta (float): the upper bound of neighborhood x
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1, num_style=20, gamma=0.5, beta=2.0

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/stm/resnet18 --attack stm --model=resnet18
    Notes: 
        Download checkpoints ('checkpoint_transformer.pth' and 'checkpoint_embeddings.pth') from https://github.com/Zhijin-Ge/STM,
        and put them in the path '/path/to/checkpoints/'
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., num_style=20, gamma=0.5, beta = 2.0, targeted=False, 
                random_start=False, norm='linfty', loss='crossentropy', device=None, attack='STM', checkpoint_path='./path/to/checkpoints/', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.num_style = num_style
        self.epsilon = epsilon
        self.gamma = gamma
        self.beta = beta
        self.checkpoint_path = checkpoint_path


    def transform(self, x, **kwargs):
        """
        Use an arbitrary style transfer network to transform the images into different domains
        Mix up the generated images added by random noise with the original images to maintain semantic consistency and boost input diversity

        Arguments:
            x: (N, C, H, W) tensor for input images
        """
        augmentor = StyleAugmentor(self.checkpoint_path)
        x_aug = augmentor(x)
        x_sty = self.gamma*x + (1-self.gamma)*x_aug.detach().clone() + torch.randn_like(x).uniform_(-self.epsilon*self.beta, self.epsilon*self.beta).cuda()

        return x_sty

    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):
            grads = 0
            for _ in range(self.num_style):
                # Obtain the stylized data
                x_s = self.transform(data + delta)

                # Obtain the output
                logits = self.get_logits(x_s)

                # Calculate the loss
                loss = self.get_loss(logits, label)

                # Calculate the gradients on x_s
                grad = self.get_grad(loss, x_s)
                grads += grad

            grads /= self.num_style

            # Calculate the momentum
            momentum = self.get_momentum(grads, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()

"""The style transfer model"""
class ConvInRelu(nn.Module):
    def __init__(self,channels_in,channels_out,kernel_size,stride=1):
        super(ConvInRelu,self).__init__()
        self.n_params = 0
        self.channels = channels_out
        self.reflection_pad = nn.ReflectionPad2d(int(np.floor(kernel_size/2)))
        self.conv = nn.Conv2d(channels_in,channels_out,kernel_size,stride,padding=0)
        self.instancenorm = nn.InstanceNorm2d(channels_out)
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x):
        # x: B x C_in x H x W

        x = self.reflection_pad(x)
        x = self.conv(x) # B x C_out x H x W
        x = self.instancenorm(x) # B x C_out x H x W
        x = self.relu(x) # B x C_out x H x W
        return x


class UpsampleConvInRelu(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, upsample, stride=1, activation=nn.ReLU):
        super(UpsampleConvInRelu, self).__init__()
        self.n_params = channels_out * 2
        self.upsample = upsample
        self.channels = channels_out

        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, stride)
        self.instancenorm = nn.InstanceNorm2d(channels_out)
        self.fc_beta = nn.Linear(100,channels_out)
        self.fc_gamma = nn.Linear(100,channels_out)
        if activation:
            self.activation = activation(inplace=False)
        else:
            self.activation = None
        
    def forward(self, x, style):
        # x: B x C_in x H x W
        # style: B x 100

        beta = self.fc_beta(style).unsqueeze(2).unsqueeze(3) # B x C_out x 1 x 1
        gamma = self.fc_gamma(style).unsqueeze(2).unsqueeze(3) # B x C_out x 1 x 1

        if self.upsample:
            x = self.upsample_layer(x)
        x = self.reflection_pad(x)
        x = self.conv(x)
        x = self.instancenorm(x)
        x = gamma * x
        x += beta
        if self.activation:
            x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    # modelled after that used by Johnson et al. (2016)
    def __init__(self,channels):
        super(ResidualBlock,self).__init__()
        self.n_params = channels * 4
        self.channels = channels

        self.reflection_pad = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(channels,channels,3,stride=1,padding=0)
        self.instancenorm = nn.InstanceNorm2d(channels)
        self.fc_beta1 = nn.Linear(100,channels)
        self.fc_gamma1 = nn.Linear(100,channels)
        self.fc_beta2 = nn.Linear(100,channels)
        self.fc_gamma2 = nn.Linear(100,channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(channels,channels,3,stride=1,padding=0)
        
    def forward(self, x, style):
        # x: B x C x H x W  
        # style: B x self.n_params
        
        beta1 = self.fc_beta1(style).unsqueeze(2).unsqueeze(3) # B x C x 1 x 1
        gamma1 = self.fc_gamma1(style).unsqueeze(2).unsqueeze(3) # B x C x 1 x 1
        beta2 = self.fc_beta2(style).unsqueeze(2).unsqueeze(3) # B x C x 1 x 1
        gamma2 = self.fc_gamma2(style).unsqueeze(2).unsqueeze(3) # B x C x 1 x 1

        y = self.reflection_pad(x)
        y = self.conv1(y)
        y = self.instancenorm(y)
        y = gamma1 * y
        y += beta1
        y = self.relu(y)
        y = self.reflection_pad(y)
        y = self.conv2(y)
        y = self.instancenorm(y)
        y = gamma2 * y
        y += beta2
        return x + y


class Ghiasi(nn.Module):
    def __init__(self):
        super(Ghiasi,self).__init__()
        self.layers = nn.ModuleList([
            ConvInRelu(3,32,9,stride=1),
            ConvInRelu(32,64,3,stride=2),
            ConvInRelu(64,128,3,stride=2),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            UpsampleConvInRelu(128,64,3,upsample=2),
            UpsampleConvInRelu(64,32,3,upsample=2),
            UpsampleConvInRelu(32,3,9,upsample=None,activation=None)
        ])

        self.n_params = sum([layer.n_params for layer in self.layers])
    
    def forward(self,x,styles):
        # x: B x 3 x H x W
        # styles: B x 100 batch of style embeddings
        
        for i, layer in enumerate(self.layers):
            if i < 3:
                # first three layers do not perform renormalization (see style_normalization_activations in the original source)
                x = layer(x)
            else:
                x = layer(x, styles)
        
        return torch.sigmoid(x)

"""Style Augument"""
class StyleAugmentor(nn.Module):
    def __init__(self, checkpoint_path):
        super(StyleAugmentor, self).__init__()

        # create transformer and style predictor networks:
        self.ghiasi = Ghiasi()
        self.ghiasi.cuda()
        # Checkpoints are from https://github.com/Zhijin-Ge/STM:
        checkpoint_ghiasi_name = os.path.join(checkpoint_path, 'checkpoint_transformer.pth')
        checkpoint_embeddings_name = os.path.join(checkpoint_path, 'checkpoint_embeddings.pth')

        if os.path.exists(checkpoint_ghiasi_name) and os.path.exists(checkpoint_embeddings_name):
            pass
        else:
            raise ValueError("Please download checkpoints from 'https://drive.google.com/drive/folders/1NkD91e3NbSQlZUflc63kgjqlgXIhzcxg?usp=sharing', and put them into the path './path/to/checkpoints'.")
        
        checkpoint_ghiasi = torch.load(checkpoint_ghiasi_name)
        checkpoint_embeddings = torch.load(checkpoint_embeddings_name)
        
        # load weights for ghiasi and stylePredictor, and mean / covariance for the embedding distribution:
        self.ghiasi.load_state_dict(checkpoint_ghiasi['state_dict_ghiasi'],strict=False)

        # load mean imagenet embedding:
        self.imagenet_embedding = checkpoint_embeddings['imagenet_embedding_mean'] # mean style embedding for ImageNet
        self.imagenet_embedding = self.imagenet_embedding.cuda()

        # get mean and covariance of PBN style embeddings:
        self.mean = checkpoint_embeddings['pbn_embedding_mean']
        self.mean = self.mean.cuda() # 1 x 100
        self.cov = checkpoint_embeddings['pbn_embedding_covariance']
        
        # compute SVD of covariance matrix:
        u, s, vh = np.linalg.svd(self.cov.numpy())
        
        self.A = np.matmul(u,np.diag(s**0.5))
        self.A = torch.tensor(self.A).float().cuda() # 100 x 100
        # self.cov = cov(Ax), x ~ N(0,1)
    
    def sample_embedding(self, n):
        # n: number of embeddings to sample
        # returns n x 100 embedding tensor
        embedding = torch.randn(n,100).cuda() # n x 100
        embedding = torch.mm(embedding,self.A.transpose(1,0)) + self.mean # n x 100
        return embedding

    def forward(self, x, alpha=1.0, downsamples=0, embedding=None):
        # augments a batch of images with style randomization
        # x: B x C x H x W image tensor
        # alpha: float in [0,1], controls interpolation between random style and original style
        # downsamples: int, number of times to downsample by factor of 2 before applying style transfer
        # embedding: B x 100 tensor, or None. Use this embedding if provided.
        # style embedding for when alpha=0:

        if downsamples:
            assert(x.size(2) % 2**downsamples == 0)
            assert(x.size(3) % 2**downsamples == 0)
            for i in range(downsamples):
                x = nn.functional.avg_pool2d(x,2)

        if embedding is None:
            # sample a random embedding
            embedding = self.sample_embedding(x.size(0))
        # interpolate style embeddings:
        embedding = alpha*embedding
        
        restyled = self.ghiasi(x,embedding)

        if downsamples:
            restyled = nn.functional.upsample(restyled,scale_factor=2**downsamples,mode='bilinear')
        
        return restyled.detach() # detach prevents the user from accidentally backpropagating errors into stylePredictor or ghiasi while training a downstream model
