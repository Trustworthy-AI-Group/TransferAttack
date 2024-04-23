import torch
from torchvision.transforms import RandomResizedCrop
import torch.nn.functional as F

import scipy.stats as st
import numpy as np

from ..utils import *
from ..gradient.mifgsm import MIFGSM

class SU(MIFGSM):
    """
    SU Attack (Self-University attack)
    'Enhancing the Self-Universality for Transferable Targeted Attacks'(https://arxiv.org/pdf/2209.03716.pdf)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        resize_rate (float): the relative size of the resized image
        diversity_prob (float): the probability for transforming the input image
        lamb (float): the the weights of the similarity loss.
        kernel_type (str): the type of kernel (gaussian/uniform/linear).
        kernel_size (int): the size of kernel.
        targeted (bool): targeted/untargeted attack.
        feature_layer (str): the specific intermediate layer for feature extraction.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=2/255, epoch=300, decay=1, resize_rate=1.1, diversity_prob=0.5, lamb=0.4, kernel_type='gaussian', kernel_size=15, feature_layer='layer3'

    Example script:
        python main.py --attack su --output_dir adv_data/su/resnet18 --targeted
    """

    def __init__(self, model_name, epsilon=16/255, alpha=2/255, epoch=300, decay=1., lamb=0.001, scale=(0.1, 0.), feature_layer='layer3', targeted=True, random_start=False, norm='linfty', loss='crossentropy', device=None, attack='SU', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.start, self.interval = scale
        self.feature_layer = self.find_layer(feature_layer)
        self.lamb = lamb
        self.local_transform = RandomResizedCrop(img_height, scale=(self.start, self.start+self.interval))
        self.kernel = self.generate_kernel('gaussian', 5)
        self.resize_rate = 1.1
        self.diversity_prob = 0.7
    
    def transform(self, x, **kwargs):
        """
        Random transform the input images
        """
        # do not transform the input image
        if torch.rand(1) > self.diversity_prob:
            return x
        
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        # resize the input image to random size
        rnd = torch.randint(low=min(img_size, img_resize), high=max(img_size, img_resize), size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)

        # randomly add padding
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        # resize the image back to img_size
        return F.interpolate(padded, size=[img_size, img_size], mode='bilinear', align_corners=False)

    def generate_kernel(self, kernel_type, kernel_size, nsig=3):
        """
        Generate the gaussian/uniform/linear kernel

        Arguments:
            kernel_type (str): the method for initilizing the kernel
            kernel_size (int): the size of kernel
        """
        if kernel_type.lower() == 'gaussian':
            x = np.linspace(-nsig, nsig, kernel_size)
            kern1d = st.norm.pdf(x)
            kernel_raw = np.outer(kern1d, kern1d)
            kernel = kernel_raw / kernel_raw.sum()
        elif kernel_type.lower() == 'uniform':
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        elif kernel_type.lower() == 'linear':
            kern1d = 1 - np.abs(np.linspace((-kernel_size+1)//2, (kernel_size-1)//2, kernel_size)/(kernel_size**2))
            kernel_raw = np.outer(kern1d, kern1d)
            kernel = kernel_raw / kernel_raw.sum()
        else:
            raise Exception("Unspported kernel type {}".format(kernel_type))

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return torch.from_numpy(stack_kernel.astype(np.float32)).to(self.device)

    def get_grad(self, loss, delta, **kwargs):
        """
        Overridden for TIM attack.
        """
        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
        grad = F.conv2d(grad, self.kernel, stride=1, padding='same', groups=3)
        return grad

    def find_layer(self,layer_name):
        parser = layer_name.split(' ')
        m = self.model[1]
        for layer in parser:
            if layer not in m._modules.keys():
                print("Selected layer is not in Model")
                exit()
            else:
                m = m._modules.get(layer)
        return m

    def __forward_hook(self,m,i,o):
        global mid_output
        mid_output = o

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
        batch_size = data.shape[0]

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        h = self.feature_layer.register_forward_hook(self.__forward_hook)

        momentum = 0
        for _ in range(self.epoch):
            # Obtain the global and local input
            li_inputs = self.local_transform(data)
            accom_inputs = torch.concat([data+delta, li_inputs+delta], dim=0)

            # Get the logits output after DI transform
            logits = self.get_logits(self.transform(accom_inputs))

            # Calculate the loss of global and local input
            loss = self.get_loss(logits, torch.cat([label, label], dim=0))

            # # Get feature of input
            fs_loss = torch.nn.functional.cosine_similarity(mid_output[:batch_size].view(batch_size, -1), mid_output[-batch_size:].view(batch_size, -1))

            fs_loss = torch.mean(fs_loss)

            loss += self.lamb * fs_loss

            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        h.remove()
        return delta.detach()