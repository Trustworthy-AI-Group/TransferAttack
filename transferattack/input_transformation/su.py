import torch
from torchvision.transforms import RandomResizedCrop
import torch.nn.functional as F

import scipy.stats as st
import numpy as np

from ..utils import *
from ..gradient.mifgsm import MIFGSM

class LogitLoss(nn.Module):
    def __init__(self, ):
        super(LogitLoss, self).__init__()

    def forward(self, logits, labels):
        real = logits.gather(1,labels.unsqueeze(1)).squeeze(1)
        logit_dists = ( -1 * real)
        loss = logit_dists.mean()
        return loss

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
        python main.py --input_dir ./path/to/data --output_dir adv_data/su/resnet50_targeted --attack su --model=resnet50 --targeted --epoch=300
        python main.py --input_dir ./path/to/data --output_dir adv_data/su/resnet50_targeted --eval --targeted
    """

    def __init__(self, model_name, epsilon=16/255, alpha=2/255, epoch=300, decay=1., coef=0.001, scale=(0.1, 0.0), depth=3, targeted=True, random_start=False, norm='linfty', loss='crossentropy', device=None, attack='SU', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.model_name = model_name
        self.start, self.interval = scale
        self.coef = coef
        self.depth = depth

        self.local_transform = RandomResizedCrop(img_height, scale=(self.start, self.start+self.interval))
        self.gaussian_kernel = self._TI_kernel()
        self.resize_rate = 1.1
        self.diversity_prob = 0.7
        self._register_forward()

        self.loss_fn = LogitLoss()

    def _target_layer(self, model_name, depth):
        '''
        'inception_v3', 'resnet50', 'densenet121', 'vgg16_bn'
        depth: [1, 2, 3, 4]
        '''
        if model_name == 'resnet50':
            return getattr(self.model[1], 'layer{}'.format(depth))[-1]
        elif model_name == 'vgg16_bn':
            depth_to_layer = {1:12,2:22,3:32,4:42}
            return getattr(self.model[1], 'features')[depth_to_layer[depth]]
        elif model_name == 'densenet121':
            return getattr(getattr(self.model[1], 'features'), 'denseblock{}'.format(depth))
        elif model_name == 'inception_v3':
            depth_to_layer = {1:'Conv2d_4a_3x3', 2:'Mixed_5d', 3:'Mixed_6e', 4:'Mixed_7c'}
            return getattr(self.model[1], '{}'.format(depth_to_layer[depth]))

    def _register_forward(self):
        '''
        'inception_v3', 'resnet50', 'densenet121', 'vgg16_bn'
        '''
        self.activations = []
        def forward_hook(module, input, output):
            self.activations += [output]
            return None
        target_layer = self._target_layer(self.model_name, self.depth)    
        target_layer.register_forward_hook(forward_hook)

    def _DI(self, X_in):
        img_resize = int(img_height * self.resize_rate)
        rnd = np.random.randint(img_height, img_resize, size=1)[0]
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = np.random.randint(0, h_rem,size=1)[0]
        pad_bottom = h_rem - pad_top
        pad_left = np.random.randint(0, w_rem,size=1)[0]
        pad_right = w_rem - pad_left

        c = np.random.rand(1)
        if c <= 0.7:
            X_out = F.pad(F.interpolate(X_in, size=(rnd,rnd)),(pad_left,pad_right,pad_top,pad_bottom),mode='constant', value=0)
            return  X_out 
        else:
            return  X_in
            
    def _TI_kernel(self):
        def gkern(kernlen=15, nsig=3):
            x = np.linspace(-nsig, nsig, kernlen)
            kern1d = st.norm.pdf(x)
            kernel_raw = np.outer(kern1d, kern1d)
            kernel = kernel_raw / kernel_raw.sum()
            return kernel
        channels=3
        kernel_size=5
        kernel = gkern(kernel_size, 3).astype(np.float32)
        gaussian_kernel = np.stack([kernel, kernel, kernel])
        gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
        gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()
        return gaussian_kernel

    def get_grad(self, loss, delta, **kwargs):
        """
        Overridden for TIM attack.
        """
        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
        return grad

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
            used_coef = -1
        else:
            used_coef = 1

        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        batch_size = data.shape[0]

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):
            self.activations = []
            # Obtain the global and local input
            li_inputs = self.local_transform(data)
            accom_inputs = torch.concat([data+delta, li_inputs+delta], dim=0)
            logits = self.model(self._DI(accom_inputs))
            
            loss_label = torch.cat([label, label], dim=0)

            classifier_loss = self.loss_fn(logits, loss_label)

            # # Get feature of input
            fs_loss = torch.nn.functional.cosine_similarity(self.activations[0][:batch_size].view(batch_size, -1), self.activations[0][-batch_size:].view(batch_size, -1))

            fs_loss = torch.mean(fs_loss)

            loss = -(classifier_loss + self.coef * used_coef * fs_loss)

            grad = self.get_grad(loss, delta)
            grad = F.conv2d(grad, self.gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3)
            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()
