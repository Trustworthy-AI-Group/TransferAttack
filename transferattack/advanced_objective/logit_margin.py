import torch
import torch.nn.functional as F

from ..utils import *
from ..gradient.mifgsm import MIFGSM
import scipy.stats as st
import numpy as np

class Logit_Margin(MIFGSM):
    """
    Logit_Margin Attack
    'Logit Margin Matters: Improving Transferable Targeted Adversarial Attack by Logit Calibration (TIFS 2023)'(https://arxiv.org/abs/2303.03680)

    Arguments:
        model (torch.nn.Module): the surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        resize_rate (float): the relative size of the resized image
        diversity_prob (float): the probability for transforming the input image
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
    
    Official arguments:
        epsilon=16/255, alpha=2.0/255, epoch=200, decay=1, resize_rate=1.1, diversity_prob=0.7
    Example Script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/logit_margin/resnet18_targeted --attack logit_margin --model=resnet18 --targeted
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=2/255, epoch=300, decay=1., temperature=5, resize_rate=1.1, diversity_prob=0.7,
                kernel_type='gaussian', kernel_size=5, targeted=True, feature_layer='fc',
                random_start=False, norm='linfty', loss='crossentropy', device=None, attack='Logit_Margin', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        if resize_rate < 1:
            raise Exception("Error! The resize rate should be larger than 1.")
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.kernel = self.generate_kernel(kernel_type, kernel_size)
        self.loss_type = 'Margin-based' # (Temperature-based, Margin-based, Angle-based)
        self.temperature = temperature
        self.feature_layer = self.find_layer(feature_layer)
    
    def find_layer(self, layer_name):
        if layer_name not in self.model[1]._modules.keys():
            print("Selected layer is not in Model")
            exit()
        else:
            return self.model[1]._modules.get(layer_name)
    def __forward_hook(self,m,i,o):
        global mid_input
        mid_input = i

    # TIM
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
    
    # DIM
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

    def forward(self, data, label, **kwargs):
        """
        The Logit_Margin attack procedure

        Arguments:
            data (N, C, H, W): tensor for input images
            labels (N,): tensor for ground-truth labels if untargetd
            labels (2,N): tensor for [ground-truth, targeted labels] if targeted
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)
        h = self.feature_layer.register_forward_hook(self.__forward_hook)

        momentum = 0
        for _ in range(self.epoch):
            # Obtain the output, use DIM transform
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))

            # Calculate the loss
            if self.loss_type == 'Temperature-based':
                logits = logits / self.temperature
                loss = self.get_loss(logits, label)
            elif self.loss_type == 'Margin-based':
                value, _ = torch.sort(logits, dim=1, descending=True)
                logits = logits / torch.unsqueeze(value[:, 0] - value[:, 1], 1).detach()
                loss = self.get_loss(logits, label)
            else: # Angle-based
                model_weight = self.model[1].fc.weight.data
                feature = mid_input[0]
                output = F.linear(F.normalize(feature), F.normalize(model_weight))
                real = output.gather(1, label.unsqueeze(1)).squeeze(1)
                logit_dists = (-1 * real)
                loss = -logit_dists.sum()

            # Calculate the gradients, use TIM
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        h.remove()
        return delta.detach()
