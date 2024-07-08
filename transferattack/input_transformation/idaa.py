import torch
import torch.nn.functional as F

from ..utils import *
from ..gradient.mifgsm import MIFGSM

import scipy.stats as st

class IDAA(MIFGSM):
    """
    IDAA(Input-Diversity-based Adaptive Attack)
    'Boosting the Transferability of Adversarial Examples via Local Mixup and Adaptive Step Size'(https://arxiv.org/pdf/2401.13205)
    
    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_scale (int): the number of shuffled copies in each iteration.
        num_block (int): the number of block in the image.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., num_scale=10, num_block=3
    
    Example script:
        python main.py --attack sia --output_dir adv_data/sia/resnet18
    """
    
    def __init__(self, model_name, epsilon=0.07, alpha=1, epoch=10, decay=1., num_scale=20, num_block=3, crop_size=0.7, targeted=False, random_start=False, 
                norm='linfty', loss='crossentropy', device=None, attack='IDAA', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.num_scale = num_scale
        self.num_block = num_block
        self.kernel = self.gkern()
        self.crop_size = crop_size
        self.op = [self.vertical_shift, self.horizontal_shift, self.vertical_flip, self.horizontal_flip, self.rotate180, self.scale, self.add_noise]

    def vertical_shift(self, x):
        _, _, w, _ = x.shape
        step = np.random.randint(low = 0, high=w, dtype=np.int32)
        return x.roll(step, dims=2)

    def horizontal_shift(self, x):
        _, _, _, h = x.shape
        step = np.random.randint(low = 0, high=h, dtype=np.int32)
        return x.roll(step, dims=3)

    def vertical_flip(self, x):
        return x.flip(dims=(2,))

    def horizontal_flip(self, x):
        return x.flip(dims=(3,))

    def rotate180(self, x):
        return x.rot90(k=2, dims=(2,3))
    
    def scale(self, x):
        return torch.rand(1)[0] * x

    def add_noise(self, x):
        return torch.clip(x + torch.zeros_like(x).uniform_(-16/255,16/255), 0, 1)

    def gkern(self, kernel_size=3, nsig=3):
        x = np.linspace(-nsig, nsig, kernel_size)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return torch.from_numpy(stack_kernel.astype(np.float32)).to(self.device)

    def blur(self, x):
        return F.conv2d(x, self.kernel, stride=1, padding='same', groups=3)

    def blocktransform(self, x, choice=-1):
        _, _, w, h = x.shape
        y_axis = [0,] + np.random.choice(list(range(1, h)), self.num_block-1, replace=False).tolist() + [h,]
        x_axis = [0,] + np.random.choice(list(range(1, w)), self.num_block-1, replace=False).tolist() + [w,]
        y_axis.sort()
        x_axis.sort()
        
        x_copy = x.clone()
        for i, idx_x in enumerate(x_axis[1:]):
            for j, idx_y in enumerate(y_axis[1:]):
                chosen = choice if choice >= 0 else np.random.randint(0, high=len(self.op), dtype=np.int32)
                x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y] = self.op[chosen](x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y])

        return x_copy

    def transform(self, x, **kwargs):
        """
        Scale the input for BSR
        """
        return torch.cat([self.blocktransform(x) for _ in range(self.num_scale)])

    def get_loss(self, logits, label):
        """
        Calculate the loss
        """
        return -self.loss(logits, label.repeat(self.num_scale)) if self.targeted else self.loss(logits, label.repeat(self.num_scale))
    
    
    def get_bound(self, x):
        lower_bound = -torch.min(x,self.epsilon*torch.ones_like(x))
        upper_bound = torch.min(1-x, self.epsilon*torch.ones_like(x))
        return lower_bound, upper_bound
    
    def compute_perturbation(self, w, lb, ub):
        return lb + (ub-lb) * (torch.tanh(w)/2 + 1/2) 
    
    def update_delta(self, delta, data, grad, alpha, **kwargs):
        delta = delta + alpha * grad.sign()
        return delta
    
    
    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

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
        ub, lb = self.get_bound(data)
        
        # Initialize adversarial perturbation
        delta = self.init_delta(data)
        r = self.compute_perturbation(delta, lb, ub)
        crop_H = int(data.shape[2] * self.crop_size)
        crop_W = int(data.shape[3] * self.crop_size)

        momentum = 0
        for _ in range(self.epoch):
            # Obtain the output
            
            B1 = self.transform(data+self.compute_perturbation(delta, lb, ub), momentum=momentum)
            B2 = self.transform(data+self.compute_perturbation(delta, lb, ub), momentum=momentum)
            
            # randomly select sxs from B2 and mix it up with sxs in  B1
            
            start_h = np.random.randint(0, data.shape[2]-crop_H)
            start_w = np.random.randint(0, data.shape[3]-crop_W)
            croped_B2 = B2[:, :, start_h:start_h+crop_H, start_w:start_w+crop_W]
            
            start_h = np.random.randint(0, data.shape[2]-crop_H)
            start_w = np.random.randint(0, data.shape[3]-crop_W)
            B1[:, :, start_h:start_h+crop_H, start_w:start_w+crop_W] = croped_B2
            
            
            
            logits = self.get_logits(B1)

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return self.compute_perturbation(delta, lb, ub)
    
    