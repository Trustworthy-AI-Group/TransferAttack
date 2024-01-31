import torch
import torch.nn.functional as F
import scipy.stats as st

from ..utils import *
from ..gradient.mifgsm import MIFGSM

class POTRIP(MIFGSM):
    """
    Po+Trip Attack
    'Towards Transferable Targeted Attack (CVPR 2020)'(https://ieeexplore.ieee.org/document/9156367)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        resize_rate (float): the relative size of the resized image
        diversity_prob (float): the probability for transforming the input image
        lamb (float): the weight of triplet loss.
        gamma (float): the margin value.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model.
    
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=2.0/255, epoch=300, decay=1, resize_rate=1.1, diversity_prob=0.5
    
    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/potrip/resnet18_targeted --attack potrip --targeted
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=2/255, epoch=300, decay=1., resize_rate=1.1, 
                lamb = 0.01, gamma = 0.007, targeted=True, random_start=False, kernel_type='gaussian', kernel_size=5,
                norm='linfty', loss='crossentropy', device=None, attack='POTRIP', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        if resize_rate < 1:
            raise Exception("Error! The resize rate should be larger than 1.")
        self.resize_rate = resize_rate
        self.num_classes = 1000
        self.lamb =  lamb
        self.gamma = gamma
        self.kernel = self.generate_kernel(kernel_type, kernel_size)

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

    def Poincare_dis(self, a, b):
        L2_a = torch.sum(torch.square(a), 1)
        L2_b = torch.sum(torch.square(b), 1)

        theta = 2 * torch.sum(torch.square(a - b), 1) / ((1 - L2_a) * (1 - L2_b))
        distance = torch.mean(torch.acosh(1.0 + theta))
        return distance

    def Cos_dis(self, a, b):
        a_b = torch.abs(torch.sum(torch.multiply(a, b), 1))
        L2_a = torch.sum(torch.square(a), 1)
        L2_b = torch.sum(torch.square(b), 1)
        distance = torch.mean(a_b / torch.sqrt(L2_a * L2_b))
        return distance

    def get_loss(self, logits, labels_true, labels):
        batch_size = labels_true.shape[0]
        labels_onehot = torch.zeros(batch_size, 1000, device=self.device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        labels_true_onehot = torch.zeros(batch_size, 1000, device=self.device)
        labels_true_onehot.scatter_(1, labels_true.unsqueeze(1), 1)
        labels_infhot = torch.zeros_like(labels_onehot).scatter_(1, labels.unsqueeze(1), float('inf'))

        loss_po = self.Poincare_dis(logits / torch.sum(torch.abs(logits), 1, keepdim=True),torch.clamp((labels_onehot - 0.00001), 0.0, 1.0))
        loss_cos = torch.clamp(self.Cos_dis(labels_onehot, logits) - self.Cos_dis(labels_true_onehot, logits) + self.gamma, 0.0, 2.1)
        loss_total = loss_po + self.lamb * loss_cos
        return -loss_total if self.targeted else loss_total

    def get_grad(self, loss, delta, **kwargs):
        """
        Overridden for TIM attack.
        """
        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
        grad = F.conv2d(grad, self.kernel, stride=1, padding='same', groups=3)
        return grad
    
    def transform(self, x, **kwargs):
        """
        Random transform the input images
        """
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
        The general attack procedure

        Arguments:
            data (N, C, H, W): tensor for input images
            labels (N,): tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        if self.targeted:
            assert len(label) == 2
            ori_label = label[0]
            target_label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        ori_label = ori_label.clone().detach().to(self.device)
        target_label = target_label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))

            # Calculate the loss
            loss = self.get_loss(logits, ori_label, target_label)
            # loss = self.get_loss(logits, target_label)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()
