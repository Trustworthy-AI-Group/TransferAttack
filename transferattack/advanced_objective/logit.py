import torch
import torch.nn.functional as F
from ..utils import *
from ..attack import Attack
import scipy.stats as st

class LOGIT(Attack):
    """
    Logit Attack
    'On Success and Simplicity: A Second Look at Transferable Targeted Attacks (NeurIPS 2021)'(https://arxiv.org/abs/2012.11207)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        targeted (bool): targeted/untargeted attack
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=2.0/255, epoch=300, decay=1.0

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/logit/resnet18_targeted --attack logit --model=resnet18 --targeted
    """

    def __init__(self, model_name, epsilon=16/255, alpha=2/255, epoch=300, decay=1.0, resize_rate=1.1, kernel_size=5, targeted=True, 
                 random_start=False, norm='linfty', loss='crossentropy', attack='Logit', device=None, **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device, **kwargs)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.resize_rate = resize_rate
        self.kernel = self.generate_kernel('gaussian', kernel_size)
        
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

    def get_grad(self, loss, delta, **kwargs):
        """
        Overridden for TIM attack.
        """
        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
        grad = F.conv2d(grad, self.kernel, stride=1, padding='same', groups=3)
        return grad

    def get_loss(self, logits, data, label):
        loss = 0
        for batch_i in range(data.shape[0]):
            loss += logits[batch_i][label[batch_i]]
        return loss
        
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

        momentum = 0.
        delta = self.init_delta(data).to(self.device)
        for _ in range(self.epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))

            # Calculate the loss
            loss = self.get_loss(logits, data, label)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()
