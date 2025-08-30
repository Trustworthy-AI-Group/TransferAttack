import random
import functools
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TFF

from ..utils import *
from ..attack import Attack

class OPS(Attack):
    ''' 
     OPS (Operator-Perturbation-based Stochastic optimization) Attack
    'Boosting Adversarial Transferability through Augmentation in Hypothesis Space (CVPR 2025)'(https://openaccess.thecvf.com/content/CVPR2025/papers/Guo_Boosting_Adversarial_Transferability_through_Augmentation_in_Hypothesis_Space_CVPR_2025_paper.pdf)

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
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1.0, beta=2., num_sample_neighbor=10, num_sample_operator=20, sample_levels = range(2, 5), sample_ratios = np.arange(0., 1.5, 0.25) + 0.25
    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/ops/resnet50 --attack=ops --model resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/ops/resnet50 --eval
    '''
    
    def __init__(self, model_name, epsilon=16/255, beta=2., epoch=10, num_sample_neighbor=10, num_sample_operator=20, sample_levels = range(2, 5), sample_ratios = np.arange(0., 1.5, 0.25) + 0.25, decay=1., 
                 targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None, attack='OPS', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = epsilon / epoch
        self.epoch = epoch
        self.decay = decay

        self.using_sampling = (num_sample_operator * num_sample_neighbor > 0)

        if self.using_sampling:
            # NOTE: operator sampling
            self.num_sample_operator = num_sample_operator
            self.basic_ops = [
                identity, vertical_flip, horizontal_flip, vertical_shift, horizontal_shift, 
                rotate(5), rotate(-5), rotate(15), rotate(-15), rotate(45), rotate(-45), rotate(90), rotate(-90), rotate(180), 
                scaling(2), scaling(3), scaling(4), scaling(5), scaling(6), scaling(7), scaling(8),
                dim(1.1), dim(1.3), dim(1.5), dim(1.7), dim(1.9), dim(2.1), dim(2.3), dim(2.5), dim(2.7), dim(2.9),
            ]
            self.sample_levels = sample_levels
            self.op_list = []
            self.num_extra_ops = len(self.basic_ops)

            # NOTE: perturbation sampling
            self.num_sample_neighbor = num_sample_neighbor
            self.sample_radius = beta * epsilon * sample_ratios
            self.eps_list = []
            self.num_extra_eps = self.num_sample_neighbor

    # NOTE: operator sampling
    @property
    def op_num(self):
        return len(self.op_list)

    def get_new_ops(self, k=2):
        sel_ops = random.choices(self.basic_ops, k=k)
        new_op = lambda x: x
        new_op = functools.reduce(lambda f, g: lambda x: f(g(x)), sel_ops, new_op)
        return new_op

    def expand_op_list(self, k=2):
        for _ in range(self.num_extra_ops):
            self.op_list.append(self.get_new_ops(k=k))

    def init_op_list(self):
        self.op_list = []
        for level in self.sample_levels:
            if level == 1:
                self.op_list.append(self.basic_ops.copy())
            else:
                self.expand_op_list(level)

    # NOTE: perturbation sampling
    @property
    def eps_num(self):
        return len(self.eps_list)

    def expand_eps_list(self, delta, radius=1.):
        shape = (self.num_extra_eps, *delta.shape[1:])
        noise = torch.zeros(shape).uniform_(-radius, radius).to(self.device)
        self.eps_list.extend(noise)

    def init_eps_list(self, delta):
        self.eps_list = []
        for radius in self.sample_radius:
            self.expand_eps_list(delta, radius)

    def get_averaged_gradient(self, data, delta, label, **kwargs):
        """
        Calculate the averaged updated gradient
        """
        averaged_gradient = self.get_surrogate_gradient(data, delta, label)
        if not self.using_sampling:
            return averaged_gradient

        selected_eps = random.sample(self.eps_list, min(self.num_sample_neighbor, self.eps_num))
        for eps in selected_eps:
            x_near = data + delta + eps

            self.init_op_list()
            selected_ops = random.sample(self.op_list, min(self.num_sample_operator, self.op_num))
            for op in selected_ops:
                logits = self.get_logits(op(x_near))
                loss   = self.get_loss(logits, label)
                grad   = self.get_grad(loss, delta)
                averaged_gradient += grad

        return averaged_gradient / (self.num_sample_neighbor * self.num_sample_operator + 1)

    def get_surrogate_gradient(self, data, delta, label, **kwargs):
        logits = self.get_logits(data + delta)
        loss   = self.get_loss(logits, label)
        grad   = self.get_grad(loss, delta)
        return grad

    def forward(self, data, label, **kwargs):
        """
        The attack procedure for PGN

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
        if self.using_sampling:
            self.init_eps_list(delta)

        momentum, averaged_gradient = 0, 0
        for _ in range(self.epoch):
            averaged_gradient = self.get_averaged_gradient(data, delta, label)
            momentum = self.get_momentum(averaged_gradient, momentum)
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()

def vertical_shift(x):
        _, _, w, _ = x.shape
        step = np.random.randint(low = 0, high=w, dtype=np.int32)
        return x.roll(step, dims=2)

def horizontal_shift(x):
    _, _, _, h = x.shape
    step = np.random.randint(low = 0, high=h, dtype=np.int32)
    return x.roll(step, dims=3)

def vertical_flip(x):
    return x.flip(dims=(2,))

def horizontal_flip(x):
    return x.flip(dims=(3,))

class scaling():
    def __init__(self, scale) -> None:
        self.scale = scale
    
    def __call__(self, x):
        return x / self.scale

class dim():
    def __init__(self, resize_rate=1.1, diversity_prob=0.5) -> None:
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        
    def __call__(self, x):
        """
        Random transform the input images
        """
        # do not transform the input image
        #if torch.rand(1) > self.diversity_prob:
        #    return x
        
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

def identity(x):
    return x

class rotate():
    def __init__(self, angle) -> None:
        self.angle = angle
    
    def __call__(self, x):
        return TFF.rotate(img=x, angle=self.angle)

