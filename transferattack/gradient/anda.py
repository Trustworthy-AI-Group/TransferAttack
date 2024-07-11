import torch
import math
from torch.nn import functional as F

from ..utils import *
from ..attack import Attack


class ANDA(Attack):
    """
    ANDA Attack
    'Strong Transferable Adversarial Attacks via Ensembled Asymptotically Normal Distribution Learning (CVPR 2024)'(https://openaccess.thecvf.com/content/CVPR2024/papers/Fang_Strong_Transferable_Adversarial_Attacks_via_Ensembled_Asymptotically_Normal_Distribution_Learning_CVPR_2024_paper.pdf)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        n_ens (int): the augmentation number.
        sample (bool): the output options, use sample or not.
        aug_max (float): the augmentation degree of the attack.
        epoch (int): the number of iterations.
        targeted (bool): targeted/untargeted attack
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epoch = 10
        n_ens = 25
        aug_max = 0.3
        sample = False
        epsilon = 16/255
        alpha = epsilon/epoch=1.6/255
    
    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/anda/resnet18 --attack anda --model=resnet18 --batchsize=1
        python main.py --input_dir ./path/to/data --output_dir adv_data/anda/resnet18 --eval
    
    Notes:
        - batchsize=1 only
        - MultiANDA requires torch.distributed, please refer to https://github.com/CLIAgroup/ANDA for more details
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, n_ens=25, aug_max=0.3, sample=False, targeted=False, random_start=False,
                norm='linfty', loss='crossentropy', device=None, attack='ANDA', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = 0
        self.n_ens = n_ens
        self.aug_max = aug_max
        self.sample = sample

        def is_sqr(n):
            a = int(math.sqrt(n))
            return a * a == n
        assert is_sqr(self.n_ens), "n_ens must be square number."

        self.thetas = self.get_thetas(int(math.sqrt(self.n_ens)), -self.aug_max, self.aug_max)

    def get_theta(self, i, j):
        theta = torch.tensor([[[1, 0, i], [0, 1, j]]], dtype=torch.float)
        return theta

    def get_thetas(self, n, min_r=-0.5, max_r=0.5):
        range_r = torch.linspace(min_r, max_r, n)
        thetas = []
        for i in range_r:
            for j in range_r:
                thetas.append(self.get_theta(i, j))
        thetas = torch.cat(thetas, dim=0)
        return thetas

    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data (N, C, H, W): tensor for input images
            labels (N,): tensor for ground-truth labels if untargetd
            labels (2,N): tensor for [ground-truth, targeted labels] if targeted
        """
        
        assert data.shape[0] == 1, "ANDA currently only supports batchsize=1"
        if label.ndim == 2:
            assert label.shape[1] == 1, "ANDA currently only supports batchsize=1"
        else:
            assert label.shape[0] == 1, "ANDA currently only supports batchsize=1"

        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        
        data = data.clone().detach().to(self.device)
        xt = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        min_x = data - self.epsilon
        max_x = data + self.epsilon

        data_shape = data.shape[1:]
        stat = ANDA_STATISTICS(data_shape=(1,) + data_shape, device=self.device)

        for _ in range(self.epoch):
            
            # Augment the data
            xt_batch = xt.repeat(self.n_ens, 1, 1, 1)
            xt_batch.requires_grad = True            
            aug_xt_batch = self.transform(thetas=self.thetas, data=xt_batch)
            labels = label.repeat(xt_batch.shape[0])

            # Obtain the output
            logits = self.get_logits(aug_xt_batch)

            # Calculate the loss
            loss = self.get_loss(logits, labels)

            # Calculate the gradients
            grad = self.get_grad(loss, xt_batch)

            # Collect the grads
            stat.collect_stat(grad)

            # Get mean of grads
            sample_noise = stat.noise_mean

            if self.sample and i == self.epoch - 1:
                # Sample noise
                sample_noises = stat.sample(n_sample=1, scale=1)
                sample_xt = self.alpha * sample_noises.squeeze().sign() + xt
                sample_xt = torch.clamp(sample_xt, 0.0, 1.0).detach()
                sample_xt = torch.max(torch.min(sample_xt, max_x), min_x).detach()

            # Update adv using mean of grads
            xt = xt + self.alpha * sample_noise.sign()

            # Clamp data into valid range
            xt = torch.clamp(xt, 0.0, 1.0).detach()
            xt = torch.max(torch.min(xt, max_x), min_x).detach()
        
        if self.sample:
            adv = sample_xt.detach().clone()
        else:
            adv = xt.detach().clone()
        
        delta = adv - data

        return delta.detach()
    
    def transform(self, thetas, data):
        grids = F.affine_grid(thetas, data.size(), align_corners=False).to(data.device)
        output = F.grid_sample(data, grids, align_corners=False)
        return output
    
    def get_loss(self, logits, labels):
        loss = F.cross_entropy(logits, labels, reduction="sum")
        return loss


class ANDA_STATISTICS:
    def __init__(self, device, data_shape=(1, 3, 224, 224)):
        self.data_shape = data_shape
        self.device = device

        self.n_models = 0
        self.noise_mean = torch.zeros(data_shape, dtype=torch.float).to(device)
        self.noise_cov_mat_sqrt = torch.empty((0, np.prod(data_shape)), dtype=torch.float).to(device)

    def sample(self, n_sample=1, scale=0.0, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        mean = self.noise_mean
        cov_mat_sqrt = self.noise_cov_mat_sqrt

        if scale == 0.0:
            assert n_sample == 1
            return mean.unsqueeze(0)

        assert scale == 1.0
        k = cov_mat_sqrt.shape[0]
        cov_sample = cov_mat_sqrt.new_empty((n_sample, k), requires_grad=False).normal_().matmul(cov_mat_sqrt)
        cov_sample /= (k - 1)**0.5

        rand_sample = cov_sample.reshape(n_sample, *self.data_shape)
        sample = mean.unsqueeze(0) + scale * rand_sample
        sample = sample.reshape(n_sample, *self.data_shape)
        return sample

    def collect_stat(self, noise):
        mean = self.noise_mean
        cov_mat_sqrt = self.noise_cov_mat_sqrt
        assert noise.device == cov_mat_sqrt.device
        bs = noise.shape[0]
        # first moment
        mean = mean * self.n_models / (self.n_models + bs) + noise.data.sum(dim=0, keepdim=True) / (self.n_models + bs)

        # square root of covariance matrix
        dev = (noise.data - mean).view(bs, -1)
        cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev), dim=0)

        self.noise_mean = mean
        self.noise_cov_mat_sqrt = cov_mat_sqrt
        self.n_models += bs

    def clear(self):
        self.n_models = 0
        self.noise_mean = torch.zeros(self.data_shape, dtype=torch.float).to(self.device)
        self.noise_cov_mat_sqrt = torch.empty((0, np.prod(self.data_shape)), dtype=torch.float).to(self.device)
