import torch

from ..utils import *
from ..gradient.mifgsm import MIFGSM

class DeCowA(MIFGSM):
    """
    DeCowA(Wapring Attack)
    'Boosting Adversarial Transferability across Model Genus by Deformation-Constrained Warping (AAAI 2024)'(https://arxiv.org/abs/2402.03951)
    
    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        mesh_width: the number of the control points
        mesh_height: the number of the control points = 3 * 3 = 9
        noise_scale: random noise strength
        num_warping: the number of warping transformation samples
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1.
    
    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/decowa/resnet18 --attack decowa --model=resnet18
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., mesh_width=3, mesh_height=3, rho=0.01, 
                 num_warping=20, noise_scale=2, targeted=False, random_start=False, norm='linfty', 
                 loss='crossentropy', device=None, attack='DeCowA', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.num_warping = num_warping
        self.noise_scale = noise_scale
        self.mesh_width = mesh_width
        self.mesh_height = mesh_height
        self.epsilon = epsilon
        self.rho = rho

    def vwt(self, x, noise_map):
        n, c, w, h = x.size()
        X = grid_points_2d(self.mesh_width, self.mesh_height, self.device)
        Y = noisy_grid(self.mesh_width, self.mesh_height, noise_map, self.device)
        tpsb = TPS(size=(h, w), device=self.device)
        warped_grid_b = tpsb(X[None, ...], Y[None, ...])
        warped_grid_b = warped_grid_b.repeat(x.shape[0], 1, 1, 1)
        vwt_x = torch.grid_sampler_2d(x, warped_grid_b, 0, 0, False)
        return vwt_x
    
    def update_noise_map(self, x, label):
        x.requires_grad = False
        noise_map = (torch.rand([self.mesh_height - 2, self.mesh_width - 2, 2]) - 0.5) * self.noise_scale
        for _ in range(1):
            noise_map.requires_grad = True
            vwt_x = self.vwt(x, noise_map)
            logits = self.get_logits(vwt_x)
            loss = self.get_loss(logits, label)
            grad = self.get_grad(loss, noise_map)
            noise_map = noise_map.detach() - self.rho * grad  
        return noise_map.detach()

    def forward(self, data, label, **kwargs):
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
            for _ in range(self.num_warping):

                # Obtain the data after warping
                adv = (data + delta).clone().detach()
                noise_map_hat = self.update_noise_map(adv, label)
                vwt_x = self.vwt(data+delta, noise_map_hat)
                
                # Obtain the output
                logits = self.get_logits(vwt_x)

                # Calculate the loss
                loss = self.get_loss(logits, label)

                # Calculate the gradients on x_idct
                grad = self.get_grad(loss, delta)
                grads += grad

            grads /= self.num_warping

            # Calculate the momentum
            momentum = self.get_momentum(grads, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()


def K_matrix(X, Y):
    eps = 1e-9

    D2 = torch.pow(X[:, :, None, :] - Y[:, None, :, :], 2).sum(-1)
    K = D2 * torch.log(D2 + eps)
    return K

def P_matrix(X):
    n, k = X.shape[:2]
    device = X.device

    P = torch.ones(n, k, 3, device=device)
    P[:, :, 1:] = X
    return P


class TPS_coeffs(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X, Y):

        n, k = X.shape[:2]  # n = 77, k =2
        device = X.device

        Z = torch.zeros(1, k + 3, 2, device=device)
        P = torch.ones(n, k, 3, device=device)
        L = torch.zeros(n, k + 3, k + 3, device=device) # [1, 80, 80]
        K = K_matrix(X, X)

        P[:, :, 1:] = X
        Z[:, :k, :] = Y
        L[:, :k, :k] = K
        L[:, :k, k:] = P
        L[:, k:, :k] = P.permute(0, 2, 1)

        # Q = torch.solve(Z, L)[0]
        Q = torch.linalg.solve(L, Z)
        return Q[:, :k], Q[:, k:]

class TPS(torch.nn.Module):
    def __init__(self, size: tuple = (256, 256), device=None):
        super().__init__()
        h, w = size
        self.size = size
        self.device = device
        self.tps = TPS_coeffs()
        grid = torch.ones(1, h, w, 2, device=device)
        grid[:, :, :, 0] = torch.linspace(-1, 1, w)
        grid[:, :, :, 1] = torch.linspace(-1, 1, h)[..., None]
        self.grid = grid.view(-1, h * w, 2)

    def forward(self, X, Y):
        """Override abstract function."""
        h, w = self.size
        W, A = self.tps(X, Y)  
        U = K_matrix(self.grid, X) 
        P = P_matrix(self.grid)
        grid = P @ A + U @ W
        return grid.view(-1, h, w, 2) 

def grid_points_2d(width, height, device):
    xx, yy = torch.meshgrid(
        [torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device)])
    return torch.stack([yy, xx], dim=-1).contiguous().view(-1, 2)

def noisy_grid(width, height, noise_map, device):
    """
    Make uniform grid points, and add noise except for edge points.
    """
    grid = grid_points_2d(width, height, device)
    mod = torch.zeros([height, width, 2], device=device)
    mod[1:height - 1, 1:width - 1, :] = noise_map
    return grid + mod.reshape(-1, 2)
