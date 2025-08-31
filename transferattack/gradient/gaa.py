import torch
import torch.nn as nn

from ..utils import *
from ..attack import Attack

class GAA(Attack):
    """
    Gradient Aggregation Attack (GAA) Method
    
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
        rho (float): hyperparameter for calculating t_hat
        lambda_param (float): hyperparameter for gradient aggregation
        xi (float): upper bound of random sampling in xi-ball
        N (int): number of randomly sampled examples

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/gaa/resnet50 --attack gaa --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/gaa/resnet50 --eval
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, 
                 random_start=False, norm='linfty', loss='crossentropy', device=None, attack='GAA',
                 rho=1.6/255, lambda_param=0.2, xi=0.1, N=20, **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.rho = rho
        self.lambda_param = lambda_param
        self.xi = 3.5 * epsilon
        self.N = N

    def forward(self, data, label, **kwargs):
        """
        The GAA attack procedure

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargeted, otherwise targeted labels
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]  # the second element is the targeted label tensor
        
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)
        
        # Initialize momentum
        momentum = torch.zeros_like(delta).to(self.device)
        
        # Calculate step size
        alpha = self.epsilon / self.epoch

        for t in range(self.epoch):
            # Initialize gradient aggregation
            g_bar = torch.zeros_like(delta).to(self.device)
            
            # Sample N random examples and aggregate gradients
            for i in range(self.N):
                # Randomly sample an example x' in B_epsilon(x)
                x_prime = self.sample_random_example(data, delta)
                
                # Calculate gradient g' = ∇_x L(x', y_t; θ)
                g_prime = self.calculate_gradient(x_prime, label)
                
                # Calculate t_hat = ρ * (g' / ||g'||_1) by Eq.(8)
                g_prime_norm = torch.norm(g_prime, p=1, dim=(1, 2, 3), keepdim=True)
                t_hat = self.rho * (g_prime / (g_prime_norm + 1e-8))
                
                # Get x_hat = x' + t_hat
                x_hat = x_prime + t_hat
                
                # Calculate gradient g_hat = ∇_x L(x_hat, y_t; θ)
                g_hat = self.calculate_gradient(x_hat, label)
                
                # Get the gradient g' = g_hat + (1-λ)g' + (1+λ)g_hat by Eq.(9)
                g_prime = g_hat + (1 - self.lambda_param) * g_prime + (1 + self.lambda_param) * g_hat
                
                # Accumulate gradients
                g_bar += g_prime
            
            # Average gradient
            g_bar = g_bar / self.N
            
            # Update the enhanced momentum g_t
            g_bar_norm = torch.norm(g_bar, p=1, dim=(1, 2, 3), keepdim=True)
            momentum = self.decay * momentum + (g_bar / (g_bar_norm + 1e-8))
            
            # Update x_adv by applying the gradient sign
            delta = self.update_delta(delta, data, momentum, alpha)

        return delta.detach()

    def sample_random_example(self, data, delta):
        """
        Randomly sample an example x' in B_epsilon(x)
        """
        # Create random perturbation within xi-ball on the same device as data
        if self.norm == 'linfty':
            random_pert = torch.rand_like(data, device=data.device).uniform_(-self.xi, self.xi)
        else:
            random_pert = torch.randn_like(data, device=data.device) * self.xi
            # Normalize to L2 norm
            pert_norm = torch.norm(random_pert.view(random_pert.size(0), -1), p=2, dim=1).view(-1, 1, 1, 1)
            random_pert = random_pert / (pert_norm + 1e-8) * self.xi
        
        # Add to current adversarial example
        x_prime = data + delta + random_pert
        
        # Clip to valid range
        x_prime = torch.clamp(x_prime, 0, 1)
        
        return x_prime

    def calculate_gradient(self, x, label):
        """
        Calculate gradient ∇_x L(x, y; θ)
        """
        x.requires_grad_(True)
        
        # Forward pass
        logits = self.model(x)
        
        # Calculate loss
        loss = self.get_loss(logits, label)
        
        # Calculate gradient
        grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
        
        return grad

    def update_delta(self, delta, data, grad, alpha):
        """
        Update adversarial perturbation
        """
        if self.norm == 'linfty':
            delta = torch.clamp(delta + alpha * grad.sign(), -self.epsilon, self.epsilon)
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
            scaled_grad = grad / (grad_norm + 1e-20)
            delta = (delta + scaled_grad * alpha).view(delta.size(0), -1).renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(delta)
        
        delta = clamp(delta, img_min-data, img_max-data)
        return delta.detach().requires_grad_(True)
