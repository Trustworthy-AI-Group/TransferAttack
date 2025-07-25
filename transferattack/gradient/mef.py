import torch
from ..utils import *
from ..attack import Attack
import torch.nn as nn

class MEF(Attack):
    """
    MEF (Maximin Expected Flatness)
    'Boosting Adversarial Transferability with Low-Cost Optimization via Maximin Expected Flatness' (https://arxiv.org/abs/2405.16181)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        num_neighbor (int): the number of randomly sampled samples.
        gamma (float): the upper bound of random sampling.
        kesai (float): the upper bound of sub-regions around sample points.
        epoch (int): the number of iterations.
        inner_decay (float): the decay factor for inner momentum calculation.
        decay (float): the decay factor for outer momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model.
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, num_neighbor=20, gamma=2., kesai=0.15, epoch=20, inner_decay=0.9, decay=0.5.

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/mef/resnet50 --attack mef --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/mef/resnet50 --eval
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, num_neighbor=20, gamma=2., kesai=0.15, lamada=0.16/255, epoch=20, inner_decay=0.9, decay=0.5, targeted=False, 
                random_start=False, norm='linfty', loss='crossentropy_no_reduction', device=None, attack='MEF', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.kesai = kesai * epsilon
        self.gamma = gamma * epsilon
        self.lamada = lamada
        self.epoch = epoch
        self.inner_decay = inner_decay
        self.decay = decay
        self.num_neighbor = num_neighbor
    
    def loss_function(self, loss):
        """
        Get the loss function
        """
        if loss == 'crossentropy':
            return nn.CrossEntropyLoss()
        elif loss == 'crossentropy_no_reduction':
            return nn.CrossEntropyLoss(reduction='none')
        else:
            raise Exception("Unsupported loss {}".format(loss))
            
    def get_conditional_sampled_points(self, delta, grad_pgia):
        """
        Neighborhood conditional sampling
        """
        sample_delta = self.transform(delta + torch.zeros_like(grad_pgia).uniform_(-self.gamma, self.gamma))
        sample_delta = self.transform(sample_delta + self.kesai * grad_pgia)
        return sample_delta
        
    def get_points_gradient(self, data, delta, label, **kwargs):
        """
        Calculate the gradients of the sampled points
        """
        b, c, h, w = data.shape
        loss_list = torch.zeros([self.num_neighbor, b]).to(self.device)
        grad_list = torch.zeros([self.num_neighbor, b, c, h, w]).to(self.device)
        for i in range(self.num_neighbor):

            # Get the conditional sampled points x_min
            x_min = self.transform(data + delta[i])

            # Calculate the output of the x_min
            logits = self.get_logits(x_min)

            # Calculate the loss of the x_min
            loss_list[i] = self.get_loss(logits, label)

            # Calculate the gradient of the x_min
            grad_list[i] = self.get_grad(loss_list[i].mean(), x_min)
        
        # Calculate the gradient of the loss function
        grad = (1/self.num_neighbor) * grad_list

        return grad

    def forward(self, data, label, **kwargs):
        """
        The attack procedure for MEF

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

        momentum = 0
        b, c, h, w = data.shape
        grad_pgia = torch.zeros([self.num_neighbor, b, c, h, w]).to(self.device)
        for _ in range(self.epoch):

            # Neighborhood conditional sampling
            sample_delta = self.get_conditional_sampled_points(delta, grad_pgia)

            # Calculate the gradient of each point
            gradient = self.get_points_gradient(data, sample_delta, label)

            # Update gradient for previous gradient inversion approximation
            grad_pgia = ((gradient / torch.mean(torch.abs(gradient), (2, 3, 4), keepdim=True)).detach() - self.inner_decay * grad_pgia)

            # Calculate the momentum
            momentum = self.get_momentum(gradient.sum(0), momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()