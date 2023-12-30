import torch

from ..utils import *
from ..attack import Attack

class GRA(Attack):
    """
    GRA Attack
    'Boosting Adversarial Transferability via Gradient Relevance Attack (ICCV 2023)'(https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_Boosting_Adversarial_Transferability_via_Gradient_Relevance_Attack_ICCV_2023_paper.pdf)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        beta (float): the upper bound factor of neighborhood.
        num_neighbor (int): the number of samples for estimating the gradient variance.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, beta=3.5, num_neighbor=20, epoch=10, decay=1.

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/gra/resnet18 --attack gra --model=resnet18
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, beta=3.5, num_neighbor=20, epoch=10, decay=1., targeted=False,
                random_start=False, norm='linfty', loss='crossentropy', device=None, attack='GRA', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.radius = beta * epsilon
        self.epoch = epoch
        self.decay = decay
        self.num_neighbor = num_neighbor

    def get_average_gradient(self, data, delta, label, momentum, **kwargs):
        """
        Calculate the average gradient of the samples
        """
        grad = 0
        for _ in range(self.num_neighbor):
            # Obtain the output
            # This is inconsistent for transform!
            logits = self.get_logits(self.transform(data+delta+torch.zeros_like(delta).uniform_(-self.radius, self.radius).to(self.device), momentum=momentum))

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad += self.get_grad(loss, delta)

        return grad / self.num_neighbor

    def get_cosine_similarity(self, cur_grad, sam_grad, **kwargs):
        """
        Calculate cosine similarity to find the score
        """

        cur_grad = cur_grad.view(cur_grad.size(0), -1)
        sam_grad = sam_grad.view(sam_grad.size(0), -1)

        cos_sim = torch.sum(cur_grad * sam_grad, dim=1) / (
                    torch.sqrt(torch.sum(cur_grad ** 2, dim=1)) * torch.sqrt(torch.sum(sam_grad ** 2, dim=1)))
        cos_sim = cos_sim.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return cos_sim

    def get_decay_indicator(self, M, delta, cur_noise, last_noise, eta, **kwargs):
        """
        Define the decay indicator
        """
    
        if isinstance(last_noise, int):
            last_noise = torch.full(cur_noise.shape, last_noise)
        else:
            last_noise = last_noise

        if torch.cuda.is_available():
            last_noise = last_noise.cuda()

        last = last_noise.sign()
        cur = cur_noise.sign()
        eq_m = (last == cur).float()
        di_m = torch.ones_like(delta) - eq_m
        M = M * (eq_m + di_m * eta)

        return M


    def forward(self, data, label, **kwargs):
        """
        The attack procedure for GRA

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

        # Initialize the attenuation factor for decay indicator
        eta = 0.94

        # Initialize the decay indicator
        M = torch.full_like(delta, 1 / eta)

        momentum = 0
        for _ in range(self.epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the current gradients
            grad = self.get_grad(loss, delta)

            # Calculate the average gradients
            samgrad = self.get_average_gradient(data, delta, label, momentum)

            # Calculate the cosine similarity
            s = self.get_cosine_similarity(grad, samgrad)

            # Calculate the global weighted gradient
            current_grad = s * grad + (1 - s) * samgrad

            # Save the previous perturbation
            last_momentum = momentum

            # Calculate the momentum
            momentum = self.get_momentum(current_grad, momentum)

            # Update decay indicator
            M = self.get_decay_indicator(M, delta, momentum, last_momentum, eta)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, M * self.alpha)

        return delta.detach()
