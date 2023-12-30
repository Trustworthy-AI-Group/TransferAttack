import torch

from ..utils import *
from ..attack import Attack

class RAP(Attack):
    """
    RAP Attack
    'Boosting the Transferability of Adversarial Attacks with Reverse Adversarial Perturbation (NeurIPS 2022)'(https://arxiv.org/abs/2210.05968)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        transpoint (int): the step start to use RAP attack.
            - transpoint 400: baseline method
            - transpoint 0: baseline + RAP
            - transpoint 100: baselien +RAP-LS
        epsilon_n (float): the perturbation budget for inner maximizaiton
        alpha_n (float): the step size for inner maximization
        adv_steps (int): the number of iterations for inner maximization
        targeted (bool): targeted/untargeted attack
        random_start (bool): whether using random initialization for delta and n_rap
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        Untargeted Attack:
            epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, transpoint=100, epsilon_n=16/255, alpha_n=2/255, adv_steps=8
        Targeted Attack:
            epsilon=, alpha=, epoch, transpoint, epsilon_n, alpha_n, adv_step=
    Example script:
        python main.py --attack rap --output_dir adv_data/rap/resnet18
    """

    def __init__(self, model_name, epsilon=16/255, alpha=2/255, epoch=400, transpoint=100, epsilon_n=16/255, alpha_n=2/255, adv_steps=8,
                targeted=True, random_start=False,
                norm='linfty', loss='crossentropy', device=None, attack='RAP', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = 1.
        self.alpha_n = alpha_n
        self.adv_steps = adv_steps
        self.transpoint = transpoint
        self.epsilon_n = epsilon_n

    def get_loss(self, logits, label):
        if not self.targeted:
            real = logits.gather(1, label.unsqueeze(1)).squeeze(1)
            logit_dists = -1 * real
            loss = logit_dists.mean()
        else:
            real = logits.gather(1, label.unsqueeze(1)).squeeze(1)
            loss = real.mean()
        return loss

    def update_n_rap(self, delta, data, grad, alpha, **kwargs):
        # if self.norm == 'linfty':
        delta = torch.clamp(delta + alpha * grad.sign(), -self.epsilon_n, self.epsilon_n)
        # else:
        #     grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
        #     scaled_grad = grad / (grad_norm + 1e-20)
        #     delta = (delta + scaled_grad * alpha).view(delta.size(0), -1).renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(delta)
        delta = clamp(delta, img_min-data, img_max-data)
        return delta

    def init_n_rap(self, data, random_start, **kwargs):
        delta = torch.zeros_like(data).to(self.device)
        if random_start:
            # if self.norm == 'linfty':
            delta.uniform_(-self.epsilon_n, self.epsilon_n)
            # else:
            #     delta.normal_(-self.epsilon, self.epsilon)
            #     d_flat = delta.view(delta.size(0), -1)
            #     n = d_flat.norm(p=2, dim=10).view(delta.size(0), 1, 1, 1)
            #     r = torch.zeros_like(data).uniform_(0,1).to(self.device)
            #     delta *= r/n*self.epsilon
            delta = clamp(delta, img_min-data, img_max-data)
        delta.detach().requires_grad = True
        return delta

    def get_n_rap(self, data, label):
        n_rap = self.init_n_rap(data, random_start=True)

        for _ in range(self.adv_steps):
            # Obtain the output
            logits = self.get_logits(self.transform(data+n_rap))

            # Calculate the loss
            loss = -self.get_loss(logits, label)

            # Calculate the gradients
            grad = self.get_grad(loss, n_rap)

            # Update the n_rap
            n_rap = self.update_n_rap(n_rap, data, grad, self.alpha_n)

        return n_rap.detach()

    def forward(self, data, label, **kwargs):
        """
        The RAP attack procedure

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
        n_rap = torch.zeros_like(data).to(self.device)
        for iter in range(self.epoch):
            # Late start
            if iter >= self.transpoint:
                n_rap = self.get_n_rap(data+delta, label)

            # Obtain the output
            logits = self.get_logits(self.transform(data+delta+n_rap, momentum=momentum))

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()