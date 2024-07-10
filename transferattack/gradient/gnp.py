import torch
from ..utils import *
from ..attack import Attack

class GNP(Attack):
    """
    GNP (Gradient Norm Penalty)
    'GNP Attack: Transferable Adversarial Examples via Gradient Norm Penalty (ICIP 2023)' (https://ieeexplore.ieee.org/abstract/document/10223158)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        r (float): the step length.
        beta (float): the regularization coefficient.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1, r=0.01, beta=0.8.
    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/gnp/resnet18 --attack gnp --model=resnet18
        python main.py --input_dir ./path/to/data --output_dir adv_data/gnp/resnet18 --eval
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., r=0.01, beta=0.8, targeted=False, random_start=False,
                norm='linfty', loss='crossentropy', device=None, attack='GNP', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.r = r
        self.beta = beta
    

    def forward(self, data, label, **kwargs):
        """
        The GNP attack procedure

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

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta))

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            g1 = self.get_grad(loss, delta)

            # Calculate the neighborhood point
            g_p = g1 / (g1.abs().mean(dim=(1,2,3), keepdim=True))

            # self.model.zero_grad()

            # Obtain the anticipatory output
            logits = self.get_logits(self.transform(data+delta+self.r*g_p))

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            g2 = self.get_grad(loss, delta)

            gt = (1 + self.beta) * g1 + self.beta * g2

            # Calculate the momentum
            momentum = self.get_momentum(gt, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()
