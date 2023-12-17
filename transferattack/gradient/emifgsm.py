import torch

from ..utils import *
from .mifgsm import MIFGSM

class EMIFGSM(MIFGSM):
    """
    EMI-FGSM Attack
    'Boosting Adversarial Transferability through Enhanced Momentum (BMVC 2021)'(https://arxiv.org/abs/2103.10609)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_sample (int): the number of samples to enhance the momentum.
        radius (float): the relative radius for sampling.
        sample_method (str): the sampling method (linear/uniform/gaussian).
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., num_sample=11, radius=7, sample_method='linear'
    """
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., num_sample=11, radius=7, sample_method='linear', 
                targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None, attack='EMI-FGSM', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.num_sample = num_sample
        self.radius = radius
        self.sample_method = sample_method.lower()

    def get_factors(self):
        """
        Generate the sampling factors
        """
        if self.sample_method == 'linear':
            return np.linspace(-self.radius, self.radius, num=self.num_sample)
        elif self.sample_method == 'uniform':
            return np.random.uniform(-self.radius, self.radius, size=self.num_sample)
        elif self.sample_method == 'gaussian':
            return np.clip(np.random.normal(size=self.num_sample)/3, -1, 1)*self.radius
        else:
            raise Exception('Unsupported sampling method {}!'.format(self.sample_method)) 

    def transform(self, x, grad, **kwargs):
        """
        Admix the input for Admix Attack
        """
        factors = np.linspace(-self.radius, self.radius, num=self.num_sample)
        return torch.concat([x+factor*self.alpha*grad for factor in factors])

    def get_loss(self, logits, label):
        """
        Calculate the loss
        """
        return -self.loss(logits, label.repeat(self.num_sample)) if self.targeted else self.loss(logits, label.repeat(self.num_sample))
    
    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

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

        bar_grad = 0
        for _ in range(self.epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta, grad=bar_grad))

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            bar_grad = grad / (grad.abs().mean(dim=(1,2,3), keepdim=True))

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()
