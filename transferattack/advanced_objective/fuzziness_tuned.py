import torch
import torch.nn.functional as F
from ..utils import *
from ..attack import Attack

class Fuzziness_Tuned(Attack):
    """
    Fuzziness_Tuned Attack
    'Fuzziness-tuned: Improving the Transferability of Adversarial Examples'(https://arxiv.org/abs/2303.10078)

    Arguments:
        model (torch.nn.Module): the surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        K (float): the confidence scaling mechanism.
        T (float): the temperature scaling mechanism.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=1.6/255, epoch=10, decay=1, K=1.0, T=2.0.
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., K=0.8, T=2.0, targeted=False, random_start=False, 
                norm='linfty', loss='crossentropy', device=None, attack='Fuzziness_Tuned', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.K = K
        self.T = T
    
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
        data_size = label.shape[0]

        # Initialize adversarial perturbation
        delta = self.init_delta(data)
        index_label = [i for i in range(data_size)]
        momentum = 0
        for _ in range(self.epoch):
            # Obtain the output: fuzziness-tuned method with confidence scaling mechanism and temperature scaling mechanism.
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))
            logits[[index_label, label[index_label]]] *= self.K
            logits /= self.T
            
            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()