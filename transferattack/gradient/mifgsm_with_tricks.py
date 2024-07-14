import torch

from ..utils import *
from ..attack import Attack

"""
Results of the MI-FGSM with tricks on this library:
| 100.0 | 42.6 | 46.0 | 74.3 | 16.7 | 24.5 | 34.3 | 41.7 | MI-FGSM
| 100.0 | 50.9 | 54.6 | 82.6 | 19.3 | 27.3 | 40.9 | 47.1 | RGI-MIFGSM
|  99.8 | 44.8 | 48.4 | 73.7 | 20.5 | 26.6 | 36.5 | 45.0 | Dual MI-FGSM (FGSM)
|  99.8 | 51.5 | 55.5 | 79.7 | 24.6 | 32.8 | 43.1 | 48.8 | Ensemble Dual MI-FGSM (FGSM)
| 100.0 | 51.4 | 55.2 | 81.3 | 23.5 | 31.3 | 42.1 | 48.3 | Ensemble Dual MI-FGSM (I-FGSM)
"""




class RGMIFGSM(Attack):
    """
    RGI-FGSM Attack
    'Bag of tricks to boost the adversarial transferability'(https://arxiv.org/abs/2401.08734)

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
        device (torch.device): the device for data. If it is None, the device would be same as model.
        pre_epoch (int): the pre-convergence iterations.
        s (int): the global search factor.
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., pre_epoch=5, s=10

    Example script:
        python main.py --attack gifgsm --output_dir adv_data/gifgsm/resnet18
        python main.py --attack gifgsm --output_dir adv_data/gifgsm/resnet18 --eval
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, random_start=False, 
                norm='linfty', loss='crossentropy', device=None, attack='GI-FGSM', pre_epoch=5, s=10, num_directions=5, **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device, **kwargs)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.pre_epoch = pre_epoch
        self.s = s
        self.num_directions = num_directions

    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        momentum = 0.
        self.random_start = True
        delta = self.init_delta(data).to(self.device)
        for di in range(self.num_directions):
            direction_momentum = 0.
            delta = self.init_delta(data).to(self.device)
            for _ in range(self.pre_epoch):
                # Obtain the output
                logits = self.get_logits(self.transform(data+delta, momentum=momentum))
                # Calculate the loss
                loss = self.get_loss(logits, label)
                # Calculate the gradients
                grad = self.get_grad(loss, delta)
                # Calculate the momentum
                momentum = self.get_momentum(grad, momentum)
                # Update adversarial perturbation
                delta = self.update_delta(delta, data, momentum, self.alpha*self.s)
            momentum += direction_momentum
        momentum /= self.num_directions
        self.random_start = False
        delta = self.init_delta(data).to(self.device)
        for _ in range(self.epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))
            # Calculate the loss
            loss = self.get_loss(logits, label)
            # Calculate the gradients
            grad = self.get_grad(loss, delta)
            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)
            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)
        # exit()
        return delta.detach()


class DualMIFGSM(Attack):
    """
    MI-FGSM Attack with dual example
    'Bag of tricks to boost the adversarial transferability'(https://arxiv.org/abs/2401.08734)

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
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1.

    Example script:
        python main.py --attack mifgsm --output_dir adv_data/mifgsm/resnet18
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, random_start=False,
                norm='linfty', loss='crossentropy', device=None, attack='DualMIFGSM', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
    
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
        delta_dual = delta.clone().detach().to(self.device)

        momentum = 0.
        momentum_dual = 0.

        for _ in range(self.epoch):
            # Obtain the output
            self.random_start = True
            delta = self.init_delta(data).to(self.device)
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))
            self.random_start = False


            # Calculate the loss  
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)
            momentum_dual = self.get_momentum(grad, momentum_dual,decay=self.decay)
            # Update adversarial perturbation
            delta = self.update_delta(delta, data, grad, self.alpha)
            delta_dual = self.update_delta(delta_dual, data, momentum_dual, self.alpha)


        return delta_dual.detach()


class Ens_FGSM_MIFGSM(Attack):
    """
    MI-FGSM Attack with the ensemble of dual example
    'Bag of tricks to boost the adversarial transferability'(https://arxiv.org/abs/2401.08734)

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
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1.

    Example script:
        python main.py --attack mifgsm --output_dir adv_data/mifgsm/resnet18
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, random_start=False,
                norm='linfty', loss='crossentropy', device=None, attack='Ens_DualMIFGSM', num_d=5, **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.num_directions = num_d
    
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
        delta_dual = delta.clone().detach().to(self.device)
        # self.ai = self.normalize(self.op, self.epoch)
        momentum = 0.
        momentum_dual = 0.

        for _ in range(self.epoch):
            # Obtain the output
            grad_c = 0.
            for nd in range(self.num_directions):
                self.random_start = True
                delta = self.init_delta(data).to(self.device)
                logits = self.get_logits(self.transform(data+delta, momentum=momentum))
                self.random_start = False


                # Calculate the loss  
                loss = self.get_loss(logits, label)

                # Calculate the gradients
                grad = self.get_grad(loss, delta)
                grad_c += grad
            grad_c /= self.num_directions
            grad = grad_c

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)
            momentum_dual = self.get_momentum(grad, momentum_dual,decay=self.decay)
            # Update adversarial perturbation
            delta_dual = self.update_delta(delta_dual, data, momentum_dual, self.alpha)


        return delta_dual.detach()




