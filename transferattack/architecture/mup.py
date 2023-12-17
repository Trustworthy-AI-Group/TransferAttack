import torch
import torch.nn as nn
from ..utils import *
from ..gradient.mifgsm import MIFGSM


class MUP(MIFGSM):
    """
    MUP Attack
    'Generating Adversarial Examples with Better Transferability via Masking Unimportant Parameters of Surrogate Model (IJCNN 2023)'(https://arxiv.org/abs/2304.06908)

    Arguments:
        model (torch.nn.Module): the surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        mask_ratio (float): the masking ratio.
        mask_type (str): mask type for calculating the parameter importance score.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=2/255, epoch=10, decay=1.0

    Example scripts:
        python main.py --attack=mup --input_dir=./data --output_dir=./results/mup/resnet18 --model resnet18 --batchsize 1
        python main.py --attack=mup --input_dir=./data --output_dir=./results/mup/resnet18 -e

    NOTE:
        1) --batchsize=1 is necessary for MUP attack since gradient and taylor score are calculated w.r.t. each sample.
        2) resnet18 is not used in the paper. According to Fig. 3 in the paper, the mask_ratio is set to 0.15 for most models to achieve the best performance.
    """

    def __init__(self, **kwargs):
        kwargs['attack'] = 'MUP'
        kwargs['alpha'] = 2/255
        super().__init__(**kwargs)

        self.mask_ratio = 0.15
        self.mask_type = 'taylor'

    def forward(self, data, label, **kwargs):
        """
        Override the forward function of MIFGSM to add pruning.

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
            # First forward-backward step: get pruning mask and prune the model
            self.model = self.restore_weight(self.model)
            logits_1 = self.get_logits(self.transform(data+delta, momentum=momentum))
            loss_1 = self.get_loss(logits_1, label)
            loss_1.backward(retain_graph=True)
            self.model = self.prune(self.model, p=self.mask_ratio, type=self.mask_type)

            # Second forward-backward step: get gradient w.r.t. data on pruned model
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))
            loss = self.get_loss(logits, label)
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()

    def prune(self, model, p=0.1, type='taylor'):
        """
        Prune the model parameters with the smallest 'p' weights

        Arguments:
            model (torch.nn.Module): the surrogate model to be pruned.
            p (float): the percentage of weights to be pruned.
            type (str): the type of pruning, 'taylor', 'l1', 'grad'
        """
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                if type == 'taylor':
                    scores = torch.abs(module.weight.data * module.weight.grad)
                elif type == 'l1':
                    scores = torch.norm(module.weight.data, p=1, dim=(2, 3))
                elif type == 'grad':
                    scores = torch.abs(module.weight.grad)
                else:
                    raise ValueError('Type {} not supported'.format(type))

                idx = int(scores.numel() * p)
                values, _ = scores.view(-1).sort()
                threshold = values[idx]
                mask = (scores > threshold).float().cuda()
                self.prune_from_mask(module, mask)
        return model

    def prune_from_mask(self, module, mask):
        """
        Inplace prune the module with mask, and save the original weights in module.weight_orig
        """
        module.weight_orig = module.weight.clone()  # must use .clone()
        module.weight = nn.Parameter(module.weight * mask)

    def restore_weight(self, model):
        """
        Restore the original weights in model
        """
        for module in model.modules():
            if isinstance(module, nn.Conv2d) and hasattr(module, 'weight_orig'):
                module.weight = nn.Parameter(
                    module.weight_orig.clone())  # must use .clone()
                del module.weight_orig
        return model
