import torch
from ..utils import *
from ..gradient.mifgsm import MIFGSM
import copy
import torch.nn as nn

class InteractionLoss(nn.Module):
    def __init__(self, target=None, label=None):
        super(InteractionLoss, self).__init__()
        assert (target is not None) and (label is not None)
        self.target = target
        self.label = label

    def logits_interaction(self, outputs, leave_one_outputs,
                           only_add_one_outputs, zero_outputs):
        complete_score = outputs[:, self.target] - outputs[:, self.label]
        leave_one_out_score = (
            leave_one_outputs[:, self.target] -
            leave_one_outputs[:, self.label])
        only_add_one_score = (
            only_add_one_outputs[:, self.target] -
            only_add_one_outputs[:, self.label])
        zero_score = (
            zero_outputs[:, self.target] - zero_outputs[:, self.label])

        average_pairwise_interaction = (complete_score - leave_one_out_score -
                                        only_add_one_score +
                                        zero_score).mean()

        return average_pairwise_interaction

    def forward(self, outputs, leave_one_outputs, only_add_one_outputs,
                zero_outputs):
        return self.logits_interaction(outputs, leave_one_outputs,
                                       only_add_one_outputs, zero_outputs)

def sample_grids(sample_grid_num=16,
                 grid_scale=16,
                 img_size=224,
                 sample_times=8):
    grid_size = img_size // grid_scale
    sample = []
    for _ in range(sample_times):
        grids = []
        ids = np.random.randint(0, grid_scale**2, size=sample_grid_num)
        rows, cols = ids // grid_scale, ids % grid_scale
        for r, c in zip(rows, cols):
            grid_range = (slice(r * grid_size, (r + 1) * grid_size),
                          slice(c * grid_size, (c + 1) * grid_size))
            grids.append(grid_range)
        sample.append(grids)
    return sample

def sample_for_interaction(delta,
                           sample_grid_num,
                           grid_scale,
                           img_size,
                           times=16):
    samples = sample_grids(
        sample_grid_num=sample_grid_num,
        grid_scale=grid_scale,
        img_size=img_size,
        sample_times=times)
    only_add_one_mask = torch.zeros_like(delta).repeat(times, 1, 1, 1)
    for i in range(times):
        grids = samples[i]
        for grid in grids:
            only_add_one_mask[i:i + 1, :, grid[0], grid[1]] = 1
    leave_one_mask = 1 - only_add_one_mask
    only_add_one_perturbation = delta * only_add_one_mask
    leave_one_out_perturbation = delta * leave_one_mask

    return only_add_one_perturbation, leave_one_out_perturbation


def get_features(
    model,
    x,
    perturbation,
    leave_one_out_perturbation,
    only_add_one_perturbation,
):

    outputs = model(x + perturbation)
    leave_one_outputs = model(x + leave_one_out_perturbation)
    only_add_one_outputs = model(x + only_add_one_perturbation)
    zero_outputs = model(x)

    return (outputs, leave_one_outputs, only_add_one_outputs, zero_outputs)

class IR(MIFGSM):
    """
    IR Attack
    A Unified Approach to Interpreting and Boosting Adversarial Transferability (ICLR 2021)(https://arxiv.org//abs/2309.15696)

    Arguments:
        model (torch.nn.Module): the surrogate model for attack.
        epsilon (float): the perturbation budget.
        grid_num (int) : the numbers of divided image blocks
        grid_scale (int): the size of the grid
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., num_scale=10
    """

    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1.,
                 targeted=False, random_start=False,grid_scale=16,grid_num=32,sample_times=1,lam=1,
                 norm='linfty', loss='crossentropy', device=None, attack='ir', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.grid_scale = grid_scale
        #self.feature_layer = self.find_layer(feature_layer)
        self.grid_num = grid_num
        self.sample_times = sample_times
        self.lam = lam


    def forward(self, data, label, **kwargs):
        """
        The general attack procedure
        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)


        # Initialize adversarial perturbation
        delta = self.init_delta(data)
        momentum = 0
        for _ in range(self.epoch):
            # Obtain the output
            out = self.model(data+delta)
            outputs_c = copy.deepcopy(out.detach())
            outputs_c[:, label] = -np.inf
            other_max = outputs_c.max(1)[1]
            interaction_loss = InteractionLoss(
                target=other_max, label=label)
            average_pairwise_interaction =0
            for i in range(5):
                only_add_one_perturbation, leave_one_out_perturbation = \
                    sample_for_interaction(delta, self.grid_num,
                                           self.grid_scale, 224,
                                           self.sample_times)

                (outputs, leave_one_outputs, only_add_one_outputs,
                 zero_outputs) = get_features(self.model, data, delta,
                                              leave_one_out_perturbation,
                                              only_add_one_perturbation)
                average_pairwise_interaction += interaction_loss(
                    outputs, leave_one_outputs, only_add_one_outputs,
                    zero_outputs)
            # Calculate the loss
            loss1 = -self.loss(outputs,label)
            loss = loss1 - self.lam *average_pairwise_interaction/32
            self.model.zero_grad()
            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, -momentum, self.alpha)

        return delta.detach()