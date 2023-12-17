import torch
import torch.nn.functional as F

from ..gradient.mifgsm import MIFGSM
from ..utils import *


class DEM(MIFGSM):
    """
    DEM (Diversity-Ensemble)
    'Improving the Transferability of Adversarial Examples with Resized-Diverse-Inputs, Diversity-Ensemble and Region Fitting (ECCV 2020)'(https://arxiv.org/abs/2112.06011)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        resize_rate (float): the relative size of the resized image
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1, resize_rates=[1.14, 1.27, 1.4, 1.53, 1.66]

    Example script:
        python main.py --attack dem --output_dir adv_data/dem/resnet18

    Compared with DIM:
        1. Remove the diversity_prob in DIM, set diversity_prob=1 in DEM
        2. Use larger resize_rate in DEM (>1.1)
        3. Use ensemble logits in DEM
        4. Use epsilon to replace alpha when updating delta in DEM, clip has been implemented in attack (Region Fitting)
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., resize_rates=[1.14, 1.27, 1.4, 1.53, 1.66], targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None, attack='DEM', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        if not isinstance(resize_rates, list):
            raise Exception("Error! The resize rates should be a list.")
        for resize_rate in resize_rates:
            if resize_rate < 1:
                raise Exception("Error! The resize rate should be larger than 1.")
        self.resize_rates = resize_rates
        self.alpha = epsilon

    def transform(self, x, resize_rate, **kwargs):
        """
        Random transform the input images
        """

        img_size = x.shape[-1]
        img_resize = int(img_size * resize_rate)

        # resize the input image to random size
        rnd = torch.randint(low=min(img_size, img_resize), high=max(img_size, img_resize), size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)

        # randomly add padding
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        # resize the image back to img_size
        return F.interpolate(padded, size=[img_size, img_size], mode='bilinear', align_corners=False)


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

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):
            logits_ensemble=0
            #ensemble
            for resize_rate in self.resize_rates:
                # Obtain the output
                logits = self.get_logits(self.transform(data+delta, resize_rate, momentum=momentum))

                # ensemble the logits
                logits_ensemble += logits
            logits_ensemble /= len(self.resize_rates)

            # Calculate the loss
            loss = self.get_loss(logits_ensemble, label)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()
