import torch

from ...gradient.mifgsm import MIFGSM
from ...utils import *
from .yaila_utils import *


class YAILA(MIFGSM):
    """
    YAILA (Yet Another Intermediate-Level Attack)
    'Yet Another Intermediate-Level Attack (ECCV 2020)'(https://arxiv.org/abs/2008.08847)

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
        epsilon=0.03, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1.

    Example script:
        python main.py --attack=yaila --output_dir  adv_data/yaila/resnet50
    """

    def __init__(self, model_name, epsilon=0.03, alpha=1.6/255, epoch=10, decay=1., targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None, attack='YAILA', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)

    def load_model(self, model_name):
        print('=> load resnet50')
        model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        model.eval()
        model = model.cuda()
        return wrap_model(model)

    def forward(self, data, label, **kwargs):
        """
        The yaila attack procedure, following the official implementation https://github.com/qizhangli/ila-plus-plus.

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        batch_size = data.shape[0]

        # Step 1: Calculate the intermediate features mappings
        mid_layer_index = '3_1'
        bi, ui = mid_layer_index.split('_')
        mid_layer_index = '{}_{}'.format(bi, int(ui)-1)

        H, r = attack(False, None, data, label, self.device, self.epoch, 'tap', self.epsilon, self.model, mid_layer_index, batch_size=batch_size, lr=1./255)

        w = calculate_w(H=H, r=r, lam=1.0, normalize_H = True)

        # Step 2: Calculate the adversarial perturbation
        attacked_imgs = attack(True, torch.from_numpy(w), data, label, self.device, niters=50, baseline_method='tap', epsilon=self.epsilon, model=self.model, mid_layer_index=mid_layer_index, batch_size=batch_size, lr=1./255)

        # adversarial perturbation
        delta = attacked_imgs - data
        return delta