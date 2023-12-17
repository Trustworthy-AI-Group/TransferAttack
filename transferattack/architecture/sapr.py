import torch

from ..gradient.mifgsm import MIFGSM
from ..utils import *


class SAPR(MIFGSM):
    """
    SAPR (Self-Attention Patches Restructure)
    'Improving the Transferability of Adversarial Examples with Restructure Embedded Patches'(https://arxiv.org/abs/2204.12680)

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
        prob (float): the probability of using self token permutation.

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., prob=0.15 for vit

    Example script:
        python main.py --attack sapr --output_dir adv_data/sapr/vit
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1.,  targeted=False, random_start=False,
                norm='linfty', loss='crossentropy', device=None, attack='SAPR', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)

        add_mix_token_hook(self.model, prob=0.15)

    def load_model(self, model_name):
        import timm
        print('load vit_base_patch16_224')
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True).eval().cuda()

        self.model = wrap_model(self.model)
        return self.model

def add_mix_token_hook(model, prob=0.5):
    for module in model.modules():
        from timm.models.vision_transformer import Attention
        if isinstance(module, Attention):
            module.register_forward_pre_hook(SelfTokenMixHook_pre(prob=prob))

class SelfTokenMixHook_pre:
    def __init__(self, prob=0.5):
        self.prob = prob # have self.prob probablity to permute the self token

    def __call__(self, module, input):
        import random
        if random.uniform(0, 1) > self.prob:
            return input

        bs, num_token ,_ = input[0].shape

        idx_token = torch.randperm(num_token-1) + 1
        idx_token = torch.cat([torch.tensor([0]), idx_token], dim=0)

        input_new = input[0][:,idx_token]
        return tuple([input_new])
