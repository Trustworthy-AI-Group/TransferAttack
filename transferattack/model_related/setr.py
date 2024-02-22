from timm.models import create_model

from ..gradient.mifgsm import MIFGSM
from ..utils import *


class SETR(MIFGSM):
    """
    SETR (Self-Ensembling & Token Refinement)
    'On Improving Adversarial Transferability of Vision Transformers (ICLR 2022)'(https://arxiv.org/abs/2106.04169)

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
        attack (str): the name of attack.

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1.

    Example script:
        python main.py --attack=setr --model=tiny --output_dir adv_data/setr/tiny
    """

    def __init__(self, model_name='tiny', epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, gamma=0.2, random_start=False, norm='linfty', loss='crossentropy', device=None, attack='SETR',trm=True, **kwargs):
        self.trm = trm
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)

    def load_model(self, model_name):
        if model_name not in ['tiny', 'small', 'base']:
            raise ValueError(f'Model:{model_name} should be one of tiny, small, base')

        if self.trm: # SE + TRM, load pretrained model from retrained model before with pth saved in github
            model_name_detail = f"{model_name}_patch16_224_hierarchical"   # "tiny_patch16_224_hierarchical"
            model_path = f'https://github.com/Muzammal-Naseer/ATViT/releases/download/v0/deit_{model_name}_trm.pth'
        else: # SE only, load pretrained model from timm
            model_name_detail = f"deit_{model_name}_patch16_224"  # 'deit_tiny_patch16_224'

        # import the module only when setr attack is called
        from .setr_networks import tiny_patch16_224_hierarchical, small_patch16_224_hierarchical, base_patch16_224_hierarchical

        src_model, src_mean, src_std = get_model(model_name_detail)
        if model_path is not None:
            if model_path.startswith("https://"):
                src_checkpoint = torch.hub.load_state_dict_from_url(model_path, map_location='cpu')
            else:
                src_checkpoint = torch.load(model_path, map_location='cpu')

            # print(src_checkpoint.keys())
            src_model.load_state_dict(src_checkpoint['state_dict'])

        return wrap_model(src_model.eval().cuda())

    def get_loss(self, logits, label):
        """
        The loss calculation, which is overrideen because of emsemble-loss.
        """
        if isinstance(logits, list) :
            loss = 0
            for logits_one in logits:
                loss += -self.loss(logits_one, label) if self.targeted else self.loss(logits_one, label)
        else:
            loss = -self.loss(logits, label) if self.targeted else self.loss(logits, label)

        return loss

def get_model(model_name):
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    # get the source model
    if model_name in model_names:
        model = models.__dict__[model_name](pretrained=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'deit' in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'hierarchical' in model_name or "ensemble" in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'vit' in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif 'T2t' in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'tnt' in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif 'swin' in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise NotImplementedError(f"Please provide correct model names: {model_names}")

    return model, mean, std