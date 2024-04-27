import torch.nn as nn

from ..utils import *
from ..gradient.mifgsm import MIFGSM
from torchvision.models import resnet50
import copy
from collections import OrderedDict

class MBA(MIFGSM):
    """
    MBA Attack
    'Making Substitute Models More Bayesian Can Enhance Transferability of Adversarial Examples (ICLR 2023)'(https://arxiv.org/abs/2302.05086)

    Arguments:
        model (torch.nn.Module): the surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        targeted (bool): targeted/untargeted attack
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/mba/ens --attack mba --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/mba/ens --eval
    """

    def __init__(self, model_name='resnet50', epsilon=16/255, alpha=1.6/255, epoch=10, decay=1.0, targeted=False, random_start=True, 
                 norm='linfty', loss='crossentropy', device=None, attack='MBA', checkpoint_path='./path/to/checkpoints/', **kwargs):
        self.checkpoint_path = checkpoint_path
        self.source_model_path = 'resnet50_morebayesian_attack.pt'
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        

    def load_model(self, model_name):
        # download model checkpoints from: https://drive.google.com/drive/folders/1rOa4nFGsxrw-30_DJ77X_xqj__vhE_TN
        if model_name == 'resnet50':
            model_path = os.path.join(self.checkpoint_path, self.source_model_path)
        else:
            raise ValueError('model:{} not supported, only supported "resnet50"'.format(model_name))

        if os.path.exists(model_path):
            pass
        else:
            raise ValueError("""Please download checkpoints from 'https://drive.google.com/drive/folders/1rOa4nFGsxrw-30_DJ77X_xqj__vhE_TN', and put them into the path './path/to/checkpoints'.""")
        
        state_dict = torch.load(model_path)
        mean_model = build_model(state_dict["mean_state_dict"])
        sqmean_model = build_model(state_dict["sqmean_state_dict"])
        mean_model = nn.DataParallel(mean_model)

        model_list = []
        for model_ind in range(20):
            model_list.append(copy.deepcopy(mean_model))
            noise_dict = OrderedDict()
            for (name, param_mean), param_sqmean, param_cur in zip(mean_model.named_parameters(), sqmean_model.parameters(), model_list[-1].parameters()):
                var = torch.clamp(param_sqmean.data - param_mean.data**2, 1e-30)
                var = var + 0
                noise_dict[name] = var.sqrt() * torch.randn_like(param_mean, requires_grad=False)
            for (name, param_cur), (_, noise) in zip(model_list[-1].named_parameters(), noise_dict.items()):
                param_cur.data.add_(noise, alpha=1.5)
        return EnsembleModel([model for model in model_list])
    

def build_model(state_dict=False):
    model = resnet50()
    if "module" in list(state_dict.keys())[0]:
        model = nn.DataParallel(model)
        model.load_state_dict(state_dict)
        model = model.module
    else:
        model.load_state_dict(state_dict)
    return wrap_model(model.eval().cuda())
