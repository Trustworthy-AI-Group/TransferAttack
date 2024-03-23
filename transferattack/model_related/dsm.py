import os

from ..gradient.mifgsm import MIFGSM
from ..utils import *


class DSM(MIFGSM):
    """
    DSM (Dark Surrogate Model)
    'Boosting the Adversarial Transferability of Surrogate Model with Dark Knowledge'(https://arxiv.org/abs/2206.08316)

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
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1.,
        Optimizer: MI-FGSM from [FGSM, MI-FGSM, M-DI2-FGSM]

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/dsm/resnet18 --attack dsm --model=SR_resnet18_cutmix

    Notes:
        Download the checkpoint ('SD_resnet18_cutmix.pth.tar') from official repository: https://github.com/ydc123/Dark_Surrogate_Model, and put it in the path '/path/to/checkpoints/'.
        TransferAttack framework provides an alternative download link: https://huggingface.co/Trustworthy-AI-Group/TransferAttack/resolve/main/DSM.zip
    """

    def __init__(self, model='SR_resnet18_cutmix', epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, random_start=False,
                 norm='linfty', loss='crossentropy', device=None, attack='DSM', checkpoint_path='/path/to/checkpoints/', **kwargs):
        model = 'SR_0.1_resnet18_cutmix'
        self.checkpoint_path = checkpoint_path
        super().__init__(model, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)

    def load_model(self, model_name):
        # download model: https://huggingface.co/Trustworthy-AI-Group/TransferAttack/resolve/main/DSM.zip
        if model_name == 'resnet18_CE':
            model_path = os.path.join(self.checkpoint_path, 'resnet18_CE.pth.tar')
        elif model_name == 'SD_resnet18_cutmix':
            model_path = os.path.join(self.checkpoint_path, 'SD_resnet18_cutmix.pth.tar')
        elif model_name == 'SR_0.1_resnet18_cutmix':
            model_path = os.path.join(self.checkpoint_path, 'SD_resnet18_cutmix.pth.tar')
        else:
            raise ValueError('model:{} not supported'.format(model_name))

        if os.path.exists(model_path):
            pass
        else:
            raise ValueError("Please download checkpoints, and put them into the path './path/to/checkpoints'.")

        model = models.__dict__['resnet18'](pretrained=True).eval().cuda()
        info = torch.load(model_path, 'cpu')
        if 'state_dict' in info.keys():  # our models
            state_dict = info['state_dict']
        else:  # Pretrained slightly robust model
            state_dict = info['model']
        cur_state_dict = model.state_dict()
        state_dict_keys = state_dict.keys()
        for key in cur_state_dict:
            if key in state_dict_keys:
                cur_state_dict[key].copy_(state_dict[key])
            elif key.replace('module.', '') in state_dict_keys:
                cur_state_dict[key].copy_(state_dict[key.replace('module.', '')])
            elif 'module.' + key in state_dict_keys:
                cur_state_dict[key].copy_(state_dict['module.' + key])
            elif 'module.attacker.model.' + key in state_dict_keys:
                cur_state_dict[key].copy_(state_dict['module.attacker.model.' + key])
        model.load_state_dict(cur_state_dict)

        return wrap_model(model.eval().cuda())
