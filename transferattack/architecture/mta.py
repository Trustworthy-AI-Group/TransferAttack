# example bash: python main.py --input_dir ./path/to/data --output_dir adv_data/mta/resnet18 --attack mta --model=resnet_MTA
from ..utils import *
from ..gradient.mifgsm import MIFGSM


class MTA(MIFGSM):
    """
    Meta-Transfer Attack
    'Training Meta-Surrogate Model for Transferable Adversarial Attack (AAAI 2023)'(https://ojs.aaai.org/index.php/AAAI/article/view/26139)

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

    Example scripts:
        python main.py --input_dir ./path/to/data --output_dir adv_data/mta/resnet18 --attack mta --model=resnet_MTA
    """

    def __init__(self, model_name='resnet_MTA', epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, gamma=0.2, random_start=False, norm='linfty', loss='crossentropy', device=None, attack='MTA', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)

    def load_model(self, model_name):
        # download model: https://github.com/ydc123/Meta_Surrogate_Model/blob/main/README.md or https://github.com/ydc123/Dark_Surrogate_Model/blob/main/README.md
        if model_name == 'resnet_MTA':
            model_path = './path/to/checkpoints/resnet18_MTA_stage3.pth'
        else:
            raise ValueError('model:{} not supported, please set the model name to "resnet_MTA" and download it from https://github.com/ydc123/Meta_Surrogate_Model/blob/main/README.md'.format(model_name))

        model = models.__dict__['resnet18'](weights='DEFAULT').eval().cuda()
        info = torch.load(model_path, 'cpu')
        if 'state_dict' in info.keys(): # our models
            state_dict = info['state_dict']
        else: # Pretrained slightly robust model
            state_dict = info['model']
        cur_state_dict = model.state_dict()
        state_dict_keys = state_dict.keys()
        for key in cur_state_dict:
            if key in state_dict_keys:
                cur_state_dict[key].copy_(state_dict[key])
            elif key.replace('module.','') in state_dict_keys:
                cur_state_dict[key].copy_(state_dict[key.replace('module.','')])
            elif 'module.'+key in state_dict_keys:
                cur_state_dict[key].copy_(state_dict['module.'+key])
            elif 'module.attacker.model.'+key in state_dict_keys:
                cur_state_dict[key].copy_(state_dict['module.attacker.model.'+key])
        model.load_state_dict(cur_state_dict)

        return wrap_model(model.eval().cuda())
