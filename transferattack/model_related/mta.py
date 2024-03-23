from ..gradient.mifgsm import MIFGSM
from ..utils import *


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

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/mta/resnet18 --attack mta --model=resnet_MTA

    Notes:
        Download the checkpoint ('resnet18_MTA_stage3.pth') from official repository: https://github.com/ydc123/Meta_Surrogate_Model, and put it in the path '/path/to/checkpoints/'.
        TransferAttack framework provides an alternative download link: https://huggingface.co/Trustworthy-AI-Group/TransferAttack/resolve/main/MTA.zip
    """

    def __init__(self, model_name='resnet_MTA', epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, random_start=False,
                 norm='linfty', loss='crossentropy', device=None, attack='MTA', checkpoint_path='/path/to/checkpoints/', **kwargs):
        self.checkpoint_path = checkpoint_path
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)

    def load_model(self, model_name):
        # download model: https://huggingface.co/Trustworthy-AI-Group/TransferAttack/resolve/main/MTA.zip
        if model_name == 'resnet_MTA':
            model_path = os.path.join(self.checkpoint_path, 'resnet18_MTA_stage3.pth')
        else:
            raise ValueError('model:{} not supported'.format(model_name))

        if os.path.exists(model_path):
            pass
        else:
            raise ValueError("Please download checkpoints, and put them into the path './path/to/checkpoints'.")

        model = models.__dict__['resnet18'](weights='DEFAULT').eval().cuda()
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
