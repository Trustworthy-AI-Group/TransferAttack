import os
import pretrainedmodels
from ..gradient.mifgsm import MIFGSM
from ..utils import *
import torch.nn as nn

class DRA(MIFGSM):
    """
    DRA Attack
    'Towards Understanding and Boosting Adversarial Transferability from a Distribution Perspective (TIP 2022)'(https://ieeexplore.ieee.org/document/9917370?denied=)

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

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/dra/resnet50 --attack dra --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/dra/resnet50 --eval

    Notes:
        Download the checkpoint ('DRA_resnet50.pth') from official repository: https://github.com/alibaba/easyrobust/tree/main/examples/attacks/dra, and put it in the path '/path/to/checkpoints/'.
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, random_start=False,
                 norm='linfty', loss='crossentropy', device=None, attack='DRA', checkpoint_path='./path/to/checkpoints/', **kwargs):
        self.checkpoint_path = checkpoint_path
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)


    def load_model(self, model_name):
        weight_path = os.path.join(self.checkpoint_path, 'DRA_resnet50.pth')

        if not os.path.exists(weight_path):
            raise ValueError("Please download the checkpoint of the 'DRA_resnet50.pth' from 'https://drive.google.com/drive/folders/1JAkrWOEU4qLUEMy0X5LcSUUJMNTOoyE0?usp=sharing', and put it into the path '{}'.".format(self.checkpoint_path))
        
        net = pretrainedmodels.__dict__[model_name](num_classes=1000,pretrained='imagenet') 
        net = torch.nn.DataParallel(net).cuda()

        ckpt = torch.load(weight_path)
        if "model_state_dict" in ckpt:
            net.load_state_dict(ckpt["model_state_dict"])
            if "accuracy" in ckpt:
                print("The loaded model has Validation accuracy of: {:.2f} %\n".format(ckpt["accuracy"]))
        else:
            net.load_state_dict(ckpt)  

        model = models.__dict__[model_name](weights="DEFAULT")

        model = nn.DataParallel(model).cuda()
        model_dict = model.state_dict()
        pre_dict = net.state_dict()
        state_dict = {k:v for k,v in pre_dict.items() if k in model_dict.keys()}
        state_dict['module.fc.weight'] = pre_dict['module.last_linear.weight']
        state_dict['module.fc.bias'] = pre_dict['module.last_linear.bias']
        print("Loaded pretrained weight. Len :", len(pre_dict.keys()), len(state_dict.keys()))
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

        return wrap_model(model.eval().cuda())
