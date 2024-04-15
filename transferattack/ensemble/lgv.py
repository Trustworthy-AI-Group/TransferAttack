import torch.nn as nn

import numpy as np
from ..utils import *
from ..gradient.mifgsm import MIFGSM
from torchvision.models import resnet50
from random import shuffle, sample
import glob

class LGV(MIFGSM):
    """
    LGV Attack
    'LGV: Boosting Adversarial Example Transferability from Large Geometric Vicinity (ECCV 2022)'(https://arxiv.org/abs/2207.13129)

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
        python main.py --input_dir ./path/to/data --output_dir adv_data/lgv/ens --attack lgv --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/lgv/ens --eval
    """

    def __init__(self, model_name='resnet50', epsilon=16/255, alpha=1.6/255, epoch=10, decay=1.0, targeted=False, random_start=True, 
                 norm='linfty', loss='crossentropy', device=None, attack='LGV', checkpoint_path='./path/to/checkpoints/', **kwargs):
        self.checkpoint_path = checkpoint_path
        self.path_lgv_models = 'models/ImageNet/resnet50/cSGD/seed0'
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        

    def load_model(self, model_name):
        # download model checkpoints from: https://figshare.com/ndownloader/files/36698862
        if model_name == 'resnet50':
            model_path = os.path.join(self.checkpoint_path, self.path_lgv_models)
        else:
            raise ValueError('model:{} not supported, only supported "resnet50"'.format(model_name))

        if os.path.exists(model_path):
            pass
        else:
            raise ValueError("""Please download checkpoints from 'https://figshare.com/ndownloader/files/36698862', and put them into the path './path/to/checkpoints'.\\
                             More details: you can use the linux command: "!wget -O lgv_models.zip https://figshare.com/ndownloader/files/36698862" then "!unzip lgv_models.zip" at checkpoints path, it will be quickly!""")
        
        # LGV surrogate
        paths_models = glob.glob(f'{model_path}/*.pt')
        paths_models = sorted(paths_models)
        list_models = []
        for path in paths_models:
            model = resnet50()
            model.load_state_dict(torch.load(path)['state_dict'])
            model = wrap_model(model.eval().cuda())
            list_models.append(model)

        return LightEnsemble(list_models, order="shuffle", n_grad=1)


class LightEnsemble(nn.Module):
    def __init__(self, list_models, order="shuffle", n_grad=1):
        """
        Perform a single forward pass to one of the models when call forward()

        Arguments:
            list_models (list of nn.Module): list of LGV models.
            order (str): 'shuffle' draw a model without replacement (default), 'random' draw a model with replacement,
            None cycle in provided order.
            n_grad (int): number of models to ensemble in each forward pass (fused logits). Select models according to
            `order`. If equal to -1, use all models and order is ignored.
        """
        super(LightEnsemble, self).__init__()
        self.n_models = len(list_models)
        if self.n_models < 1:
            raise ValueError("Empty list of models")
        if not (n_grad > 0 or n_grad == -1):
            raise ValueError("n_grad should be strictly positive or equal to -1")
        if order == "shuffle":
            shuffle(list_models)
        elif order in [None, "random"]:
            pass
        else:
            raise ValueError("Not supported order")
        self.models = nn.ModuleList(list_models)
        self.order = order
        self.n_grad = n_grad
        self.f_count = 0

    def forward(self, x):
        if self.n_grad >= self.n_models or self.n_grad < 0:
            indexes = list(range(self.n_models))
        elif self.order == "random":
            indexes = sample(range(self.n_models), self.n_grad)
        else:
            indexes = [
                i % self.n_models
                for i in list(range(self.f_count, self.f_count + self.n_grad))
            ]
            self.f_count += self.n_grad
        if self.n_grad == 1:
            x = self.models[indexes[0]](x)
        else:
            # clone to make sure x is not changed by inplace methods
            x_list = [
                model(x.clone()) for i, model in enumerate(self.models) if i in indexes
            ]
            x = torch.stack(x_list)
            x = torch.mean(x, dim=0, keepdim=False)
        return x
