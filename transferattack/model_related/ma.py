import torch
import torch.nn.functional as F
import torchvision.models as models
from ..utils import *
from ..gradient.mifgsm import MIFGSM


class MA(MIFGSM):
    """
    MA Attack
    Improving Adversarial Transferability via Model Alignment (ECCV 2024) (https://eccv2024.ecva.net//virtual/2024/poster/1952)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_ens (int): the number of gradients to aggregate
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1.

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/ma/resnet50 --attack ma --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/ma/resnet50 --eval
    """

    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1.,
                 targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None, attack='MA', eta=28.0, gamma=2.0, checkpoint_path='./path/to/checkpoints/', **kwargs):
        self.checkpoint_path = checkpoint_path
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.eta = eta
        self.gamma = gamma

    def load_model(self, model_name):
        model = models.resnet50(pretrained=False).cuda().eval()
        model_path = os.path.join(self.checkpoint_path, 'source_model.pth')
        if os.path.exists(model_path):
            pass
        else:
            raise ValueError("Please download checkpoints from https://drive.google.com/file/d/1HqQFhSgBJvJZ2RVfxCP2WCi3-xck-bHx/view?usp=sharing, and put them into the path './path/to/checkpoints'.")
        
        model.load_state_dict(torch.load(model_path))
        return wrap_model(model.eval().cuda())
