import torch
import torch.nn as nn
from ..utils import *
from ..gradient.mifgsm import MIFGSM
import torchvision.transforms as transforms
import os

class ATTA(MIFGSM):
    """
    ATTA Attack
    'Improving the Transferability of Adversarial Samples with Adversarial Transformations (CVPR 2021)'(https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_Improving_the_Transferability_of_Adversarial_Samples_With_Adversarial_Transformations_CVPR_2021_paper.pdf)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        gamma (float): the scalar weight to trade-off the contributions of each loss  function.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., gamma=1.0

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/atta/resnet18 --attack atta --model=resnet18
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., gamma=1.0, targeted=False, random_start=False, 
                norm='linfty', loss='crossentropy', device=None, attack='ATTA', checkpoint_path='./path/to/checkpoints/', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.gamma = gamma
        self.checkpoint_path = checkpoint_path
        self.atta_model = self.load_atta_model()
        
    def load_atta_model(self, **kwargs):
        atta_model = torch.nn.Sequential(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ATTA_Model())

        weight_name = os.path.join(self.checkpoint_path, 'atta_model_weight.pth')

        if not os.path.exists(weight_name):
            raise ValueError("Please download the checkpoint of the 'ATTA_Model' from 'https://drive.google.com/drive/folders/1QrL3MGuQH-Jx4jwZ5CWO8zHBtquUQkBZ?usp=sharing', and put it into the path '{}'.".format(self.checkpoint_path))

        atta_model.load_state_dict(torch.load(weight_name))

        return atta_model.eval().to(self.device)

    def lattack_loss(self, x_adv, label, **kwargs): # L_attack loss function
        return self.loss(self.model(x_adv), label) + self.gamma * self.loss(self.model(self.atta_model(x_adv)), label)

    def get_loss(self, x_adv, label):
        """
        Calculate the loss
        """
        return -self.lattack_loss(x_adv, label) if self.targeted else self.lattack_loss(x_adv, label)
    
    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data (N, C, H, W): tensor for input images
            labels (N,): tensor for ground-truth labels if untargetd
            labels (2,N): tensor for [ground-truth, targeted labels] if targeted
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):
            # Calculate the loss
            loss = self.get_loss(data + delta, label)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()
    
class ATTA_Model(nn.Module):
    def __init__(self):
        super(ATTA_Model, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 3, 3, stride=1, padding=1),  #   N = W - F + 2P + 1 = W ->  2P = F - 1
            nn.LeakyReLU(True),
            nn.Conv2d(3, 3, 15, stride=1, padding=7))

    def forward(self, x):
        out = self.conv(x)
        return out