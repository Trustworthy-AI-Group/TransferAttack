import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from ..utils import *
from ..attack import Attack
import torch.nn as nn
from ..gradient.mifgsm import MIFGSM

class Mid_layer_target_Loss(nn.Module):
    def __init__(self):
        super(Mid_layer_target_Loss, self).__init__()

    def forward(self, old_attack_mid, new_mid, original_mid, coeff):
        x = (old_attack_mid - original_mid).view(1, -1)
        y = (new_mid - original_mid).view(1, -1)

        x_norm = x / x.norm()
        if (y == 0).all():
            y_norm = y
        else:
            y_norm = y / y.norm()
        angle_loss = torch.mm(x_norm, y_norm.transpose(0, 1))
        magnitude_gain = y.norm() / x.norm()
        return angle_loss + magnitude_gain * coeff
 
class TRAP(MIFGSM):
    """
    Transferable and Robust Adversarial Perturbation Generation (TRAP)
    'Exploring Transferable and Robust Adversarial Perturbation Generation from the Perspective of Network Hierarchy'(https://arxiv.org/pdf/2108.07033v1)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        beta (float): the relative value for the neighborhood.
        num_neighbor (int): the number of samples for estimating the gradient variance.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, beta=1.5, num_scale=20, epoch=10, decay=1.

    Example script:
        python main.py --attack trap --output_dir adv_data/trap/resnet18
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, beta=0.8, epoch=300, baseline_epoch=4, decay=1., targeted=False, probb=0.9,
                random_start=False, norm='linfty', loss='crossentropy', device=None, attack='TRAP', feature_layer='layer4', coeff=0.8, **kwargs):
        super().__init__(model_name, epsilon, epsilon/baseline_epoch, baseline_epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.beta = beta
        self.enhance_epoch = epoch - baseline_epoch
        self.feature_layer = self.find_layer(feature_layer)
        self.coeff = coeff
        self.affine_trans = transforms.RandomAffine(degrees=90, translate=(0.1,0.1), scale=(0.5,1.5), shear=(-30,30,-30,30),interpolation=transforms.InterpolationMode.BILINEAR)
        self.trap_loss = Mid_layer_target_Loss()
        self.probb = probb

    def find_layer(self,layer_name):
        if layer_name not in self.model[1]._modules.keys():
            print("Selected layer is not in Model")
            exit()
        else:
            return self.model[1]._modules.get(layer_name)
        
    def __forward_hook(self,m,i,o):
        global mid_output
        mid_output = o
        
    def get_trap_loss(self, old_attack_mid, new_mid, ori_mid):
        return -self.trap_loss(old_attack_mid, new_mid, ori_mid, self.coeff) if self.targeted else self.trap_loss(old_attack_mid, new_mid, ori_mid, self.coeff)
        
    def transform(self, data, **kwargs):
        if torch.rand(1) > self.probb:
            return data
        return self.affine_trans(data)

    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data (N, C, H, W): tensor for input images
            labels (N,): tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        init_delta = super().forward(data, label, **kwargs)
        
        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        h = self.feature_layer.register_forward_hook(self.__forward_hook)

        # Calculate h_ori
        out = self.model(data)
        h_ori = torch.zeros(mid_output.size()).cuda()
        h_ori.copy_(mid_output)
        
        # Calculate h_adv
        out = self.model(data + init_delta)
        h_guide = torch.zeros(mid_output.size()).cuda()
        h_guide.copy_(mid_output)

        # h_guide = h_adv
        momentum = 0
        delta = self.init_delta(data)
        self.alpha = self.epsilon / self.enhance_epoch
        for _ in range(self.enhance_epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta))

            h_adv = torch.zeros(mid_output.size()).cuda()
            h_adv.copy_(mid_output)

            # Calculate the loss
            loss = self.get_trap_loss(h_guide, h_adv, h_ori)
            # print(loss.data)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)
            
            h_guide = (1 - self.beta) * h_adv + self.beta * h_guide
        
        h.remove()
        return delta.detach()