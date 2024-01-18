import torch
import torchvision.transforms as transforms
import torch.nn as nn

from ..utils import *

from ..gradient.mifgsm import MIFGSM

class Mid_layer_target_Loss(nn.Module):
    def __init__(self):
        super(Mid_layer_target_Loss, self).__init__()

    def forward(self, h_star, h_adv, h_x, coeff):
        x = (h_star - h_x).view(1, -1)
        y = (h_adv - h_x).view(1, -1)

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
        beta (float): the balance coefficient.
        epoch (int): the number of iterations.
        baseline_epoch (int): the number of iterations for baseline attack phase.
        probb (float): the execution probability of applying affine-transformation.
        coeff (float): the tradeoff parameter.
        feature_layer (str): the targeted layer for the hidden output.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, beta=0.8, epoch=300, baseline_epoch=150, feature_layer='layer3', probb=0.9, coeff=0.8 decay=1.

    Example script:
        python main.py --attack trap --output_dir adv_data/trap/resnet18
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, beta=0.8, epoch=300, baseline_epoch=150, decay=1., targeted=False, probb=0.9,
                random_start=False, norm='linfty', loss='crossentropy', device=None, attack='TRAP', feature_layer='layer3', coeff=0.8, **kwargs):
        super().__init__(model_name, epsilon, epsilon/baseline_epoch, baseline_epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.beta = beta
        self.enhance_epoch = epoch - baseline_epoch
        self.feature_layer = self.find_layer(feature_layer)
        self.coeff = coeff
        self.affine_trans = transforms.RandomAffine(degrees=90, translate=(0.1,0.1), scale=(0.5,1.5), shear=(-30,30,-30,30),interpolation=transforms.InterpolationMode.BILINEAR)
        self.trap_loss = Mid_layer_target_Loss()
        self.probb = probb

    def find_layer(self,layer_name):
        parser = layer_name.split(' ')
        m = self.model[1]
        for layer in parser:
            if layer not in m._modules.keys():
                print("Selected layer is not in Model")
                exit() 
            else:
                m = m._modules.get(layer)
        return m
        
    def __forward_hook(self,m,i,o):
        global mid_output
        mid_output = o
        
    def get_trap_loss(self, h_star, h_adv, h_x):
        loss = self.trap_loss(h_star, h_adv, h_x, self.coeff)
        return -loss if self.targeted else loss
        
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
        # Perform t1 baseline attack (AIM)
        init_delta = super().forward(data, label, **kwargs)

        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        delta = self.init_delta(data)

        h = self.feature_layer.register_forward_hook(self.__forward_hook)

        # Initialize h_x and obtain its hidden outputs
        logits = self.get_logits(data)
        h_x = mid_output
        
        # Initialize h_star and obtain its hidden outputs
        logits = self.get_logits(data+init_delta)
        h_star = mid_output

        # Update step size alpha
        self.alpha = self.epsilon / self.enhance_epoch

        momentum = 0
        for _ in range(self.enhance_epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta))
            
            # Obtain the hidden output of x_adv
            h_adv = mid_output

            # Calculate the loss
            loss = self.get_trap_loss(h_star, h_adv, h_x)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)
            
            # Update h_star
            h_star = (1 - self.beta) * h_adv + self.beta * h_star
        
        h.remove()
        return delta.detach()