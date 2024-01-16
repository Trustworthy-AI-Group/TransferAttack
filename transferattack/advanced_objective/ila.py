import torch

from ..utils import *
from ..attack import Attack
import torch.nn as nn
mid_outputs = None

class Proj_Loss(torch.nn.Module):
    def __init__(self):
        super(Proj_Loss, self).__init__()

    def forward(self, old_attack_mid, new_mid, original_mid, coeff):
        x = (old_attack_mid - original_mid).reshape(1, -1)
        y = (new_mid - original_mid).reshape(1, -1)
        x_norm = x / x.norm()

        proj_loss = torch.mm(y, x_norm.transpose(0, 1)) / x.norm()
        return proj_loss


class ILA(Attack):
    """
    ILA (Intermediate Level Attack)
    'Enhancing Adversarial Example Transferability with an Intermediate Level Attack (ICCV 2019)'(https://arxiv.org/abs/1907.10823)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        coeff (float): coefficient.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., coeff=1.0

    Example script:
        python main.py --attack=ila --output_dir adv_data/ila/resnet18
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, random=False, epoch=10, decay=1., targeted=False, 
                random_start=False, norm='linfty', loss='crossentropy', device=None, attack='ILA', coeff=1.0, **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.random = random
        self.coeff = coeff
        
        
    def get_ila_loss(self, mid_attack_original, mid_output, mid_original):
        """
        Overriden for ILA
        """
        return Proj_Loss()(mid_attack_original, mid_output, mid_original, self.coeff)
        
    
    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        init_delta = super().forward(data, label, **kwargs)
        
        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)
        
        global mid_outputs
        
        feature_layers = self.model[1]._modules.keys()
        
        hs = []
        def get_mid_output(model_, input_, o):
            global mid_outputs
            mid_outputs = o
        count = 0
        names = []
        for layer_name in feature_layers:
            if isinstance(self.model[1]._modules.get(layer_name), nn.Sequential):
                for i in range(len(self.model[1]._modules.get(layer_name))):
                    count += 1
                    names.append([layer_name,i])
            else:
                count = count + 1
                names.append([layer_name])
        mid_layer = int(count/2)
        featureLayer = names[mid_layer]
        if len(featureLayer) == 2:
            hs.append(self.model[1]._modules.get(featureLayer[0])[featureLayer[1]].register_forward_hook(get_mid_output))
        else:
            hs.append(self.model[1]._modules.get(featureLayer[0]).register_forward_hook(get_mid_output))
        out = self.model(data)
        mid_original = torch.zeros(mid_outputs.size()).cuda()
        mid_original.copy_(mid_outputs)
        
        out = self.model(data + init_delta)
        mid_attack_original = torch.zeros(mid_outputs.size()).cuda()
        mid_attack_original.copy_(mid_outputs)
        

        for _ in range(self.epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta))
            
            # Calculate the loss
            loss = self.get_ila_loss(mid_attack_original, mid_outputs, mid_original)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, grad, self.alpha)
            
            mid_outputs = []
        
        for h in hs:
            h.remove()

        return delta.detach()

