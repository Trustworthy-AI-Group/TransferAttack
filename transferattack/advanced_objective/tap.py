import torch

from ..utils import *
from ..attack import Attack
import torch.nn as nn
mid_outputs = []

class TAP(Attack):
    """
    TAP Attack
    'Transferable Adversarial Perturbations (ECCV 2018)'(https://openaccess.thecvf.com/content_ECCV_2018/papers/Bruce_Hou_Transferable_Adversarial_Perturbations_ECCV_2018_paper.pdf)

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
        python main.py --attack tap --output_dir adv_data/tap/resnet18
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, beta=1.5, num_scale=30, random=False, epoch=100, decay=1., targeted=False, 
                random_start=False, norm='linfty', loss='crossentropy', device=None, attack='TAP', lam=0.005,alpha_tap=0.5,s=3,yita=0.01,learning_rate=0.006,**kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.radius = beta * epsilon
        self.epoch = epoch
        self.decay = decay
        self.num_scale = num_scale
        self.random = random
        self.lam = lam
        self.alpha_tap = alpha_tap
        self.s = s
        self.yita = yita
        self.learning_rate = learning_rate
        
        
    def get_loss(self, logits, label, x, x_adv, original_mids, new_mids):
        """
        Overriden for TAP
        """
        l1 = nn.CrossEntropyLoss()(logits, label)
        l2 = 0.
        for i, new_mid in enumerate(new_mids):
            a = torch.sign(original_mids[i] ) * torch.pow(torch.abs(original_mids[i]),self.alpha_tap)
            b = torch.sign(new_mid) * torch.pow(torch.abs(new_mid),self.alpha_tap)
            l2 += self.lam * (a-b).norm() **2
        l3 = self.yita * torch.abs(nn.AvgPool2d(self.s)(x-x_adv)).sum()
        return -(l1 + l2 + l3) if self.targeted else l1 + l2 + l3
        
    
    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
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
            mid_outputs.append(o)
        for layer_name in feature_layers:
            if isinstance(self.model[1]._modules.get(layer_name), nn.Sequential):
                for i in range(len(self.model[1]._modules.get(layer_name))):
                    hs.append(self.model[1]._modules.get(layer_name)[i].register_forward_hook(get_mid_output))
            else:
                hs.append(self.model[1]._modules.get(layer_name).register_forward_hook(get_mid_output))
        out = self.model(data)
        
        mid_originals = []
        for mid_output in mid_outputs:
            mid_original = torch.zeros(mid_output.size()).to(self.device)
            mid_originals.append(mid_original.copy_(mid_output))
        mid_outputs = []

        for _ in range(self.epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta))
            
            mid_originals_ = []
            for mid_original in mid_originals:
                mid_originals_.append(mid_original.detach())
            
            # Calculate the loss
            loss = self.get_loss(logits, label,data, data+delta,mid_originals_,mid_outputs)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, grad, self.alpha)
            
            mid_outputs = []
        
        for h in hs:
            h.remove()

        return delta.detach()

