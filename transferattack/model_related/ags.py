import torch
from ..utils import *
from ..attack import Attack

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50 as regular_resnet50

mid_outputs = []

class AGS(Attack):
    """
    AGS Attack
    'AGS: Affordable and Generalizable Substitute Training for Transferable Adversarial Attack (AAAI 2024)'(comming soon..)

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
        
    Official arguments:
        epsilon=25.5/255, alpha=1.0/255, epoch=300, decay=1.

    Example script:
        python main.py --batchsize=20 --input_dir ./path/to/data --output_dir adv_data/ags/ags_coco --attack ags --model ags_coco
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.0/255, epoch=300, decay=1., targeted=False, 
                random_start=True, norm='linfty', loss='crossentropy', device=None, attack='AGS', checkpoint_path='./path/to/checkpoints/', **kwargs):
        self.checkpoint_path = checkpoint_path
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay

    def load_model(self, model_name):
        # download model: https://github.com/lwmming/AGS/README.md
        if model_name == 'ags_coco':
            model_path = os.path.join(self.checkpoint_path, 'coco_ags_100.pth')
        elif model_name == 'ags_comics':
            model_path = os.path.join(self.checkpoint_path, 'comics_ags_100.pth')
        elif model_name == 'ags_paintings':
            model_path = os.path.join(self.checkpoint_path, 'paintings_ags_100.pth')
        else:
            raise ValueError('model:{} not supported'.format(model_name))

        if not os.path.exists(model_path):
            raise ValueError("Please download checkpoints from 'https://github.com/lwmming/AGS', and put them into the path './path/to/checkpoints'.")
        
        model = Basic_SSL_Model(128)
        model.load_state_dict(torch.load(model_path))

        return model.eval().cuda()

    def get_loss(self, original_mids, new_mids):
        """
        Overriden for AGS
        """
        loss_mid = 0.
        for i, new_mid in enumerate(new_mids):
            n_img = original_mids[i].shape[0]
            loss_mid += (1 - F.cosine_similarity(original_mids[i].reshape(n_img, -1), new_mid.reshape(n_img, -1)).mean())
        return loss_mid
        

    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: AGS does no need labels
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        # Initialize adversarial perturbation
        delta = self.init_delta(data)
        
        global mid_outputs
        
        feature_layers = ['5']
        
        hs = []
        def get_mid_output(model_, input_, o):
            global mid_outputs
            mid_outputs.append(o)
        for layer_name in feature_layers:
            hs.append(self.model.f._modules.get(layer_name).register_forward_hook(get_mid_output))

        out = self.model(data)
        mid_originals = []
        for mid_output in mid_outputs:
            mid_original = torch.zeros(mid_output.size()).to(self.device)
            mid_originals.append(mid_original.copy_(mid_output))
        mid_outputs = []

        for _ in range(self.epoch):
            # Obtain the output
            _ = self.model(self.transform(data+delta))

            mid_originals_ = []
            for mid_original in mid_originals:
                mid_originals_.append(mid_original.detach())
            
            # Calculate the loss
            loss = self.get_loss(mid_originals_, mid_outputs)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, grad, self.alpha)
            
            mid_outputs = []
        
        for h in hs:
            h.remove()

        return delta.detach()


class Basic_SSL_Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Basic_SSL_Model, self).__init__()
        self.f = []
        for name, module in regular_resnet50().named_children():
            if not isinstance(module, nn.Linear):
                self.f.append(module)
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        # return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
        return feature, F.normalize(out, dim=-1)
