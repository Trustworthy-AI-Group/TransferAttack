import torch
from torch import nn

from ..utils import *
from ..gradient.mifgsm import MIFGSM


class IAA(MIFGSM):
    '''
    IAA Attack
    'Rethinking adversarial transferability from a data distribution perspective (ICLR 2022)'(https://openreview.net/pdf?id=gVRhIEajG1k)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        beta (dir): the beta for softplus.
        lamb (dir): the lamb for residual modules.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=2/255, epoch=10, decay=0., random_start=True

    Example script:
        python main.py --attack=iaa --output_dir adv_data/iaa/resnet18
    '''

    def __init__(self, model_name, epsilon=16/255, alpha=2/255, epoch=10, decay=1., targeted=False, random_start=True, 
                 norm='linfty', loss='crossentropy', device=None, attack='IAA', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay,
                         targeted, random_start, norm, loss, device, attack)
        self.beta = {'resnet18': 20, 'resnet34': 20, 'resnet50': 20,
                     'resnet152': 32, 'densenet121': 35, 'densenet201': 35}
        self.lamb = {'resnet18': {'layer1': 0.98, 'layer2': 0.87, 'layer3': 0.73, 'layer4': 0.19},
                     'resnet34': {'layer1': 0.98, 'layer2': 0.87, 'layer3': 0.73, 'layer4': 0.19},
                     'resnet50': {'layer1': 0.98, 'layer2': 0.87, 'layer3': 0.73, 'layer4': 0.19},
                     'resnet152': {'layer1': 0.89, 'layer2': 0.88, 'layer3': 0.70, 'layer4': 0.20},
                     'densenet121': {'denseblock1': 0.80, 'denseblock2': 0.80, 'denseblock3': 0.80, 'denseblock4': 0.44},
                     'densenet201': {'denseblock1': 0.80, 'denseblock2': 0.80, 'denseblock3': 0.80, 'denseblock4': 0.44}}
        self.modify_model(self.model, model_name)

    def modify_model(self, model, model_name):
        """
        Modify the model with IAA
            - Replace ReLU with Softplus
            - Decrease the weight for certain residual modules
        """
        if model_name not in ['resnet18', 'resnet34', 'resnet50', 'resnet152', 'densenet121', 'densenet201']:
            raise ValueError('Model {} not supported'.format(model_name))

        # Replace ReLU with Softplus
        beta = self.beta[model_name]
        self.replace_layers(model, nn.ReLU, nn.Softplus(beta=beta))

        # Decrease the weight for certain residual modules
        lamb = self.lamb[model_name]
        if model_name in ['resnet18', 'resnet34']:
            for name, module in model.named_modules():
                if 'bn2' in name:
                    layer_name = name.split('.')[1]
                    module.register_forward_hook(
                        self.resnet_forward_hook(lamb[layer_name]))
        elif model_name in ['resnet50', 'resnet152']:
            for name, module in model.named_modules():
                if 'bn3' in name:
                    layer_name = name.split('.')[1]
                    module.register_forward_hook(
                        self.resent_forward_hook(lamb[layer_name]))
        elif model_name in ['densenet121', 'densenet201']:
            for name, module in model.named_modules():
                if module.__class__.__name__ == '_DenseLayer':
                    block_name = name .split('.')[2]
                    layer_name = name.split('.')[3]
                    module.register_forward_hook(
                        self.densenet_forward_hook(lamb[block_name]))

    def replace_layers(self, model, old, new):
        """
        Replace the old layer with the new layer in the model

        Inputs:
            model (nn.Module): the model to be modified
            old (nn.Module): the old layer to be replaced
            new (nn.Module): the new layer to replace the old layer
        """
        for n, module in model.named_children():
            if len(list(module.children())) > 0:
                # compound module, go inside it
                self.replace_layers(module, old, new)

            if isinstance(module, old):
                setattr(model, n, new)

    def resnet_forward_hook(self, lamb):
        """
        Forward hook for residual modules

        Inputs:
            lamb (float): the weight for the residual modules
        """
        def __forward_hook(module, inputs, outputs):
            return outputs * lamb
        return __forward_hook
    
    def densenet_forward_hook(self, lamb):
        """
        Forward hook for residual modules

        Inputs:
            lamb (float): the weight for the residual modules
        """
        def __forward_hook(module, inputs, outputs):
            features = inputs[0]
            features[-1] *= lamb
            return (features,)
        
        return __forward_hook
