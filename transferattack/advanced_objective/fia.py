import torch

from ..utils import *
from ..gradient.mifgsm import MIFGSM

mid_outputs = []
mid_grads = []
class FIA(MIFGSM):
    """
    FIA Attack
    Feature Importance-aware Transferable Adversarial Attacks (ICCV 2021) (https://arxiv.org/pdf/2107.14185.pdf)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        featur_layer (str): the feature layer name
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_ens (int): the number of gradients to aggregate
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        feature_layer: feature layer to launch the attack
        drop_rate : probability to drop random pixel

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., num_scale=10
    """

    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1., num_ens=30,
                 targeted=False, random_start=False, feature_layer='layer2',
                 norm='linfty', loss='crossentropy', device=None, attack='FIA',drop_rate=0.3, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.num_ens = num_ens
        self.feature_layer = self.find_layer(feature_layer)
        self.drop_rate = drop_rate

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

    def __backward_hook(self,m,i,o):
        global mid_grad
        mid_grad = o


    def drop(self,data):
        x_drop = torch.zeros(data.size()).cuda()
        x_drop.copy_(data).detach()
        x_drop.requires_grad = True
        Mask = torch.bernoulli(torch.ones_like(x_drop) * (1 - self.drop_rate))
        x_drop = x_drop * Mask
        return x_drop
    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        h = self.feature_layer.register_forward_hook(self.__forward_hook)
        h2 = self.feature_layer.register_full_backward_hook(self.__backward_hook)

        agg_grad = 0
        for _ in range(self.num_ens):
            x_drop = self.drop(data)
            output_random = self.model(x_drop)
            output_random = torch.softmax(output_random, 1)
            loss = 0
            for batch_i in range(data.shape[0]):
                loss += output_random[batch_i][label[batch_i]]
            self.model.zero_grad()
            loss.backward()
            agg_grad += mid_grad[0].detach()
        for batch_i in range(data.shape[0]):
            agg_grad[batch_i] /= agg_grad[batch_i].norm(2)
        h2.remove()

        for _ in range(self.epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data + delta))
            # Calculate the loss
            loss = (mid_output * agg_grad).sum()
            self.model.zero_grad()
            # Calculate the gradients
            grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
            # Update adversarial perturbation
            delta = self.update_delta(delta, data,-grad, self.alpha)
        h.remove()
        return delta.detach()
