import torch

from ..utils import *
from ..gradient.mifgsm import MIFGSM

mid_outputs = []
mid_grads = []
class FMAA(MIFGSM):
    """
    FMAA(Feature Momentum Adversarial Attack)
    'Enhancing the Transferability via Feature-Momentum Adversarial Attack' (https://arxiv.org/pdf/2204.10606.pdf)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_ens (int): the number of gradients to aggregate.
        lamb (float): the decay factor for feature momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        feature_layer: feature layer to launch the attack
        drop_rate : probability to drop random pixel

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., num_ens=30, lamb = 1.1
        drop_rate=0.4 for 1st iteration and 0.1 for the rest iterations

    Example script:
        python main.py --attack=fmaa --output_dir adv_data/fmaa/resnet18
    """

    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1., num_ens=30, lamb=1.1,
                 targeted=False, random_start=False,
                 norm='linfty', loss='crossentropy', device=None, attack='FMAA',feature_layer='layer2',drop_rate=0.3, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.num_ens = num_ens
        self.feature_layer = self.find_layer(feature_layer)
        self.drop_rate = drop_rate
        self.lamb = lamb

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

    def get_beta(self, agg_grad, beta, **kwargs):
        """
        The momentum calculation
        """
        return beta * self.lamb + agg_grad / (agg_grad.abs().mean(dim=(1,2,3), keepdim=True))

    def get_agg_grad(self, data, delta, label, **kwargs):
        # add hook
        h2 = self.feature_layer.register_full_backward_hook(self.__backward_hook)

        # get adversarial image
        adv_image = data + delta

        # calculate aggregate gradient
        agg_grad = 0
        for _ in range(self.num_ens):
            # generate random pixel-wise mask
            Mask = torch.bernoulli(torch.ones_like(data) * (1 - self.drop_rate))

            # get logits
            output_random = self.get_logits(self.transform(adv_image*Mask))

            # get probability
            output_random = torch.softmax(output_random, 1)

            # calculate the loss
            loss = 0
            for batch_i in range(data.shape[0]):
                loss += output_random[batch_i][label[batch_i]]

            # get gradient w.r.t. intermediate layer feature
            grad = self.get_grad(loss, delta)
            agg_grad += mid_grad[0].detach()
            
        # get average gradient
        agg_grad /= self.num_ens

        # remove hook
        h2.remove()

        return agg_grad

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

        h = self.feature_layer.register_forward_hook(self.__forward_hook)

        beta = 0
        for _ in range(self.epoch):
            # set drop_rate
            if _ == 0:
                self.drop_rate = 0.4
            else:
                self.drop_rate = 0.1

            # get agg_grad
            agg_grad = self.get_agg_grad(data, delta, label)

            # update beta
            beta = self.get_beta(agg_grad, beta)

            # Obtain the output
            logits = self.get_logits(self.transform(data + delta))

            # Calculate the loss
            loss = (mid_output * beta).sum()

            self.model.zero_grad()

            # Calculate the gradients
            grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, -grad, self.alpha)

        h.remove()
        return delta.detach()
