import torch
from ..utils import *
from ..gradient.mifgsm import MIFGSM

mid_output = None
mid_grad = None

class DANAA(MIFGSM):
    """
    DANAA Attack
    DANAA: Towards Transferable Attacks with Double Adversarial Neuron Attribution (ADMA 2023)(https://arxiv.org/pdf/2310.10427)
    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_ens (int): the number of gradients to aggregate.
        scale (float): the scale of random perturbation for non-linear path-based attribution.
        lr (float): the learning rate for non-linear path-based attribution.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        feature_layer: feature layer to launch the attack

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., num_ens=30, scale=0.25, lr=0.0025,

    Example script:
        python main.py --attack danaa --output_dir adv_data/danaa/resnet18
        python main.py --attack danaa --output_dir adv_data/danaa/resnet18 --eval
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., num_ens=30, scale=0.25, lr=0.0025,
                 targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None, attack='DANAA',feature_layer='layer2', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.scale = scale
        self.lr = lr
        self.num_ens = num_ens
        self.feature_layer = self.find_layer(feature_layer)

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

    def get_loss(self, mid_feature, base_feature, agg_grad):
        """
        Overwrite the loss function for DANAA

        Arguments:
            mid_feature: the intermediate feature of adversarial input
            base_feature: the intermediate feature of zero input
            agg_grad: the aggregated gradients w.r.t. the intermediate features
        """
        gamma = 1.0
        attribution = (mid_feature - base_feature) * agg_grad
        blank = torch.zeros_like(attribution)
        positive = torch.where(attribution >= 0, attribution, blank)
        negative = torch.where(attribution < 0, attribution, blank)
        balance_attribution = positive + gamma * negative
        loss = torch.mean(balance_attribution)

        return -loss if self.targeted else loss


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

        # Register the forward and backward hooks
        h = self.feature_layer.register_forward_hook(self.__forward_hook)
        h2 = self.feature_layer.register_full_backward_hook(self.__backward_hook)

        # Initialize the original input for Non-linear path-based attribution
        x_t = data.clone().detach().to(self.device)
        x_t.requires_grad = True

        agg_grad = 0
        for _ in range(self.num_ens):
            # Move along the non-linear path
            x = x_t + torch.randn_like(x_t).cuda() * self.scale

            # Obtain the output
            logits = self.get_logits(x)

            # Calculate the loss
            loss = torch.softmax(logits, 1)[torch.arange(logits.shape[0]), label].sum()

            # Calculate the gradients w.r.t. the input
            x_grad = self.get_grad(loss, x_t)

            # Update the input
            x_t = x_t + self.lr * x_grad.sign()

            # Aggregate the gradients w.r.t. the intermidiate features
            agg_grad += mid_grad[0].detach()

        # Normalize the aggregated gradients
        agg_grad = -agg_grad / torch.sqrt(torch.sum(agg_grad ** 2, dim=(1, 2, 3), keepdim=True))
        h2.remove()

        # Obtain the base features
        self.model(x_t)
        y_base = mid_output.clone().detach()

        momentum = 0
        for _ in range(self.epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data + delta))

            # Calculate the loss
            loss = self.get_loss(mid_output, y_base, agg_grad)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        h.remove()
        return delta.detach()