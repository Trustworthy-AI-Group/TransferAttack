import torch

from ..gradient.mifgsm import MIFGSM


class ILPD(MIFGSM):
    """
    ILPD Attack
    'Improving Adversarial Transferability via Intermediate-level Perturbation Decay'(https://arxiv.org/abs/2304.13410)

    Arguments:
        attack (str): the name of attack.
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        coef (float): coeffcient gamma
        sigma (float): noise size
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epoch=100, sigma=0.05, coef=0.1, N=1, il_pos="layer2.3"
    """

    def __init__(self, **kwargs):
        kwargs["model_name"] = "resnet50"
        kwargs["attack"] = "ILPD"
        kwargs["epoch"] = 100
        kwargs["alpha"] = 1 / 255

        super().__init__(**kwargs)

        self.il_module = self.model[1].layer2[3]
        self.sigma = 0.05
        self.coef = 0.1
        self.N = 1

    def prep_hook(self, ori_img):
        if hasattr(self, "hook"):
            self.hook.remove()
        with torch.no_grad():
            ilout_hook = self.il_module.register_forward_hook(hook_ilout)
            self.model(
                ori_img + self.sigma * torch.randn(ori_img.size()).to(ori_img.device)
            )
            ori_ilout = self.il_module.output
            ilout_hook.remove()
        hook_func = get_hook_pd(ori_ilout, self.coef)
        self.hook = self.il_module.register_forward_hook(hook_func)

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
            label = label[1]  # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data.clone())

        momentum = 0
        for _ in range(self.epoch):
            self.prep_hook(data)

            # Obtain the output
            logits = self.get_logits(
                self.transform(data + delta, momentum=momentum)
            )

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()


def hook_ilout(module, input, output):
    module.output = output


def get_hook_pd(ori_ilout, gamma):
    def hook_pd(module, input, output):
        return gamma * output + (1 - gamma) * ori_ilout

    return hook_pd
