import math
from ..utils import *
from ..gradient.mifgsm import MIFGSM


class AdaMSI_FGM(MIFGSM):
    """
    AdaMSI-FGM Attack
    On the Convergence of an Adaptive Momentum Method for Adversarial Attacks (AAAI 2024) (https://ojs.aaai.org/index.php/AAAI/article/view/29323)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        featur_layer (str): the feature layer name
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        lambda_ (float): the lambda for momentum calculation

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1.,

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/adamsi_fgm/resnet50 --attack adamsi_fgm --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/adamsi_fgm/resnet50 --eval
    """
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6/255, epoch=10, decay=1.0,
                 targeted=False, random_start=False,
                 norm='linfty', loss='crossentropy', device=None, attack='AdaMSI_FGM', lambda_=0.6):
        super(AdaMSI_FGM, self).__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device,
                                   attack)
        self.lambda_ = lambda_

    def get_momentum(self, grad, momentum, **kwargs):
        g_norm1 = grad.abs().view(grad.size(0), -1).sum(dim=1)
        s_t = self.lambda_ * (self.t ** 2) * g_norm1
        beta1_t = self.s_prev / (s_t + 1.0)
        beta2_t = 1.0 - 1.0 / self.t
        self.v = beta2_t * self.v + (1.0 - beta2_t) * (grad * grad)
        V_hat = self.v.sqrt() + 1e-16 / math.sqrt(self.t)
        momentum = momentum * self.decay + beta1_t.view(-1, 1, 1, 1) * (self.x0 + self.delta - self.x_prev)
        momentum = grad / V_hat + momentum
        self.s_prev = s_t
        return momentum


    def update_delta(self, delta, data, grad, alpha, **kwargs):
        if self.norm == 'linfty':
            delta = torch.clamp(delta + alpha * grad, -self.epsilon, self.epsilon)
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
            scaled_grad = grad / (grad_norm + 1e-20)
            delta = (delta + scaled_grad * alpha).view(delta.size(0), -1).renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(delta)
        delta = clamp(delta, img_min-data, img_max-data)
        return delta.detach().requires_grad_(True)


    def forward(self, data, label, **kwargs):
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        self.x0 = data.clone().detach()
        self.x_prev = self.x0.clone()
        self.v = torch.zeros_like(self.x0)
        self.s_prev = torch.zeros(self.x0.size(0), device=self.x0.device)
        delta = self.init_delta(data)
        self.delta = delta
        self.t = 0
        momentum = 0
        for _ in range(self.epoch):
            self.t += 1
            logits = self.get_logits(self.transform(data + delta, momentum=momentum))
            loss = self.get_loss(logits, label)
            grad = self.get_grad(loss, delta)
            momentum = self.get_momentum(grad, momentum)
            delta = self.update_delta(delta, data, momentum, self.alpha)
            self.delta = delta
        return delta
