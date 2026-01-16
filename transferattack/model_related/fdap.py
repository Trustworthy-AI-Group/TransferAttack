from ..utils import *
from ..gradient.mifgsm import MIFGSM

class FDAP(MIFGSM):
    """
    FDAP Attack
    Attacking Transformers with Feature Diversity Adversarial Perturbation (AAAI 2024) (https://ojs.aaai.org/index.php/AAAI/article/view/28365)

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
        device (torch.device): the device for data. If it is None, the device would be same as model.
        beta (float): The power for calculating Frobenius norm.
        gamma (float): The trade-off parameter for diversity loss.

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=3/255, epoch=30, decay=1.

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/fdap/vit_l_16 --attack fdap --model=vit_l_16
        python main.py --input_dir ./path/to/data --output_dir adv_data/fdap/vit_l_16 --eval
    """
    def __init__(self, model_name, epsilon=16/255, alpha=3/255, epoch=30, decay=1.0,
                 targeted=False, random_start=False,
                 norm='linfty', loss='crossentropy', device=None, attack='FDAP', beta=2.0, gamma=0.1):
        super(FDAP, self).__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.decay = decay
        self.beta = beta
        self.layer_names = [f"1.encoder.layers.encoder_layer_{i}.ln_2" for i in range(5, 10)]
        self.features = {}
        self.gamma = gamma
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                module.register_forward_hook(self._hook_fn(name))

    def _hook_fn(self, layer_name):
        def hook(module, input, output):
            self.features[layer_name] = output
        return hook

    def forward(self, data, label, **kwargs):
        data = data.clone().detach().to(self.device)
        momentum = torch.zeros_like(data)
        label = label.clone().detach().to(self.device)
        delta = self.init_delta(data)
        for _ in range(self.epoch):
            self.features.clear()
            logits = self.get_logits(self.transform(data + delta, momentum=momentum))

            loss_div = 0.0
            for layer in self.layer_names:
                feat = self.features[layer]
                B, L, C = feat.shape
                mean_feat = feat.mean(dim=1, keepdim=True)
                res = feat - mean_feat
                # Frobenius norm per sample
                r = res.view(B, -1).norm(p=2, dim=1) + 1e-8
                loss_div += (torch.log(r).pow(self.beta)).mean()
            loss_div = -loss_div

            loss = self.get_loss(logits, label)
            loss = loss + self.gamma * loss_div
            grad = self.get_grad(loss, delta)
            momentum = self.get_momentum(grad, momentum)
            delta = self.update_delta(delta, data, momentum, self.alpha)
        return delta
