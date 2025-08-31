import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn, Tensor
from ..utils import *
from ..gradient.mifgsm import MIFGSM


def imagenet_normalize(x):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalize(x)

def imagenet_denormalize(x):
    denormalize = T.Normalize(mean=[-2.1179, -2.0357, -1.8044], std=[4.3668, 4.4643, 4.4444])
    return denormalize(x)


class P2FA(MIFGSM):
    """
    P2FA Attack
    Pixel2Feature Attack (P2FA): Rethinking the Perturbed Space to Enhance Adversarial Transferability (ICML 2025) (https://openreview.net/pdf?id=bPJo5uSkOJ)

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

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/p2fa/resnet50 --attack p2fa --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/p2fa/resnet50 --eval
    """

    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1., num_ens=30,
                 targeted=False, random_start=False, feature_layer='1.layer2',
                 norm='linfty', loss='crossentropy', device=None, attack='P2FA', eta=28.0, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.ensemble_number = num_ens
        self.layer_name = feature_layer
        self.feature_maps = None
        self.register_hook()
        self.normalize = imagenet_normalize
        self.denormalize = imagenet_denormalize
        self.loss_fn_ce = nn.CrossEntropyLoss()
        self.eta = eta

    def get_maskgrad(self, images: Tensor, labels: Tensor) -> Tensor:
        images = images.clone().detach()
        images.requires_grad = True
        logits = self.model(images)
        labels = labels.long()
        loss = self.loss_fn_ce(logits, labels)
        maskgrad = torch.autograd.grad(loss, images)[0]
        maskgrad /= torch.sqrt(torch.sum(torch.square(maskgrad), dim=(1, 2, 3), keepdim=True))
        return maskgrad.detach()

    def hook(self, module, input, output):
        self.feature_maps = output
        return None

    def register_hook(self):
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(hook=self.hook)


    def get_aggregate_gradient(self, images: Tensor, labels: Tensor) -> Tensor:
        _ = self.model(images)
        images_denorm = self.denormalize(images)
        images_masked = images.clone().detach()
        aggregate_grad = torch.zeros_like(self.feature_maps)
        targets = F.one_hot(labels.type(torch.int64), 1000).float().to(self.device)
        for _ in range(self.ensemble_number):
            g = self.get_maskgrad(images_masked, labels)
            images_masked = self.normalize(images_denorm + self.eta * g)
            logits = self.model(images_masked)
            loss = torch.sum(logits * targets, dim=1).mean()
            aggregate_grad += torch.autograd.grad(loss, self.feature_maps)[0]
        aggregate_grad /= torch.sqrt(torch.sum(torch.square(aggregate_grad), dim=(1, 2, 3), keepdim=True))
        return -aggregate_grad


    def forward(self, data, label, **kwargs):
        data = data.clone().detach().to(self.device)
        labels = label.clone().detach().to(self.device)
        delta = self.init_delta(data)
        gg = torch.zeros_like(data)
        _ = self.model(data)
        g = torch.zeros_like(self.feature_maps)
        for _ in range(self.epoch):
            aggregate_grad = self.get_aggregate_gradient(data + delta, labels)
            _ = self.get_logits(self.transform(data + delta))
            feature_maps = self.feature_maps.clone()
            g = self.decay * g + aggregate_grad
            feature_maps += 100000.0 * g / torch.sqrt(torch.sum(torch.square(g), dim=(1, 2, 3), keepdim=True))
            for _ in range(10):
                _ = self.get_logits(self.transform(data + delta))
                loss = torch.sum(torch.square(self.feature_maps - feature_maps), dim=(1, 2, 3)).mean()
                grad = torch.autograd.grad(loss, delta)[0]
                gg = self.decay * gg + grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                delta = self.update_delta(delta, data, -grad, self.alpha)
        return delta
