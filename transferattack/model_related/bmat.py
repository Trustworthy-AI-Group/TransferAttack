import copy
import os
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from ..attack import Attack
from ..utils import wrap_model


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MBA_CHECKPOINT = _REPOSITORY_ROOT / 'checkpoints' / 'resnet50_morebayesian_attack.pt'
_DEFAULT_INCEPTION_CHECKPOINT = _REPOSITORY_ROOT / 'checkpoints' / 'inception_v3_google-1a9a5a14.pth'


def _require_higher():
    """Load the BMAT-only dependency without affecting other attacks."""
    try:
        import higher
    except ModuleNotFoundError as exc:
        raise ImportError(
            'BMAT requires the optional package "higher==0.2.1". '
            'Install it with: pip install higher==0.2.1'
        ) from exc
    return higher


class _PerturbationParameter(nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.p = nn.Parameter(tensor)

    def forward(self):
        return self.p


def _input_diversity(x, diversity_prob=0.5, resize_rate=1.1):
    """Input diversity with the settings used by the rebuttal implementation."""
    if torch.rand(1) > diversity_prob:
        return x

    image_size = x.shape[-1]
    resized_size = int(image_size * resize_rate)
    random_size = torch.randint(
        low=min(image_size, resized_size),
        high=max(image_size, resized_size),
        size=(1,),
        dtype=torch.int32,
    ).item()
    resized = F.interpolate(x, size=[random_size, random_size], mode='bilinear', align_corners=False)
    remaining_height = resized_size - random_size
    remaining_width = resized_size - random_size
    pad_top = torch.randint(0, remaining_height, (1,), dtype=torch.int32).item()
    pad_left = torch.randint(0, remaining_width, (1,), dtype=torch.int32).item()
    padded = F.pad(
        resized,
        [pad_left, remaining_width - pad_left, pad_top, remaining_height - pad_top],
        value=0,
    )
    return F.interpolate(padded, size=[image_size, image_size], mode='bilinear', align_corners=False)


def _build_bayesian_resnet50(state_dict):
    model = models.resnet50()
    if next(iter(state_dict)).startswith('module.'):
        model = nn.DataParallel(model)
        model.load_state_dict(state_dict)
        return model.module
    model.load_state_dict(state_dict)
    return model


def _load_mba_pseudo_surrogate(checkpoint_path, device, num_subnets=6):
    """Sample the final of six MBA subnets, matching the released single-surrogate setup."""
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise ValueError(
            'BMAT MBA mode requires resnet50_morebayesian_attack.pt. '
            f'Place it at {checkpoint_path} or pass mba_checkpoint_path when '
            'constructing BMAT from Python.'
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    mean_model = _build_bayesian_resnet50(checkpoint['mean_state_dict']).to(device)
    sqmean_model = _build_bayesian_resnet50(checkpoint['sqmean_state_dict']).to(device)

    sampled_model = None
    for _ in range(num_subnets):
        sampled_model = copy.deepcopy(mean_model)
        noise_dict = OrderedDict()
        for (name, parameter_mean), parameter_sqmean in zip(
            mean_model.named_parameters(), sqmean_model.parameters()
        ):
            variance = torch.clamp(parameter_sqmean.data - parameter_mean.data.square(), min=1e-30)
            noise_dict[name] = variance.sqrt() * torch.randn_like(parameter_mean)
        with torch.no_grad():
            for (name, parameter), (_, noise) in zip(sampled_model.named_parameters(), noise_dict.items()):
                parameter.add_(noise, alpha=1.5)

    return wrap_model(sampled_model.eval().to(device))


def _vector_to_grads(vector, parameters):
    grads, offset = [], 0
    for parameter in parameters:
        size = parameter.numel()
        grads.append(vector[offset:offset + size].view_as(parameter).detach())
        offset += size
    return grads


def _hessian_vector_product(inner_grad, vector, parameters, damping, retain_graph):
    hessian_vector = torch.autograd.grad(
        inner_grad, parameters, grad_outputs=vector, retain_graph=retain_graph
    )
    return torch.nn.utils.parameters_to_vector(hessian_vector).detach() / damping + vector


def _conjugate_gradient(inner_grad, outer_grad, parameters, steps, damping):
    solution = outer_grad.detach().clone()
    residual = outer_grad.detach().clone() - _hessian_vector_product(
        inner_grad, solution, parameters, damping, retain_graph=True
    )
    direction = residual.detach().clone()
    for step in range(steps):
        hessian_direction = _hessian_vector_product(
            inner_grad, direction, parameters, damping, retain_graph=step + 1 < steps
        )
        scale = (residual @ residual) / (direction @ hessian_direction)
        solution = solution + scale * direction
        next_residual = residual - scale * hessian_direction
        direction = next_residual + (next_residual @ next_residual) / (residual @ residual) * direction
        residual = next_residual.detach().clone()
    return _vector_to_grads(solution, parameters)


class BMAT(Attack):
    """
    Learning with Bilevel-Minimax Optimization for Efficient and Reliable Transfer Attacks (ECCV 2026)
    This implementation preserves BMAT's two-stage update order: finite-step SWM/IGA trajectory seeding followed by a standard MI-FGSM deployment phase.
    This public TransferAttack benchmark uses the ResNet-50 source surrogate.
    Paper: https://arxiv.org/abs/2608.11815
    Official code: https://github.com/callous-youth/BMAT

    Dependencies:
        higher==0.2.1: pip install higher==0.2.1

    Arguments:
        model_name (str): the source surrogate model. BMAT supports ResNet-50.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of deployment-stage iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device is inferred from the source model.
        pseudo_surrogate (str): ``'mba'`` (default) or ``'inception_v3'``.
        pseudo_checkpoint (str): checkpoint for the optional Inc-v3 pseudo-surrogate.
        mba_checkpoint_path (str): checkpoint containing MBA mean and variance.
        mba_num_subnets (int): number of MBA draws; the final of six matches the released single-surrogate setting.
        transform (str): ``'di'`` for TransferAttack's DIM-style input diversity, or ``'none'``.
        momentum (str): ``'mi'`` for TransferAttack's MI-FGSM-style momentum update, or ``'none'``.
        meta_steps (int): number of trajectory-seeding outer updates.
        inner_steps (int): number of SWM inner adaptation steps per meta step.
        attack_lr (float): SWM perturbation and surrogate adaptation rate.
        cg_steps (int): number of implicit-gradient CG iterations.
        damping (float): damping coefficient in the CG linear system.
        first_order (bool): whether to use the first-order inner response.

    Official arguments:
        epsilon=16/255, alpha=1.6/255, epoch=10, decay=1.0, pseudo_surrogate='mba', mba_num_subnets=6, transform='none', momentum='mi', meta_steps=3, inner_steps=10, attack_lr=2.0, cg_steps=1, damping=1.0, first_order=True.

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/bmat/resnet50 --attack bmat --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/bmat/resnet50 --eval
    """

    def __init__(
        self,
        model_name='resnet50',
        epsilon=16 / 255,
        alpha=1.6 / 255,
        epoch=10,
        decay=1.0,
        targeted=False,
        random_start=False,
        norm='linfty',
        loss='crossentropy',
        device=None,
        attack='BMAT',
        pseudo_surrogate='mba',
        pseudo_checkpoint=None,
        mba_checkpoint_path=None,
        mba_num_subnets=6,
        transform='none',
        momentum='mi',
        meta_steps=3,
        inner_steps=10,
        attack_lr=2.0,
        cg_steps=1,
        damping=1.0,
        first_order=True,
        **kwargs,
    ):
        if model_name != 'resnet50':
            raise ValueError('BMAT currently supports the published ResNet-50 source surrogate only.')
        self.higher = _require_higher()
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        if not isinstance(self.model, nn.Sequential) or len(self.model) != 2:
            raise RuntimeError('BMAT expects TransferAttack to wrap the source as preprocessing plus backbone.')

        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.meta_steps = meta_steps
        self.inner_steps = inner_steps
        self.attack_lr = attack_lr
        self.cg_steps = cg_steps
        self.damping = damping
        self.first_order = first_order
        self.pseudo_surrogate = pseudo_surrogate
        self.use_di = transform == 'di'
        self.use_momentum = momentum == 'mi'
        self.preprocess = self.model[0]
        self.inner_model = self.model[1]

        if transform not in {'di', 'none'}:
            raise ValueError(f'Unsupported BMAT transform: {transform}')
        if momentum not in {'mi', 'none'}:
            raise ValueError(f'Unsupported BMAT momentum: {momentum}')

        if self.pseudo_surrogate == 'mba':
            self.pseudo_model = _load_mba_pseudo_surrogate(
                str(mba_checkpoint_path or _DEFAULT_MBA_CHECKPOINT), self.device, mba_num_subnets
            )
        elif self.pseudo_surrogate == 'inception_v3':
            pseudo_checkpoint = pseudo_checkpoint or _DEFAULT_INCEPTION_CHECKPOINT
            if not os.path.isfile(pseudo_checkpoint):
                raise ValueError(
                    'BMAT Inc-v3 mode requires inception_v3_google-1a9a5a14.pth. '
                    f'Place it at {pseudo_checkpoint} or pass pseudo_checkpoint '
                    'when constructing BMAT from Python.'
                )
            self.pseudo_model = models.inception_v3(
                weights=None, aux_logits=True, init_weights=False
            ).to(self.device)
            checkpoint = torch.load(pseudo_checkpoint, map_location=self.device)
            self.pseudo_model.load_state_dict(checkpoint, strict=True)
        else:
            raise ValueError(f'Unsupported BMAT pseudo-surrogate: {self.pseudo_surrogate}')
        self.pseudo_model.eval()
        for parameter in self.pseudo_model.parameters():
            parameter.requires_grad_(False)

    def _transform(self, image):
        return _input_diversity(image) if self.use_di else image

    def _accumulate_gradient(self, gradient, momentum):
        if not self.use_momentum:
            return gradient, momentum
        gradient = self.get_momentum(gradient, momentum)
        return gradient, gradient.detach()

    def _phase_one_gradient(self, data, label, momentum):
        image = data
        for _ in range(self.meta_steps):
            image = image.detach().requires_grad_(True)
            perturbation = _PerturbationParameter(image)
            self.inner_model.p_cur = perturbation
            model_parameters = [
                parameter for parameter in self.inner_model.parameters()
                if parameter is not perturbation()
            ]
            perturbation_lr = self.attack_lr if self.targeted else -self.attack_lr
            inner_optimizer = torch.optim.SGD(
                [
                    {'params': model_parameters, 'lr': self.attack_lr * 5e-5},
                    {'params': [perturbation()], 'lr': perturbation_lr},
                ],
                lr=self.attack_lr,
            )
            with self.higher.innerloop_ctx(
                self.inner_model,
                inner_optimizer,
                copy_initial_weights=True,
                track_higher_grads=not self.first_order,
            ) as (functional_model, diffopt):
                for _ in range(self.inner_steps):
                    clean_logits = functional_model(self.preprocess(self._transform(data)))
                    adversarial_logits = functional_model(
                        self.preprocess(self._transform(functional_model.p_cur()))
                    )
                    inner_loss = self.loss(adversarial_logits, label) + 0.1 * self.loss(clean_logits, label)
                    # Match the released trajectory optimizer: retain the last
                    # finite response and form the outer gradient from it.
                    if torch.isnan(inner_loss):
                        break
                    diffopt.step(inner_loss)

                outer_loss = self.loss(self.pseudo_model(functional_model.p_cur()), label)
                final_inner_loss = self.loss(
                    functional_model(self.preprocess(self._transform(functional_model.p_cur()))), label
                )
                inner_gradient = torch.nn.utils.parameters_to_vector(
                    torch.autograd.grad(final_inner_loss, functional_model.p_cur.parameters(), create_graph=True)
                )
                outer_gradient = torch.nn.utils.parameters_to_vector(
                    torch.autograd.grad(outer_loss, [functional_model.p_cur()])
                )
                hypergradient = _conjugate_gradient(
                    inner_gradient,
                    outer_gradient,
                    list(functional_model.p_cur.parameters()),
                    self.cg_steps,
                    self.damping,
                )[0]

            gradient = hypergradient.detach()
            gradient, momentum = self._accumulate_gradient(gradient, momentum)
            if self.targeted:
                gradient = -gradient
            image = image.detach() + self.alpha * gradient.sign()
            image = torch.max(torch.min(image, data + self.epsilon), data - self.epsilon)
            image = torch.clamp(image, 0.0, 1.0)
        return image.detach(), momentum.detach()

    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1]
        data = data.detach().to(self.device)
        label = label.detach().to(self.device)
        self.inner_model.eval()
        self.pseudo_model.eval()

        seeded, momentum = self._phase_one_gradient(data, label, torch.zeros_like(data))
        delta = (seeded - data).detach()
        for _ in range(self.epoch):
            delta = delta.detach().requires_grad_(True)
            logits = self.model(self._transform(data + delta))
            loss = self.get_loss(logits, label)
            gradient = torch.autograd.grad(loss, delta)[0]
            gradient, momentum = self._accumulate_gradient(gradient, momentum)
            delta = self.update_delta(delta, data, gradient, self.alpha)
        return delta.detach()
