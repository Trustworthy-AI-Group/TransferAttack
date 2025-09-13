import torch
import torch.nn as nn

from ..gradient.mifgsm import MIFGSM


class FAUG(MIFGSM):
    """
    FAUG Attack
    'Improving the Transferability of Adversarial Examples by Feature Augmentation (TNNLS 2025)'(https://ieeexplore.ieee.org/document/10993300)

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
        noise_type (str): the type of noise, normal/uniform
        layer_names: target layer names to inject noise; if None, use 'conv1' for resnet50, otherwise the first Conv2d.
        mean1 (float): mean for normal noise (default 0.0).
        std1 (float): scale for normal noise (default 0.3).
        lower1 (float): lower bound for uniform noise (default -0.2).
        upper1 (float): upper bound for uniform noise (default 0.2). 

    Official arguments:
        epsilon=16/255, alpha=2/255, epoch=10, decay=1., mean1=0., std1=0.3

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/faug/resnet50 --attack faug --model resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/faug/resnet50 --eval
    """

    def __init__(self, model_name,
                 epsilon=16/255, alpha=2/255, epoch=10, decay=1., targeted=False,
                 random_start=False, norm='linfty', loss='crossentropy', device=None, attack='FAUG',
                 layer_names=None, noise_type='normal', mean1=0.0, std1=0.3, lower1=-0.2, upper1=0.2,
                 burn_in_steps=1, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)

        name = (model_name or "").lower()
        if layer_names is not None:
            if isinstance(layer_names, str):
                toks: List[str] = []
                for sep in (",", ";"):
                    for t in layer_names.split(sep):
                        t = t.strip()
                        if t:
                            toks.append(t)
                layer_names = toks if toks else [layer_names]
            self._target_layers = _resolve_by_patterns(self.model, layer_names)
        else:
            if "resnet50" in name:
                self._target_layers = _resolve_by_patterns(self.model, ["conv1"])
            else:
                n, m = _first_conv(self.model)
                self._target_layers = [(n, m)]
        self._noise_type = noise_type.lower()
        self._mean1 = float(mean1)
        self._std1 = float(std1)
        self._lower1 = float(lower1)
        self._upper1 = float(upper1)
        self._burn_in = max(0, int(burn_in_steps))

    def get_logits(self, x, use_noise=False, **kwargs):
        if not use_noise:
            return self.model(x)
        with _FAUGHook(
            self._target_layers,
            noise_type=self._noise_type,
            mean1=self._mean1,
            std1=self._std1,
            lower1=self._lower1,
            upper1=self._upper1,
        ):
            return self.model(x)

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
            label = label[1] # the second element is the targeted label tensor

        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)
        momentum = 0

        for i in range(self.epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data + delta, momentum=momentum), use_noise=(i >= self._burn_in))

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()



def _resolve_by_patterns(model, patterns):
    named_list = list(model.named_modules())
    named_map = dict(named_list)
    results: List[Tuple[str, nn.Module]] = []
    for pat in patterns:
        if pat in named_map:
            results.append((pat, named_map[pat]))
            continue
        found = None
        for n, m in named_list:
            if n.endswith(pat):
                found = (n, m)
                break
        if found is None:
            raise ValueError(f"[FAUG] Pattern '{pat}' not found in model.named_modules().")
        results.append(found)
    return results


def _first_conv(module):
    for n, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            return n, m
    raise ValueError("[FAUG] No Conv2d layer found.")


class _FAUGHook:
    def __init__(self, modules, *, noise_type='normal', mean1=0.0, std1=0.3, lower1=-0.2, upper1=0.2):
        self.modules = modules
        self.noise_type = noise_type.lower()
        self.mean1 = float(mean1)
        self.std1 = float(std1)
        self.lower1 = float(lower1)
        self.upper1 = float(upper1)
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def _noise(self, feat):
        if self.noise_type == "normal":
            scale = feat.std().item()
            std = max(0.0, self.std1 * scale)
            return torch.zeros_like(feat).normal_(mean=self.mean1, std=std)
        elif self.noise_type == "uniform":
            return torch.zeros_like(feat).uniform_(self.lower1, self.upper1)
        else:
            raise ValueError(f"[FAUG] Unsupported noise_type '{self.noise_type}'.")

    def __enter__(self):
        def _hook(_m, _inp, out):
            return out + self._noise(out)
        for _, m in self.modules:
            self._handles.append(m.register_forward_hook(_hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        return False

