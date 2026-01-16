import torch
import torch.nn as nn
import numpy as np
import scipy.stats as st
from ..gradient.mifgsm import MIFGSM
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from typing import Callable
from torch import nn, Tensor


class FTM(MIFGSM):
    """
    FTM Attack
    'Improving Transferable Targeted Attacks with Feature Tuning Mixup (CVPR 2025)'(https://arxiv.org/pdf/2411.15553)

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
        attack_type: String indicating which attack components to use:
                    'D' - Diverse Input (DI)
                    'R' - Resized Diverse Input (RDI) 
                    'M' - Momentum (MI)
                    'T' - Translation Invariance (TI)
                    'F' - Feature Tuning Mixup (FTM)

        mu: Momentum decay factor for MI

    Official arguments:
        epsilon=16/255, alpha=2/255, epoch=300, decay=1.0, prob=1.0, targeted=True

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/ftm/resnet50 --attack ftm --model resnet50 --targeted
        python main.py --input_dir ./path/to/data --output_dir adv_data/ftm/resnet50 --eval --targeted
    """

    def __init__(self, model_name, epsilon=16/255, alpha=2/255, epoch=300, decay=1., prob=1.0, targeted=True,
                 random_start=False, norm='linfty', loss='crossentropy', device=None, attack='FTM', attack_type='RTMF', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.attack_type = attack_type
        self.prob = prob

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

        ti_kernel_size = 5
        if 'T' in self.attack_type:
            kernel = gkern(ti_kernel_size, 3).astype(np.float32)
            gaussian_kernel = np.stack([kernel, kernel, kernel])
            gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
            gaussian_kernel = torch.from_numpy(gaussian_kernel).to(self.device)

        # Set loss function
        loss_fn = LogitLoss(label, targeted=True)

        consumed_iteration = 0
        if 'F' in self.attack_type:  # Storing clean features at the first iteration
            with torch.no_grad():
                img_width = data.size()[-1]
                x_f = data
                models = []
                for source_model in [self.model]:
                    models.append(FeatureTuning(source_model, img_width, self.device))
                for model in models:
                    model.start_feature_record()  # Set feature recoding mode
                    model(x_f)  # Feature recording
                    model.end_feature_record()  # Set feature mixup inference mode
                consumed_iteration = 1  # Deduct 1 iteration in total iterations for fair comparisons
                assert consumed_iteration < self.epoch, "consumed_iteration should be less than num_iter"
        else:
            models = [self.model]

        for i in range(consumed_iteration, self.epoch):

            if 'D' in self.attack_type:
                x_adv_or_nes = DI(data+delta, self.prob)
            elif 'R' in self.attack_type:
                x_adv_or_nes = RDI(data+delta)
            else:
                x_adv_or_nes = data+delta
            
            total_loss = 0
            for model in models:
                total_loss += loss_fn(model(x_adv_or_nes))
            # gradient calculation
            if 'F' in self.attack_type:
                all_params = [data+delta]  # first add input
                all_active_layers = []  # for recording each model's selected layers

                for model in models:
                    tuning_params = []
                    active_layer_indices = []
                    for layer_idx, was_triggered in model.mixing_triggered.items():
                        if was_triggered:
                            tuning_params.append(model.outputs_tuning[layer_idx])
                            active_layer_indices.append(layer_idx)
                    all_params.extend(tuning_params)
                    all_active_layers.append((active_layer_indices, len(tuning_params)))
                all_grads = torch.autograd.grad(total_loss, delta, retain_graph=False, create_graph=False)
                # gradient of input
                grad_x = all_grads[0]
                # update each model's feature perturbations
                current_idx = 1  # start from 1 because 0 is input's gradient
                for model_idx, (active_indices, num_params) in enumerate(all_active_layers):
                    model = models[model_idx]
                    model_grads = all_grads[current_idx:current_idx + num_params]

                    for layer_idx, grad in zip(active_indices, model_grads):
                        model.outputs_tuning[layer_idx] = (model.outputs_tuning[layer_idx] - grad).detach().requires_grad_(True)

                    current_idx += num_params
            else:
                grad_x = torch.autograd.grad(total_loss, delta, retain_graph=False, create_graph=False)[0]
            # Update g
            if 'T' in self.attack_type:
                grad_x = F.conv2d(grad_x, gaussian_kernel, bias=None, stride=1,
                                padding=((ti_kernel_size - 1) // 2, (ti_kernel_size - 1) // 2), groups=3)

            if 'M' in self.attack_type:
                momentum = self.get_momentum(grad_x, momentum)
            else:
                momentum = grad_x

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        if 'F' in self.attack_type:
            for model in models:
                model.remove_hooks()
        torch.cuda.empty_cache()
        
        return delta.detach()


class FeatureTuning(nn.Module):
    def __init__(self, model: nn.Module, input_size, device):
        super().__init__()
        self.device = device
        self.mixup_layer = 'conv_linear_include_last'
        self.prob = 0.1
        self.channelwise = True

        self.model = model
        self.input_size = input_size
        self.record = False

        self.outputs = {}
        self.outputs_tuning = {}  # feature perturbations for tuning
        self.mixing_triggered = {}
        self.forward_hooks = []

        def get_children(model: torch.nn.Module):
            children = list(model.children())
            flattened_children = []
            if children == []:
                if self.mixup_layer == 'conv_linear_no_last' or self.mixup_layer == 'conv_linear_include_last':
                    if type(model) == torch.nn.Conv2d or type(model) == torch.nn.Linear:
                        return model
                    else:
                        return []
                elif self.mixup_layer == 'bn' or self.mixup_layer == 'relu':
                    if type(model) == torch.nn.BatchNorm2d:
                        return model
                    else:
                        return []
                else:
                    if type(model) == torch.nn.Conv2d:
                        return model
                    else:
                        return []
            else:
                for child in children:
                    try:
                        flattened_children.extend(get_children(child))
                    except TypeError:
                        flattened_children.append(get_children(child))
            return flattened_children

        mod_list = get_children(model)
        self.layer_num = len(mod_list)

        for i, m in enumerate(mod_list):
            self.forward_hooks.append(m.register_forward_hook(self.save_outputs_hook(i)))

    def save_outputs_hook(self, layer_idx) -> Callable:
        mix_upper_bound_feature = 0.75
        mix_lower_bound_feature = 0.0
        shuffle_image_feature = 'SelfShuffle'
        blending_mode_feature = 'M'
        mixed_image_type_feature = 'C'
        divisor = 4
        ftm_beta = 0.01

        def hook_fn(module, input, output):
            if type(module) == torch.nn.Linear or output.size()[-1] <= self.input_size // divisor:

                if self.mixup_layer == 'conv_linear_no_last' and (layer_idx + 1) == self.layer_num and type(module) == torch.nn.Linear:
                    pass  # exclude the last fc layer
                else:
                    if layer_idx in self.outputs and self.record == False:  # Feature mixup inference mode
                        c = torch.rand(1).item()
                        # Record selected layers for update
                        self.mixing_triggered[layer_idx] = (c <= self.prob)

                        # If selected, mix the output with clean features and feature perturbations
                        if self.mixing_triggered[layer_idx]:
                            # Configuration for mixing clean features
                            if mixed_image_type_feature == 'A':  # Mix features of other images
                                prev_feature = output.clone().detach()
                            else:  # Mix clean features
                                prev_feature = self.outputs[layer_idx].clone().detach()  # Get stored clean features

                            if shuffle_image_feature == 'SelfShuffle':  # Image-wise feature shuffling
                                idx = torch.randperm(output.shape[0])
                                prev_feature_shuffle = prev_feature[idx].view(prev_feature.size())
                                del idx
                            elif shuffle_image_feature == 'None':
                                prev_feature_shuffle = prev_feature

                            # Random mixing ratio
                            mix_ratio = mix_upper_bound_feature - mix_lower_bound_feature
                            if self.channelwise == True:
                                if output.dim() == 4:
                                    a = (torch.rand(prev_feature.shape[0],
                                                    prev_feature.shape[1]) * mix_ratio + mix_lower_bound_feature).view(
                                        prev_feature.shape[0], prev_feature.shape[1], 1, 1).to(self.device)
                                elif output.dim() == 3:
                                    a = (torch.rand(prev_feature.shape[0],
                                                    prev_feature.shape[1]) * mix_ratio + mix_lower_bound_feature).view(
                                        prev_feature.shape[0], prev_feature.shape[1], 1).to(self.device)
                                else:
                                    a = (torch.rand(prev_feature.shape[0],
                                                    prev_feature.shape[1]) * mix_ratio + mix_lower_bound_feature).view(
                                        prev_feature.shape[0], prev_feature.shape[1]).to(self.device)
                            else:
                                if output.dim() == 4:
                                    a = (torch.rand(prev_feature.shape[0]) * mix_ratio + mix_lower_bound_feature).view(
                                        prev_feature.shape[0], 1, 1, 1).to(self.device)
                                elif output.dim() == 3:
                                    a = (torch.rand(prev_feature.shape[0]) * mix_ratio + mix_lower_bound_feature).view(
                                        prev_feature.shape[0], 1, 1).to(self.device)
                                else:
                                    a = (torch.rand(prev_feature.shape[0]) * mix_ratio + mix_lower_bound_feature).view(
                                        prev_feature.shape[0], 1).to(self.device)

                            if self.mixup_layer == 'relu':
                                output = F.relu(output, inplace=True)

                            # mix with feature perturbations
                            output_flat = output.detach().view(output.size(0), -1)  # [B, *]
                            tuning_flat = self.outputs_tuning[layer_idx].detach().view(output.size(0), -1)  # [B, *]

                            output_norm = output_flat.norm(dim=1)  # [B]
                            tuning_norm = tuning_flat.norm(dim=1)  # [B]
                            scale = ftm_beta * output_norm / (tuning_norm + 1e-7)  # [B]

                            for _ in range(len(output.shape) - 1):
                                scale = scale.unsqueeze(-1)

                            output1 = output + self.outputs_tuning[layer_idx] * scale

                            # mix with clean features
                            if blending_mode_feature == 'M':  # Linear interpolation
                                output2 = (1 - a) * output1 + a * prev_feature_shuffle
                            elif blending_mode_feature == 'A':  # Addition
                                output2 = output1 + a * prev_feature_shuffle

                            return output2
                        # If not selected, mix the output with feature perturbations
                        else:
                            output_flat = output.detach().view(output.size(0), -1)  # [B, *]
                            tuning_flat = self.outputs_tuning[layer_idx].detach().view(output.size(0), -1)  # [B, *]

                            output_norm = output_flat.norm(dim=1)  # [B]
                            tuning_norm = tuning_flat.norm(dim=1)  # [B]
                            scale = ftm_beta * output_norm / (tuning_norm + 1e-7)  # [B]

                            for _ in range(len(output.shape) - 1):
                                scale = scale.unsqueeze(-1)

                            output_perturbed = output + self.outputs_tuning[layer_idx].detach() * scale

                            return output_perturbed 

                    elif self.record == True:  # Feature recording mode
                        self.outputs[layer_idx] = output.clone().detach()
                        # Learnable feature perturbations
                        self.outputs_tuning[layer_idx] = torch.zeros_like(output).clone().detach().requires_grad_(True)
                        self.mixing_triggered[layer_idx] = False
                        return

        return hook_fn

    def start_feature_record(self):
        self.record = True

    def end_feature_record(self):
        self.record = False

    def remove_hooks(self):
        for fh in self.forward_hooks:
            fh.remove()
        del self.outputs
        del self.outputs_tuning
        del self.mixing_triggered

    def forward(self, x: Tensor) -> Tensor:
        # Clear mixing triggers at the start of each forward pass
        self.mixing_triggered = {}
        return self.model(x)



class LogitLoss(nn.Module):
    def __init__(self, labels, targeted=True):
        super(LogitLoss, self).__init__()
        self.labels = labels
        self.targeted = targeted
        self.labels.requires_grad = False

    def forward(self, logits):
        real = logits.gather(1, self.labels.unsqueeze(1)).squeeze(1)
        logit_dists = (1 * real)
        loss = logit_dists.sum()
        if self.targeted == False:
            loss = -loss
        return loss


def DI(X_in, prob):
    prob = 0.7
    img_width = X_in.size()[-1]  # B X C X H X W
    enlarged_img_width = int(img_width * 330. / 299.)
    rnd = np.random.randint(img_width, enlarged_img_width, size=1)[0]
    h_rem = enlarged_img_width - rnd
    w_rem = enlarged_img_width - rnd
    pad_top = np.random.randint(0, h_rem, size=1)[0]
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem, size=1)[0]
    pad_right = w_rem - pad_left
    c = np.random.rand(1)
    if c <= prob:
        X_out = F.pad(F.interpolate(X_in, size=(rnd, rnd)), (pad_left, pad_top, pad_right, pad_bottom), mode='constant', value=0)
        return X_out
    else:
        return X_in


def RDI(x_adv):
    x_di = x_adv
    img_width = x_adv.size()[-1]
    enlarged_img_width = int(img_width * 340. / 299.)
    di_pad_amount = enlarged_img_width - img_width
    di_pad_value = 0
    ori_size = x_di.shape[-1]
    rnd = int(torch.rand(1) * di_pad_amount) + ori_size
    x_di = transforms.Resize((rnd, rnd), interpolation=InterpolationMode.NEAREST)(x_di)
    pad_max = ori_size + di_pad_amount - rnd
    pad_left = int(torch.rand(1) * pad_max)
    pad_right = pad_max - pad_left
    pad_top = int(torch.rand(1) * pad_max)
    pad_bottom = pad_max - pad_top
    x_di = F.pad(x_di, (pad_left, pad_right, pad_top, pad_bottom), 'constant', di_pad_value)
    if img_width > 64:  # For the CIFAR-10 dataset, we skip the image size reduction.
        x_di = transforms.Resize((ori_size, ori_size), interpolation=InterpolationMode.NEAREST)(x_di)
    return x_di


def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel
