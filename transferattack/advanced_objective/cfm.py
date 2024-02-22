import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.transforms import InterpolationMode

from ..utils import *
from ..attack import Attack
from typing import Callable
import scipy.stats as st


class CFM(Attack):
    """
    Clean Feature Mixup Attack
    'Introducing Competition to Boost the Transferability of Targeted Adversarial Examples through Clean Feature Mixup (CVPR 2023) (https://arxiv.org/abs/2305.14846)'

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
        feature_layer: feature layer to launch the attack.

    Official arguments:
        epsilon=0.07, alpha=epsilon/epoch=0.007, epoch=300, decay=1.

    Example script:
        python main.py --attack=cfm --output_dir adv_data/cfm/resnet18 --targeted
    """
    def __init__(self, model_name, epsilon=16/255, alpha=2/255, epoch=300, decay=1., targeted=True, random_start=False, norm='linfty', loss='crossentropy', device=None, attack='CFM', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.model = FeatureMixup(self.model)
        self.kernel_type = 'gaussian'
        self.kernel_size = 5
        self.kernel = self.generate_kernel(self.kernel_type, self.kernel_size)

    def generate_kernel(self, kernel_type, kernel_size, nsig=3):
        """
        Generate the gaussian/uniform/linear kernel

        Arguments:
            kernel_type (str): the method for initilizing the kernel
            kernel_size (int): the size of kernel
        """
        if kernel_type.lower() == 'gaussian':
            x = np.linspace(-nsig, nsig, kernel_size)
            kern1d = st.norm.pdf(x)
            kernel_raw = np.outer(kern1d, kern1d)
            kernel = kernel_raw / kernel_raw.sum()
        elif kernel_type.lower() == 'uniform':
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        elif kernel_type.lower() == 'linear':
            kern1d = 1 - np.abs(np.linspace((-kernel_size+1)//2, (kernel_size-1)//2, kernel_size)/(kernel_size**2))
            kernel_raw = np.outer(kern1d, kern1d)
            kernel = kernel_raw / kernel_raw.sum()
        else:
            raise Exception("Unspported kernel type {}".format(kernel_type))

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return torch.from_numpy(stack_kernel.astype(np.float32)).to(self.device)

    def get_grad(self, loss, delta, **kwargs):
        """
        Overridden for TIM attack.
        """
        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
        grad = F.conv2d(grad, self.kernel, stride=1, padding='same', groups=3)
        return grad

    def get_loss(self, logits, label): # logits loss
        real = logits.gather(1, label.unsqueeze(1)).squeeze(1)
        logit_dists = (1 * real)
        loss = logit_dists.sum()
        if self.targeted==False:
            loss=-loss
        return loss

    def transform(self, data, **kwargs):
        x_di = data
        img_width=data.size()[-1] # B X C X H X W
        enlarged_img_width=int(img_width*340./299.)
        di_pad_amount=enlarged_img_width-img_width
        di_pad_value=0
        ori_size = x_di.shape[-1]
        rnd = int(torch.rand(1) * di_pad_amount) + ori_size
        x_di = transforms.Resize((rnd, rnd), interpolation=InterpolationMode.NEAREST)(x_di)
        pad_max = ori_size + di_pad_amount - rnd
        pad_left = int(torch.rand(1) * pad_max)
        pad_right = pad_max - pad_left
        pad_top = int(torch.rand(1) * pad_max)
        pad_bottom = pad_max - pad_top
        x_di = F.pad(x_di, (pad_left, pad_right, pad_top, pad_bottom), 'constant', di_pad_value)
        if img_width>64: # For the CIFAR-10 dataset, we skip the image size reduction.
            x_di = transforms.Resize((ori_size, ori_size), interpolation=InterpolationMode.NEAREST)(x_di)
        return x_di

    def forward(self, data, label, **kwargs):
        """
        The CFM attack procedure

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
        for _ in range(self.epoch):
            if _ == 0:
                # Store clean feature
                with torch.no_grad():
                    self.model.start_feature_record() # Set feature recoding mode
                    self.model(data) # Feature recording
                    self.model.end_feature_record() # Set feature mixup inference mode
                continue
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()

exp_configuration = {
        'targeted':True,
        'epsilon':16,
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'p':1.,  # "prob for DI and RE"

        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,

        'channelwise':True,
        'mixup_layer':'conv_linear_include_last',
    }

# Clean Feature Mixup
class FeatureMixup(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        exp_settings=exp_configuration
        self.mixup_layer=exp_settings['mixup_layer']
        self.prob=exp_settings['mix_prob']
        self.channelwise=exp_settings['channelwise']

        self.model = model
        self.input_size=img_height
        self.record=False


        self.outputs={}
        self.forward_hooks=[]

        def get_children(model: torch.nn.Module):
            children = list(model.children())
            flattened_children = []
            if children == []:
                # if model is the last child
                if self.mixup_layer=='conv_linear_no_last' or self.mixup_layer=='conv_linear_include_last':
                    if type(model)==torch.nn.Conv2d or type(model)==torch.nn.Linear:
                        return model
                    else:
                        return []
                elif self.mixup_layer=='bn' or self.mixup_layer=='relu':
                    if type(model)==torch.nn.BatchNorm2d:
                        return model
                    else:
                        return []
                else:
                    if type(model)==torch.nn.Conv2d:
                        return model
                    else:
                        return []
            else:
                # look for children
                for child in children:
                        try:
                            flattened_children.extend(get_children(child))
                        except TypeError:
                            flattened_children.append(get_children(child))
            return flattened_children
        mod_list=get_children(model)
        self.layer_num=len(mod_list)
        #print(mod_list)

        for i, m in enumerate(mod_list):
            self.forward_hooks.append(m.register_forward_hook(self.save_outputs_hook(i)))

    def save_outputs_hook(self, layer_idx) -> Callable:
        # Load experiment configurations
        exp_settings=exp_configuration
        mix_upper_bound_feature=exp_settings['mix_upper_bound_feature']
        mix_lower_bound_feature=exp_settings['mix_lower_bound_feature']
        shuffle_image_feature=exp_settings['shuffle_image_feature']
        blending_mode_feature=exp_settings['blending_mode_feature']
        mixed_image_type_feature=exp_settings['mixed_image_type_feature']
        divisor=exp_settings['divisor']


        def hook_fn(module, input, output):
            if type(module)==torch.nn.Linear or output.size()[-1]<=self.input_size//divisor:

                if self.mixup_layer=='conv_linear_no_last' and (layer_idx+1)==self.layer_num and type(module)==torch.nn.Linear:
                    pass # exclude the last fc layer
                else:
                    if layer_idx in self.outputs and self.record==False: # Feature mixup inference mode
                        c = torch.rand(1).item()
                        if c <= self.prob: # With probability p

                            if mixed_image_type_feature=='A': # Mix features of other images
                                prev_feature=output.clone().detach()
                            else: # Mix clean features
                                prev_feature=self.outputs[layer_idx].clone().detach() # Get stored clean features


                            if shuffle_image_feature=='SelfShuffle': # Image-wise feature shuffling
                                idx = torch.randperm(output.shape[0])
                                prev_feature_shuffle = prev_feature[idx].view(prev_feature.size())
                                del idx
                            elif shuffle_image_feature=='None':
                                prev_feature_shuffle=prev_feature

                            # Random mixing ratio
                            mix_ratio=mix_upper_bound_feature-mix_lower_bound_feature
                            if self.channelwise==True:
                                if output.dim()==4:
                                    a = (torch.rand(prev_feature.shape[0],prev_feature.shape[1])*mix_ratio+mix_lower_bound_feature).view(prev_feature.shape[0],prev_feature.shape[1],1,1).cuda()
                                elif output.dim()==3:
                                    a = (torch.rand(prev_feature.shape[0],prev_feature.shape[1])*mix_ratio+mix_lower_bound_feature).view(prev_feature.shape[0],prev_feature.shape[1],1).cuda()
                                else:
                                    a = (torch.rand(prev_feature.shape[0],prev_feature.shape[1])*mix_ratio+mix_lower_bound_feature).view(prev_feature.shape[0],prev_feature.shape[1]).cuda()
                            else:
                                if output.dim()==4:
                                    a = (torch.rand(prev_feature.shape[0])*mix_ratio+mix_lower_bound_feature).view(prev_feature.shape[0],1,1,1).cuda()
                                elif output.dim()==3:
                                    a = (torch.rand(prev_feature.shape[0])*mix_ratio+mix_lower_bound_feature).view(prev_feature.shape[0],1,1).cuda()
                                else:
                                    a = (torch.rand(prev_feature.shape[0])*mix_ratio+mix_lower_bound_feature).view(prev_feature.shape[0],1).cuda()
                            # Blending
                            if self.mixup_layer=='relu':
                                output=F.relu(output,inplace=True)

                            if blending_mode_feature=='M': # Linear interpolation
                                output2=(1-a)*output+a*prev_feature_shuffle
                            elif blending_mode_feature=='A': # Addition
                                output2=output+a*prev_feature_shuffle

                            return output2
                        else:
                            return output

                    elif self.record==True: # Feature recording mode
                        self.outputs[layer_idx]= output.clone().detach()
                        return
        return hook_fn
    def start_feature_record(self):
        self.record=True
    def end_feature_record(self):
        self.record=False

    def remove_hooks(self):
        for fh in self.forward_hooks:
            fh.remove()
        del self.outputs

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)