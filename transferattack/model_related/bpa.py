import torch
import torch.nn.functional as F

from ..utils import *
from ..gradient.mifgsm import MIFGSM


class BPA(MIFGSM):
    """
    BPA: Backward Propagation Attack
    'Rethinking the Backward Propagation for Adversarial Transferability (NeurIPS 2023)'(https://arxiv.org/abs/2306.12685)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        bpa_layer (str): start bpa from this layer.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=8/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., bpa_layer=3_1 for resnet50

    Example script:
        python main.py --attack=bpa --output_dir adv_data/bpa/resnet18
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., bpa_layer='3_1', targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None, attack='BPA', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.bpa_layer = bpa_layer

    def forward(self, data, label, **kwargs):
        """
        The BPA attack procedure

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

        momentum=0
        for _ in range(self.epoch):
            # Obtain the logits
            logits = bpa_forw_resnet(self.model, data+delta, self.bpa_layer)

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Get the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update the delta
            delta = self.update_delta(delta, data, momentum, self.alpha)
        
        return delta.detach()



class BPA_MaxPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, kernel_size, stride=None, padding=0, temp_coef=10):
        # forward uses original maxpool function
        outputs = F.max_pool2d(inputs, kernel_size, stride, padding)
        ctx.save_for_backward(inputs, outputs)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.temp_coef = temp_coef
        return outputs

    @staticmethod
    def backward(ctx, grad_out):
        inputs, outputs = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        padding = ctx.padding
        temp_coef = ctx.temp_coef
        n_in, c_in, _, _ = inputs.shape
        _, _, h_out, w_out = outputs.shape

        patches = F.unfold(inputs, kernel_size=kernel_size, padding=padding, stride=stride) #[N,C*(kernel_size**2), patch_num], patch_num = h_out * w_out
        patches = torch.reshape(patches, (n_in, c_in, kernel_size**2, h_out, w_out))
        softmax_patches = F.softmax(patches * temp_coef, dim=2)
        grad_out = torch.tile(grad_out[:,:,None,:,:], (1,1,kernel_size**2,1,1))
        grad_in = grad_out * softmax_patches
        grad_in = torch.reshape(grad_in, (n_in, c_in*(kernel_size**2), -1))
        grad_in = F.fold(grad_in, inputs.shape[2:], kernel_size=kernel_size, padding=padding, stride=stride)
        return grad_in, None, None, None, None

class BPA_ReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        result = F.relu(inputs)
        ctx.save_for_backward(inputs)
        return result

    @staticmethod
    def backward(ctx, grad_out):
        inputs, =ctx.saved_tensors
        inputs_grad = torch.sigmoid(inputs) * (1 + inputs * (1 - torch.sigmoid(inputs)))
        return grad_out * inputs_grad

def bpa_maxpool(inputs, kernel_size, stride, padding, temp_coef):
    return BPA_MaxPool.apply(inputs, kernel_size, stride, padding, temp_coef)

def bpa_relu(inputs):
    return BPA_ReLU.apply(inputs)

def bpa_forw_resnet(model, x, bpa_layer):
    jj = int(bpa_layer.split('_')[0])
    kk = int(bpa_layer.split('_')[1])

    x = model[0](x)
    x = model[1].conv1(x)
    x = model[1].bn1(x)
    x = model[1].relu(x)
    x = bpa_maxpool(x, kernel_size=3, stride=2, padding=1, temp_coef=10)
    
    def layer_forw(jj, kk, jj_now, kk_now, x, mm):
        if jj < jj_now:
            x = block_func(mm, x, do_bpa=True)
        elif jj == jj_now:
            if kk_now >= kk:
                x = block_func(mm, x, do_bpa=True)
            else:
                x = block_func(mm, x, do_bpa=False)
        else:
            x = block_func(mm, x, do_bpa=False)
        return x
    
    for ind, mm in enumerate(model[1].layer1):
        x = layer_forw(jj, kk, 1, ind, x, mm)
    for ind, mm in enumerate(model[1].layer2):
        x = layer_forw(jj, kk, 2, ind, x, mm)
    for ind, mm in enumerate(model[1].layer3):
        x = layer_forw(jj, kk, 3, ind, x, mm)
    for ind, mm in enumerate(model[1].layer4):
        x = layer_forw(jj, kk, 4, ind, x, mm)
    
    x = model[1].avgpool(x)
    x = torch.flatten(x, 1)
    x = model[1].fc(x)
    return x

def block_func(block, x, do_bpa):
    identity = x

    out = block.conv1(x)
    out = block.bn1(out)
    if do_bpa:
        out = bpa_relu(out)
    else:
        out = block.relu(out)

    out = block.conv2(out)
    out = block.bn2(out)

    if block.downsample is not None:
        identity = block.downsample(identity)

    out += identity
    out = block.relu(out)
    return out
