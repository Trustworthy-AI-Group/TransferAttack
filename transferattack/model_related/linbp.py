import torch
import torch.nn.functional as F
import torch.nn as nn
from ..utils import *
from ..gradient.mifgsm import MIFGSM

class LinBP(MIFGSM):
    """
    LinBP Attack
    'Backpropagating Linearly Improves Transferability of Adversarial Examples (NeurIPS 2020)' (https://proceedings.neurips.cc/paper_files/paper/2020/file/00e26af6ac3b1c1c49d7c3d79c60d000-Paper.pdf)

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
        linbp_layer: feature layer to launch the attack.

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., linbp_layer=3_1 for resnet50
    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/linbp/resnet50 --attack linbp --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/linbp/resnet50 --eval
    """

    def __init__(self, model_name, epsilon=16.0 / 255, alpha=1.6 / 255, epoch=300, decay=1.,
                 targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None, attack ='LinBP', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.linbp_layer = '3_1'
        self.sgm_lambda = 1.0

    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

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

        momentum = 0
        for _ in range(self.epoch):
            # Obtain the output
            att_out, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls = linbp_forw_resnet50(self.model, data+delta, True,
                                                                                                self.linbp_layer)
            # Calculate the loss
            pred = torch.argmax(att_out, dim=1).view(-1)

            loss = nn.CrossEntropyLoss()(att_out, label.cuda())

            self.model.zero_grad()

            # Calculate the gradients
            grad = linbp_backw_resnet50(delta, loss, conv_out_ls, ori_mask_ls, relu_out_ls, conv_input_ls, xp=self.sgm_lambda)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            #Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()

def linbp_forw_resnet50(model, x, do_linbp, linbp_layer):
    jj = int(linbp_layer.split('_')[0])
    kk = int(linbp_layer.split('_')[1])
    x = model[0](x)
    x = model[1].conv1(x)
    x = model[1].bn1(x)
    x = model[1].relu(x)
    x = model[1].maxpool(x)
    ori_mask_ls = []
    conv_out_ls = []
    relu_out_ls = []
    conv_input_ls = []
    def layer_forw(jj, kk, jj_now, kk_now, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp):
        if jj < jj_now:
            x, ori_mask, conv_out, relu_out, conv_in = block_func(mm, x, linbp=True)
            ori_mask_ls.append(ori_mask)
            conv_out_ls.append(conv_out)
            relu_out_ls.append(relu_out)
            conv_input_ls.append(conv_in)
        elif jj == jj_now:
            if kk_now >= kk:
                x, ori_mask, conv_out, relu_out, conv_in = block_func(mm, x, linbp=True)
                ori_mask_ls.append(ori_mask)
                conv_out_ls.append(conv_out)
                relu_out_ls.append(relu_out)
                conv_input_ls.append(conv_in)
            else:
                x, _, _, _, _ = block_func(mm, x, linbp=False)
        else:
            x, _, _, _, _ = block_func(mm, x, linbp=False)
        return x, ori_mask_ls
    for ind, mm in enumerate(model[1].layer1):
        x, ori_mask_ls = layer_forw(jj, kk, 1, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp)
    for ind, mm in enumerate(model[1].layer2):
        x, ori_mask_ls = layer_forw(jj, kk, 2, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp)
    for ind, mm in enumerate(model[1].layer3):
        x, ori_mask_ls = layer_forw(jj, kk, 3, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp)
    for ind, mm in enumerate(model[1].layer4):
        x, ori_mask_ls = layer_forw(jj, kk, 4, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls, do_linbp)
    x = model[1].avgpool(x)
    x = torch.flatten(x, 1)
    x = model[1].fc(x)
    return x, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls

def block_func(block, x, linbp):
    identity = x
    conv_in = x+0
    out = block.conv1(conv_in)
    out = block.bn1(out)
    out_0 = out + 0
    if linbp:
        out = linbp_relu(out_0)
    else:
        out = block.relu(out_0)
    ori_mask_0 = out.data.bool().int()

    out = block.conv2(out)
    out = block.bn2(out)
    out_1 = out + 0
    if linbp:
        out = linbp_relu(out_1)
    else:
        out = block.relu(out_1)
    ori_mask_1 = out.data.bool().int()

    out = block.conv3(out)
    out = block.bn3(out)

    if block.downsample is not None:
        identity = block.downsample(identity)
    identity_out = identity + 0
    x_out = out + 0


    out = identity_out + x_out
    out = block.relu(out)
    ori_mask_2 = out.data.bool().int()
    return out, (ori_mask_0, ori_mask_1, ori_mask_2), (identity_out, x_out), (out_0, out_1), (0, conv_in)


def linbp_relu(x):
    x_p = F.relu(-x)
    x = x + x_p.data
    return x

def linbp_backw_resnet50(img, loss, conv_out_ls, ori_mask_ls, relu_out_ls, conv_input_ls, xp):
    for i in range(-1, -len(conv_out_ls)-1, -1):
        if i == -1:
            grads = torch.autograd.grad(loss, conv_out_ls[i])
        else:
            grads = torch.autograd.grad((conv_out_ls[i+1][0], conv_input_ls[i+1][1]), conv_out_ls[i], grad_outputs=(grads[0], main_grad_norm))
        normal_grad_2 = torch.autograd.grad(conv_out_ls[i][1], relu_out_ls[i][1], grads[1]*ori_mask_ls[i][2],retain_graph=True)[0]
        normal_grad_1 = torch.autograd.grad(relu_out_ls[i][1], relu_out_ls[i][0], normal_grad_2 * ori_mask_ls[i][1], retain_graph=True)[0]
        normal_grad_0 = torch.autograd.grad(relu_out_ls[i][0], conv_input_ls[i][1], normal_grad_1 * ori_mask_ls[i][0], retain_graph=True)[0]
        del normal_grad_2, normal_grad_1
        main_grad = torch.autograd.grad(conv_out_ls[i][1], conv_input_ls[i][1], grads[1])[0]
        alpha = normal_grad_0.norm(p=2, dim = (1,2,3), keepdim = True) / main_grad.norm(p=2,dim = (1,2,3), keepdim=True)
        main_grad_norm = xp * alpha * main_grad
    input_grad = torch.autograd.grad((conv_out_ls[0][0], conv_input_ls[0][1]), img, grad_outputs=(grads[0], main_grad_norm))
    return input_grad[0].data
