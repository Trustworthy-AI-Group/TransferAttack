# Additive Power-of-Two Quantization: An Efficient Non-uniform Discretization For Neural Networks
# Yuhang Li, Xin Dong, Wei Wang
# International Conference on Learning Representations (ICLR), 2020.


import torch.nn as nn
import torch
import torch.nn.functional as F


# this function construct an additive pot quantization levels set, with clipping threshold = 1,
def build_power_value(B=2, additive=True):
    base_a = [0.]
    base_b = [0.]
    base_c = [0.]
    if additive:
        if B == 2:
            for i in range(3):
                base_a.append(2 ** (-i - 1))
        elif B == 4:
            for i in range(3):
                base_a.append(2 ** (-2 * i - 1))
                base_b.append(2 ** (-2 * i - 2))
        elif B == 6:
            for i in range(3):
                base_a.append(2 ** (-3 * i - 1))
                base_b.append(2 ** (-3 * i - 2))
                base_c.append(2 ** (-3 * i - 3))
        elif B == 3:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-i - 1))
                else:
                    base_b.append(2 ** (-i - 1))
                    base_a.append(2 ** (-i - 2))
        elif B == 5:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-2 * i - 1))
                    base_b.append(2 ** (-2 * i - 2))
                else:
                    base_c.append(2 ** (-2 * i - 1))
                    base_a.append(2 ** (-2 * i - 2))
                    base_b.append(2 ** (-2 * i - 3))
        else:
            pass
    else:
        for i in range(2 ** B - 1):
            base_a.append(2 ** (-i - 1))
    values = []
    for a in base_a:
        for b in base_b:
            for c in base_c:
                values.append((a + b + c))
    values = torch.Tensor(list(set(values)))
    values = values.mul(1.0 / torch.max(values))
    return values


def gradient_scale(x, scale):
    yout = x
    ygrad = x * scale
    y = (yout - ygrad).detach() + ygrad
    return y


def apot_quantization(tensor, alpha, proj_set, is_weight=True, grad_scale=None):
    def power_quant(x, value_s):
        if is_weight:
            shape = x.shape
            xhard = x.view(-1)
            sign = x.sign()
            value_s = value_s.type_as(x)
            xhard = xhard.abs()
            idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
            xhard = value_s[idxs].view(shape).mul(sign)
            xhard = xhard
        else:
            shape = x.shape
            xhard = x.view(-1)
            value_s = value_s.type_as(x)
            xhard = xhard
            idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
            xhard = value_s[idxs].view(shape)
            xhard = xhard
        xout = (xhard - x).detach() + x
        return xout

    if grad_scale:
        alpha = gradient_scale(alpha, grad_scale)
    data = tensor / alpha
    if is_weight:
        data = data.clamp(-1, 1)
        data_q = power_quant(data, proj_set)
        data_q = data_q * alpha
    else:
        data = data.clamp(0, 1)
        data_q = power_quant(data, proj_set)
        data_q = data_q * alpha
    return data_q


def uq_with_calibrated_graditens(grad_scale=None):
    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input.div_(alpha)
            input_c = input.clamp(min=-1, max=1)
            input_q = input_c.round()
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)  # rescale to the original range
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()  # calibration: grad for weights will not be clipped
            input, input_q = ctx.saved_tensors
            i = (input.abs() > 1.).float()
            sign = input.sign()
            grad_alpha = (grad_output * (sign * i + (input_q - input) * (1 - i))).sum()
            if grad_scale:
                grad_alpha = grad_alpha * grad_scale
            return grad_input, grad_alpha

    return _uq().apply


def uniform_quantization(tensor, alpha, bit, is_weight=True, grad_scale=None):
    if grad_scale:
        alpha = gradient_scale(alpha, grad_scale)
    data = tensor / alpha
    if is_weight:
        data = data.clamp(-1, 1)
        data = data * (2 ** (bit - 1) - 1)
        data_q = (data.round() - data).detach() + data
        data_q = data_q / (2 ** (bit - 1) - 1) * alpha
    else:
        data = data.clamp(0, 1)
        data = data * (2 ** bit - 1)
        data_q = (data.round() - data).detach() + data
        data_q = data_q / (2 ** bit - 1) * alpha
    return data_q


class QuantConv2d(nn.Conv2d):
    """Generates quantized convolutional layers.

    args:
        bit(int): bitwidth for the quantization,
        power(bool): (A)PoT or Uniform quantization
        additive(float): Use additive or vanilla PoT quantization

    procedure:
        1. determine if the bitwidth is illegal
        2. if using PoT quantization, then build projection set. (For 2-bit weights quantization, PoT = Uniform)
        3. generate the clipping thresholds

    forward:
        1. if bit = 32(full precision), call normal convolution
        2. if not, first normalize the weights and then quantize the weights and activations
        3. if bit = 2, apply calibrated gradients uniform quantization to weights
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, bit=5, power=True, additive=True, grad_scale=None):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias)
        self.layer_type = 'QuantConv2d'
        self.bit = bit
        self.power = power
        self.grad_scale = grad_scale
        self.a_quant = True   # stochastic depth
        if power:
            if self.bit > 2:
                self.proj_set_weight = build_power_value(B=self.bit - 1, additive=additive)
            self.proj_set_act = build_power_value(B=self.bit, additive=additive)
        self.act_alpha = torch.nn.Parameter(torch.tensor(6.0))
        self.weight_alpha = torch.nn.Parameter(torch.tensor(3.0))

    def forward(self, x):
        # weight normalization
        # print('self.a_quant: {}'.format(self.a_quant))
        if self.bit == 32:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups) + (self.act_alpha + self.weight_alpha) * 0.0
        else:
            mean = self.weight.mean()
            std = self.weight.std()
            weight = self.weight.add(-mean).div(std)
            if self.power:
                if self.bit > 2:
                    weight = apot_quantization(weight, self.weight_alpha, self.proj_set_weight, True, self.grad_scale)
                else:
                    weight = uq_with_calibrated_graditens(self.grad_scale)(weight, self.weight_alpha)
                if self.a_quant:
                    x = apot_quantization(x, self.act_alpha, self.proj_set_act, False, self.grad_scale)
                else:
                    x = x + self.act_alpha * 0.0
            else:
                if self.bit > 2:
                    weight = uniform_quantization(weight, self.weight_alpha, self.bit, True, self.grad_scale)
                else:
                    weight = uq_with_calibrated_graditens(self.grad_scale)(weight, self.weight_alpha)
                if self.a_quant:
                    x  = uniform_quantization(x, self.act_alpha, self.bit, False, self.grad_scale)
                else:
                    x = x
            return F.conv2d(x, weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

    def show_params(self):
        if self.bit != 32:
            wgt_alpha = round(self.weight_alpha.data.item(), 3)
            act_alpha = round(self.act_alpha.data.item(), 3)
            print('clipping threshold weight alpha: {:2f}, activation alpha: {:2f}'.format(wgt_alpha, act_alpha))


# 8-bit quantization for the first and the last layer
class first_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(first_conv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                         bias)
        self.layer_type = 'FConv2d'

    def forward(self, x):
        max = self.weight.data.max()
        weight_q = self.weight.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q - self.weight).detach() + self.weight
        return F.conv2d(x, weight_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class last_fc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(last_fc, self).__init__(in_features, out_features, bias)
        self.layer_type = 'LFC'

    def forward(self, x):
        max = self.weight.data.max()
        weight_q = self.weight.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q - self.weight).detach() + self.weight
        return F.linear(x, weight_q, self.bias)