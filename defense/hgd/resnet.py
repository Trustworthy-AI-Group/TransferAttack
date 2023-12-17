import torch
from torch import nn
from collections import OrderedDict
import numpy as np

class Conv(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    def __init__(self, n_in, n_out, stride = 1, expansion = 4):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(n_out)
        self.conv3 = nn.Conv2d(n_out, n_out * expansion, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(n_out * expansion)

        self.downsample = None
        if stride != 1 or n_in != n_out * expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(n_in, n_out * expansion, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(n_out * expansion))

        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Denoise(nn.Module):
    def __init__(self, h_in, w_in, block, fwd_in, fwd_out, num_fwd, back_out, num_back):
        super(Denoise, self).__init__()

        h, w = [], []
        for i in range(len(num_fwd)):
            h.append(h_in)
            w.append(w_in)
            h_in = int(np.ceil(float(h_in) / 2))
            w_in = int(np.ceil(float(w_in) / 2))

        if block is Bottleneck:
            expansion = 4
        else:
            expansion = 1
        
        fwd = []
        n_in = fwd_in
        for i in range(len(num_fwd)):
            group = []
            for j in range(num_fwd[i]):
                if j == 0:
                    if i == 0:
                        stride = 1
                    else:
                        stride = 2
                    group.append(block(n_in, fwd_out[i], stride = stride))
                else:
                    group.append(block(fwd_out[i] * expansion, fwd_out[i]))
            n_in = fwd_out[i] * expansion
            fwd.append(nn.Sequential(*group))
        self.fwd = nn.ModuleList(fwd)

        upsample = []
        back = []
        n_in = (fwd_out[-2] + fwd_out[-1]) * expansion
        for i in range(len(num_back) - 1, -1, -1):
            upsample.insert(0, nn.Upsample(size = (h[i], w[i]), mode = 'bilinear'))
            group = []
            for j in range(num_back[i]):
                if j == 0:
                    group.append(block(n_in, back_out[i]))
                else:
                    group.append(block(back_out[i] * expansion, back_out[i]))
            if i != 0:
                n_in = (back_out[i] + fwd_out[i - 1]) * expansion
            back.insert(0, nn.Sequential(*group))
        self.upsample = nn.ModuleList(upsample)
        self.back = nn.ModuleList(back)

        self.final = nn.Conv2d(back_out[0] * expansion, fwd_in, kernel_size = 1, bias = False)

    def forward(self, x):
        out = x
        outputs = []
        for i in range(len(self.fwd)):
            out = self.fwd[i](out)
            if i != len(self.fwd) - 1:
                outputs.append(out)
        
        for i in range(len(self.back) - 1, -1, -1):
            out = self.upsample[i](out)
            out = torch.cat((out, outputs[i]), 1)
            out = self.back[i](out)
        out = self.final(out)
        out += x
        return out

class Null(nn.Module):
    def __init__(self):
        super(Null, self).__init__()

    def forward(self, x):
        return x

class ResNet(nn.Module):
    def __init__(self, net_type, input_size, denoise_block, fwd_out, num_fwd, back_out, num_back, num_classes = 1000):
        super(ResNet, self).__init__()

        if net_type == 'resnet50':
            block = Bottleneck
            num_blocks = [3, 4, 6, 3]
        elif net_type == 'resnet152':
            block = Bottleneck
            num_blocks = [3, 8, 36, 3]
        else:
            exit('Wrong base net type')

        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        if block is Bottleneck:
            expansion  = 4
        else:
            expansion = 1
        n_in = 64
        n_out = [64, 128, 256, 512]

        for i in range(len(num_blocks)):
            group = []
            for j in range(num_blocks[i]):
                if j == 0:
                    if i == 0:
                        stride = 1
                    else:
                        stride = 2
                    group.append(block(n_in, n_out[i], stride = stride))
                else:
                    group.append(block(n_out[i] * expansion, n_out[i]))
            n_in = n_out[i] * expansion
            group = nn.Sequential(*group)
            setattr(self, 'layer' + str(i + 1), group)
        
        self.fc = nn.Linear(512 * expansion, num_classes)

        h_in, w_in = [], []
        h, w, = input_size
        for i in range(5):
            h_in.append(h)
            w_in.append(w)
            h = int(np.ceil(float(h) / 2))
            w = int(np.ceil(float(w) / 2))
            if i == 0:
                h = int(np.ceil(float(h) / 2))
                w = int(np.ceil(float(w) / 2))

        denoise = []
        n_in = [3, 64 * expansion, 128 * expansion, 256 * expansion, 512 * expansion]
        block = denoise_block
        for i in range(len(block)):
            if block[i] is None:
                denoise.append(Null())
            else:
                denoise.append(Denoise(h_in[i], w_in[i], block[i], n_in[i], fwd_out[i], num_fwd[i], back_out[i], num_back[i]))
        self.denoise = nn.ModuleList(denoise)
        self.denoise_relu = nn.ReLU()
 
    def forward(self, x, defense = False):
        outputs = []

        out = x
        if defense:
            out = self.denoise[0](out)
        outputs.append(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        for i in range(1, 5):
            out = getattr(self, 'layer' + str(i))(out)
            if defense:
                out = self.denoise[i](out)
                out = self.denoise_relu(out)
            outputs.append(out)

        size = out.size()
        out = out.view(size[0], size[1], -1)
        out = out.mean(2)
        out = self.fc(out)
        outputs.append(out)

        return outputs

class DenoiseLoss(nn.Module):
    def __init__(self, n, hard_mining = 0, norm = False):
        super(DenoiseLoss, self).__init__()
        self.n = n
        assert(hard_mining >= 0 and hard_mining <= 1)
        self.hard_mining = hard_mining
        self.norm = norm

    def forward(self, x, y):
        loss = torch.pow(torch.abs(x - y), self.n) / self.n
        if self.hard_mining > 0:
            loss = loss.view(-1)
            k = int(loss.size(0) * self.hard_mining)
            loss, idcs = torch.topk(loss, k)
            y = y.view(-1)[idcs]
            
        loss = loss.mean()
        if self.norm:
            norm = torch.pow(torch.abs(y), self.n)
            norm = norm.data.mean()
            loss = loss / norm
        return loss

class Loss(nn.Module):
    def __init__(self, n, hard_mining = 0, norm = False):
        super(Loss, self).__init__()
        self.loss = DenoiseLoss(n, hard_mining, norm)
    
    def forward(self, x, y):
        z = []
        for i in range(len(x)):
            z.append(self.loss(x[i], y[i]))
        return z

class Net(nn.Module):
    def __init__(self, net_type, input_size, block, fwd_out, num_fwd, back_out, num_back, n, hard_mining = 0, loss_norm = False):
        super(Net, self).__init__()
        self.net = ResNet(net_type, input_size, block, fwd_out, num_fwd, back_out, num_back)
        self.loss = Loss(n, hard_mining, loss_norm)

    def forward(self, orig_x, adv_x, requires_control = True, train = True):
        orig_outputs = self.net(orig_x)

        if requires_control:
            control_outputs = self.net(adv_x)
            control_loss = self.loss(control_outputs, orig_outputs)

        if train:
            adv_x.volatile = False
            for i in range(len(orig_outputs)):
                orig_outputs[i].volatile = False
        adv_outputs = self.net(adv_x, defense = True)
        loss = self.loss(adv_outputs, orig_outputs)

        if not requires_control:
            return orig_outputs[-1], adv_outputs[-1], loss
        else:
            return orig_outputs[-1], adv_outputs[-1], loss, control_outputs[-1], control_loss

def get_net(net_type, input_size, block, fwd_out, num_fwd, back_out, num_back, n, hard_mining = 0, loss_norm = False):
    net = Net(net_type, input_size, block, fwd_out, num_fwd, back_out, num_back, n, hard_mining, loss_norm)
    
        
    return net
