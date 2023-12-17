import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(192, 48, kernel_size=1, stride=1),
            BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2)
        ) 

        self.branch2 = nn.Sequential(
            BasicConv2d(192, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out

class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()
        
        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out

class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 160, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(160, 192, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out

class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()
        
        self.branch0 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(2080, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv2d(224, 256, kernel_size=(3,1), stride=1, padding=(1,0))
        )

        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


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


class Null(nn.Module):
    def __init__(self):
        super(Null, self).__init__()

    def forward(self, x):
        return x


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


class InceptionResnetV2(nn.Module):

    def __init__(self, input_size, denoise_block, fwd_out, num_fwd, back_out, num_back, num_classes=1001):
        super(InceptionResnetV2, self).__init__()
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17)
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10)
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20)
        )
        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)
        self.classif = nn.Linear(1536, num_classes)

        h_in, w_in = [299, 71, 35, 17, 8], [299, 71, 35, 17, 8]
        denoise = []
        n_in = [3, 192, 320, 1088, 1536]
        block = denoise_block
        for i in range(len(block)):
            if block[i] is None:
                denoise.append(Null())
            else:
                denoise.append(Denoise(h_in[i], w_in[i], block[i], n_in[i], fwd_out[i], num_fwd[i], back_out[i], num_back[i]))
        self.denoise = nn.ModuleList(denoise)

    def forward(self, x, defense = False):
        outputs = []
        
        if defense:
            x = self.denoise[0](x)
        outputs.append(x)
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        if defense:
            x = self.denoise[1](x)
        outputs.append(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        if defense:
            x = self.denoise[2](x)
        outputs.append(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        if defense:
            x = self.denoise[3](x)
        outputs.append(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        if defense:
            x = self.denoise[4](x)
        outputs.append(x)
        x = self.avgpool_1a(x)
        x = x.view(x.size(0), -1)
        x = self.classif(x)
        outputs.append(x[:, 1:])
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
    def __init__(self, input_size, block, fwd_out, num_fwd, back_out, num_back, n, hard_mining = 0, loss_norm = False):
        super(Net, self).__init__()
        self.net = InceptionResnetV2(input_size, block, fwd_out, num_fwd, back_out, num_back)
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

def get_net(input_size, block, fwd_out, num_fwd, back_out, num_back, n, hard_mining = 0, loss_norm = False):
    net = Net(input_size, block, fwd_out, num_fwd, back_out, num_back, n, hard_mining, loss_norm)

    return net
