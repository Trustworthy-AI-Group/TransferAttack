# Modified  from https://github.com/xinntao/BasicSR
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.module_util as mutil


###############################################################
# NRP network
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class NRP(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(NRP, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = mutil.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))

        return trunk


#################################################################
# NRP based on ResNet Generator
class NRP_resG(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23):
        super(NRP_resG, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(mutil.ResidualBlock_noBN, nf=nf)
        self.recon_trunk = mutil.make_layer(basic_block, nb)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out = self.conv_last(self.recon_trunk(fea))
        return out


if __name__ == '__main__':
    netG = NRP_resG(3, 3, 64, 23)
    netG.load_state_dict(torch.load('pretrained_purifiers/NRP_resG.pth'))
    test_sample = torch.rand(1, 3, 256, 256)
    print(netG(test_sample).size())
    #print(netG(test_sample).size())
    print(sum(p.numel() for p in netG.parameters() if p.requires_grad))