
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision

from PIL import Image


# copy from advertorch
class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

def jpeg2png(name):
    name_list = list(name)
    name_list[-4:-1] = 'png'
    name_list.pop(-1)
    return ''.join(name_list)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, transform=None, png=True):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.labels = pd.read_csv(self.label_dir).to_numpy()
        self.png = png

    def __getitem__(self, index):
        file_name, label = self.labels[index]
        label = torch.tensor(label) - 1
        file_dir = os.path.join(self.img_dir, file_name)
        if self.png:
            file_dir = jpeg2png(file_dir)
        img = Image.open(file_dir).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.labels)

def save_images(adv, i, batch_size, label_dir, output_dir, png=True):
    '''
    save the adversarial images
    :param adv: adversarial images in [0, 1]
    :param i: batch index of images
    :return:
    '''
    dest_dir = output_dir
    labels = pd.read_csv(label_dir).to_numpy()
    base_idx = i * batch_size
    for idx, img in enumerate(adv):
        fname = labels[idx + base_idx][0]
        dest_name = os.path.join(dest_dir, fname)
        if png:
            dest_name = jpeg2png(dest_name)
        torchvision.utils.save_image(img, dest_name)
    return

class LGVModel(nn.Module):

    def __init__(self, arch, qaa):
        super(LGVModel, self).__init__()
        self.sequence = np.random.permutation(np.arange(1, 41))
        self.id = 0
        self.arch = arch
        self.qaa = qaa
        if self.qaa:
            if self.arch == 'resnet34' or self.arch == 'resnet50':
                import archs.apot as models
                model = models.__dict__[self.arch](pretrained=False, bit=2, stochastic=True)
                model = torch.nn.DataParallel(model).cuda()
                ckpt_dir = './checkpoints/apot/{}_w2a2_stochastic_120603.pth.tar'.format(self.arch)
                model.load_state_dict(torch.load(ckpt_dir)["state_dict"])
                self.qnn = model.eval().cuda()
            else:
                raise Exception('arch {} not implemented!'.format(self.arch))
            
    def forward(self, x):
        if self.qaa:
            if self.id % 2 == 0:
                out = self.qnn(x)
            else:
                import torchvision.models as models
                model = models.__dict__[self.arch](pretrained=False).cuda().eval()
                ckpt_dir = './checkpoints/fp/lgv/%s/cSGD/seed0/iter-%05d.pt' % (self.arch, self.sequence[self.id])
                model.load_state_dict(torch.load(ckpt_dir)['state_dict'])
                out = model(x)
            self.id = (self.id + 1) % 40
        else:
            import torchvision.models as models
            model = models.__dict__[self.arch](pretrained=False).cuda().eval()
            ckpt_dir = './checkpoints/fp/lgv/%s/cSGD/seed0/iter-%05d.pt' % (self.arch, self.sequence[self.id])
            model.load_state_dict(torch.load(ckpt_dir)['state_dict'])
            out = model(x)
            self.id = (self.id + 1) % 40
        return out

def load_model(args):
    if args.quantize_method == "pytorch":
        assert args.w_bit == 8 and args.a_bit == 8
        import torchvision.models.quantization as models
        model = models.__dict__[args.arch](pretrained=True, quantize=True)
        print("8-bit model {} loaded successfully!".format(args.arch))
    elif args.quantize_method == "apot":
        if args.stochastic == True and args.ckpt_id == '120603':
            import archs.apot as models
            model = models.__dict__[args.arch](pretrained=False, bit=args.w_bit, stochastic=True)
            model = torch.nn.DataParallel(model).cuda()
            model_dir = os.path.join(args.ckpt_dir, "apot", args.arch + "_w{}a{}_stochastic_120603.pth.tar".format(args.w_bit, args.a_bit))
            model.load_state_dict(torch.load(model_dir)["state_dict"])
        else:
            import archs.apot as models
            model = models.__dict__[args.arch](pretrained=False, bit=args.w_bit, stochastic=False)
            model = torch.nn.DataParallel(model).cuda()
            model_dir = os.path.join(args.ckpt_dir, "apot", args.arch + "_w{}a{}.pth.tar".format(args.w_bit, args.a_bit))
            model.load_state_dict(torch.load(model_dir)["model"])
        print("model successfully loaded from {}".format(model_dir))
    else:
        raise Exception('quantize method {} not implemented!'.format(args.quantize_method))

    return model