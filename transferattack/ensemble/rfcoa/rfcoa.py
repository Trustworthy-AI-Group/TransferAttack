import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchmetrics
from collections import OrderedDict

from transferattack.utils import wrap_model
from ...attack import Attack
from .networks import vgg, resnet
from ... utils import generation_target_classes

class RFCoA(Attack):
    """
    RFCoA
    'Breaking Barriers in Physical-World Adversarial Examples: Improving Robustness and Transferability via Robust Feature (AAAI 2025)'(https://arxiv.org/abs/2412.16958)

    Arguments:
        model_name (str): the name of surrogate models for attack.
        epsilon (float): the perturbation budget.
        epoch (int): the number of iterations.
        arch (str): the architecture of autoencoder
        autoencoder_path(str): the path of pretrained autoencoder.
        targeted (bool): targeted/untargeted attack.
        feature_path(str):the path of extracted feature.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        img_size(int): the size of input images.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, epoch=300, autoencoder_path='./transferattack/ensemble/rfcoa/imagenet-vgg16.pth', feature_path='./transferattack/ensemble/rfcoa/24.npz'

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/rfcoa/ --attack rfcoa --targeted --ensemble --model resnet50,vgg16,densenet121
        python main.py --input_dir ./path/to/data --output_dir adv_data/rfcoa --eval --targeted
    """
    def __init__(self, model_name, attack='RFCoA', epsilon=16/255, targeted=True, random_start=True, norm='linfty', 
                loss='crossentropy', arch="vgg16", autoencoder_path='./transferattack/ensemble/rfcoa/imagenet-vgg16.pth', 
                feature_path='./transferattack/ensemble/rfcoa/24.npz', epoch=300, device=None):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        
        self.autoencoder = self.buildautoencoder(arch,autoencoder_path).to(self.device)
        self.ssim_calculator = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.feature_path =feature_path
        self.epoch=epoch
        # self.normalize = transforms.Compose([
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        # ])

    def total_variation(self, tensor):
        _, _, h, w = tensor.size()
        tv_h = torch.sum(torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :]))
        tv_w = torch.sum(torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1]))
        return tv_h + tv_w

    def buildautoencoder(self, arch, path):

        if arch in ["vgg11", "vgg13", "vgg16", "vgg19"]:
            configs = vgg.get_configs(arch)
            model = vgg.VGGAutoEncoder(configs)

        elif arch in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
            configs, bottleneck = resnet.get_configs(arch)
            model = resnet.ResNetAutoEncoder(configs, bottleneck)
        
 
        checkpoint = torch.load(path)
        model_dict = model.state_dict()
        model_dict.update(checkpoint['state_dict'])
        new_state_dict = OrderedDict()
        for k, v in model_dict.items():
            new_key = k.replace("module.", "")  # 去掉 "module."
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)
        return model
    
    def spatial_attention_map (self, feat, label):
        feat.requires_grad_()
        loss = 0
        decode = self.autoencoder.decode(feat)
        # decode = self.normalize(decode)

        for model in self.model.models:
            output = model(decode)
            loss += self.loss(output, label)

        loss /= len(self.model.models)

        grad = torch.autograd.grad(loss, feat)[0]
        grad = torch.abs(grad)

        sam = torch.sigmoid(grad)
        feat = feat.detach()

        return feat, sam

    def forward(self, images, labels, idx, **kwargs):
        feature_path = self.feature_path.replace("24", str(generation_target_classes[idx]))
        data = np.load(feature_path)
        self.feature = torch.from_numpy(data['array']).to(self.device)
        if self.targeted:
            assert len(labels) == 2
            label = labels[0] 
            adv_label = labels[1]
        images = images.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        adv_label = adv_label.clone().detach().to(self.device)

        with torch.no_grad():
            org = self.autoencoder.get_feature(images)
            org = org.to(self.device)

        org, sam = self.spatial_attention_map(org, label)

        alpha = torch.rand(org.shape).to(self.device)
        mask = torch.rand(images.shape).to(self.device)

        for step in range(self.epoch):
            alpha.requires_grad_()
            mask.requires_grad_()

            optimizer = optim.Adam([
                {'params': mask, 'lr': 0.002},
                {'params': alpha, 'lr': 0.04}
            ])

            encode = alpha * self.feature + (1 - sam) * org
            decoded = self.autoencoder.decode(encode)
            decoded = mask * decoded + (1 - mask) * images

            per_loss = torch.norm(mask, p=1)
            tv_loss = self.total_variation(mask)
            ssim_loss = self.ssim_calculator(decoded, images)

            # decoded_norm = self.normalize(decoded)
            outputs = [m(decoded) for m in self.model.models]

            adv_loss_1 = sum(self.loss(o, adv_label) for o in outputs) / len(outputs)
            adv_loss_2 = sum(self.loss(o, label) for o in outputs) / len(outputs)

            adv_loss = 5 * adv_loss_1 - 2 * adv_loss_2
            cog_loss = 0.005 * per_loss + 0.002 * tv_loss - 200 * ssim_loss

            total_loss = adv_loss + cog_loss

            # decoded = self.normalize(decoded)


            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            alpha = torch.clamp(alpha.detach(), 0, 1)
            mask = torch.clamp(mask.detach(), 0, 1)


        encode = alpha * self.feature + sam * org
        decoded = self.autoencoder.decode(encode)


        adv_images = mask * decoded + (1 - mask) * images

        return (adv_images-images).detach()
