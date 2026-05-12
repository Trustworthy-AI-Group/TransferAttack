import torch
import torchvision.transforms as T
import timm
import numpy as np
import random
import torch.nn.functional as F
from ..utils import *
from ..gradient.mifgsm import MIFGSM
from scipy.optimize import brentq

class SID(MIFGSM):
    """
    SID Attack
    Leveraging Spatial Invariance to Boost Adversarial Transferability (https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_Leveraging_Spatial_Invariance_to_Boost_Adversarial_Transferability_ICCV_2025_paper.pdf)
    
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
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1.0,
                 targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None, attack='SID', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        
    def dct(self, x, norm=None):
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)
        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
        Vc = torch.fft.fft(v)
        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)
        V = Vc.real * W_r - Vc.imag * W_i
        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2
        V = 2 * V.view(*x_shape)
        return V
    
    def idct(self, x, norm=None):
        x_shape = x.shape
        N = x_shape[-1]
        x_v = x.contiguous().view(-1, x_shape[-1]) / 2
        if norm == 'ortho':
            x_v[:, 0] *= np.sqrt(N) * 2
            x_v[:, 1:] *= np.sqrt(N / 2) * 2
        k = torch.arange(x_shape[-1], dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)
        V_t_r = x_v
        V_t_i = torch.cat([x_v[:, :1] * 0, -x_v.flip([1])[:, :-1]], dim=1)
        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r
        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
        tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
        v = torch.fft.ifft(tmp)
        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]
        return x.view(*x_shape).real
    
    def dct_2d(self, x, norm=None):
        x1 = self.dct(x, norm=norm)
        x2 = self.dct(x1.transpose(-1, -2), norm=norm)
        return x2.transpose(-1, -2)
    
    def idct_2d(self, x, norm=None):
        x1 = self.idct(x, norm=norm)
        x2 = self.idct(x1.transpose(-1, -2), norm=norm)
        return x2.transpose(-1, -2)
    
    def get_length(self, length, num_block):
        length = int(length)
        rand = np.random.uniform(size=num_block, low=0.1, high=0.9)
        rand_norm = np.round(rand*length/rand.sum()).astype(np.int32)
        rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
        return tuple(rand_norm)
    
    def random_flip(self, x):
        ret = x.clone()
        if torch.rand(1) < 0.5:
            ret = torch.flip(ret, dims=(3,))
        return ret
    
    def frequency_fusion(self, patch, x):
        org_x = x.clone()
        _, _, patch_w, patch_h = patch.shape
        rescale_x = F.interpolate(org_x, size=[patch_w, patch_h], mode='bilinear', align_corners=False)
        rescale_flip_x = self.random_flip(rescale_x)
        dctx = self.dct_2d(rescale_flip_x)  
        dctp = self.dct_2d(patch)
        _, _, w, h = dctx.shape
        low_ratio = 0.4
        low_w = int(w * low_ratio)
        low_h = int(h * low_ratio)
        # patch_low = dctp[:, :, 0:low_w, 0:low_h]
        dctx[:, :, 0:low_w, 0:low_h] = dctp[:, :, 0:low_w, 0:low_h]
        idctx = self.idct_2d(dctx)
        return idctx
    
    def linear_fusion(self, patch, x, omega=0.5):
        org_x = x.clone()
        _, _, patch_w, patch_h = patch.shape
        rescale_x = F.interpolate(org_x, size=[patch_w, patch_h], mode='bilinear', align_corners=False)

        rescale_flip_x = self.random_flip(rescale_x)
        ret = rescale_flip_x * omega + patch * (1 - omega)
        return ret
    
    def block_fusion(self, patch, x, probabilities=0.5, omega=0.5):
        if torch.rand(1) < probabilities:
            return patch
        else:
            if torch.rand(1) < 0.5:
                return self.frequency_fusion(patch, x)
            else:
                return self.linear_fusion(patch, x, omega)
            
    def local_fusion(self, x, num_block=2, probabilities=0.5, omega=0.5):
        batch_size, _, w, h = x.shape
        width_length, height_length = self.get_length(w, num_block), self.get_length(h, num_block)
        x_split_w = torch.split(x, width_length, dim=2)
        x_split_h_l = [torch.split(x_split_w[i], height_length, dim=3) for i in range(num_block)]

        ret_list = []
        for strip in x_split_h_l:
            temp_list = []
            for i in range(num_block):
                x_enh = self.block_fusion(strip[i], x, probabilities, omega)
                x_enh_flip= self.random_flip(x_enh)
                temp_list.append(x_enh_flip)
            temp = torch.cat(temp_list, dim=3)
            ret_list.append(temp)
        x_h_perm = torch.cat(ret_list, dim=2)
        return x_h_perm
    
    def multi_scale(self, x, resize_ratio):
        img_size = x.shape[-1]
        if resize_ratio == 1:
            ret =  x
        else:
            img_resize = int(img_size * resize_ratio)
            rescaled = F.interpolate(x, size=[img_resize, img_resize], mode='bilinear', align_corners=False)
            h_rem = img_size - img_resize
            w_rem = img_size - img_resize
            pad_top = torch.randint(low=0, high=h_rem, size=(1,), dtype=torch.int32)
            pad_bottom = h_rem - pad_top
            pad_left = torch.randint(low=0, high=w_rem, size=(1,), dtype=torch.int32)
            pad_right = w_rem - pad_left
            ret = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
        ret = self.random_flip(ret)
        return ret
    
    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        # Initialize adversarial perturbation
        delta = self.init_delta(data)
        momentum = 0
        N = 20
        final_grad = 0
        for _ in range(self.epoch):
            avg_grad = 0
            for n in range(N):
                x = data+delta
                x_emb = self.local_fusion(x, num_block=2, probabilities=0.5, omega=0.5)
                resize_ratio = 1 - (n*0.1/N)
                x_enh = self.multi_scale(x_emb, resize_ratio)
                logits = self.get_logits(x_enh, momentum=momentum)
                loss = self.get_loss(logits, label)
                grad = self.get_grad(loss, delta)
                avg_grad += grad
            final_grad = avg_grad / N
            momentum = self.get_momentum(final_grad, momentum)
            delta = self.update_delta(delta, data, momentum, self.alpha)
        return delta.detach()

