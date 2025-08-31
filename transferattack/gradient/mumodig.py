import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import kornia.augmentation as K
from ..utils import *
from ..gradient.mifgsm import MIFGSM
import scipy.stats as st
# import random

class MUMODIG(MIFGSM):
    """
    MUMODIG Attack
    'Improving Integrated Gradient-based Transferable Adversarial Examples by Refining the Integration Path (AAAI 2025)'(https://arxiv.org/abs/2412.18844) 

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        N_trans (int):  the number of total auxiliary inputs
        N_base (int): baseline number
        N_intepolate (int): interpolation point number
        region_num (int): the region number
        lamb (float): the position factor
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, N_trans=6, N_base=1, N_intepolate=1, region_num=2, lamb=0.65

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data5/mumodig/resnet50 --attack mumodig --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data5/mumodig/resnet50 --eval    
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., 
                 N_trans = 6, N_base = 1, N_intepolate = 1, region_num = 2, lamb = 0.65, # 6,1,1,2,0.65
                 targeted=False, random_start=False, 
                 norm='linfty', loss='crossentropy', device=None, attack='MUMODIG', **kwargs): 

        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack) 
        self.N_trans = N_trans    
        self.N_base = N_base
        self.N_intepolate = N_intepolate    
        self.quant = LBQuantization(region_num)
        self.lamb = lamb


    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

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
        
        for iter_out in range(self.epoch):\
            
            sole_grad = self.ig(data, delta, label)
            exp_grad = self.exp_ig(data, delta, label)
            ig_grad = sole_grad  + exp_grad

            momentum = self.get_momentum(ig_grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)


        return delta.detach()


    def ig(self, data, delta, label, **kwargs): 
        
        ig = 0

        for i_base in range(self.N_base):
            baseline = self.quant(data+delta).clone().detach().to(self.device) 
            path = data+delta - baseline
            acc_grad = 0   
            for i_inter in range(self.N_intepolate):

                x_interplotate = baseline + (i_inter + self.lamb) * path / self.N_intepolate 
                logits = self.get_logits(x_interplotate)
                loss = self.get_loss(logits, label)

                if i_base + 1  == self.N_base and i_inter + 1 == self.N_intepolate:
                    each_ig_grad = self.get_grad(loss, delta)
                else:
                    each_ig_grad = self.get_repeat_grad(loss, delta) 
            
                # accumulate grads
                acc_grad += each_ig_grad 
            ig += acc_grad * path

        return ig


    def exp_ig(self, data, delta, label, **kwargs):
        
        ig = 0

        for i_trans in range(self.N_trans):

            x_transform = self.select_transform_apply(data+delta)

            for i_base in range(self.N_base):

                baseline = self.quant(x_transform).clone().detach().to(self.device) # quant baseline

                path = x_transform - baseline 
                
                acc_grad = 0            
                for i_inter in range(self.N_intepolate):

                    x_interplotate = baseline + (i_inter + self.lamb) / self.N_intepolate * path  

                    logits = self.get_logits(x_interplotate)

                    loss = self.get_loss(logits, label)
                     
                    if i_base + 1  == self.N_base and i_inter + 1 == self.N_intepolate:
                        each_ig_grad = self.get_grad(loss, delta)
                    else:
                        each_ig_grad = self.get_repeat_grad(loss, delta) # 

                    acc_grad += each_ig_grad 

                ig += acc_grad * path  



        return ig


    def get_repeat_grad(self, loss, delta, **kwargs):
        """
        The gradient calculation, which should be overridden when the attack need to tune the gradient (e.g., TIM, variance tuning, enhanced momentum, etc.)
        """
        return torch.autograd.grad(loss, delta, retain_graph=True, create_graph=False)[0]


    def vertical_shift(self, x):
        _, _, w, _ = x.shape          
        step = np.random.randint(low = 0, high=w, dtype=np.int32) # w = 224
        return x.roll(step, dims=2)   

    def horizontal_shift(self, x):
        _, _, _, h = x.shape          
        step = np.random.randint(low = 0, high=h, dtype=np.int32)
        return x.roll(step, dims=3)

    def vertical_flip(self, x):
        return x.flip(dims=(2,))      

    def horizontal_flip(self, x):
        return x.flip(dims=(3,))

    def random_rotate(self, x):
        rotation_transform = K.RandomRotation(p =1, degrees=45)
        return rotation_transform(x)
    
    def random_affine(self, x):
        trans_list = [self.vertical_shift, self.horizontal_shift, self.vertical_flip, self.horizontal_flip, self.random_rotate]

        i = torch.randint(0, len(trans_list), [1]).item()
        trans = trans_list[i]
        return trans(x)

    def random_resize_and_pad(self, x, img_large_size = 245, **kwargs):

        img_inter_size = torch.randint(low=min(x.shape[-1], img_large_size), high=max(x.shape[-1], img_large_size), size=(1,), dtype=torch.int32)
        img_inter = F.interpolate(x, size=[img_inter_size, img_inter_size], mode='bilinear', align_corners=False)
        res_space = img_large_size - img_inter_size
        res_top = torch.randint(low=0, high=res_space.item(), size=(1,), dtype=torch.int32)
        res_bottom = res_space - res_top
        res_left = torch.randint(low=0, high=res_space.item(), size=(1,), dtype=torch.int32)
        res_right = res_space - res_left
        padded = F.pad(img_inter, [res_left.item(), res_right.item(), res_top.item(), res_bottom.item()], value=0)
        x_trans = F.interpolate(padded, size=[x.shape[-1], x.shape[-1]], mode='bilinear', align_corners=False)
        return x_trans
    

    def select_transform_apply(self, x, **kwargs):

        T_set = [self.random_affine, self.random_resize_and_pad] # 

        i = torch.randint(0, len(T_set), [1]).item()
        trans = T_set[i]

        return trans(x)

class LBQuantization(nn.Module):
    def __init__(self, region_num, transforms_like=False):
        """
        region_num: int;
        """
        super().__init__()
        self.region_num = region_num
        self.transforms_like = transforms_like

    def get_params(self, x):
        """
        x: (C, H, W)·
        returns (C), (C), (C)
        """
        C, _, _ = x.size() # one batch img
        min_val, max_val = x.reshape(C, -1).min(1)[0], x.reshape(C, -1).max(1)[0] 


        total_region_percentile_number = (torch.ones(C) * (self.region_num - 1)).int() 

        return min_val, max_val, total_region_percentile_number

    def forward(self, x):
        """
        x: (B, c, H, W) or (C, H, W)
        """
        EPSILON = 1

        if not self.transforms_like:
            B, c, H, W = x.shape
            C = B * c                  
            x = x.reshape(C, H, W)
        else:
            C, H, W = x.shape
        min_val, max_val, total_region_percentile_number_per_channel = self.get_params(x) 


        # region percentiles for each channel
        region_percentiles = torch.rand(total_region_percentile_number_per_channel.sum(), device=x.device)

        region_percentiles_per_channel = region_percentiles.reshape([-1, self.region_num - 1])

        # ordered region ends 
        region_percentiles_pos = (region_percentiles_per_channel * (max_val - min_val).reshape(C, 1) + min_val.reshape(C, 1)).reshape(C, -1, 1, 1)
        
        ordered_region_right_ends_for_checking = torch.cat([region_percentiles_pos, max_val.reshape(C, 1, 1, 1)+EPSILON], dim=1).sort(1)[0]
        ordered_region_right_ends = torch.cat([region_percentiles_pos, max_val.reshape(C, 1, 1, 1)+1e-6], dim=1).sort(1)[0]
        
        ordered_region_left_ends = torch.cat([min_val.reshape(C, 1, 1, 1), region_percentiles_pos], dim=1).sort(1)[0]

        is_inside_each_region = (x.reshape(C, 1, H, W) < ordered_region_right_ends_for_checking) * (x.reshape(C, 1, H, W) >= ordered_region_left_ends) # -> (C, self.region_num, H, W); boolean
        assert (is_inside_each_region.sum(1) == 1).all()# sanity check: each pixel falls into one sub_range

        associated_region_id = torch.argmax(is_inside_each_region.int(), dim=1, keepdim=True)  # -> (C, 1, H, W)


        proxy_vals = torch.gather(ordered_region_left_ends.expand([-1, -1, H, W]), 1, associated_region_id)[:,0]
        x = proxy_vals.type(x.dtype)



        if not self.transforms_like:
            x = x.reshape(B, c, H, W)   

        return x

