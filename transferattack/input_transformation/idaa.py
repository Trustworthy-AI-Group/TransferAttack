import torch
import torch.nn.functional as F
from ..utils import *
from ..gradient.mifgsm import MIFGSM
import kornia

from kornia.augmentation import (
    RandomHorizontalFlip,           
    RandomPerspective,              
    RandomRotation,                 
    RandomVerticalFlip,             
    RandomThinPlateSpline,          
    Resize,                          
    RandomAffine,                   
    RandomErasing,                  
    RandomElasticTransform,         
    RandomFisheye,                  
)
import scipy.stats as st

class IDAA(MIFGSM):
    """
    IDAA(Input-Diversity-based Adaptive Attack)
    'Boosting the Transferability of Adversarial Examples via Local Mixup and Adaptive Step Size'(https://arxiv.org/pdf/2401.13205) ICASSP2025
    
    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_scale (int): the number of copies for transformation.
        gamma (float): the weight controling deviation strength.
        mixup_num (float): the mixup operation executed times.
        mixup_alpha (float): the parameter to adjust beta distribution in local mixup function.
        crop_size (float): the resize local region in Resize operation.
        aug_p (float): the probability of applying input transformations.
        beta1 (float): calculated for first-order gradient.
        beta2 (float): calculated for second-order gradient.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., num_scale=10, mixup_num=3, mixup_alpha=0.4, crop_size=0.7, 

    To reproduce results of the original paper, please change settings as follows:
        in attack.py: 
            model = models.__dict__[model_name](weights="IMAGENET1K_V1")
        
        in utils.py:
            img_height, img_width = 299, 299
            def load_pretrained_model(cnn_model=[], vit_model=[]):
                for model_name in cnn_model:
                    yield model_name, models.__dict__[model_name](weights="IMAGENET1K_V1")
            
        
    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/idaa/resnet50_targeted --attack idaa --model=resnet50 --targeted 
        python main.py --input_dir ./path/to/data --output_dir adv_data/idaa/resnet50_targeted --eval --targeted
    """
    
    def __init__(self, model_name, epsilon=0.07, alpha=1, epoch=10, decay=1., num_scale=10, gamma=0.1, mixup_num=3, mixup_alpha=0.4, crop_size=0.7, aug_p=1.0, targeted=True, random_start=False, 
                norm='linfty', loss='crossentropy', device=None, attack='IDAA', **kwargs):
        super().__init__(model_name,  epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.alpha = alpha
        self.num_scale = num_scale
        self.gamma = gamma
        self.mixup_num = mixup_num
        self.mixup_alpha = mixup_alpha
        self.crop_size = crop_size
        self.aug_p = aug_p
        self.beta1 = 0.99
        self.beta2 = 0.999
        
        self.op = torch.nn.Sequential(
            RandomHorizontalFlip(same_on_batch=False, keepdim=False, p=self.aug_p),           
            RandomPerspective(0.5, "nearest", align_corners=True, same_on_batch=False, keepdim=False, p=self.aug_p),              
            RandomRotation(15.0, "nearest", align_corners=True, same_on_batch=False, keepdim=False, p=self.aug_p),                 
            RandomVerticalFlip(same_on_batch=False, keepdim=False, p=0.6, p_batch=self.aug_p),             
            RandomThinPlateSpline(0.3, align_corners=True, same_on_batch=False, keepdim=False, p=self.aug_p),          
            RandomResize(0.9, p=self.aug_p),                   
            RandomAffine((-1.0, 5.0), (0.3, 1.0), (0.4, 1.3), 0.5, resample="nearest",
                    padding_mode="reflection", align_corners=True, same_on_batch=False, keepdim=False, p=self.aug_p),
            RandomErasing(scale=(0.01, 0.04), ratio=(0.3, 1.0), value=1, same_on_batch=False, keepdim=False, p=self.aug_p),                
            RandomElasticTransform((27, 27), (33, 31), (0.1, 1.0), align_corners=True, padding_mode="reflection", same_on_batch=False, keepdim=False, p=self.aug_p),         
            RandomFisheye(kornia.core.tensor([-0.3, 0.3]), kornia.core.tensor([-0.3, 0.3]), kornia.core.tensor([0.9, 1.0]), same_on_batch=False, keepdim=False, p=self.aug_p)
            )                  

    
    def rand_bbox(self, shape_size, ws=None, hs=None):
        W, H = shape_size[2], shape_size[3]
        if ws is None or hs is None:
            cut_w, cut_h = np.int32(W * self.crop_size), np.int32(H * self.crop_size)
        else:
            cut_w, cut_h = ws, hs
        if ws is not None and hs is not None:
            cx, cy = np.random.randint(W - ws), np.random.randint(H - hs)
            bbx1, bby1, bbx2, bby2 = cx, cy, cx + ws, cy + hs
        else:
            cx, cy = np.random.randint(W), np.random.randint(H)
            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    def local_mix(self, B1):
        B1_prime = B1.clone()
        length = len(B1)
        sz = B1.size()
        for _ in range(self.mixup_num): 
            indexes = torch.randperm(length)
            B2 = B1_prime[indexes]
            for i in range(length):
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                lam = max(lam, 1 - lam)
                bb1, bb2, bb3, bb4 = self.rand_bbox(sz)
                bb5, bb6, bb7, bb8 = self.rand_bbox(sz, bb3-bb1, bb4-bb2)
                B1_prime[i][:, bb1:bb3, bb2:bb4] = lam * B1_prime[i][:, bb1:bb3, bb2:bb4] + (1 - lam) * B2[i][:, bb5:bb7, bb6:bb8]
        return B1_prime
    
    def get_loss(self, logits, y_src, y_tgt, **kwargs):
        if self.targeted:
            l_pos = self.loss(logits, y_tgt.repeat(self.num_scale+1))
            l_neg = self.loss(logits, y_src.repeat(self.num_scale+1))
            return l_pos - self.gamma*l_neg 
        else:
            return self.loss(logits, y_src.repeat(self.num_scale+1))
        
    def get_bound(self, x):
        lower_bound = -torch.min(x,self.epsilon*torch.ones_like(x))
        upper_bound = torch.min(1-x, self.epsilon*torch.ones_like(x))
        return lower_bound, upper_bound
    
    def compute_perturbation(self, w, lb, ub):
        return lb + (ub-lb) * (torch.tanh(w)/2 + 1/2) 
    
    def update_delta(self, delta, grad, m, v, alpha, **kwargs):
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * grad**2
        eps = 1e-8
        delta = delta - alpha * m /(torch.sqrt(v) + eps)
        return delta, m, v

    def init_delta_and_normal_distribute(self, data, **kwargs):
        delta = torch.randn_like(data).to(self.device)
        delta.requires_grad = True
        return delta
    
    def get_grad(self, loss, delta, **kwargs):
        return torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
    
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
            y_src = label[0].clone().detach().to(self.device)
            y_tgt = label[1].clone().detach().to(self.device)
        else:
            y_src = label.clone().detach().to(self.device)
            y_tgt = None

        data = data.clone().detach().to(self.device)
        ub, lb = self.get_bound(data)
        delta = self.init_delta_and_normal_distribute(data)
        N_repeat = self.num_scale + 1
        m = torch.zeros_like(delta)
        v = torch.zeros_like(delta)
        batch_size = data.shape[0]
        for _ in range(self.epoch):
            deltas = delta.repeat(N_repeat, 1, 1, 1)
            r_t = self.compute_perturbation(
                deltas,
                lb.repeat(N_repeat, 1, 1, 1),
                ub.repeat(N_repeat, 1, 1, 1),
            )
            x_adv_repeats = data.repeat(N_repeat,1,1,1) + r_t
            num_ops = len(self.op)
            transformed_list = []
            for k in range(x_adv_repeats.shape[0]):
                img_k = x_adv_repeats[k:k+1]
                transformed_list.append(self.op[k%num_ops](img_k))
            
            B1 = torch.cat(transformed_list, dim=0)
            B1_prime = self.local_mix(B1)
            logits = self.get_logits(B1_prime)
            loss = self.get_loss(logits, y_src, y_tgt)
            grad_clone = torch.autograd.grad(loss, deltas)[0]            
            grad_norm = torch.mean(torch.abs(grad_clone), dim=(1, 2, 3), keepdim=True)
            grad = grad_clone / (grad_norm + 1e-8)
            grad_aggregated = grad.view(N_repeat, batch_size, *delta.shape[1:]).mean(0)
            delta, m, v = self.update_delta(delta, grad_aggregated, m, v, self.alpha)

        return self.compute_perturbation(delta, lb, ub)

class RandomResize(torch.nn.Module):
    def __init__(self, resize_ratio=0.9, p=1.0):
        super().__init__()
        self.resize_ratio = resize_ratio
        self.p = p

    def forward(self, imgs):
        if np.random.rand() > self.p:
            return imgs
        H, W = imgs.shape[-2:]
        resize_limit = int(W * self.resize_ratio)
        low = min(W, resize_limit)
        high = max(W, resize_limit)
        rnd_size = np.random.randint(low, high + 1)
        rescaled = F.interpolate(imgs, size=[rnd_size, rnd_size], mode='bilinear', align_corners=False)
        pad_h = H - rnd_size
        pad_w = W - rnd_size
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top        
        padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom), value=0)
        return padded
